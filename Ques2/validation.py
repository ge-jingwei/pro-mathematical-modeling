from dataclasses import asdict, replace

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.signal import butter, hilbert, sosfiltfilt
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.special import ndtr
from scipy.stats import ttest_rel
from Ques1.statistics import cluster_sign_test

from Ques2.model import ModelParameters, simulate, vary


def _scales(model: dict, erp: dict, times: np.ndarray, task: int) -> np.ndarray:
    mask = (times >= 0) & (times <= 0.6)
    result = []
    for channel in range(3):
        x = np.concatenate([model[side].eeg[channel, mask] for side in (-1, 1)])
        y = np.concatenate([erp[(task, side)]["mean"][channel, mask] for side in (-1, 1)])
        result.append(float(x @ y / max(x @ x, np.finfo(float).eps)))
    return np.asarray(result)


def calibrated_prediction(model: dict, scales: np.ndarray) -> dict:
    return {side: item.eeg * scales[:, None] for side, item in model.items()}


def _lag_design(signal: np.ndarray, maximum: int = 160, step: int = 4) -> np.ndarray:
    columns = []
    for lag in range(0, maximum + 1, step):
        shifted = np.zeros_like(signal)
        shifted[lag:] = signal[:-lag] if lag else signal
        columns.append(shifted)
    return np.column_stack(columns)


def fit_observation_kernel(model: dict, reference: dict, times: np.ndarray, maximum: int = 160, alpha: float = 0.1) -> tuple[dict, list[dict]]:
    mask = (times >= 0) & (times <= 0.75)
    kernels = []
    for channel in range(3):
        design = np.vstack([_lag_design(model[side].eeg[channel], maximum)[mask] for task in (1, 2) for side in (-1, 1)])
        target = np.concatenate([reference[(task, side)]["mean"][channel, mask] for task in (1, 2) for side in (-1, 1)])
        scale = np.linalg.norm(design, axis=0)
        scale[scale == 0] = 1.0
        fitted = Ridge(alpha=alpha).fit(design / scale, target)
        kernels.append({"coef": fitted.coef_, "intercept": fitted.intercept_, "scale": scale})
    return apply_observation_kernel(model, kernels, times, maximum), kernels


def apply_observation_kernel(model: dict, kernels: list[dict], times: np.ndarray, maximum: int = 160) -> dict:
    result = {}
    for side in (-1, 1):
        channels = []
        for channel, kernel in enumerate(kernels):
            design = _lag_design(model[side].eeg[channel], maximum) / kernel["scale"]
            channels.append(design @ kernel["coef"] + kernel["intercept"])
        values = np.asarray(channels)
        values -= values[:, times < 0].mean(axis=1, keepdims=True)
        result[side] = values
    return result


def validation_metrics(prediction: dict, erp: dict, times: np.ndarray, task: int, rate: float) -> dict:
    mask = (times >= 0) & (times <= 0.75)
    correlations = []
    spectra = []
    for side in (-1, 1):
        for channel in range(3):
            observed = erp[(task, side)]["mean"][channel, mask]
            predicted = prediction[side][channel, mask]
            correlations.append(np.corrcoef(observed, predicted)[0, 1])
            _, obs_psd = welch(observed, rate, nperseg=min(128, len(observed)))
            _, pred_psd = welch(predicted, rate, nperseg=min(128, len(predicted)))
            spectra.append(float(obs_psd @ pred_psd / np.sqrt((obs_psd @ obs_psd) * (pred_psd @ pred_psd))))
    return {"waveform_correlation": float(np.nanmean(correlations)), "spectral_cosine": float(np.nanmean(spectra))}


def _lovo_mean(epochs: np.ndarray, fraction: float) -> np.ndarray:
    center = np.median(epochs, axis=0)
    distance = np.linalg.norm((epochs - center).reshape(len(epochs), -1), axis=1)
    keep = max(2, int(np.ceil(len(epochs) * fraction)))
    return epochs[np.argsort(distance)[:keep]].mean(axis=0)


def observation_cross_validation_folds(dataset, model: dict, rate: float, folds: int = 5, maximum: int = 160, alpha: float = 0.1, lovo_fraction: float | None = None) -> pd.DataFrame:
    splits = {}
    for task in (1, 2):
        for side in (-1, 1):
            indices = np.flatnonzero((dataset.tasks == task) & (dataset.labels == side))
            splits[(task, side)] = [(indices[train], indices[test]) for train, test in KFold(folds, shuffle=True, random_state=20260923).split(indices)]
    rows = []
    for fold in range(folds):
        train = {}
        test = {}
        for key, values in splits.items():
            train_indices, test_indices = values[fold]
            epochs = dataset.epochs[train_indices]
            train[key] = {"mean": _lovo_mean(epochs, lovo_fraction) if lovo_fraction else epochs.mean(axis=0)}
            test[key] = {"mean": dataset.epochs[test_indices].mean(axis=0)}
        prediction, _ = fit_observation_kernel(model, train, dataset.times, maximum, alpha)
        window, _, _ = lateral_window(prediction, dataset.times)
        for task in (1, 2):
            rows.append({"task": task, "fold": fold + 1, **validation_metrics(prediction, test, dataset.times, task, rate), "sign_agreement": sign_agreement(prediction, test, dataset.times, task, window)})
    return pd.DataFrame(rows)


def aggregate_validation(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby("task", as_index=False).agg(waveform_correlation=("waveform_correlation", "mean"), waveform_correlation_sd=("waveform_correlation", "std"), spectral_cosine=("spectral_cosine", "mean"), spectral_cosine_sd=("spectral_cosine", "std"), sign_agreement=("sign_agreement", "mean"))


def observation_cross_validation(dataset, model: dict, rate: float, folds: int = 5, maximum: int = 160, alpha: float = 0.1, lovo_fraction: float | None = None) -> pd.DataFrame:
    return aggregate_validation(observation_cross_validation_folds(dataset, model, rate, folds, maximum, alpha, lovo_fraction))


def sham_paired_tests(real: pd.DataFrame, shams: dict, task: int = 2) -> pd.DataFrame:
    rows = []
    for name, frame in shams.items():
        reference = real.query("task == @task").sort_values("fold")
        compare = frame.query("task == @task").sort_values("fold")
        difference = reference["waveform_correlation"].to_numpy() - compare["waveform_correlation"].to_numpy()
        statistic, probability = ttest_rel(reference["waveform_correlation"], compare["waveform_correlation"])
        rows.append({"model": name, "mean_difference": float(difference.mean()), "difference_sd": float(difference.std(ddof=1)), "t_statistic": float(statistic), "p_value": float(probability), "folds": len(difference)})
    return pd.DataFrame(rows)


def lateral_window(prediction: dict, times: np.ndarray) -> tuple[np.ndarray, float, float]:
    contrast = (prediction[1][2] - prediction[1][1]) - (prediction[-1][2] - prediction[-1][1])
    search = (times >= 0.08) & (times <= 0.5)
    center = int(np.flatnonzero(search)[np.argmax(np.abs(contrast[search]))])
    mask = np.abs(times - times[center]) <= 0.06
    return mask, float(times[mask][0]), float(times[mask][-1])


def sign_agreement(prediction: dict, erp: dict, times: np.ndarray, task: int, mask: np.ndarray) -> float:
    agreements = []
    for channel in (1, 2):
        predicted = prediction[1][channel, mask].mean() - prediction[-1][channel, mask].mean()
        observed = erp[(task, 1)]["mean"][channel, mask].mean() - erp[(task, -1)]["mean"][channel, mask].mean()
        agreements.append(np.sign(predicted) == np.sign(observed))
    return float(np.mean(agreements))


def calibrate(erp: dict, times: np.ndarray, rate: float, config: dict) -> tuple[ModelParameters, dict, np.ndarray, pd.DataFrame]:
    rows = []
    grid = config["q2"]["calibration"]
    for delay in grid["delay"]:
        for coupling in grid["coupling"]:
            for input_gain in grid["input_gain"]:
                parameters = ModelParameters(delay=float(delay), coupling=float(coupling), input_gain=float(input_gain))
                model = {side: simulate(side, times, rate, parameters, config["q2"]["grid_size"]) for side in (-1, 1)}
                metrics = validation_metrics(calibrated_prediction(model, _scales(model, erp, times, 1)), erp, times, 1, rate)
                rows.append({"delay_ms": delay * 1000, "coupling": coupling, "input_gain": input_gain, **metrics})
    frame = pd.DataFrame(rows)
    near = frame[frame["waveform_correlation"] >= frame["waveform_correlation"].max() - float(grid["tolerance"])]
    chosen = near.sort_values(["delay_ms", "coupling", "input_gain"]).iloc[0]
    parameters = ModelParameters(delay=float(chosen.delay_ms) / 1000.0, coupling=float(chosen.coupling), input_gain=float(chosen.input_gain))
    model = {side: simulate(side, times, rate, parameters, config["q2"]["grid_size"]) for side in (-1, 1)}
    return parameters, model, _scales(model, erp, times, 1), frame


def sensitivity(parameters: ModelParameters, kernels: list[dict], erp: dict, times: np.ndarray, rate: float, config: dict, maximum: int = 160) -> pd.DataFrame:
    rows = []
    for name in ("delay", "coupling", "input_gain", "tau_e", "tau_i"):
        for factor in (0.8, 1.2):
            current = vary(parameters, name, factor)
            model = {side: simulate(side, times, rate, current, config["q2"]["grid_size"]) for side in (-1, 1)}
            prediction = apply_observation_kernel(model, kernels, times, maximum)
            metrics = validation_metrics(prediction, erp, times, 2, rate)
            mask, _, _ = lateral_window(prediction, times)
            rows.append({"parameter": name, "factor": factor, "value": getattr(current, name), **metrics, "sign_agreement": sign_agreement(prediction, erp, times, 2, mask)})
    return pd.DataFrame(rows)


def _lateralization_metrics(model: dict) -> dict:
    midline = max(np.max(np.abs(model[side].eeg[0])) for side in (-1, 1))
    midline = max(float(midline), np.finfo(float).eps)
    contrast = (model[1].eeg[2] - model[1].eeg[1]) - (model[-1].eeg[2] - model[-1].eeg[1])
    peak = int(np.argmax(np.abs(contrast)))
    swapped = np.vstack([model[-1].eeg[0], model[-1].eeg[2], model[-1].eeg[1]])
    return {
        "condition_contrast_peak": float(np.max(np.abs(contrast))),
        "contrast_to_fz": float(np.max(np.abs(contrast)) / midline),
        "mirror_residual": float(np.max(np.abs(model[1].eeg - swapped))),
        "mirror_residual_to_fz": float(np.max(np.abs(model[1].eeg - swapped)) / midline),
        "fz_condition_to_fz": float(np.max(np.abs(model[1].eeg[0] - model[-1].eeg[0])) / midline),
        "lateralization_sign": int(np.sign(contrast[peak])),
        "fz_peak": midline,
    }


def model_diagnostics(parameters: ModelParameters, times: np.ndarray, rate: float, grid_size: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for mode in ("collapsed", "shape"):
        model = {side: simulate(side, times, rate, parameters, grid_size, drive_mode=mode) for side in (-1, 1)}
        rows.append({"projection": mode, **_lateralization_metrics(model)})
    return pd.DataFrame(rows), geometry_sensitivity(parameters, times, rate, grid_size)


def geometry_sensitivity(parameters: ModelParameters, times: np.ndarray, rate: float, grid_size: int) -> pd.DataFrame:
    rows = []
    for shift in (0.0, -0.2, 0.2):
        model = {side: simulate(side, times, rate, parameters, grid_size, geometry_shift=shift) for side in (-1, 1)}
        rows.append({"geometry_shift": shift, "projection": "shape", **_lateralization_metrics(model)})
    return pd.DataFrame(rows)


def group_level_tests(dataset, permutations: int, rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    rows = []
    windows = {}
    for lock, epochs, labels, tasks, groups in (("cue", dataset.epochs, dataset.labels, dataset.tasks, dataset.groups), ("target", dataset.target_epochs, dataset.target_labels, dataset.target_tasks, dataset.target_groups)):
        time_mask = (dataset.times >= 0) & (dataset.times <= 0.6)
        times = dataset.times[time_mask]
        for task in (1, 2):
            for feature in ("F3", "F4", "F4-F3", "normalized_F4-F3", "theta_ITPC", "alpha_ITPC"):
                values = []
                band = (4.0, 8.0) if feature == "theta_ITPC" else (8.0, 12.0)
                if feature in {"theta_ITPC", "alpha_ITPC"}:
                    sos = butter(3, band, btype="bandpass", fs=256.0, output="sos")
                    filtered = sosfiltfilt(sos, epochs, axis=-1)
                    phase = np.angle(hilbert(filtered, axis=-1))
                for group in np.unique(groups[tasks == task]):
                    selected = (tasks == task) & (groups == group)
                    left = epochs[selected & (labels == -1)]
                    right = epochs[selected & (labels == 1)]
                    if not len(left) or not len(right):
                        continue
                    if feature in {"theta_ITPC", "alpha_ITPC"}:
                        right_phase = phase[selected & (labels == 1)]
                        left_phase = phase[selected & (labels == -1)]
                        right_itpc = np.abs(np.exp(1j * right_phase).mean(axis=0)).mean(axis=0)
                        left_itpc = np.abs(np.exp(1j * left_phase).mean(axis=0)).mean(axis=0)
                        signal = right_itpc - left_itpc
                    elif feature == "F3":
                        signal = right[:, 1].mean(axis=0) - left[:, 1].mean(axis=0)
                    elif feature == "F4":
                        signal = right[:, 2].mean(axis=0) - left[:, 2].mean(axis=0)
                    elif feature == "F4-F3":
                        signal = (right[:, 2] - right[:, 1]).mean(axis=0) - (left[:, 2] - left[:, 1]).mean(axis=0)
                    else:
                        right_norm = (right[:, 2] - right[:, 1]) / np.maximum(np.sqrt((right[:, 1] ** 2 + right[:, 2] ** 2) / 2), 1e-6)
                        left_norm = (left[:, 2] - left[:, 1]) / np.maximum(np.sqrt((left[:, 1] ** 2 + left[:, 2] ** 2) / 2), 1e-6)
                        signal = right_norm.mean(axis=0) - left_norm.mean(axis=0)
                    values.append(signal[time_mask])
                if len(values) >= 2:
                    clusters = cluster_sign_test(np.asarray(values), permutations, rng, alpha=0.05 / 24)
                    if clusters:
                        windows[(lock, task, feature)] = [(int(a), int(b), float(p)) for a, b, p in clusters]
                    rows.append({"lock": lock, "task": task, "feature": feature, "n_groups": len(values), "significant_clusters": len(clusters), "clusters": ";".join(f"{times[a]:.3f}-{times[b-1]:.3f}:p={p:.4f}" for a, b, p in clusters), "max_abs_d": float(np.nanmax(np.abs(np.mean(values, axis=0)) / np.maximum(np.std(values, axis=0, ddof=1), 1e-9)))})
                else:
                    rows.append({"lock": lock, "task": task, "feature": feature, "n_groups": len(values), "significant_clusters": 0, "clusters": "insufficient groups", "max_abs_d": np.nan})
    return pd.DataFrame(rows), windows


def lateral_feature_gate(dataset, windows: dict) -> np.ndarray:
    index = np.flatnonzero((dataset.times >= 0) & (dataset.times <= 0.6))
    gate = np.zeros(len(dataset.times), dtype=bool)
    for (lock, _task, feature), clusters in windows.items():
        if lock != "cue" or feature not in {"F4-F3", "normalized_F4-F3"}:
            continue
        for start, stop, _p in clusters:
            gate[index[start:stop]] = True
    return gate


def positive_controls(dataset, permutations: int, rng: np.random.Generator) -> pd.DataFrame:
    mask = (dataset.times >= 0.05) & (dataset.times <= 0.6)
    cue_features = dataset.epochs[:, :, mask].mean(axis=2)
    task_result = classify(cue_features, (dataset.tasks == 2).astype(int), dataset.subjects, permutations, rng)[0]
    target = dataset.target_tasks == 1
    target_features = dataset.target_epochs[target][:, :, mask].mean(axis=2)
    location_result = classify(target_features, (dataset.target_labels[target] == 1).astype(int), dataset.target_subjects[target], permutations, rng)[0]
    return pd.DataFrame([{ "control": "task1_vs_task2", **task_result }, { "control": "task1_target_position", **location_result }])


def injection_control(dataset, prediction: dict, window: np.ndarray, amplitudes: tuple[float, ...], permutations: int, rng: np.random.Generator) -> pd.DataFrame:
    mask = (dataset.times >= 0.10) & (dataset.times <= 0.30)
    profile = np.hanning(int(mask.sum()))
    scale = float(np.sqrt(np.mean(dataset.epochs[:, 1:3] ** 2)))
    rows = []
    for amplitude in amplitudes:
        epochs = dataset.epochs.copy()
        epochs[:, 2, mask] += amplitude * scale * profile * dataset.labels[:, None]
        epochs[:, 1, mask] -= amplitude * scale * profile * dataset.labels[:, None]
        features, _ = feature_matrix(replace(dataset, epochs=epochs), prediction, window)
        rows.append({"injected_amplitude_rms": amplitude, **classify(features, dataset.labels, dataset.groups, permutations, rng)[0]})
    return pd.DataFrame(rows)


def decodability_bounds(dataset) -> pd.DataFrame:
    rows = []
    for lock, epochs, labels, tasks in (("cue", dataset.epochs, dataset.labels, dataset.tasks), ("target", dataset.target_epochs, dataset.target_labels, dataset.target_tasks)):
        for task in (1, 2):
            selected = tasks == task
            values = (epochs[selected, 2] - epochs[selected, 1])
            y = labels[selected]
            for start in np.arange(0, 0.6, 0.1):
                window = (dataset.times >= start) & (dataset.times < start + 0.1)
                per_trial = values[:, window].mean(axis=1)
                a, b = per_trial[y == -1], per_trial[y == 1]
                pooled = np.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) / max(len(a)+len(b)-2, 1))
                d = (b.mean() - a.mean()) / max(pooled, np.finfo(float).eps)
                rows.append({"lock": lock, "task": task, "window_start_s": start, "window_stop_s": start + 0.1, "cohens_d": d, "auc_upper_bound": float(ndtr(abs(d) / np.sqrt(2)))})
    return pd.DataFrame(rows)


def feature_matrix(dataset, prediction: dict, window: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model_mask = (dataset.times >= 0) & (dataset.times <= 0.6)
    template = prediction[1][:, model_mask] - prediction[-1][:, model_mask]
    template /= max(np.linalg.norm(template), np.finfo(float).eps)
    matched = np.einsum("nct,ct->n", dataset.epochs[:, :, model_mask], template)
    f3 = dataset.epochs[:, 1, window].mean(axis=1)
    f4 = dataset.epochs[:, 2, window].mean(axis=1)
    scale = np.sqrt(np.mean(dataset.epochs[:, 1:3, window] ** 2, axis=(1, 2)))
    lateral = (f4 - f3) / np.maximum(scale, np.finfo(float).eps)
    return np.column_stack([matched, lateral]), np.column_stack([matched])


def classify(features: np.ndarray, labels: np.ndarray, groups: np.ndarray, permutations: int, rng: np.random.Generator) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    splitter = StratifiedGroupKFold(n_splits=len(np.unique(groups)), shuffle=True, random_state=20260923)

    def predict(y):
        probability = np.zeros(len(y))
        prediction = np.zeros(len(y), dtype=int)
        for train, test in splitter.split(features, y, groups):
            model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
            model.fit(features[train], y[train])
            probability[test] = model.predict_proba(features[test])[:, 1]
            prediction[test] = model.predict(features[test])
        return probability, prediction

    binary = (labels == 1).astype(int)
    probability, predicted = predict(binary)
    auc = roc_auc_score(binary, probability)
    accuracy = accuracy_score(binary, predicted)
    null = np.zeros(permutations)
    for i in range(permutations):
        shuffled = binary.copy()
        for group in np.unique(groups):
            selected = groups == group
            shuffled[selected] = rng.permutation(shuffled[selected])
        null[i] = roc_auc_score(shuffled, predict(shuffled)[0])
    fpr, tpr, _ = roc_curve(binary, probability)
    result = {"auc": auc, "accuracy": accuracy, "p_value": (1 + np.sum(null >= auc)) / (permutations + 1), "n_trials": len(labels), "folds": len(np.unique(groups))}
    return result, fpr, tpr, null


def parameter_table(parameters: ModelParameters, kernels: list[dict]) -> pd.DataFrame:
    meanings = {
        "delay": "LGN-cortex delay (s)", "coupling": "spatial excitatory coupling", "input_gain": "visual drive gain",
        "tau_e": "excitatory time constant (s)", "tau_i": "inhibitory time constant (s)", "sigmoid_gain": "population response gain",
        "w_ee": "E-to-E coupling", "w_ei": "I-to-E coupling", "w_ie": "E-to-I coupling", "w_ii": "I-to-I coupling",
    }
    rows = [{"parameter": name, "meaning": meanings[name], "value": value, "source": "calibrated" if name in {"delay", "coupling", "input_gain"} else "physiological prior"} for name, value in asdict(parameters).items()]
    rows.extend({"parameter": f"kernel_{name}", "meaning": f"{name} causal observation-kernel norm", "value": np.linalg.norm(kernel["coef"]), "source": "split-half calibration"} for name, kernel in zip(("Fz", "F3", "F4"), kernels, strict=True))
    return pd.DataFrame(rows)
