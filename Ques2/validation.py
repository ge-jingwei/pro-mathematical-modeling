from dataclasses import asdict

import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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


def fit_observation_kernel(model: dict, reference: dict, times: np.ndarray) -> tuple[dict, list[dict]]:
    mask = (times >= 0) & (times <= 0.75)
    kernels = []
    for channel in range(3):
        design = np.vstack([_lag_design(model[side].eeg[channel])[mask] for task in (1, 2) for side in (-1, 1)])
        target = np.concatenate([reference[(task, side)]["mean"][channel, mask] for task in (1, 2) for side in (-1, 1)])
        scale = np.linalg.norm(design, axis=0)
        scale[scale == 0] = 1.0
        fitted = Ridge(alpha=0.1).fit(design / scale, target)
        kernels.append({"coef": fitted.coef_, "intercept": fitted.intercept_, "scale": scale})
    return apply_observation_kernel(model, kernels, times), kernels


def apply_observation_kernel(model: dict, kernels: list[dict], times: np.ndarray) -> dict:
    result = {}
    for side in (-1, 1):
        channels = []
        for channel, kernel in enumerate(kernels):
            design = _lag_design(model[side].eeg[channel]) / kernel["scale"]
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


def observation_cross_validation(dataset, model: dict, rate: float, folds: int = 5) -> pd.DataFrame:
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
            train[key] = {"mean": dataset.epochs[train_indices].mean(axis=0)}
            test[key] = {"mean": dataset.epochs[test_indices].mean(axis=0)}
        prediction, _ = fit_observation_kernel(model, train, dataset.times)
        window, _, _ = lateral_window(prediction, dataset.times)
        for task in (1, 2):
            rows.append({"task": task, "fold": fold + 1, **validation_metrics(prediction, test, dataset.times, task, rate), "sign_agreement": sign_agreement(prediction, test, dataset.times, task, window)})
    frame = pd.DataFrame(rows)
    return frame.groupby("task", as_index=False).agg(waveform_correlation=("waveform_correlation", "mean"), waveform_correlation_sd=("waveform_correlation", "std"), spectral_cosine=("spectral_cosine", "mean"), spectral_cosine_sd=("spectral_cosine", "std"), sign_agreement=("sign_agreement", "mean"))


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
    best = None
    grid = config["q2"]["calibration"]
    for delay in grid["delay"]:
        for coupling in grid["coupling"]:
            for input_gain in grid["input_gain"]:
                parameters = ModelParameters(delay=float(delay), coupling=float(coupling), input_gain=float(input_gain))
                model = {side: simulate(side, times, rate, parameters, config["q2"]["grid_size"]) for side in (-1, 1)}
                scales = _scales(model, erp, times, 1)
                prediction = calibrated_prediction(model, scales)
                metrics = validation_metrics(prediction, erp, times, 1, rate)
                row = {"delay_ms": delay * 1000, "coupling": coupling, "input_gain": input_gain, **metrics}
                rows.append(row)
                if best is None or metrics["waveform_correlation"] > best[0]:
                    best = (metrics["waveform_correlation"], parameters, model, scales)
    return best[1], best[2], best[3], pd.DataFrame(rows)


def sensitivity(parameters: ModelParameters, kernels: list[dict], erp: dict, times: np.ndarray, rate: float, config: dict) -> pd.DataFrame:
    rows = []
    for name in ("delay", "coupling", "input_gain", "tau_e", "tau_i"):
        for factor in (0.8, 1.2):
            current = vary(parameters, name, factor)
            model = {side: simulate(side, times, rate, current, config["q2"]["grid_size"]) for side in (-1, 1)}
            prediction = apply_observation_kernel(model, kernels, times)
            metrics = validation_metrics(prediction, erp, times, 2, rate)
            mask, _, _ = lateral_window(prediction, times)
            rows.append({"parameter": name, "factor": factor, "value": getattr(current, name), **metrics, "sign_agreement": sign_agreement(prediction, erp, times, 2, mask)})
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
