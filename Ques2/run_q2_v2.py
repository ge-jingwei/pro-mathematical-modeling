"""Question 2: LGN-to-scalp mechanism, lateralisation and discriminative features."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.signal import welch
from scipy.special import ndtr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from neuro.accel import device, field_response_batch
from neuro.dataset import Epochs, load
from neuro.dynamics import FieldParameters, kuramoto, micro_transfer, normalised_transfer, order_parameter, regional_labels, regional_signals
from neuro.encoding import EncodingParameters, cortical_density, encode
from neuro.forward import LeadFieldParameters, forward, lead_field

GRID = 24
ANCHOR = 0.0
SPACING = 0.25
MAP_SIGMA = 0.25
WINDOW = (0.0, 0.75)
LOCKS = ("cue", "target")


def robust_mean(values: np.ndarray, fraction: float = 0.2) -> np.ndarray:
    """Trimmed mean: this dataset carries heavy-tailed artefacts that bias the mean."""
    trim = int(values.shape[0] * fraction)
    ordered = np.sort(values, axis=0)
    return ordered[trim: values.shape[0] - trim].mean(axis=0)


def locked(dataset: Epochs, lock: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if lock == "cue":
        return dataset.cue, dataset.cue_side, dataset.task, dataset.file
    return dataset.target, dataset.target_side, dataset.target_task, dataset.target_file


def erp_table(dataset: Epochs, lock: str) -> dict:
    epochs, side, task, _ = locked(dataset, lock)
    return {(int(item), int(condition)): robust_mean(epochs[(task == item) & (side == condition)]) for item in np.unique(task) for condition in (-1, 1)}


def densities(parameters: EncodingParameters, anchor: float = ANCHOR, spacing: float = SPACING) -> dict:
    return {side: cortical_density(encode(side, parameters), GRID, anchor, spacing, MAP_SIGMA) for side in (-1, 1)}


def predict(density: dict, times: np.ndarray, excitatory, inhibitory, fields: np.ndarray, parameters: FieldParameters) -> dict:
    return {side: forward(density[side], times, excitatory, inhibitory, fields, parameters) for side in (-1, 1)}


def correlation(observed: np.ndarray, predicted: np.ndarray, times: np.ndarray) -> float:
    mask = (times >= WINDOW[0]) & (times <= WINDOW[1])
    values = [np.corrcoef(observed[channel, mask], predicted[channel, mask])[0, 1] for channel in range(len(observed))]
    return float(np.nanmean(values))


def cosine(observed: np.ndarray, predicted: np.ndarray, rate: float) -> float:
    values = []
    for channel in range(len(observed)):
        _, first = welch(observed[channel], rate, nperseg=min(128, observed.shape[-1]))
        _, second = welch(predicted[channel], rate, nperseg=min(128, predicted.shape[-1]))
        values.append(float(first @ second / np.sqrt((first @ first) * (second @ second))))
    return float(np.nanmean(values))


def calibrate(table: dict, times: np.ndarray, excitatory, inhibitory, fields: np.ndarray, density: dict, task: int) -> tuple[FieldParameters, pd.DataFrame]:
    """Sweep the field parameters in one batched call (GPU when available)."""
    levels = {
        "delay": (0.02, 0.04, 0.06, 0.08),
        "gain": (0.3, 0.6, 1.0, 1.5),
        "feedback_gain": (0.8, 1.2, 1.8, 2.4),
        "feedback_inhibition": (0.5, 1.0, 1.5, 2.0, 2.5),
        "feedback_delay": (0.30, 0.36, 0.42, 0.48),
        "delay_gradient": (0.0, 0.05),
    }
    names = list(levels)
    combinations = list(itertools.product(*(levels[name] for name in names)))
    sweep = {name: np.array([row[index] for row in combinations]) for index, name in enumerate(names)}
    rows = []
    for side in (-1, 1):
        potentials = field_response_batch(density[side], times, excitatory, inhibitory, FieldParameters(), sweep, fields=fields)
        mask = (times >= WINDOW[0]) & (times <= WINDOW[1])
        observed = table[(task, side)][:, mask]
        for index, configuration in enumerate(combinations):
            predicted = potentials[index][:, mask]
            values = [np.corrcoef(observed[channel], predicted[channel])[0, 1] for channel in range(3)]
            rows.append({**dict(zip(names, configuration)), "side": side, "correlation": float(np.nanmean(values))})
    frame = pd.DataFrame(rows).pivot_table(index=names, columns="side", values="correlation").reset_index()
    frame.columns = [*names, "r_side_left", "r_side_right"]
    frame["objective"] = frame[["r_side_left", "r_side_right"]].mean(axis=1).abs()
    best = frame.sort_values("objective", ascending=False).iloc[0]
    chosen = FieldParameters(**{name: float(best[name]) for name in names})
    return chosen, frame


def polarity(table: dict, prediction: dict, times: np.ndarray) -> float:
    values = [correlation(table[(1, side)], prediction[side], times) for side in (-1, 1)]
    return float(np.sign(np.mean(values)) or 1.0)


def shams(times: np.ndarray, density: dict, excitatory, inhibitory, fields: np.ndarray, parameters: FieldParameters, seed: int = 20260923) -> dict:
    """Ablations: removing one model ingredient isolates its contribution."""
    from dataclasses import replace

    rng = np.random.default_rng(seed)
    result = {"full": predict(density, times, excitatory, inhibitory, fields, parameters)}
    result["no_feedback"] = predict(density, times, excitatory, inhibitory, fields, replace(parameters, feedback_gain=0.0))
    result["no_adaptation"] = predict(density, times, excitatory, inhibitory, fields, replace(parameters, adaptation=0.0))
    result["no_shape"] = predict({side: np.ones((GRID, GRID)) for side in (-1, 1)}, times, excitatory, inhibitory, fields, parameters)
    result["random_shape"] = predict({side: np.clip(rng.normal(size=(GRID, GRID)), 0.0, None) for side in (-1, 1)}, times, excitatory, inhibitory, fields, parameters)
    return result


def lateralisation(density: dict, excitatory, inhibitory, fields: np.ndarray, parameters: FieldParameters, times: np.ndarray) -> dict:
    prediction = predict(density, times, excitatory, inhibitory, fields, parameters)
    mask = (times >= WINDOW[0]) & (times <= WINDOW[1])
    contrast = ((prediction[1][2] - prediction[1][1]) - (prediction[-1][2] - prediction[-1][1]))[mask]
    midline = max(float(np.abs(prediction[1][0][mask]).max()), float(np.abs(prediction[-1][0][mask]).max()), 1e-12)
    mirror = (prediction[1] - prediction[-1][[0, 2, 1]])[:, mask]
    return {
        "contrast_peak": float(np.abs(contrast).max()),
        "contrast_to_midline": float(np.abs(contrast).max() / midline),
        "mirror_residual": float(np.abs(mirror).max()),
        "mirror_residual_to_midline": float(np.abs(mirror).max() / midline),
        "midline_condition_gap": float(np.abs((prediction[1][0] - prediction[-1][0])[mask]).max() / midline),
        "sign": int(np.sign(contrast[np.argmax(np.abs(contrast))])),
    }


def layout_scan(excitatory, inhibitory, fields: np.ndarray, parameters: FieldParameters, times: np.ndarray, encoding: EncodingParameters) -> pd.DataFrame:
    rows = []
    for anchor in np.round(np.arange(-0.4, 0.41, 0.1), 2):
        for spacing in (0.0, 0.15, 0.3, 0.45):
            rows.append({"anchor": float(anchor), "spacing": float(spacing), **lateralisation(densities(encoding, float(anchor), float(spacing)), excitatory, inhibitory, fields, parameters, times)})
    return pd.DataFrame(rows)


def lead_sensitivity(excitatory, inhibitory, parameters: FieldParameters, times: np.ndarray, encoding: EncodingParameters) -> pd.DataFrame:
    rows = []
    base = LeadFieldParameters()
    for name in ("spread", "depth", "anterior"):
        for factor in (0.8, 1.0, 1.2):
            settings = {**base.__dict__, name: base.__dict__[name] * factor}
            rows.append({"parameter": name, "factor": factor, **lateralisation(densities(encoding), excitatory, inhibitory, lead_field(GRID, LeadFieldParameters(**settings)), parameters, times)})
    return pd.DataFrame(rows)


def observation_kernel(table: dict, prediction: dict, times: np.ndarray) -> pd.DataFrame:
    """Negative control: a flexible ridge lag kernel is fitted to the measured ERP.

    A large fitted correlation here says nothing about the model, because the
    kernel has enough freedom to fit a sham drive equally well.  It is kept as a
    documented contrast to the parameter-free forward model.
    """
    mask = (times >= WINDOW[0]) & (times <= WINDOW[1])
    rows = []
    for maximum in (60, 160):
        for alpha in (1.0, 0.1, 0.01):
            values = []
            for task in (1, 2):
                for side in (-1, 1):
                    for channel in range(3):
                        column = prediction[side][channel]
                        design = np.column_stack([np.r_[np.zeros(lag), column[: len(column) - lag]] for lag in range(0, maximum + 1, 4)])
                        target = table[(task, side)][channel]
                        fitted = Ridge(alpha=alpha).fit(design[mask], target[mask])
                        values.append(float(np.corrcoef(target[mask], fitted.predict(design[mask]))[0, 1]))
            rows.append({"maximum_lag": maximum, "alpha": alpha, "fitted_correlation": float(np.mean(values))})
    return pd.DataFrame(rows)


def _spans(statistic: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    active = np.abs(statistic) >= threshold
    edges = np.flatnonzero(np.diff(np.r_[False, active, False]))
    return [(int(a), int(b)) for a, b in edges.reshape(-1, 2)]


def cluster_test(right: np.ndarray, left: np.ndarray, times: np.ndarray, permutations: int, rng: np.random.Generator, threshold: float = 1.96) -> dict:
    """Two-sample cluster-mass permutation test on the single-trial lateralisation."""

    def statistic(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        spread = np.sqrt(first.var(axis=0, ddof=1) / len(first) + second.var(axis=0, ddof=1) / len(second))
        return (second.mean(axis=0) - first.mean(axis=0)) / np.maximum(spread, 1e-12)

    observed = statistic(right, left)
    spans = _spans(observed, threshold)
    masses = [float(np.abs(observed[a:b]).sum()) for a, b in spans]
    best = int(np.argmax(masses)) if masses else None
    mass = masses[best] if best is not None else 0.0
    joined = np.r_[right, left]
    count = len(right)
    null = np.zeros(permutations)
    for draw in range(permutations):
        shuffled = joined[rng.permutation(len(joined))]
        value = statistic(shuffled[:count], shuffled[count:])
        null[draw] = max([float(np.abs(value[a:b]).sum()) for a, b in _spans(value, threshold)], default=0.0)
    return {
        "n_left": int(len(left)),
        "n_right": int(len(right)),
        "max_abs_d": float(np.max(np.abs(observed))),
        "cluster_mass": mass,
        "p_value": float((1.0 + np.sum(null >= mass)) / (permutations + 1.0)),
        "start_s": float(times[spans[best][0]]) if best is not None else np.nan,
        "stop_s": float(times[spans[best][1] - 1]) if best is not None else np.nan,
    }


def group_tests(dataset: Epochs, times: np.ndarray, permutations: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    mask = (times >= 0.0) & (times <= 0.6)
    for lock in LOCKS:
        epochs, side, task, _ = locked(dataset, lock)
        lateral = epochs[:, 2] - epochs[:, 1]
        for item in (1, 2):
            selected = (task == item) & mask[None, :] if False else task == item
            left = lateral[selected & (side == -1)][:, mask]
            right = lateral[selected & (side == 1)][:, mask]
            rows.append({"lock": lock, "task": item, **cluster_test(right, left, times[mask], permutations, rng)})
    return pd.DataFrame(rows)


def features(epochs: np.ndarray, prediction: dict, times: np.ndarray, polarity: float) -> np.ndarray:
    mask = (times >= 0.05) & (times <= 0.6)
    template = polarity * (prediction[1][:, mask] - prediction[-1][:, mask])
    template = template / max(np.linalg.norm(template), 1e-12)
    matched = np.einsum("nct,ct->n", epochs[:, :, mask], template)
    lateral = (epochs[:, 2] - epochs[:, 1])[:, mask]
    lateral_template = polarity * (prediction[1][2] - prediction[1][1] - prediction[-1][2] + prediction[-1][1])[mask]
    projection = lateral @ lateral_template / max(np.linalg.norm(lateral_template), 1e-12)
    peak = np.abs(lateral).max(axis=1) * np.sign(lateral[:, np.argmax(np.abs(lateral_template))])
    return np.column_stack([matched, projection, peak])


def classify(matrix: np.ndarray, labels: np.ndarray, groups: np.ndarray, permutations: int, rng: np.random.Generator) -> dict:
    binary = (labels == 1).astype(int)
    folds = len(np.unique(groups))
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=20260923)
    probability = np.zeros(len(binary))
    predicted = np.zeros(len(binary), dtype=int)
    for train, test in splitter.split(matrix, binary, groups):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        model.fit(matrix[train], binary[train])
        probability[test] = model.predict_proba(matrix[test])[:, 1]
        predicted[test] = model.predict(matrix[test])
    from sklearn.metrics import accuracy_score, roc_auc_score
    auc = roc_auc_score(binary, probability)
    accuracy = accuracy_score(binary, predicted)
    null = np.zeros(permutations)
    for draw in range(permutations):
        shuffled = binary.copy()
        for group in np.unique(groups):
            selected = groups == group
            shuffled[selected] = rng.permutation(shuffled[selected])
        null[draw] = roc_auc_score(shuffled, _predict(matrix, shuffled, groups, splitter))
    return {"auc": float(auc), "accuracy": float(accuracy), "p_value": float((1 + np.sum(null >= auc)) / (permutations + 1)), "n_trials": int(len(binary)), "folds": int(folds)}


def _predict(matrix: np.ndarray, labels: np.ndarray, groups: np.ndarray, splitter) -> np.ndarray:
    probability = np.zeros(len(labels))
    for train, test in splitter.split(matrix, labels, groups):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        model.fit(matrix[train], labels[train])
        probability[test] = model.predict_proba(matrix[test])[:, 1]
    return probability


def bound(epochs: np.ndarray, side: np.ndarray, task: np.ndarray, times: np.ndarray) -> pd.DataFrame:
    rows = []
    lateral = epochs[:, 2] - epochs[:, 1]
    for item in (1, 2):
        for start in np.arange(0.0, 0.6, 0.1):
            window = (times >= start) & (times < start + 0.1)
            values = lateral[task == item][:, window].mean(axis=1)
            labels = side[task == item]
            first, second = values[labels == -1], values[labels == 1]
            pooled = np.sqrt(((len(first) - 1) * first.var(ddof=1) + (len(second) - 1) * second.var(ddof=1)) / max(len(first) + len(second) - 2, 1))
            effect = (second.mean() - first.mean()) / max(pooled, 1e-12)
            rows.append({"task": item, "window_start_s": float(start), "cohens_d": float(effect), "auc_upper_bound": float(ndtr(abs(effect) / np.sqrt(2)))})
    return pd.DataFrame(rows)


def macro(density: dict, times: np.ndarray, excitatory, inhibitory, parameters: FieldParameters, rate: float) -> dict:
    """Regional Kuramoto order parameter, from the field and from an explicit oscillator network."""
    from neuro.dynamics import field_response

    labels = regional_labels(GRID, 4)
    regions = np.unique(labels)
    axis = np.linspace(-1.0, 1.0, GRID)
    grid_x, grid_y = np.meshgrid(axis, axis)
    centres = np.array([[grid_x[labels == region].mean(), grid_y[labels == region].mean()] for region in regions])
    distance = np.linalg.norm(centres[:, None, :] - centres[None, :, :], axis=-1)
    coupling = np.exp(-(distance**2) / (2.0 * 0.6**2))
    np.fill_diagonal(coupling, 0.0)
    network_delays = distance / 6.0
    frequencies = 10.0 + np.linspace(-1.5, 1.5, len(regions))
    phases = kuramoto(times, frequencies, 0.6 * coupling, network_delays, rate)
    result = {"kuramoto": np.abs(np.exp(1j * phases).mean(axis=0)), "network_delays": network_delays}
    for side in (-1, 1):
        source = field_response(density[side], times, excitatory, inhibitory, parameters)
        signals = regional_signals(source, labels)
        radius, _ = order_parameter(signals, rate)
        result[f"order_{side}"] = radius
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cti.yaml")
    parser.add_argument("--permutations", type=int, default=1000)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    with (root / args.config).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    output = root / "outputs" / "v2" / "q2"
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config["random_seed"])
    rate = float(config["sample_rate"])
    dataset = load(root)
    times = dataset.times
    encoding = EncodingParameters()
    excitatory, inhibitory, *_ = micro_transfer()
    excitatory, inhibitory = normalised_transfer(excitatory, inhibitory)
    fields = lead_field(GRID)
    density = densities(encoding)
    cue_table = erp_table(dataset, "cue")
    parameters = {}
    for task in (1, 2):
        parameters[task], grid = calibrate(cue_table, times, excitatory, inhibitory, fields, density, task)
        grid.to_csv(output / f"calibration_task{task}.csv", index=False)
    prediction = {task: predict(density, times, excitatory, inhibitory, fields, parameters[task]) for task in (1, 2)}
    sign = polarity(cue_table, prediction[1], times)

    rows = []
    for source_task, params in parameters.items():
        signed = {side: sign * value for side, value in prediction[source_task].items()}
        for lock in LOCKS:
            table = cue_table if lock == "cue" else erp_table(dataset, "target")
            for task in (1, 2):
                for side in (-1, 1):
                    for channel, name in enumerate(("Fz", "F3", "F4")):
                        rows.append({"calibrated_on": source_task, "lock": lock, "task": task, "side": side, "channel": name, "correlation": correlation(table[(task, side)][channel][None, :], signed[side][channel][None, :], times), "spectral_cosine": cosine(table[(task, side)][channel][None, :], signed[side][channel][None, :], rate)})
    validation = pd.DataFrame(rows)
    validation.to_csv(output / "validation.csv", index=False)

    sham_rows = []
    for task, params in parameters.items():
        sham = shams(times, density, excitatory, inhibitory, fields, params)
        for name, value in sham.items():
            sham_rows.append({"task": task, "model": name, "correlation": float(np.mean([correlation(cue_table[(task, side)], sign * value[side], times) for side in (-1, 1)]))})
    sham_table = pd.DataFrame(sham_rows)
    sham_table.to_csv(output / "shams.csv", index=False)

    lateral = lateralisation(density, excitatory, inhibitory, fields, parameters[1], times)
    layout = layout_scan(excitatory, inhibitory, fields, parameters[1], times, encoding)
    layout.to_csv(output / "layout_scan.csv", index=False)
    sensitivity = lead_sensitivity(excitatory, inhibitory, parameters[1], times, encoding)
    sensitivity.to_csv(output / "lead_sensitivity.csv", index=False)
    kernel = observation_kernel(cue_table, prediction[1], times)
    kernel.to_csv(output / "observation_kernel.csv", index=False)

    tests = group_tests(dataset, times, args.permutations, rng)
    tests.to_csv(output / "group_tests.csv", index=False)
    signed_model = {side: sign * value for side, value in prediction[1].items()}
    matrix = features(dataset.cue, signed_model, times, sign)
    classifier = classify(matrix, dataset.cue_side, dataset.file, 200, rng)
    bounds = bound(dataset.cue, dataset.cue_side, dataset.task, times)
    bounds.to_csv(output / "bounds.csv", index=False)
    order = macro(density, times, excitatory, inhibitory, parameters[1], rate)
    np.savez_compressed(output / "order.npz", left=order["order_-1"], right=order["order_1"], kuramoto=order["kuramoto"], network_delays=order["network_delays"])

    summary = {"parameters_task1": parameters[1].__dict__, "parameters_task2": parameters[2].__dict__, "polarity": sign, "lateralisation": lateral, "classifier": classifier}
    pd.Series({**{f"p1_{key}": value for key, value in summary["parameters_task1"].items()}, **{f"p2_{key}": value for key, value in summary["parameters_task2"].items()}, "polarity": sign, **{f"lateral_{key}": value for key, value in lateral.items()}, **{f"classifier_{key}": value for key, value in classifier.items()}}).to_csv(output / "summary.csv")
    print("task1", pd.Series(parameters[1].__dict__).round(4).to_string())
    print("task2", pd.Series(parameters[2].__dict__).round(4).to_string())
    print("polarity", sign)
    print(validation.groupby(["calibrated_on", "task"])["correlation"].mean().round(3).to_string())
    print(sham_table.round(3).to_string(index=False))
    print(pd.Series(lateral).round(4).to_string())
    print(tests.round(4).to_string(index=False))
    print(pd.Series(classifier).to_string())


if __name__ == "__main__":
    main()
