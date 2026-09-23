"""Question 1 v2: robust preprocessing, model-matched denoising, P3a extraction."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import curve_fit
from scipy.stats import t as t_dist

import pywt

from neuro.dataset import Epochs, load
from neuro.dynamics import FieldParameters, micro_transfer, normalised_transfer
from neuro.encoding import EncodingParameters, cortical_density, encode
from neuro.forward import forward, lead_field

CHANNELS = ("Fz", "F3", "F4")
GRID = 24
PRIMARY = (0.25, 0.5)


def trimmed(values: np.ndarray, fraction: float = 0.2) -> np.ndarray:
    """Trimmed mean, robust to the heavy-tailed artefacts in this dataset."""
    trim = int(values.shape[0] * fraction)
    ordered = np.sort(values, axis=0)
    return ordered[trim: values.shape[0] - trim].mean(axis=0)


def wavelet_shrink(values: np.ndarray, alpha: float = 48.0) -> np.ndarray:
    """Single-channel wavelet denoising, kept as a comparison method."""
    output = np.empty_like(values)
    for index, trial in enumerate(values):
        coefficients = pywt.wavedec(trial, "sym6", mode="symmetric", level=4)
        sigma = np.median(np.abs(coefficients[-1] - np.median(coefficients[-1]))) / 0.6745
        threshold = alpha * sigma * np.sqrt(2.0 * np.log(trial.size))
        shrunk = [coefficients[0], *[pywt.threshold(part, threshold, mode="soft") for part in coefficients[1:]]]
        output[index] = pywt.waverec(shrunk, "sym6", mode="symmetric")[: trial.size]
    return output


def split_half_reliability(values: np.ndarray, seed: int = 0) -> float:
    """Correlation between the ERPs of two random halves: higher means cleaner."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(values))
    first, second = order[: len(order) // 2], order[len(order) // 2:]
    left, right = values[first].mean(axis=0), values[second].mean(axis=0)
    return float(np.mean([np.corrcoef(left[channel], right[channel])[0, 1] for channel in range(values.shape[1])]))


def shape_fidelity(reference: np.ndarray, candidate: np.ndarray) -> float:
    return float(np.mean([np.corrcoef(reference[channel], candidate[channel])[0, 1] for channel in range(reference.shape[0])]))


def discriminability(values: np.ndarray, sides: np.ndarray, templates: np.ndarray) -> float:
    """Effect size of left versus right along the model's shape direction."""
    direction = (templates[0] - templates[1]).ravel()
    scores = values.reshape(len(values), -1) @ direction
    left, right = scores[sides == -1], scores[sides == 1]
    pooled = np.sqrt(((len(left) - 1) * left.var(ddof=1) + (len(right) - 1) * right.var(ddof=1)) / max(len(left) + len(right) - 2, 1))
    return float((right.mean() - left.mean()) / max(pooled, 1e-12))


def matched_filter_denoise(values: np.ndarray, templates: np.ndarray) -> np.ndarray:
    """Project each trial onto the model's common and difference templates."""
    common = templates[0] + templates[1]
    difference = templates[0] - templates[1]
    basis = np.stack([common.ravel(), difference.ravel()])
    gram = basis @ basis.T
    gram_inv = np.linalg.inv(gram + 1e-6 * np.eye(2))
    flat = values.reshape(len(values), -1)
    coefficients = flat @ basis.T @ gram_inv
    return (coefficients @ basis).reshape(values.shape)


def cluster_vs_zero(epochs: np.ndarray, times: np.ndarray, permutations: int, rng: np.random.Generator) -> pd.DataFrame:
    """One-sample cluster permutation test of the ERP against baseline."""
    rows = []
    mask = (times >= 0.0) & (times <= 0.6)
    window_times = times[mask]
    for channel, name in enumerate(CHANNELS):
        data = epochs[:, channel, mask]
        statistic = data.mean(axis=0) / (data.std(axis=0, ddof=1) / np.sqrt(len(data)) + 1e-12)
        threshold = float(t_dist.ppf(0.975, len(data) - 1))
        spans = _spans(statistic, threshold)
        masses = [float(np.abs(statistic[a:b]).sum()) for a, b in spans]
        null = np.zeros(permutations)
        for draw in range(permutations):
            signed = data * rng.choice((-1.0, 1.0), size=(len(data), 1))
            value = signed.mean(axis=0) / (signed.std(axis=0, ddof=1) / np.sqrt(len(data)) + 1e-12)
            null[draw] = max([float(np.abs(value[a:b]).sum()) for a, b in _spans(value, threshold)], default=0.0)
        for index, (start, stop) in enumerate(spans):
            rows.append({"channel": name, "start_ms": float(window_times[start] * 1000), "stop_ms": float(window_times[stop - 1] * 1000), "mass": masses[index], "p_value": float((1.0 + np.sum(null >= masses[index])) / (permutations + 1.0))})
    return pd.DataFrame(rows)


def _spans(statistic: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    active = np.abs(statistic) >= threshold
    edges = np.flatnonzero(np.diff(np.r_[False, active, False]))
    return [(int(a), int(b)) for a, b in edges.reshape(-1, 2)]


def gamma_wave(t, amplitude, onset, tau, shape, offset):
    scaled = np.clip((np.asarray(t) - onset) / tau, 1e-3, 30.0)
    return offset + amplitude * np.exp(shape * (np.log(scaled) - scaled + 1.0))


def fit_p3a(erp: np.ndarray, times: np.ndarray) -> dict:
    """Fit an asymmetric Gamma wave to the P3a window (250-500 ms)."""
    window = (times >= PRIMARY[0]) & (times <= PRIMARY[1])
    t = times[window]
    y = erp[window]
    amplitude = float(np.max(np.abs(y - y.mean()))) or 1.0
    guess = [amplitude, 0.25, 0.05, 2.0, float(y.mean())]
    bounds = ([-5 * amplitude, 0.2, 0.01, 0.5, -amplitude], [5 * amplitude, 0.5, 0.3, 6.0, amplitude])
    try:
        fitted, _ = curve_fit(gamma_wave, t, y, p0=guess, bounds=bounds, maxfev=20000)
    except Exception:
        return {"r2": np.nan, "amplitude": np.nan, "latency_ms": np.nan, "width_ms": np.nan}
    prediction = gamma_wave(t, *fitted)
    residual = y - prediction
    r2 = 1.0 - float(residual @ residual / max(float(np.sum((y - y.mean()) ** 2)), 1e-12))
    peak_time = float(fitted[1] + fitted[2] * fitted[3])
    return {"r2": float(r2), "amplitude": float(fitted[0]), "latency_ms": float(peak_time * 1000), "width_ms": float(fitted[2] * 1000)}


def templates(density: dict, times: np.ndarray, excitatory, inhibitory, fields: np.ndarray) -> np.ndarray:
    parameters = FieldParameters(delay=0.04, gain=1.0, feedback_gain=1.2, feedback_inhibition=2.0, feedback_delay=0.36)
    return np.stack([forward(density[side], times, excitatory, inhibitory, fields, parameters) for side in (-1, 1)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cti.yaml")
    parser.add_argument("--permutations", type=int, default=1000)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    with (root / args.config).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    output = root / "outputs" / "v2" / "q1"
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config["random_seed"])
    dataset = load(root)
    times = dataset.times
    primary = (times >= PRIMARY[0]) & (times <= PRIMARY[1])

    encoding = EncodingParameters()
    excitatory, inhibitory, *_ = micro_transfer()
    excitatory, inhibitory = normalised_transfer(excitatory, inhibitory)
    fields = lead_field(GRID)
    density = {side: cortical_density(encode(side, encoding), GRID, 0.0, 0.25, 0.25) for side in (-1, 1)}
    model_templates = templates(density, times, excitatory, inhibitory, fields)

    method_rows = []
    asymmetry_rows = []
    for task in (1, 2):
        selection = dataset.task == task
        values = dataset.cue[selection]
        sides = dataset.cue_side[selection]
        reference = trimmed(values)
        methods = {"mean": values, "trimmed": values, "wavelet": wavelet_shrink(values), "matched": matched_filter_denoise(values, model_templates)}
        for name, processed in methods.items():
            erp = processed.mean(axis=0) if name == "mean" else trimmed(processed)
            method_rows.append({"task": task, "method": name, "reliability": split_half_reliability(processed), "fidelity": shape_fidelity(reference, erp), "discriminability": discriminability(processed, sides, model_templates)})
        p3a_rows = []
        for side in (-1, 1):
            erp = trimmed(values[sides == side])
            for channel, name in enumerate(CHANNELS):
                p3a_rows.append({"task": task, "side": side, "channel": name, **fit_p3a(erp[channel], times)})
            f3 = float(erp[1, primary].mean())
            f4 = float(erp[2, primary].mean())
            asymmetry_rows.append({"task": task, "side": side, "f3_mean": f3, "f4_mean": f4, "asymmetry_index": (f3 - f4) / max(abs(f3) + abs(f4), 1e-12)})
        pd.DataFrame(p3a_rows).to_csv(output / f"p3a_task{task}.csv", index=False)
        response = cluster_vs_zero(values, times, args.permutations, rng)
        response.insert(0, "task", task)
        response.to_csv(output / f"clusters_task{task}.csv", index=False)

    pd.DataFrame(method_rows).to_csv(output / "method_comparison.csv", index=False)
    pd.DataFrame(asymmetry_rows).to_csv(output / "asymmetry.csv", index=False)
    print(pd.DataFrame(method_rows).round(3).to_string(index=False))
    print(pd.DataFrame(asymmetry_rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
