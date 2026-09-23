"""Question 3: V-H-P cognitive model with theta-gamma PAC validation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.signal import butter, hilbert, sosfiltfilt

from neuro.dataset import Epochs, load
from neuro.dynamics import kuramoto

BANDS = {"theta": (4.0, 8.0), "low_gamma": (30.0, 50.0)}


def bandpass(values: np.ndarray, rate: float, low: float, high: float) -> np.ndarray:
    sos = butter(3, [low, high], btype="bandpass", fs=rate, output="sos")
    return sosfiltfilt(sos, values, axis=-1)


def modulation_index(phase_signal: np.ndarray, amplitude_signal: np.ndarray, bins: int = 18) -> float:
    """Normalised Kullback-Leibler modulation index of amplitude over phase bins."""
    phase = np.angle(hilbert(phase_signal))
    amplitude = np.abs(hilbert(amplitude_signal))
    edges = np.linspace(-np.pi, np.pi, bins + 1)
    indices = np.digitize(phase, edges) - 1
    indices = np.clip(indices, 0, bins - 1)
    distribution = np.array([amplitude[indices == bin].mean() for bin in range(bins)])
    distribution = distribution / max(distribution.sum(), 1e-12)
    entropy = -float(np.sum(distribution[distribution > 0] * np.log(distribution[distribution > 0])))
    return (np.log(bins) - entropy) / np.log(bins)


def surrogate_pac(phase_signal: np.ndarray, amplitude_signal: np.ndarray, permutations: int, rng: np.random.Generator) -> tuple[float, float]:
    observed = modulation_index(phase_signal, amplitude_signal)
    null = np.zeros(permutations)
    for draw in range(permutations):
        shift = int(rng.integers(1, len(phase_signal)))
        null[draw] = modulation_index(np.roll(phase_signal, shift), amplitude_signal)
    return observed, float((1.0 + np.sum(null >= observed)) / (permutations + 1.0))


def three_node_kuramoto(times: np.ndarray, rate: float, project: int) -> np.ndarray:
    """V-H-P oscillator network: coupling from the hippocampus grows faster in project 1."""
    frequencies = np.array([10.0, 8.0, 6.0])
    coupling = np.array([[0.0, 3.0, 0.0], [0.5, 0.0, 1.5], [0.0, 2.0 if project == 1 else 1.2, 0.0]])
    delays = np.array([[0.0, 0.004, 0.0], [0.004, 0.0, 0.012], [0.0, 0.012, 0.0]])
    return kuramoto(times, frequencies, coupling, delays, rate)


def cognition_segments(dataset: Epochs, subject: str, task: int, rate: float) -> tuple[np.ndarray, np.ndarray]:
    """Target-locked broadband, truncated per trial to 100 ms before the response."""
    selected = (dataset.target_subject == subject) & (dataset.target_task == task)
    values = dataset.target_broadband[selected][:, 0]
    reaction = dataset.target_reaction_time[selected]
    start = int(0.2 * rate)
    segments = []
    for trial, rt in zip(values, reaction, strict=True):
        stop = min(int((rt + 0.1) * rate), trial.size)
        if stop > start + int(0.4 * rate):
            segments.append(trial[start:stop])
    return values, np.asarray(reaction), segments


def electrode_synchrony(dataset: Epochs, subject: str, task: int, rate: float) -> tuple[np.ndarray, np.ndarray]:
    """Three-electrode theta-band order parameter aligned to the response.

    The model predicts the frontal synchrony (order parameter R) rises during
    recognition and peaks just before the response.
    """
    selected = (dataset.target_subject == subject) & (dataset.target_task == task)
    values = dataset.target_broadband[selected]
    reaction = dataset.target_reaction_time[selected]
    start = int(0.2 * rate)
    grid = np.linspace(-1.5, 0.0, 60)
    curves = []
    for trial, rt in zip(values, reaction, strict=True):
        stop = min(int((rt + 0.1) * rate), trial.shape[-1])
        if stop - start < int(0.4 * rate):
            continue
        phases = np.angle(hilbert(bandpass(trial[:, start:stop], rate, *BANDS["theta"]), axis=-1))
        radius = np.abs(np.exp(1j * phases).mean(axis=0))
        time_to_response = (np.arange(stop - start) - (stop - start)) / rate
        curves.append(np.interp(grid, time_to_response, radius))
    return grid, np.asarray(curves)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cti.yaml")
    parser.add_argument("--permutations", type=int, default=500)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    with (root / args.config).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    output = root / "outputs" / "v2" / "q3"
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config["random_seed"])
    rate = float(config["sample_rate"])
    dataset = load(root)

    rows = []
    for subject in ("A", "B"):
        for task in (1, 2):
            _, reaction, segments = cognition_segments(dataset, subject, task, rate)
            median = float(np.median(reaction))
            theta = [bandpass(segment, rate, *BANDS["theta"]) for segment in segments]
            gamma = [bandpass(segment, rate, *BANDS["low_gamma"]) for segment in segments]
            for label, mask in (("all", np.ones(len(segments), dtype=bool)), ("fast", reaction < median), ("slow", reaction >= median)):
                chosen = [index for index in np.flatnonzero(mask) if index < len(segments)]
                if len(chosen) < 2:
                    continue
                phase = np.concatenate([theta[index] for index in chosen])
                amplitude = np.concatenate([gamma[index] for index in chosen])
                observed, probability = surrogate_pac(phase, amplitude, args.permutations, rng)
                rows.append({"subject": subject, "task": task, "group": label, "reaction_median_s": round(median, 3), "n_trials": int(len(chosen)), "mi": observed, "p_value": probability})
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "pac_target_locked.csv", index=False)
    print(frame.round(4).to_string(index=False))

    synchrony_rows = []
    for subject in ("A", "B"):
        for task in (1, 2):
            grid, curves = electrode_synchrony(dataset, subject, task, rate)
            mean = curves.mean(axis=0)
            early, late = float(mean[:12].mean()), float(mean[-12:].mean())
            synchrony_rows.append({"subject": subject, "task": task, "n_trials": int(len(curves)), "sync_early": early, "sync_late": late, "sync_rise": late - early})
            np.savez(output / f"synchrony_{subject}_task{task}.npz", grid=grid, curves=curves)
    sync = pd.DataFrame(synchrony_rows)
    sync.to_csv(output / "synchrony.csv", index=False)
    print(sync.round(4).to_string(index=False))

    times = dataset.times
    for project in (1, 2):
        phases = three_node_kuramoto(times, rate, project)
        order = np.abs(np.exp(1j * phases).mean(axis=0))
        np.savez(output / f"kuramoto_project{project}.npz", phases=phases, order=order)


if __name__ == "__main__":
    main()
