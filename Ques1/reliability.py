from pathlib import Path

import numpy as np

from Ques1.quality import QualityThresholds, _plateau_mask, fit_thresholds
from tools.io_mat import Record, load_records


def compute_reliability(
    eeg: np.ndarray,
    thresholds: QualityThresholds,
    saturation_value: float,
    plateau_minimum: int,
) -> np.ndarray:
    saturation = np.isclose(np.abs(eeg), saturation_value)
    plateau = np.vstack([_plateau_mask(signal, plateau_minimum) for signal in eeg])
    difference = np.diff(eeg, axis=1, prepend=eeg[:, :1])
    transient = 1.0 / (1.0 + (np.abs(difference) / thresholds.transient[:, None]) ** 2)
    common = np.median(eeg, axis=0)
    residual = eeg - common
    cross_channel = 1.0 / (
        1.0 + (np.abs(residual) / thresholds.cross_channel[:, None]) ** 2
    )
    reliability = (~saturation & ~plateau) * transient * cross_channel
    return np.clip(reliability, 0.0, 1.0)


def fit_record_reliability(
    record: Record,
    config: dict,
    training_samples: np.ndarray,
) -> tuple[np.ndarray, QualityThresholds]:
    settings = config["quality"]
    thresholds = fit_thresholds(
        record.eeg,
        settings["transient_mad_scale"],
        settings["cross_channel_mad_scale"],
        training_samples,
    )
    reliability = compute_reliability(
        record.eeg,
        thresholds,
        settings["saturation"],
        settings["plateau_min_samples"],
    )
    return reliability, thresholds


def trial_weights(
    reliability: np.ndarray,
    cue_onsets: np.ndarray,
    sample_rate: float,
) -> np.ndarray:
    weights = []
    for onset in cue_onsets.astype(int):
        start = max(0, int(onset - 0.5 * sample_rate))
        stop = min(reliability.shape[1], int(onset + sample_rate))
        weights.append(reliability[:, start:stop].mean(axis=1))
    return np.asarray(weights)


def write_reliability(data_dir: str | Path, output_path: str | Path, config: dict) -> None:
    arrays = {}
    for record in load_records(data_dir):
        training_samples = np.ones(record.eeg.shape[1], dtype=bool)
        reliability, thresholds = fit_record_reliability(record, config, training_samples)
        key = record.path.stem.replace("-", "_")
        arrays[f"{key}_weights"] = reliability
        arrays[f"{key}_transient_thresholds"] = thresholds.transient
        arrays[f"{key}_cross_thresholds"] = thresholds.cross_channel
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
