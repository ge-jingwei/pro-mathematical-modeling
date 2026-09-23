from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from Ques1.events import TRIAL_COLUMNS, build_trials
from tools.io_mat import Record, load_records


@dataclass(frozen=True)
class QualityThresholds:
    transient: np.ndarray
    cross_channel: np.ndarray


@dataclass(frozen=True)
class QualityFlags:
    saturation: np.ndarray
    plateau: np.ndarray
    transient: np.ndarray
    cross_channel: np.ndarray

    @property
    def bad(self) -> np.ndarray:
        return self.saturation | self.plateau | self.transient | self.cross_channel


def _mad(values: np.ndarray, axis: int) -> np.ndarray:
    center = np.median(values, axis=axis, keepdims=True)
    return np.median(np.abs(values - center), axis=axis)


def _plateau_mask(signal: np.ndarray, minimum: int) -> np.ndarray:
    boundaries = np.r_[0, np.flatnonzero(np.diff(signal) != 0) + 1, len(signal)]
    mask = np.zeros(len(signal), dtype=bool)
    for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
        if stop - start >= minimum:
            mask[start:stop] = True
    return mask


def _longest_plateau(signal: np.ndarray) -> int:
    boundaries = np.r_[0, np.flatnonzero(np.diff(signal) != 0) + 1, len(signal)]
    return int(np.diff(boundaries).max(initial=0))


def fit_thresholds(
    eeg: np.ndarray,
    transient_scale: float,
    cross_scale: float,
    sample_mask: np.ndarray | None = None,
) -> QualityThresholds:
    selected = np.ones(eeg.shape[1], dtype=bool) if sample_mask is None else sample_mask
    difference_mask = selected & np.r_[False, selected[:-1]]
    if not difference_mask.any() or not selected.any():
        raise ValueError("Training samples are insufficient for quality thresholds")
    difference = np.diff(eeg, axis=1, prepend=eeg[:, :1])[:, difference_mask]
    common = np.median(eeg, axis=0)
    residual = (eeg - common)[..., selected]
    floor = np.finfo(float).eps
    return QualityThresholds(
        transient=np.maximum(transient_scale * _mad(difference, axis=1), floor),
        cross_channel=np.maximum(cross_scale * _mad(residual, axis=1), floor),
    )


def mark_quality(
    eeg: np.ndarray,
    thresholds: QualityThresholds,
    saturation_value: float,
    plateau_minimum: int,
) -> QualityFlags:
    saturation = np.isclose(np.abs(eeg), saturation_value)
    plateau = np.vstack([_plateau_mask(signal, plateau_minimum) for signal in eeg])
    difference = np.diff(eeg, axis=1, prepend=eeg[:, :1])
    transient = np.abs(difference) > thresholds.transient[:, None]
    common = np.median(eeg, axis=0)
    cross_channel = np.abs(eeg - common) > thresholds.cross_channel[:, None]
    return QualityFlags(saturation, plateau, transient, cross_channel)


def enrich_trials(
    record: Record,
    trials: pd.DataFrame,
    flags: QualityFlags,
    peak_to_peak_threshold: float,
) -> tuple[pd.DataFrame, int]:
    result = trials.copy()
    usable = []
    for index, row in result.iterrows():
        start = max(0, int(row["t_cue_on"] - 0.5 * record.sample_rate))
        stop = min(record.eeg.shape[1], int(row["t_cue_on"] + record.sample_rate))
        window = slice(start, stop)
        for channel_index, label in enumerate(record.eeg_labels):
            result.at[index, f"bad_frac_{label}"] = flags.bad[channel_index, window].mean()
        result.at[index, "n_sat"] = int(flags.saturation[:, window].sum())
        rejected = flags.saturation[:, window].any() or np.ptp(record.eeg[:, window], axis=1).max() > peak_to_peak_threshold
        usable.append(not rejected)
    return result[TRIAL_COLUMNS], int(np.sum(usable))


def summarize_record(
    record: Record,
    flags: QualityFlags,
    usable_trials: int,
) -> pd.DataFrame:
    rows = []
    for index, label in enumerate(record.eeg_labels):
        signal = record.eeg[index]
        rows.append(
            {
                "file": record.path.name,
                "subject": record.subject,
                "task": record.task,
                "channel": label,
                "n_samples": len(signal),
                "sat_negative": int(np.isclose(signal, -1000.0).sum()),
                "sat_positive": int(np.isclose(signal, 1000.0).sum()),
                "plateau_samples": int(flags.plateau[index].sum()),
                "max_plateau_samples": _longest_plateau(signal),
                "transient_samples": int(flags.transient[index].sum()),
                "cross_channel_samples": int(flags.cross_channel[index].sum()),
                "bad_fraction": float(flags.bad[index].mean()),
                "usable_trials": usable_trials,
            }
        )
    return pd.DataFrame(rows)


def build_quality_outputs(data_dir: str | Path, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_trials = build_trials(data_dir)
    trial_frames = []
    report_frames = []
    settings = config["quality"]
    for record in load_records(data_dir):
        trials = all_trials[all_trials["file"] == record.path.name].copy()
        thresholds = fit_thresholds(
            record.eeg,
            settings["transient_mad_scale"],
            settings["cross_channel_mad_scale"],
        )
        flags = mark_quality(
            record.eeg,
            thresholds,
            settings["saturation"],
            settings["plateau_min_samples"],
        )
        enriched, usable = enrich_trials(
            record,
            trials,
            flags,
            settings["hard_trial_peak_to_peak"],
        )
        trial_frames.append(enriched)
        report_frames.append(summarize_record(record, flags, usable))
    return pd.concat(trial_frames, ignore_index=True), pd.concat(report_frames, ignore_index=True)


def write_quality_outputs(
    data_dir: str | Path,
    output_dir: str | Path,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trials, report = build_quality_outputs(data_dir, config)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trials.to_csv(output_dir / "trials.csv", index=False)
    report.to_csv(output_dir / "quality_report.csv", index=False)
    return trials, report
