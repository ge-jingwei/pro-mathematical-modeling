"""Cached epoched dataset for the second-generation model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

from Ques1.events import parse_events
from tools.io_mat import load_records

CHANNELS = ("Fz", "F3", "F4")


@dataclass(frozen=True)
class Epochs:
    cue: np.ndarray
    cue_broadband: np.ndarray
    target: np.ndarray
    cue_side: np.ndarray
    target_side: np.ndarray
    action_side: np.ndarray
    reaction_time: np.ndarray
    subject: np.ndarray
    task: np.ndarray
    file: np.ndarray
    times: np.ndarray
    target_task: np.ndarray
    target_file: np.ndarray
    target_subject: np.ndarray

    def subset(self, mask: np.ndarray) -> "Epochs":
        return Epochs(self.cue[mask], self.cue_broadband[mask], self.target[mask], self.cue_side[mask], self.target_side[mask], self.action_side[mask], self.reaction_time[mask], self.subject[mask], self.task[mask], self.file[mask], self.times, self.target_task[mask], self.target_file[mask], self.target_subject[mask])


def bandpass(data: np.ndarray, rate: float, low: float = 0.5, high: float = 30.0) -> np.ndarray:
    sos = butter(4, [low, high], btype="bandpass", fs=rate, output="sos")
    notch = butter(2, [49.0, 51.0], btype="bandstop", fs=rate, output="sos")
    return sosfiltfilt(notch, sosfiltfilt(sos, data, axis=-1), axis=-1)


def _ecg_regression(eeg: np.ndarray, ecg: np.ndarray, rate: float) -> np.ndarray:
    lags = np.arange(-int(0.2 * rate), int(0.2 * rate) + 1, 8)
    design = np.column_stack([np.roll(ecg, lag) for lag in lags])
    design -= np.median(design, axis=0)
    edge = int(np.max(np.abs(lags)))
    valid = np.ones(len(ecg), dtype=bool)
    valid[:edge] = False
    valid[-edge:] = False
    valid &= np.all(np.isfinite(design), axis=1)
    sampled = np.flatnonzero(valid)[::4]
    gram = design[sampled].T @ design[sampled]
    gram.flat[:: len(lags) + 1] += np.trace(gram) * 1e-6 / len(lags)
    weights = np.linalg.solve(gram, design[sampled].T @ eeg[:, sampled].T)
    return eeg - (design @ weights).T


def _epochs(signal: np.ndarray, onsets: np.ndarray, rate: float, window: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.arange(round(window[0] * rate), round(window[1] * rate), dtype=int)
    indices = onsets[:, None].astype(int) + offsets
    return signal[:, indices].transpose(1, 0, 2), offsets / rate


def _baseline(values: np.ndarray, times: np.ndarray, window: tuple[float, float]) -> np.ndarray:
    mask = (times >= window[0]) & (times < window[1])
    return values - np.median(values[..., mask], axis=-1, keepdims=True)


def _reject(raw: np.ndarray, saturation: float = 1000.0, peak_to_peak: float = 400.0) -> np.ndarray:
    return ~(np.any(np.abs(raw) >= saturation, axis=(1, 2)) | (np.max(np.ptp(raw, axis=2), axis=1) > peak_to_peak))


def build(root: Path, cue_window: tuple[float, float] = (-0.2, 0.8), target_window: tuple[float, float] = (-0.2, 0.8)) -> Epochs:
    cue_values, cue_broadband, target_values, cue_side, target_side, action_side, reaction_time = [], [], [], [], [], [], []
    subject, task, file, target_task, target_file, target_subject = [], [], [], [], [], []
    times = None
    for record in load_records(root / "C题" / "dataset"):
        rate = record.sample_rate
        events = parse_events(record)
        clean = _ecg_regression(bandpass(record.eeg, rate), bandpass(record.ecg[None, :], rate, 5.0, 25.0)[0], rate)
        broadband = _ecg_regression(bandpass(record.eeg, rate, 0.5, 55.0), bandpass(record.ecg[None, :], rate, 5.0, 25.0)[0], rate)
        raw_cue, cue_times = _epochs(record.eeg, events["cue_on"].to_numpy(), rate, cue_window)
        raw_target, target_times = _epochs(record.eeg, events["target_on"].to_numpy(), rate, target_window)
        kept_cue, kept_target = _reject(raw_cue), _reject(raw_target)
        clean_cue, _ = _epochs(clean, events["cue_on"].to_numpy(), rate, cue_window)
        broadband_cue, _ = _epochs(broadband, events["cue_on"].to_numpy(), rate, cue_window)
        clean_target, _ = _epochs(clean, events["target_on"].to_numpy(), rate, target_window)
        cue_values.append(_baseline(clean_cue[kept_cue], cue_times, (-0.2, 0.0)))
        cue_broadband.append(_baseline(broadband_cue[kept_cue], cue_times, (-0.2, 0.0)))
        target_values.append(_baseline(clean_target[kept_target], target_times, (-0.2, 0.0)))
        cue_side.append(events["cue_side"].to_numpy()[kept_cue])
        target_side.append(events["target_side"].to_numpy()[kept_target])
        action_side.append(events["response_side"].to_numpy()[kept_target])
        reaction_time.append(events["reaction_time_s"].to_numpy()[kept_cue])
        subject.append(np.full(int(kept_cue.sum()), record.subject))
        task.append(np.full(int(kept_cue.sum()), record.task))
        file.append(np.full(int(kept_cue.sum()), record.path.name))
        target_subject.append(np.full(int(kept_target.sum()), record.subject))
        target_task.append(np.full(int(kept_target.sum()), record.task))
        target_file.append(np.full(int(kept_target.sum()), record.path.name))
        times = cue_times
    stack = lambda values: np.concatenate(values)
    return Epochs(stack(cue_values), stack(cue_broadband), stack(target_values), stack(cue_side), stack(target_side), stack(action_side), stack(reaction_time), stack(subject), stack(task), stack(file), times, stack(target_task), stack(target_file), stack(target_subject))


def load(root: Path) -> Epochs:
    cache = root / "outputs" / "v2" / "epochs.npz"
    names = ("cue", "cue_broadband", "target", "cue_side", "target_side", "action_side", "reaction_time", "subject", "task", "source", "times", "target_task", "target_source", "target_subject")
    if cache.exists():
        payload = np.load(cache, allow_pickle=True)
        if all(name in payload for name in names):
            return Epochs(*(payload[name] for name in names))
    values = build(root)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, cue=values.cue, cue_broadband=values.cue_broadband, target=values.target, cue_side=values.cue_side, target_side=values.target_side, action_side=values.action_side, reaction_time=values.reaction_time, subject=values.subject, task=values.task, source=values.file, times=values.times, target_task=values.target_task, target_source=values.target_file, target_subject=values.target_subject)
    return values
