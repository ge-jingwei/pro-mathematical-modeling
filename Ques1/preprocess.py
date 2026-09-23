from dataclasses import dataclass

import numpy as np
import pywt
from scipy.signal import butter, sosfiltfilt

from tools.io_mat import Record


@dataclass(frozen=True)
class EpochSet:
    raw: np.ndarray
    decon: np.ndarray
    conventional: np.ndarray
    proposed: np.ndarray
    keep: np.ndarray
    times: np.ndarray
    target_proposed: np.ndarray
    target_keep: np.ndarray
    target_times: np.ndarray
    alpha: float
    tradeoff: list[dict]


def _filter(data: np.ndarray, sample_rate: float, highpass: float, lowpass: float) -> np.ndarray:
    band = butter(4, [highpass, lowpass], btype="bandpass", fs=sample_rate, output="sos")
    notch = butter(2, [49.0, 51.0], btype="bandstop", fs=sample_rate, output="sos")
    return sosfiltfilt(notch, sosfiltfilt(band, data, axis=-1), axis=-1)


def _ecg_regression(eeg: np.ndarray, ecg: np.ndarray, sample_rate: float) -> np.ndarray:
    lags = np.arange(-int(0.2 * sample_rate), int(0.2 * sample_rate) + 1, 8)
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


def extract_epochs(data: np.ndarray, onsets: np.ndarray, sample_rate: float, window: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.arange(round(window[0] * sample_rate), round(window[1] * sample_rate), dtype=int)
    indices = onsets[:, None].astype(int) + offsets
    if indices.min() < 0 or indices.max() >= data.shape[-1]:
        raise ValueError("Epoch exceeds signal bounds")
    return data[:, indices].transpose(1, 0, 2), offsets / sample_rate


def _baseline(epochs: np.ndarray, times: np.ndarray, window: tuple[float, float]) -> np.ndarray:
    mask = (times >= window[0]) & (times < window[1])
    return epochs - np.median(epochs[..., mask], axis=-1, keepdims=True)


def _shrink_epoch(epoch: np.ndarray, alpha: float) -> np.ndarray:
    coeffs = pywt.wavedec(epoch, "sym6", mode="symmetric", level=4)
    sigma = np.median(np.abs(coeffs[-1] - np.median(coeffs[-1]))) / 0.6745
    threshold = alpha * sigma * np.sqrt(2.0 * np.log(epoch.size))
    shrunk = [coeffs[0], *[pywt.threshold(part, threshold, mode="soft") for part in coeffs[1:]]]
    return pywt.waverec(shrunk, "sym6", mode="symmetric")[: epoch.size]


def wavelet_shrink(epochs: np.ndarray, alpha: float) -> np.ndarray:
    if alpha == 0.0:
        return epochs.copy()
    return np.asarray([[_shrink_epoch(channel, alpha) for channel in trial] for trial in epochs])


def _erp_fidelity(reference: np.ndarray, candidate: np.ndarray, labels: np.ndarray, times: np.ndarray) -> float:
    mask = (times >= 0.0) & (times <= 0.6)
    values = []
    for side in (-1, 1):
        chosen = labels == side
        for channel in range(reference.shape[1]):
            x = reference[chosen, channel][:, mask].mean(axis=0)
            y = candidate[chosen, channel][:, mask].mean(axis=0)
            values.append(np.corrcoef(x, y)[0, 1])
    return float(np.nanmean(values))


def prepare_epochs(record: Record, events, config: dict) -> EpochSet:
    rate = record.sample_rate
    window = tuple(config["windows"]["cue"])
    baseline = tuple(config["windows"]["baseline"])
    raw_unscaled, times = extract_epochs(record.eeg, events["cue_on"].to_numpy(), rate, window)
    decon, _ = extract_epochs(_filter(record.decon, rate, 0.5, 30.0), events["cue_on"].to_numpy(), rate, window)
    filtered = _filter(record.eeg, rate, 0.5, 30.0)
    ecg = _filter(record.ecg[None, :], rate, 5.0, 25.0)[0]
    conventional_signal = _ecg_regression(filtered, ecg, rate)
    conventional, _ = extract_epochs(conventional_signal, events["cue_on"].to_numpy(), rate, window)
    raw = _baseline(raw_unscaled, times, baseline)
    decon = _baseline(decon, times, baseline)
    conventional = _baseline(conventional, times, baseline)
    saturated = np.any(np.abs(raw_unscaled) >= config["quality"]["saturation"], axis=(1, 2))
    excessive = np.max(np.ptp(raw_unscaled, axis=2), axis=1) > config["quality"]["hard_trial_peak_to_peak"]
    keep = ~(saturated | excessive)
    labels = events["cue_side"].to_numpy()[keep]
    reference = conventional[keep]
    baseline_mask = (times >= baseline[0]) & (times < baseline[1])
    reference_rms = float(np.sqrt(np.mean(reference[..., baseline_mask] ** 2)))
    tradeoff = []
    candidates = {}
    for alpha in config["wavelet"]["alpha_candidates"]:
        candidate = wavelet_shrink(reference, float(alpha))
        fidelity = _erp_fidelity(reference, candidate, labels, times)
        rms = float(np.sqrt(np.mean(candidate[..., baseline_mask] ** 2)))
        reduction = 20.0 * np.log10(reference_rms / max(rms, np.finfo(float).eps))
        tradeoff.append({"alpha": float(alpha), "fidelity": fidelity, "noise_reduction_db": reduction})
        candidates[float(alpha)] = candidate
    eligible = [row for row in tradeoff if row["fidelity"] >= config["wavelet"]["minimum_fidelity"]]
    eligible = eligible or tradeoff
    best_reduction = max(row["noise_reduction_db"] for row in eligible)
    selected = min((row for row in eligible if row["noise_reduction_db"] >= best_reduction - 0.01), key=lambda row: row["alpha"])
    proposed = conventional.copy()
    proposed[keep] = candidates[selected["alpha"]]
    target_window = tuple(config["windows"]["target"])
    target_raw, target_times = extract_epochs(record.eeg, events["target_on"].to_numpy(), rate, target_window)
    target_conventional, _ = extract_epochs(conventional_signal, events["target_on"].to_numpy(), rate, target_window)
    target_saturated = np.any(np.abs(target_raw) >= config["quality"]["saturation"], axis=(1, 2))
    target_excessive = np.max(np.ptp(target_raw, axis=2), axis=1) > config["quality"]["hard_trial_peak_to_peak"]
    target_keep = ~(target_saturated | target_excessive)
    target_conventional = _baseline(target_conventional, target_times, baseline)
    target_proposed = target_conventional.copy()
    target_proposed[target_keep] = wavelet_shrink(target_conventional[target_keep], selected["alpha"])
    return EpochSet(raw, decon, conventional, proposed, keep, times, target_proposed, target_keep, target_times, selected["alpha"], tradeoff)
