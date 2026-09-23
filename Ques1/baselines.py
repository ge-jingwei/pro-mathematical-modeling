import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.stats import trim_mean


def extract_epochs(
    signal: np.ndarray,
    onsets: np.ndarray,
    sample_rate: float,
    window: tuple[float, float],
) -> np.ndarray:
    offsets = np.arange(int(round(window[0] * sample_rate)), int(round(window[1] * sample_rate)))
    indices = onsets.astype(int)[:, None] + offsets
    if indices.min() < 0 or indices.max() >= signal.shape[-1]:
        raise ValueError("Epoch exceeds signal bounds")
    return signal[:, indices].transpose(1, 0, 2)


def bandpass_epochs(
    epochs: np.ndarray,
    sample_rate: float,
    highpass: float,
    lowpass: float,
    notch: float,
) -> np.ndarray:
    band = butter(4, [highpass, lowpass], btype="bandpass", fs=sample_rate, output="sos")
    stop = butter(2, [notch - 1.0, notch + 1.0], btype="bandstop", fs=sample_rate, output="sos")
    return sosfiltfilt(stop, sosfiltfilt(band, epochs, axis=-1), axis=-1)


def hard_trial_mask(
    raw_epochs: np.ndarray,
    saturation_value: float,
    peak_to_peak_threshold: float,
) -> np.ndarray:
    saturated = np.isclose(np.abs(raw_epochs), saturation_value).any(axis=(1, 2))
    excessive = np.ptp(raw_epochs, axis=2).max(axis=1) > peak_to_peak_threshold
    return ~(saturated | excessive)


def _interpolate_short(signal: np.ndarray, bad: np.ndarray, maximum: int) -> np.ndarray:
    result = signal.copy()
    boundaries = np.flatnonzero(np.diff(np.r_[False, bad, False]))
    for start, stop in boundaries.reshape(-1, 2):
        if stop - start <= maximum and start > 0 and stop < len(signal):
            result[start:stop] = np.linspace(result[start - 1], result[stop], stop - start + 2)[1:-1]
    return result


def short_interpolation(
    raw_epochs: np.ndarray,
    reliability_epochs: np.ndarray,
    maximum_samples: int,
) -> np.ndarray:
    return np.asarray(
        [
            [_interpolate_short(signal, weight == 0.0, maximum_samples) for signal, weight in zip(epoch, weights, strict=True)]
            for epoch, weights in zip(raw_epochs, reliability_epochs, strict=True)
        ]
    )


def robust_average(
    epochs: np.ndarray,
    valid: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    proportion: float = 0.2,
) -> np.ndarray:
    selected = epochs if valid is None else epochs[valid]
    if not len(selected):
        return np.full(epochs.shape[1:], np.nan)
    if weights is None:
        return trim_mean(selected, proportiontocut=proportion, axis=0)
    selected_weights = weights if valid is None else weights[valid]
    order = np.argsort(selected, axis=0)
    sorted_values = np.take_along_axis(selected, order, axis=0)
    sorted_weights = np.take_along_axis(selected_weights, order, axis=0)
    cut = int(np.floor(proportion * len(selected)))
    if cut:
        sorted_values = sorted_values[cut:-cut]
        sorted_weights = sorted_weights[cut:-cut]
    denominator = sorted_weights.sum(axis=0)
    return np.divide(
        (sorted_values * sorted_weights).sum(axis=0),
        denominator,
        out=np.full_like(denominator, np.nan, dtype=float),
        where=denominator > 0,
    )
