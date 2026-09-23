from dataclasses import dataclass, replace

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import butter, fftconvolve, hilbert, sosfiltfilt
from scipy.special import expit


@dataclass(frozen=True)
class ModelParameters:
    delay: float = 0.06
    coupling: float = 1.2
    input_gain: float = 5.0
    tau_e: float = 0.012
    tau_i: float = 0.020
    sigmoid_gain: float = 1.4
    w_ee: float = 10.0
    w_ei: float = 12.0
    w_ie: float = 10.0
    w_ii: float = 2.0


@dataclass(frozen=True)
class Simulation:
    stimulus: np.ndarray
    features: np.ndarray
    cortical_input: np.ndarray
    source: np.ndarray
    eeg: np.ndarray
    order_global: np.ndarray
    order_regional: np.ndarray


def triangle_image(side: int, size: int = 64) -> np.ndarray:
    y, x = np.mgrid[-1:1:complex(size), -1:1:complex(size)]
    vertices = np.array([[-0.58 * side, -0.58], [-0.58 * side, 0.58], [0.62 * side, 0.0]])
    points = np.column_stack([x.ravel(), y.ravel()])
    signs = []
    for a, b in zip(vertices, np.roll(vertices, -1, axis=0), strict=True):
        signs.append((points[:, 0] - a[0]) * (b[1] - a[1]) - (points[:, 1] - a[1]) * (b[0] - a[0]))
    values = np.stack(signs)
    filled = (np.all(values >= 0, axis=0) | np.all(values <= 0, axis=0)).reshape(size, size).astype(float)
    edge = np.hypot(*np.gradient(filled))
    return gaussian_filter(edge, 0.7) / max(edge.max(), np.finfo(float).eps)


def _gabor(theta: float, size: int = 17, sigma: float = 3.0, wavelength: float = 6.0) -> np.ndarray:
    axis = np.arange(size) - (size - 1) / 2
    y, x = np.meshgrid(axis, axis)
    xr = x * np.cos(theta) + y * np.sin(theta)
    yr = -x * np.sin(theta) + y * np.cos(theta)
    kernel = np.exp(-(xr**2 + 0.5 * yr**2) / (2 * sigma**2)) * np.cos(2 * np.pi * xr / wavelength)
    return kernel - kernel.mean()


def encode_shape(stimulus: np.ndarray) -> np.ndarray:
    orientations = np.linspace(0, np.pi, 4, endpoint=False)
    maps = np.asarray([np.abs(fftconvolve(stimulus, _gabor(theta), mode="same")) for theta in orientations])
    maps /= np.maximum(maps.max(axis=(1, 2), keepdims=True), np.finfo(float).eps)
    return maps


def cortical_projection(features: np.ndarray, grid_size: int = 16) -> np.ndarray:
    image = features.mean(axis=0)
    bins = image.shape[0] // grid_size
    pooled = image[: bins * grid_size, : bins * grid_size].reshape(grid_size, bins, grid_size, bins).mean(axis=(1, 3))
    projected = np.fliplr(pooled)
    return projected / max(projected.max(), np.finfo(float).eps)


def lead_fields(grid_size: int) -> np.ndarray:
    y, x = np.mgrid[-1:1:complex(grid_size), -1:1:complex(grid_size)]
    centers = ((0.0, 0.28), (-0.55, 0.30), (0.55, 0.30))
    fields = np.asarray([np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * 0.52**2)) for cx, cy in centers])
    return fields / fields.sum(axis=(1, 2), keepdims=True)


def _bandpass(data: np.ndarray, rate: float, low: float, high: float) -> np.ndarray:
    sos = butter(3, [low, high], btype="bandpass", fs=rate, output="sos")
    return sosfiltfilt(sos, data, axis=-1)


def simulate(side: int, times: np.ndarray, rate: float, parameters: ModelParameters, grid_size: int = 16) -> Simulation:
    stimulus = triangle_image(side)
    features = encode_shape(stimulus)
    drive_map = cortical_projection(features, grid_size)
    e = np.full((grid_size, grid_size), 0.035)
    i = np.full_like(e, 0.025)
    source = np.zeros((grid_size, grid_size, len(times)))
    dt = 1.0 / rate
    onset = parameters.delay
    for k, time in enumerate(times):
        envelope = float(time >= onset) * np.exp(-max(time - onset, 0.0) / 0.22)
        lateral = gaussian_filter(e, 1.0)
        se = expit(parameters.sigmoid_gain * (parameters.w_ee * e - parameters.w_ei * i + parameters.coupling * lateral + parameters.input_gain * envelope * drive_map - 3.1))
        si = expit(parameters.sigmoid_gain * (parameters.w_ie * e - parameters.w_ii * i - 2.8))
        e = np.clip(e + dt * (-e + (1 - e) * se) / parameters.tau_e, 0, 1)
        i = np.clip(i + dt * (-i + (1 - i) * si) / parameters.tau_i, 0, 1)
        source[..., k] = e - 0.72 * i
    baseline = times < 0
    source -= source[..., baseline].mean(axis=-1, keepdims=True)
    fields = lead_fields(grid_size)
    eeg = np.einsum("cyx,yxt->ct", fields, source)
    eeg = _bandpass(eeg, rate, 0.5, 30.0)
    phase = np.angle(hilbert(_bandpass(source.reshape(-1, len(times)), rate, 4.0, 12.0), axis=-1)).reshape(source.shape)
    unit = np.exp(1j * phase)
    order_global = np.abs(unit.mean(axis=(0, 1)))
    order_regional = np.abs(np.einsum("cyx,yxt->ct", fields, unit))
    return Simulation(stimulus, features, drive_map, source, eeg, order_global, order_regional)


def vary(parameters: ModelParameters, name: str, factor: float) -> ModelParameters:
    return replace(parameters, **{name: getattr(parameters, name) * factor})
