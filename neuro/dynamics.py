"""Microscopic, mesoscopic and macroscopic model layers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from scipy.signal import hilbert
from scipy.special import expit


@dataclass(frozen=True)
class MicroParameters:
    tau: float = 0.020
    rest: float = -70.0
    threshold: float = -50.0
    reset: float = -65.0
    refractory: float = 0.004
    resistance: float = 12.0
    noise: float = 1.0
    excitation: float = 6.0
    inhibition: float = 10.0
    synapse: float = 0.90


@dataclass(frozen=True)
class Sigmoid:
    amplitude: float
    slope: float
    threshold: float

    def __call__(self, x: np.ndarray | float) -> np.ndarray:
        return self.amplitude * expit(self.slope * (np.asarray(x) - self.threshold))


def lif_circuit(inputs: np.ndarray, parameters: MicroParameters = MicroParameters(), neurons: int = 128, duration: float = 0.6, step: float = 2e-4, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Pyramidal cell driving an interneuron that inhibits it back (equation 1).

    Returns the steady-state firing rates of the pyramidal and interneuron
    populations for every input level.
    """
    rng = np.random.default_rng(seed)
    levels = np.asarray(inputs, dtype=float)
    shape = (neurons, len(levels))
    pyramidal = np.full(shape, parameters.rest + 4.0)
    interneuron = np.full(shape, parameters.rest + 4.0)
    refractory = np.zeros(shape)
    excitatory_synapse = np.zeros(shape)
    inhibitory_synapse = np.zeros(shape)
    spikes_pyramidal = np.zeros(shape)
    spikes_interneuron = np.zeros(shape)
    decay = float(np.exp(-step / (4.0 * parameters.tau)))
    steps = int(round(duration / step))
    warmup = steps // 3
    for index in range(steps):
        active = refractory <= 0.0
        current_pyramidal = levels[None, :] - parameters.inhibition * inhibitory_synapse
        current_interneuron = parameters.excitation * excitatory_synapse
        pyramidal += step / parameters.tau * (-(pyramidal - parameters.rest) + parameters.resistance * current_pyramidal) + parameters.noise * rng.normal(scale=np.sqrt(step), size=shape)
        interneuron += step / parameters.tau * (-(interneuron - parameters.rest) + parameters.resistance * current_interneuron) + parameters.noise * rng.normal(scale=np.sqrt(step), size=shape)
        fired_pyramidal = (pyramidal >= parameters.threshold) & active
        fired_interneuron = (interneuron >= parameters.threshold) & active
        excitatory_synapse = parameters.synapse * excitatory_synapse + fired_pyramidal
        inhibitory_synapse = parameters.synapse * inhibitory_synapse + fired_interneuron
        pyramidal[fired_pyramidal] = parameters.reset
        interneuron[fired_interneuron] = parameters.reset
        refractory = np.maximum(refractory - step, 0.0)
        refractory[fired_pyramidal | fired_interneuron] = parameters.refractory
        if index >= warmup:
            spikes_pyramidal += fired_pyramidal
            spikes_interneuron += fired_interneuron
    window = (steps - warmup) * step
    return spikes_pyramidal.sum(axis=0) / (neurons * window), spikes_interneuron.sum(axis=0) / (neurons * window)


def fit_transfer(inputs: np.ndarray, rates: np.ndarray) -> Sigmoid:
    """Sigmoid transfer function fitted to the microscopic f-I curve."""
    def model(x, amplitude, slope, threshold):
        return amplitude * expit(slope * (x - threshold))

    amplitude = max(float(rates.max()), 1e-6)
    guess = [amplitude, 1.0, float(inputs[np.argmax(rates >= 0.5 * amplitude)] if np.any(rates >= 0.5 * amplitude) else np.median(inputs))]
    bounds = ([0.0, 1e-3, float(inputs.min()) - 10.0], [amplitude * 2.0 + 1.0, 20.0, float(inputs.max()) + 10.0])
    fitted, _ = curve_fit(model, inputs, rates, p0=guess, bounds=bounds, maxfev=20000)
    return Sigmoid(float(fitted[0]), float(fitted[1]), float(fitted[2]))


def micro_transfer(parameters: MicroParameters = MicroParameters(), levels: int = 26, seed: int = 0) -> tuple[Sigmoid, Sigmoid, np.ndarray, np.ndarray, np.ndarray]:
    inputs = np.linspace(0.0, 12.0, levels)
    pyramidal, interneuron = lif_circuit(inputs, parameters, seed=seed)
    return fit_transfer(inputs, pyramidal), fit_transfer(inputs, interneuron), inputs, pyramidal, interneuron


def normalised_transfer(excitatory: Sigmoid, inhibitory: Sigmoid) -> tuple[Sigmoid, Sigmoid]:
    """Rescale both curves to unit gain, with the excitatory threshold as unit."""
    unit = excitatory.threshold
    return Sigmoid(1.0, excitatory.slope * unit, 1.0), Sigmoid(1.0, inhibitory.slope * unit, inhibitory.threshold / unit)


@dataclass(frozen=True)
class FieldParameters:
    delay: float = 0.08
    envelope_width: float = 0.09
    gain: float = 2.0
    feedback_gain: float = 0.6
    feedback_delay: float = 0.52
    feedback_width: float = 0.10
    feedback_inhibition: float = 1.6
    coupling: float = 0.5
    spatial_sigma: float = 1.5
    bias: float = 0.9
    w_ee: float = 4.0
    w_ei: float = 6.0
    w_ie: float = 4.0
    w_ii: float = 2.0
    tau_e: float = 0.012
    tau_i: float = 0.020
    adaptation: float = 1.6
    tau_adaptation: float = 0.09
    delay_gradient: float = 0.02
    step: float = 0.001


def alpha_envelope(times: np.ndarray, onset: float | np.ndarray, width: float) -> np.ndarray:
    scaled = np.maximum(times - onset, 0.0) / width
    return np.where(times >= onset, scaled * np.exp(1.0 - scaled), 0.0)


def delay_map(grid: int, delay: float, gradient: float, axis: int = 1) -> np.ndarray:
    """Spatially distributed conduction delay across the cortical patch."""
    coordinate = np.linspace(-1.0, 1.0, grid)
    profile = coordinate if axis == 1 else coordinate[::-1]
    shape = (1, grid) if axis == 1 else (grid, 1)
    return delay + gradient * profile.reshape(shape)


def field_response(density: np.ndarray, times: np.ndarray, excitatory: Sigmoid, inhibitory: Sigmoid, parameters: FieldParameters = FieldParameters(), feedforward: np.ndarray | None = None, feedback: np.ndarray | None = None) -> np.ndarray:
    """Two-dimensional Wilson-Cowan neural field (equations 2 and 3)."""
    steps = int(round((times[-1] - times[0]) / parameters.step)) + 1
    grid_times = times[0] + np.arange(steps) * parameters.step
    feedforward = feedforward if feedforward is not None else delay_map(density.shape[-1], parameters.delay, parameters.delay_gradient)
    feedback = feedback if feedback is not None else delay_map(density.shape[-1], parameters.feedback_delay, parameters.delay_gradient)
    direct = alpha_envelope(grid_times[None, None, :], feedforward[..., None], parameters.envelope_width)
    recurrent = alpha_envelope(grid_times[None, None, :], feedback[..., None], parameters.feedback_width)
    excitatory_drive = parameters.gain * direct + parameters.feedback_gain * recurrent
    inhibitory_drive = parameters.feedback_gain * parameters.feedback_inhibition * recurrent
    e = np.full_like(density, 0.02)
    i = np.full_like(density, 0.02)
    adaptation = np.zeros_like(density)
    source = np.zeros((*density.shape, len(times)))
    cursor = 0
    for index, level in enumerate(grid_times):
        lateral = gaussian_filter(e, parameters.spatial_sigma)
        input_e = parameters.w_ee * e - parameters.w_ei * i + parameters.coupling * (lateral - e) + excitatory_drive[..., index] * density - parameters.bias - parameters.adaptation * adaptation
        input_i = parameters.w_ie * e - parameters.w_ii * i + inhibitory_drive[..., index] * density - parameters.bias
        e = np.clip(e + parameters.step / parameters.tau_e * (-e + (1.0 - e) * excitatory(input_e)), 0.0, 1.0)
        i = np.clip(i + parameters.step / parameters.tau_i * (-i + (1.0 - i) * inhibitory(input_i)), 0.0, 1.0)
        adaptation += parameters.step / parameters.tau_adaptation * (-adaptation + e)
        if cursor < len(times) and level >= times[cursor] - 0.5 * parameters.step:
            source[..., cursor] = e - 0.72 * i
            cursor += 1
    return source


def regional_signals(source: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Average source time course inside each labelled region."""
    regions = np.unique(labels)
    return np.stack([source[labels == region].mean(axis=0) for region in regions])


def order_parameter(signals: np.ndarray, rate: float) -> tuple[np.ndarray, np.ndarray]:
    """Kuramoto order parameter of equation 5 read out from regional signals."""
    analytic = hilbert(signals, axis=-1)
    phases = np.angle(analytic)
    unit = np.exp(1j * phases)
    return np.abs(unit.mean(axis=0)), phases


def kuramoto(times: np.ndarray, frequencies: np.ndarray, coupling: np.ndarray, delays: np.ndarray, rate: float, step: float = 0.001) -> np.ndarray:
    """Explicit coupled-oscillator network of equation 4."""
    steps = int(round((times[-1] - times[0]) / step)) + 1
    nodes = len(frequencies)
    lag = np.maximum(np.round(delays * rate).astype(int), 0)
    history = np.zeros((int(lag.max()) + 1, nodes))
    phases = np.zeros(nodes)
    collected = np.empty((nodes, len(times)))
    cursor = 0
    history[-1] = phases
    column = np.arange(nodes)[None, :]
    for index in range(steps):
        delayed = history[history.shape[0] - 1 - lag, column]
        interaction = (coupling * np.sin(delayed - phases[None, :])).sum(axis=1)
        phases = phases + step * (2.0 * np.pi * frequencies + interaction)
        history = np.roll(history, 1, axis=0)
        history[0] = phases
        now = times[0] + index * step
        if cursor < len(times) and now >= times[cursor] - 0.5 * step:
            collected[:, cursor] = phases
            cursor += 1
    return collected


def regional_labels(grid: int, blocks: int) -> np.ndarray:
    axis = np.arange(grid)
    block = np.minimum(axis * blocks // grid, blocks - 1)
    return block[:, None] * blocks + block[None, :]
