"""Batched GPU implementation of the mesoscopic field used for parameter sweeps."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as functional

from neuro.dynamics import FieldParameters, Sigmoid

SWEPT = ("delay", "gain", "feedback_gain", "feedback_delay", "feedback_width", "feedback_inhibition", "coupling", "bias", "w_ee", "w_ei", "w_ie", "w_ii", "tau_e", "tau_i", "adaptation", "tau_adaptation", "delay_gradient")
FIXED = ("envelope_width",)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gaussian_kernel(sigma: float, current: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Matches scipy.ndimage.gaussian_filter: truncation at 4 sigma, reflect boundary."""
    radius = int(4.0 * sigma + 0.5)
    axis = torch.arange(-radius, radius + 1, device=current, dtype=dtype)
    kernel = torch.exp(-(axis**2) / (2.0 * sigma**2))
    kernel = kernel / kernel.sum()
    return (kernel[:, None] * kernel[None, :])[None, None]


def _alpha(times: torch.Tensor, onset: torch.Tensor, width: torch.Tensor) -> torch.Tensor:
    scaled = torch.clamp((times - onset) / width, min=0.0)
    return torch.where(times >= onset, scaled * torch.exp(1.0 - scaled), torch.zeros_like(scaled))


def field_response_batch(density: np.ndarray, times: np.ndarray, excitatory: Sigmoid, inhibitory: Sigmoid, parameters: FieldParameters, sweep: dict[str, np.ndarray] | None = None, chunk: int = 128, dtype: torch.dtype = torch.float64, fields: np.ndarray | None = None) -> np.ndarray:
    """Run the field for every parameter combination in ``sweep`` at once.

    ``sweep`` maps field parameter names to one-dimensional arrays that run
    along the leading batch axis.  When ``fields`` is given, the cortical sheet
    is projected to scalp potentials inside the batch loop and the result has
    shape ``(size, electrodes, time)`` with the pre-stimulus baseline removed.
    """
    sweep = sweep or {}
    size = len(next(iter(sweep.values()))) if sweep else 1
    order = {name: np.asarray(values) for name, values in sweep.items()}
    current = device()
    grid = density.shape[-1]
    steps = int(round((times[-1] - times[0]) / parameters.step)) + 1
    grid_times = times[0] + np.arange(steps) * parameters.step
    coordinate = np.linspace(-1.0, 1.0, grid)
    baseline = (times >= -0.2) & (times < 0.0)
    lead = torch.as_tensor(fields, device=current, dtype=dtype) if fields is not None else None
    result = np.empty((size, lead.shape[0] if lead is not None else grid, len(times)) if lead is not None else (size, grid, grid, len(times)))
    for start in range(0, size, chunk):
        stop = min(start + chunk, size)
        span = stop - start
        values = {}
        for name in (*SWEPT, *FIXED):
            if name in sweep:
                values[name] = torch.as_tensor(order[name][start:stop], device=current, dtype=dtype).reshape(span, 1, 1)
            else:
                values[name] = torch.full((span, 1, 1), float(getattr(parameters, name)), device=current, dtype=dtype)
        profile = torch.as_tensor(coordinate, device=current, dtype=dtype).reshape(1, 1, grid)
        onset = values["delay"] + values["delay_gradient"] * profile
        feedback = values["feedback_delay"] + values["delay_gradient"] * profile
        density_t = torch.as_tensor(density, device=current, dtype=dtype).expand(span, grid, grid)
        kernel = gaussian_kernel(float(parameters.spatial_sigma), current, dtype)
        padding = kernel.shape[-1] // 2
        e = torch.full((span, grid, grid), 0.02, device=current, dtype=dtype)
        i = torch.full_like(e, 0.02)
        adaptation = torch.zeros_like(e)
        source = torch.zeros((span, grid, grid, len(times)), device=current, dtype=dtype)
        cursor = 0
        for index in range(steps):
            moment = torch.full((span, 1, 1), float(grid_times[index]), device=current, dtype=dtype)
            lateral = functional.conv2d(functional.pad(e[:, None], (padding,) * 4, mode="reflect"), kernel)[:, 0]
            direct = _alpha(moment, onset, values["envelope_width"])
            recurrent = _alpha(moment, feedback, values["feedback_width"])
            drive_e = (values["gain"] * direct + values["feedback_gain"] * recurrent) * density_t
            drive_i = values["feedback_gain"] * values["feedback_inhibition"] * recurrent * density_t
            input_e = values["w_ee"] * e - values["w_ei"] * i + values["coupling"] * (lateral - e) + drive_e - values["bias"] - values["adaptation"] * adaptation
            input_i = values["w_ie"] * e - values["w_ii"] * i + drive_i - values["bias"]
            e = torch.clamp(e + parameters.step / values["tau_e"] * (-e + (1.0 - e) * torch.sigmoid(excitatory.slope * (input_e - excitatory.threshold))), 0.0, 1.0)
            i = torch.clamp(i + parameters.step / values["tau_i"] * (-i + (1.0 - i) * torch.sigmoid(inhibitory.slope * (input_i - inhibitory.threshold))), 0.0, 1.0)
            adaptation = adaptation + parameters.step / values["tau_adaptation"] * (-adaptation + e)
            if cursor < len(times) and grid_times[index] >= times[cursor] - 0.5 * parameters.step:
                source[..., cursor] = e - 0.72 * i
                cursor += 1
        if lead is None:
            result[start:stop] = source.cpu().numpy()
        else:
            potential = torch.einsum("cxy,bxyt->bct", lead, source)
            potential = potential - potential[..., baseline].mean(dim=-1, keepdim=True)
            result[start:stop] = potential.cpu().numpy()
    return result


def parity(density: np.ndarray, times: np.ndarray, excitatory: Sigmoid, inhibitory: Sigmoid, parameters: FieldParameters = FieldParameters()) -> float:
    """Largest deviation between the batched GPU path and the reference implementation."""
    from neuro.dynamics import field_response

    reference = field_response(density, times, excitatory, inhibitory, parameters)
    batched = field_response_batch(density, times, excitatory, inhibitory, parameters)[0]
    return float(np.abs(reference - batched).max())
