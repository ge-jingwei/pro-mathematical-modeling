"""Forward chain from cortical sources to scalp potentials."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import welch

from neuro.dynamics import FieldParameters, Sigmoid, field_response

ELECTRODES = ("Fz", "F3", "F4")


@dataclass(frozen=True)
class LeadFieldParameters:
    spread: float = 0.52
    depth: float = 0.42
    anterior: float = 0.30


def electrode_positions(parameters: LeadFieldParameters = LeadFieldParameters()) -> np.ndarray:
    return np.array([[0.0, parameters.anterior], [-parameters.spread, parameters.anterior], [parameters.spread, parameters.anterior]])


def lead_field(grid: int, parameters: LeadFieldParameters = LeadFieldParameters()) -> np.ndarray:
    """Inverse-square volume conduction from the cortical patch to Fz, F3 and F4."""
    axis = np.linspace(-1.0, 1.0, grid)
    x, y = np.meshgrid(axis, axis)
    fields = []
    for x0, y0 in electrode_positions(parameters):
        distance = (x - x0) ** 2 + (y - y0) ** 2 + parameters.depth**2
        fields.append(1.0 / distance)
    fields = np.asarray(fields)
    return fields / fields.sum(axis=(1, 2), keepdims=True)


def scalp_potential(source: np.ndarray, fields: np.ndarray) -> np.ndarray:
    return np.einsum("cyx,yxt->ct", fields, source)


def forward(density: np.ndarray, times: np.ndarray, excitatory: Sigmoid, inhibitory: Sigmoid, fields: np.ndarray, parameters: FieldParameters = FieldParameters()) -> np.ndarray:
    source = field_response(density, times, excitatory, inhibitory, parameters)
    potential = scalp_potential(source, fields)
    baseline = (times >= -0.2) & (times < 0.0)
    return potential - potential[:, baseline].mean(axis=1, keepdims=True)


def waveform_correlation(observed: np.ndarray, predicted: np.ndarray, times: np.ndarray, low: float = 0.0, high: float = 0.75) -> float:
    mask = (times >= low) & (times <= high)
    values = [np.corrcoef(observed[channel, mask], predicted[channel, mask])[0, 1] for channel in range(len(observed))]
    return float(np.nanmean(values))


def spectral_cosine(observed: np.ndarray, predicted: np.ndarray, rate: float) -> float:
    values = []
    for channel in range(len(observed)):
        _, first = welch(observed[channel], rate, nperseg=min(128, observed.shape[-1]))
        _, second = welch(predicted[channel], rate, nperseg=min(128, predicted.shape[-1]))
        values.append(float(first @ second / np.sqrt((first @ first) * (second @ second))))
    return float(np.nanmean(values))
