"""Retina, LGN and cortical shape encoding."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve
from scipy.special import expit

IMAGE_SIZE = 64
VERTICES = (np.array([-0.72, 0.0]), np.array([0.28, 0.95]), np.array([0.28, -0.95]))


@dataclass(frozen=True)
class EncodingParameters:
    size: int = IMAGE_SIZE
    scales: tuple[float, ...] = (0.9, 1.8, 3.2)
    orientations: int = 8
    gabor_sigma: float = 2.0
    gabor_wavelength: float = 5.0
    gabor_elongation: float = 0.6
    apex_offset: int = 4
    apex_samples: int = 2
    apex_sigma: float = 0.6
    opponent_width: float = 0.10


def triangle(size: int = IMAGE_SIZE, side: int = -1, softness: float = 0.7) -> np.ndarray:
    """Filled triangle pointing left (side=-1) or right (side=+1)."""
    y, x = np.mgrid[-1:1:complex(size), -1:1:complex(size)]
    apex, upper, lower = VERTICES

    def half_plane(start: np.ndarray, stop: np.ndarray) -> np.ndarray:
        return (x - start[0]) * (stop[1] - start[1]) - (y - start[1]) * (stop[0] - start[0])

    inside = (half_plane(apex, upper) >= 0) & (half_plane(upper, lower) >= 0) & (half_plane(lower, apex) >= 0)
    image = gaussian_filter(inside.astype(float), softness)
    return np.fliplr(image) if side > 0 else image


def dog_kernel(sigma_center: float, sigma_surround: float) -> np.ndarray:
    radius = int(np.ceil(3.0 * sigma_surround))
    axis = np.arange(-radius, radius + 1, dtype=float)
    x, y = np.meshgrid(axis, axis)
    squared = x**2 + y**2
    kernel = np.exp(-squared / (2.0 * sigma_center**2)) / (2.0 * np.pi * sigma_center**2)
    kernel -= np.exp(-squared / (2.0 * sigma_surround**2)) / (2.0 * np.pi * sigma_surround**2)
    return kernel - kernel.mean()


def gabor_kernel(theta: float, parameters: EncodingParameters, phase: float = 0.0) -> np.ndarray:
    """Gabor filter whose ``theta`` is the preferred edge orientation."""
    radius = int(np.ceil(2.5 * parameters.gabor_sigma))
    axis = np.arange(-radius, radius + 1, dtype=float)
    x, y = np.meshgrid(axis, axis)
    along = x * np.cos(theta) + y * np.sin(theta)
    across = -x * np.sin(theta) + y * np.cos(theta)
    envelope = np.exp(-(along**2 + (parameters.gabor_elongation * across) ** 2) / (2.0 * parameters.gabor_sigma**2))
    kernel = envelope * np.cos(2.0 * np.pi * across / parameters.gabor_wavelength + phase)
    return kernel - kernel.mean()


def lgn_response(image: np.ndarray, parameters: EncodingParameters) -> np.ndarray:
    """Centre-surround band-pass stage with local contrast gain control."""
    channels = []
    for sigma in parameters.scales:
        response = fftconvolve(image, dog_kernel(sigma, 2.0 * sigma), mode="same")
        channels.append(response / (np.sqrt(gaussian_filter(response**2, 4.0 * sigma)) + 1e-6))
    return np.asarray(channels)


def orientation_energy(lgn: np.ndarray, parameters: EncodingParameters) -> np.ndarray:
    """V1 complex-cell energy per orientation, pooled over scales."""
    thetas = np.linspace(0.0, np.pi, parameters.orientations, endpoint=False)
    energy = np.empty((parameters.orientations, *lgn.shape[-2:]))
    for index, theta in enumerate(thetas):
        pair = [np.maximum(fftconvolve(lgn.sum(axis=0), gabor_kernel(theta, parameters, phase), mode="same"), 0.0) for phase in (0.0, np.pi / 2.0)]
        energy[index] = np.hypot(pair[0], pair[1])
    peak = float(energy.max())
    return energy / (peak if peak > 0 else 1.0)


def orientation_index(theta: float, parameters: EncodingParameters) -> int:
    thetas = np.linspace(0.0, np.pi, parameters.orientations, endpoint=False)
    distance = np.abs(np.mod(theta - thetas + np.pi / 2.0, np.pi) - np.pi / 2.0)
    return int(np.argmin(distance))


def edge_geometry(side: int, parameters: EncodingParameters) -> list[tuple[np.ndarray, float]]:
    """Edge directions leaving the apex, as (unit offset, orientation angle)."""
    apex, upper, lower = VERTICES
    if side > 0:
        flip = np.array([-1.0, 1.0])
        apex, upper, lower = apex * flip, lower * flip, upper * flip

    def to_pixel(point: np.ndarray) -> np.ndarray:
        return (point + 1.0) * 0.5 * (parameters.size - 1)

    origin = to_pixel(apex)
    pairs = []
    for stop in (upper, lower):
        direction = to_pixel(stop) - origin
        direction = direction / np.linalg.norm(direction)
        pairs.append((direction, float(np.mod(np.arctan2(direction[1], direction[0]), np.pi))))
    return pairs


def apex_maps(energy: np.ndarray, parameters: EncodingParameters) -> tuple[np.ndarray, np.ndarray]:
    """Structural detectors for a leftward and a rightward apex.

    An apex is the conjunction of two diagonal edges at fixed relative
    positions, in the spirit of the structural feature detectors hinted at in
    the task statement.
    """
    step = parameters.apex_offset

    def sample(channel: np.ndarray, offset: np.ndarray) -> np.ndarray:
        return np.roll(np.roll(channel, -int(round(offset[1])), axis=0), -int(round(offset[0])), axis=1)

    maps = []
    for side in (-1, 1):
        product = np.ones_like(energy[0])
        for direction, angle in edge_geometry(side, parameters):
            channel = energy[orientation_index(angle, parameters)]
            for multiple in range(1, parameters.apex_samples + 1):
                product = product * sample(channel, multiple * step * direction)
        maps.append(gaussian_filter(product ** (1.0 / (2.0 * parameters.apex_samples)), parameters.apex_sigma))
    return maps[0], maps[1]


def population_signal(left: np.ndarray, right: np.ndarray) -> float:
    """Normalised mass difference of the two opponent feature populations."""
    total = float(left.sum() + right.sum())
    return float((left.sum() - right.sum()) / total) if total > 1e-12 else 0.0


def opponent_activation(offset: float, width: float) -> tuple[float, float]:
    """Smooth opponent read-out of the signed apex signal."""
    left = float(expit(offset / width))
    return left, 1.0 - left


@dataclass(frozen=True)
class Encoding:
    image: np.ndarray
    lgn: np.ndarray
    energy: np.ndarray
    apex_left: np.ndarray
    apex_right: np.ndarray
    offset: float
    population: tuple[float, float]


def encode(side: int, parameters: EncodingParameters = EncodingParameters()) -> Encoding:
    image = triangle(parameters.size, side)
    lgn = lgn_response(image, parameters)
    energy = orientation_energy(lgn, parameters)
    left, right = apex_maps(energy, parameters)
    offset = population_signal(left, right)
    return Encoding(image, lgn, energy, left, right, offset, opponent_activation(offset, parameters.opponent_width))


def population_map(grid: int, x0: float, sigma: float) -> np.ndarray:
    axis = np.linspace(-1.0, 1.0, grid)
    x, y = np.meshgrid(axis, axis)
    return np.exp(-((x - x0) ** 2 + y**2) / (2.0 * sigma**2))


def cortical_density(encoding: Encoding, grid: int, anchor: float, spacing: float, sigma: float) -> np.ndarray:
    """Map the two opponent feature populations onto the cortical sheet.

    ``anchor`` is the map position of the foveal representation and ``spacing``
    the separation of the two opponent feature domains.  The pair is a mirror
    image of itself only when ``anchor`` sits exactly on the midline, which is
    what decides whether the scalp projection keeps a first-order term.
    """
    left = population_map(grid, anchor - 0.5 * spacing, sigma)
    right = population_map(grid, anchor + 0.5 * spacing, sigma)
    return encoding.population[0] * left + encoding.population[1] * right
