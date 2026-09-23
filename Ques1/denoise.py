import numpy as np


BASIS = np.column_stack(
    [
        np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0),
        np.array([0.0, 1.0, -1.0]) / np.sqrt(2.0),
        np.array([2.0, -1.0, -1.0]) / np.sqrt(6.0),
    ]
)


def fit_noise_scales(training_epochs: np.ndarray) -> tuple[np.ndarray, float]:
    center = np.median(training_epochs, axis=(0, 2), keepdims=True)
    residual = training_epochs - center
    observation = 1.4826 * np.median(np.abs(residual), axis=(0, 2))
    mode_epochs = np.einsum("ij,njt->nit", BASIS.T, training_epochs)
    differences = np.diff(mode_epochs, axis=2)
    process = float(np.median((1.4826 * np.median(np.abs(differences), axis=(0, 2))) ** 2))
    floor = np.finfo(float).eps
    return np.maximum(observation, floor), max(process, floor)


def reliability_kalman(
    epochs: np.ndarray,
    reliability: np.ndarray,
    observation_scale: np.ndarray,
    process_variance: float,
    reliability_floor: float,
) -> np.ndarray:
    output = np.empty_like(epochs, dtype=float)
    identity = np.eye(3)
    base_variance = observation_scale**2
    state = np.zeros((len(epochs), 3))
    covariance = np.tile(np.diag(base_variance), (len(epochs), 1, 1))
    codes = np.sum((reliability > 0.0) * (1 << np.arange(3))[None, :, None], axis=1)
    for time in range(epochs.shape[2]):
        covariance = covariance + process_variance * identity
        for code in range(1, 8):
            selected = np.flatnonzero(codes[:, time] == code)
            if not len(selected):
                continue
            available = (code & (1 << np.arange(3))) > 0
            design = BASIS[available]
            current_covariance = covariance[selected]
            design_covariance = np.einsum("ai,nij->naj", design, current_covariance)
            innovation = np.einsum("nai,bi->nab", design_covariance, design)
            diagonal = base_variance[available] / np.maximum(
                reliability[selected][:, available, time], reliability_floor
            )
            innovation[:, np.arange(available.sum()), np.arange(available.sum())] += diagonal
            gain = np.linalg.solve(innovation, design_covariance).transpose(0, 2, 1)
            residual = epochs[selected][:, available, time] - state[selected] @ design.T
            state[selected] += np.einsum("nij,nj->ni", gain, residual)
            covariance[selected] = np.einsum(
                "nij,njk->nik",
                identity - np.einsum("nia,aj->nij", gain, design),
                current_covariance,
            )
        output[:, :, time] = state @ BASIS.T
    return output
