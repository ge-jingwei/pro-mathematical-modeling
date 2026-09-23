import numpy as np
from scipy.stats import t, ttest_ind


def bootstrap_mean_ci(epochs: np.ndarray, samples: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = epochs.mean(axis=0)
    indices = rng.integers(0, len(epochs), size=(samples, len(epochs)))
    boot = epochs[indices].mean(axis=1)
    low, high = np.percentile(boot, [2.5, 97.5], axis=0)
    return mean, low, high


def _clusters(statistic: np.ndarray, threshold: float) -> list[np.ndarray]:
    active = np.abs(statistic) >= threshold
    edges = np.flatnonzero(np.diff(np.r_[False, active, False]))
    return [np.arange(a, b) for a, b in edges.reshape(-1, 2)]


def cluster_sign_test(data: np.ndarray, permutations: int, rng: np.random.Generator) -> list[tuple[int, int, float]]:
    n = len(data)
    std = data.std(axis=0, ddof=1)
    observed = np.divide(data.mean(axis=0), std / np.sqrt(n), out=np.zeros(data.shape[1]), where=std > 0)
    threshold = float(t.ppf(0.975, n - 1))
    clusters = _clusters(observed, threshold)
    maxima = np.zeros(permutations)
    for i in range(permutations):
        signed = data * rng.choice((-1.0, 1.0), size=(n, 1))
        spread = signed.std(axis=0, ddof=1)
        statistic = np.divide(signed.mean(axis=0), spread / np.sqrt(n), out=np.zeros(data.shape[1]), where=spread > 0)
        masses = [np.abs(statistic[cluster]).sum() for cluster in _clusters(statistic, threshold)]
        maxima[i] = max(masses, default=0.0)
    result = []
    for cluster in clusters:
        mass = float(np.abs(observed[cluster]).sum())
        p = (1.0 + np.sum(maxima >= mass)) / (permutations + 1.0)
        if p < 0.05:
            result.append((int(cluster[0]), int(cluster[-1] + 1), float(p)))
    return result


def cluster_independent_test(left: np.ndarray, right: np.ndarray, permutations: int, rng: np.random.Generator) -> list[tuple[int, int, float]]:
    observed = np.nan_to_num(ttest_ind(left, right, axis=0, equal_var=False).statistic)
    threshold = float(t.ppf(0.975, min(len(left), len(right)) - 1))
    clusters = _clusters(observed, threshold)
    joined = np.r_[left, right]
    maxima = np.zeros(permutations)
    for i in range(permutations):
        shuffled = joined[rng.permutation(len(joined))]
        statistic = np.nan_to_num(ttest_ind(shuffled[:len(left)], shuffled[len(left):], axis=0, equal_var=False).statistic)
        masses = [np.abs(statistic[cluster]).sum() for cluster in _clusters(statistic, threshold)]
        maxima[i] = max(masses, default=0.0)
    result = []
    for cluster in clusters:
        mass = float(np.abs(observed[cluster]).sum())
        p = (1.0 + np.sum(maxima >= mass)) / (permutations + 1.0)
        if p < 0.05:
            result.append((int(cluster[0]), int(cluster[-1] + 1), float(p)))
    return result


def side_difference(left: np.ndarray, right: np.ndarray, mask: np.ndarray, permutations: int, rng: np.random.Generator) -> tuple[float, float, float]:
    left_values = left[:, mask].mean(axis=1)
    right_values = right[:, mask].mean(axis=1)
    effect = float(right_values.mean() - left_values.mean())
    pooled = np.sqrt(((len(left_values) - 1) * left_values.var(ddof=1) + (len(right_values) - 1) * right_values.var(ddof=1)) / (len(left_values) + len(right_values) - 2))
    d = effect / pooled if pooled > 0 else np.nan
    joined = np.r_[left_values, right_values]
    exceed = 0
    for _ in range(permutations):
        shuffled = rng.permutation(joined)
        delta = shuffled[len(left_values):].mean() - shuffled[:len(left_values)].mean()
        exceed += abs(delta) >= abs(effect)
    return effect, float(d), (exceed + 1.0) / (permutations + 1.0)
