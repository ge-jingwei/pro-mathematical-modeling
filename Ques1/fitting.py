import numpy as np
from scipy.optimize import curve_fit


def gaussian2(t, offset, a1, m1, s1, a2, m2, s2):
    return offset + a1 * np.exp(-0.5 * ((t - m1) / s1) ** 2) + a2 * np.exp(-0.5 * ((t - m2) / s2) ** 2)


def gaussian3(t, offset, a1, m1, s1, a2, m2, s2, a3, m3, s3):
    return gaussian2(t, offset, a1, m1, s1, a2, m2, s2) + a3 * np.exp(-0.5 * ((t - m3) / s3) ** 2)


def _criterion(y: np.ndarray, fitted: np.ndarray, parameters: int) -> tuple[float, float, float]:
    residual = y - fitted
    rss = max(float(residual @ residual), np.finfo(float).eps)
    n = len(y)
    aic = n * np.log(rss / n) + 2 * parameters
    bic = n * np.log(rss / n) + parameters * np.log(n)
    r2 = 1.0 - rss / max(float(np.sum((y - y.mean()) ** 2)), np.finfo(float).eps)
    return r2, aic, bic


def fit_components(times: np.ndarray, values: np.ndarray) -> dict:
    mask = (times >= 0.05) & (times <= 0.6)
    t = times[mask]
    y = values[mask]
    scale = max(float(np.max(np.abs(y))), 1.0)
    bounds2 = ([-2 * scale, -4 * scale, 0.07, 0.015, -4 * scale, 0.15, 0.015], [2 * scale, 4 * scale, 0.18, 0.12, 4 * scale, 0.28, 0.15])
    p02 = [float(np.median(y)), -scale / 2, 0.12, 0.05, scale / 2, 0.22, 0.07]
    p2, _ = curve_fit(gaussian2, t, y, p0=p02, bounds=bounds2, maxfev=30000)
    fit2 = gaussian2(t, *p2)
    bounds3 = (bounds2[0] + [-4 * scale, 0.25, 0.02], bounds2[1] + [4 * scale, 0.5, 0.18])
    p03 = [*p2, scale / 2, 0.35, 0.1]
    p3, _ = curve_fit(gaussian3, t, y, p0=p03, bounds=bounds3, maxfev=50000)
    fit3 = gaussian3(t, *p3)
    r22, aic2, bic2 = _criterion(y, fit2, len(p2))
    r23, aic3, bic3 = _criterion(y, fit3, len(p3))
    return {"params": p3, "fitted": gaussian3(times, *p3), "r2": r23, "aic": aic3, "bic": bic3, "aic_two": aic2, "bic_two": bic2, "r2_two": r22}
