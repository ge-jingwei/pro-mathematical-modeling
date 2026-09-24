"""Leakage-safe reuse of the existing model and trialwise ridge inversion."""
from pathlib import Path
import importlib.util
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def load_legacy():
    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    helper = load("q2_utils", ROOT / "Ques2" / "utils.py")
    previous = sys.modules.get("utils")
    sys.modules["utils"] = helper
    try:
        model = load("q2_model", ROOT / "Ques2" / "model.py")
    finally:
        if previous is None:
            del sys.modules["utils"]
        else:
            sys.modules["utils"] = previous
    return helper, model


HELPER, MODEL = load_legacy()
TIME = HELPER.TIME
SOURCE_NAMES = ["左源投影", "右源投影", "公共源投影", "左右投影差", "归一选择指数", "源轨迹重建误差", "左右模板相关差", "最佳偏移", "源平均阿尔法能量", "源平均西塔能量", "左右阿尔法能量差", "左右阿尔法能量比"]
SCALP_NAMES = ["左额峰幅", "中额峰幅", "右额峰幅", "左额均幅", "中额均幅", "右额均幅", "左额峰时", "中额峰时", "右额峰时", "左额晚期均幅", "中额晚期均幅", "右额晚期均幅", "双侧差", "中额与双侧差", "晚期双侧差"]
AUXILIARY = [0, 1, 2, 12, 13]
ALPHAS = [1e-5, 1e-3, .03, .3]


def fit_state(X, meta, indices, device):
    kept, thresholds = HELPER.training_quality(X, np.asarray(indices))
    erp = HELPER.condition_means(X, meta, kept, device)
    model = MODEL.fit(erp, variant="simple", starts=3, max_nfev=220)
    G = MODEL.mixing(model["coef"])
    baseline = X[kept][..., TIME < 0].transpose(1, 0, 2).reshape(3, -1)
    cov = np.cov(baseline)
    cov = .9*cov + .1*np.trace(cov)/3*np.eye(3)
    values, vectors = np.linalg.eigh(cov)
    whitening = (vectors/np.sqrt(np.maximum(values, 1e-10)))@vectors.T
    return dict(model=model, G=G, whitening=whitening, training_indices=kept,
                input_indices=np.asarray(indices), thresholds=thresholds, noise_covariance=cov)


def inverse_operator(state, alpha):
    W, G = state["whitening"], state["G"]
    A = W@G
    strength = float(alpha*np.linalg.norm(A, 2)**2)
    return np.linalg.solve(A.T@A+strength*np.eye(3), A.T@W), strength


def invert(X, state, alpha, device):
    operator, strength = inverse_operator(state, alpha)
    q = torch.einsum("sc,nct->nst", torch.as_tensor(operator, dtype=torch.float64, device=device),
                     torch.as_tensor(X, dtype=torch.float64, device=device)).cpu().numpy()
    return q, operator, strength


def scalp_features(X):
    mid, late = (TIME >= .25) & (TIME <= .5), (TIME >= .5) & (TIME <= .8)
    z = X[..., mid]
    return np.column_stack([z.max(-1), z.mean(-1), TIME[mid][z.argmax(-1)], X[..., late].mean(-1),
                            (z[:, 0]-z[:, 2]).mean(-1), (z[:, 1]-(z[:, 0]+z[:, 2])/2).mean(-1),
                            (X[:, 0, late]-X[:, 2, late]).mean(-1)])


def source_features(X, tasks, state, alpha, device):
    q, operator, strength = invert(X, state, alpha, device)
    _, templates = MODEL.predict(state["model"])
    use = (TIME >= 0) & (TIME <= .8)
    z = q[..., use]
    out = np.zeros((len(X), 8))
    for task in [1, 2]:
        ids = np.flatnonzero(tasks == task)
        if not len(ids):
            continue
        k = 2*(task-1)
        base = np.stack([templates[k, 0], templates[k+1, 1], templates[k, 2]])
        observations = z[ids]
        best = np.full(len(ids), np.inf)
        for shift in range(-3, 4):
            template = np.stack([np.interp(TIME-shift/256, TIME, curve, left=0., right=0.) for curve in base])[:, use]
            coefficients = np.sum(observations*template[None], axis=-1)/(np.sum(template**2, axis=-1)+1e-12)
            reconstructed = coefficients[..., None]*template
            error = np.mean((observations-reconstructed)**2, axis=(1, 2))/(np.mean(observations**2, axis=(1, 2))+1e-12)
            improve = error < best
            if not improve.any():
                continue
            correlations = []
            centered = observations.reshape(len(ids), -1)
            centered = centered-centered.mean(-1, keepdims=True)
            for d in [0, 1]:
                candidate = np.stack([np.interp(TIME-shift/256, TIME, curve, left=0., right=0.) for curve in templates[k+d]])[:, use].ravel()
                candidate -= candidate.mean()
                correlations.append(centered@candidate/(np.linalg.norm(centered, axis=1)*np.linalg.norm(candidate)+1e-12))
            delta = coefficients[:, 0]-coefficients[:, 1]
            features = np.column_stack([coefficients, delta, delta/(np.abs(coefficients[:, 0])+np.abs(coefficients[:, 1])+1e-12),
                                        np.sqrt(error), correlations[0]-correlations[1], np.full(len(ids), shift)])
            out[ids[improve]] = features[improve]
            best[improve] = error[improve]
    signal = torch.as_tensor(z, device=device, dtype=torch.float64)
    grid = torch.linspace(-1, 1, z.shape[-1], device=device, dtype=torch.float64)
    signal = signal-signal.mean(-1, keepdim=True)
    signal = signal-(signal@grid)[..., None]/(grid@grid)*grid
    window = torch.hann_window(z.shape[-1], device=device, dtype=torch.float64)
    power = torch.abs(torch.fft.rfft(signal*window, dim=-1))**2/(256*torch.sum(window**2))
    power[..., 1:] *= 2
    if z.shape[-1] % 2 == 0:
        power[..., -1] /= 2
    frequency = torch.fft.rfftfreq(z.shape[-1], 1/256, device=device)
    alpha_power = (power[..., (frequency >= 8) & (frequency <= 13)].sum(-1)*256/z.shape[-1]).cpu().numpy()
    theta_power = (power[..., (frequency >= 4) & (frequency < 8)].sum(-1)*256/z.shape[-1]).cpu().numpy()
    spectral = np.column_stack([np.log(alpha_power.mean(-1)+1e-12), np.log(theta_power.mean(-1)+1e-12),
                                alpha_power[:, 0]-alpha_power[:, 1], np.log((alpha_power[:, 0]+1e-12)/(alpha_power[:, 1]+1e-12))])
    result = np.column_stack([out, spectral])
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite source features")
    return result, q, operator, strength


def feature_sets(X, tasks, state, alpha, device):
    scalp = scalp_features(X)
    source, q, operator, strength = source_features(X, tasks, state, alpha, device)
    return {"erp": scalp, "source": source, "combined": np.column_stack([source[:, :10], scalp[:, AUXILIARY]])}, q, operator, strength


def feature_names(scheme):
    if scheme == "erp":
        return SCALP_NAMES
    if scheme == "source":
        return SOURCE_NAMES
    return SOURCE_NAMES[:10]+[SCALP_NAMES[i] for i in AUXILIARY]

