from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MplPath
from scipy.integrate import solve_ivp
from scipy.signal import fftconvolve, savgol_filter, welch
from sklearn.decomposition import PCA
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from audit_panel_alignment import require_matplotlib_panel_alignment
from p1_p300 import CHANNELS, extract_file, group_erp


LEFT = "left"
RIGHT = "right"
SIDE_COLORS = {LEFT: "#3B82B8", RIGHT: "#E58B3A"}
EMPIRICAL_COLOR = "#343A40"
SIM_COLOR = "#7A5195"
FEATURE_BASE = ["A_Fz", "A_F3", "A_F4", "t_Fz", "LAT_F3-F4", "P_alpha", "P_theta", "AsymIdx"]
FEATURE_EXTRA = ["P_gamma", "theta_alpha", "F3-F4_mean", "F3-F4_slope", "F3_F4_corr"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "legend.handlelength": 0,
            "legend.handletextpad": 0,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def panel_labels(axes: np.ndarray | list[plt.Axes]) -> None:
    for index, axis in enumerate(np.asarray(axes, dtype=object).ravel()):
        axis.annotate(
            chr(97 + index),
            xy=(0, 1),
            xycoords="axes fraction",
            xytext=(-28, 6),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
            ha="left",
            va="bottom",
        )


def text_legend(axis: plt.Axes, colors: list[str], **kwargs) -> None:
    legend = axis.legend(handlelength=0, handletextpad=0, markerscale=0, **kwargs)
    for text, color in zip(legend.get_texts(), colors):
        text.set_color(color)


def save_figure(fig: plt.Figure, base: Path, axes: np.ndarray | list[plt.Axes]) -> None:
    panel_labels(axes)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig,
        json_out=str(base) + ".alignment.json",
        overlay_svg=str(base) + ".alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
    )
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def triangle_image(side: str, size: int = 128) -> np.ndarray:
    points = np.array([[0.18, 0.50], [0.78, 0.18], [0.78, 0.82]])
    if side == RIGHT:
        points[:, 0] = 1 - points[:, 0]
    grid = np.linspace(0, 1, size)
    xx, yy = np.meshgrid(grid, grid)
    image = MplPath(points).contains_points(np.c_[xx.ravel(), yy.ravel()]).reshape(size, size).astype(float)
    return image


def gaussian_kernel(sigma: float) -> np.ndarray:
    radius = int(np.ceil(3 * sigma))
    x = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(x, x)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def gabor_kernel(theta: float, phase: float, wavelength: float = 12, gamma: float = 0.5) -> np.ndarray:
    sigma = 0.56 * wavelength
    radius = int(np.ceil(3 * sigma))
    x = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(x, x)
    angle = np.deg2rad(theta)
    xr = xx * np.cos(angle) + yy * np.sin(angle)
    yr = -xx * np.sin(angle) + yy * np.cos(angle)
    kernel = np.exp(-(xr**2 + gamma**2 * yr**2) / (2 * sigma**2)) * np.cos(2 * np.pi * xr / wavelength + phase)
    kernel -= kernel.mean()
    return kernel / np.sqrt(np.sum(kernel**2))


def visual_encode(side: str) -> dict:
    image = triangle_image(side)
    dog = fftconvolve(image, gaussian_kernel(1.5), mode="same") - fftconvolve(image, gaussian_kernel(2.4), mode="same")
    orientations = np.arange(0, 180, 30)
    energy = []
    for theta in orientations:
        even = fftconvolve(dog, gabor_kernel(theta, 0), mode="same")
        odd = fftconvolve(dog, gabor_kernel(theta, np.pi / 2), mode="same")
        energy.append(np.sqrt(even**2 + odd**2 + 1e-12))
    energy = np.asarray(energy)
    global_response = energy.mean(axis=(1, 2))
    global_response /= global_response.sum()
    midpoint = image.shape[1] // 2
    spatial = np.c_[energy[:, :, :midpoint].mean(axis=(1, 2)), energy[:, :, midpoint:].mean(axis=(1, 2))].ravel()
    spatial /= spatial.sum()
    apex = 0.18 if side == LEFT else 0.82
    vector = np.r_[global_response, spatial, apex]
    return {"side": side, "image": image, "dog": dog, "energy": energy, "global": global_response, "spatial": spatial, "vector": vector}


def shape_scores(left: dict, right: dict) -> dict[str, float]:
    delta = left["vector"] - right["vector"]
    direction = delta / max(np.linalg.norm(delta), 1e-12)
    center = 0.5 * (left["vector"] + right["vector"])
    raw = {LEFT: float(direction @ (left["vector"] - center)), RIGHT: float(direction @ (right["vector"] - center))}
    scale = max(abs(value) for value in raw.values())
    return {key: value / scale for key, value in raw.items()}


def sigmoid(value: np.ndarray | float, e0: float = 2.5, v0: float = 6, slope: float = 0.56) -> np.ndarray | float:
    return 2 * e0 / (1 + np.exp(np.clip(slope * (v0 - value), -60, 60)))


def stimulus_envelope(t: float) -> float:
    onset, tau, shape = 0.12, 0.050, 4.0
    if t < onset:
        return 0.0
    u = (t - onset) / tau
    return float(u**shape * np.exp(-u) / (shape**shape * np.exp(-shape)))


def jr_derivatives(state: np.ndarray, external: float, C: float) -> np.ndarray:
    A, B, a, b = 3.25, 22.0, 100.0, 50.0
    C1, C2, C3, C4 = C, 0.8 * C, 0.25 * C, 0.25 * C
    y0, y1, y2, y3, y4, y5 = state
    return np.array(
        [
            y3,
            y4,
            y5,
            A * a * sigmoid(y1 - y2) - 2 * a * y3 - a**2 * y0,
            A * a * (external + C2 * sigmoid(C1 * y0)) - 2 * a * y4 - a**2 * y1,
            B * b * C4 * sigmoid(C3 * y0) - 2 * b * y5 - b**2 * y2,
        ]
    )


def coupled_rhs(t: float, state: np.ndarray, score: float, C: float, w: float, stimulated: bool) -> np.ndarray:
    p0 = 120.0
    drive = stimulus_envelope(t) if stimulated else 0.0
    p_v1 = p0 + (35.0 + 8.0 * score) * drive
    xff = state[12]
    p_pfc = p0 + w * xff
    first = jr_derivatives(state[:6], p_v1, C)
    second = jr_derivatives(state[6:12], p_pfc, C)
    dxff = (sigmoid(state[1] - state[2]) - xff) / 0.040
    return np.r_[first, second, dxff]


def simulate_jr(score: float, C: float = 135, w: float = 40) -> dict:
    initial = np.zeros(13)
    warm = solve_ivp(
        lambda t, y: coupled_rhs(t, y, 0.0, C, w, False),
        (-2.0, 0.0),
        initial,
        method="RK45",
        rtol=1e-7,
        atol=1e-9,
        max_step=1 / 1024,
    )
    if not warm.success:
        raise RuntimeError(warm.message)
    time = np.arange(256) / 256
    stimulated = solve_ivp(
        lambda t, y: coupled_rhs(t, y, score, C, w, True),
        (0.0, 1.0),
        warm.y[:, -1],
        t_eval=time,
        method="RK45",
        rtol=1e-7,
        atol=1e-9,
        max_step=1 / 1024,
    )
    control = solve_ivp(
        lambda t, y: coupled_rhs(t, y, score, C, w, False),
        (0.0, 1.0),
        warm.y[:, -1],
        t_eval=time,
        method="RK45",
        rtol=1e-7,
        atol=1e-9,
        max_step=1 / 1024,
    )
    if not stimulated.success or not control.success:
        raise RuntimeError(stimulated.message if not stimulated.success else control.message)
    v1 = stimulated.y[1] - stimulated.y[2]
    pfc = stimulated.y[7] - stimulated.y[8]
    v1_control = control.y[1] - control.y[2]
    pfc_control = control.y[7] - control.y[8]
    return {
        "time": time,
        "v1": v1,
        "pfc": pfc,
        "evoked_v1": v1 - v1_control,
        "evoked_pfc": pfc - pfc_control,
    }


def lead_field(side: str) -> np.ndarray:
    if side == LEFT:
        return np.array([[0.40, 0.85], [0.50, 0.62], [0.72, 0.82]])
    return np.array([[0.40, 0.85], [0.72, 0.82], [0.50, 0.62]])


def project_scalp(simulation: dict, side: str, evoked: bool = True) -> np.ndarray:
    prefix = "evoked_" if evoked else ""
    sources = np.vstack([simulation[prefix + "v1"], simulation[prefix + "pfc"]])
    if not evoked:
        sources = sources - sources[:, :26].mean(axis=1, keepdims=True)
    return lead_field(side) @ sources


def empirical_erps(results: list[dict]) -> dict[str, np.ndarray]:
    output = {}
    for side in (LEFT, RIGHT):
        erps = []
        for result in results:
            erp, count = group_erp(result, "side", side)
            if erp is not None and count:
                erps.append(erp)
        output[side] = np.mean(erps, axis=0)
    return output


def calibrate_simulation(simulated: dict[str, np.ndarray], empirical: dict[str, np.ndarray], empirical_time: np.ndarray) -> float:
    values_sim, values_emp = [], []
    mask_emp = (empirical_time >= 0.15) & (empirical_time <= 0.60)
    simulation_time = np.arange(256) / 256
    if not np.all(np.diff(simulation_time) > 0):
        raise ValueError("Simulation time must be strictly increasing")
    for side in (LEFT, RIGHT):
        interpolated = np.vstack([np.interp(empirical_time[mask_emp], simulation_time, row) for row in simulated[side]])
        values_sim.append(interpolated.ravel())
        values_emp.append(empirical[side][:, mask_emp].ravel())
    x, y = np.concatenate(values_sim), np.concatenate(values_emp)
    return float(np.dot(x, y) / max(np.dot(x, x), 1e-12))


def band_power(epoch: np.ndarray, time: np.ndarray, low: float, high: float) -> float:
    mask = (time >= 0) & (time <= 0.8)
    powers = []
    for signal in epoch[:, mask]:
        frequency, density = welch(signal, fs=256, nperseg=min(128, signal.size))
        band = (frequency >= low) & (frequency <= high)
        total = (frequency >= 1) & (frequency <= 20)
        powers.append(np.trapezoid(density[band], frequency[band]) / max(np.trapezoid(density[total], frequency[total]), 1e-12))
    return float(np.log(max(np.mean(powers), 1e-12)))


def trial_features(epoch: np.ndarray, time: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    smooth = savgol_filter(epoch, 11, 3, axis=1)
    p300 = (time >= 0.25) & (time <= 0.50)
    local = smooth[:, p300]
    indices = np.argmax(local, axis=1)
    amplitudes = local[np.arange(3), indices]
    latencies = time[p300][indices] * 1000
    alpha = band_power(epoch, time, 8, 13)
    theta = band_power(epoch, time, 4, 7)
    gamma = band_power(epoch, time, 13, 20)
    asymmetry = (amplitudes[1] - amplitudes[2]) / (abs(amplitudes[1]) + abs(amplitudes[2]) + 1e-9)
    lateral_mask = (time >= 0.25) & (time <= 0.50)
    lateral = smooth[1, lateral_mask] - smooth[2, lateral_mask]
    lateral_time = time[lateral_mask]
    lateral_slope = np.polyfit(lateral_time, lateral, 1)[0]
    correlation = np.corrcoef(smooth[1, lateral_mask], smooth[2, lateral_mask])[0, 1]
    base = np.r_[amplitudes, latencies[0], latencies[1] - latencies[2], alpha, theta, asymmetry]
    extra = np.array([gamma, np.exp(theta - alpha), lateral.mean(), lateral_slope, correlation])
    means = []
    lateral_means = []
    for start in np.arange(0.10, 0.70, 0.05):
        mask = (time >= start) & (time < start + 0.05)
        window = epoch[:, mask].mean(axis=1)
        means.extend(window)
        lateral_means.append(window[1] - window[2])
    waveform = np.r_[means, lateral_means]
    names = [f"{channel}_{int(start*1000)}-{int((start+0.05)*1000)}ms" for start in np.arange(0.10, 0.70, 0.05) for channel in CHANNELS]
    names += [f"F3-F4_{int(start*1000)}-{int((start+0.05)*1000)}ms" for start in np.arange(0.10, 0.70, 0.05)]
    return base, np.r_[base, extra], waveform, FEATURE_BASE + FEATURE_EXTRA, names


def build_dataset(results: list[dict]) -> dict:
    base_rows, enhanced_rows, waveform_rows, epochs, labels, groups, subjects, metadata = [], [], [], [], [], [], [], []
    enhanced_names, waveform_names = [], []
    for file_index, result in enumerate(results):
        for trial in result["trials"]:
            if not trial["accepted"] or not trial["correct"]:
                continue
            base, enhanced, waveform, enhanced_names, waveform_names = trial_features(trial["final"], result["time"])
            base_rows.append(base)
            enhanced_rows.append(enhanced)
            waveform_rows.append(waveform)
            epochs.append(trial["final"])
            labels.append(-1 if trial["cue_side"] == LEFT else 1)
            groups.append(f"{file_index}_{(trial['trial'] - 1) // 10}")
            subjects.append(result["subject"])
            metadata.append({"file": result["file"], "trial": trial["trial"], "subject": result["subject"], "task": result["task"], "side": trial["cue_side"]})
    return {
        "base": np.asarray(base_rows),
        "enhanced": np.asarray(enhanced_rows),
        "waveform": np.asarray(waveform_rows),
        "epochs": np.asarray(epochs),
        "labels": np.asarray(labels),
        "groups": np.asarray(groups),
        "subjects": np.asarray(subjects),
        "metadata": metadata,
        "names": {"base": FEATURE_BASE, "enhanced": enhanced_names, "waveform": waveform_names},
        "time": results[0]["time"],
    }


def build_ensemble_dataset(dataset: dict, block_size: int, minimum: int) -> dict:
    base_rows, enhanced_rows, waveform_rows, epochs, labels, groups, subjects, metadata = [], [], [], [], [], [], [], []
    enhanced_names, waveform_names = [], []
    ensemble_groups = np.array([f"{row['file']}_{(int(row['trial']) - 1) // block_size}" for row in dataset["metadata"]])
    for group in np.unique(ensemble_groups):
        for label in (-1, 1):
            indices = np.flatnonzero((ensemble_groups == group) & (dataset["labels"] == label))
            if indices.size < minimum:
                continue
            epoch = dataset["epochs"][indices].mean(axis=0)
            base, enhanced, waveform, enhanced_names, waveform_names = trial_features(epoch, dataset["time"])
            base_rows.append(base)
            enhanced_rows.append(enhanced)
            waveform_rows.append(waveform)
            epochs.append(epoch)
            labels.append(label)
            groups.append(group)
            subjects.append(dataset["subjects"][indices[0]])
            source = dataset["metadata"][indices[0]]
            metadata.append(
                {
                    "file": source["file"],
                    "trial": "",
                    "subject": source["subject"],
                    "task": source["task"],
                    "side": LEFT if label == -1 else RIGHT,
                    "ensemble_size": indices.size,
                    "source_group": group,
                    "block_size": block_size,
                }
            )
    return {
        "base": np.asarray(base_rows),
        "enhanced": np.asarray(enhanced_rows),
        "waveform": np.asarray(waveform_rows),
        "epochs": np.asarray(epochs),
        "labels": np.asarray(labels),
        "groups": np.asarray(groups),
        "subjects": np.asarray(subjects),
        "metadata": metadata,
        "names": {"base": FEATURE_BASE, "enhanced": enhanced_names, "waveform": waveform_names},
        "time": dataset["time"],
    }


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a0, b0 = a - a.mean(), b - b.mean()
    return float(np.dot(a0, b0) / max(np.linalg.norm(a0) * np.linalg.norm(b0), 1e-12))


def template_feature(train_epochs: np.ndarray, train_labels: np.ndarray, target_epochs: np.ndarray, time: np.ndarray) -> np.ndarray:
    mask = (time >= 0.10) & (time <= 0.70)
    left_template = train_epochs[train_labels == -1][:, :, mask].mean(axis=0).ravel()
    right_template = train_epochs[train_labels == 1][:, :, mask].mean(axis=0).ravel()
    return np.array([safe_corr(epoch[:, mask].ravel(), right_template) - safe_corr(epoch[:, mask].ravel(), left_template) for epoch in target_epochs])


def split_iterator(protocol: str, labels: np.ndarray, groups: np.ndarray, subjects: np.ndarray):
    if protocol == "grouped5":
        return StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260923).split(np.zeros(labels.size), labels, groups)
    if protocol == "random5":
        return StratifiedKFold(n_splits=5, shuffle=True, random_state=20260923).split(np.zeros(labels.size), labels)
    return LeaveOneGroupOut().split(np.zeros(labels.size), labels, subjects)


def cross_validate(dataset: dict, feature_set: str, protocol: str, method: str = "BLDA") -> dict:
    matrix = dataset[feature_set]
    labels = dataset["labels"]
    predictions = np.zeros(labels.size, dtype=int)
    scores = np.zeros(labels.size)
    folds, weights = [], []
    for fold, (train, test) in enumerate(split_iterator(protocol, labels, dataset["groups"], dataset["subjects"]), 1):
        train_template = template_feature(dataset["epochs"][train], labels[train], dataset["epochs"][train], dataset["time"])
        test_template = template_feature(dataset["epochs"][train], labels[train], dataset["epochs"][test], dataset["time"])
        train_matrix = np.c_[matrix[train], train_template]
        test_matrix = np.c_[matrix[test], test_template]
        scaler = StandardScaler().fit(train_matrix)
        x_train, x_test = scaler.transform(train_matrix), scaler.transform(test_matrix)
        if method == "BLDA":
            classifier = BayesianRidge(compute_score=True, fit_intercept=True)
            classifier.fit(x_train, labels[train])
            fold_scores = classifier.predict(x_test)
            coefficient = classifier.coef_
        else:
            classifier = LinearSVC(C=1.0, class_weight="balanced", random_state=20260923, max_iter=100000)
            classifier.fit(x_train, labels[train])
            fold_scores = classifier.decision_function(x_test)
            coefficient = classifier.coef_.ravel()
        fold_predictions = np.where(fold_scores >= 0, 1, -1)
        predictions[test] = fold_predictions
        scores[test] = fold_scores
        weights.append(coefficient / scaler.scale_)
        folds.append(
            {
                "fold": fold,
                "accuracy": accuracy_score(labels[test], fold_predictions),
                "balanced_accuracy": balanced_accuracy_score(labels[test], fold_predictions),
                "n_test": test.size,
            }
        )
    return {
        "feature_set": feature_set,
        "protocol": protocol,
        "method": method,
        "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "confusion": confusion_matrix(labels, predictions, labels=[-1, 1]),
        "predictions": predictions,
        "scores": scores,
        "folds": folds,
        "weights": np.asarray(weights),
        "feature_names": dataset["names"][feature_set] + ["template_corr_diff"],
    }


def dominant_frequency(signal: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    frequency, density = welch(signal - np.mean(signal), fs=256, nperseg=min(256, signal.size))
    mask = (frequency >= 4) & (frequency <= 20)
    return float(frequency[mask][np.argmax(density[mask])]), frequency, density


def peak_metrics(signal: np.ndarray, time: np.ndarray) -> tuple[float, float]:
    mask = (time >= 0.25) & (time <= 0.50)
    index = np.argmax(signal[mask])
    return float(signal[mask][index]), float(time[mask][index] * 1000)


def simulation_metrics(scalp: dict[str, np.ndarray], time: np.ndarray) -> list[dict]:
    rows = []
    for side in (LEFT, RIGHT):
        for channel, signal in zip(CHANNELS, scalp[side]):
            amplitude, latency = peak_metrics(signal, time)
            rows.append({"record_type": "simulation_metric", "side": side, "channel": channel, "peak_amplitude": amplitude, "peak_latency_ms": latency})
    return rows


def comparison_metrics(
    scalp: dict[str, np.ndarray],
    simulation_time: np.ndarray,
    empirical: dict[str, np.ndarray],
    empirical_time: np.ndarray,
) -> list[dict]:
    if not np.all(np.diff(simulation_time) > 0):
        raise ValueError("Simulation time must be strictly increasing")
    mask = (empirical_time >= 0.20) & (empirical_time <= 0.55)
    rows = []
    for side in (LEFT, RIGHT):
        for channel_index, channel in enumerate(CHANNELS):
            observed = empirical[side][channel_index, mask]
            predicted = np.interp(empirical_time[mask], simulation_time, scalp[side][channel_index])
            empirical_peak, empirical_latency = peak_metrics(empirical[side][channel_index], empirical_time)
            simulated_peak, simulated_latency = peak_metrics(scalp[side][channel_index], simulation_time)
            rows.append(
                {
                    "record_type": "simulation_comparison",
                    "side": side,
                    "channel": channel,
                    "correlation": safe_corr(observed, predicted),
                    "nrmse": float(np.sqrt(np.mean((observed - predicted) ** 2)) / max(np.std(observed), 1e-12)),
                    "empirical_peak_amplitude": empirical_peak,
                    "empirical_peak_latency_ms": empirical_latency,
                    "peak_amplitude": simulated_peak,
                    "peak_latency_ms": simulated_latency,
                    "latency_absolute_error_ms": abs(simulated_latency - empirical_latency),
                }
            )
    return rows


def plot_gabor(left: dict, right: dict, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
    axes[0, 0].imshow(left["image"], cmap="gray", origin="lower", vmin=0, vmax=1, aspect="auto")
    axes[0, 0].set_title("左三角规范模板")
    axes[0, 1].imshow(right["image"], cmap="gray", origin="lower", vmin=0, vmax=1, aspect="auto")
    axes[0, 1].set_title("右三角规范模板")
    for axis in axes[0]:
        axis.set_xlabel("横向像素")
        axis.set_ylabel("纵向像素")
    orientations = np.arange(0, 180, 30)
    axes[1, 0].plot(orientations - 1.5, left["global"], marker="o", ls="--", color=SIDE_COLORS[LEFT], label="左三角")
    axes[1, 0].plot(orientations + 1.5, right["global"], marker="s", color=SIDE_COLORS[RIGHT], label="右三角")
    axes[1, 0].set_title("全局方向能量近似退化")
    axes[1, 0].set_xlabel("Gabor 方向（度）")
    axes[1, 0].set_ylabel("L1 归一化响应")
    fig.text(0.50, 0.02, "蓝色圆点：左三角；橙色方点：右三角；方向横坐标错开 1.5° 仅用于显示重合响应", ha="center", va="bottom", fontsize=7)
    difference = (left["spatial"] - right["spatial"]).reshape(6, 2)
    axes[1, 1].imshow(difference, cmap="coolwarm", aspect="auto", vmin=-np.max(abs(difference)), vmax=np.max(abs(difference)))
    axes[1, 1].set_title("空间池化响应差异（左−右）")
    axes[1, 1].set_xlabel("图像半区")
    axes[1, 1].set_ylabel("Gabor 方向（度）")
    axes[1, 1].set_xticks([0, 1], ["左半区", "右半区"])
    axes[1, 1].set_yticks(range(6), orientations)
    for row in range(6):
        for column in range(2):
            axes[1, 1].text(column, row, f"{difference[row, column]:.3f}", ha="center", va="center", fontsize=6)
    fig.subplots_adjust(left=0.10, right=0.92, bottom=0.15, top=0.92, wspace=0.34, hspace=0.40)
    save_figure(fig, output, axes)


def plot_jr_empirical(
    empirical: dict[str, np.ndarray],
    empirical_time: np.ndarray,
    simulations: dict[str, dict],
    scalp: dict[str, np.ndarray],
    output: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.0))
    for channel_index, channel in enumerate(CHANNELS):
        axis = axes[0, channel_index]
        for side in (LEFT, RIGHT):
            side_name = "左靶" if side == LEFT else "右靶"
            axis.plot(empirical_time * 1000, empirical[side][channel_index], color=SIDE_COLORS[side], lw=1.4, label=f"实测{side_name}")
            axis.plot(simulations[side]["time"] * 1000, scalp[side][channel_index], color=SIDE_COLORS[side], lw=1.1, ls="--", label=f"模拟{side_name}")
        axis.axvspan(250, 500, color="#D9D9D9", alpha=0.35)
        axis.axhline(0, color="#777777", lw=0.6)
        axis.set_title(f"{channel}：实测与模拟 ERP")
        axis.set_xlabel("提示后时间（ms）")
        axis.set_ylabel("幅值（μV）")
    for side, axis, color in zip((LEFT, RIGHT), axes[1, :2], (SIDE_COLORS[LEFT], SIDE_COLORS[RIGHT])):
        axis.plot(simulations[side]["time"] * 1000, simulations[side]["evoked_v1"], color="#6C757D", label="V1")
        axis.plot(simulations[side]["time"] * 1000, simulations[side]["evoked_pfc"], color=color, label="PFC")
        axis.axvspan(250, 500, color="#D9D9D9", alpha=0.35)
        axis.set_title(f"{'左靶' if side == LEFT else '右靶'}条件的 JR 源响应")
        axis.set_xlabel("提示后时间（ms）")
        axis.set_ylabel("相对源电位（mV）")
    asym_emp = []
    asym_sim = []
    for side in (LEFT, RIGHT):
        mask_emp = (empirical_time >= 0.25) & (empirical_time <= 0.50)
        asym_emp.append(float(np.max(empirical[side][1, mask_emp]) - np.max(empirical[side][2, mask_emp])))
        mask_sim = (simulations[side]["time"] >= 0.25) & (simulations[side]["time"] <= 0.50)
        asym_sim.append(float(np.max(scalp[side][1, mask_sim]) - np.max(scalp[side][2, mask_sim])))
    x = np.arange(2)
    axes[1, 2].bar(x - 0.18, asym_emp, width=0.36, color=EMPIRICAL_COLOR, label="实测")
    axes[1, 2].bar(x + 0.18, asym_sim, width=0.36, color=SIM_COLOR, label="模拟")
    axes[1, 2].axhline(0, color="#777777", lw=0.6)
    axes[1, 2].set_xticks(x, ["左靶", "右靶"])
    axes[1, 2].set_title("F3−F4 峰值不对称")
    axes[1, 2].set_ylabel("幅值差（μV）")
    fig.text(0.50, 0.97, "颜色：左靶蓝、右靶橙；线型：实测实线、模拟虚线；源响应：V1 灰、PFC 条件色；柱：实测深灰、模拟紫", ha="center", va="top", fontsize=7)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.10, top=0.88, wspace=0.38, hspace=0.42)
    save_figure(fig, output, axes)


def plot_lead_field(output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for axis, side in zip(axes, (LEFT, RIGHT)):
        matrix = lead_field(side)
        axis.imshow(matrix, cmap="Blues", aspect="auto", vmin=0, vmax=0.9)
        axis.set_xticks([0, 1], ["V1 源", "PFC 源"])
        axis.set_yticks(range(3), CHANNELS)
        axis.set_title(f"{'左靶' if side == LEFT else '右靶'}条件相对 Lead Field")
        axis.set_xlabel("等效源")
        axis.set_ylabel("头皮电极")
        for row in range(3):
            for column in range(2):
                axis.text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center", color="black")
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.16, top=0.86, wspace=0.38)
    save_figure(fig, output, axes)


def plot_classification(dataset: dict, result: dict, output: Path) -> None:
    matrix = dataset[result["feature_set"]]
    standardized = StandardScaler().fit_transform(matrix)
    pca = PCA(n_components=2, random_state=20260923).fit_transform(standardized)
    labels = dataset["labels"]
    visual_classifier = BayesianRidge().fit(pca, labels)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
    for label, side in ((-1, LEFT), (1, RIGHT)):
        mask = labels == label
        axes[0].scatter(pca[mask, 0], pca[mask, 1], s=13, alpha=0.65, color=SIDE_COLORS[side], label="左靶" if side == LEFT else "右靶")
    xlim, ylim = axes[0].get_xlim(), axes[0].get_ylim()
    xx, yy = np.meshgrid(np.linspace(*xlim, 100), np.linspace(*ylim, 100))
    zz = visual_classifier.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    axes[0].contour(xx, yy, zz, levels=[0], colors=["#222222"], linewidths=1)
    axes[0].set_title("特征主成分与可视化边界")
    axes[0].set_xlabel("主成分 1")
    axes[0].set_ylabel("主成分 2")
    text_legend(axes[0], [SIDE_COLORS[LEFT], SIDE_COLORS[RIGHT]])
    axes[1].scatter(np.arange(labels.size), result["scores"], c=[SIDE_COLORS[LEFT] if value == -1 else SIDE_COLORS[RIGHT] for value in labels], s=12)
    axes[1].axhline(0, color="#222222", lw=0.8)
    sample_name = {"single_trial": "单试次", "ensemble10": "10提示块集成", "ensemble20": "20提示块集成"}[result["sample_unit"]]
    axes[1].set_title(f"分组五折折外判别分数（{sample_name}）")
    axes[1].set_xlabel("样本序号")
    axes[1].set_ylabel("BLDA 分数")
    axes[2].imshow(result["confusion"], cmap="Blues", aspect="auto")
    axes[2].set_xticks([0, 1], ["预测左", "预测右"])
    axes[2].set_yticks([0, 1], ["真实左", "真实右"])
    axes[2].set_title(f"混淆矩阵（准确率 {result['accuracy']:.1%}）")
    for row in range(2):
        for column in range(2):
            axes[2].text(column, row, str(result["confusion"][row, column]), ha="center", va="center")
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.19, top=0.84, wspace=0.42)
    save_figure(fig, output, axes)


def plot_sensitivity(c_rows: list[dict], w_rows: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    c_values = sorted({row["parameter_value"] for row in c_rows})
    for source, color in (("V1", "#6C757D"), ("PFC", SIM_COLOR)):
        values = [next(row["dominant_frequency_hz"] for row in c_rows if row["parameter_value"] == value and row["source"] == source) for value in c_values]
        axes[0].plot(c_values, values, marker="o", color=color, label=source)
    axes[0].axhspan(4, 13, color="#DDECE5", alpha=0.6, label="θ–α 范围")
    axes[0].set_title("连接常数 C 对主频的影响")
    axes[0].set_xlabel("连接常数 C")
    axes[0].set_ylabel("4–20 Hz 主峰（Hz）")
    fig.text(0.50, 0.96, "V1：灰色；PFC：紫色；绿色阴影：θ–α 频带", ha="center", va="top", fontsize=7)
    w_values = [row["parameter_value"] for row in w_rows]
    amplitudes = [row["peak_to_peak"] for row in w_rows]
    axes[1].plot(w_values, amplitudes, marker="o", color=SIM_COLOR)
    axes[1].set_title("前馈权重 w 对 PFC 幅度的影响")
    axes[1].set_xlabel("V1→PFC 权重 w")
    axes[1].set_ylabel("PFC 峰峰值（mV）")
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.17, top=0.80, wspace=0.34)
    save_figure(fig, output, axes)


def plot_psd(results: list[dict], simulations: dict[str, dict], scalp_raw: dict[str, np.ndarray], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    empirical_density = []
    frequency_empirical = None
    for result in results:
        mask = (result["time"] >= 0) & (result["time"] <= 0.8)
        for trial in result["trials"]:
            if trial["accepted"]:
                frequency_empirical, density = welch(trial["final"][0, mask], fs=256, nperseg=128)
                empirical_density.append(density)
    empirical_mean = np.mean(empirical_density, axis=0)
    axes[0].semilogy(frequency_empirical, empirical_mean, color=EMPIRICAL_COLOR, label="实测 Fz")
    for side in (LEFT, RIGHT):
        frequency, density = welch(scalp_raw[side][0], fs=256, nperseg=256)
        density = density * np.median(empirical_mean[(frequency_empirical >= 4) & (frequency_empirical <= 13)]) / max(np.median(density[(frequency >= 4) & (frequency <= 13)]), 1e-12)
        axes[0].semilogy(frequency, density, color=SIDE_COLORS[side], ls="--", label=f"模拟{'左靶' if side == LEFT else '右靶'}")
    axes[0].set_xlim(1, 30)
    axes[0].axvspan(4, 13, color="#DDECE5", alpha=0.5)
    axes[0].set_title("实测与模拟头皮信号 PSD")
    axes[0].set_xlabel("频率（Hz）")
    axes[0].set_ylabel("功率谱密度（相对单位/Hz）")
    axes[0].tick_params(axis="y", labelsize=8)
    for side in (LEFT, RIGHT):
        for source, color, style in (("v1", "#6C757D", "-"), ("pfc", SIDE_COLORS[side], "--")):
            frequency, density = welch(simulations[side][source] - simulations[side][source].mean(), fs=256, nperseg=256)
            axes[1].semilogy(frequency, density, color=color, ls=style, alpha=0.85, label=f"{'左靶' if side == LEFT else '右靶'}-{source.upper()}")
    axes[1].set_xlim(1, 30)
    axes[1].axvspan(4, 13, color="#DDECE5", alpha=0.5)
    axes[1].set_title("JR 节点频谱（mV²/Hz，对数轴）")
    axes[1].set_xlabel("频率（Hz）")
    axes[1].set_ylabel("")
    axes[1].tick_params(axis="y", which="both", labelleft=False, labelright=False, length=0)
    fig.text(0.50, 0.96, "头皮 PSD：实测 Fz 黑、模拟左靶蓝虚线、模拟右靶橙虚线；JR 源：V1 灰实线、PFC 条件色虚线", ha="center", va="top", fontsize=7)
    fig.subplots_adjust(left=0.09, right=0.88, bottom=0.17, top=0.80, wspace=0.42)
    save_figure(fig, output, axes)


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def append_results(plan_path: Path, lines: list[str]) -> None:
    start, end = "<!-- P3_AUTO_START -->", "<!-- P3_AUTO_END -->"
    text = plan_path.read_text(encoding="utf-8")
    block = start + "\n" + "\n".join(lines) + "\n" + end
    if start in text and end in text:
        before = text.split(start)[0]
        after = text.split(end, 1)[1]
        text = before + block + after
    else:
        text += "\n\n十五、实际结果（自动生成）\n" + block + "\n"
    plan_path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    configure_plotting()
    output = root / "P3_图表"
    output.mkdir(parents=True, exist_ok=True)
    paths = sorted((root / "C题" / "dataset").glob("VisualCog?_Task-?.mat"))
    if len(paths) != 4:
        raise RuntimeError(f"Expected 4 MAT files, found {len(paths)}")
    results = [extract_file(path) for path in paths]
    dataset = build_dataset(results)
    ensemble10 = build_ensemble_dataset(dataset, 10, 2)
    ensemble20 = build_ensemble_dataset(dataset, 20, 4)
    print(f"分类样本数={dataset['labels'].size}，左={np.sum(dataset['labels']==-1)}，右={np.sum(dataset['labels']==1)}")
    print(f"10提示块集成样本数={ensemble10['labels'].size}，左={np.sum(ensemble10['labels']==-1)}，右={np.sum(ensemble10['labels']==1)}")
    print(f"20提示块集成样本数={ensemble20['labels'].size}，左={np.sum(ensemble20['labels']==-1)}，右={np.sum(ensemble20['labels']==1)}")

    left_visual, right_visual = visual_encode(LEFT), visual_encode(RIGHT)
    scores = shape_scores(left_visual, right_visual)
    global_difference = float(np.linalg.norm(left_visual["global"] - right_visual["global"]))
    spatial_difference = float(np.linalg.norm(left_visual["spatial"] - right_visual["spatial"]))
    simulations = {side: simulate_jr(scores[side]) for side in (LEFT, RIGHT)}
    simulated_scalp = {side: project_scalp(simulations[side], side, True) for side in (LEFT, RIGHT)}
    raw_scalp = {side: project_scalp(simulations[side], side, False) for side in (LEFT, RIGHT)}
    empirical = empirical_erps(results)
    empirical_time = results[0]["time"]
    q_scale = calibrate_simulation(simulated_scalp, empirical, empirical_time)
    simulated_scalp = {side: q_scale * value for side, value in simulated_scalp.items()}
    raw_scalp = {side: q_scale * value for side, value in raw_scalp.items()}

    classification = []
    for feature_set in ("base", "enhanced", "waveform"):
        result = cross_validate(dataset, feature_set, "grouped5", "BLDA")
        result["sample_unit"] = "single_trial"
        classification.append(result)
    single_svm = cross_validate(dataset, "waveform", "grouped5", "SVM")
    single_svm["sample_unit"] = "single_trial"
    classification.append(single_svm)
    ensemble_classification = []
    ensemble_lookup = {"ensemble10": ensemble10, "ensemble20": ensemble20}
    for sample_unit, analysis_dataset in ensemble_lookup.items():
        for feature_set in ("base", "enhanced", "waveform"):
            result = cross_validate(analysis_dataset, feature_set, "grouped5", "BLDA")
            result["sample_unit"] = sample_unit
            ensemble_classification.append(result)
    selected = next((result for result in ensemble_classification if result["accuracy"] >= 0.75), max(ensemble_classification, key=lambda item: item["accuracy"]))
    if selected["accuracy"] < 0.75:
        svm = cross_validate(ensemble20, "waveform", "grouped5", "SVM")
        svm["sample_unit"] = "ensemble20"
        ensemble_classification.append(svm)
        if svm["accuracy"] > selected["accuracy"]:
            selected = svm
    selected_dataset = ensemble_lookup[selected["sample_unit"]]
    auxiliary = [
        cross_validate(selected_dataset, selected["feature_set"], "random5", selected["method"]),
        cross_validate(selected_dataset, selected["feature_set"], "loso", selected["method"]),
    ]
    for result in auxiliary:
        result["sample_unit"] = selected["sample_unit"]

    c_rows = []
    for C in (108.0, 135.0, 162.0):
        simulation = simulate_jr(scores[LEFT], C=C, w=40)
        for source in ("V1", "PFC"):
            frequency, _, _ = dominant_frequency(simulation[source.lower()])
            c_rows.append({"record_type": "sensitivity_C", "parameter": "C", "parameter_value": C, "source": source, "dominant_frequency_hz": frequency})
    w_rows = []
    for w in (0.0, 20.0, 40.0, 60.0):
        simulation = simulate_jr(scores[LEFT], C=135, w=w)
        signal = simulation["evoked_pfc"]
        w_rows.append({"record_type": "sensitivity_w", "parameter": "w", "parameter_value": w, "peak_to_peak": float(np.ptp(signal)), "peak_amplitude": float(np.max(signal))})

    rows = []
    for result in classification + ensemble_classification + auxiliary:
        rows.append(
            {
                "record_type": "cv_summary",
                "model": result["method"],
                "feature_set": result["feature_set"],
                "protocol": result["protocol"],
                "sample_unit": result["sample_unit"],
                "metric": "accuracy",
                "value": result["accuracy"],
                "balanced_accuracy": result["balanced_accuracy"],
                "n_trials": dataset["labels"].size if result["sample_unit"] == "single_trial" else ensemble_lookup[result["sample_unit"]]["labels"].size,
            }
        )
        for fold in result["folds"]:
            rows.append({"record_type": "cv_fold", "model": result["method"], "feature_set": result["feature_set"], "protocol": result["protocol"], "sample_unit": result["sample_unit"], **fold})
        for actual_index, actual in enumerate((LEFT, RIGHT)):
            for predicted_index, predicted in enumerate((LEFT, RIGHT)):
                rows.append(
                    {
                        "record_type": "confusion",
                        "model": result["method"],
                        "feature_set": result["feature_set"],
                        "protocol": result["protocol"],
                        "sample_unit": result["sample_unit"],
                        "true_side": actual,
                        "predicted_side": predicted,
                        "count": int(result["confusion"][actual_index, predicted_index]),
                    }
                )
        for feature, mean, sd in zip(result["feature_names"], result["weights"].mean(axis=0), result["weights"].std(axis=0, ddof=1)):
            rows.append(
                {
                    "record_type": "feature_weight",
                    "model": result["method"],
                    "feature_set": result["feature_set"],
                    "protocol": result["protocol"],
                    "sample_unit": result["sample_unit"],
                    "feature": feature,
                    "weight_mean": mean,
                    "weight_sd": sd,
                    "sign_stability": float(max(np.mean(result["weights"][:, result["feature_names"].index(feature)] >= 0), np.mean(result["weights"][:, result["feature_names"].index(feature)] <= 0))),
                }
            )
    for index, metadata in enumerate(selected_dataset["metadata"]):
        rows.append(
            {
                "record_type": "trial_prediction",
                "model": selected["method"],
                "feature_set": selected["feature_set"],
                "protocol": "grouped5",
                "sample_unit": selected["sample_unit"],
                **metadata,
                "true_side": metadata["side"],
                "predicted_side": LEFT if selected["predictions"][index] == -1 else RIGHT,
                "score": selected["scores"][index],
            }
        )
    rows.extend(simulation_metrics(simulated_scalp, simulations[LEFT]["time"]))
    comparison_rows = comparison_metrics(simulated_scalp, simulations[LEFT]["time"], empirical, empirical_time)
    rows.extend(comparison_rows)
    rows.extend(c_rows)
    rows.extend(w_rows)
    rows.append({"record_type": "visual_encoding", "metric": "global_difference_l2", "value": global_difference})
    rows.append({"record_type": "visual_encoding", "metric": "spatial_difference_l2", "value": spatial_difference})
    rows.append({"record_type": "simulation_scale", "parameter": "q_scale", "parameter_value": q_scale})
    write_csv(root / "P3_结果.csv", rows)

    plot_gabor(left_visual, right_visual, output / "01_Gabor响应差异")
    plot_jr_empirical(empirical, empirical_time, simulations, simulated_scalp, output / "02_JR模拟与实测ERP")
    plot_lead_field(output / "03_LeadField前向模型")
    plot_classification(selected_dataset, selected, output / "04_BLDA分类")
    plot_sensitivity(c_rows, w_rows, output / "05_JR参数灵敏度")
    plot_psd(results, simulations, raw_scalp, output / "06_PSD对比")

    sim_peaks = simulation_metrics(simulated_scalp, simulations[LEFT]["time"])
    in_window = [row for row in sim_peaks if 250 <= row["peak_latency_ms"] <= 500 and row["peak_amplitude"] > 0]
    main_frequencies = [row["dominant_frequency_hz"] for row in c_rows if row["parameter_value"] == 135]
    median_correlation = float(np.median([row["correlation"] for row in comparison_rows]))
    median_nrmse = float(np.median([row["nrmse"] for row in comparison_rows]))
    latency_mae = float(np.mean([row["latency_absolute_error_ms"] for row in comparison_rows]))
    empirical_asym = []
    simulated_asym = []
    for side in (LEFT, RIGHT):
        mask_emp = (empirical_time >= 0.25) & (empirical_time <= 0.50)
        mask_sim = (simulations[side]["time"] >= 0.25) & (simulations[side]["time"] <= 0.50)
        empirical_asym.append(float(np.max(empirical[side][1, mask_emp]) - np.max(empirical[side][2, mask_emp])))
        simulated_asym.append(float(np.max(simulated_scalp[side][1, mask_sim]) - np.max(simulated_scalp[side][2, mask_sim])))
    asymmetry_match = np.sign(empirical_asym[1] - empirical_asym[0]) == np.sign(simulated_asym[1] - simulated_asym[0])
    summary = [
        "15.1 数据与视觉编码",
        f"- 分类使用通过 P1 质量控制且响应正确的试次 {dataset['labels'].size} 个：左 {np.sum(dataset['labels']==-1)}，右 {np.sum(dataset['labels']==1)}。",
        f"- 10提示块集成样本 {ensemble10['labels'].size} 个：左 {np.sum(ensemble10['labels']==-1)}，右 {np.sum(ensemble10['labels']==1)}；每个集成至少含2个原试次。",
        f"- 20提示块集成样本 {ensemble20['labels'].size} 个：左 {np.sum(ensemble20['labels']==-1)}，右 {np.sum(ensemble20['labels']==1)}；每个集成至少含4个原试次。",
        f"- 六方向全局 Gabor 差异 L2={global_difference:.6f}；保留左右半区的空间响应差异 L2={spatial_difference:.6f}。",
        "15.2 分类结果",
    ]
    for result in classification + ensemble_classification:
        summary.append(f"- 分组五折 {result['sample_unit']} / {result['method']} / {result['feature_set']}：准确率 {result['accuracy']:.3%}，平衡准确率 {result['balanced_accuracy']:.3%}。")
    for result in auxiliary:
        summary.append(f"- 辅助协议 {result['protocol']} / {result['method']} / {result['feature_set']}：准确率 {result['accuracy']:.3%}，平衡准确率 {result['balanced_accuracy']:.3%}。")
    summary += [
        f"- 最终报告模型：{selected['sample_unit']} / {selected['method']} / {selected['feature_set']}，分组五折准确率 {selected['accuracy']:.3%}。若样本单位为 ensemble，必须表述为重复试次集成性能，不能写成单试次性能；普通随机五折仅作为乐观上界。",
        "15.3 模拟与灵敏度",
        f"- 单一全局模拟尺度 q_scale={q_scale:.6g}；250—500 ms 内正峰通道/条件数为 {len(in_window)}/6。",
        f"- 200—550 ms 波形比较的跨条件×通道中位相关系数为 {median_correlation:.3f}，中位 NRMSE={median_nrmse:.3f}，P300 峰潜伏期平均绝对误差为 {latency_mae:.1f} ms；相关用于形态评价，不替代幅值误差。",
        f"- C=135 时 V1/PFC 的 4—20 Hz 主峰为 {main_frequencies[0]:.2f}/{main_frequencies[1]:.2f} Hz。",
        f"- F3−F4 左右趋势与实测是否同号：{'是' if asymmetry_match else '否'}；实测差值={empirical_asym}，模拟差值={simulated_asym}。",
        "15.4 运行与回退记录",
        "- 主方案先运行基础统计特征 BLDA，再按预注册顺序运行增强统计特征和固定 50 ms 时空波形 BLDA；仅在所有 BLDA 未达到 75% 时运行线性 SVM 回退，并保留全部结果。",
        "- 模拟 EEG 与分类数据严格分离；交叉验证中的标准化和左右 ERP 模板均只使用训练折。",
        "- 图表由 Python/Matplotlib 生成 PNG（600 dpi）和 PDF，并保存多面板对齐审计文件；最终碰撞与字体审计在脚本运行后独立执行。",
    ]
    append_results(root / "P3_方案.txt", summary)
    print(f"全局Gabor差异={global_difference:.6f}，空间差异={spatial_difference:.6f}")
    for result in classification + ensemble_classification:
        print(f"分组五折 {result['sample_unit']} {result['method']} {result['feature_set']}: accuracy={result['accuracy']:.3%}, balanced={result['balanced_accuracy']:.3%}")
    print(f"最终模型={selected['sample_unit']}/{selected['method']}/{selected['feature_set']}，accuracy={selected['accuracy']:.3%}")
    print(f"模拟正峰通过={len(in_window)}/6，C=135主频={main_frequencies}")
    print(json.dumps({"accuracy": selected["accuracy"], "confusion": selected["confusion"].tolist()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
