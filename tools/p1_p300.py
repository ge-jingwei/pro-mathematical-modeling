from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pywt
from scipy.io import loadmat
from scipy.optimize import curve_fit
from scipy.signal import butter, fftconvolve, filtfilt, iirnotch, savgol_filter
from scipy.stats import pearsonr, ttest_rel

from audit_panel_alignment import require_matplotlib_panel_alignment


CHANNELS = ["Fz", "F3", "F4"]
SIDE_COLORS = {"left": "#3B82B8", "right": "#E58B3A"}
ACC_COLORS = {"correct": "#3C7D74", "error": "#A15C86"}
ALPHAS = [0.00, 0.08, 0.50, 0.90, 1.10]
FREQUENCIES = [1, 2, 4, 8, 12, 16]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def labels_from_cell(value: np.ndarray) -> list[str]:
    labels = []
    for item in np.asarray(value).ravel():
        while isinstance(item, np.ndarray) and item.size == 1:
            item = item.item()
        labels.append(str(item))
    return labels


def run_starts(values: np.ndarray, accepted: set[int], legal_codes: set[int] | None = None) -> list[dict]:
    quantized = np.zeros(values.size, dtype=int)
    for code in legal_codes or (accepted | {0}):
        quantized[np.abs(values - code) <= 0.25] = code
    starts = np.r_[True, quantized[1:] != quantized[:-1]]
    indices = np.flatnonzero(starts)
    ends = np.r_[indices[1:], values.size]
    return [
        {"sample": int(i), "code": int(quantized[i]), "length": int(j - i)}
        for i, j in zip(indices, ends)
        if quantized[i] in accepted
    ]


def pair_trials(cues: list[dict], responses: list[dict]) -> list[dict]:
    trials = []
    cursor = 0
    for i, cue in enumerate(cues):
        next_sample = cues[i + 1]["sample"] if i + 1 < len(cues) else np.iinfo(np.int64).max
        while cursor < len(responses) and responses[cursor]["sample"] < cue["sample"]:
            cursor += 1
        response = responses[cursor] if cursor < len(responses) and responses[cursor]["sample"] < next_sample else None
        if response is not None:
            cursor += 1
        cue_side = "left" if cue["code"] < 0 else "right"
        response_side = "missing" if response is None else ("left" if response["code"] < 0 else "right")
        trials.append(
            {
                "trial": i + 1,
                "cue_sample": cue["sample"],
                "cue_side": cue_side,
                "response_side": response_side,
                "correct": response_side == cue_side if response is not None else False,
            }
        )
    return trials


def preprocess(eeg: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    notch_b, notch_a = iirnotch(50, 30, fs)
    band_b, band_a = butter(4, [0.5, 20], btype="bandpass", fs=fs)
    pre = np.vstack([filtfilt(band_b, band_a, filtfilt(notch_b, notch_a, x)) for x in eeg])
    denoised = []
    thresholds = []
    for channel, signal in zip(CHANNELS, pre):
        coeffs = pywt.wavedec(signal, "sym9", level=5, mode="symmetric")
        rebuilt = [coeffs[0]]
        layer_rows = []
        for level, (detail, alpha) in enumerate(zip(coeffs[1:], ALPHAS), 1):
            sigma = np.median(np.abs(detail - np.median(detail))) / 0.6745
            threshold = float(sigma * np.sqrt(2 * np.log(detail.size)) * alpha)
            rebuilt.append(pywt.threshold(detail, threshold, mode="soft"))
            layer_rows.append({"channel": channel, "detail": f"d{6-level}", "alpha": alpha, "sigma": float(sigma), "threshold": threshold})
        denoised.append(pywt.waverec(rebuilt, "sym9", mode="symmetric")[: signal.size])
        thresholds.extend(layer_rows)
    return pre, np.vstack(denoised), thresholds


def baseline(epoch: np.ndarray, time: np.ndarray) -> np.ndarray:
    mask = (time >= -0.2) & (time < 0)
    return epoch - epoch[:, mask].mean(axis=1, keepdims=True)


def gabor_features(signal: np.ndarray, time: np.ndarray, fs: float) -> np.ndarray:
    feature_mask = (time >= 0.2) & (time <= 0.6)
    values = []
    for frequency in FREQUENCIES:
        sigma = 3 / (2 * np.pi * frequency)
        half = min(signal.size // 2 - 1, max(8, int(np.ceil(3 * sigma * fs))))
        kernel_time = np.arange(-half, half + 1) / fs
        kernel = np.exp(-(kernel_time**2) / (2 * sigma**2)) * np.exp(2j * np.pi * frequency * kernel_time)
        kernel /= np.sqrt(np.sum(np.abs(kernel) ** 2))
        response = np.abs(fftconvolve(signal, kernel, mode="same"))[feature_mask]
        values.extend([float(response.mean()), float(np.sqrt(np.mean(response**2)))])
    return np.asarray(values)


def vector_angle(before: np.ndarray, after: np.ndarray) -> float:
    denominator = np.linalg.norm(before) * np.linalg.norm(after)
    if denominator < 1e-12:
        return float("nan")
    cosine = float(np.clip(np.dot(before, after) / denominator, -1, 1))
    return float(np.degrees(np.arccos(cosine)))


def snr_db(epochs: np.ndarray, time: np.ndarray) -> float:
    mask = (time >= 0.25) & (time <= 0.50)
    mean_erp = epochs.mean(axis=0)
    residual = epochs - mean_erp
    signal_power = float(np.mean(mean_erp[:, mask] ** 2))
    noise_power = float(np.mean(residual[:, :, mask] ** 2))
    return float(10 * np.log10(max(signal_power, 1e-12) / max(noise_power, 1e-12)))


def gamma_component(t: np.ndarray, amplitude: float, onset: float, tau: float, shape: float) -> np.ndarray:
    u = np.maximum((t - onset) / tau, 0)
    return amplitude * np.power(u, shape) * np.exp(-u)


def gamma_model(t: np.ndarray, amplitude: float, onset: float, tau: float, shape: float, baseline_value: float) -> np.ndarray:
    return gamma_component(t, amplitude, onset, tau, shape) + baseline_value


def gamma_trend_model(
    t: np.ndarray,
    peak_amplitude: float,
    onset: float,
    peak_time: float,
    shape: float,
    baseline_value: float,
    linear: float,
    quadratic: float,
) -> np.ndarray:
    tau = (peak_time - onset) / shape
    scale = shape**shape * np.exp(-shape)
    amplitude = peak_amplitude / scale
    centered = t - 0.375
    return gamma_model(t, amplitude, onset, tau, shape, baseline_value) + linear * centered + quadratic * centered**2


def predict_fit(t: np.ndarray, fit: dict) -> np.ndarray:
    return gamma_model(t, fit["A"], fit["t0_s"], fit["tau_s"], fit["k"], fit["B"]) + fit["C1"] * (t - 0.375) + fit["C2"] * (t - 0.375) ** 2


def fit_gamma(time: np.ndarray, erp: np.ndarray) -> dict:
    low, high = 0.20, 0.55
    mask = (time >= low) & (time <= high)
    t = time[mask]
    y = erp[mask]
    span = max(float(np.ptp(y)), 1.0)
    starts = [0.28, 0.33, 0.38, 0.43, 0.48]
    best = None
    bounds = (
        [0, 0.08, 0.25, 0.5, float(np.min(y) - span), -20 * span, -100 * span],
        [2 * span, 0.24, 0.50, 10, float(np.max(y) + span), 20 * span, 100 * span],
    )
    for peak_start in starts:
        try:
            params, _ = curve_fit(
                gamma_trend_model,
                t,
                y,
                p0=[span, 0.18, peak_start, 3, float(np.median(y)), 0, 0],
                bounds=bounds,
                maxfev=200000,
            )
        except (RuntimeError, ValueError, FloatingPointError):
            continue
        predicted = gamma_trend_model(t, *params)
        denominator = float(np.sum((y - y.mean()) ** 2))
        r2 = 1 - float(np.sum((y - predicted) ** 2)) / denominator if denominator > 1e-12 else float("nan")
        if np.isfinite(r2) and (best is None or r2 > best[0]):
            best = (r2, params)
    if best is None:
        return {"fit_status": "failed", "r2": float("nan")}
    r2, params = best
    peak_amplitude, onset, peak_time, shape, baseline_value, linear, quadratic = [float(x) for x in params]
    tau = (peak_time - onset) / shape
    amplitude = peak_amplitude / (shape**shape * np.exp(-shape))
    dense = np.linspace(onset, max(0.8, peak_time + 0.4), 10000)
    component = gamma_component(dense, amplitude, onset, tau, shape)
    above = dense[component >= 0.5 * peak_amplitude]
    width = float(above[-1] - above[0]) if above.size > 1 else float("nan")
    return {
        "fit_status": "ok",
        "model": "gamma_quadratic_baseline",
        "A": amplitude,
        "t0_s": onset,
        "tau_s": tau,
        "k": shape,
        "B": baseline_value,
        "C1": linear,
        "C2": quadratic,
        "peak_amplitude": peak_amplitude,
        "peak_latency_ms": peak_time * 1000,
        "fwhm_ms": width * 1000,
        "r2": float(r2),
        "fit_window_start_s": low,
        "fit_window_end_s": high,
    }


def empirical_trial_features(epoch: np.ndarray, time: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = (time >= 0.25) & (time <= 0.50)
    smooth = savgol_filter(epoch, 11, 3, axis=1)
    indices = np.argmax(smooth[:, mask], axis=1)
    window_time = time[mask]
    amplitudes = np.take_along_axis(smooth[:, mask], indices[:, None], axis=1).ravel()
    latencies = window_time[indices] * 1000
    return amplitudes, latencies


def extract_file(path: Path) -> dict:
    mat = loadmat(path, squeeze_me=False, struct_as_record=False)
    fs = float(np.asarray(mat["SampleRate"]).squeeze())
    labels = labels_from_cell(mat["DataLabel"])
    data = np.asarray(mat["data"], dtype=float)
    if data.shape[0] != len(labels) and data.shape[1] == len(labels):
        data = data.T
    subject = path.stem.replace("VisualCog", "").split("_")[0]
    task = int(path.stem.split("Task-")[1])
    cues = run_starts(data[7], {-1, 1})
    response_codes = {-1, 1} if task == 1 else {-2, 2}
    legal = response_codes | ({-1, 0, 1} if task == 2 else {0})
    responses = run_starts(data[8], response_codes, legal)
    event_trials = pair_trials(cues, responses)
    pre, denoised, thresholds = preprocess(data[:3], fs)
    start, stop = -round(0.2 * fs), round(0.8 * fs)
    time = np.arange(start, stop + 1) / fs
    trials = []
    for event in event_trials:
        cue = event["cue_sample"]
        left, right = cue + start, cue + stop + 1
        row = {**event, "boundary": left < 0 or right > data.shape[1]}
        if row["boundary"]:
            row.update(raw=None, pre=None, final=None, saturation=False, max_abs=float("nan"), peak_to_peak=float("nan"))
            trials.append(row)
            continue
        raw_epoch = baseline(data[:3, left:right], time)
        pre_epoch = baseline(pre[:, left:right], time)
        final_epoch = baseline(denoised[:, left:right], time)
        row.update(
            raw=raw_epoch,
            pre=pre_epoch,
            final=final_epoch,
            saturation=bool(np.any(np.abs(data[:3, left:right]) >= 999)),
            max_abs=float(np.max(np.abs(final_epoch))),
            peak_to_peak=float(np.max(np.ptp(final_epoch, axis=1))),
        )
        trials.append(row)
    eligible = [x for x in trials if not x["boundary"]]
    candidate_thresholds = [100.0, 150.0, 175.0, 200.0, 225.0, 250.0]
    for threshold in candidate_thresholds:
        for row in eligible:
            row[f"accepted_{int(threshold)}"] = not row["saturation"] and row["max_abs"] <= threshold and row["peak_to_peak"] <= 2 * threshold
    primary_retention = np.mean([x["accepted_100"] for x in eligible]) if eligible else 0
    selected_threshold = candidate_thresholds[-1]
    for threshold in candidate_thresholds:
        if np.mean([x[f"accepted_{int(threshold)}"] for x in eligible]) > 0.70:
            selected_threshold = threshold
            break
    for row in trials:
        row["accepted"] = False if row["boundary"] else row[f"accepted_{int(selected_threshold)}"]
        reasons = []
        if row["boundary"]:
            reasons.append("boundary")
        if row.get("saturation", False):
            reasons.append("saturation")
        if not row["boundary"] and (row["max_abs"] > selected_threshold or row["peak_to_peak"] > 2 * selected_threshold):
            reasons.append("amplitude_artifact")
        row["rejection_reason"] = "+".join(reasons) if reasons else "accepted"
    accepted = [x for x in trials if x["accepted"]]
    if not accepted:
        raise RuntimeError(f"No retained trials in {path.name}")
    pre_epochs = np.stack([x["pre"] for x in accepted])
    final_epochs = np.stack([x["final"] for x in accepted])
    gabor_angles = []
    for row in accepted:
        row["gabor_angles"] = []
        for channel in range(3):
            angle = vector_angle(gabor_features(row["pre"][channel], time, fs), gabor_features(row["final"][channel], time, fs))
            row["gabor_angles"].append(angle)
            gabor_angles.append(angle)
        amplitudes, latencies = empirical_trial_features(row["final"], time)
        row["trial_peak_amplitudes"] = amplitudes
        row["trial_peak_latencies_ms"] = latencies
    correlations = []
    for channel in range(3):
        mask = (time >= 0.2) & (time <= 0.6)
        correlations.append(float(pearsonr(pre_epochs[:, channel].mean(axis=0)[mask], final_epochs[:, channel].mean(axis=0)[mask]).statistic))
    return {
        "file": path.name,
        "subject": subject,
        "task": task,
        "fs": fs,
        "time": time,
        "trials": trials,
        "thresholds": thresholds,
        "artifact_threshold": selected_threshold,
        "primary_retention": primary_retention,
        "retention": len(accepted) / len(trials),
        "pre_snr_db": snr_db(pre_epochs, time),
        "final_snr_db": snr_db(final_epochs, time),
        "gabor_angles": np.asarray(gabor_angles),
        "waveform_correlations": correlations,
    }


def selected_epochs(result: dict, group_type: str, condition: str) -> list[dict]:
    retained = [x for x in result["trials"] if x["accepted"]]
    if group_type == "side":
        return [x for x in retained if x["cue_side"] == condition]
    if group_type == "accuracy":
        target = condition == "correct"
        return [x for x in retained if x["correct"] == target]
    if group_type == "side_correct":
        return [x for x in retained if x["correct"] and x["cue_side"] == condition]
    raise ValueError(group_type)


def group_erp(result: dict, group_type: str, condition: str, stage: str = "final") -> tuple[np.ndarray | None, int]:
    rows = selected_epochs(result, group_type, condition)
    if not rows:
        return None, 0
    return np.stack([x[stage] for x in rows]).mean(axis=0), len(rows)


def task_erp(results: list[dict], task: int, group_type: str, condition: str) -> tuple[np.ndarray | None, np.ndarray | None, int, int]:
    subject_erps = []
    total_trials = 0
    for result in results:
        if result["task"] != task:
            continue
        erp, count = group_erp(result, group_type, condition)
        if erp is not None:
            subject_erps.append(erp)
            total_trials += count
    if not subject_erps:
        return None, None, 0, total_trials
    stack = np.stack(subject_erps)
    sem = stack.std(axis=0, ddof=1) / np.sqrt(stack.shape[0]) if stack.shape[0] > 1 else np.zeros_like(stack[0])
    return stack.mean(axis=0), sem, stack.shape[0], total_trials


def panel_labels(axes: np.ndarray | list) -> None:
    for i, ax in enumerate(np.atleast_1d(axes).ravel()):
        ax.annotate(chr(97 + i), xy=(0, 1), xycoords="axes fraction", xytext=(-20, 4), textcoords="offset points", fontsize=10, fontweight="bold", ha="left", va="bottom")


def save_figure(fig: plt.Figure, base: Path, require_labels: bool) -> None:
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig,
        json_out=str(base) + ".alignment.json",
        overlay_svg=str(base) + ".alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=require_labels,
        strict=True,
    )
    fig.savefig(base.with_suffix(".png"), dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_denoise(result: dict, output: Path) -> None:
    retained = [x for x in result["trials"] if x["accepted"]]
    scores = np.asarray([x["max_abs"] for x in retained])
    representative = retained[int(np.argmin(np.abs(scores - np.median(scores))))]
    time_ms = result["time"] * 1000
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.2), sharex=True)
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.10, top=0.84, hspace=0.34)
    for channel, ax in enumerate(axes):
        ax.plot(time_ms, representative["raw"][channel], color="#999999", linewidth=0.6, label="原始")
        ax.plot(time_ms, representative["final"][channel], color="#315B7D", linewidth=0.9, label="降噪")
        ax.axvspan(250, 500, color="#E8B04A", alpha=0.14)
        ax.axvline(0, color="#444444", linestyle="--", linewidth=0.7)
        ax.set_ylabel(f"{CHANNELS[channel]}\n幅值（μV）")
        ax.grid(alpha=0.16, linewidth=0.5)
    axes[-1].set_xlabel("相对提示时间（ms）")
    angles = [x for x in representative["gabor_angles"] if np.isfinite(x)]
    fig.suptitle(
        f"被试 {result['subject']} 项目 {result['task']}：代表试次 {representative['trial']} 原始与降噪\n"
        f"灰=原始，蓝=降噪，黄色=P300窗；本试次 Gabor 角中位数={np.median(angles):.2f}°",
        fontsize=10,
    )
    panel_labels(axes)
    save_figure(fig, output, True)


def plot_task_erp(results: list[dict], task: int, group_type: str, output: Path) -> None:
    conditions = ["left", "right"] if group_type == "side" else ["correct", "error"]
    colors = SIDE_COLORS if group_type == "side" else ACC_COLORS
    labels = {"left": "左靶", "right": "右靶", "correct": "正确", "error": "错误"}
    time = next(x["time"] for x in results if x["task"] == task)
    time_ms = time * 1000
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.2), sharex=True)
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.10, top=0.82, hspace=0.34)
    counts = []
    for channel, ax in enumerate(axes):
        for condition in conditions:
            erp, sem, subjects, trials = task_erp(results, task, group_type, condition)
            counts.append((condition, subjects, trials))
            if erp is None or trials < 5:
                continue
            ax.plot(time_ms, erp[channel], color=colors[condition], linewidth=1.1, linestyle="-" if condition in {"left", "correct"} else "--")
            if subjects > 1:
                ax.fill_between(time_ms, erp[channel] - sem[channel], erp[channel] + sem[channel], color=colors[condition], alpha=0.16, linewidth=0)
        ax.axvspan(250, 500, color="#E8B04A", alpha=0.10)
        ax.axvline(0, color="#444444", linestyle="--", linewidth=0.7)
        ax.axhline(0, color="#777777", linewidth=0.5)
        ax.set_ylabel(f"{CHANNELS[channel]}\n幅值（μV）")
        ax.grid(alpha=0.16, linewidth=0.5)
    axes[-1].set_xlabel("相对提示时间（ms）")
    unique = {condition: (subjects, trials) for condition, subjects, trials in counts}
    note = "，".join(f"{labels[c]}：被试 n={unique[c][0]}，试次 n={unique[c][1]}" for c in conditions)
    encoding = "蓝实线=左靶，橙虚线=右靶" if group_type == "side" else "绿实线=正确，紫虚线=错误"
    fig.suptitle(f"项目 {task} Grand Average ERP：{labels[conditions[0]]}与{labels[conditions[1]]}\n{encoding}；阴影=被试间 SEM；{note}", fontsize=10)
    panel_labels(axes)
    save_figure(fig, output, True)


def plot_gamma(result: dict, fit_lookup: dict, output: Path) -> None:
    time = result["time"]
    time_ms = time * 1000
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.4), sharex=True)
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.10, top=0.82, hspace=0.34)
    for channel, ax in enumerate(axes):
        notes = []
        for side in ["left", "right"]:
            erp, count = group_erp(result, "side", side)
            fit = fit_lookup.get((result["file"], CHANNELS[channel], "side", side))
            if erp is None or fit is None or fit["fit_status"] != "ok":
                continue
            ax.plot(time_ms, erp[channel], color=SIDE_COLORS[side], linewidth=0.9)
            dense = np.linspace(fit["fit_window_start_s"], fit["fit_window_end_s"], 500)
            ax.plot(dense * 1000, predict_fit(dense, fit), color=SIDE_COLORS[side], linestyle="--", linewidth=1.1)
            notes.append(f"{('左' if side == 'left' else '右')}：R²={fit['r2']:.3f}, {fit['peak_latency_ms']:.0f} ms, {fit['peak_amplitude']:.2f} μV")
        ax.axvspan(250, 500, color="#E8B04A", alpha=0.10)
        ax.axvline(0, color="#444444", linestyle="--", linewidth=0.7)
        ax.axhline(0, color="#777777", linewidth=0.5)
        ax.set_ylabel(f"{CHANNELS[channel]}\n幅值（μV）")
        ax.set_title("；".join(notes), fontsize=8)
        ax.grid(alpha=0.16, linewidth=0.5)
    axes[-1].set_xlabel("相对提示时间（ms）")
    fig.suptitle(f"被试 {result['subject']} 项目 {result['task']}：P300 Gamma 拟合\n蓝/橙=左/右，实线=实测 ERP，虚线=拟合", fontsize=10)
    panel_labels(axes)
    save_figure(fig, output, True)


def plot_task_gamma(results: list[dict], task: int, output: Path) -> None:
    time = next(x["time"] for x in results if x["task"] == task)
    time_ms = time * 1000
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.4), sharex=True)
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.10, top=0.82, hspace=0.34)
    for channel, ax in enumerate(axes):
        notes = []
        for side in ["left", "right"]:
            erp, _, _, trials = task_erp(results, task, "side", side)
            if erp is None:
                continue
            fit = fit_gamma(time, erp[channel])
            ax.plot(time_ms, erp[channel], color=SIDE_COLORS[side], linewidth=0.9)
            if fit["fit_status"] == "ok":
                dense = np.linspace(fit["fit_window_start_s"], fit["fit_window_end_s"], 500)
                ax.plot(dense * 1000, predict_fit(dense, fit), color=SIDE_COLORS[side], linestyle="--", linewidth=1.1)
                notes.append(f"{('左' if side == 'left' else '右')}：R²={fit['r2']:.3f}, 峰={fit['peak_latency_ms']:.0f} ms, n={trials}")
        ax.axvspan(250, 500, color="#E8B04A", alpha=0.10)
        ax.axvline(0, color="#444444", linestyle="--", linewidth=0.7)
        ax.axhline(0, color="#777777", linewidth=0.5)
        ax.set_ylabel(f"{CHANNELS[channel]}\n幅值（μV）")
        ax.set_title("；".join(notes), fontsize=8)
        ax.grid(alpha=0.16, linewidth=0.5)
    axes[-1].set_xlabel("相对提示时间（ms）")
    fig.suptitle(f"项目 {task} 被试等权 ERP 与 Gamma 拟合\n蓝/橙=左/右，实线=实测，虚线=拟合", fontsize=10)
    panel_labels(axes)
    save_figure(fig, output, True)


def plot_asymmetry(asymmetry_rows: list[dict], output: Path) -> None:
    groups = [(1, "left"), (1, "right"), (2, "left"), (2, "right")]
    means, sems = [], []
    fig, ax = plt.subplots(figsize=(6.8, 4.3), constrained_layout=True)
    for index, (task, side) in enumerate(groups):
        rows = [x for x in asymmetry_rows if x["task"] == task and x["side"] == side and np.isfinite(x["asymmetry_index"])]
        values = np.asarray([x["asymmetry_index"] for x in rows])
        means.append(float(values.mean()) if values.size else np.nan)
        sems.append(float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0)
        for offset, row in enumerate(rows):
            ax.scatter(index + (-0.08 if offset % 2 == 0 else 0.08), row["asymmetry_index"], color="#333333", s=24, zorder=3)
    colors = [SIDE_COLORS[side] for _, side in groups]
    ax.bar(np.arange(4), means, yerr=sems, color=colors, alpha=0.72, capsize=4, edgecolor="white")
    ax.axhline(0, color="#555555", linewidth=0.7)
    ax.set_xticks(np.arange(4), ["项目1\n左靶", "项目1\n右靶", "项目2\n左靶", "项目2\n右靶"])
    ax.set_ylabel("F3—F4 不对称指数")
    ax.set_title("P300 峰值 F3—F4 不对称\n柱=被试均值，误差线=被试 SEM，点=A/B")
    ax.grid(axis="y", alpha=0.16, linewidth=0.5)
    save_figure(fig, output, False)


def plot_project_comparison(fit_rows: list[dict], tests: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), constrained_layout=True)
    metrics = [("peak_amplitude", "P300 峰值振幅（μV）"), ("peak_latency_ms", "P300 峰值潜伏期（ms）")]
    groups = [(1, "left"), (1, "right"), (2, "left"), (2, "right")]
    for ax, (metric, ylabel) in zip(axes, metrics):
        means, sems = [], []
        for task, side in groups:
            values = np.asarray([x[metric] for x in fit_rows if x["group_type"] == "side" and x["task"] == task and x["condition"] == side and x["fit_status"] == "ok"], dtype=float)
            means.append(values.mean())
            sems.append(values.std(ddof=1) / np.sqrt(values.size) if values.size > 1 else 0)
        ax.bar(np.arange(4), means, yerr=sems, color=[SIDE_COLORS[x[1]] for x in groups], alpha=0.72, capsize=4, edgecolor="white")
        for index, (task, side) in enumerate(groups):
            values = [x[metric] for x in fit_rows if x["group_type"] == "side" and x["task"] == task and x["condition"] == side and x["fit_status"] == "ok"]
            ax.scatter(np.full(len(values), index), values, color="#333333", s=14, alpha=0.65, zorder=3)
        test_metric = "amplitude" if metric == "peak_amplitude" else "latency"
        p1 = next((x["p_raw"] for x in tests if x["analysis"] == "file_channel" and x["scope"] == "task1" and x["metric"] == test_metric), np.nan)
        p2 = next((x["p_raw"] for x in tests if x["analysis"] == "file_channel" and x["scope"] == "task2" and x["metric"] == test_metric), np.nan)
        ax.set_title(f"项目1 左右 p={p1:.3g}；项目2 左右 p={p2:.3g}", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_xticks(np.arange(4), ["任务1\n左", "任务1\n右", "任务2\n左", "任务2\n右"])
        ax.grid(axis="y", alpha=0.16, linewidth=0.5)
    fig.suptitle("项目与提示侧的 P300 参数对比\n柱=文件×通道均值±SEM，点=文件×通道", fontsize=10)
    panel_labels(axes)
    save_figure(fig, output, True)


def fit_all(results: list[dict]) -> list[dict]:
    rows = []
    for result in results:
        for group_type, conditions in [("side", ["left", "right"]), ("accuracy", ["correct", "error"]), ("side_correct", ["left", "right"])]:
            for condition in conditions:
                erp, count = group_erp(result, group_type, condition)
                if erp is None or count < 5:
                    for channel in CHANNELS:
                        rows.append({"file": result["file"], "subject": result["subject"], "task": result["task"], "channel": channel, "group_type": group_type, "condition": condition, "n_trials": count, "fit_status": "insufficient_trials", "r2": float("nan")})
                    continue
                for channel_index, channel in enumerate(CHANNELS):
                    fit = fit_gamma(result["time"], erp[channel_index])
                    rows.append({"file": result["file"], "subject": result["subject"], "task": result["task"], "channel": channel, "group_type": group_type, "condition": condition, "n_trials": count, **fit})
    return rows


def holm_adjust(p_values: list[float]) -> list[float]:
    result = [float("nan")] * len(p_values)
    valid = [(i, p) for i, p in enumerate(p_values) if np.isfinite(p)]
    ordered = sorted(valid, key=lambda x: x[1])
    running = 0.0
    for rank, (index, p) in enumerate(ordered):
        adjusted = min(1.0, (len(ordered) - rank) * p)
        running = max(running, adjusted)
        result[index] = running
    return result


def paired_test(left: np.ndarray, right: np.ndarray) -> dict:
    if left.size < 2 or right.size != left.size:
        return {"n_pairs": int(min(left.size, right.size)), "t": float("nan"), "df": float("nan"), "p_raw": float("nan"), "mean_difference": float("nan"), "cohen_dz": float("nan")}
    test = ttest_rel(left, right, nan_policy="omit")
    difference = left - right
    sd = float(np.std(difference, ddof=1))
    return {"n_pairs": int(difference.size), "t": float(test.statistic), "df": int(difference.size - 1), "p_raw": float(test.pvalue), "mean_difference": float(np.mean(difference)), "cohen_dz": float(np.mean(difference) / sd) if sd > 1e-12 else float("nan")}


def make_tests(fit_rows: list[dict], results: list[dict]) -> list[dict]:
    tests = []
    for group_type, analysis_name in [("side", "file_channel"), ("side_correct", "file_channel_correct")]:
        for scope, tasks in [("task1", {1}), ("task2", {2}), ("all", {1, 2})]:
            for metric, field in [("amplitude", "peak_amplitude"), ("latency", "peak_latency_ms")]:
                left_map = {(x["file"], x["channel"]): x[field] for x in fit_rows if x["fit_status"] == "ok" and x["group_type"] == group_type and x["condition"] == "left" and x["task"] in tasks}
                right_map = {(x["file"], x["channel"]): x[field] for x in fit_rows if x["fit_status"] == "ok" and x["group_type"] == group_type and x["condition"] == "right" and x["task"] in tasks}
                keys = sorted(set(left_map) & set(right_map))
                test = paired_test(np.asarray([left_map[k] for k in keys]), np.asarray([right_map[k] for k in keys]))
                tests.append({"analysis": analysis_name, "scope": scope, "metric": metric, **test})
    for scope, tasks in [("task1", {1}), ("task2", {2}), ("all", {1, 2})]:
        for metric, field in [("amplitude", "peak_amplitude"), ("latency", "peak_latency_ms")]:
            left, right = [], []
            for result in results:
                if result["task"] not in tasks:
                    continue
                left_values = [x[field] for x in fit_rows if x["fit_status"] == "ok" and x["group_type"] == "side" and x["condition"] == "left" and x["file"] == result["file"]]
                right_values = [x[field] for x in fit_rows if x["fit_status"] == "ok" and x["group_type"] == "side" and x["condition"] == "right" and x["file"] == result["file"]]
                if left_values and right_values:
                    left.append(np.mean(left_values))
                    right.append(np.mean(right_values))
            tests.append({"analysis": "file_mean_sensitivity", "scope": scope, "metric": metric, **paired_test(np.asarray(left), np.asarray(right))})
    for result in results:
        retained = [x for x in result["trials"] if x["accepted"]]
        left_rows = sorted([x for x in retained if x["cue_side"] == "left"], key=lambda x: x["cue_sample"])
        right_rows = sorted([x for x in retained if x["cue_side"] == "right"], key=lambda x: x["cue_sample"])
        pairs = []
        available_left, available_right = left_rows[:], right_rows[:]
        while available_left and available_right:
            distances = np.abs(np.subtract.outer([x["cue_sample"] for x in available_left], [x["cue_sample"] for x in available_right]))
            i, j = np.unravel_index(np.argmin(distances), distances.shape)
            pairs.append((available_left.pop(i), available_right.pop(j)))
        result["trial_pairs"] = pairs
    for scope, tasks in [("task1", {1}), ("task2", {2}), ("all", {1, 2})]:
        for metric in ["amplitude", "latency"]:
            left, right = [], []
            for result in results:
                if result["task"] not in tasks:
                    continue
                for lrow, rrow in result["trial_pairs"]:
                    key = "trial_peak_amplitudes" if metric == "amplitude" else "trial_peak_latencies_ms"
                    left.extend(lrow[key])
                    right.extend(rrow[key])
            tests.append({"analysis": "trial_time_matched_exploratory", "scope": scope, "metric": metric, **paired_test(np.asarray(left), np.asarray(right))})
    for analysis in sorted(set(x["analysis"] for x in tests)):
        for scope in ["task1", "task2", "all"]:
            indices = [i for i, x in enumerate(tests) if x["analysis"] == analysis and x["scope"] == scope]
            adjusted = holm_adjust([tests[i]["p_raw"] for i in indices])
            for i, value in zip(indices, adjusted):
                tests[i]["p_holm"] = value
    return tests


def make_asymmetry(fit_rows: list[dict]) -> list[dict]:
    rows = []
    files = sorted(set(x["file"] for x in fit_rows))
    for file in files:
        for side in ["left", "right"]:
            lookup = {x["channel"]: x for x in fit_rows if x["file"] == file and x["group_type"] == "side" and x["condition"] == side and x["fit_status"] == "ok"}
            if "F3" not in lookup or "F4" not in lookup:
                continue
            a3, a4 = lookup["F3"]["peak_amplitude"], lookup["F4"]["peak_amplitude"]
            denominator = a3 + a4
            base = lookup["F3"]
            rows.append({"file": file, "subject": base["subject"], "task": base["task"], "side": side, "amplitude_f3": a3, "amplitude_f4": a4, "asymmetry_index": (a3 - a4) / denominator if abs(denominator) >= 1e-8 else float("nan")})
    return rows


def make_parameter_summary(fit_rows: list[dict]) -> list[dict]:
    rows = []
    for task in [1, 2]:
        for side in ["left", "right"]:
            for channel in CHANNELS:
                selected = [
                    x
                    for x in fit_rows
                    if x["fit_status"] == "ok"
                    and x["group_type"] == "side"
                    and x["task"] == task
                    and x["condition"] == side
                    and x["channel"] == channel
                ]
                row = {
                    "record_type": "parameter_summary",
                    "task": task,
                    "condition": side,
                    "channel": channel,
                    "n_subjects": len(selected),
                }
                for field in ["peak_amplitude", "peak_latency_ms", "fwhm_ms", "r2"]:
                    values = np.asarray([x[field] for x in selected], dtype=float)
                    row[f"{field}_mean"] = float(np.mean(values))
                    row[f"{field}_sem"] = float(np.std(values, ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0
                rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plan_summary(results: list[dict], fit_rows: list[dict], tests: list[dict]) -> str:
    lines = ["", "十八、实际结果（自动生成）", "<!-- P1_AUTO_START -->"]
    for index, result in enumerate(results, 1):
        angles = result["gabor_angles"][np.isfinite(result["gabor_angles"])]
        side_fits = [x for x in fit_rows if x["file"] == result["file"] and x["group_type"] == "side" and x["fit_status"] == "ok"]
        lines.extend(
            [
                f"18.{index} {result['file']}",
                f"- 伪影阈值：±{result['artifact_threshold']:.0f} μV；主规则保留率 {result['primary_retention']:.1%}；最终保留率 {result['retention']:.1%}。",
                f"- SNR：小波前 {result['pre_snr_db']:.3f} dB，小波后 {result['final_snr_db']:.3f} dB，增益 {result['final_snr_db']-result['pre_snr_db']:.3f} dB。",
                f"- Gabor 角：中位数 {np.median(angles):.3f}°，95%分位数 {np.percentile(angles,95):.3f}°，<30° 比例 {np.mean(angles<30):.1%}。",
                f"- 小波前后 ERP 相关系数 Fz/F3/F4：{'/'.join(f'{x:.4f}' for x in result['waveform_correlations'])}。",
                f"- 左右靶文件级 Gamma 主拟合：成功 {len(side_fits)}/6，中位 R² {np.median([x['r2'] for x in side_fits]):.3f}，R²>0.9 比例 {np.mean([x['r2']>0.9 for x in side_fits]):.1%}。",
            ]
        )
    lines.append("19. 左右靶配对检验")
    for test in tests:
        if test["analysis"] not in {"file_channel", "file_channel_correct"}:
            continue
        lines.append(
            f"- {test['analysis']} / {test['scope']} / {test['metric']}：n={test['n_pairs']}，t={test['t']:.4f}，p={test['p_raw']:.6g}，Holm p={test.get('p_holm',np.nan):.6g}，dz={test['cohen_dz']:.4f}。"
        )
    retention_ok = all(x["retention"] > 0.70 for x in results)
    angles = np.concatenate([x["gabor_angles"][np.isfinite(x["gabor_angles"])] for x in results])
    side_fits = [x for x in fit_rows if x["group_type"] == "side" and x["fit_status"] == "ok"]
    latency_ok = np.mean([250 <= x["peak_latency_ms"] <= 500 for x in side_fits]) if side_fits else 0
    r2_high = np.mean([x["r2"] > 0.9 for x in side_fits]) if side_fits else 0
    lines.extend(
        [
            "20. P1 自检与阶段结论",
            f"1. 四文件保留率>70%：{'通过' if retention_ok else '未通过'}。",
            f"2. Gabor 角中位数 {np.median(angles):.3f}°，<30°比例 {np.mean(angles<30):.1%}：{'通过' if np.median(angles)<30 else '未通过'}。",
            f"3. 左右靶主拟合中 P300 潜伏期位于 250—500 ms 的比例 {latency_ok:.1%}。",
            f"4. 左右靶主拟合 R²>0.90 比例 {r2_high:.1%}，中位 R² {np.median([x['r2'] for x in side_fits]):.3f}。",
            "5. 统计显著性以第19节原始及 Holm 校正 p 值为准；不通过数据选择追求显著。",
            "<!-- P1_AUTO_END -->",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    dataset = root / "C题" / "dataset"
    figure_dir = root / "P1_图表"
    figure_dir.mkdir(parents=True, exist_ok=True)
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
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    paths = sorted(dataset.glob("VisualCog?_Task-?.mat"))
    if len(paths) != 4:
        raise RuntimeError(f"Expected 4 MAT files, found {len(paths)}")
    results = []
    for path in paths:
        print(f"处理 {path.name}")
        result = extract_file(path)
        results.append(result)
        angles = result["gabor_angles"][np.isfinite(result["gabor_angles"])]
        print(
            f"  保留率={result['retention']:.2%}, 阈值=±{result['artifact_threshold']:.0f}, "
            f"SNR增益={result['final_snr_db']-result['pre_snr_db']:.3f} dB, Gabor中位角={np.median(angles):.3f}°"
        )
    fit_rows = fit_all(results)
    tests = make_tests(fit_rows, results)
    asymmetry_rows = make_asymmetry(fit_rows)
    parameter_summary = make_parameter_summary(fit_rows)
    fit_lookup = {(x["file"], x["channel"], x["group_type"], x["condition"]): x for x in fit_rows}
    output_rows = []
    for result in results:
        angles = result["gabor_angles"][np.isfinite(result["gabor_angles"])]
        output_rows.append(
            {
                "record_type": "file_quality",
                "file": result["file"],
                "subject": result["subject"],
                "task": result["task"],
                "artifact_threshold": result["artifact_threshold"],
                "primary_retention": result["primary_retention"],
                "retention": result["retention"],
                "snr_before_db": result["pre_snr_db"],
                "snr_after_db": result["final_snr_db"],
                "snr_gain_db": result["final_snr_db"] - result["pre_snr_db"],
                "gabor_angle_median_deg": float(np.median(angles)),
                "gabor_angle_p95_deg": float(np.percentile(angles, 95)),
                "gabor_below_30_ratio": float(np.mean(angles < 30)),
            }
        )
        for threshold in result["thresholds"]:
            output_rows.append({"record_type": "wavelet_threshold", "file": result["file"], "subject": result["subject"], "task": result["task"], **threshold})
        for trial in result["trials"]:
            trial_row = {
                "record_type": "trial_quality",
                "file": result["file"],
                "subject": result["subject"],
                "task": result["task"],
                "trial": trial["trial"],
                "cue_sample": trial["cue_sample"],
                "cue_side": trial["cue_side"],
                "response_side": trial["response_side"],
                "correct": trial["correct"],
                "accepted": trial["accepted"],
                "rejection_reason": trial["rejection_reason"],
                "saturation": trial.get("saturation", False),
                "max_abs": trial.get("max_abs", np.nan),
                "peak_to_peak": trial.get("peak_to_peak", np.nan),
            }
            output_rows.append(trial_row)
            if trial["accepted"]:
                for channel_index, channel in enumerate(CHANNELS):
                    output_rows.append(
                        {
                            "record_type": "trial_p300",
                            "file": result["file"],
                            "subject": result["subject"],
                            "task": result["task"],
                            "trial": trial["trial"],
                            "channel": channel,
                            "cue_side": trial["cue_side"],
                            "correct": trial["correct"],
                            "peak_amplitude": trial["trial_peak_amplitudes"][channel_index],
                            "peak_latency_ms": trial["trial_peak_latencies_ms"][channel_index],
                            "gabor_angle_deg": trial["gabor_angles"][channel_index],
                        }
                    )
        stem = f"{result['subject']}_Task{result['task']}"
        plot_denoise(result, figure_dir / f"{stem}_denoise_comparison")
        plot_gamma(result, fit_lookup, figure_dir / f"{stem}_gamma_fit")
    for fit in fit_rows:
        output_rows.append({"record_type": "gamma_fit", **fit})
    for test in tests:
        output_rows.append({"record_type": "ttest", **test})
        if test["analysis"] in {"file_channel", "file_channel_correct"}:
            print(
                f"t检验 {test['analysis']} {test['scope']} {test['metric']}: "
                f"t={test['t']:.4f}, p={test['p_raw']:.6g}, Holm p={test.get('p_holm',np.nan):.6g}, n={test['n_pairs']}"
            )
    for row in asymmetry_rows:
        output_rows.append({"record_type": "asymmetry", **row})
    output_rows.extend(parameter_summary)
    for task in [1, 2]:
        plot_task_erp(results, task, "side", figure_dir / f"Task{task}_grand_erp_side")
        plot_task_erp(results, task, "accuracy", figure_dir / f"Task{task}_grand_erp_accuracy")
        plot_task_gamma(results, task, figure_dir / f"Task{task}_gamma_summary")
    plot_asymmetry(asymmetry_rows, figure_dir / "p300_asymmetry")
    plot_project_comparison(fit_rows, tests, figure_dir / "task_comparison")
    write_csv(root / "P1_结果.csv", output_rows)
    plan_path = root / "P1_方案.txt"
    plan = plan_path.read_text(encoding="utf-8")
    if "<!-- P1_AUTO_START -->" in plan:
        plan = plan.split("\n十八、实际结果（自动生成）", 1)[0].rstrip() + "\n"
    plan_path.write_text(plan + plan_summary(results, fit_rows, tests), encoding="utf-8")
    (figure_dir / "P1_图表清单.json").write_text(
        json.dumps({"figures": sorted(x.name for x in figure_dir.glob("*.png")), "dpi": 400, "backend": "Python/Matplotlib"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"完成：{root / 'P1_方案.txt'}")
    print(f"完成：{root / 'P1_结果.csv'}")
    print(f"图表：{figure_dir}")


if __name__ == "__main__":
    main()
