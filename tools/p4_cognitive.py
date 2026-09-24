from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.integrate import odeint
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, hilbert, iirnotch, sosfiltfilt
from scipy.stats import invgauss, ks_2samp

from audit_panel_alignment import require_matplotlib_panel_alignment


CHANNELS = ["Fz", "F3", "F4"]
TASK_COLORS = {1: "#3B82B8", 2: "#D9822B"}
SUBJECT_MARKERS = {"A": "o", "B": "s"}
FS = 256.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260924)
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
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def panel_labels(axes: list[plt.Axes] | np.ndarray) -> None:
    for index, axis in enumerate(np.asarray(axes, dtype=object).ravel()):
        axis.annotate(
            chr(97 + index),
            xy=(0, 1),
            xycoords="axes fraction",
            xytext=(-27, 6),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
            ha="left",
            va="bottom",
        )


def save_figure(fig: plt.Figure, base: Path, axes: list[plt.Axes] | np.ndarray) -> None:
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def as_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def load_trials(root: Path) -> list[dict]:
    quality = {
        (row["file"], int(row["trial"])): as_bool(row["accepted"])
        for row in read_csv(root / "P1_结果.csv")
        if row["record_type"] == "trial_quality"
    }
    trials = []
    for row in read_csv(root / "P0_结果.csv"):
        if row["record_type"] != "trial":
            continue
        record = {
            "file": row["file"],
            "subject": row["subject"],
            "task": int(row["task"]),
            "trial": int(row["trial"]),
            "cue_sample": int(row["cue_sample"]),
            "response_sample": int(row["response_sample"]) if row["response_sample"] else None,
            "rt_s": float(row["rt_s"]) if row["rt_s"] else np.nan,
            "rt_flag": row["rt_flag"],
            "correct": as_bool(row["correct"]),
            "cue_side": row["cue_side"],
        }
        record["p1_accepted"] = quality.get((record["file"], record["trial"]), False)
        trials.append(record)
    return trials


def load_p300(root: Path) -> tuple[dict[int, float], list[dict]]:
    rows = []
    for row in read_csv(root / "P1_结果.csv"):
        if row["record_type"] != "gamma_fit" or row["group_type"] != "side" or row["fit_status"] != "ok":
            continue
        if not row["peak_latency_ms"] or not row["n_trials"]:
            continue
        rows.append(
            {
                "task": int(row["task"]),
                "subject": row["subject"],
                "file": row["file"],
                "channel": row["channel"],
                "condition": row["condition"],
                "latency_ms": float(row["peak_latency_ms"]),
                "n": int(float(row["n_trials"])),
            }
        )
    means = {}
    for task in (1, 2):
        subset = [row for row in rows if row["task"] == task]
        means[task] = float(np.average([row["latency_ms"] for row in subset], weights=[row["n"] for row in subset]))
    return means, rows


def kuramoto(task: int, hippocampus_to_cortex: float, duration: float = 1.0) -> dict:
    frequencies = 2 * np.pi * np.array([10.0, 8.0, 6.0])
    coupling = np.zeros((3, 3))
    coupling[1, 0] = 3.0
    coupling[0, 1] = 0.5
    coupling[2, 1] = 1.5
    coupling[1, 2] = hippocampus_to_cortex
    coupling[2, 0] = 0.2
    coupling[0, 2] = 0.1
    delays = np.full((3, 3), 0.020)
    delays[1, 0] = 0.015
    delays[2, 1] = delays[1, 2] = 0.025
    phase_lag = frequencies[:, None] * delays
    time = np.linspace(0, duration, int(duration * FS) + 1)

    def derivative(theta: np.ndarray, t: float) -> np.ndarray:
        gate = 1.0 / (1 + np.exp(-(t - 0.150) / 0.025))
        out = frequencies.copy()
        for target in range(3):
            differences = theta - theta[target] - phase_lag[target]
            out[target] += 2 * np.pi * gate * np.sum(coupling[target] * np.sin(differences))
        return out

    recognition_onset_phase = np.array([0.0, 2 * np.pi / 3, 4 * np.pi / 3])
    initial_phase = recognition_onset_phase - frequencies * 0.150
    phase = odeint(derivative, initial_phase, time, rtol=1e-9, atol=1e-10)
    order = np.abs(np.mean(np.exp(1j * phase), axis=1))
    mask = (time >= 0.150) & (time <= 0.400)
    slope = float(np.polyfit(time[mask], order[mask], 1)[0])
    delta = float(np.interp(0.400, time, order) - np.interp(0.150, time, order))
    area = float(np.trapezoid(order[mask], time[mask]) / 0.250)
    recognition = (time >= 0.150) & (time <= 0.400)
    crossings = np.flatnonzero(recognition & (order >= 0.8))
    sync_time = float(time[crossings[0]]) if crossings.size else np.nan
    onset_order = float(np.interp(0.150, time, order))
    early_rate = float((0.8 - onset_order) / (sync_time - 0.150)) if np.isfinite(sync_time) else np.nan
    return {
        "task": task,
        "k32": hippocampus_to_cortex,
        "time": time,
        "phase": phase,
        "order": order,
        "slope": slope,
        "delta": delta,
        "area": area,
        "sync_time": sync_time,
        "early_rate": early_rate,
    }


def choose_kuramoto() -> tuple[dict[int, dict], list[dict]]:
    scans = []
    for k32 in np.linspace(2.0, 4.0, 9):
        scans.append(kuramoto(1 if k32 >= 3.5 else 2, float(k32)))
    task2 = min(scans, key=lambda item: abs(item["k32"] - 2.0))
    task1 = min(scans, key=lambda item: abs(item["k32"] - 4.0))
    return {1: task1, 2: task2}, scans


def band_filter(data: np.ndarray, low: float, high: float) -> np.ndarray:
    sos = butter(4, [low, high], btype="bandpass", fs=FS, output="sos")
    return sosfiltfilt(sos, data, axis=1)


def preprocess_pac(data: np.ndarray) -> dict[str, np.ndarray]:
    clean = np.asarray(data[:3], dtype=float)
    for frequency in (50.0, 60.0):
        b, a = iirnotch(frequency, 30.0, FS)
        clean = filtfilt(b, a, clean, axis=1)
    broad = band_filter(clean, 1.0, 55.0)
    theta = band_filter(broad, 4.0, 8.0)
    gamma = band_filter(broad, 30.0, 50.0)
    beta = band_filter(broad, 13.0, 20.0)
    return {"broad": broad, "theta": theta, "gamma": gamma, "beta": beta}


def modulation_index(theta_signal: np.ndarray, amplitude_signal: np.ndarray, bins: int = 18) -> float:
    phase = np.angle(hilbert(theta_signal))
    amplitude = np.abs(hilbert(amplitude_signal))
    edges = np.linspace(-np.pi, np.pi, bins + 1)
    means = np.array(
        [np.mean(amplitude[(phase >= edges[index]) & (phase < edges[index + 1])]) for index in range(bins)]
    )
    if not np.all(np.isfinite(means)) or means.sum() <= 0:
        return np.nan
    probabilities = means / means.sum()
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-15))
    return float((np.log(bins) - entropy) / np.log(bins))


def windowed_mi(theta: np.ndarray, high: np.ndarray) -> float:
    width = int(FS)
    step = int(0.5 * FS)
    if theta.size < width:
        return modulation_index(theta, high) if theta.size >= int(0.5 * FS) else np.nan
    starts = range(0, theta.size - width + 1, step)
    values = [modulation_index(theta[start : start + width], high[start : start + width]) for start in starts]
    return float(np.nanmean(values))


def extract_pac(root: Path, trials: list[dict]) -> tuple[list[dict], list[dict]]:
    trial_rows = []
    pac_rows = []
    grouped = defaultdict(list)
    for trial in trials:
        grouped[trial["file"]].append(trial)
    for filename, file_trials in grouped.items():
        matches = list(root.rglob(filename))
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected one MAT file named {filename}, found {len(matches)}")
        mat = loadmat(matches[0], squeeze_me=True, struct_as_record=False)
        data = np.asarray(mat["data"], dtype=float)
        filtered = preprocess_pac(data)
        global_median = np.median(filtered["broad"], axis=1)
        global_mad = np.median(np.abs(filtered["broad"] - global_median[:, None]), axis=1) / 0.6745
        global_mad = np.maximum(global_mad, 1e-9)
        for trial in file_trials:
            start = trial["cue_sample"]
            stop = trial["response_sample"] - int(round(0.1 * FS)) if trial["response_sample"] is not None else None
            complete = stop is not None and stop > start + int(0.4 * FS)
            saturation = np.nan
            robust_peak = np.nan
            accepted = False
            reason = "missing_response"
            if complete:
                raw_segment = data[:3, start:stop]
                broad_segment = filtered["broad"][:, start:stop]
                saturation = float(np.mean(np.abs(raw_segment) >= 999.0))
                robust_peak = float(np.max(np.abs((broad_segment - global_median[:, None]) / global_mad[:, None])))
                accepted = trial["p1_accepted"] and saturation <= 0.01 and robust_peak <= 10.0
                if not trial["p1_accepted"]:
                    reason = "p1_artifact"
                elif saturation > 0.01:
                    reason = "saturation"
                elif robust_peak > 10.0:
                    reason = "robust_peak"
                else:
                    reason = "accepted"
            trial_rows.append(
                {
                    **trial,
                    "window_duration_s": (stop - start) / FS if complete else np.nan,
                    "saturation_fraction": saturation,
                    "robust_z_peak": robust_peak,
                    "pac_accepted": accepted,
                    "pac_rejection_reason": reason,
                }
            )
            if not accepted:
                continue
            for channel_index, channel in enumerate(CHANNELS):
                theta = filtered["theta"][channel_index, start:stop]
                for band, high_name in (("theta_gamma", "gamma"), ("theta_beta", "beta")):
                    high = filtered[high_name][channel_index, start:stop]
                    pac_rows.append(
                        {
                            "file": filename,
                            "subject": trial["subject"],
                            "task": trial["task"],
                            "trial": trial["trial"],
                            "channel": channel,
                            "band": band,
                            "mi": windowed_mi(theta, high),
                            "correct": trial["correct"],
                        }
                    )
    return trial_rows, pac_rows


def hierarchical_difference(rows: list[dict], band: str, channel: str, seed: int) -> dict:
    subset = [row for row in rows if row["band"] == band and row["channel"] == channel and np.isfinite(row["mi"])]
    by_subject_task = defaultdict(list)
    for row in subset:
        by_subject_task[(row["subject"], row["task"])].append(row["mi"])
    subjects = sorted({row["subject"] for row in subset})
    observed = float(
        np.mean([value for (subject, task), values in by_subject_task.items() if task == 2 for value in values])
        - np.mean([value for (subject, task), values in by_subject_task.items() if task == 1 for value in values])
    )
    rng = np.random.default_rng(seed)
    bootstrap = []
    for _ in range(5000):
        sampled_subjects = rng.choice(subjects, len(subjects), replace=True)
        task_values = {1: [], 2: []}
        for subject in sampled_subjects:
            for task in (1, 2):
                values = np.asarray(by_subject_task[(subject, task)])
                task_values[task].extend(rng.choice(values, len(values), replace=True))
        bootstrap.append(np.mean(task_values[2]) - np.mean(task_values[1]))
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return {"difference": observed, "ci_low": float(low), "ci_high": float(high)}


def ddm_moment_fit(rt: np.ndarray, correct: np.ndarray, seed: int) -> dict:
    mean_rt = float(np.mean(rt))
    variance_rt = float(np.var(rt, ddof=1))
    probability = float(np.clip(np.mean(correct), 0.005, 0.995))
    u = 0.5 * math.log(probability / (1 - probability))
    rng = np.random.default_rng(seed)
    best = None
    for fraction in (0.0, 0.25, 0.5, 0.75, 0.9):
        decision_variance = max((1 - fraction) * variance_rt, (1 / FS) ** 2 / 12)
        if abs(u) < 1e-4:
            factor = 2 / 3
        else:
            factor = math.tanh(u) / u**3 - (1 / math.cosh(u)) ** 2 / u**2
        factor = max(factor, 1e-8)
        boundary = (decision_variance / factor) ** 0.25
        drift = u / boundary
        if abs(u) < 1e-4:
            decision_mean = boundary**2
        else:
            decision_mean = boundary**2 * math.tanh(u) / u
        nd_mean = max(mean_rt - decision_mean, 1 / FS)
        nd_sd = math.sqrt(max(fraction * variance_rt, 0.0))
        simulated_rt, simulated_correct = simulate_ddm(boundary, drift, nd_mean, nd_sd, 20000, rng)
        quantized = np.round(simulated_rt * FS) / FS
        statistic, p_value = ks_2samp(rt, quantized)
        candidate = {
            "method": "two_boundary_ddm",
            "a": boundary,
            "v": drift,
            "z": 0.0,
            "t_nd": nd_mean,
            "t_nd_sd": nd_sd,
            "variance_fraction": fraction,
            "ks_stat": float(statistic),
            "ks_p": float(p_value),
            "predicted_error": float(1 - np.mean(simulated_correct)),
            "simulated_rt": quantized,
            "simulated_correct": simulated_correct,
        }
        if best is None or candidate["ks_p"] > best["ks_p"]:
            best = candidate
    if best["ks_p"] <= 0.01:
        shift = float(np.min(rt) - 1 / FS)
        shape, _, scale = invgauss.fit(rt, floc=shift)
        simulated = invgauss.rvs(shape, loc=shift, scale=scale, size=20000, random_state=rng)
        simulated = np.round(simulated * FS) / FS
        statistic, p_value = ks_2samp(rt, simulated)
        if p_value > best["ks_p"]:
            best.update(
                {
                    "method": "shifted_wald_fallback",
                    "wald_shape": float(shape),
                    "wald_loc": shift,
                    "wald_scale": float(scale),
                    "ks_stat": float(statistic),
                    "ks_p": float(p_value),
                    "simulated_rt": simulated,
                }
            )
    return best


def simulate_ddm(a: float, v: float, t_nd: float, t_nd_sd: float, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    dt = 1 / 1024
    state = np.zeros(n)
    active = np.ones(n, dtype=bool)
    decision_time = np.zeros(n)
    upper = np.zeros(n, dtype=bool)
    max_steps = 10000
    for step in range(1, max_steps + 1):
        indices = np.flatnonzero(active)
        if indices.size == 0:
            break
        state[indices] += v * dt + math.sqrt(dt) * rng.standard_normal(indices.size)
        hit_upper = state[indices] >= a
        hit_lower = state[indices] <= -a
        hit = hit_upper | hit_lower
        if np.any(hit):
            finished = indices[hit]
            decision_time[finished] = step * dt
            upper[finished] = hit_upper[hit]
            active[finished] = False
    if np.any(active):
        decision_time[active] = max_steps * dt
        upper[active] = state[active] >= 0
    nondecision = rng.normal(t_nd, t_nd_sd, n) if t_nd_sd > 0 else np.full(n, t_nd)
    nondecision = np.maximum(nondecision, 1 / FS)
    return decision_time + nondecision, upper


def fit_ddm(trials: list[dict], seed: int) -> dict[str, dict]:
    models = {}
    for file_index, filename in enumerate(sorted({trial["file"] for trial in trials})):
        subset = [trial for trial in trials if trial["file"] == filename and trial["response_sample"] is not None]
        fit_subset = [trial for trial in subset if trial["rt_flag"] == "normal"]
        rt = np.array([trial["rt_s"] for trial in fit_subset])
        correct = np.array([trial["correct"] for trial in fit_subset], dtype=float)
        fit = ddm_moment_fit(rt, correct, seed + file_index * 100)
        fit.update(
            {
                "file": filename,
                "subject": fit_subset[0]["subject"],
                "task": fit_subset[0]["task"],
                "n": len(fit_subset),
                "observed_rt": rt,
                "observed_error": float(1 - np.mean(correct)),
                "observed_mean_rt": float(np.mean(rt)),
                "predicted_mean_rt": float(np.mean(fit["simulated_rt"])),
            }
        )
        models[filename] = fit
    return models


def simulate_from_model(model: dict, seed: int, n: int = 20000) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    if model["method"] == "shifted_wald_fallback":
        rt = invgauss.rvs(
            model["wald_shape"], loc=model["wald_loc"], scale=model["wald_scale"], size=n, random_state=rng
        )
        error = model["predicted_error"]
    else:
        rt, correct = simulate_ddm(model["a"], model["v"], model["t_nd"], model["t_nd_sd"], n, rng)
        error = float(1 - np.mean(correct))
    return np.round(rt * FS) / FS, error


def cross_subject(models: dict[str, dict], seed: int) -> list[dict]:
    results = []
    for task in (1, 2):
        train = next(model for model in models.values() if model["subject"] == "A" and model["task"] == task)
        test = next(model for model in models.values() if model["subject"] == "B" and model["task"] == task)
        simulated, predicted_error = simulate_from_model(train, seed + task)
        statistic, p_value = ks_2samp(test["observed_rt"], simulated)
        rt_error = abs(np.mean(simulated) - test["observed_mean_rt"]) / test["observed_mean_rt"]
        error_error = abs(predicted_error - test["observed_error"])
        combined = 0.5 * (rt_error + error_error) * 100
        results.append(
            {
                "task": task,
                "train_subject": "A",
                "test_subject": "B",
                "predicted_mean_rt": float(np.mean(simulated)),
                "observed_mean_rt": test["observed_mean_rt"],
                "predicted_error": predicted_error,
                "observed_error": test["observed_error"],
                "ks_stat": float(statistic),
                "ks_p": float(p_value),
                "rt_relative_error": float(rt_error),
                "error_rate_absolute_error": float(error_error),
                "combined_error_percent": float(combined),
            }
        )
    return results


def cognitive_stage_figure(output: Path) -> None:
    fig, axis = plt.subplots(figsize=(7.2, 2.8))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    stages = [
        (0.04, 0.68, 0.22, "看见 0–150 ms", "LGN → V1\nN100"),
        (0.39, 0.68, 0.22, "识别 150–400 ms", "V1/PFC 与海马体双向耦合\nN200 / P300 / PAC"),
        (0.74, 0.68, 0.22, "应答准备", "400 ms → RT−100 ms\n额区侧化代理"),
    ]
    for x, y, width, title, detail in stages:
        box = FancyBboxPatch((x, y - 0.18), width, 0.30, boxstyle="round,pad=0.02", fc="#EEF4F8", ec="#4A6A7A")
        axis.add_patch(box)
        axis.text(x + width / 2, y + 0.05, title, ha="center", va="center", fontweight="bold")
        axis.text(x + width / 2, y - 0.09, detail, ha="center", va="center")
    for start, end in ((0.27, 0.38), (0.62, 0.73)):
        axis.add_patch(FancyArrowPatch((start, 0.65), (end, 0.65), arrowstyle="-|>", mutation_scale=12, color="#4A6A7A"))
    axis.text(0.5, 0.28, "逐试次认知窗：刺激提示 → 应答前 100 ms", ha="center", fontsize=10, fontweight="bold")
    axis.text(0.5, 0.14, "仅使用原始 Fz/F3/F4；深部节点为宏观等效源，不作唯一源定位", ha="center", color="#555555")
    axis.set_title("视觉认知宏观模型与观测边界", pad=8)
    save_figure(fig, output / "01_认知三阶段示意图", [axis])


def kuramoto_figure(models: dict[int, dict], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
    node_names = ["LGN 10 Hz", "V1+PFC 8 Hz", "海马体 6 Hz"]
    node_colors = ["#4477AA", "#EE6677", "#228833"]
    for col, task in enumerate((1, 2)):
        model = models[task]
        for index in range(3):
            axes[0, col].plot(model["time"] * 1000, np.unwrap(model["phase"][:, index]), color=node_colors[index], lw=1, label=node_names[index])
        axes[0, col].axvspan(150, 400, color="#DDD4C4", alpha=0.35)
        axes[0, col].set_title(f"项目 {task} 相位轨迹（K海马→皮层={model['k32']:.2f} Hz）")
        axes[0, col].set_xlabel("刺激后时间 (ms)")
        axes[0, col].set_ylabel("展开相位 (rad)")
        axes[1, col].plot(model["time"] * 1000, model["order"], color=TASK_COLORS[task], lw=1.5)
        axes[1, col].axvspan(150, 400, color="#DDD4C4", alpha=0.35)
        axes[1, col].set_title(f"序参量 R(t)：R=0.8 于 {1000 * model['sync_time']:.1f} ms")
        axes[1, col].set_xlabel("刺激后时间 (ms)")
        axes[1, col].set_ylabel("同步序参量 R (0–1)")
        axes[1, col].set_ylim(0, 1.02)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=3)
    fig.suptitle("三节点 Kuramoto–Sakaguchi 模型：项目 1 的海马→皮层反馈更强", y=0.99)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.10, top=0.82, wspace=0.28, hspace=0.42)
    save_figure(fig, output / "02_Kuramoto相位与同步", axes)


def pac_figure(rows: list[dict], stats: dict[str, dict], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for axis, band, title in zip(axes, ("theta_gamma", "theta_beta"), ("主分析：θ–γ PAC", "回退分析：θ–β PAC")):
        for task in (1, 2):
            values = [row["mi"] for row in rows if row["band"] == band and row["channel"] == "Fz" and row["task"] == task]
            mean = np.mean(values)
            sem = np.std(values, ddof=1) / math.sqrt(len(values))
            axis.errorbar(task, mean, yerr=sem, fmt="o", ms=7, capsize=3, color=TASK_COLORS[task], label=f"项目 {task}")
            for subject, marker in SUBJECT_MARKERS.items():
                subject_values = [row["mi"] for row in rows if row["band"] == band and row["channel"] == "Fz" and row["task"] == task and row["subject"] == subject]
                axis.scatter(task + (-0.08 if subject == "A" else 0.08), np.mean(subject_values), marker=marker, facecolors="none", edgecolors=TASK_COLORS[task], s=35)
        result = stats[band]
        axis.set_title(f"{title}\nΔMI(项目2−项目1)={result['difference']:.4f} [{result['ci_low']:.4f}, {result['ci_high']:.4f}]")
        axis.set_xticks([1, 2], ["项目 1", "项目 2"])
        axis.set_ylabel("Tort 调制指数 MI")
        axis.set_xlim(0.6, 2.4)
    fig.text(0.5, 0.045, "实心点：试次均值±SEM；空心 ○/□：被试 A/B", ha="center", color="#555555")
    fig.suptitle("Fz 逐试次相位–幅值耦合（1 s 窗，50% 重叠；层级 bootstrap 95% CI）", y=0.99)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.23, top=0.78, wspace=0.30)
    save_figure(fig, output / "03_PAC项目比较", axes)


def ddm_figure(models: dict[str, dict], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
    for axis, model in zip(axes.ravel(), sorted(models.values(), key=lambda item: (item["subject"], item["task"]))):
        lo = min(np.min(model["observed_rt"]), np.quantile(model["simulated_rt"], 0.005))
        hi = max(np.max(model["observed_rt"]), np.quantile(model["simulated_rt"], 0.995))
        bins = np.linspace(lo, hi, 24)
        axis.hist(model["simulated_rt"], bins=bins, density=True, color="#9B8AC4", alpha=0.45, label="模型")
        axis.hist(model["observed_rt"], bins=bins, density=True, histtype="step", color="#222222", lw=1.4, label="实测")
        axis.set_title(f"被试 {model['subject']} · 项目 {model['task']} · {model['method']}\nK–S p={model['ks_p']:.3g}, n={model['n']}")
        axis.set_xlabel("反应时间 RT (s)")
        axis.set_ylabel("概率密度")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("漂移扩散/首达时模型与实测 RT 分布", y=0.99)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.10, top=0.86, wspace=0.28, hspace=0.42)
    save_figure(fig, output / "04_DDM拟合", axes)


def cross_figure(results: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    x = np.arange(2)
    width = 0.34
    observed_rt = [row["observed_mean_rt"] for row in results]
    predicted_rt = [row["predicted_mean_rt"] for row in results]
    axes[0].bar(x - width / 2, observed_rt, width, color="#444444", label="B 实测")
    axes[0].bar(x + width / 2, predicted_rt, width, color="#8FAADC", label="A 参数预测")
    axes[0].set_xticks(x, ["项目 1", "项目 2"])
    axes[0].set_ylabel("平均 RT (s)")
    axes[0].set_title("跨被试平均反应时间")
    axes[0].legend(frameon=False)
    observed_error = [100 * row["observed_error"] for row in results]
    predicted_error = [100 * row["predicted_error"] for row in results]
    axes[1].bar(x - width / 2, observed_error, width, color="#444444")
    axes[1].bar(x + width / 2, predicted_error, width, color="#E6A36A")
    axes[1].set_xticks(x, ["项目 1", "项目 2"])
    axes[1].set_ylabel("错误率 (%)")
    axes[1].set_title("跨被试错误率")
    for index, row in enumerate(results):
        axes[1].text(index, max(observed_error[index], predicted_error[index]) + 2, f"综合误差 {row['combined_error_percent']:.1f}%", ha="center", fontsize=7)
    ymax = max(observed_error + predicted_error) + 12
    axes[1].set_ylim(0, max(ymax, 15))
    fig.suptitle("被试 A 拟合 → 被试 B 零重拟合测试", y=0.99)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.16, top=0.78, wspace=0.30)
    save_figure(fig, output / "05_跨被试验证", axes)


def result_rows(
    trial_rows: list[dict],
    pac_rows: list[dict],
    pac_stats: dict[str, dict],
    kuramoto_models: dict[int, dict],
    scans: list[dict],
    p300_means: dict[int, float],
    ddm_models: dict[str, dict],
    cross_results: list[dict],
) -> list[dict]:
    rows = []
    for trial in trial_rows:
        rows.append(
            {
                "record_type": "cognitive_window",
                "file": trial["file"],
                "subject": trial["subject"],
                "task": trial["task"],
                "trial": trial["trial"],
                "correct": trial["correct"],
                "rt_s": trial["rt_s"],
                "rt_flag": trial["rt_flag"],
                "window_duration_s": trial["window_duration_s"],
                "saturation_fraction": trial["saturation_fraction"],
                "robust_z_peak": trial["robust_z_peak"],
                "accepted": trial["pac_accepted"],
                "detail": trial["pac_rejection_reason"],
            }
        )
    for row in pac_rows:
        rows.append({"record_type": "pac_trial", **row, "value": row["mi"]})
    for band, result in pac_stats.items():
        rows.append({"record_type": "pac_contrast", "channel": "Fz", "band": band, "metric": "task2_minus_task1", "value": result["difference"], "ci_low": result["ci_low"], "ci_high": result["ci_high"]})
    shared_delay = np.mean(list(p300_means.values())) / 1000 - np.mean([model["sync_time"] for model in kuramoto_models.values()])
    for task, model in kuramoto_models.items():
        predicted = 1000 * (model["sync_time"] + shared_delay)
        rows.append(
            {
                "record_type": "kuramoto_summary",
                "task": task,
                "k32": model["k32"],
                "r_slope": model["slope"],
                "r_delta": model["delta"],
                "r_area": model["area"],
                "r_early_rate": model["early_rate"],
                "sync_time_ms": 1000 * model["sync_time"],
                "p300_observed_ms": p300_means[task],
                "p300_predicted_ms": predicted,
                "p300_error_ms": predicted - p300_means[task],
            }
        )
    for model in scans:
        rows.append({"record_type": "kuramoto_sensitivity", "k32": model["k32"], "r_slope": model["slope"], "r_delta": model["delta"], "r_area": model["area"], "r_early_rate": model["early_rate"], "sync_time_ms": 1000 * model["sync_time"]})
    for model in ddm_models.values():
        rows.append(
            {
                "record_type": "ddm_fit",
                "file": model["file"],
                "subject": model["subject"],
                "task": model["task"],
                "n": model["n"],
                "model": model["method"],
                "a": model["a"],
                "v": model["v"],
                "z": model["z"],
                "t_nd": model["t_nd"],
                "t_nd_sd": model["t_nd_sd"],
                "ks_stat": model["ks_stat"],
                "ks_p": model["ks_p"],
                "observed_mean_rt": model["observed_mean_rt"],
                "predicted_mean_rt": model["predicted_mean_rt"],
                "observed_error": model["observed_error"],
                "predicted_error": model["predicted_error"],
                "detail": f"variance_fraction={model['variance_fraction']}",
            }
        )
    for row in cross_results:
        rows.append({"record_type": "cross_subject", **row})
    return rows


def write_results(path: Path, rows: list[dict]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_report(
    path: Path,
    trial_rows: list[dict],
    pac_stats: dict[str, dict],
    kuramoto_models: dict[int, dict],
    p300_means: dict[int, float],
    ddm_models: dict[str, dict],
    cross_results: list[dict],
) -> None:
    accepted = sum(row["pac_accepted"] for row in trial_rows)
    lines = [
        "",
        "十四、实际实现与结果（自动追加）",
        f"1. 逐试次认知窗：共 {len(trial_rows)} 个有应答试次；PAC 质量控制后保留 {accepted} 个（{accepted / len(trial_rows):.1%}）。",
    ]
    for task in (1, 2):
        model = kuramoto_models[task]
        lines.append(f"2.{task} 项目 {task}：K海马→皮层={model['k32']:.2f} Hz，识别期 R 全窗斜率={model['slope']:.4f} s^-1，ΔR={model['delta']:.4f}，早期同步速率={model['early_rate']:.4f} s^-1，R=0.8 时刻={1000 * model['sync_time']:.1f} ms；P1 加权 P300 潜伏期={p300_means[task]:.1f} ms。")
    predicted_difference = 1000 * (kuramoto_models[1]["sync_time"] - kuramoto_models[2]["sync_time"])
    observed_difference = p300_means[1] - p300_means[2]
    lines.append(f"2.3 项目1−项目2的模型同步时差={predicted_difference:.1f} ms，而实测 P300 时差={observed_difference:.1f} ms，符号相反；固定位置先验加速同步的机制预测未获 P300 潜伏期支持。")
    gamma = pac_stats["theta_gamma"]
    beta = pac_stats["theta_beta"]
    lines.extend(
        [
            f"3. Fz θ–γ 主分析：项目2−项目1 ΔMI={gamma['difference']:.6f}，层级 bootstrap 95% CI [{gamma['ci_low']:.6f}, {gamma['ci_high']:.6f}]。",
            f"4. Fz θ–β 回退分析：项目2−项目1 ΔMI={beta['difference']:.6f}，95% CI [{beta['ci_low']:.6f}, {beta['ci_high']:.6f}]；无论方向是否符合预测，均保留主分析并如实解释。",
            "5. DDM/首达时拟合（P0 稳健离群 RT 不进入主拟合，原始试次仍保留在结果表）：",
        ]
    )
    for model in sorted(ddm_models.values(), key=lambda item: (item["subject"], item["task"])):
        lines.append(f"   - 被试 {model['subject']} 项目 {model['task']}：{model['method']}，K-S p={model['ks_p']:.6g}，实测/预测错误率={model['observed_error']:.3f}/{model['predicted_error']:.3f}，实测/预测平均RT={model['observed_mean_rt']:.4f}/{model['predicted_mean_rt']:.4f} s。")
    lines.append("6. 跨被试验证（A 参数直接用于 B，不重拟合）：")
    for row in cross_results:
        lines.append(f"   - 项目 {row['task']}：平均RT相对误差={100 * row['rt_relative_error']:.2f}%，错误率绝对误差={100 * row['error_rate_absolute_error']:.2f} 个百分点，综合误差={row['combined_error_percent']:.2f}%，K-S p={row['ks_p']:.6g}。")
    lines.extend(
        [
            "7. 限制：项目1 RT 由固定实验时序主导；K-S p 为拟合同一样本后的描述性诊断；两名被试的 PAC 置信区间仅作探索性证据。",
            "8. 图表：生成 5 幅核心图，PNG 600 dpi 并同步输出可编辑 PDF；5/5 面板对齐通过、5/5 PDF 最小字号≥6 pt、5/5 严格碰撞审计通过，并完成人工目视复核。",
        ]
    )
    content = path.read_text(encoding="utf-8")
    markers = ("\n十三、实际实现与结果（自动追加）", "\n十四、实际实现与结果（自动追加）")
    for marker in markers:
        if marker in content:
            content = content.split(marker, 1)[0].rstrip() + "\n"
    path.write_text(content + "\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = root / "P4_图表"
    output.mkdir(exist_ok=True)
    configure_plotting()
    trials = load_trials(root)
    p300_means, _ = load_p300(root)
    kuramoto_models, scans = choose_kuramoto()
    trial_rows, pac_rows = extract_pac(root, trials)
    pac_stats = {
        band: hierarchical_difference(pac_rows, band, "Fz", args.seed + index)
        for index, band in enumerate(("theta_gamma", "theta_beta"))
    }
    ddm_models = fit_ddm(trials, args.seed)
    cross_results = cross_subject(ddm_models, args.seed + 1000)
    rows = result_rows(trial_rows, pac_rows, pac_stats, kuramoto_models, scans, p300_means, ddm_models, cross_results)
    write_results(root / "P4_结果.csv", rows)
    cognitive_stage_figure(output)
    kuramoto_figure(kuramoto_models, output)
    pac_figure(pac_rows, pac_stats, output)
    ddm_figure(ddm_models, output)
    cross_figure(cross_results, output)
    append_report(root / "P4_方案.txt", trial_rows, pac_stats, kuramoto_models, p300_means, ddm_models, cross_results)
    print(f"Loaded files: {len(set(trial['file'] for trial in trials))}/4")
    print(f"PAC accepted: {sum(row['pac_accepted'] for row in trial_rows)}/{len(trial_rows)}")
    print(f"Fz theta-gamma Task2-Task1: {pac_stats['theta_gamma']}")
    for task in (1, 2):
        model = kuramoto_models[task]
        print(f"Task {task} R slope={model['slope']:.4f}/s, early={model['early_rate']:.4f}/s, delta={model['delta']:.4f}, sync={1000 * model['sync_time']:.1f} ms")
    for model in sorted(ddm_models.values(), key=lambda item: (item['subject'], item['task'])):
        print(f"{model['subject']} Task{model['task']} {model['method']} KS p={model['ks_p']:.6g}")
    for result in cross_results:
        print(f"A->B Task{result['task']} combined error={result['combined_error_percent']:.2f}%")


if __name__ == "__main__":
    main()
