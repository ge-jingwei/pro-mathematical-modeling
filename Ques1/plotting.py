from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tools.figure_qa import save_aligned


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Microsoft YaHei"]
plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})
plt.rcParams["font.size"] = 7
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["legend.frameon"] = False

BLUE = "#0F4D92"
RED = "#B64342"
GRAY = "#767676"
TEAL = "#42949E"
EXPORT_SUFFIXES = (".svg", ".pdf", ".png", ".tiff")
EXPORT_DPI = 600


def require_matplotlib_panel_alignment(fig, axes, base: Path, panel_ids: list[str]) -> None:
    save_aligned(fig, axes, base, panel_ids)


def _label(axes) -> None:
    for i, axis in enumerate(np.ravel(axes)):
        axis.text(-0.13, 1.04, chr(97 + i), transform=axis.transAxes, weight="bold", va="bottom")


def plot_quality(records, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6), constrained_layout=True)
    for axis, record in zip(axes.ravel(), records, strict=True):
        saturated = np.any(np.abs(record.eeg) >= 1000.0, axis=0)
        center = int(np.flatnonzero(saturated)[0]) if saturated.any() else int(np.argmax(np.max(np.abs(record.eeg), axis=0)))
        half = int(2.0 * record.sample_rate)
        start = max(0, min(center - half, record.eeg.shape[1] - 2 * half))
        stop = start + 2 * half
        time = np.arange(stop - start) / record.sample_rate
        for i, (signal, name, color) in enumerate(zip(record.eeg[:, start:stop], ("Fz", "F3", "F4"), (BLUE, TEAL, RED), strict=True)):
            scale = max(np.median(np.abs(signal - np.median(signal))) * 1.4826, 1.0)
            axis.plot(time, signal / scale + 3 * (2 - i), color=color, linewidth=0.7, label=name)
            bad = np.abs(signal) >= 1000.0
            axis.scatter(time[bad], signal[bad] / scale + 3 * (2 - i), color="#FFD700", s=5, zorder=3)
        ecg = record.ecg[start:stop]
        ecg_scale = max(np.median(np.abs(ecg - np.median(ecg))) * 1.4826, 1.0)
        axis.plot(time, ecg / ecg_scale - 3, color=GRAY, linewidth=0.6, label="ECG")
        axis.set(title=f"Subject {record.subject} · Task {record.task}", xlabel="Time (s)", yticks=[-3, 0, 3, 6], yticklabels=["ECG", "F4", "F3", "Fz"])
    _label(axes)
    require_matplotlib_panel_alignment(fig, axes.ravel(), output / "fig1-1_quality_diagnostic", list("abcd"))
    plt.close(fig)


def plot_tradeoff(tradeoff: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), constrained_layout=True)
    for name, group in tradeoff.groupby("file"):
        label = name.replace("VisualCog", "").replace(".mat", "")
        axes[0].plot(group["alpha"], group["fidelity"], marker="o", label=label)
        axes[1].plot(group["alpha"], group["noise_reduction_db"], marker="o", label=label)
    axes[0].axhline(0.9, color=RED, linestyle="--", linewidth=1)
    axes[0].set(xlabel="Wavelet threshold multiplier", ylabel="ERP shape correlation", ylim=(0.85, 1.01))
    axes[1].set(xlabel="Wavelet threshold multiplier", ylabel="Baseline noise reduction (dB)")
    axes[1].legend(ncol=2, fontsize=6)
    _label(axes)
    require_matplotlib_panel_alignment(fig, axes, output / "fig1-3_tradeoff", ["a", "b"])
    plt.close(fig)


def plot_method_comparison(metrics: pd.DataFrame, output: Path) -> None:
    summary = metrics.groupby("method").agg(fidelity=("fidelity", "mean"), reduction=("noise_reduction_db", "mean")).reindex(["Raw", "Decon", "Conventional", "Proposed"])
    colors = [GRAY, "#B4C0E4", TEAL, RED]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), constrained_layout=True)
    axes[0].bar(summary.index, summary["fidelity"], color=colors)
    axes[1].bar(summary.index, summary["reduction"], color=colors)
    axes[0].set(ylabel="ERP shape correlation", ylim=(0, 1.05))
    axes[1].set(ylabel="Baseline noise reduction (dB)")
    for axis in axes:
        for label in axis.get_xticklabels():
            label.set_rotation(20)
            label.set_rotation_mode("anchor")
            label.set_ha("right")
    _label(axes)
    require_matplotlib_panel_alignment(fig, axes, output / "fig1-4_method_comparison", ["a", "b"])
    plt.close(fig)


def plot_erp(erp: dict, output: Path, alignment: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6), sharex=True, constrained_layout=True)
    for task in (1, 2):
        for channel, name in enumerate(("Fz", "F3", "F4")):
            axis = axes[task - 1, channel]
            for side, color, label in ((-1, BLUE, "Left triangle"), (1, RED, "Right triangle")):
                item = erp[(task, side, channel)]
                axis.plot(item["times"] * 1000, item["mean"], color=color, label=label)
                axis.fill_between(item["times"] * 1000, item["low"], item["high"], color=color, alpha=0.16)
            axis.axvline(0, color=GRAY, linewidth=0.8)
            axis.axhline(0, color=GRAY, linewidth=0.6)
            axis.set_title(f"Task {task} · {name}")
            if channel == 0:
                axis.set_ylabel("Amplitude (µV)")
            if task == 2:
                axis.set_xlabel(f"Time from {alignment} onset (ms)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=6, ncol=2, loc="outside upper center")
    _label(axes)
    require_matplotlib_panel_alignment(fig, axes.ravel(), output / f"fig1-5_{alignment}_erp", list("abcdef"))
    plt.close(fig)


def plot_heatmaps(pooled, output: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6), sharex=True, constrained_layout=True)
    for task in (1, 2):
        task_sets = [item for item in pooled if item[0].task == task]
        times = task_sets[0][3]
        frames = pd.concat([item[1] for item in task_sets], ignore_index=True)
        values = np.concatenate([item[2] for item in task_sets])
        order = np.argsort(frames["cue_side"].to_numpy())
        boundary = int((frames["cue_side"] == -1).sum())
        limit = np.percentile(np.abs(values), 97)
        for channel, name in enumerate(("Fz", "F3", "F4")):
            axis = axes[task - 1, channel]
            axis.imshow(values[order, channel], aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit, extent=[times[0] * 1000, times[-1] * 1000, len(values), 0], interpolation="nearest")
            axis.axvline(0, color="black", linewidth=0.7)
            axis.axhline(boundary, color="#FFD700", linewidth=1)
            axis.set_title(f"Task {task} · {name}")
            if channel == 0:
                axis.set_ylabel("Trials: left then right")
            if task == 2:
                axis.set_xlabel("Time from target onset (ms)")
    _label(axes)
    require_matplotlib_panel_alignment(fig, axes.ravel(), output / "fig1-5_single_trial_heatmaps", list("abcdef"))
    plt.close(fig)


def plot_difference(erp: dict, clusters: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6), sharex=True, constrained_layout=True)
    for task in (1, 2):
        for channel, name in enumerate(("Fz", "F3", "F4")):
            axis = axes[task - 1, channel]
            left = erp[(task, -1, channel)]
            right = erp[(task, 1, channel)]
            difference = right["mean"] - left["mean"]
            axis.plot(left["times"] * 1000, difference, color=RED)
            axis.axhline(0, color=GRAY, linewidth=0.7)
            subset = clusters[(clusters["task"] == task) & (clusters["channel"] == name)]
            for row in subset.itertuples():
                axis.axvspan(row.start_ms, row.stop_ms, color="#FFD700", alpha=0.25)
            axis.set_title(f"Task {task} · {name}")
            if channel == 0:
                axis.set_ylabel("Right − left (µV)")
            if task == 2:
                axis.set_xlabel("Time from target onset (ms)")
    _label(axes)
    require_matplotlib_panel_alignment(fig, axes.ravel(), output / "fig1-6_side_difference", list("abcdef"))
    plt.close(fig)


def plot_fits(erp: dict, fits: dict, output: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6), sharex=True, sharey="row", constrained_layout=True)
    for task in (1, 2):
        for channel, name in enumerate(("Fz", "F3", "F4")):
            axis = axes[task - 1, channel]
            for side, color, label in ((-1, BLUE, "Left"), (1, RED, "Right")):
                item = erp[(task, side, channel)]
                mask = (item["times"] >= 0) & (item["times"] <= 0.6)
                axis.plot(item["times"][mask] * 1000, item["mean"][mask], color=color, alpha=0.35)
                axis.plot(item["times"][mask] * 1000, fits[(task, side, channel)]["fitted"][mask], color=color, linewidth=1.4, label=label)
            axis.set(xlim=(0, 600), title=f"Task {task} · {name}")
            if channel == 0:
                axis.set_ylabel("Amplitude (µV)")
            if task == 2:
                axis.set_xlabel("Time from target onset (ms)")
    axes[0, 2].legend(fontsize=6)
    _label(axes)
    require_matplotlib_panel_alignment(fig, axes.ravel(), output / "fig1-7_curve_fits", list("abcdef"))
    plt.close(fig)
