from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Microsoft YaHei", "DejaVu Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})

BLUE = "#315B8A"
RED = "#B44B4B"
TEAL = "#4F8C8D"
GRAY = "#747474"
GOLD = "#D69A2D"


def _labels(axes) -> None:
    for index, axis in enumerate(np.ravel(axes)):
        axis.annotate(chr(97 + index), xy=(0, 1), xycoords="axes fraction", xytext=(-18, 8), textcoords="offset points", weight="bold", fontsize=8, va="bottom")


def _save(fig, axes, path: Path, labels: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    try:
        from audit_panel_alignment import require_matplotlib_panel_alignment
        require_matplotlib_panel_alignment(fig, json_out=path.with_suffix(".alignment.json"), overlay_svg=path.with_suffix(".alignment.svg"), require_panel_labels=labels, strict=True)
    except ImportError:
        from tools.figure_qa import save_aligned
        save_aligned(fig, np.ravel(axes), path, [chr(97 + i) for i in range(len(np.ravel(axes)))])
        return
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(path.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})


def _box(axis, x, y, width, height, text, color):
    patch = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.02", facecolor=color, edgecolor="white", linewidth=0.8)
    axis.add_patch(patch)
    axis.text(x + width / 2, y + height / 2, text, ha="center", va="center", color="white", weight="bold", fontsize=8)


def _arrow(axis, start, stop):
    axis.add_patch(FancyArrowPatch(start, stop, arrowstyle="-|>", mutation_scale=10, linewidth=1.2, color=GRAY))


def plot_architecture(output: Path) -> None:
    fig, axis = plt.subplots(figsize=(7.2, 2.3), constrained_layout=True)
    axis.set(xlim=(0, 1), ylim=(0, 1))
    axis.axis("off")
    items = [(0.03, "Triangle\nstimulus", BLUE), (0.22, "Gabor\nfeatures", TEAL), (0.41, "LGN-cortex\ndelay", GOLD), (0.60, "Wilson-Cowan\nfield", RED), (0.79, "Scalp\nlead field", BLUE)]
    for x, text, color in items:
        _box(axis, x, 0.35, 0.15, 0.30, text, color)
    for left, right in zip(items[:-1], items[1:], strict=True):
        _arrow(axis, (left[0] + 0.15, 0.50), (right[0], 0.50))
    axis.text(0.5, 0.12, "Spatial code → population dynamics → synchrony → Fz/F3/F4", ha="center", color=GRAY)
    _save(fig, [axis], output / "fig2-1_model_architecture", labels=False)
    plt.close(fig)


def plot_shape_encoding(model: dict, output: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.3), constrained_layout=True)
    for row, side in enumerate((-1, 1)):
        item = model[side]
        images = (item.stimulus, item.features.mean(axis=0), item.cortical_input)
        titles = ("Stimulus", "Orientation features", "Cortical projection")
        for column, (image, title) in enumerate(zip(images, titles, strict=True)):
            axes[row, column].imshow(image, cmap="magma", origin="lower")
            axes[row, column].set(xticks=[], yticks=[], title=f"{('Left' if side == -1 else 'Right')} · {title}")
    _labels(axes)
    _save(fig, axes, output / "fig2-2_shape_encoding")
    plt.close(fig)


def plot_order(model: dict, times: np.ndarray, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), sharex=True, constrained_layout=True)
    axes[0].plot(times * 1000, model[-1].order_global, color=BLUE, label="Left")
    axes[0].plot(times * 1000, model[1].order_global, color=RED, linestyle="--", label="Right")
    axes[0].set(title="Global order parameter", ylabel="Synchrony r(t)")
    for axis, channel, name in zip(axes[1:], (1, 2), ("F3-weighted", "F4-weighted"), strict=True):
        axis.plot(times * 1000, model[-1].order_regional[channel], color=BLUE)
        axis.plot(times * 1000, model[1].order_regional[channel], color=RED, linestyle="--")
        axis.set_title(name)
    for axis in axes:
        axis.axvline(0, color=GRAY, linewidth=0.7)
        axis.set_xlabel("Time from cue onset (ms)")
    axes[0].legend()
    _labels(axes)
    _save(fig, axes, output / "fig2-3_order_parameter")
    plt.close(fig)


def plot_erp_validation(erp: dict, prediction: dict, times: np.ndarray, output: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.5), sharex=True, constrained_layout=True)
    for task in (1, 2):
        for channel, name in enumerate(("Fz", "F3", "F4")):
            axis = axes[task - 1, channel]
            for side, color, label in ((-1, BLUE, "Left"), (1, RED, "Right")):
                item = erp[(task, side)]
                axis.plot(times * 1000, item["mean"][channel], color=color, label=f"{label} observed")
                axis.fill_between(times * 1000, item["low"][channel], item["high"][channel], color=color, alpha=0.13)
                axis.plot(times * 1000, prediction[side][channel], color=color, linestyle="--", linewidth=1.0, label=f"{label} model")
            axis.axvline(0, color=GRAY, linewidth=0.7)
            axis.axhline(0, color=GRAY, linewidth=0.5)
            axis.set_title(f"Task {task} · {name}")
            if channel == 0:
                axis.set_ylabel("Amplitude (µV)")
            if task == 2:
                axis.set_xlabel("Time from cue onset (ms)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="outside upper center", fontsize=6)
    _labels(axes)
    _save(fig, axes, output / "fig2-4_model_validation")
    plt.close(fig)


def plot_mechanism(output: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.1), constrained_layout=True)
    texts = ("Mirror edge\narrangement", "Contralateral\nfeature map", "E-I population\nsynchrony", "Lateralized\nscalp potential")
    colors = (BLUE, TEAL, RED, GOLD)
    for axis, text, color in zip(axes, texts, colors, strict=True):
        axis.set(xlim=(0, 1), ylim=(0, 1))
        axis.axis("off")
        _box(axis, 0.08, 0.30, 0.84, 0.40, text, color)
    for axis in axes[:-1]:
        axis.annotate("", xy=(1.18, 0.5), xytext=(0.94, 0.5), xycoords="axes fraction", arrowprops={"arrowstyle": "-|>", "color": GRAY, "lw": 1.1})
    _labels(axes)
    _save(fig, axes, output / "fig2-5_mechanism_chain")
    plt.close(fig)


def plot_lateralization(dataset, window: np.ndarray, output: Path, rng: np.random.Generator) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True, constrained_layout=True)
    baseline = dataset.times < 0
    rms = np.sqrt(np.mean(dataset.epochs[:, 1:3, baseline] ** 2, axis=(1, 2)))
    lateral = (dataset.epochs[:, 2] - dataset.epochs[:, 1]) / np.maximum(rms[:, None], np.finfo(float).eps)
    for task, axis in zip((1, 2), axes, strict=True):
        for side, color, label in ((-1, BLUE, "Left"), (1, RED, "Right")):
            selected = lateral[(dataset.tasks == task) & (dataset.labels == side)]
            boot = selected[rng.integers(0, len(selected), size=(1000, len(selected)))].mean(axis=1)
            low, high = np.percentile(boot, [2.5, 97.5], axis=0)
            axis.plot(dataset.times * 1000, selected.mean(axis=0), color=color, label=f"{label}, n={len(selected)}")
            axis.fill_between(dataset.times * 1000, low, high, color=color, alpha=0.14)
        axis.axvspan(dataset.times[window][0] * 1000, dataset.times[window][-1] * 1000, color=GOLD, alpha=0.18)
        axis.axvline(0, color=GRAY, linewidth=0.7)
        axis.axhline(0, color=GRAY, linewidth=0.5)
        axis.set(title=f"Task {task}", xlabel="Time from cue onset (ms)")
        axis.legend(fontsize=6)
    axes[0].set_ylabel("Normalized lateral contrast (F4 − F3)")
    _labels(axes)
    _save(fig, axes, output / "fig2-6_lateralization")
    plt.close(fig)


def plot_classifier(results: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), constrained_layout=True)
    for name, item, color in (("Matched template", results["template"], TEAL), ("Template + lateral", results["combined"], RED)):
        axes[0].plot(item[1], item[2], color=color, label=f"{name} · AUC={item[0]['auc']:.2f}")
    axes[0].plot([0, 1], [0, 1], color=GRAY, linestyle="--")
    axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="Cross-file ROC")
    axes[0].legend(fontsize=6)
    axes[1].hist(results["combined"][3], bins=20, color="#D9D9D9", edgecolor="white")
    axes[1].axvline(results["combined"][0]["auc"], color=RED, linewidth=1.5, label="Observed AUC")
    axes[1].set(xlabel="Permuted AUC", ylabel="Count")
    axes[1].set_title("Within-file label permutation", pad=10)
    axes[1].legend(fontsize=6)
    _labels(axes)
    _save(fig, axes, output / "fig2-7_classifier")
    plt.close(fig)


def plot_controls(sham, injection, bounds, group_tests, real_correlation: float, real_auc: float, output: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.4), constrained_layout=True)
    names = []
    values = []
    errors = []
    colors = []
    for kernel, color in (("loose", "#BFBFBF"), ("tight", BLUE)):
        subset = sham[sham["kernel"] == kernel]
        for row in subset.itertuples():
            names.append(f"{kernel}\n{row.model}")
            values.append(row.waveform_correlation)
            errors.append(row.waveform_correlation_sd)
            colors.append(color)
    axes[0].bar(range(len(values)), values, yerr=errors, color=colors, width=0.6)
    axes[0].axhline(real_correlation, color=RED, linestyle="--", linewidth=1.0)
    axes[0].set(xticks=range(len(names)), xticklabels=names, ylabel="Held-out correlation", ylim=(0, 1.05))
    axes[0].set_title("Real model (red line) vs shams", fontsize=7)
    axes[0].tick_params(axis="x", labelsize=5)

    axes[1].plot(injection["injected_amplitude_rms"], injection["auc"], marker="o", color=TEAL, markersize=3)
    axes[1].axhline(0.70, color=GRAY, linestyle="--", linewidth=0.8)
    axes[1].axhline(real_auc, color=RED, linestyle=":", linewidth=1.0)
    axes[1].set(xlabel="Injected lateral amplitude (RMS units)", ylabel="Cross-file AUC", ylim=(0.3, 1.0))
    axes[1].set_title("Injection sensitivity", fontsize=7)

    for (lock, task), color, style in zip(sorted({(row.lock, row.task) for row in bounds.itertuples()}), (BLUE, RED, TEAL, GOLD), ("-", "--", "-.", ":")):
        subset = bounds[(bounds["lock"] == lock) & (bounds["task"] == task)]
        axes[2].plot(subset["window_start_s"] + 0.05, subset["auc_upper_bound"], marker="o", markersize=2.5, color=color, linestyle=style, label=f"{lock} T{task}")
    axes[2].axhline(0.60, color=GRAY, linestyle="--", linewidth=0.8)
    axes[2].set(xlabel="Window centre (s)", ylabel="AUC upper bound", ylim=(0.5, 0.7))
    axes[2].set_title("Decodability bound", fontsize=7)
    axes[2].legend(fontsize=5, ncol=2)

    order = group_tests.reset_index(drop=True)
    axes[3].bar(range(len(order)), order["max_abs_d"], color=[RED if value else "#BFBFBF" for value in order["significant_clusters"]], width=0.7)
    axes[3].set(xticks=range(len(order)), ylabel="Max |Cohen d|")
    axes[3].set_xticklabels([f"{row.lock[0]}{int(row.task)}" for row in order.itertuples()], fontsize=5, rotation=90)
    axes[3].set_title("Group-level lateral tests", fontsize=7)
    _labels(axes)
    _save(fig, axes, output / "fig2-8_controls")
    plt.close(fig)
