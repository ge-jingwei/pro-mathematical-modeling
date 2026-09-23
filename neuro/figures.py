"""Generate figures for Question 1 and Question 3 from saved v2 outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from neuro.dataset import load

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7})
BLUE, RED, GRAY, TEAL = "#0F4D92", "#B64342", "#767676", "#42949E"


def robust_mean(values, fraction=0.2):
    trim = int(values.shape[0] * fraction)
    ordered = np.sort(values, axis=0)
    return ordered[trim: values.shape[0] - trim].mean(axis=0)


def q1_figures(root: Path) -> None:
    output = root / "outputs" / "v2" / "q1" / "figures"
    output.mkdir(parents=True, exist_ok=True)
    dataset = load(root)
    times = dataset.times

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.6), constrained_layout=True)
    for row, task in enumerate((1, 2)):
        axis = axes[row]
        for side in (-1, 1):
            erp = robust_mean(dataset.cue[(dataset.task == task) & (dataset.cue_side == side)])
            axis.plot(times * 1000, erp[0], color=BLUE if side == -1 else RED, lw=1, label="left" if side == -1 else "right")
        axis.axvline(0, color=GRAY, lw=0.5)
        axis.axvspan(250, 500, color=TEAL, alpha=0.08)
        axis.set_title(f"project {task}: robust Fz ERP")
        axis.set_xlabel("time (ms)")
        axis.set_ylabel("amplitude (uV)")
        axis.legend(frameon=False, fontsize=6)
    fig.savefig(output / "fig1-1_erp.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    methods = pd.read_csv(root / "outputs" / "v2" / "q1" / "method_comparison.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)
    for index, metric in enumerate(("reliability", "fidelity")):
        axis = axes[index]
        for task, color in ((1, BLUE), (2, RED)):
            subset = methods[methods["task"] == task]
            axis.plot(subset["method"], subset[metric], "o-", color=color, lw=1, label=f"project {task}")
        axis.set_title(metric)
        axis.tick_params(axis="x", rotation=20)
        axis.legend(frameon=False, fontsize=6)
    fig.savefig(output / "fig1-2_methods.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def q3_figures(root: Path) -> None:
    output = root / "outputs" / "v2" / "q3" / "figures"
    output.mkdir(parents=True, exist_ok=True)
    dataset = load(root)
    rt1 = dataset.reaction_time[dataset.task == 1]
    rt2 = dataset.reaction_time[dataset.task == 2]

    fig, axis = plt.subplots(figsize=(3.4, 2.6), constrained_layout=True)
    axis.hist(rt1, bins=20, alpha=0.6, color=BLUE, label="project 1")
    axis.hist(rt2, bins=20, alpha=0.6, color=RED, label="project 2")
    axis.axvline(rt1.mean(), color=BLUE, lw=1)
    axis.axvline(rt2.mean(), color=RED, lw=1)
    axis.set_xlabel("reaction time (s)")
    axis.set_ylabel("trials")
    axis.legend(frameon=False, fontsize=6)
    fig.savefig(output / "fig3-1_rt.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(3.6, 2.6), constrained_layout=True)
    for subject, task, color, label in (("A", 1, BLUE, "A p1"), ("A", 2, RED, "A p2"), ("B", 1, TEAL, "B p1"), ("B", 2, GRAY, "B p2")):
        payload = np.load(root / "outputs" / "v2" / "q3" / f"synchrony_{subject}_task{task}.npz")
        axis.plot(payload["grid"] * 1000, payload["curves"].mean(axis=0), color=color, lw=1, label=label)
    axis.set_xlabel("time to response (ms)")
    axis.set_ylabel("theta order parameter R")
    axis.legend(frameon=False, fontsize=6)
    fig.savefig(output / "fig3-2_synchrony.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(3.6, 2.6), constrained_layout=True)
    times = dataset.times
    for project, color in ((1, BLUE), (2, RED)):
        payload = np.load(root / "outputs" / "v2" / "q3" / f"kuramoto_project{project}.npz")
        axis.plot(times * 1000, payload["order"], color=color, lw=1, label=f"project {project}")
    axis.set_xlabel("time (ms)")
    axis.set_ylabel("order parameter R")
    axis.legend(frameon=False, fontsize=6)
    fig.savefig(output / "fig3-3_order.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    q1_figures(root)
    q3_figures(root)
    print("figures written")


if __name__ == "__main__":
    main()
