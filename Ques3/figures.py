"""Compact, source-linked plots and a factual Chinese report."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Ques3.utils import dump, CHANNELS
from Ques1.src.plotting import save

COLORS = ["#0072B2", "#CC79A7", "#D55E00"]


def grid(rows=1, cols=1, height=None):
    fig, axes = plt.subplots(rows, cols, figsize=(10 if cols > 1 else 6.4, height or (3.8 if rows == 1 else 6.7)), squeeze=False)
    fig.subplots_adjust(left=.09 if cols > 1 else .13, right=.97, bottom=.17 if rows == 1 else .11, top=.80 if rows == 1 else .87, wspace=.35, hspace=.62)
    return fig, axes


def make_figures(trials, predictions, trajectories, choice, timing, confusion, eeg, out):
    figures = out
    sources, selected = [], []
    categories = [("正确左应答", (predictions.response_status == "correct") & (predictions.response_direction == -1)),
                  ("正确右应答", (predictions.response_status == "correct") & (predictions.response_direction == 1)),
                  ("错误应答", predictions.response_status == "incorrect")]
    available = [(label, predictions[mask & predictions.eeg_eligible]) for label, mask in categories]
    available = [(label, subset) for label, subset in available if len(subset)]
    fig, axes = grid(cols=len(available), height=4.2)
    for ax, (label, subset) in zip(axes.flat, available):
        row = subset.sort_values("row_id").iloc[0]
        state = trajectories[row.fold]
        t = state["time"]
        use = t <= row.prediction_horizon
        i = int(row.row_id)
        selected.append(dict(category=label, row_id=i, record=row.record, trial_id=int(row.trial_id), selection="first_eligible_test_trial"))
        for name, key, color in zip(["感知证据", "记忆状态", "决策状态"], ["evidence", "memory", "decision"], COLORS):
            ax.plot(t[use], state[key][i, use], label=name, color=color)
            sources.extend(dict(row_id=i, time_s=float(tt), variable=key, value=float(v)) for tt, v in zip(t[use], state[key][i, use]))
        ax.axhline(row.decision_threshold, color=".45", ls=":", lw=1, label="决策阈值")
        ax.axhline(-row.decision_threshold, color=".45", ls=":", lw=1)
        ax.axvline(row.cue_to_response_latency, color=".25", ls="--", lw=1.2, label="实际点击")
        if np.isfinite(row.predicted_crossing_time):
            ax.axvline(row.predicted_crossing_time, color="#009E73", ls="-.", lw=1.2, label="预测越阈")
        ax.set(title=f"{label} · 试次{int(row.trial_id)}", xlabel="提示后时间（秒）", ylabel="模型状态（任意单位）")
    handles, labels = [], []
    for ax in axes.flat:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in labels:
                handles.append(handle); labels.append(label)
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(.5, .995))
    save(fig, figures / "representative_trajectories")
    pd.DataFrame(selected).to_csv(out / "representative_trials.csv", index=False)
    pd.DataFrame(sources).to_csv(out / "representative_trajectory_data.csv", index=False)
    mean_rows = []
    fig, axes = grid(rows=2, cols=2)
    for r, (fold, state) in enumerate(trajectories.items()):
        p = state["predictions"]
        for c, (status, label) in enumerate([("correct", "正确"), ("incorrect", "错误")]):
            ax = axes[r, c]
            ids = p.loc[p.response_status == status, "row_id"].to_numpy(int)
            horizon = float(p.prediction_horizon.iloc[0])
            use = state["time"] <= horizon
            for key, name, color in zip(["evidence", "memory", "decision"], ["感知证据", "记忆状态", "决策状态"], COLORS):
                mean = state[key][ids][:, use].mean(0)
                ax.plot(state["time"][use], mean, color=color, label=name)
                mean_rows.extend(dict(fold=fold, response_status=status, n=len(ids), time_s=float(t), variable=key, mean=float(v)) for t, v in zip(state["time"][use], mean))
            ax.axhline(state["params"][-1], color=".55", ls=":", lw=.8)
            ax.axhline(-state["params"][-1], color=".55", ls=":", lw=.8)
            direction = "甲组训练、乙组测试" if r == 0 else "乙组训练、甲组测试"
            ax.set(title=f"{direction} · {label} · {len(ids)}次", xlabel="提示后时间（秒）", ylabel="模型状态（任意单位）")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=3)
    save(fig, figures / "mean_trajectories")
    pd.DataFrame(mean_rows).to_csv(out / "mean_trajectory_data.csv", index=False)
    fig, axes = grid(cols=2, height=4.0)
    for ax, mode, labels, title in zip(axes.flat, ["binary_readout", "threshold"], [[-1, 1], [-1, 0, 1]], ["二分类读出", "真实阈值应答"]):
        cm = confusion[(confusion.fold == "pooled") & (confusion.classifier == mode)].pivot(index="actual", columns="predicted", values="count").reindex(index=[-1, 1], columns=labels).to_numpy()
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(cm.max(), 1), aspect="auto")
        for i in range(2):
            for j in range(len(labels)):
                ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", fontsize=12, color="white" if cm[i, j] > cm.max() * .55 else "#222222")
        ax.set(xticks=range(len(labels)), xticklabels=[{-1:"左应答", 0:"未越阈", 1:"右应答"}[v] for v in labels],
               yticks=[0, 1], yticklabels=["左应答", "右应答"], xlabel="模型预测", ylabel="实际点击", title=title)
    fig.suptitle("双向留组预测合并 · 二百个真实应答", y=.98)
    save(fig, figures / "confusion_matrix")
    fig, axes = grid()
    ax = axes[0, 0]
    for (fold, p), color in zip(predictions.groupby("fold"), COLORS):
        valid = p.predicted_crossing_time.notna()
        label = "甲组训练、乙组测试" if fold == "A_to_B" else "乙组训练、甲组测试"
        ax.scatter(p.loc[valid, "cue_to_response_latency"], p.loc[valid, "predicted_crossing_time"], color=color, s=17, alpha=.7, label=label)
    predicted = predictions.predicted_crossing_time.dropna()
    low = min(predicted.min(), predictions.cue_to_response_latency.min()) - .1
    high = max(predicted.max(), predictions.cue_to_response_latency.max()) + .1
    ax.plot([low, high], [low, high], color=".5", ls="--", lw=1)
    n = int(predictions.predicted_crossing_time.notna().sum())
    ax.set(xlabel="实际提示到应答时间间隔（秒）", ylabel="预测首次越阈时间（秒）", title=f"有越阈预测{n}次；未越阈{len(predictions)-n}次不填补")
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=2)
    save(fig, figures / "time_prediction_scatter")
    fig, axes = grid(cols=2, height=4.0)
    for ax, fold, title in zip(axes.flat, ["A_to_B", "B_to_A"], ["甲组训练、乙组测试", "乙组训练、甲组测试"]):
        p = eeg[(eeg.fold == fold) & (eeg.scope == "full_pre_response")]
        for model, name, color in zip(["q2_only", "q2_plus_states"], ["问题二模型", "增加认知状态"], COLORS):
            values = p[p.model == model].set_index("channel").loc[CHANNELS, "r2"].to_numpy()
            ax.plot(range(3), values, color=color, marker="o", label=name)
        ax.set(xticks=range(3), xticklabels=["左额", "中额", "右额"], xlabel="脑电通道", ylabel="决定系数", title=title)
        ax.axhline(0, color=".7", lw=.8)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=2)
    save(fig, figures / "eeg_reconstruction_comparison")
    dump(out / "figure_manifest.json", dict(figures=5, selection_rule="first EEG-eligible held-out trial in original order",
         absent_no_response_category=True, task1_omissions_not_plotted=True, arbitrary_units=True,
         source_tables=["representative_trajectory_data.csv", "mean_trajectory_data.csv", "confusion_matrix.csv", "out_of_fold_predictions.csv", "eeg_metrics.csv"]))


