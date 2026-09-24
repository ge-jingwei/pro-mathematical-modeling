"""Compact, source-linked plots and a factual Chinese report."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Ques3.utils import table, dump, CHANNELS
from plotting import save

COLORS = ["#0072B2", "#CC79A7", "#D55E00"]


def grid(rows=1, cols=1, height=None):
    fig, axes = plt.subplots(rows, cols, figsize=(10 if cols > 1 else 6.4, height or (3.8 if rows == 1 else 6.7)), squeeze=False)
    fig.subplots_adjust(left=.09 if cols > 1 else .13, right=.97, bottom=.17 if rows == 1 else .11, top=.80 if rows == 1 else .87, wspace=.35, hspace=.62)
    return fig, axes


def make_figures(trials, predictions, trajectories, choice, timing, confusion, eeg, out):
    figures = out / "figures"
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
    pd.DataFrame(selected).to_csv(out / "tables/representative_trials.csv", index=False)
    pd.DataFrame(sources).to_csv(out / "tables/representative_trajectory_data.csv", index=False)
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
    pd.DataFrame(mean_rows).to_csv(out / "tables/mean_trajectory_data.csv", index=False)
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
    dump(out / "model/figure_manifest.json", dict(figures=5, selection_rule="first EEG-eligible held-out trial in original order",
         absent_no_response_category=True, task1_omissions_not_plotted=True, arbitrary_units=True,
         source_tables=["representative_trajectory_data.csv", "mean_trajectory_data.csv", "confusion_matrix.csv", "out_of_fold_predictions.csv", "eeg_metrics.csv"]))


def write_report(trials, predictions, choice, timing, eeg, checks, out):
    pooled = choice[choice.fold == "pooled"].iloc[0]
    times = timing[(timing.fold == "pooled") & (timing.model == "threshold_crossed_only")].iloc[0]
    baseline_time = timing[(timing.fold == "pooled") & (timing.model == "training_median_baseline")].iloc[0]
    full = eeg[eeg.scope == "full_pre_response"]
    comparison = full.groupby("model")[["nrmse", "r2", "pearson"]].mean().reset_index()
    paired = full.pivot(index=["fold", "channel"], columns="model", values="r2")
    improved = int(sum(paired.q2_plus_states > paired.q2_only))
    counts = pd.read_csv(out / "tables/behavior_counts.csv")
    eeg_n = int(predictions.eeg_eligible.sum())
    choice_table = choice[["fold", "accuracy", "balanced_accuracy", "macro_f1", "recall_left", "recall_right", "threshold_accuracy", "threshold_response_coverage"]].rename(columns=dict(fold="验证方向", accuracy="二分类准确率", balanced_accuracy="二分类平衡准确率", macro_f1="宏平均调和分数", recall_left="左召回率", recall_right="右召回率", threshold_accuracy="阈值准确率", threshold_response_coverage="阈值覆盖率"))
    time_table = timing.rename(columns=dict(fold="验证方向", model="方法", actual_responses="真实应答数", evaluated="评价数", coverage="覆盖率", mae="平均绝对误差", rmse="均方根误差", spearman="秩相关"))
    eeg_table = comparison.rename(columns=dict(model="模型", nrmse="归一化均方根误差", r2="决定系数", pearson="线性相关"))
    text = f"""# 问题三最小认知模型报告

## 一、结论

三个阶段已完成，保留真实事件、源空间证据、隐藏状态、双向留组预测及五组检查图。行为二分类的合并准确率为{pooled.accuracy:.2%}、平衡准确率为{pooled.balanced_accuracy:.2%}；真实阈值应答准确率为{pooled.threshold_accuracy:.2%}，阈值覆盖率为{pooled.threshold_response_coverage:.2%}。不能据此宣称已经建立有效的单试次认知解码器。

有越阈结果的{int(times.evaluated)}个试次，提示到应答时间间隔的平均绝对误差为{times.mae:.4f}秒，均方根误差为{times.rmse:.4f}秒，秩相关为{times.spearman:.4f}。训练记录中位数基线的平均绝对误差为{baseline_time.mae:.4f}秒。未越阈试次不填造应答时间；时间成绩必须与覆盖率一起阅读，不能作为全部二百次均已预测成功。

应答前脑电的{improved}/6个“测试记录×通道”比较中，扩展模型的决定系数高于问题二模型。改进仅证明增加的低维状态对离线重建有帮助，不证明其来自海马，也不证明认知机制已被验证。

## 二、真实数据与标签边界

四条记录共四百次提示；任务一二百次均无点击编码，只用于视觉对照。任务二二百次均存在真实点击，其中九十五次提示方向与点击方向一致、一百零五次不一致。客观提示、实际点击、模型内部状态分别保存。

“正确／错误”严格使用本次要求的方向一致性定义。题目任务二涉及提示形状与目标位置的匹配，因此方向不一致不必意味着真实任务错误；没有完整答案键，不能据此报告实验正确率或诊断认知障碍。

由于无应答试次极度缺乏，模型无法稳定验证无应答机制，仅对左右应答方向进行验证。具体而言，任务二无应答为零；任务一记录缺失不计入遗漏类别，不画伪造的无应答轨迹。

每个有应答窗口为提示至点击前一百毫秒，离散终点向下取整。缺失应答窗口按要求使用提示后二秒的人为假定，受下一提示或记录结束限制。这不是经证实的实验超时；不能用于否定实际三至四秒后的点击。后续正负一标记语义未确认，不将其当作已确认目标时间，全文使用提示到应答时间间隔。

## 三、感知、记忆与决策模型

复用问题一清洗方法及问题二固定早期片段。每折加载仅由该训练记录拟合的问题二动力学和混合矩阵，核对其训练编号；随后重新按该折训练记录校准证据尺度。单试次反演调用问题二既有源投影函数，输入只有脑电和任务编号，没有提示方向、点击方向或点击时间。

感知证据为右源减左源，经训练尺度标准化。前三电极的直接波形差未被用作证据；源反演也不能凭空恢复已经丢失的方向信息。先收集固定约零点八秒脑电，之前输入证据设为零，之后的源曲线由问题二动力学外推，不是实测的晚期源活动。头皮反对称增益弱意味着反演可能不稳定，不保证性能提高。

记忆状态表示海马记忆匹配功能的潜在状态：

`tau_m * dm/dt = -m + e`

决策状态满足：

`tau_z * dz/dt = -z + w_e*e + w_m*m + bias`

每试次初始记忆和决策均为零；记忆输入尺度固定为一，任务二先验为零。跨阈值首次决定方向与时间；未跨阈值为模型无应答。二分类读出另行保存，跨阈值时取首越阈方向，未跨阈值时才取截止状态符号；这不等于生成了真实阈值应答，混淆矩阵分别展示两者。比例归一化固定决策驱动向量范数，避免共同缩放不可辨识。

总计六个标量参数，其中整体尺度受约束；每折一百六十个候选加一次局部精调，不做测试集调参。参数搜索保留全部候选损失；局部优化保留实际访问过的最优点，不假定终点一定优于初值。样本少、反演不稳定及参数边界均限制生理解释。

在问题二重建上加入三个通道各自的记忆和决策线性映射；映射通过训练折岭回归估计。认知拟合损失由方向错配、平滑方向损失、时间误差及弱阈值匹配组成，不联合强行最优化脑电。

## 四、训练测试隔离与覆盖率

甲组训练乙组测试、乙组训练甲组测试，每折一百次行为训练、一百次独立行为测试；合并二百次测试且每次只出现一次。记录组不等于已确认的两个被试，故不声称跨被试泛化。

问题二参数复用同一训练组既有结果；源尺度、认知参数、岭回归权重及预测期限均只依赖该训练组。预测期限为训练点击最大时间加零点七五秒，再向上取整至六十四赫兹网格；不以测试点击时间裁剪决策状态。真实点击时间只用于训练标签及测试评价窗口。所有行为样本均保留，包含方向不一致试次。

脑电评价按固定饱和、非有限与既有硬质控规则保留{eeg_n}/200试次。各折具体覆盖见检查表，未按正确性或误差选择试次；不合格试次仍参与行为验证。脑电的后期扩展调用问题一原有清洗函数，不重新实现问题一或问题二。

重要限制：既有片段使用零相位块滤波，扩展脑电使用各记录独立的零相位全记录滤波，可能含点击附近的滤波拖尾。训练、测试记录从不混合，但这是离线解码与重建，不能当作严格在线因果预测。源反演也已经使用测试试次早期脑电，因此全窗脑电重建不是完全独立的生成预测；另保存证据窗结束后的重建成绩用于透明比较，仍不消除零相位处理的非因果性。

## 五、应答方向验证

{table(choice_table)}

二分类混淆矩阵以及保留未越阈列的阈值混淆矩阵同时交付。准确率不能单独反映偏向某一方向的退化解，需结合平衡准确率和左右召回率。模型无应答预测若出现在实际有应答试次中，是模型失败而不是发现真实遗漏。

## 六、提示到应答时间间隔验证

{table(time_table)}

误差单位为秒。无越阈时刻保留缺失；未以零、均值或截止时间补造点。训练中位数仅是明确标记的参照，不计作模型首次越阈时间。

## 七、脑电重建比较

以下是两折三个通道等权平均，非把所有采样点混成单个通道。归一化误差以相同测试目标的标准差归一化，决定系数使用相同测试窗的中心化平方和，相关系数为线性相关。逐折逐通道及后期窗口完整指标另存。

{table(eeg_table)}

增加状态等于增加解释自由度；此处只完成用户要求的一项比较，没有额外大规模消融或机制识别。不能把映射权重视为解剖脑源定位，也不能把负面行为结果隐藏在改善的脑电曲线后。

## 八、三类行为的模型解释

- 正确：按操作定义，真实提示与真实点击一致；模型若首越同侧阈值，则与该行为相符。代表性图按原始顺序选第一个脑电合格的测试试次，不按拟合优劣挑选。
- 错误：真实点击与提示方向不一致；模型反向首越阈值可作为低维解释，但弱感知证据、总体偏向、任务形状与位置差异都可能产生相同表象，无法辨认真实生理病因。
- 无应答：模型未在训练确定的观察期限内越阈；这是一项模型假设。任务二无真实遗漏样本，因此无应答机制、灵敏度和三分类效果均不可验证；任务一没有记录点击不能充当证据。

## 九、交付与复现

交付包含行为表、两折完整源及状态数组、逐试次真实阈值预测、独立二分类读出、参数和搜索损失、训练编号、脑电映射、方向与时间指标、脑电比較及五组图形。源轨迹文件也保留任务一视觉对照，任务一记忆与决策数组明确留空。固定随机种子为二零二六零九二四。

统一入口为 `python problem3/run_problem3.py`，默认通过现有远程工具运行；数据处理和模型拟合在指定环境及空闲显存较多的显卡上执行。已有输出禁止静默覆盖，明确复现需新输出目录，阶段续跑需显式指定续跑选项。
"""
    (out / "report.md").write_text(text, encoding="utf-8")
    (out / "stage3_check.md").write_text(f"# 阶段三检查\n\n状态：运行与隔离检查通过；不代表模型效果达标。\n\n两折各一百个测试试次，共{len(predictions)}次且编号唯一。二分类准确率{pooled.accuracy:.2%}、平衡准确率{pooled.balanced_accuracy:.2%}。真实阈值准确率{pooled.threshold_accuracy:.2%}；时间指标覆盖{int(times.evaluated)}/200次。脑电覆盖{eeg_n}/200次，六项通道比较中{improved}项决定系数改善。无真实遗漏，未输出伪造三分类成绩。\n\n全表、模型数组、参数和图形源表已保存；性能不足与滤波非因果性均见主报告。\n", encoding="utf-8")
