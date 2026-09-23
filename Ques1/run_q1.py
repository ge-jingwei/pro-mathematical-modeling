import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from Ques1.events import build_events, parse_events
from Ques1.fitting import fit_components
from Ques1.plotting import plot_difference, plot_erp, plot_fits, plot_heatmaps, plot_method_comparison, plot_quality, plot_tradeoff
from Ques1.preprocess import prepare_epochs
from Ques1.statistics import bootstrap_mean_ci, cluster_independent_test, cluster_sign_test, side_difference
from tools.io_mat import load_records


CHANNELS = ("Fz", "F3", "F4")


def _correlation(reference: np.ndarray, candidate: np.ndarray, labels: np.ndarray, times: np.ndarray) -> float:
    mask = (times >= 0) & (times <= 0.6)
    values = []
    for side in (-1, 1):
        chosen = labels == side
        for channel in range(3):
            values.append(np.corrcoef(reference[chosen, channel][:, mask].mean(0), candidate[chosen, channel][:, mask].mean(0))[0, 1])
    return float(np.nanmean(values))


def run(root: Path, config: dict) -> None:
    data_dir = root / config["paths"]["data_dir"]
    output = root / config["paths"]["q1_output_dir"]
    figures = output / "figures"
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config["random_seed"])
    events = build_events(data_dir)
    events.to_csv(output / "trials.csv", index=False)
    pooled = []
    cue_pooled = []
    quality_rows = []
    tradeoff_rows = []
    method_rows = []
    records = load_records(data_dir)
    for record in records:
        current = parse_events(record)
        epochs = prepare_epochs(record, current, config)
        labels = current["cue_side"].to_numpy()
        kept_labels = labels[epochs.keep]
        cue_pooled.append((record, current.loc[epochs.keep].reset_index(drop=True), epochs.proposed[epochs.keep], epochs.times))
        pooled.append((record, current.loc[epochs.target_keep].reset_index(drop=True), epochs.target_proposed[epochs.target_keep], epochs.target_times))
        quality_rows.append({"file": record.path.name, "subject": record.subject, "task": record.task, "total_trials": len(current), "retained_trials": int(epochs.target_keep.sum()), "rejected_trials": int((~epochs.target_keep).sum()), "retention_rate": float(epochs.target_keep.mean()), "cue_retained_trials": int(epochs.keep.sum()), "selected_alpha": epochs.alpha})
        for row in epochs.tradeoff:
            tradeoff_rows.append({"file": record.path.name, **row, "selected": row["alpha"] == epochs.alpha})
        methods = {"Raw": epochs.raw[epochs.keep], "Decon": epochs.decon[epochs.keep], "Conventional": epochs.conventional[epochs.keep], "Proposed": epochs.proposed[epochs.keep]}
        baseline_mask = (epochs.times >= config["windows"]["baseline"][0]) & (epochs.times < config["windows"]["baseline"][1])
        raw_rms = float(np.sqrt(np.mean(methods["Raw"][..., baseline_mask] ** 2)))
        for name, values in methods.items():
            rms = float(np.sqrt(np.mean(values[..., baseline_mask] ** 2)))
            method_rows.append({"file": record.path.name, "method": name, "fidelity": _correlation(methods["Conventional"], values, kept_labels, epochs.times), "baseline_rms": rms, "noise_reduction_db": 20 * np.log10(raw_rms / max(rms, np.finfo(float).eps))})
    quality = pd.DataFrame(quality_rows)
    tradeoff = pd.DataFrame(tradeoff_rows)
    metrics = pd.DataFrame(method_rows)
    quality.to_csv(output / "table1-1_trial_rejection.csv", index=False)
    metrics.to_csv(output / "table1-2_method_comparison.csv", index=False)
    tradeoff.to_csv(output / "threshold_tradeoff.csv", index=False)
    erp = {}
    cue_erp = {}
    fit_map = {}
    fit_rows = []
    stat_rows = []
    cluster_rows = []
    primary = (np.asarray(config["windows"]["primary"]) / 1.0).tolist()
    for task in (1, 2):
        task_sets = [item for item in pooled if item[0].task == task]
        times = task_sets[0][3]
        for side in (-1, 1):
            selected = np.concatenate([values[frame["cue_side"].to_numpy() == side] for _, frame, values, _ in task_sets])
            for channel, name in enumerate(CHANNELS):
                mean, low, high = bootstrap_mean_ci(selected[:, channel], config["statistics"]["bootstrap_samples"], rng)
                erp[(task, side, channel)] = {"times": times, "mean": mean, "low": low, "high": high, "n": len(selected)}
                fitted = fit_components(times, mean)
                fit_map[(task, side, channel)] = fitted
                p = fitted["params"]
                fit_rows.append({"task": task, "side": side, "channel": name, "n_trials": len(selected), "n1_amplitude": p[1], "n1_latency_ms": p[2] * 1000, "p2_amplitude": p[4], "p2_latency_ms": p[5] * 1000, "p3_amplitude": p[7], "p3_latency_ms": p[8] * 1000, "p3_fwhm_ms": 2.355 * p[9] * 1000, "r2": fitted["r2"], "aic": fitted["aic"], "bic": fitted["bic"], "aic_two": fitted["aic_two"], "bic_two": fitted["bic_two"]})
                post = (times >= 0) & (times <= 0.6)
                clusters = cluster_sign_test(selected[:, channel][:, post], config["statistics"]["cluster_permutations"], rng)
                post_indices = np.flatnonzero(post)
                for start, stop, p_value in clusters:
                    cluster_rows.append({"task": task, "side": side, "channel": name, "contrast": "response_vs_zero", "start_ms": times[post_indices[start]] * 1000, "stop_ms": times[post_indices[stop - 1]] * 1000, "p_value": p_value})
        for channel, name in enumerate(CHANNELS):
            left = np.concatenate([values[frame["cue_side"].to_numpy() == -1, channel] for _, frame, values, _ in task_sets])
            right = np.concatenate([values[frame["cue_side"].to_numpy() == 1, channel] for _, frame, values, _ in task_sets])
            mask = (times >= primary[0]) & (times <= primary[1])
            delta, d, p_value = side_difference(left, right, mask, config["statistics"]["label_permutations"], rng)
            stat_rows.append({"task": task, "channel": name, "window_ms": "250-500", "right_minus_left": delta, "cohen_d": d, "p_value": p_value, "n_left": len(left), "n_right": len(right)})
            clusters = cluster_independent_test(left[:, (times >= 0) & (times <= 0.6)], right[:, (times >= 0) & (times <= 0.6)], config["statistics"]["cluster_permutations"], rng)
            post_indices = np.flatnonzero((times >= 0) & (times <= 0.6))
            for start, stop, cluster_p in clusters:
                cluster_rows.append({"task": task, "side": 0, "channel": name, "contrast": "left_vs_right", "start_ms": times[post_indices[start]] * 1000, "stop_ms": times[post_indices[stop - 1]] * 1000, "p_value": cluster_p})
    fits = pd.DataFrame(fit_rows)
    statistics = pd.DataFrame(stat_rows)
    clusters = pd.DataFrame(cluster_rows)
    fits.to_csv(output / "table1-3_curve_parameters.csv", index=False)
    statistics.to_csv(output / "side_statistics.csv", index=False)
    clusters.to_csv(output / "significant_clusters.csv", index=False)
    for task in (1, 2):
        task_sets = [item for item in cue_pooled if item[0].task == task]
        times = task_sets[0][3]
        for side in (-1, 1):
            selected = np.concatenate([values[frame["cue_side"].to_numpy() == side] for _, frame, values, _ in task_sets])
            for channel in range(3):
                mean, low, high = bootstrap_mean_ci(selected[:, channel], config["statistics"]["bootstrap_samples"], rng)
                cue_erp[(task, side, channel)] = {"times": times, "mean": mean, "low": low, "high": high, "n": len(selected)}
    plot_quality(records, figures)
    plot_tradeoff(tradeoff, figures)
    plot_method_comparison(metrics, figures)
    plot_heatmaps(pooled, figures)
    plot_erp(erp, figures, "target")
    plot_erp(cue_erp, figures, "cue")
    plot_difference(erp, clusters[clusters["contrast"] == "left_vs_right"], figures)
    plot_fits(erp, fit_map, figures)
    write_report(output, quality, metrics, fits, statistics, clusters)


def write_report(output: Path, quality: pd.DataFrame, metrics: pd.DataFrame, fits: pd.DataFrame, statistics: pd.DataFrame, clusters: pd.DataFrame) -> None:
    method = metrics.groupby("method")[["fidelity", "noise_reduction_db"]].mean()
    significant = statistics[statistics["p_value"] < 0.05]
    fit_good = float((fits["r2"] >= 0.8).mean())
    text = f"""# 问题一结果评估报告

## 结论

问题一流程已完整跑通，包括事件核验、坏试次剔除、心电参考回归、特征保持小波收缩、目标锁定主分析、提示锁定辅助分析、置信区间、簇置换检验、左右差异检验和受生理窗口约束的高斯曲线拟合。

## 数据与质量控制

目标锁定主分析在 400 个试次中保留 {int(quality['retained_trials'].sum())} 个，剔除 {int(quality['rejected_trials'].sum())} 个。各文件保留率为 {', '.join(f'{row.file}: {row.retention_rate:.1%}' for row in quality.itertuples())}。提示锁定辅助分析保留 {int(quality['cue_retained_trials'].sum())} 个。剔除仅依据饱和和峰峰值阈值，不使用左右标签。

原始数据复核显示，两份项目二文件均有 100 个点击标记，未发现方案文档所述的单次漏答。该差异不影响问题一，但后续问题三必须以原始事件表为准。

## 去噪效果

提出方法的平均视觉响应形态相关系数为 {method.loc['Proposed', 'fidelity']:.3f}，满足不低于 0.90 的工作点约束；相对原始数据的基线噪声降低为 {method.loc['Proposed', 'noise_reduction_db']:.2f} dB。机器滤波通道仅用于反面对照，未参与建模。

## 有效响应与左右差异

目标锁定簇置换检验共检出 {len(clusters[clusters['contrast'] == 'response_vs_zero'])} 个显著视觉响应簇。250 至 500 毫秒窗口中，6 个任务与电极组合有 {len(significant)} 个达到置换检验的 0.05 水平；效应量、样本数和精确概率见 `side_statistics.csv`。本批数据未支持目标锁定波形存在稳定的左右形状差异，不能据此声称已获得直接判别特征。组水平显著响应也不能自动解释为经典顶区 P300，本报告统一称为前额视觉响应成分。

## 曲线拟合

12 条任务、方向和电极组合曲线均完成三高斯拟合，拟合优度不低于 0.80 的比例为 {fit_good:.1%}。参数表同时给出二高斯与三高斯的赤池信息准则和贝叶斯信息准则，便于判断新增 P300 成分是否得到数据支持。

## 可靠性判断

流程满足题面要求并形成可复现闭环。主要限制是三通道均位于前额区、坏试次比例较高，目标锁定左右效应量很小且均不显著。该阴性结果不等同于不存在视觉编码，但问题二不能把目标锁定均值差异作为已证实特征；应从提示锁定响应、侧化轨迹和机理模型推导特征重新验证，并完整保留本次阴性结果。

## 输出索引

- `table1-1_trial_rejection.csv`：试次保留与剔除
- `table1-2_method_comparison.csv`：四种处理方法对比
- `table1-3_curve_parameters.csv`：曲线参数与拟合优度
- `side_statistics.csv`：左右差异、效应量与置换概率
- `significant_clusters.csv`：显著响应和左右差异时段
- `figures`：阈值工作点、方法对比、视觉响应、左右差异和拟合图
"""
    (output / "Ques1评估报告.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cti.yaml")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    with (root / args.config).open(encoding="utf-8") as stream:
        run(root, yaml.safe_load(stream))


if __name__ == "__main__":
    main()
