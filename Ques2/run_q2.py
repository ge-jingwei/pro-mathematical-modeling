import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from Ques2.data import erp_summary, load_cue_dataset
from Ques2.model import simulate
from Ques2.plotting import plot_architecture, plot_classifier, plot_erp_validation, plot_lateralization, plot_mechanism, plot_order, plot_shape_encoding
from Ques2.validation import calibrate, calibrated_prediction, classify, decodability_bounds, feature_matrix, fit_observation_kernel, geometry_sensitivity, group_level_tests, lateral_window, model_diagnostics, observation_cross_validation, parameter_table, positive_controls, sensitivity


def write_report(output: Path, dataset, validation: pd.DataFrame, sensitivity_table: pd.DataFrame, classifiers: pd.DataFrame, sham: pd.DataFrame, direction: pd.DataFrame, geometry: pd.DataFrame, group_tests: pd.DataFrame, bounds: pd.DataFrame, controls: pd.DataFrame, lovo: pd.DataFrame, start: float, stop: float, maximum: int, alpha: float) -> None:
    held_out = validation.query("task == 2").iloc[0]
    first_task = validation.query("task == 1").iloc[0]
    combined = classifiers.query("feature_set == 'Template + lateral'").iloc[0]
    best_sham = sham.loc[sham["waveform_correlation"].idxmax()]
    delta = held_out.waveform_correlation - best_sham.waveform_correlation
    max_bound = bounds.loc[bounds["auc_upper_bound"].idxmax()]
    task_control = controls.loc[controls["control"] == "task1_vs_task2"].iloc[0]
    position_control = controls.loc[controls["control"] == "task1_target_position"].iloc[0]
    collapsed = direction.query("projection == 'collapsed'")["side_lobe_ratio"].max()
    directional = direction.query("projection == 'shape'")["side_lobe_ratio"].max()
    significant = int(group_tests["significant_clusters"].sum())
    geometry_stable = geometry["lateralization_sign"].nunique() == 1 and int(geometry["lateralization_sign"].iloc[0]) != 0
    best_lovo = lovo.loc[lovo["waveform_correlation"].idxmax()]
    text = f"""# 问题二结果评估报告

## 结论

本轮按方案重做了方向特征投影、导联场几何扰动、伪模型对照、组水平检验、阳性对照、效应量上界和LOVO稳健性分析。结果是否达标以本报告的数字为准，不把实现了代码当作通过验收。

项目二留出波形相关为 {held_out.waveform_correlation:.3f}±{held_out.waveform_correlation_sd:.3f}，谱余弦为 {held_out.spectral_cosine:.3f}±{held_out.spectral_cosine_sd:.3f}；左右侧化符号一致率为 {held_out.sign_agreement:.1%}。最佳伪模型的相关为 {best_sham.waveform_correlation:.3f}，真模型与其差值为 {delta:.3f}。因此，波形复现仅在差值和对照结果支持时成立；当前仍不能仅凭真模型相关系数宣称机理得到验证。

项目一留出相关为 {first_task.waveform_correlation:.3f}±{first_task.waveform_correlation_sd:.3f}。拟合核设置为最多 {maximum} 个滞后样本、岭惩罚 {alpha:g}；若初始伪模型相关达到0.85，流程自动收紧到60个滞后样本和惩罚1.0。

## 数据与校准

沿用问题一冻结的特征保持预处理，提示锁定纳入 {len(dataset.labels)} 个试次，目标锁定纳入 {len(dataset.target_labels)} 个试次。形状标签取提示符号；位置标签取目标位置符号，两者分别分析。模型侧化窗口为提示后 {start * 1000:.0f} 至 {stop * 1000:.0f} 毫秒。数据文件作为分组单位进行五折留出。

## 机理解释

方向通道分别投影后，F4−F3峰值与Fz峰值之比由方向平均基线的 {collapsed:.4f} 变为 {directional:.4f}。导联场使用反平方距离衰减近似，并对电极横向位置和深度作±20%扰动；扰动下侧化符号{ '稳定' if geometry_stable else '不稳定或为零' }。该导联场不是三层球模型，也没有个体头部几何数据，因此只能视为待验证的简化正向映射，不能称作生物物理精确头模型。

## 特征判别

提示锁定形状侧、目标锁定位置侧共完成 {len(group_tests)} 项组水平簇检验，显著簇总数为 {significant}。未经组水平显著性筛选的模型特征仅作探索性分类：联合特征AUC={combined.auc:.3f}、置换p={combined.p_value:.4f}。阳性对照中，项目一与项目二AUC={task_control.auc:.3f}、p={task_control.p_value:.4f}；项目一目标位置AUC={position_control.auc:.3f}、p={position_control.p_value:.4f}。阳性对照是否达到预设AUC≥0.70由这些数字判断；若未达到，阴性形状分类不能归因于数据本身。

最大窗口效应量为d={max_bound.cohens_d:.3f}，正态近似单试次AUC上界为{max_bound.auc_upper_bound:.3f}（{max_bound['lock']}锁定、项目{int(max_bound.task)}、{max_bound.window_start_s:.1f}至{max_bound.window_stop_s:.1f}秒）。该上界使用标签侧的单试次F4−F3指标计算，不能替代交叉验证结果。

## 灵敏度与可靠性

LOVO保留比例扫描的最高留出相关为{best_lovo.waveform_correlation:.3f}（保留{best_lovo.retained_fraction:.0%}试次），全试次基准为{held_out.waveform_correlation:.3f}；此处仅评价观测核训练均值的稳健性，不作为判别改善证据。按预注册口径，LOVO还须使符号一致率超过60%或置换p<0.05才算有效改进。

## 输出索引

- `table2-1_model_parameters.csv`：模型参数、含义与来源
- `table2-2_sensitivity.csv`：关键参数上下 20% 扰动
- `table2-3_validation.csv`：五折留出试次验证指标
- `classifier_performance.csv`：分组交叉验证与置换检验
- `calibration_grid.csv`：参数网格搜索记录
- `sham_comparison.csv`、`directional_projection.csv`、`geometry_sensitivity.csv`：伪模型、方向通道与导联几何验收
- `group_level_tests.csv`、`positive_controls.csv`、`decodability_upper_bounds.csv`：组水平前置检验、阳性对照和上界
- `lovo_comparison.csv`：LOVO保留比例扫描
- `figures`：模型结构、形状编码、序参数、真实数据验证、机制链、侧化轨迹与判别图
"""
    (output / "Ques2评估报告.md").write_text(text, encoding="utf-8")


def run(root: Path, config: dict) -> None:
    output = root / config["paths"]["q2_output_dir"]
    figures = output / "figures"
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config["random_seed"])
    dataset = load_cue_dataset(root, config)
    rate = float(config["sample_rate"])
    erp = erp_summary(dataset, config["statistics"]["bootstrap_samples"], rng)
    parameters, model, scales, grid = calibrate(erp, dataset.times, rate, config)
    mechanistic_prediction = calibrated_prediction(model, scales)
    window, start, stop = lateral_window(mechanistic_prediction, dataset.times)
    maximum, alpha = 160, 0.1
    validation_table = observation_cross_validation(dataset, model, rate, maximum=maximum, alpha=alpha)
    sham_rows = []
    sham_models = {}
    for mode in ("constant", "random"):
        sham_models[mode] = {side: simulate(side, dataset.times, rate, parameters, config["q2"]["grid_size"], drive_mode=mode, seed=20260923) for side in (-1, 1)}
        sham = observation_cross_validation(dataset, sham_models[mode], rate, maximum=maximum, alpha=alpha)
        row = sham.query("task == 2").iloc[0]
        sham_rows.append({"model": mode, **row.to_dict()})
    sham_table = pd.DataFrame(sham_rows)
    if sham_table["waveform_correlation"].max() >= 0.85:
        maximum, alpha = 60, 1.0
        validation_table = observation_cross_validation(dataset, model, rate, maximum=maximum, alpha=alpha)
        sham_rows = []
        for mode, sham_model in sham_models.items():
            sham = observation_cross_validation(dataset, sham_model, rate, maximum=maximum, alpha=alpha)
            row = sham.query("task == 2").iloc[0]
            sham_rows.append({"model": mode, **row.to_dict()})
        sham_table = pd.DataFrame(sham_rows)
    prediction, kernels = fit_observation_kernel(model, erp, dataset.times, maximum, alpha)
    sensitivity_table = sensitivity(parameters, kernels, erp, dataset.times, rate, config, maximum)
    direction_table, geometry_table = model_diagnostics(parameters, dataset.times, rate, config["q2"]["grid_size"])
    group_table = group_level_tests(dataset, config["statistics"]["cluster_permutations"], rng)
    bounds_table = decodability_bounds(dataset)
    control_table = positive_controls(dataset, config["q2"]["classifier_permutations"], rng)
    combined, template = feature_matrix(dataset, mechanistic_prediction, window)
    results = {}
    classifier_rows = []
    for name, values, key in (("Matched template", template, "template"), ("Template + lateral", combined, "combined")):
        result = classify(values, dataset.labels, dataset.groups, config["q2"]["classifier_permutations"], rng)
        results[key] = result
        classifier_rows.append({"feature_set": name, **result[0]})
    parameters_table = parameter_table(parameters, kernels)
    classifiers = pd.DataFrame(classifier_rows)
    lovo_rows = []
    for fraction in (0.6, 0.75, 0.9, 0.95):
        robust_cv = observation_cross_validation(dataset, model, rate, maximum=maximum, alpha=alpha, lovo_fraction=fraction)
        task2 = robust_cv.query("task == 2").iloc[0]
        lovo_rows.append({"retained_fraction": fraction, **task2.to_dict()})
    lovo_table = pd.DataFrame(lovo_rows)
    parameters_table.to_csv(output / "table2-1_model_parameters.csv", index=False)
    sensitivity_table.to_csv(output / "table2-2_sensitivity.csv", index=False)
    validation_table.to_csv(output / "table2-3_validation.csv", index=False)
    classifiers.to_csv(output / "classifier_performance.csv", index=False)
    grid.to_csv(output / "calibration_grid.csv", index=False)
    sham_table.to_csv(output / "sham_comparison.csv", index=False)
    direction_table.to_csv(output / "directional_projection.csv", index=False)
    geometry_table.to_csv(output / "geometry_sensitivity.csv", index=False)
    group_table.to_csv(output / "group_level_tests.csv", index=False)
    bounds_table.to_csv(output / "decodability_upper_bounds.csv", index=False)
    control_table.to_csv(output / "positive_controls.csv", index=False)
    lovo_table.to_csv(output / "lovo_comparison.csv", index=False)
    plot_architecture(figures)
    plot_shape_encoding(model, figures)
    plot_order(model, dataset.times, figures)
    plot_erp_validation(erp, prediction, dataset.times, figures)
    plot_mechanism(figures)
    plot_lateralization(dataset, window, figures, rng)
    plot_classifier(results, figures)
    write_report(output, dataset, validation_table, sensitivity_table, classifiers, sham_table, direction_table, geometry_table, group_table, bounds_table, control_table, lovo_table, start, stop, maximum, alpha)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cti.yaml")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    with (root / args.config).open(encoding="utf-8") as stream:
        run(root, yaml.safe_load(stream))


if __name__ == "__main__":
    main()
