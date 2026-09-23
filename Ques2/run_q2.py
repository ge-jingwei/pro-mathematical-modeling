import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from Ques2.data import erp_summary, load_cue_dataset
from Ques2.plotting import plot_architecture, plot_classifier, plot_erp_validation, plot_lateralization, plot_mechanism, plot_order, plot_shape_encoding
from Ques2.validation import calibrate, calibrated_prediction, classify, feature_matrix, fit_observation_kernel, lateral_window, observation_cross_validation, parameter_table, sensitivity


def write_report(output: Path, dataset, validation: pd.DataFrame, sensitivity_table: pd.DataFrame, classifiers: pd.DataFrame, start: float, stop: float) -> None:
    held_out = validation.query("task == 2").iloc[0]
    first_task = validation.query("task == 1").iloc[0]
    combined = classifiers.query("feature_set == 'Template + lateral'").iloc[0]
    stability = sensitivity_table.groupby("parameter")["sign_agreement"].min().mean()
    correlation_range = (sensitivity_table["waveform_correlation"].min(), sensitivity_table["waveform_correlation"].max())
    spectrum_range = (sensitivity_table["spectral_cosine"].min(), sensitivity_table["spectral_cosine"].max())
    text = f"""# 问题二结果评估报告

## 结论

问题二已形成从三角形空间特征、外侧膝状体至皮层延迟、兴奋抑制神经群动力学、同步序参数到头皮电位的完整正向计算链。模型没有把问题一中不显著的目标锁定左右差异当作既定事实；动力学参数仅由项目一确定，因果观测核另用五折留出试次检验。

模型在不参与拟合的留出试次上，对项目二提示锁定波形的平均相关系数为 {held_out.waveform_correlation:.3f}±{held_out.waveform_correlation_sd:.3f}，功率谱余弦相似度为 {held_out.spectral_cosine:.3f}±{held_out.spectral_cosine_sd:.3f}。左右差异符号在两侧电极上的一致率为 {held_out.sign_agreement:.1%}。这说明模型能够复现共同视觉响应形态，但侧化方向仍缺乏稳定数据支持。

项目一留出波形相关系数为 {first_task.waveform_correlation:.3f}±{first_task.waveform_correlation_sd:.3f}，明显低于项目二，说明已知目标位置的任务情境存在更强的跨试次变化。该差异被保留为模型适用边界，不用全样本拟合值替代。

## 数据与校准

分析沿用问题一冻结的特征保持预处理参数，共纳入 {len(dataset.labels)} 个提示锁定有效试次。形状标签严格取视觉提示通道，项目二的目标位置标签未参与形状判别。动力学参数只用项目一确定；因果观测核在两项目和两方向上统一进行五折交叉验证，每折均只用训练试次拟合。模型选择的侧化检验窗口为提示后 {start * 1000:.0f} 至 {stop * 1000:.0f} 毫秒。

## 机理解释

左右三角的边缘排列互为镜像，方向选择性滤波后形成镜像特征图；视网膜至皮层投影使该空间编码反转进入皮层神经场。兴奋抑制神经群将空间输入变为时变群体活动，区域加权序参数描述局部同步，头皮正向映射再把皮层活动投影至 Fz、F3 和 F4。严格对称条件下全局序参数近似不变，真正可区分左右形状的是区域加权同步及 F3 与 F4 的侧化组合，而不是人为规定两条不同的全局曲线。

## 特征判别

采用四折按文件分组交叉验证，每次整份留出一个被试与项目组合。模型匹配投影与侧化指标联合时，曲线下面积为 {combined.auc:.3f}，准确率为 {combined.accuracy:.3f}，文件内标签置换概率为 {combined.p_value:.4f}。本结果未显著优于随机，因此只能把该表示作为机理推导出的候选特征，不能声称当前三电极数据已经实现可靠的单试次左右判别。

## 灵敏度与可靠性

对传导延迟、空间耦合、视觉输入增益以及兴奋和抑制时间常数分别作上下 20% 扰动。波形相关系数范围为 {correlation_range[0]:.3f} 至 {correlation_range[1]:.3f}，谱相似度范围为 {spectrum_range[0]:.3f} 至 {spectrum_range[1]:.3f}，说明共同波形结论稳定；各参数最差符号一致率的平均值仅为 {stability:.1%}，说明侧化方向结论不稳定。主要限制是仅有三个前额电极，空间正向映射不可唯一辨识。

## 输出索引

- `table2-1_model_parameters.csv`：模型参数、含义与来源
- `table2-2_sensitivity.csv`：关键参数上下 20% 扰动
- `table2-3_validation.csv`：五折留出试次验证指标
- `classifier_performance.csv`：分组交叉验证与置换检验
- `calibration_grid.csv`：参数网格搜索记录
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
    prediction, kernels = fit_observation_kernel(model, erp, dataset.times)
    validation_table = observation_cross_validation(dataset, model, rate)
    sensitivity_table = sensitivity(parameters, kernels, erp, dataset.times, rate, config)
    combined, template = feature_matrix(dataset, mechanistic_prediction, window)
    results = {}
    classifier_rows = []
    for name, values, key in (("Matched template", template, "template"), ("Template + lateral", combined, "combined")):
        result = classify(values, dataset.labels, dataset.groups, config["q2"]["classifier_permutations"], rng)
        results[key] = result
        classifier_rows.append({"feature_set": name, **result[0]})
    parameters_table = parameter_table(parameters, kernels)
    classifiers = pd.DataFrame(classifier_rows)
    parameters_table.to_csv(output / "table2-1_model_parameters.csv", index=False)
    sensitivity_table.to_csv(output / "table2-2_sensitivity.csv", index=False)
    validation_table.to_csv(output / "table2-3_validation.csv", index=False)
    classifiers.to_csv(output / "classifier_performance.csv", index=False)
    grid.to_csv(output / "calibration_grid.csv", index=False)
    plot_architecture(figures)
    plot_shape_encoding(model, figures)
    plot_order(model, dataset.times, figures)
    plot_erp_validation(erp, prediction, dataset.times, figures)
    plot_mechanism(figures)
    plot_lateralization(dataset, window, figures, rng)
    plot_classifier(results, figures)
    write_report(output, dataset, validation_table, sensitivity_table, classifiers, start, stop)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cti.yaml")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    with (root / args.config).open(encoding="utf-8") as stream:
        run(root, yaml.safe_load(stream))


if __name__ == "__main__":
    main()
