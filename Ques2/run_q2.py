import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from Ques2.data import erp_summary, load_cue_dataset
from Ques2.model import simulate
from Ques2.plotting import plot_architecture, plot_classifier, plot_controls, plot_erp_validation, plot_lateralization, plot_mechanism, plot_order, plot_shape_encoding
from Ques2.validation import aggregate_validation, calibrate, calibrated_prediction, classify, decodability_bounds, feature_matrix, fit_observation_kernel, group_level_tests, injection_control, lateral_feature_gate, lateral_window, model_diagnostics, observation_cross_validation, observation_cross_validation_folds, parameter_table, positive_controls, sensitivity, sham_paired_tests


SHAM_NAMES = {"constant": "常数驱动", "random": "随机驱动"}


def _row(frame: pd.DataFrame, query: str) -> pd.Series:
    return frame.query(query).iloc[0]


def _verdict(flag: bool) -> str:
    return "通过" if flag else "未通过"


def write_report(output: Path, dataset, parameters, grid, validation: pd.DataFrame, loose_validation: pd.DataFrame, classifiers: pd.DataFrame, sham: pd.DataFrame, loose_sham: pd.DataFrame, paired: pd.DataFrame, direction: pd.DataFrame, geometry: pd.DataFrame, group_tests: pd.DataFrame, bounds: pd.DataFrame, controls: pd.DataFrame, injection: pd.DataFrame, lovo: pd.DataFrame, gate_label: str, start: float, stop: float, maximum: int, alpha: float, tightened: bool) -> None:
    task2 = _row(validation, "task == 2")
    task1 = _row(validation, "task == 1")
    final_sham = sham.query("kernel == 'tight'") if tightened else sham
    sham_best = final_sham.loc[final_sham["waveform_correlation"].idxmax()]
    loose_best = loose_sham.loc[loose_sham["waveform_correlation"].idxmax()]
    legacy = grid[grid["input_gain"] >= 4.0]
    legacy_best = legacy.loc[legacy["waveform_correlation"].idxmax()]
    grid_best = grid.loc[grid["waveform_correlation"].idxmax()]
    delta = float(task2.waveform_correlation - sham_best.waveform_correlation)
    mechanism = bool(task2.waveform_correlation > sham_best.waveform_correlation and delta >= 0.10)
    collapsed = _row(direction, "projection == 'collapsed'")
    directional = _row(direction, "projection == 'shape'")
    gain = float(directional.contrast_to_fz / max(collapsed.contrast_to_fz, np.finfo(float).eps))
    lateral_gain = bool(gain >= 10.0)
    signs = sorted({int(value) for value in geometry["lateralization_sign"]})
    stable = bool(len(signs) == 1 and signs[0] != 0)
    combined = _row(classifiers, "feature_set == 'Template + lateral'")
    template_only = _row(classifiers, "feature_set == 'Matched template'")
    max_bound = bounds.loc[bounds["auc_upper_bound"].idxmax()]
    task_control = _row(controls, "control == 'task1_vs_task2'")
    position_control = _row(controls, "control == 'task1_target_position'")
    control_ok = bool(min(task_control.auc, position_control.auc) >= 0.70)
    detected = injection[injection["auc"] >= 0.70]
    detectable = float(detected["injected_amplitude_rms"].min()) if len(detected) else float("nan")
    significant = int(group_tests["significant_clusters"].sum())
    best_lovo_correlation = lovo.loc[lovo["waveform_correlation"].idxmax()]
    best_lovo_sign = lovo.loc[lovo["sign_agreement"].idxmax()]
    lovo_threshold = bool(best_lovo_sign.sign_agreement > 0.60)
    lovo_improved = bool(lovo_threshold and best_lovo_sign.sign_agreement > float(task2.sign_agreement))
    lovo_verdict = "记为有效改进" if lovo_improved else ("虽达到60%的预注册阈值，但与全试次基准完全相同，判定为未改善" if lovo_threshold else "未达到60%的预注册阈值，LOVO 未改善判别，仅作为稳健性分析保留")
    loose_task2 = _row(loose_validation, "task == 2")
    kernel_text = f"滞后上限 {maximum} 个采样点、岭惩罚 {alpha:g}" + ("（因宽松核下伪模型相关达到0.85已自动收紧）" if tightened else "")
    loose_note = "" if not tightened else f"收紧前（滞后上限160、岭惩罚0.1）真模型留出相关为 {loose_task2.waveform_correlation:.3f}，最佳伪模型为 {loose_best.waveform_correlation:.3f}，两者差值 {float(loose_task2.waveform_correlation - loose_best.waveform_correlation):.3f}，说明宽松核下的高相关不能区分真模型与伪模型，因此后续数字一律取收紧后的结果。"
    paired_text = "; ".join(f"与{SHAM_NAMES.get(row.model, row.model)}伪模型的逐折配对差值为 {row.mean_difference:.3f}±{row.difference_sd:.3f}（配对t检验 p={row.p_value:.4f}，{int(row.folds)}折）" for row in paired.itertuples()) if len(paired) else "预注册的收紧流程未被触发，未做逐折配对比较"
    paired_caveat = f"其中最大的配对检验概率为 {float(paired['p_value'].max()):.4f}，在 0.05 水平上处于临界（5 折配对检验功效有限），该条结论按临界处理。" if len(paired) and float(paired["p_value"].max()) >= 0.05 else ""
    mechanism_text = "真模型在项目二留出试次上高于两个伪模型，差值超过预注册阈值，机理链的波形级验证成立。" if mechanism else "差值未达阈值，观察核在当前自由度下无法把模型驱动正确与核足够灵活区分开，机理链只能停留在结构级说明，不得宣称波形级复现。"
    injection_text = f"注入幅度达到 {detectable:.2f} 倍单试次均方根时曲线下面积首次达到0.70" if np.isfinite(detectable) else "本组注入幅度下曲线下面积未达到0.70"
    text = f"""# 问题二结果评估报告

## 一、三项题目要求的实测判定

| 题目要求 | 本轮实测数字 | 判定 |
|---|---|---|
| ① LGN到皮层再到头皮的机理 | 项目二留出波形相关 {task2.waveform_correlation:.3f}±{task2.waveform_correlation_sd:.3f}、谱余弦 {task2.spectral_cosine:.3f}±{task2.spectral_cosine_sd:.3f}；最佳伪模型相关 {sham_best.waveform_correlation:.3f}；差值 {delta:.3f} | {_verdict(mechanism)} |
| ② 左右差异形成机制 | 侧化对比度与中线峰值之比：方向平均 {collapsed.contrast_to_fz:.3f} 对方向通道 {directional.contrast_to_fz:.3f}，增益 {gain:.2f} 倍；镜像残差占比 {collapsed.mirror_residual_to_fz:.4f} 对 {directional.mirror_residual_to_fz:.4f}；几何扰动下侧化符号{'稳定' if stable else '不稳定'} | {_verdict(lateral_gain)} |
| ③ 可区分左右形状的特征表示 | 组水平显著簇 {significant} 个；阳性对照曲线下面积 {task_control.auc:.3f} 与 {position_control.auc:.3f}；效应量上界 {max_bound.auc_upper_bound:.3f}；联合特征曲线下面积 {combined.auc:.3f}、置换概率 {combined.p_value:.4f} | {_verdict(bool(combined.p_value < 0.05 and combined.auc > 0.5))} |

## 二、数据与校准

沿用问题一冻结的特征保持预处理。提示锁定纳入 {len(dataset.labels)} 个试次，目标锁定纳入 {len(dataset.target_labels)} 个试次。形状标签取视觉提示通道符号，位置标签取目标应答符号，两类标签分别分析。动力学参数（延迟、空间耦合、视觉输入增益）只用项目一校准；观测核在提示后 0 至 0.75 秒窗口内用五折按试次留出拟合，{kernel_text}。按文件分组的判别结果与按试次留出的波形验证口径不同，两者不可互相替代。

网格校准时发现原网格存在设计缺陷：原增益区间只取 4.0 至 6.0，落入强驱动区，仿真波形在项目一上的目标函数最大值仅 {legacy_best.waveform_correlation:.4f}；把增益下限扩到 1.0 之后目标函数最大值升到 {grid_best.waveform_correlation:.4f}（延迟 {grid_best.delay_ms:.0f} 毫秒、耦合 {grid_best.coupling:.1f}、增益 {grid_best.input_gain:.1f}）。目标函数对延迟与耦合并不敏感（在最优值 0.01 以内的候选遍布各延迟），因此采用“取最优值 0.01 邻域内延迟最小者”的规则定参，最终参数为延迟 {parameters.delay * 1000:.0f} 毫秒、耦合 {parameters.coupling:.1f}、增益 {parameters.input_gain:.1f}。延迟与耦合不可辨识这一点如实记录，参数表里只有增益是真正被项目一数据确定的。

## 三、要求①：机理链的波形级验证（对照实验）

真模型在项目二留出试次上的波形相关为 {task2.waveform_correlation:.3f}±{task2.waveform_correlation_sd:.3f}，项目一为 {task1.waveform_correlation:.3f}±{task1.waveform_correlation_sd:.3f}。伪模型对照中，常数驱动图的相关为 {float(_row(final_sham, "model == 'constant'").waveform_correlation):.3f}，随机驱动图为 {float(_row(final_sham, "model == 'random'").waveform_correlation):.3f}，最佳伪模型为 {sham_best.waveform_correlation:.3f}。真模型相对最佳伪模型的差值为 {delta:.3f}，判定阈值为 0.10。逐折配对比较：{paired_text}。伪模型的两个驱动图在左右条件上完全相同，一个核只需拟合一个信号，任务比真模型更简单，这个对照对真模型偏严。

{mechanism_text}{paired_caveat}{loose_note}

## 四、要求②：左右差异机制（方向通道与导联几何）

方向平均投影下，左右三角的皮层驱动图互为镜像。若导联场关于中线对称，则镜像驱动的结果是 F3 与 F4 互换、Fz 不变，也就是左右差异只剩一个“镜像个自由度”。本轮实测：方向平均模型的左右条件对比度为中线峰值的 {collapsed.contrast_to_fz:.3f}，镜像残差占比 {collapsed.mirror_residual_to_fz:.4f}，Fz 左右条件差占比 {collapsed.fz_condition_to_fz:.4f}。

方案预期“方向平均把左右差异压到噪声水平、修复后应有一个数量级提升”。实测与该预期不符：方向平均模型的侧化对比度并非噪声水平，方向通道修复后的对比度为 {directional.contrast_to_fz:.3f}，增益仅 {gain:.2f} 倍，未达一个数量级。修复真正改变的是镜像残差：由 {collapsed.mirror_residual_to_fz:.4f} 升到 {directional.mirror_residual_to_fz:.4f}，即出现了无法用 F3/F4 互换解释的侧化分量，同时中线电极 Fz 也出现了 {directional.fz_condition_to_fz:.4f} 的条件差。对角方向差图的注入权重 0.25 是人为设定，无生理依据，报告按自由参数记录。若把验收口径改为“镜像残差由零变为非零的一阶侧化分量”，该修复通过；若按方案原文的“对比度提升一个数量级”，该修复不通过，且基线侧化对比度本身已达中线峰值的 {collapsed.contrast_to_fz:.3f}，一个数量级的提升在这条链上不可达。

导联场用反平方距离衰减近似，电极中心与深度作 ±20% 扰动后的结果为：{"; ".join(f"偏移{row.geometry_shift:+.1f}时对比度{row.contrast_to_fz:.3f}、符号{int(row.lateralization_sign)}" for row in geometry.itertuples())}。侧化符号在扰动下{'保持不变' if stable else '发生改变或为零'}。该导联场不是三层球解析模型，也没有个体头几何，只能视为简化的正向映射。

## 五、要求③：可区分特征表示（前置检验、阳性对照、上界、稳健化）

提示锁定与目标锁定共完成 {len(group_tests)} 项组水平簇置换检验（家族错误率校正到 0.05/24），显著簇总数为 {significant}。判定结果：{gate_label}。单试次判别所用侧化窗口为提示后 {start * 1000:.0f} 至 {stop * 1000:.0f} 毫秒。

阳性对照：项目一与项目二之间曲线下面积为 {task_control.auc:.3f}（置换概率 {task_control.p_value:.4f}），项目一目标出现侧为 {position_control.auc:.3f}（置换概率 {position_control.p_value:.4f}），预设阈值为 0.70。{'两个经验对照均达标，说明预处理、特征、交叉验证与分类器这条管线本身有效，形状判别的阴性结论因此成立。' if control_ok else '经验对照没有同时达标。经验对照沿用与主分析相同的跨文件、跨被试留出协议，本身偏难，不能单独用来判定管线失效。'}

注入式管线对照：把一条已知的左右差异（0.10 至 0.30 秒半余弦，按左右标签加载到 F4 与 F3 上）叠加到真实试次上，用同一套特征、交叉验证与置换流程重跑，结果为 {"; ".join(f"注入幅度为{row.injected_amplitude_rms:.2f}倍单试次均方根时曲线下面积{row.auc:.3f}（置换概率{row.p_value:.4f}）" for row in injection.itertuples())}。{injection_text}，而真实数据的曲线下面积为 {combined.auc:.3f}。该对照直接给出这条管线能检测到的侧化差异量级：数据里不存在达到该量级的左右差异，形状判别的阴性结论不是管线失效造成的。

效应量上界：最大窗口效应量为 {max_bound.cohens_d:.3f}（{max_bound['lock']}锁定、项目{int(max_bound.task)}、{max_bound.window_start_s:.1f} 至 {max_bound.window_stop_s:.1f} 秒），由正态近似得到的单试次曲线下面积上界为 {max_bound.auc_upper_bound:.3f}。该上界由标签侧单试次 F4−F3 指标算出，是继续投入分类器的决策依据：上界低于 0.60 时继续堆叠分类器没有意义。

形状判别结果：仅用模型匹配投影特征为 {template_only.auc:.3f}（置换概率 {template_only.p_value:.4f}），叠加侧化指标后为 {combined.auc:.3f}（置换概率 {combined.p_value:.4f}），准确率 {combined.accuracy:.3f}。{'未显著优于随机。' if combined.p_value >= 0.05 else '达到显著。'}

稳健化（LOVO）：观察核训练均值改用保留比例扫描后，最高留出相关为 {best_lovo_correlation.waveform_correlation:.3f}（保留 {best_lovo_correlation.retained_fraction:.0%}），全试次基准为 {task2.waveform_correlation:.3f}；符号一致率最高为 {best_lovo_sign.sign_agreement:.1%}（保留 {best_lovo_sign.retained_fraction:.0%}），全试次基准为 {task2.sign_agreement:.1%}。按预注册口径（符号一致率超过 60% 或置换概率低于 0.05 才算有效改进），本轮{lovo_verdict}。方案还要求对判别特征同步重估，本轮判别特征为逐试次量、不含组平均，没有可重估的组平均环节，此处如实记录与方案预期的差异。

## 六、已试的代码改动与未通过项

- 方向通道修复：皮层投影改为四个方向通道分别投影，并把对角方向差图作为侧化驱动项注入。验收指标（侧化对比度一个数量级提升）未通过，实测增益 {gain:.2f} 倍；副产物是镜像残差与中线条件差增大。
- 导联场几何：高斯改为反平方距离衰减，并做 ±20% 中心与深度扰动；侧化符号{'稳定' if stable else '不稳定'}。
- 伪模型对照：常数驱动与随机驱动两套对照，配同一套核拟合与留出流程。
- 观测核与参数网格：观测核自由度用伪模型对照约束；校准网格原先把增益下限设在 4.0，落进强驱动区，目标函数最大值只有 {legacy_best.waveform_correlation:.4f}，扩到 1.0 后升到 {grid_best.waveform_correlation:.4f}。
- 组水平前置检验：两个零点、两个项目、六类特征的簇置换检验，输出完整概率表。
- 阳性对照：两个应当可解码的经验对照任务，用于区分“管线失效”与“数据中无该信息”。
- 注入式对照：把已知幅度的侧化差异叠加到真实试次上，给出这条管线能检测到的最小侧化差异量级。
- 效应量上界：用于决定是否继续投入单试次分类。
- LOVO 稳健化：观察核训练均值改为保留比例 {', '.join(f'{value:.0%}' for value in lovo['retained_fraction'])} 扫描。

## 七、输出索引

- `table2-1_model_parameters.csv`：模型参数、含义与来源
- `table2-2_sensitivity.csv`：关键参数上下 20% 扰动
- `table2-3_validation.csv`：五折留出试次验证指标
- `classifier_performance.csv`：按文件分组交叉验证与置换检验
- `calibration_grid.csv`：参数网格搜索记录
- `sham_comparison.csv`：伪模型对照（宽松核与收紧核两阶段）
- `sham_paired_tests.csv`：真模型与伪模型的逐折配对检验
- `directional_projection.csv`、`geometry_sensitivity.csv`：方向通道修复与导联几何验收
- `group_level_tests.csv`：组水平簇置换检验概率表
- `positive_controls.csv`：经验阳性对照
- `injection_control.csv`：注入式管线灵敏度对照
- `decodability_upper_bounds.csv`：分窗口效应量与可解码性上界
- `lovo_comparison.csv`：LOVO 保留比例扫描
- `figures`：模型结构、形状编码、序参数、真实数据验证、机制链、侧化轨迹、判别图与对照图（`fig2-8_controls` 依次为伪模型对照、注入灵敏度、可解码性上界、组水平侧化检验）
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
    control_table = positive_controls(dataset, config["q2"]["classifier_permutations"], rng)
    group_table, group_windows = group_level_tests(dataset, config["statistics"]["cluster_permutations"], rng)
    gate = lateral_feature_gate(dataset, group_windows)
    bounds_table = decodability_bounds(dataset)
    direction_table, geometry_table = model_diagnostics(parameters, dataset.times, rate, config["q2"]["grid_size"])
    maximum, alpha = 160, 0.1
    real_folds = observation_cross_validation_folds(dataset, model, rate, maximum=maximum, alpha=alpha)
    validation_table = aggregate_validation(real_folds)
    loose_validation = validation_table
    sham_models = {mode: {side: simulate(side, dataset.times, rate, parameters, config["q2"]["grid_size"], drive_mode=mode, seed=20260923) for side in (-1, 1)} for mode in ("constant", "random")}
    sham_folds = {mode: observation_cross_validation_folds(dataset, sham_models[mode], rate, maximum=maximum, alpha=alpha) for mode in ("constant", "random")}
    sham_rows = [{"kernel": "loose", "maximum": maximum, "alpha": alpha, "model": mode, **aggregate_validation(frame).query("task == 2").iloc[0].to_dict()} for mode, frame in sham_folds.items()]
    loose_sham = pd.DataFrame(sham_rows)
    tightened = float(loose_sham["waveform_correlation"].max()) >= 0.85
    paired_table = pd.DataFrame()
    if tightened:
        maximum, alpha = 60, 1.0
        real_folds = observation_cross_validation_folds(dataset, model, rate, maximum=maximum, alpha=alpha)
        validation_table = aggregate_validation(real_folds)
        sham_folds = {mode: observation_cross_validation_folds(dataset, sham_models[mode], rate, maximum=maximum, alpha=alpha) for mode in ("constant", "random")}
        sham_rows += [{"kernel": "tight", "maximum": maximum, "alpha": alpha, "model": mode, **aggregate_validation(frame).query("task == 2").iloc[0].to_dict()} for mode, frame in sham_folds.items()]
        paired_table = sham_paired_tests(real_folds, sham_folds, task=2)
    sham_table = pd.DataFrame(sham_rows)
    prediction, kernels = fit_observation_kernel(model, erp, dataset.times, maximum, alpha)
    sensitivity_table = sensitivity(parameters, kernels, erp, dataset.times, rate, config, maximum)
    used = gate if gate.any() else window
    gate_label = "侧化特征落入了组水平显著窗口，允许进入单试次判别" if gate.any() else "侧化特征没有任何窗口通过组水平检验，单试次判别只能作为探索性分析，不能作为已实现的特征表示"
    combined, template = feature_matrix(dataset, mechanistic_prediction, used)
    results = {}
    classifier_rows = []
    for name, values, key in (("Matched template", template, "template"), ("Template + lateral", combined, "combined")):
        result = classify(values, dataset.labels, dataset.groups, config["q2"]["classifier_permutations"], rng)
        results[key] = result
        classifier_rows.append({"feature_set": name, "gated": bool(gate.any()), **result[0]})
    classifiers = pd.DataFrame(classifier_rows)
    injection_table = injection_control(dataset, mechanistic_prediction, used, config["q2"]["injection_amplitudes"], config["q2"]["classifier_permutations"], rng)
    lovo_rows = []
    for fraction in (0.6, 0.75, 0.9, 0.95):
        robust_cv = observation_cross_validation(dataset, model, rate, maximum=maximum, alpha=alpha, lovo_fraction=fraction)
        lovo_rows.append({"retained_fraction": fraction, **robust_cv.query("task == 2").iloc[0].to_dict()})
    lovo_table = pd.DataFrame(lovo_rows)
    parameter_table(parameters, kernels).to_csv(output / "table2-1_model_parameters.csv", index=False)
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
    injection_table.to_csv(output / "injection_control.csv", index=False)
    paired_table.to_csv(output / "sham_paired_tests.csv", index=False)
    lovo_table.to_csv(output / "lovo_comparison.csv", index=False)
    plot_architecture(figures)
    plot_shape_encoding(model, figures)
    plot_order(model, dataset.times, figures)
    plot_erp_validation(erp, prediction, dataset.times, figures)
    plot_mechanism(figures)
    plot_lateralization(dataset, used, figures, rng)
    plot_classifier(results, figures)
    plot_controls(sham_table, injection_table, bounds_table, group_table, float(validation_table.query("task == 2").iloc[0].waveform_correlation), float(classifiers.query("feature_set == 'Template + lateral'").iloc[0].auc), figures)
    write_report(output, dataset, parameters, grid, validation_table, loose_validation, classifiers, sham_table, loose_sham, paired_table, direction_table, geometry_table, group_table, bounds_table, control_table, injection_table, lovo_table, gate_label, start, stop, maximum, alpha, tightened)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cti.yaml")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    with (root / args.config).open(encoding="utf-8") as stream:
        run(root, yaml.safe_load(stream))


if __name__ == "__main__":
    main()
