"""Replay all three questions and write one root-level evaluation report."""
from datetime import datetime
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys

from pipeline import ROOT, DATA, dump, source_hashes


def markdown_table(frame):
    def cell(value):
        if isinstance(value, float):
            return f"{value:.4f}" if value == value else "未定义"
        return str(value).replace("|", "/").replace("\n", " ")
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |", "|" + "---|" * len(frame.columns)]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run", action="store_true", help="Regenerate all three questions before evaluation")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace recognized generated output")
    args = parser.parse_args(argv)
    import numpy as np
    import pandas as pd
    from Ques1.src.inference import choose_device
    from Ques1.src.verify import verify as verify_q1
    from Ques2.utils import verify_results
    from Ques2.iter_utils import verify as verify_nested, aggregate
    from Ques3.verify import verify as verify_q3
    from Ques1.qa.audit_figure_collisions import audit_pdf
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
    device, hardware = choose_device(args.device)
    checks, sections = [], []

    def check(name, condition, detail=""):
        checks.append(dict(check=name, passed=bool(condition), detail=str(detail)))
        return bool(condition)

    def attempt(name, action):
        try:
            action()
        except Exception as error:
            check(name, False, f"{type(error).__name__}: {error}")
        else:
            check(name, True)

    outputs = {number: ROOT / f"Ques{number}/output" for number in (1, 2, 3)}
    if args.run:
        for number in (1, 2, 3):
            command = [sys.executable, "-B", "-X", "utf8", str(ROOT / f"Ques{number}/run_problem{number}.py"),
                       "--data", str(args.data), "--device", args.device]
            if args.overwrite:
                command.append("--overwrite")
            result = subprocess.run(command, cwd=ROOT)
            if not check(f"问题{number}完整运行", result.returncode == 0, f"exit={result.returncode}"):
                break
    for number, out in outputs.items():
        check(f"问题{number}输出目录存在", out.is_dir())
        check(f"问题{number}输出目录平铺", out.is_dir() and not any(p.is_dir() for p in out.iterdir()))
        check(f"问题{number}输出无独立报告", not list(out.glob("*.md")))
        try:
            status = json.loads((out / "completion.json").read_text(encoding="utf8"))
            check(f"问题{number}运行完整", status.get("status") == "complete" and status.get("stage") == 3, status.get("status"))
            check(f"问题{number}源码与运行版本一致", status.get("source_sha256") == source_hashes(f"Ques{number}"))
        except (OSError, ValueError) as error:
            check(f"问题{number}运行状态可读取", False, error)
        for path in (ROOT / f"Ques{number}").rglob("*.py"):
            attempt(f"语法检查 {path.relative_to(ROOT)}", lambda path=path: compile(path.read_text(encoding="utf-8-sig"), str(path), "exec"))
    legacy = [p for folder in ("Ques1/results", "Ques2/outputs", "Ques3/outputs") if (p := ROOT / folder).exists()]
    old_docs = [p for number in (1, 2, 3) for p in (ROOT / f"Ques{number}").rglob("*.md")]
    check("旧输出目录已清理", not legacy, ", ".join(str(p.relative_to(ROOT)) for p in legacy))
    check("三问目录无重复说明或评估报告", not old_docs, f"剩余 {len(old_docs)} 个旧文档")

    def audit_inputs():
        recordings = pd.read_csv(outputs[1] / "recordings.csv")
        check("四条原始记录及四百次事件", len(recordings) == 4 and recordings.n_cues.sum() == 400)
        for row in recordings.itertuples():
            path = args.data / f"{row.record}.mat"
            check(f"原始数据散列 {row.record}", hashlib.sha256(path.read_bytes()).hexdigest() == row.sha256)
        sections.extend(["## 一、数据与题目覆盖", markdown_table(recordings[["record", "fs", "n_cues", "n_left", "n_right", "click_events"]]),
                         "仅分析原始前三通道，分析顺序为左额、中额、右额；不使用机器滤波通道。两个记录组的被试身份不明确，不能直接称为跨被试验证。"])
    attempt("原始数据与结果溯源", audit_inputs)
    attempt("问题一科学不变量重放", lambda: verify_q1(args.data, outputs[1], device))

    def replay_q2():
        out = outputs[2]
        with np.load(out / "validation_epochs.npz") as saved:
            signals = saved["X"].copy()
        trials = pd.read_csv(out / "trials.csv")
        verify_results(signals, trials, out, device)
        status = json.loads((out / "nested_completion.json").read_text(encoding="utf8"))
        check("问题二嵌套验证完整", status.get("status") == "complete" and status.get("outer_folds") == 12)
        verify_nested(out, signals, trials, device)
        recomputed = aggregate(pd.read_csv(out / "nested_predictions.csv")).sort_values(["mode", "group", "scheme"])
        stored = pd.read_csv(out / "nested_summary_metrics.csv").sort_values(["mode", "group", "scheme"])
        columns = recomputed.select_dtypes(include=np.number).columns
        check("问题二嵌套汇总指标复算", np.allclose(recomputed[columns], stored[columns], equal_nan=True))
        counts = pd.read_csv(out / "recording_counts.csv")
        sources = pd.read_csv(outputs[1] / "recordings.csv").set_index("record")
        for row in counts.itertuples():
            check(f"问题二数据链 {row.record}", row.sha256 == sources.loc[row.record, "sha256"] and
                  row.cache_sha256 == hashlib.sha256((outputs[1] / f"{row.record}_epochs.npz").read_bytes()).hexdigest())
    attempt("问题二训练隔离与预测重放", replay_q2)
    attempt("问题三状态与指标重放", lambda: verify_q3(outputs[3], outputs[2], device))

    def summarize_q1():
        out = outputs[1]
        quality = pd.read_csv(out / "trial_quality.csv")
        classification = pd.read_csv(out / "classification.csv")
        predictions = pd.read_csv(out / "predictions.csv")
        for key, subset in predictions.groupby(["project", "stage", "window", "scheme"]):
            row = classification[(classification.project == key[0]) & (classification.stage == key[1]) &
                                 (classification.window == key[2]) & (classification.scheme == key[3]) & (classification.channel == "all")].iloc[0]
            actual = [accuracy_score(subset.label, subset.predicted), balanced_accuracy_score(subset.label, subset.predicted), roc_auc_score(subset.label, subset.score)]
            check(f"问题一折外指标 {key}", np.allclose(actual, row[["accuracy", "balanced_accuracy", "auc"]].to_numpy(float)))
        selected = classification[(classification.stage == "clean") & (classification.window == "p300") & (classification.channel == "all")]
        fit = pd.read_csv(out / "fit_cross_record.csv")
        recovery = pd.read_csv(out / "synthetic_recovery.csv")
        peaks = pd.read_csv(out / "p300_metrics.csv")
        second = peaks[(peaks.project == 2) & (peaks.stage == "clean")]
        sections.extend(["## 二、问题一：去噪、视觉响应与拟合",
            f"描述性试次保留 {int(quality.keep.sum())}/{len(quality)}；合成脉冲峰幅保留比例为 {recovery.peak_retention.min():.6f}–{recovery.peak_retention.max():.6f}。这只证明所设合成波形的处理保真，不能证明真实形状信息已全部保留。",
            markdown_table(selected[["project", "scheme", "n", "accuracy", "balanced_accuracy", "auc", "permutation_p"]]),
            "跨记录拟合检查：", markdown_table(fit),
            f"项目二去噪曲线中 {int(second.peak_at_window_edge.sum())}/{len(second)} 个窗口最大值位于窗口边缘。不能将边缘最大值或晚期慢波直接认定为经典三百毫秒正成分。问题一组内结果使用描述性全记录预处理，不能当作严格在线或完全折内预处理的确认性分类；更严格的分块分类见问题二。"])
    attempt("问题一指标与结果摘要", summarize_q1)

    def summarize_q2():
        out = outputs[2]
        primary = json.loads((out / "primary_metrics.json").read_text(encoding="utf8"))
        summary = pd.read_csv(out / "nested_summary_metrics.csv")
        mechanism = pd.read_csv(out / "fit_metrics.csv")
        selected = summary[summary.group == "all"]
        sections.extend(["## 三、问题二：形成机制、反演与左右刺激特征",
            f"原机制模型训练内选定方案的跨记录准确率为 {primary['accuracy']:.2%}，平衡准确率为 {primary['balanced_accuracy']:.2%}，折间平均曲线下面积为 {primary['roc_auc']:.4f}。",
            markdown_table(mechanism.groupby(["scope", "task"])[["rmse", "r2"]].mean().reset_index()),
            "嵌套验证包含两次跨记录与十次组内分块外层。各方案及训练内选定方案全部列出，不按测试成绩改选主方案。", 
            markdown_table(selected[["mode", "scheme", "n", "accuracy", "balanced_accuracy", "auc"]]),
            "左右源和公共源通过对称传导矩阵形成头皮波形；源特征从未知方向的脑电反演，不能把测试刺激标签送入生成器后再分类。反演是确定变换，不会凭空增加刺激标签信息。模型可用于解释假说和特征构造，但弱分类结果不支持可靠形状识别。三电极不能唯一定位深部源；这些已被反复观察的数据只支持探索性迭代，不能称为全新的确认性外部测试。"])
    attempt("问题二指标与结果摘要", summarize_q2)

    def summarize_q3():
        out = outputs[3]
        choice = pd.read_csv(out / "choice_metrics.csv")
        timing = pd.read_csv(out / "time_metrics.csv")
        eeg = pd.read_csv(out / "eeg_metrics.csv")
        counts = pd.read_csv(out / "behavior_counts.csv")
        comparison = eeg[eeg.scope == "full_pre_response"].groupby("model")[["nrmse", "r2", "pearson"]].mean().reset_index()
        sections.extend(["## 四、问题三：认知宏观模型",
            markdown_table(counts), markdown_table(choice[["fold", "n", "accuracy", "balanced_accuracy", "recall_left", "recall_right", "threshold_accuracy", "threshold_response_coverage"]]),
            markdown_table(timing[timing.fold == "pooled"]), markdown_table(comparison),
            "项目一无点击记录只用于视觉对照，不解释为二百次真实遗漏；项目二保留全部点击，方向一致/不一致不等于题目已提供真实答案键。本数据没有项目二真实无应答样本，不能验证无应答类别。二分类截止读出与实际越阈应答严格分开，未越阈不补造时间。记忆状态是功能性潜变量，不是测得的海马活动。离线脑电重建改善也不能替代方向、时间预测改善；零相位滤波不支持在线因果预测或疾病诊断结论。"])
    attempt("问题三指标与结果摘要", summarize_q3)

    figure_rows = []
    for number, out in outputs.items():
        paths = sorted(out.glob("*.pdf"))
        check(f"问题{number}图组数量", len(paths) == {1: 11, 2: 10, 3: 5}[number], len(paths))
        for path in paths:
            try:
                result = audit_pdf(path)
                passed = result["summary"]["fail"] == 0 and result["summary"]["warn"] == 0
                alignment = json.loads(path.with_suffix(".alignment.json").read_text(encoding="utf8"))
                passed = passed and alignment.get("verdict") in ("PASS", "NOT APPLICABLE")
                check(f"图形核验 {number}/{path.stem}", passed, result["summary"])
                figure_rows.append(dict(question=number, figure=path.name, passed=passed))
            except Exception as error:
                check(f"图形核验 {number}/{path.stem}", False, error)
        if out.is_dir():
            dump(out / "figure_audit.json", [row for row in figure_rows if row["question"] == number])
    inventory = []
    for number, out in outputs.items():
        paths = [p for p in out.iterdir() if p.is_file()] if out.exists() else []
        inventory.append(dict(问题=number, 数据表=sum(p.suffix == ".csv" for p in paths),
                              图组=sum(p.suffix == ".pdf" for p in paths), 模型与数组=sum(p.suffix in (".pkl", ".npz") for p in paths),
                              元数据=sum(p.suffix == ".json" for p in paths), 子目录=sum(p.is_dir() for p in out.iterdir()) if out.exists() else 0))
    failed = [row for row in checks if not row["passed"]]
    lines = ["# 三问总评估报告", f"生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}。本报告由根目录 `evaluate.py` 自动生成，不手工填写验证通过数。",
             f"本次检查 {len(checks)} 项，通过 {len(checks)-len(failed)} 项，未通过 {len(failed)} 项。状态：{'完整通过' if not failed else '存在未完成或未通过项，见末节'}。",
             "运行成功、数值核验通过与科学假说得到支持是不同结论。以下数值来自当前输出文件；缺失或重放失败会明确报错，不用历史报告补造。"]
    lines += sections
    lines += ["## 五、交付附件", markdown_table(pd.DataFrame(inventory)),
              "每问附件直接位于 `QuesN/output`，没有图、表或模型子目录。问题二 `nested_` 前缀表示嵌套分类扩展，不能与原机制模型参数混用。图像提供可编辑矢量版及位图，表格为对应源数据；模型、数组和训练划分保留用于复现，不是无用临时文件。详细方法、运行步骤及图表索引见根目录 `论文写作材料.md`。",
              "## 六、验证记录", markdown_table(pd.DataFrame(checks)),
              "## 七、解释边界与待处理事项",
              "原始数据没有患者标签、疾病组或独立临床验证，不能写成精神疾病诊断模型已获验证。仅两个记录组，置信区间和置换结果不是人群推广证据；通道幅值缺少物理标定，不能擅自写成微伏。问题一局部拟合、问题二机制拟合、问题三认知预测分别报告，不能用其中某个样本内高拟合值替代全链路有效性。"]
    if failed:
        lines += ["以下项目必须保留为未完成，不能将报告表述为全部通过："]
        lines += [f"- {row['check']}：{row['detail'] or '检查未通过'}" for row in failed]
    else:
        lines += ["当前输出的结构、数据链、预测重放及图形检查均通过；这不改变模型性能及科学解释边界。"]
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "pandas", "torch", "scikit-learn", "matplotlib")}
    lines += ["## 八、评估环境", f"解释器：`{sys.executable}`。计算设备：`{hardware}`。依赖版本：`{versions}`。",
              "复核命令：`python -B evaluate.py`；从原始数据重跑并评估：`python -B evaluate.py --run --overwrite`。"]
    destination = ROOT / "总评估报告.md"
    destination.write_text("\n\n".join(lines) + "\n", encoding="utf8")
    print(f"{destination}: {len(checks)-len(failed)}/{len(checks)} checks passed", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
