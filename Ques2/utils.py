"""Data audit, reused preprocessing, fold-local features, and plotting helpers."""
from pathlib import Path
import hashlib
import json
import re
import sys
import numpy as np
import pandas as pd
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Ques1" / "src"))
from preprocess import filter_record, wavelet_high_only, baseline, TIME
from plotting import setup, save
from inference import choose_device

CHANNELS = ["F3", "Fz", "F4"]
CONDITIONS = [(1, -1), (1, 1), (2, -1), (2, 1)]
SEED = 20260924


def dump(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def markdown_table(frame):
    frame = frame.copy().fillna("缺失")
    def cell(v):
        return f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v)
    rows = ["| " + " | ".join(map(str, frame.columns)) + " |", "|" + "---|" * len(frame.columns)]
    rows += ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join(rows)


def load_record(path):
    mat = loadmat(path, squeeze_me=True)
    arrays = [(k, np.asarray(v)) for k, v in mat.items() if not k.startswith("__") and isinstance(v, np.ndarray) and v.ndim == 2 and 10 in v.shape]
    if len(arrays) != 1:
        raise ValueError(f"{path.name}: cannot uniquely identify ten-channel data")
    key, raw = arrays[0]
    raw = raw if raw.shape[0] == 10 else raw.T
    raw = raw.astype(float)
    labels = [str(np.asarray(v).item()).strip() for v in np.ravel(mat.get("DataLabel", []))]
    expected = ["Fz", "F3", "F4", "FzDecon", "F3Decon", "F4Decon", "ECG", "VisCue", "TgtAct", "TimeStamp"]
    if labels[:7] != expected[:7] or not labels[7].startswith("VisCue") or not labels[8].startswith(("Action", "TgtAct")) or labels[9] != "TimeStamp" or float(mat.get("SampleRate", 0)) != 256:
        raise ValueError(f"{path.name}: channel or sampling-rate mismatch: {labels}")
    if not np.isfinite(raw[[7, 8, 9]]).all() or not np.allclose(np.diff(raw[9]), 1 / 256):
        raise ValueError(f"{path.name}: invalid events or timestamps")
    if not set(np.unique(raw[7])).issubset({-1, 0, 1}):
        raise ValueError(f"{path.name}: unrecognized visual labels")
    cue = raw[7]
    starts = np.flatnonzero(np.isin(cue, [-1, 1]) & np.r_[True, cue[1:] != cue[:-1]])
    if len(starts) < 20 or np.min(np.diff(starts)) < 256:
        raise ValueError(f"{path.name}: abnormal event counts or overlapping epochs")
    match = re.fullmatch(r"VisualCog([AB])_Task-([12])", path.stem)
    if match is None:
        raise ValueError(f"Unexpected recording name: {path.stem}")
    return raw, starts, match[1], int(match[2]), key


def stage1(data_dir, cache_dir, out, device):
    import torch
    import matplotlib.pyplot as plt
    paths = sorted(Path(data_dir).glob("VisualCog*_Task-*.mat"))
    if len(paths) != 4:
        raise ValueError("Exactly four VisualCog recording files are required")
    events = pd.read_csv(Path(cache_dir) / "events.csv")
    quality = pd.read_csv(Path(cache_dir) / "trial_quality.csv")
    expected_hashes = pd.read_csv(Path(cache_dir) / "recordings.csv").set_index("record").sha256
    all_clean, all_valid, all_meta, audit = [], [], [], []
    for path in paths:
        print(f"Stage 1: inspect {path.name}", flush=True)
        raw, starts, group, task, variable = load_record(path)
        cache = Path(cache_dir) / f"{path.stem}_epochs.npz"
        if not cache.exists():
            raise FileNotFoundError(f"Missing reusable Question 1 epochs: {cache}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected_hashes[path.stem]:
            raise ValueError(f"{path.name}: cached epochs belong to different raw data")
        with np.load(cache) as saved:
            clean = saved["clean"].copy()
            keep = saved["keep"].copy()
            if list(saved["channels"]) != CHANNELS or not np.array_equal(saved["samples"], starts) or not np.array_equal(saved["labels"], raw[7, starts]):
                raise ValueError(f"{path.name}: cached events/channels mismatch")
            if clean.shape != (len(starts), 3, len(TIME)) or not np.allclose(saved["time_s"], TIME):
                raise ValueError(f"{path.name}: cached epoch grid mismatch")
            raw_epochs = np.stack([raw[[1, 0, 2], s-51:s+206] for s in starts])
            if not np.allclose(saved["raw"], baseline(raw_epochs), equal_nan=True):
                raise ValueError(f"{path.name}: raw epoch alignment failed")
        e = events[events.record == path.stem].sort_values("trial").reset_index(drop=True)
        qc = quality[quality.record == path.stem].sort_values("trial").reset_index(drop=True)
        if not np.array_equal(e["sample"], starts) or not np.array_equal(qc.keep, keep):
            raise ValueError(f"{path.name}: cache metadata mismatch")
        eeg = raw[[1, 0, 2]]
        validation = np.zeros_like(clean)
        boundaries = [0] + [int((starts[i-1] + starts[i]) // 2) for i in range(20, len(starts), 20)] + [eeg.shape[1]]
        for block, (lo, hi) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            ids = np.flatnonzero(np.arange(len(starts)) // 20 == block)
            segment = eeg[:, lo:hi].copy()
            if not np.isfinite(segment).all():
                raise ValueError(f"{path.name}: missing continuous values require explicit review")
            filtered = filter_record(segment, wavelet=False)
            for i in ids:
                s = starts[i] - lo
                if s < 51 or s + 206 > hi - lo:
                    raise ValueError("A trial crosses its independent preprocessing block")
                validation[i] = baseline(wavelet_high_only(filtered[:, s-51:s+206]))
        hard_reasons = []
        for i, s in enumerate(starts):
            reasons = []
            if not np.isfinite(raw_epochs[i]).all(): reasons.append("nonfinite")
            if np.any(np.abs(eeg[:, max(0, s-512):s+512]) >= 999.9): reasons.append("saturation_within_2s")
            if np.any(np.std(raw_epochs[i], axis=-1) < 1e-8): reasons.append("flat")
            hard_reasons.append(";".join(reasons))
        meta = pd.DataFrame(dict(record=path.stem, group=group, task=task, trial=np.arange(len(starts)), sample=starts,
                                 y=raw[7, starts].astype(int), block=np.arange(len(starts)) // 20, keep=keep,
                                 hard_keep=[not r for r in hard_reasons], hard_reason=hard_reasons,
                                 q1_reason=qc.reason.fillna(""), response=e.click_sign,
                                 reaction_time=e.click_delay_s, later_direction=e.target_sign,
                                 later_direction_delay=e.target_delay_s,
                                 response_target_latency=e.click_delay_s-e.target_delay_s))
        meta["response_cue_agreement"] = np.where(meta.response.notna(), meta.response == meta.y, np.nan)
        meta["response_later_agreement"] = np.where(meta.response.notna(), meta.response == meta.later_direction, np.nan)
        meta["correct"] = np.nan
        for d in [-1, 1]:
            if ((meta.y == d) & meta.keep).sum() < 10:
                raise ValueError(f"{path.name}: too few valid trials in class {d}")
        all_clean.append(clean); all_valid.append(validation); all_meta.append(meta)
        audit.append(dict(record=path.stem, task=task, group=group, raw_shape=list(raw.shape), variable=variable,
                          events=len(starts), retained=int(keep.sum()), removed=int((~keep).sum()),
                          left=int(((meta.y == -1) & keep).sum()), right=int(((meta.y == 1) & keep).sum()),
                          classification_hard_valid=int(meta.hard_keep.sum()), sha256=digest,
                          cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest()))
    meta = pd.concat(all_meta, ignore_index=True)
    clean, validation = np.concatenate(all_clean), np.concatenate(all_valid)
    keep = meta.keep.to_numpy()
    if not np.isfinite(clean).all() or not np.isfinite(validation).all():
        raise ValueError("Nonfinite EEG after preprocessing")
    baseline_error = float(np.abs(clean[..., TIME < 0].mean(-1)).max())
    if baseline_error > 1e-8:
        raise ValueError("Cached baseline correction failed")
    arrays = dict(X=clean[keep], y=meta.y.to_numpy()[keep], task=meta.task.to_numpy()[keep],
                  group=meta.group.to_numpy()[keep], times=TIME, trial=meta.trial.to_numpy()[keep],
                  response=meta.response.to_numpy()[keep], reaction_time=meta.reaction_time.to_numpy()[keep],
                  correct=meta.correct.to_numpy()[keep], channels=CHANNELS)
    np.savez_compressed(out / "model" / "epochs.npz", **arrays)
    np.savez_compressed(out / "model" / "validation_epochs.npz", X=validation, times=TIME)
    meta.to_csv(out / "tables" / "trials.csv", index=False)
    pd.DataFrame(audit).drop(columns=["raw_shape"]).to_csv(out / "tables" / "recording_counts.csv", index=False)
    erps = condition_means(clean, meta, np.flatnonzero(keep), device)
    rows = [dict(task=m, direction=d, channel=ch, time_s=t, amplitude=v)
            for (m, d), wave in zip(CONDITIONS, erps) for ch, curve in zip(CHANNELS, wave) for t, v in zip(TIME, curve)]
    pd.DataFrame(rows).to_csv(out / "tables" / "erp_curves.csv", index=False)
    fig, axs = plt.subplots(2, 2, figsize=(10, 6.4))
    fig.subplots_adjust(left=.10, right=.97, bottom=.11, top=.88, wspace=.30, hspace=.47)
    for ax, (m, d), wave in zip(axs.flat, CONDITIONS, erps):
        for ch, curve, color in zip(CHANNELS, wave, ["#357AA1", "#8F6B96", "#C98245"]):
            ax.plot(TIME*1000, curve, label=ch, color=color)
        n = ((meta.task == m) & (meta.y == d) & meta.keep).sum()
        ax.set_title(f"任务{m} · {'左' if d == -1 else '右'}刺激 · {n}试次")
        ax.axvline(0, color=".6", lw=.7); ax.axhline(0, color=".7", lw=.7)
        ax.axvspan(250, 500, color=".9", zorder=0)
        ax.set(xlabel="刺激后时间（毫秒）", ylabel="幅值（原记录单位）", xlim=(-200, 800))
    fig.legend(*axs[0, 0].get_legend_handles_labels(), loc="upper center", ncol=3)
    save(fig, out / "figures" / "stage1_erp")
    summary = pd.DataFrame(audit)[["record", "events", "retained", "removed", "left", "right", "classification_hard_valid"]]
    summary.columns = ["记录", "事件数", "保留", "删除", "左", "右", "分类硬质控保留"]
    text = f"""# 阶段一检查\n\n状态：通过。原始四条记录均为十通道，采样率二百五十六赫兹；每条记录的一百次提示均与缓存样本逐一对应。\n\n{markdown_table(summary)}\n\n统一片段形状为 {arrays['X'].shape}，通道顺序为左额、中额、右额。实际离散窗口为 {TIME[0]:.6f} 至 {TIME[-1]:.6f} 秒；与问题一相同，基线采用零时刻前的五十一个采样点，零时刻属于刺激响应。最大基线均值误差为 {baseline_error:.3g}。\n\n主平均曲线复用问题一清洗片段与剔除规则，但不复用全记录估计的软权重；各条件先在记录内平均，再对记录等权平均。幅值单位未知，不擅自标成微伏。\n\n分类专用片段沿用问题一的滤波、小波和基线处理方法，但按连续二十试次块独立滤波，每个片段独立估计高频小波阈值，避免训练和验证块互相影响。分类不沿用全记录软异常阈值：训练折估计阈值只筛训练样本；测试集保留全部预先固定硬质控合格样本，并报告覆盖率。\n\n任务一没有明确点击事件，应答、反应时间和正确性不补造。任务二记录点击方向、相对提示及后续方向标记的时间。题面不足以确认后续方向标记是否等于正确点击位置，因此所有正确性字段保留缺失，同时另存两种方向一致性，不能将它们当作正确率。记录组甲、乙不等同于两个被试。\n\n图形用途：四类平均曲线用于检查事件锁定及任务差异，不作为单试次分类成绩。每个面板是一个条件，三条线对应三电极；不绘制未估计的置信区间。\n"""
    (out / "stage1_check.md").write_text(text, encoding="utf-8")
    dump(out / "model" / "data_audit.json", dict(recordings=audit, baseline_max_error=baseline_error,
          excluded_channels_zero_based=[3, 4, 5], validation_block_size=20, shape=list(arrays["X"].shape)))
    print(summary.to_string(index=False), flush=True)
    print(f"Stage 1 passed: {arrays['X'].shape}; baseline error={baseline_error:.3g}", flush=True)
    return clean, validation, meta


def condition_means(X, meta, indices, device):
    import torch
    result = []
    for task, label in CONDITIONS:
        parts = []
        for group in sorted(meta.iloc[indices].group.unique()):
            ids = indices[(meta.iloc[indices].task.to_numpy() == task) & (meta.iloc[indices].y.to_numpy() == label) & (meta.iloc[indices].group.to_numpy() == group)]
            if len(ids) < 3:
                raise ValueError("Insufficient training trials for one condition")
            parts.append(torch.as_tensor(X[ids], device=device, dtype=torch.float64).mean(0))
        result.append(torch.stack(parts).mean(0).cpu().numpy())
    return np.stack(result)


def training_quality(X, indices):
    z = X[indices]
    f = np.column_stack([np.sqrt(np.mean(z[..., TIME < 0]**2, axis=(1, 2))), np.ptp(z, axis=-1).max(1)])
    med = np.median(f, axis=0)
    limit = med + 8*np.maximum(1.4826*np.median(np.abs(f-med), axis=0), 1e-8)
    return indices[(f <= limit).all(1)], limit


def erp_features(X, times=TIME):
    values, names = [], []
    for lo, hi, tag in [(.25, .5, "中期"), (.5, .8, "晚期")]:
        use = (times >= lo) & (times <= hi)
        z, t = X[..., use], times[use]
        for c, ch in enumerate(CHANNELS):
            for key, v in [("峰幅", z[:, c].max(-1)), ("峰时", t[z[:, c].argmax(-1)]),
                           ("均幅", z[:, c].mean(-1)), ("面积", np.trapezoid(z[:, c], t, axis=-1))]:
                values.append(v); names.append(f"{ch}{tag}{key}")
        values += [(z[:, 0]-z[:, 2]).mean(-1), (z[:, 1]-(z[:, 0]+z[:, 2])/2).mean(-1)]
        names += [f"{tag}双侧差", f"{tag}中额与双侧差"]
    return np.column_stack(values), names



def verify_results(X, meta, out, device):
    import inspect
    import pickle
    import subprocess
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix
    from model import predict, mixing, model_features, all_features, PARAMETERS, LOW, HIGH, CLASSIFIERS, metrics
    checks = {}
    data = np.load(out / "model" / "epochs.npz")
    checks["shape_and_classes"] = bool(data["X"].shape[1:] == (3, 257) and set(data["y"]) == {-1, 1})
    checks["baseline"] = bool(np.abs(data["X"][..., TIME < 0].mean(-1)).max() < 1e-8)
    checks["finite_validation_epochs"] = bool(np.isfinite(X).all())
    checks["behavior_missing_not_fabricated"] = bool(meta.correct.isna().all() and meta.loc[meta.task == 1, "response"].isna().all())
    checks["features_no_label_argument"] = "y" not in inspect.signature(model_features).parameters
    folds = json.loads((out / "model" / "validation_manifest.json").read_text(encoding="utf-8"))
    selection = pd.read_csv(out / "tables" / "validation_selection.csv")
    predictions = pd.read_csv(out / "tables" / "selected_test_predictions.csv")
    for fold in folds:
        tag = fold["fold"]
        tr, te = fold["outer_train_indices"], fold["test_indices"]
        checks[tag+"_record_disjoint"] = not (set(meta.iloc[tr].group) & set(meta.iloc[te].group))
        checks[tag+"_test_complete"] = set(te) == set(np.flatnonzero((meta.group == meta.group.iloc[te[0]]) & meta.hard_keep))
        a, b = fold["inner_train_indices"], fold["inner_validation_indices"]
        checks[tag+"_blocks_disjoint"] = not (set(zip(meta.iloc[a].record, meta.iloc[a].block)) & set(zip(meta.iloc[b].record, meta.iloc[b].block)))
        checks[tag+"_inner_inside_outer"] = set(a+b).issubset(tr)
        best = selection[selection.fold == tag].sort_values("balanced_accuracy", ascending=False, kind="stable").iloc[0]
        checks[tag+"_selection_train_only"] = best.scheme == fold["selected_scheme"] and best.classifier == fold["selected_classifier"]
        with (out / "model" / f"{tag}_classifier.pkl").open("rb") as f:
            saved = pickle.load(f)
        model = saved["model"]
        pred, sources = predict(model)
        checks[tag+"_source_zero_before_stimulus"] = bool(np.all(sources[..., TIME <= 0] == 0))
        checks[tag+"_source_bounded"] = bool(np.max(np.abs(sources)) <= 1)
        checks[tag+"_direction_swap"] = bool(np.allclose(sources[0], sources[1][[1, 0, 2]]) and np.allclose(sources[2], sources[3][[1, 0, 2]]))
        checks[tag+"_midline_invariant"] = bool(np.allclose(pred[0, 1], pred[1, 1]) and np.allclose(pred[2, 1], pred[3, 1]))
        checks[tag+"_bounds"] = bool(np.all(np.asarray(model["theta"]) >= LOW) and np.all(np.asarray(model["theta"]) <= HIGH) and np.max(np.abs(model["coef"])) <= model["mixing_coefficient_bound"]+1e-8)
        checks[tag+"_not_all_data_fit"] = set(model["training_indices"]).issubset(tr) and not set(model["training_indices"]) & set(te)
        features, _ = all_features(X[te], meta.task.to_numpy()[te], model, device)
        z = predictions[predictions.fold == tag].sort_values("index")
        reproduced = saved["pipeline"].predict(features[saved["feature_scheme"]])
        checks[tag+"_prediction_reproduction"] = bool(np.array_equal(reproduced, z.prediction))
        train_features = pd.read_csv(out / "tables" / f"{tag}_{saved['feature_scheme']}_features_train.csv").iloc[:, 4:].to_numpy()
        checks[tag+"_scaler_train_only"] = bool(np.allclose(saved["pipeline"][0].mean_, train_features.mean(0)))
    primary = json.loads((out / "model" / "primary_metrics.json").read_text(encoding="utf-8"))
    cm = confusion_matrix(predictions.y, predictions.prediction, labels=[-1, 1])
    checks["reported_confusion"] = cm.ravel().tolist() == [primary[k] for k in ["left_left", "left_right", "right_left", "right_right"]]
    checks["reported_balanced_accuracy"] = bool(np.isclose(balanced_accuracy_score(predictions.y, predictions.prediction), primary["balanced_accuracy"]))
    checks["each_test_trial_once"] = bool(not predictions["index"].duplicated().any() and len(predictions) == meta.hard_keep.sum())
    audits = []
    auditor = ROOT / "Ques1" / "qa" / "audit_figure_collisions.py"
    for path in sorted((out / "figures").glob("*.pdf")):
        result = subprocess.run([sys.executable, str(auditor), str(path), "--json-out", str(path.with_suffix(".collision.json"))], capture_output=True, text=True)
        audits.append(dict(file=path.name, returncode=result.returncode, summary=result.stdout[-1000:]))
    checks["seven_figures_pass_geometry_and_collision"] = len(audits) == 7 and all(row["returncode"] == 0 for row in audits)
    checks["all_alignment_manifests_present"] = len(list((out / "figures").glob("*.alignment.json"))) == 7
    dump(out / "figure_audit.json", audits)
    dump(out / "model" / "verification.json", checks)
    if not all(checks.values()):
        raise AssertionError(f"Verification failed: {[k for k, v in checks.items() if not v]}")
    print(f"Verification passed: {len(checks)} checks; seven figure audits", flush=True)


def write_report(out):
    import importlib.metadata
    import platform
    fitting = pd.read_csv(out / "tables" / "fit_metrics.csv")
    scores = pd.read_csv(out / "tables" / "classification_metrics.csv")
    primary = json.loads((out / "model" / "primary_metrics.json").read_text(encoding="utf-8"))
    model = json.loads((out / "model" / "stage2_A_final.json").read_text(encoding="utf-8"))
    mechanism = json.loads((out / "model" / "mechanism_diagnostics.json").read_text(encoding="utf-8"))
    folds = json.loads((out / "model" / "validation_manifest.json").read_text(encoding="utf-8"))
    from model import PARAMETERS
    meanings = ["外侧膝状体输入上升时间", "输入衰减与上升时间之差", "方向源时间常数", "公共源时间常数", "非偏好输入比例", "方向源输入增益", "任务一公共输入延迟", "任务二公共输入延迟", "任务一公共源增益", "任务二公共源增益"]
    parameters = pd.DataFrame(dict(参数=PARAMETERS, 含义=meanings, 估计值=model["theta"]))
    fit_summary = fitting.groupby(["scope", "task"])[["rmse", "r2", "late_rmse"]].mean().reset_index()
    fit_summary.columns = ["数据范围", "任务", "均方根误差", "平均决定系数", "晚期均方根误差"]
    fit_summary["数据范围"] = fit_summary["数据范围"].replace({"train_A":"甲记录训练", "heldout_B":"乙记录外部检查"})
    main_scores = scores[scores.fold.isin(["A_to_B", "B_to_A"])].copy()
    main_scores["fold"] = main_scores.fold.replace({"A_to_B":"甲训练乙测试", "B_to_A":"乙训练甲测试"})
    main_scores["scheme"] = main_scores.scheme.replace({"erp":"传统脑电", "source":"反演源", "scalp":"重建头皮", "combined":"传统与源联合"})
    main_scores["classifier"] = main_scores.classifier.replace({"logistic":"逻辑回归", "lda":"线性判别", "svm":"线性支持向量机"})
    main_scores = main_scores[["fold", "scheme", "classifier", "balanced_accuracy", "accuracy", "macro_f1", "roc_auc", "selected_by_validation"]]
    main_scores.columns = ["测试方向", "特征方案", "分类器", "平衡准确率", "准确率", "宏平均分数", "曲线下面积", "内部验证选定"]
    text = r"""# 问题二最小可行版本：实测报告

## 一、结论先行

三个阶段已经实际运行，但没有达到“可靠左右解码”或“全面良好拟合”的目标。模型能描述任务二晚期上升的总体趋势，任务一与跨记录拟合明显不足；严格测试的分类主结果接近随机。该版本是可复现的机制假说基线，不是已验证的神经机制，更不是疾病诊断模型。

## 二、数据与处理

只使用原始前三脑电通道，重排为左额、中额、右额；设备滤波通道完全禁用。四记录各一百次刺激；问题一质控保留三百零八次，统一片段为三通道、二百五十七采样点。所有原始文件与既有缓存的散列值、事件起点和原始基线片段逐一核验。

复用问题一滤波参数：五十赫兹陷波、零点一至三十赫兹带通、高频小波软阈值和刺激前基线。描述性图复用原清洗片段。严格分类与生成模型验证则将相同方法限制在连续二十次刺激块内，且小波阈值逐片段独立估计，避免零相位滤波、小波阈值跨验证边界传递信息。两套片段具有相同通道、采样网格、单位和元数据，但处理边界不同，不能把它们的平均幅值不加说明地混用。

描述性平均采用记录内等权试次、记录间等权，不沿用问题一的全记录软权重。分类与模型验证预先仅按饱和、平坦、缺失硬质控保留三百一十九次；软异常阈值只由训练样本估计并只用于训练筛除。测试样本不按模型误差或预测结果剔除，因此样本数与描述性图不同。

任务一缺少正负二点击，反应时间保留缺失。任务二保存点击方向、相对提示时间、相对后续方向标记时间。由于后续方向标记的语义不足以确认正确点击位置，正确性不编造，另存方向一致性供问题三核对。甲、乙仅称记录组。

## 三、模型结构与参数

输入核为 $p(t)=e^{-t/\tau_d}-e^{-t/\tau_r}$（非负时间），负时间取零，且 $\tau_d=\tau_r+\Delta_\tau$。偏好方向得到完整输入，另一方向得到比例 $\rho$ 的输入。

源动力学使用零基线平滑非线性 $\sigma(x)=\tanh(x)$：

$$\tau_s\dot q_L=-q_L+\tanh(g_su_L+0.12q_C),\quad
\tau_s\dot q_R=-q_R+\tanh(g_su_R+0.12q_C),$$

$$\tau_c\dot q_C=-q_C+\tanh[g_{c,m}p(t-\delta_m)+0.10(q_L+q_R)].$$

任务仅允许公共输入延迟和增益不同；左右始终共享动力学。可选扩展在公共源前加入固定零点一秒的一阶上升环节，仅内部训练验证显著改善时选用。本次两方向均保留简单一阶模型。

头皮观测 $Y=GQ$，基线偏置固定为零：

$$G=\begin{bmatrix}a&b&c\\d&d&e\\b&a&c\end{bmatrix}.$$

该矩阵是低维等效神经源到头皮电极的混合矩阵，不是三维定位或经过头部解剖验证的导联场。输入十个动力学参数、五个混合系数，共十五个自由参数。固定弱反馈参数只为减少自由度，不代表由数据辨识所得。

拟合在服务器处理器上使用有界非线性最小二乘；每次三初值、每初值最多二百二十次求值，数值差分的额外残差计算不计入该计数。混合系数通过带岭惩罚和左右差惩罚的线性子问题求解，并限制在训练幅度尺度的正负二十倍。全窗口参与拟合，中期适度加权，任务间使用训练幅度归一权重。条件平均和单试次投影在启动时最空闲显卡计算，小样本分类器在服务器处理器训练。

"""
    text += markdown_table(parameters)
    text += "\n\n五个混合系数、所有上下界、边界命中和各初值结果见参数表及模型文件。多初值参数范围较大，属于不稳定拟合，不能声称找到了唯一的生理参数。\n\n## 四、拟合效果\n\n"
    text += markdown_table(fit_summary)
    text += f"""\n\n该表是各条件、电极指标的算术平均，不等于将所有曲线拼接后计算的总决定系数。原表逐条件报告七类指标，并增加晚期窗口误差。曲线均显示零至八百毫秒，而不是仅展示较好窗口。\n\n对称假设要求左右刺激的中额预测完全一致、双侧预测镜像，因此不能拟合真实左右条件的整体幅度差。另存无关动力学形式的对称拟合上限，区分结构缺陷与优化失败。外部负决定系数说明跨记录适配失败，不能用训练任务二的较高分数掩盖。\n\n## 五、分类验证\n\n{markdown_table(main_scores)}\n\n每个外层训练记录内部，以前四个时间块训练、最后一个时间块验证；边界两侧各留一个试次间隔。两次外层均重新计算训练平均、拟合生成模型、生成特征、估计标准化，再训练分类器。测试特征函数只接收脑电、已知任务编号和训练模型，不接收刺激方向。\n\n主方案由内部验证选择，两次均选反演源特征配线性支持向量机。合并测试准确率 **{primary['accuracy']:.2%}**，平衡准确率 **{primary['balanced_accuracy']:.2%}**，宏平均分数 **{primary['macro_f1']:.4f}**。两方向曲线下面积的等权平均 **{primary['roc_auc']:.4f}**；不合并不同外层未校准的评分来计算面积。左右召回率分别为 **{primary['left_recall']:.2%}**、**{primary['right_recall']:.2%}**。\n\n混淆矩阵为 $\\begin{{bmatrix}}{primary['left_left']}&{primary['left_right']}\\\\{primary['right_left']}&{primary['right_right']}\\end{{bmatrix}}$，行是真实左、右，列是预测左、右。甲训练乙测试覆盖 {folds[0]['test_hard_retained']}/200 次，乙训练甲测试覆盖 {folds[1]['test_hard_retained']}/200 次。\n\n表中传统或联合特征的部分测试成绩较高，但不能据此倒过来改变主方案。仅一个末段验证块会有选择方差；没有额外嵌套验证、置信区间或显著性检验，不能宣称略高于随机的行具有可靠优势。跨任务合并模型没有把任务编号直接作为分类特征；任务仅用于调用对应训练模板，任务分层指标另存。\n\n## 六、左右源与头皮差异的解释边界\n\n左右刺激在模型中交换两个偏好源，公共源不变。因此源差 $q_L-q_R$ 改变符号，而頭皮差异满足 $y_{{F3}}-y_{{F4}}=(a-b)(q_L-q_R)$。本次甲记录训练的反对称模态增益相对于最大奇异增益为 **{mechanism['relative_antisymmetric_gain']:.4f}**，混合矩阵条件数为 **{mechanism['condition_number']:.2f}**。这与模型内部的方向信号衰减一致，但部分来自对称结构与弱差异惩罚，不能独立验证体积传导就是真实弱解码的原因。\n\n生成源曲线使用给定刺激方向属于前向模拟；若把测试标签输入模拟器后再声称源分类接近百分之一百，就是标签泄漏。本版本没有进行这种虚假分类。实际八维源特征由未知方向的单试次脑电同时投影到三个训练字典并比较两个候选模板得到，时间偏移只从预定三值选择。无噪声模板相对间距另外保存，但不作为真实准确率。确定且可逆的线性缩放本身不会凭空抹掉可分性；衰减、混合病态与噪声必须共同讨论。\n\n## 七、局限与问题三接口\n\n- 三个额区电极不足以识别真实视觉皮层或外侧膝状体的活动，也不足以区分晚期慢波、眼动及准备活动。\n- 只有两个身份未知记录组；跨记录差异大，任务一拟合不足。不能外推到被试人群、精神疾病或临床诊断。\n- 多初值动力学解不稳定，源尺度与混合矩阵混淆；此处的潜在源仅为可解释低维表征。\n- 点击在当前窗口以外，正确性尚不明确；问题三需要独立确认事件编码，再扩展到应答前窗口，不能把方向一致性直接当正确性。\n- 保留统一片段、逐试次元数据、训练折清单、模型参数、分类器、全部折外预测和特征矩阵，均是复现所需正式产物，不是临时缓存。\n\n可复用接口：`model.predict(model, times)` 返回四条件头皮曲线及三源曲线；`model.model_features(X, tasks, model, device)` 返回无测试标签依赖的源特征、名称和重建头皮；`utils.erp_features(X, times)` 返回传统窗口特征。`epochs.npz` 是统一描述性片段；`validation_epochs.npz` 与 `trials.csv` 联合使用保留全部四百事件及固定筛除信息。\n\n## 八、复现与检查\n\n三个阶段检查报告、逐曲线指标、逐试次预测、图形源表全部保留。最终主入口会再次检查训练测试隔离、源交换对称性、参数范围、禁止标签输入、分类器复现、训练标准化统计及图形几何。实际运行环境与源文件散列见运行清单。仅加载本项目可信模型文件，不加载来源不明的序列化文件。\n"""
    text = text.replace("頭皮", "头皮")
    (out / "report.md").write_text(text, encoding="utf-8")
    versions = {n:importlib.metadata.version(n) for n in ["numpy", "scipy", "pandas", "scikit-learn", "torch", "numba", "matplotlib", "PyWavelets"]}
    hashes = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}
    dump(out / "model" / "environment.json", dict(python=platform.python_version(), platform=platform.platform(), versions=versions, source_sha256=hashes))

