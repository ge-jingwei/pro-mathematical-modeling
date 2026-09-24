"""Audited inputs, purged block splits, descriptive diagnostics, and reporting."""
from pathlib import Path
import hashlib
import json
import pickle
import subprocess
import sys
import numpy as np
import pandas as pd
from scipy.special import ndtr
from source_features import HELPER, MODEL, TIME, ROOT, feature_sets, feature_names
from classifier import metrics, evaluate_classifier, transform

DISPLAY = {"erp":"传统脑电", "source":"反演源", "combined":"联合特征", "selected":"内层选定"}


def dump(path, value):
    def convert(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=convert, allow_nan=False), encoding="utf-8")


def load_data(cache, data, out):
    audit = json.loads((cache/"model/data_audit.json").read_text(encoding="utf-8"))
    meta = pd.read_csv(cache/"tables/trials.csv")
    with np.load(cache/"model/validation_epochs.npz") as saved:
        X, times = saved["X"].copy(), saved["times"].copy()
    if X.shape != (len(meta), 3, 257) or not np.array_equal(times, TIME) or not np.isfinite(X).all():
        raise ValueError("Invalid validation epoch cache")
    if not np.allclose(X[..., TIME < 0].mean(-1), 0., atol=1e-8):
        raise ValueError("Invalid baseline")
    rows = []
    for record in audit["recordings"]:
        path = data/(record["record"]+".mat")
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Changed raw input: {path.name}")
        raw, starts, group, task, variable = HELPER.load_record(path)
        z = meta[meta.record == record["record"]]
        if not np.array_equal(starts, z["sample"]) or not np.array_equal(raw[7, starts], z.y):
            raise ValueError("Cached trial labels disagree with raw event transitions")
        if set(z.group) != {group} or set(z.task) != {task} or not np.array_equal(z.block, z.trial//20):
            raise ValueError("Invalid recording or block metadata")
        rows.append(dict(record=record["record"], total=len(z), retained=int(z.hard_keep.sum()),
                         excluded=int((~z.hard_keep).sum()), left=int(((z.y == -1)&z.hard_keep).sum()), right=int(((z.y == 1)&z.hard_keep).sum()), raw_sha256=record["sha256"]))
    if not set(meta.y) == {-1, 1} or not meta.hard_keep.dtype == bool:
        raise ValueError("Unexpected labels or hard quality mask")
    meta.to_csv(out/"tables/trials.csv", index=False)
    pd.DataFrame(rows).to_csv(out/"tables/input_audit.csv", index=False)
    dump(out/"model/data_provenance.json", dict(cache_sha256=hashlib.sha256((cache/"model/validation_epochs.npz").read_bytes()).hexdigest(),
         trials_sha256=hashlib.sha256((cache/"tables/trials.csv").read_bytes()).hexdigest(),
         channels_used_zero_based=[1, 0, 2], forbidden_channels=[3, 4, 5], shape=X.shape,
         original_stage2=json.loads((cache/"model/mechanism_diagnostics.json").read_text(encoding="utf-8")),
         baseline_max_error=float(np.max(np.abs(X[..., TIME < 0].mean(-1))))))
    print(pd.DataFrame(rows).drop(columns="raw_sha256").to_string(index=False), flush=True)
    return X, meta


def purge(train, heldout, meta):
    allowed = np.ones(len(train), dtype=bool)
    for record in meta.iloc[heldout].record.unique():
        blocks = meta.iloc[heldout][meta.iloc[heldout].record == record].block.unique()
        targets = meta.loc[(meta.record == record)&meta.block.isin(blocks), "trial"].to_numpy()
        mask = meta.iloc[train].record.to_numpy() == record
        distances = np.abs(meta.iloc[train].trial.to_numpy()[mask, None]-targets[None])
        allowed[mask] &= distances.min(1) > 1
    return np.asarray(train)[allowed]


def outer_splits(meta):
    for group, other in [("A", "B"), ("B", "A")]:
        train = np.flatnonzero((meta.group == group)&meta.hard_keep)
        test = np.flatnonzero((meta.group == other)&meta.hard_keep)
        yield f"cross_{group}_to_{other}", "cross", group, train, test
    for group in ["A", "B"]:
        indices = np.flatnonzero((meta.group == group)&meta.hard_keep)
        for block in range(5):
            test = indices[meta.block.to_numpy()[indices] == block]
            train = indices[meta.block.to_numpy()[indices] != block]
            yield f"within_{group}_block{block}", "within", group, purge(train, test, meta), test


def save_state(path, state):
    dump(path, state)


def separation(z, y):
    left, right = z[y == -1], z[y == 1]
    difference = left.mean(0)-right.mean(0)
    within = (np.var(left, axis=0, ddof=1)+np.var(right, axis=0, ddof=1))/2
    covariance = (np.cov(left, rowvar=False)+np.cov(right, rowvar=False))/2
    squared = float(difference@np.linalg.pinv(covariance, rcond=1e-12)@difference)
    return dict(fisher_trace_ratio=float(np.sum(difference**2)/max(np.sum(within), 1e-12)),
                squared_mahalanobis=max(squared, 0.),
                gaussian_plugin_accuracy=float(ndtr(np.sqrt(max(squared, 0.))/2)))


def mechanism_rows(X, qtrain, qtest, state, train, test, meta, mode, fold, alpha, operator):
    rows = []
    use = (TIME >= .25)&(TIME <= .8)
    for split, ids, q in [("train", train, qtrain), ("test", test, qtest)]:
        for task in [1, 2]:
            mask = meta.task.to_numpy()[ids] == task
            y = meta.y.to_numpy()[ids][mask]
            for space, values in [("scalp", X[ids][mask]), ("source", q[mask])]:
                z = values[..., use].mean(-1)
                rows.append(dict(mode=mode, fold=fold, split=split, task=task, space=space, n=len(y), **separation(z, y)))
    _, templates = MODEL.predict(state["model"])
    G = state["G"]
    spec = np.linalg.svd(G, compute_uv=False)
    for task in [1, 2]:
        k = 2*(task-1)
        for space, values in [("specified_source", templates), ("simulated_scalp", np.einsum("cs,kst->kct", G, templates))]:
            delta = values[k, :, use]-values[k+1, :, use]
            denom = np.linalg.norm(values[k, :, use])+np.linalg.norm(values[k+1, :, use])
            rows.append(dict(mode=mode, fold=fold, split="noise_free_assumption", task=task, space=space, n=2,
                             relative_distance=float(np.linalg.norm(delta)/max(denom, 1e-12)),
                             deterministic_template_accuracy=1. if np.linalg.norm(delta) > 1e-10 else .5,
                             relative_antisymmetric_gain=float(abs(G[0, 0]-G[0, 1])/max(spec[0], 1e-12)),
                             condition_number=float(spec[0]/max(spec[-1], 1e-12)), operator_condition=float(np.linalg.cond(operator)), alpha=alpha))
    return rows


def aggregate(predictions):
    rows = []
    for mode in ["cross", "within"]:
        for group in ["all", "A", "B"]:
            z = predictions[predictions["mode"].eq(mode)]
            if group != "all": z = z[z.group == group]
            for scheme in ["erp", "source", "combined", "selected"]:
                v = z[z.chosen] if scheme == "selected" else z[z.scheme == scheme]
                row = metrics(v.y, v.prediction, v.score)
                per_fold = [metrics(f.y, f.prediction, f.score) for _, f in v.groupby("fold")]
                row["auc"] = float(np.mean([r["auc"] for r in per_fold]))
                row["fold_balanced_accuracy_mean"] = float(np.mean([r["balanced_accuracy"] for r in per_fold]))
                row["fold_balanced_accuracy_std"] = float(np.std([r["balanced_accuracy"] for r in per_fold], ddof=1)) if len(per_fold) > 1 else 0.
                row["folds"] = len(per_fold)
                rows.append(dict(mode=mode, group=group, scheme=scheme, **row))
    return pd.DataFrame(rows)


def verify(out, X, meta, device):
    from source_features import source_features, fit_state, inverse_operator
    import inspect
    folds = json.loads((out/"model/split_manifest.json").read_text(encoding="utf-8"))
    predicted = pd.read_csv(out/"tables/predictions.csv")
    checks = dict(twelve_outer_folds=len(folds) == 12, source_features_no_label_argument="y" not in inspect.signature(source_features).parameters,
                  finite_input=bool(np.isfinite(X).all()), raw_three_channels=X.shape[1] == 3)
    for fold in folds:
        name, train, fit, test = fold["tag"], fold["train"], fold["fit"], fold["test"]
        checks[name+"_disjoint"] = not set(train)&set(test) and set(fit).issubset(train)
        checks[name+"_outer_gap"] = np.array_equal(purge(np.array(train), test, meta), train)
        for inner in fold["inner"]:
            a, b, c = inner["training"], inner["validation"], inner["fitting"]
            checks[name+f"_inner{inner['block']}"] = not set(a)&set(b) and set(a+b).issubset(train) and set(c).issubset(a) and np.array_equal(purge(np.array(a), b, meta), a)
            state = json.loads((out/"model"/(inner["state"]+".json")).read_text(encoding="utf-8"))
            if set(state["training_indices"]) != set(c): raise AssertionError("Wrong inner fitted model membership")
        if fold["mode"] == "cross":
            checks[name+"_record_isolation"] = not set(meta.iloc[train].group)&set(meta.iloc[test].group)
        else:
            checks[name+"_block_isolation"] = not set(zip(meta.iloc[train].record, meta.iloc[train].block))&set(zip(meta.iloc[test].record, meta.iloc[test].block))
        for scheme in ["erp", "source", "combined"]:
            stem = f"{name}_{scheme}"
            with (out/"model"/(stem+".pkl")).open("rb") as stream:
                artifact = pickle.load(stream)
            bundle, state = artifact["bundle"], artifact["state"]
            f = np.load(out/"model"/(stem+"_features.npz"))
            features, _, operator, _ = feature_sets(X[test], meta.task.to_numpy()[test], state, artifact["alpha"], device)
            yp, scores = evaluate_classifier(bundle, features[scheme], meta.task.to_numpy()[test])
            p = predicted[(predicted.fold == name)&(predicted.scheme == scheme)].sort_values("index")
            checks[stem+"_reproduce"] = bool(np.array_equal(yp, p.prediction) and np.allclose(scores, p.score) and np.allclose(features[scheme], f["test"]))
            checks[stem+"_dimensions"] = bundle["size"] <= 15 and f["train"].shape[1] <= 15
            train_tasks = meta.task.to_numpy()[fit]
            state_ok = all(np.allclose(bundle["transformer"]["task_means"][int(t)], f["train"][train_tasks == t].mean(0)) for t in np.unique(train_tasks))
            centered = f["train"]-np.stack([bundle["transformer"]["task_means"][int(t)] for t in train_tasks])
            checks[stem+"_train_only_scaling"] = bool(state_ok and np.allclose(bundle["transformer"]["scaler"].mean_, centered.mean(0)))
            baseline = X[fit][..., TIME < 0].transpose(1, 0, 2).reshape(3, -1)
            cov = np.cov(baseline); cov = .9*cov+.1*np.trace(cov)/3*np.eye(3)
            checks[stem+"_train_only_noise"] = bool(np.allclose(cov, state["noise_covariance"]))
            A = state["whitening"]@state["G"]
            lhs = (A.T@A+artifact["regularization"]*np.eye(3))@operator
            rhs = A.T@state["whitening"]
            checks[stem+"_ridge_normal_equation"] = bool(np.allclose(lhs, rhs, rtol=1e-7, atol=1e-9))
        search = pd.read_csv(out/"tables"/f"{name}_inner_search.csv")
        for scheme in ["erp", "source", "combined"]:
            winner = search[search.scheme == scheme].sort_values(["balanced_accuracy", "size"], ascending=[False, True], kind="stable").iloc[0]
            recorded = fold["winners"][scheme]
            checks[name+scheme+"_inner_selection"] = all(np.isclose(winner[k], recorded[k]) for k in ["alpha", "size", "parameter", "balanced_accuracy"]) and winner.family == recorded["family"]
    for mode in ["cross", "within"]:
        for scheme in ["erp", "source", "combined"]:
            z = predicted[(predicted["mode"] == mode)&(predicted.scheme == scheme)]
            checks[mode+scheme+"_all_test_trials_once"] = set(z["index"]) == set(np.flatnonzero(meta.hard_keep)) and not z["index"].duplicated().any()
    separation_table = pd.read_csv(out/"tables/separability.csv")
    actual = separation_table[separation_table.split.isin(["train", "test"])]
    paired = actual.pivot(index=["mode", "fold", "split", "task"], columns="space", values="squared_mahalanobis")
    checks["linear_inverse_mahalanobis_invariance"] = bool(np.allclose(paired.scalp, paired.source, rtol=1e-6, atol=1e-9))
    for fold in folds:
        search = fold["winners"]
        selected = max(["erp", "source", "combined"], key=lambda name:search[name]["balanced_accuracy"])
        checks[fold["tag"]+"_scheme_choice_train_only"] = fold["selected_scheme"] == selected
    dump(out/"model/verification.json", checks)
    if not all(checks.values()):
        raise AssertionError([k for k, value in checks.items() if not value])
    return len(checks)


def figures(out, summary, predictions, folds):
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, roc_curve
    colors = {"erp":"#357AA1", "source":"#C98245", "combined":"#8F6B96", "selected":"#558878"}
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.6))
    fig.subplots_adjust(left=.09, right=.98, bottom=.18, top=.85, wspace=.35)
    for ax, mode, title in zip(axs, ["cross", "within"], ["跨记录：两次外层测试", "组内时间块：十次外层测试"]):
        z = summary[(summary["mode"] == mode)&(summary.group == "all")]
        for i, scheme in enumerate(["erp", "source", "combined", "selected"]):
            row = z[z.scheme == scheme].iloc[0]
            individual = folds[(folds["mode"] == mode)&(folds.chosen if scheme == "selected" else folds.scheme == scheme)]
            ax.bar(i, row.balanced_accuracy, color=colors[scheme], alpha=.8, width=.6)
            ax.scatter(np.full(len(individual), i)+np.linspace(-.12, .12, len(individual)), individual.balanced_accuracy, s=16, color="#333333", zorder=3)
        ax.axhline(.5, color=".4", ls=":", lw=.8); ax.axhline(.65, color=".5", ls="--", lw=.8)
        ax.set(xticks=range(4), xticklabels=[DISPLAY[s] for s in ["erp", "source", "combined", "selected"]], ylabel="平衡准确率", ylim=(0, 1), title=title)
    HELPER.save(fig, out/"figures/accuracy_comparison")
    fig, axs = plt.subplots(2, 3, figsize=(10.8, 7))
    fig.subplots_adjust(left=.08, right=.98, bottom=.09, top=.94, hspace=.45, wspace=.4)
    for r, mode in enumerate(["cross", "within"]):
        for c, scheme in enumerate(["erp", "source", "combined"]):
            ax = axs[r, c]
            z = predictions[(predictions["mode"] == mode)&(predictions.scheme == scheme)]
            cm = confusion_matrix(z.y, z.prediction, labels=[-1, 1])
            ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max()*1.7)
            for i in range(2):
                for j in range(2): ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=15)
            ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["左", "右"], yticklabels=["左", "右"], xlabel="预测", ylabel="真实", title=("跨记录 · " if r == 0 else "组内 · ")+DISPLAY[scheme])
    HELPER.save(fig, out/"figures/confusion_matrices")
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.6))
    fig.subplots_adjust(left=.09, right=.98, bottom=.15, top=.9, wspace=.3)
    roc_rows = []
    for ax, mode in zip(axs, ["cross", "within"]):
        for scheme in ["erp", "source", "combined"]:
            curves = []
            grid = np.linspace(0, 1, 101)
            z = predictions[(predictions["mode"] == mode)&(predictions.scheme == scheme)]
            for fold, v in z.groupby("fold"):
                fpr, tpr, _ = roc_curve(v.y, v.score)
                curves.append(np.interp(grid, fpr, tpr))
                roc_rows += [dict(mode=mode, scheme=scheme, fold=fold, false_positive=float(f), true_positive=float(t)) for f, t in zip(fpr, tpr)]
            mean_curve = np.mean(curves, axis=0)
            auc = summary[(summary["mode"] == mode)&(summary.group == "all")&(summary.scheme == scheme)].auc.iloc[0]
            ax.plot(grid, mean_curve, color=colors[scheme], label=f"{DISPLAY[scheme]}（{auc:.3f}）")
        ax.plot([0, 1], [0, 1], color=".5", ls="--", lw=.8)
        ax.set(xlabel="假阳性率", ylabel="真阳性率", xlim=(0, 1), ylim=(0, 1), title="跨记录平均曲线" if mode == "cross" else "组内时间块平均曲线")
        ax.legend(loc="lower right", fontsize=8)
    HELPER.save(fig, out/"figures/roc_comparison")
    pd.DataFrame(roc_rows).to_csv(out/"tables/roc_curves.csv", index=False)


def finalize(out, X, meta, device):
    predicted = pd.read_csv(out/"tables/predictions.csv")
    folds = pd.read_csv(out/"tables/fold_metrics.csv")
    summary = aggregate(predicted)
    summary.to_csv(out/"tables/summary_metrics.csv", index=False)
    task_rows = []
    for (mode, fold, scheme, task), z in predicted.groupby(["mode", "fold", "scheme", "task"], sort=False):
        task_rows.append(dict(mode=mode, fold=fold, scheme=scheme, task=task, **metrics(z.y, z.prediction, z.score)))
    pd.DataFrame(task_rows).to_csv(out/"tables/task_metrics.csv", index=False)
    recording_diagnostics(out, X, meta)
    figures(out, summary, predicted, folds)
    checks = verify(out, X, meta, device)
    audits = []
    for path in sorted((out/"figures").glob("*.pdf")):
        result = subprocess.run([sys.executable, str(ROOT/"Ques1/qa/audit_figure_collisions.py"), str(path), "--json-out", str(path.with_suffix(".collision.json"))], capture_output=True, text=True)
        audits.append(dict(file=path.name, returncode=result.returncode, detail=result.stdout))
    dump(out/"figure_audit.json", audits)
    if not all(r["returncode"] == 0 for r in audits):
        raise AssertionError("Figure collision audit failed")
    write_report(out, summary, folds, checks)
    import importlib.metadata
    dump(out/"model/final_environment.json", dict(versions={n:importlib.metadata.version(n) for n in ["numpy", "scipy", "scikit-learn", "torch", "pandas", "matplotlib", "numba"]}, source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}))
    print(summary[summary.group == "all"][["mode", "scheme", "accuracy", "balanced_accuracy", "auc"]].to_string(index=False), flush=True)
    print(f"Verified {checks} checks and {len(audits)} figures", flush=True)


def write_report(out, summary, folds, checks):
    def row(mode, scheme): return summary[(summary["mode"] == mode)&(summary.group == "all")&(summary.scheme == scheme)].iloc[0]
    source_cross, source_within = row("cross", "source"), row("within", "source")
    main_cross, main_within = row("cross", "selected"), row("within", "selected")
    displayed = summary[summary.group == "all"][["mode", "scheme", "accuracy", "balanced_accuracy", "macro_f1", "auc"]].copy()
    displayed["mode"] = displayed["mode"].replace({"cross":"跨记录", "within":"组内时间块"})
    displayed["scheme"] = displayed.scheme.map(DISPLAY)
    displayed.columns = ["验证方式", "特征", "准确率", "平衡准确率", "宏平均分数", "曲线下面积"]
    cross = folds[folds["mode"] == "cross"][["fold", "scheme", "accuracy", "balanced_accuracy", "macro_f1", "auc", "size", "family"]].copy()
    cross["fold"] = cross.fold.replace({"cross_A_to_B":"甲训练乙测试", "cross_B_to_A":"乙训练甲测试"})
    cross["scheme"] = cross.scheme.map(DISPLAY)
    cross["family"] = cross.family.replace({"logistic":"逻辑回归", "svm":"线性支持向量机", "lda":"线性判别"})
    cross.columns = ["测试方向", "特征", "准确率", "平衡准确率", "宏平均分数", "曲线下面积", "维数", "分类器"]
    sep = pd.read_csv(out/"tables/separability.csv")
    separations = sep[sep.split.isin(["train", "test"])].groupby(["mode", "split", "space"])[["fisher_trace_ratio", "squared_mahalanobis", "gaussian_plugin_accuracy"]].mean().reset_index()
    separations.to_csv(out/"tables/separability_summary.csv", index=False)
    empirical = folds.groupby(["mode", "scheme"])[["accuracy", "balanced_accuracy", "auc"]].agg(["min", "mean", "max"])
    empirical.to_csv(out/"tables/observed_performance_range.csv")
    text = f"""# 问题二分类准确率迭代：实测报告\n\n## 一、结论\n\n本轮完成源轨迹岭反演、训练内特征筛选和嵌套时间块调参，两次跨记录外层及十次组内时间块外层全部执行。主方案只按各外层训练内部验证选择；跨记录准确率 **{main_cross.accuracy:.2%}**、平衡准确率 **{main_cross.balanced_accuracy:.2%}**，组内对应为 **{main_within.accuracy:.2%}**、**{main_within.balanced_accuracy:.2%}**。是否达到百分之六十五，以完整折外汇总为准，不用最好单折替代。\n\n{HELPER.markdown_table(displayed)}\n\n## 二、双向跨记录结果\n\n{HELPER.markdown_table(cross)}\n\n准确率、平衡准确率、宏平均分数由该方案所有折外预测汇总；曲线下面积报告各外层等权均值，未混合不同模型评分。另存每个外层、各记录组和各任务的指标。十个组内折是每个记录组同时留出两个任务的同序号时间块，不将另一个记录组加入组内训练。\n\n## 三、数据与嵌套验证\n\n仅复用问题二按连续二十次刺激块独立滤波的三原始通道片段，通道顺序为左额、中额、右额。四百事件中沿用预先固定硬质控的三百一十九个合格试次，原始四个文件散列、刺激转变、标签和块号逐一核对。禁用设备处理后的第四至第六通道。既有饱和、缺失、平坦排除规则不改变；测试集合完整保留，不按分数或错误删样本。\n\n外层两次跨记录测试；组内按五个原始二十试次块逐一留出，训练侧剔除与留出试次在原始完整留出块边界外相邻的一次刺激，不因边界试次质控排除而缩短间隔，测试侧不删。每个外层训练再遍历其全部可用时间块进行内层验证，内层也有相同训练侧间隔。所有条件平均、十五参数源模型、基线噪声协方差、任务中心化、标准化、稀疏特征排序、反演强度和分类超参数均只在对应训练划分计算。\n\n没有复用见过当前验证试次的旧参数；复用的是原三源方程、混合约束、边界和拟合程序。每个不同训练集合重新三初值拟合；仅完全相同的训练索引可复用本轮拟合缓存。\n\n必须披露：这两个记录组已用于此前问题二分析，本轮又在同一数据上进行预先指定的探索性迭代。折内不存在测试标签泄漏，但这不是全新的、从未被研究者观察过的确认性外部测试；百分之六十五目标若达到也需要新记录复验。\n\n## 四、源空间与特征\n\n从训练刺激前片段估计三通道噪声协方差并固定百分之十对角收缩。白化矩阵为 $W$，混合矩阵为 $G$，令 $A=WG$，采用\n\n$$\\hat Q=(A^\\mathsf{{T}}A+\\lambda I)^{{-1}}A^\\mathsf{{T}}WY,\\quad\\lambda=\\alpha\\lVert A\\rVert_2^2.$$\n\n候选相对反演强度为十万分之一、千分之一、百分之三、十分之三，仅内层平衡准确率选择。源时间轨迹不施加标签依赖约束，不从测试真方向生成反演输入。\n\n源方案共十二维：三源投影、左右差、归一指数、投影重建误差、两方向模板相关差、正负三个采样点内最佳偏移，以及源平均阿尔法和西塔对数能量、左右阿尔法能量差、左右阿尔法对数能量比。频谱用零至八百毫秒单片段去均值与线性趋势、汉宁窗离散谱估计；西塔四至八赫兹、阿尔法八至十三赫兹。相同反演算子用于训练与测试。\n\n传统方案十五维，包括三通道中期峰幅、均幅、峰时、晚期均幅及三项空间差。联合方案严格十五维，固定使用源方案前十维加五个头皮辅助特征；两个左右阿尔法差比特征只进入完整源方案，避免联合候选也超过十五维。联合头皮特征在标准化后乘零点五，增加其在线性正则模型中的相对代价；对协方差适配的线性判别，单纯重缩放不保证同样的降权效果，不能混为一谈。\n\n全部方案先用训练集内各任务均值中心化，再标准化。用固定惩罚强度的一范数逻辑回归排序，内层选择保留维数：传统八或十五，源六或十二，联合八、十二或十五。零系数并列按预先固定的特征次序处理。每个内层均独立重做排序。分类器仅为二范数逻辑回归、二范数线性支持向量机和收缩线性判别；前两者正则参数四档、后者收缩三档。不用测试结果增加候选或翻转标签。\n\n## 五、可分性与所谓分类上限\n\n保存训练和测试中的三通道平均源活动与头皮活动可分性，按任务分层：类均值距离平方除以类内方差之和，以及类内协方差度量的马氏距离平方。测试标签只在训练流程结束后用于这一描述性诊断，不反馈分类。还保存两种无噪声指定模板的相对距离。\n\n所谓“分类上限”不能由三百余个样本或一个混合矩阵的条件数被可靠计算。本报告提供两种明确标注的参照，而不冒称真实上界：第一，各外层实测准确率范围，仅是观测范围；第二，在等先验、同协方差高斯假设下的插件参照 $\\Phi(\\sqrt{{D^2}}/2)$。这些插件值使用同一批样本估计分离度，存在乐观偏差，尤其测试上的插件值不是可部署成绩。\n\n若反演算子可逆，三维线性变换前后的马氏距离在同一协方差度量下保持不变；源轨迹是脑电的确定变换，不能增加原观测中不存在的信息。岭正则可能改善有限样本的特征与估计稳定性，但不是恢复已丢失信息的保证。无噪声两个确定模板只要不同，两空间都能完全区分；差异幅度变小并不等于准确率必须接近随机。\n\n## 六、论文可用表述与边界\n\n“源空间特征在组内时间块交叉验证中的准确率为 **{source_within.accuracy:.2%}**，跨记录验证为 **{source_cross.accuracy:.2%}**。”\n\n不直接补写“降至”或“说明被非神经因素掩盖”：只有实际下降时才能描述下降，即便下降也不足以确定非神经因果因素。可能因素包括记录间漂移、接触或阻抗变化、非平稳脑状态、形态差异及样本少，但本数据没有阻抗测量或明确被试身份，不能把这些可能解释写成已证实事实。实际的同方向均值跨记录一致性和基线噪声差异可作为描述性线索。\n\n“问题二单次拟合的反对称模态相对增益约为0.0232、条件数约为43.09，提示所设混合模型对方向差异有衰减并可能放大反演噪声。本轮结果应与这一模型诊断并列讨论，不能声称其证明了低准确率的物理必然性或非神经因素的因果解释。”\n\n没有强行写成“源空间可靠高分类、头皮空间随机”的预设结论。多初值模型可能不稳定，源尺度不唯一；当前成绩必须以实际数表为准。\n\n## 七、产物与核验\n\n全部候选内层成绩、所有外层预测、划分清单、反演算子、噪声协方差、源模型、选择特征和分类器保存；三个方案所有外层均报告。当前通过 **{checks}** 项程序核验及三组图形几何与文字碰撞核验。折点显示折间波动，不是置信区间。服务器按空闲显存选择显卡，源反演和频谱计算使用显卡，生成模型有界优化与小样本分类器使用服务器处理器。\n"""
    diagnostic_table = separations.copy()
    diagnostic_table["mode"] = diagnostic_table["mode"].replace({"cross":"跨记录", "within":"组内时间块"})
    diagnostic_table["split"] = diagnostic_table["split"].replace({"train":"训练", "test":"测试描述"})
    diagnostic_table["space"] = diagnostic_table["space"].replace({"scalp":"头皮", "source":"反演源"})
    diagnostic_table.columns = ["验证", "范围", "空间", "方差比", "马氏距离平方", "高斯插件参照"]
    noise = pd.read_csv(out/"tables/recording_noise_diagnostics.csv")
    wave = pd.read_csv(out/"tables/cross_record_waveform_diagnostics.csv")
    same_distance = separations.pivot(index=["mode", "split"], columns="space", values="squared_mahalanobis")
    difference = float(np.max(np.abs(same_distance.scalp-same_distance.source)))
    plugin = separations[(separations["mode"] == "within")&(separations["split"] == "test")&(separations["space"] == "source")].gaussian_plugin_accuracy.iloc[0]
    text += "\n\n## 八、可分性与记录差异的实际数值\n\n"+HELPER.markdown_table(diagnostic_table)
    text += f"\n\n同一汇总层级中，头皮和反演源的马氏距离平方最大差异仅 {difference:.3g}，与可逆线性变换保持该距离的计算检查一致。组内小测试块的高斯插件参照为 {plugin:.2%}并不等于交叉验证准确率达到百分之六十五，不能作为达标结果。\n\n不同记录及电极刺激前均方根范围为 {noise.baseline_rms.min():.3f} 至 {noise.baseline_rms.max():.3f} 个原记录单位；十二个同任务同方向同电极的跨记录平均曲线相关中位数为 {wave.waveform_correlation.median():.3f}，范围 {wave.waveform_correlation.min():.3f} 至 {wave.waveform_correlation.max():.3f}。这些只反映记录之间存在差异，无法鉴别阻抗、眼动或神经状态的具体贡献。详细数表保留，不用于反向调参。\n\n汇总表中的跨记录甲、乙分组标识表示训练记录组，组内则表示该记录组；双向测试表已明确标注训练与测试方向。\n"
    grouped = summary[(summary["mode"] == "within")&(summary.group != "all")][["group", "scheme", "accuracy", "balanced_accuracy", "macro_f1", "auc"]].copy()
    grouped["group"] = grouped.group.replace({"A":"记录甲", "B":"记录乙"})
    grouped["scheme"] = grouped.scheme.map(DISPLAY)
    grouped.columns = ["记录组", "方案", "准确率", "平衡准确率", "宏平均分数", "曲线下面积"]
    text += "\n\n## 九、记录组内分组结果\n\n"+HELPER.markdown_table(grouped)
    if max(main_cross.accuracy, main_within.accuracy, source_cross.accuracy, source_within.accuracy) < .65:
        text += "\n\n结论：本轮完整外层验证未达到百分之六十五目标。单个高分折和高斯插件参照均不能替代这一结果。\n"
    (out/"report_iter.md").write_text(text, encoding="utf-8")



def recording_diagnostics(out, X, meta):
    rows, paired = [], []
    post = (TIME >= 0)&(TIME <= .8)
    for record, frame in meta.groupby("record", sort=True):
        ids = frame.index[frame.hard_keep].to_numpy()
        z = X[ids]
        baseline = z[..., TIME < 0]
        for c, channel in enumerate(HELPER.CHANNELS):
            rows.append(dict(record=record, group=frame.group.iloc[0], task=int(frame.task.iloc[0]), channel=channel, n=len(ids),
                             baseline_rms=float(np.sqrt(np.mean(baseline[:, c]**2))),
                             poststimulus_rms=float(np.sqrt(np.mean(z[:, c, post]**2))),
                             left_fraction=float((meta.y.iloc[ids] == -1).mean())))
    for task in [1, 2]:
        for direction in [-1, 1]:
            means = []
            for group in ["A", "B"]:
                ids = np.flatnonzero((meta.group == group)&(meta.task == task)&(meta.y == direction)&meta.hard_keep)
                means.append(X[ids].mean(0))
            for c, channel in enumerate(HELPER.CHANNELS):
                a, b = means[0][c, post], means[1][c, post]
                paired.append(dict(task=task, direction=direction, channel=channel, waveform_correlation=float(np.corrcoef(a, b)[0, 1]),
                                   waveform_rmse=float(np.sqrt(np.mean((a-b)**2)))))
    pd.DataFrame(rows).to_csv(out/"tables/recording_noise_diagnostics.csv", index=False)
    pd.DataFrame(paired).to_csv(out/"tables/cross_record_waveform_diagnostics.csv", index=False)
    return pd.DataFrame(rows), pd.DataFrame(paired)
