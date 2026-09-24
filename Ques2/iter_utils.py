"""Audited inputs, purged block splits, descriptive diagnostics, and reporting."""
from pathlib import Path
import hashlib
import json
import pickle
import numpy as np
import pandas as pd
from scipy.special import ndtr
from Ques2.source_features import HELPER, MODEL, TIME, ROOT, feature_sets, feature_names
from Ques2.classifier import metrics, evaluate_classifier, transform

DISPLAY = {"erp":"传统脑电", "source":"反演源", "combined":"联合特征", "selected":"内层选定"}


def dump(path, value):
    def convert(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=convert, allow_nan=False), encoding="utf-8")


def load_data(cache, data, out):
    audit = json.loads((cache/"data_audit.json").read_text(encoding="utf-8"))
    meta = pd.read_csv(cache/"trials.csv")
    with np.load(cache/"validation_epochs.npz") as saved:
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
    meta.to_csv(out / "nested_trials.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "nested_input_audit.csv", index=False)
    dump(out / "nested_data_provenance.json", dict(cache_sha256=hashlib.sha256((cache/"validation_epochs.npz").read_bytes()).hexdigest(),
         trials_sha256=hashlib.sha256((cache/"trials.csv").read_bytes()).hexdigest(),
         channels_used_zero_based=[1, 0, 2], forbidden_channels=[3, 4, 5], shape=X.shape,
         original_stage2=json.loads((cache/"mechanism_diagnostics.json").read_text(encoding="utf-8")),
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
    from Ques2.source_features import source_features, fit_state, inverse_operator
    import inspect
    folds = json.loads((out / "nested_split_manifest.json").read_text(encoding="utf-8"))
    predicted = pd.read_csv(out / "nested_predictions.csv")
    checks = dict(twelve_outer_folds=len(folds) == 12, source_features_no_label_argument="y" not in inspect.signature(source_features).parameters,
                  finite_input=bool(np.isfinite(X).all()), raw_three_channels=X.shape[1] == 3)
    for fold in folds:
        name, train, fit, test = fold["tag"], fold["train"], fold["fit"], fold["test"]
        checks[name+"_disjoint"] = not set(train)&set(test) and set(fit).issubset(train)
        checks[name+"_outer_gap"] = np.array_equal(purge(np.array(train), test, meta), train)
        for inner in fold["inner"]:
            a, b, c = inner["training"], inner["validation"], inner["fitting"]
            checks[name+f"_inner{inner['block']}"] = not set(a)&set(b) and set(a+b).issubset(train) and set(c).issubset(a) and np.array_equal(purge(np.array(a), b, meta), a)
            state = json.loads((out / ("nested_" + (inner["state"]+".json"))).read_text(encoding="utf-8"))
            if set(state["training_indices"]) != set(c): raise AssertionError("Wrong inner fitted model membership")
        if fold["mode"] == "cross":
            checks[name+"_record_isolation"] = not set(meta.iloc[train].group)&set(meta.iloc[test].group)
        else:
            checks[name+"_block_isolation"] = not set(zip(meta.iloc[train].record, meta.iloc[train].block))&set(zip(meta.iloc[test].record, meta.iloc[test].block))
        for scheme in ["erp", "source", "combined"]:
            stem = f"{name}_{scheme}"
            with (out / ("nested_" + (stem+".pkl"))).open("rb") as stream:
                artifact = pickle.load(stream)
            bundle, state = artifact["bundle"], artifact["state"]
            f = np.load(out / ("nested_" + (stem+"_features.npz")))
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
        search = pd.read_csv(out / f"nested_{name}_inner_search.csv")
        for scheme in ["erp", "source", "combined"]:
            winner = search[search.scheme == scheme].sort_values(["balanced_accuracy", "size"], ascending=[False, True], kind="stable").iloc[0]
            recorded = fold["winners"][scheme]
            checks[name+scheme+"_inner_selection"] = all(np.isclose(winner[k], recorded[k]) for k in ["alpha", "size", "parameter", "balanced_accuracy"]) and winner.family == recorded["family"]
    for mode in ["cross", "within"]:
        for scheme in ["erp", "source", "combined"]:
            z = predicted[(predicted["mode"] == mode)&(predicted.scheme == scheme)]
            checks[mode+scheme+"_all_test_trials_once"] = set(z["index"]) == set(np.flatnonzero(meta.hard_keep)) and not z["index"].duplicated().any()
    separation_table = pd.read_csv(out / "nested_separability.csv")
    actual = separation_table[separation_table.split.isin(["train", "test"])]
    paired = actual.pivot(index=["mode", "fold", "split", "task"], columns="space", values="squared_mahalanobis")
    checks["linear_inverse_mahalanobis_invariance"] = bool(np.allclose(paired.scalp, paired.source, rtol=1e-6, atol=1e-9))
    for fold in folds:
        search = fold["winners"]
        selected = max(["erp", "source", "combined"], key=lambda name:search[name]["balanced_accuracy"])
        checks[fold["tag"]+"_scheme_choice_train_only"] = fold["selected_scheme"] == selected
    dump(out / "nested_verification.json", checks)
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
    HELPER.save(fig, out / "nested_accuracy_comparison")
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
    HELPER.save(fig, out / "nested_confusion_matrices")
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
    HELPER.save(fig, out / "nested_roc_comparison")
    pd.DataFrame(roc_rows).to_csv(out / "nested_roc_curves.csv", index=False)


def finalize(out, X, meta, device):
    predicted = pd.read_csv(out / "nested_predictions.csv")
    folds = pd.read_csv(out / "nested_fold_metrics.csv")
    summary = aggregate(predicted)
    summary.to_csv(out / "nested_summary_metrics.csv", index=False)
    task_rows = []
    for (mode, fold, scheme, task), z in predicted.groupby(["mode", "fold", "scheme", "task"], sort=False):
        task_rows.append(dict(mode=mode, fold=fold, scheme=scheme, task=task, **metrics(z.y, z.prediction, z.score)))
    pd.DataFrame(task_rows).to_csv(out / "nested_task_metrics.csv", index=False)
    recording_diagnostics(out, X, meta)
    figures(out, summary, predicted, folds)
    import importlib.metadata
    dump(out / "nested_final_environment.json", dict(versions={n:importlib.metadata.version(n) for n in ["numpy", "scipy", "scikit-learn", "torch", "pandas", "matplotlib", "numba"]}, source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}))
    print(summary[summary.group == "all"][["mode", "scheme", "accuracy", "balanced_accuracy", "auc"]].to_string(index=False), flush=True)





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
    pd.DataFrame(rows).to_csv(out / "nested_recording_noise_diagnostics.csv", index=False)
    pd.DataFrame(paired).to_csv(out / "nested_cross_record_waveform_diagnostics.csv", index=False)
    return pd.DataFrame(rows), pd.DataFrame(paired)
