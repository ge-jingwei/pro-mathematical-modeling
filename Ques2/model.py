"""Constrained generative dynamics, train-only fitting, and honest decoding."""
from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd
import torch
from numba import njit
from scipy.optimize import least_squares, lsq_linear
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from Ques2.utils import TIME, CHANNELS, CONDITIONS, SEED, condition_means, training_quality, erp_features, dump, save

PARAMETERS = ["tau_r", "tau_gap", "tau_s", "tau_c", "rho", "g_s", "delta_1", "delta_2", "g_c1", "g_c2"]
LOW = np.array([.008, .025, .015, .025, .0, .2, .03, .03, .05, .05])
HIGH = np.array([.12, .75, .40, .65, .95, 5., .65, .70, 6., 8.])
DEFAULT = np.array([.045, .23, .085, .12, .2, 1.5, .20, .40, 1.0, 2.])
FEATURE_SCHEMES = ["erp", "source", "scalp", "combined"]
DISPLAY = {"erp": "传统脑电", "source": "反演源特征", "scalp": "重建头皮", "combined": "传统与源联合"}
CLASSIFIERS = {
    "logistic": LogisticRegression(C=.3, class_weight="balanced", max_iter=3000, random_state=SEED),
    "lda": LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
    "svm": LinearSVC(C=.1, class_weight="balanced", max_iter=10000, random_state=SEED, dual="auto"),
}


@njit(cache=False)
def sources(theta, times, cascade):
    tr, gap, ts, tc, rho, gs, d1, d2, gc1, gc2 = theta
    td = tr + gap
    Q = np.zeros((4, 3, len(times)))
    for k in range(4):
        delay, gc = (d1, gc1) if k < 2 else (d2, gc2)
        left = k % 2 == 0
        ql = qr = qc = zc = 0.
        for j in range(len(times)):
            if times[j] <= 0:
                continue
            dt = times[j] - max(times[j-1], 0.) if j else times[j]
            t = max(times[j] - dt/2, 0.)
            p = np.exp(-t/td) - np.exp(-t/tr)
            u = max(t-delay, 0.)
            pc = np.exp(-u/td) - np.exp(-u/tr) if t > delay else 0.
            drive_l = np.tanh(gs*p*(1. if left else rho) + .12*qc)
            drive_r = np.tanh(gs*p*(rho if left else 1.) + .12*qc)
            drive_c = np.tanh(gc*pc + .10*(ql+qr))
            ql += (1-np.exp(-dt/ts))*(drive_l-ql)
            qr += (1-np.exp(-dt/ts))*(drive_r-qr)
            if cascade:
                zc += (1-np.exp(-dt/.10))*(drive_c-zc)
                qc += (1-np.exp(-dt/tc))*(zc-qc)
            else:
                qc += (1-np.exp(-dt/tc))*(drive_c-qc)
            Q[k, 0, j], Q[k, 1, j], Q[k, 2, j] = ql, qr, qc
    return Q


def mixing(coef):
    a, b, c, d, e = coef
    return np.array([[a, b, c], [d, d, e], [b, a, c]])


def design(Q):
    A = np.zeros((4, 3, Q.shape[-1], 5))
    A[:, 0, :, 0], A[:, 0, :, 1], A[:, 0, :, 2] = Q[:, 0], Q[:, 1], Q[:, 2]
    A[:, 1, :, 3], A[:, 1, :, 4] = Q[:, 0]+Q[:, 1], Q[:, 2]
    A[:, 2, :, 0], A[:, 2, :, 1], A[:, 2, :, 2] = Q[:, 1], Q[:, 0], Q[:, 2]
    return A


def predict(model, times=TIME):
    Q = sources(np.asarray(model["theta"]), times, model["variant"] == "cascade")
    return np.einsum("cs,kst->kct", mixing(model["coef"]), Q), Q


def fit(erp, variant="simple", starts=3, max_nfev=220):
    times, target = TIME[::2], erp[..., ::2]
    use = (times >= 0) & (times <= .8)
    times, target = times[use], target[..., use]
    scale = max(float(np.sqrt(np.mean(target**2))), 1.)
    task_scale = np.array([max(np.sqrt(np.mean(target[:2]**2)), scale*.3)]*2 + [max(np.sqrt(np.mean(target[2:]**2)), scale*.3)]*2)
    time_weight = np.where((times >= .25) & (times <= .5), 1.5, 1.)
    weights = np.broadcast_to(np.sqrt(time_weight)[None, None, :]/task_scale[:, None, None], target.shape).ravel()
    target_flat = target.ravel()*weights
    regularizer = np.eye(5)*.035
    regularizer = np.vstack([regularizer, [.07, -.07, 0, 0, 0]])
    def solve(theta):
        Q = sources(theta, times, variant == "cascade")
        A = design(Q).reshape(-1, 5)*weights[:, None]*scale
        aug = np.vstack([A, regularizer])
        rhs = np.r_[target_flat, np.zeros(6)]
        coef = np.linalg.lstsq(aug, rhs, rcond=None)[0]
        if np.max(np.abs(coef)) > 20:
            coef = lsq_linear(aug, rhs, bounds=(-20, 20), method="bvls").x
        return coef, np.r_[A@coef-target_flat, regularizer@coef]
    def residual(theta):
        return solve(theta)[1]
    rng = np.random.default_rng(SEED)
    results = []
    for start in range(starts):
        x0 = DEFAULT.copy() if start == 0 else np.clip(DEFAULT*(1+rng.normal(0, .40, 10)), LOW+.001*(HIGH-LOW), HIGH-.001*(HIGH-LOW))
        result = least_squares(residual, x0, bounds=(LOW, HIGH), max_nfev=max_nfev,
                               x_scale="jac", ftol=2e-5, xtol=2e-5, gtol=2e-5)
        coef, r = solve(result.x)
        results.append(dict(theta=result.x.tolist(), coef=(coef*scale).tolist(), loss=float(np.mean(r**2)),
                            nfev=int(result.nfev), success=bool(result.success), status=int(result.status)))
    best = min(results, key=lambda x:x["loss"])
    model = dict(**best, variant=variant, starts=results, fixed_kappa=.12, fixed_eta=.10,
                 fixed_common_rise_s=.10 if variant == "cascade" else 0., offset=[0., 0., 0.],
                 free_parameters=15, mixing_ridge=.035, mixing_difference_penalty=.07, mixing_coefficient_bound=20*scale)
    pred, Q = predict(model)
    if not np.isfinite(pred).all() or np.max(np.abs(Q)) > 1.0001 or np.sqrt(np.mean(pred**2)) > 20*scale:
        raise ValueError("Generative model diverged")
    return model


def waveform_metrics(obs, pred, times=TIME):
    use = (times >= 0) & (times <= .8)
    middle = (times >= .25) & (times <= .5)
    late = (times >= .5) & (times <= .8)
    a, b = obs[use], pred[use]
    rmse = float(np.sqrt(np.mean((a-b)**2)))
    midt = times[middle]
    pa, pb = obs[middle], pred[middle]
    return dict(rmse=rmse, nrmse=rmse/max(float(np.ptp(a)), 1e-10),
                r2=float(1-np.sum((a-b)**2)/max(np.sum((a-a.mean())**2), 1e-10)),
                pearson=float(np.corrcoef(a, b)[0, 1]) if np.std(a)*np.std(b) > 1e-10 else 0.,
                window_rmse=float(np.sqrt(np.mean((pa-pb)**2))),
                late_rmse=float(np.sqrt(np.mean((obs[late]-pred[late])**2))),
                peak_amplitude_error=float(pb.max()-pa.max()),
                peak_latency_error_ms=float((midt[pb.argmax()]-midt[pa.argmax()])*1000),
                observed_peak_at_boundary=bool(pa.argmax() in [0, len(pa)-1]),
                predicted_peak_at_boundary=bool(pb.argmax() in [0, len(pb)-1]))


def fit_metrics(obs, pred, scope):
    return [dict(scope=scope, task=m, direction=d, channel=ch, **waveform_metrics(obs[k, c], pred[k, c]))
            for k, (m, d) in enumerate(CONDITIONS) for c, ch in enumerate(CHANNELS)]


def validation_loss(obs, pred):
    use = (TIME >= 0) & (TIME <= .8)
    a, b = obs[..., use], pred[..., use]
    scales = np.array([max(np.sqrt(np.mean(a[:2]**2)), 1.)]*2 + [max(np.sqrt(np.mean(a[2:]**2)), 1.)]*2)
    return float(np.mean(((a-b)/scales[:, None, None])**2))


def fit_training(X, meta, ids, device, variant, tag, out):
    kept, thresholds = training_quality(X, ids)
    erp = condition_means(X, meta, kept, device)
    print(f"Fit {tag}: {variant}; {len(kept)}/{len(ids)} training trials", flush=True)
    model = fit(erp, variant)
    model.update(training_indices=kept.tolist(), input_training_indices=ids.tolist(), training_thresholds=thresholds.tolist(), fit_tag=tag)
    dump(out / f"{tag}.json", model)
    pred, Q = predict(model)
    np.savez_compressed(out / f"{tag}_curves.npz", observed=erp, predicted=pred, sources=Q, times=TIME)
    return model, erp, kept


def inner_split(meta, train):
    rows = meta.iloc[train]
    inner = train[(rows.block.to_numpy() < 4) & (rows.trial.to_numpy() % 20 < 19)]
    validation = train[(rows.block.to_numpy() == 4) & (rows.trial.to_numpy() % 20 > 0)]
    if len(validation) < 15:
        raise ValueError("Insufficient held-out chronological validation trials")
    return inner, validation


def select_dynamics(X, meta, train, device, prefix, out):
    inner, valid = inner_split(meta, train)
    valid_erp = condition_means(X, meta, valid, device)
    base, _, inner_kept = fit_training(X, meta, inner, device, "simple", prefix+"_inner_simple", out)
    base_loss = validation_loss(valid_erp, predict(base)[0])
    rows = [dict(variant="simple", validation_loss=base_loss)]
    chosen = base
    if base_loss > .25:
        extended, _, _ = fit_training(X, meta, inner, device, "cascade", prefix+"_inner_cascade", out)
        extended_loss = validation_loss(valid_erp, predict(extended)[0])
        rows.append(dict(variant="cascade", validation_loss=extended_loss))
        if extended_loss < .95*base_loss:
            chosen = extended
    pd.DataFrame(rows).to_csv(out / f"{prefix}_dynamics_selection.csv", index=False)
    return chosen, inner_kept, valid, rows


def stage2(X, meta, out, device):
    import matplotlib.pyplot as plt
    train = np.flatnonzero((meta.group == "A") & meta.hard_keep)
    test = np.flatnonzero((meta.group == "B") & meta.hard_keep)
    selected, _, _, selection = select_dynamics(X, meta, train, device, "stage2_A", out)
    model, observed, kept = fit_training(X, meta, train, device, selected["variant"], "stage2_A_final", out)
    predicted, Q = predict(model)
    external = condition_means(X, meta, test, device)
    metrics = pd.DataFrame(fit_metrics(observed, predicted, "train_A") + fit_metrics(external, predicted, "heldout_B"))
    metrics.to_csv(out / "fit_metrics.csv", index=False)
    params = [dict(parameter=n, value=v, lower=l, upper=h, boundary=bool(min((v-l)/(h-l), (h-v)/(h-l)) < .01))
              for n, v, l, h in zip(PARAMETERS, model["theta"], LOW, HIGH)]
    params += [dict(parameter=n, value=v, lower=-model["mixing_coefficient_bound"], upper=model["mixing_coefficient_bound"], boundary=bool(abs(v) > .99*model["mixing_coefficient_bound"])) for n, v in zip(["a", "b", "c", "d", "e"], model["coef"])]
    pd.DataFrame(params).to_csv(out / "parameters.csv", index=False)
    G = mixing(model["coef"])
    pd.DataFrame(G, index=CHANNELS, columns=["L", "R", "C"]).to_csv(out / "mixing_matrix.csv")
    curves = []
    for k, (m, d) in enumerate(CONDITIONS):
        for c, ch in enumerate(CHANNELS):
            curves += [dict(task=m, direction=d, channel=ch, time_s=t, train_observed=a, heldout_observed=b, fitted=v)
                       for t, a, b, v in zip(TIME, observed[k, c], external[k, c], predicted[k, c])]
    pd.DataFrame(curves).to_csv(out / "fit_curves.csv", index=False)
    source_rows = [dict(task=m, direction=d, source=s, time_s=t, value=v)
                   for (m, d), q in zip(CONDITIONS, Q) for s, curve in zip(["L", "R", "C"], q) for t, v in zip(TIME, curve)]
    pd.DataFrame(source_rows).to_csv(out / "latent_sources.csv", index=False)
    for scope, obs, name in [("训练记录甲", observed, "stage2_fit_train"), ("留出记录乙", external, "stage2_fit_heldout")]:
        fig, axs = plt.subplots(4, 3, figsize=(11, 11))
        fig.subplots_adjust(left=.075, right=.98, bottom=.06, top=.93, hspace=.65, wspace=.32)
        for k, (m, d) in enumerate(CONDITIONS):
            for c, ch in enumerate(CHANNELS):
                ax = axs[k, c]
                ax.plot(TIME*1000, obs[k, c], color="#357AA1", label="真实平均")
                ax.plot(TIME*1000, predicted[k, c], color="#C98245", label="模型预测")
                ax.axvspan(250, 500, color=".92", zorder=0)
                ax.axhline(0, color=".6", lw=.6)
                ax.set_title(f"任务{m} {'左' if d == -1 else '右'}刺激 · {ch}")
                ax.set(xlabel="时间（毫秒）", ylabel="原记录单位", xlim=(-200, 800))
        fig.suptitle(scope+"：四条件三电极", y=.987, fontsize=12)
        fig.legend(*axs[0, 0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5, .974), ncol=2)
        save(fig, out / name)
    fig, axs = plt.subplots(2, 2, figsize=(10, 6.5))
    fig.subplots_adjust(left=.09, right=.98, bottom=.11, top=.89, hspace=.45, wspace=.3)
    for k, (ax, (m, d)) in enumerate(zip(axs.flat, CONDITIONS)):
        for c, (label, color) in enumerate(zip(["左选择源", "右选择源", "公共源"], ["#357AA1", "#C98245", "#8F6B96"])):
            ax.plot(TIME*1000, Q[k, c], color=color, label=label)
        ax.set(title=f"任务{m} · {'左' if d == -1 else '右'}刺激", xlabel="时间（毫秒）", ylabel="无量纲源活动", xlim=(-200, 800), ylim=(-.02, 1.02))
    fig.legend(*axs[0, 0].get_legend_handles_labels(), loc="upper center", ncol=3)
    save(fig, out / "stage2_sources")
    ceiling = []
    for m in [1, 2]:
        i = 2*(m-1)
        use = (TIME >= 0) & (TIME <= .8)
        a, b = observed[i][..., use], observed[i+1][..., use]
        ideal = (a+b[[2, 1, 0]])/2
        sse = np.sum((a-ideal)**2)+np.sum((b-ideal[[2, 1, 0]])**2)
        total = np.sum((a-a.mean(-1, keepdims=True))**2)+np.sum((b-b.mean(-1, keepdims=True))**2)
        ceiling.append(dict(task=m, symmetry_minimum_rmse=float(np.sqrt(sse/a.size/2)),
                            symmetry_maximum_r2=float(1-sse/total)))
    pd.DataFrame(ceiling).to_csv(out / "symmetry_ceiling.csv", index=False)
    spectrum = np.linalg.svd(G, compute_uv=False)
    contrast_gain = abs(G[0, 0]-G[0, 1])/max(spectrum[0], 1e-12)
    diag = dict(relative_antisymmetric_gain=float(contrast_gain), singular_values=spectrum.tolist(),
                condition_number=float(spectrum[0]/max(spectrum[-1], 1e-12)),
                exact_middle_direction_invariance=bool(np.allclose(predicted[0, 1], predicted[1, 1]) and np.allclose(predicted[2, 1], predicted[3, 1])),
                theta_multistart_range=(np.ptp([s["theta"] for s in model["starts"]], axis=0)/(HIGH-LOW)).tolist())
    dump(out / "mechanism_diagnostics.json", diag)
    print("Stage 2 completed:", metrics.groupby(["scope", "task"])[["rmse", "r2"]].mean().to_string(), flush=True)
    return model


def model_features(X, tasks, model, device):
    pred, Q = predict(model)
    G = mixing(model["coef"])
    select = (TIME >= 0) & (TIME <= .8)
    coeff = np.zeros((len(X), 3))
    errors = np.zeros((len(X), 2))
    shifts_best = np.zeros(len(X))
    reconstruction = np.zeros_like(X)
    shifts = [-10/256, 0., 10/256]
    for task in [1, 2]:
        ids = np.flatnonzero(tasks == task)
        if len(ids) == 0:
            continue
        k = 2*(task-1)
        preferred = Q[k, 0]
        dictionary = np.stack([G[:, 0, None]*preferred, G[:, 1, None]*preferred, G[:, 2, None]*Q[k, 2]], axis=-1)
        xt = torch.as_tensor(X[ids][..., select].reshape(len(ids), -1), device=device, dtype=torch.float64)
        best = np.full(len(ids), np.inf)
        direction_errors = np.full((len(ids), 2), np.inf)
        for shift in shifts:
            B = np.empty_like(dictionary)
            for c in range(3):
                for s in range(3):
                    B[c, :, s] = np.interp(TIME-shift, TIME, dictionary[c, :, s], left=0., right=0.)
            bt = torch.as_tensor(B[:, select].reshape(-1, 3), device=device, dtype=torch.float64)
            gram = bt.T@bt
            ridge = .02*torch.trace(gram)/3 + 1e-8
            weights = torch.linalg.solve(gram+ridge*torch.eye(3, device=device, dtype=torch.float64), bt.T@xt.T).T
            estimate = weights@bt.T
            mse = ((xt-estimate)**2).mean(1).cpu().numpy()
            improve = mse < best
            best[improve] = mse[improve]
            w = weights.cpu().numpy()
            coeff[ids[improve]] = w[improve]
            reconstruction[ids[improve]] = np.einsum("ns,cts->nct", w[improve], B)
            shifts_best[ids[improve]] = shift
            for d in range(2):
                template = np.stack([np.interp(TIME-shift, TIME, v, left=0., right=0.) for v in pred[k+d]])[:, select].ravel()
                t = torch.as_tensor(template, device=device, dtype=torch.float64)
                amplitude = torch.clamp((xt@t)/(t@t+1e-8), min=0.)
                err = ((xt-amplitude[:, None]*t)**2).mean(1).sqrt().cpu().numpy()
                direction_errors[:, d] = np.minimum(direction_errors[:, d], err)
        errors[ids] = direction_errors
    difference = coeff[:, 0]-coeff[:, 1]
    normalized = difference/(np.abs(coeff[:, 0])+np.abs(coeff[:, 1])+1e-6)
    features = np.column_stack([coeff, difference, normalized, errors, shifts_best])
    names = ["左源投影", "右源投影", "公共源投影", "左右源差", "归一选择指数", "左模板误差", "右模板误差", "最佳时间偏移"]
    return features, names, reconstruction


def all_features(X, tasks, model, device):
    erp, erp_names = erp_features(X)
    source, source_names, reconstructed = model_features(X, tasks, model, device)
    scalp, scalp_names = erp_features(reconstructed)
    return dict(erp=erp, source=source, scalp=scalp, combined=np.column_stack([erp, source])), dict(
        erp=erp_names, source=source_names, scalp=["重建"+n for n in scalp_names], combined=erp_names+source_names)


def metrics(y, pred, score):
    cm = confusion_matrix(y, pred, labels=[-1, 1])
    return dict(accuracy=float(accuracy_score(y, pred)), balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                macro_f1=float(f1_score(y, pred, labels=[-1, 1], average="macro", zero_division=0)),
                roc_auc=float(roc_auc_score(y, score)), left_recall=float(cm[0, 0]/cm[0].sum()),
                right_recall=float(cm[1, 1]/cm[1].sum()), n=len(y),
                left_left=int(cm[0, 0]), left_right=int(cm[0, 1]), right_left=int(cm[1, 0]), right_right=int(cm[1, 1]))


def template_separation(model):
    predicted, Q = predict(model)
    use = (TIME >= 0) & (TIME <= .8)
    rows = []
    for task in [1, 2]:
        k = 2*(task-1)
        for space, values in [("specified_source", Q), ("simulated_scalp", predicted)]:
            left, right = values[k][:, use], values[k+1][:, use]
            delta = np.linalg.norm(left-right)
            denom = np.linalg.norm(left)+np.linalg.norm(right)
            rows.append(dict(task=task, space=space, relative_template_distance=float(delta/max(denom, 1e-12)),
                             deterministic_distinct=bool(delta > 1e-9)))
    return rows


def stage3(X, meta, out, device):
    import matplotlib.pyplot as plt
    selected_predictions, all_predictions, comparisons, tuning, manifests, coefficient_rows, mechanisms = [], [], [], [], [], [], []
    for group in ["A", "B"]:
        other = "B" if group == "A" else "A"
        tag = f"{group}_to_{other}"
        train = np.flatnonzero((meta.group == group) & meta.hard_keep)
        test = np.flatnonzero((meta.group == other) & meta.hard_keep)
        if np.intersect1d(train, test).size:
            raise AssertionError("Outer train-test overlap")
        selected, inner_kept, valid, dynamics_selection = select_dynamics(X, meta, train, device, tag, out)
        inner_blocks = set(zip(meta.iloc[inner_kept].record, meta.iloc[inner_kept].block))
        valid_blocks = set(zip(meta.iloc[valid].record, meta.iloc[valid].block))
        if inner_blocks & valid_blocks:
            raise AssertionError("Chronological validation block overlap")
        fi, names = all_features(X[inner_kept], meta.task.to_numpy()[inner_kept], selected, device)
        fv, _ = all_features(X[valid], meta.task.to_numpy()[valid], selected, device)
        yi, yv = meta.y.to_numpy()[inner_kept], meta.y.to_numpy()[valid]
        winners = {}
        for scheme in FEATURE_SCHEMES:
            scores = []
            for classifier, estimator in CLASSIFIERS.items():
                pipeline = make_pipeline(StandardScaler(), clone(estimator))
                pipeline.fit(fi[scheme], yi)
                predicted, score = pipeline.predict(fv[scheme]), pipeline.decision_function(fv[scheme])
                row = dict(fold=tag, scheme=scheme, classifier=classifier, **metrics(yv, predicted, score))
                tuning.append(row); scores.append(row)
            winners[scheme] = max(scores, key=lambda r:r["balanced_accuracy"])
        best_scheme = max(FEATURE_SCHEMES, key=lambda s:winners[s]["balanced_accuracy"])
        final, observed, train_kept = fit_training(X, meta, train, device, selected["variant"], tag+"_final", out)
        ft, names = all_features(X[train_kept], meta.task.to_numpy()[train_kept], final, device)
        fe, _ = all_features(X[test], meta.task.to_numpy()[test], final, device)
        yt, ye = meta.y.to_numpy()[train_kept], meta.y.to_numpy()[test]
        for split, ids, fs in [("train", train_kept, ft), ("test", test, fe)]:
            for scheme in FEATURE_SCHEMES:
                frame = pd.DataFrame(fs[scheme], columns=names[scheme])
                frame.insert(0, "index", ids); frame.insert(1, "split", split)
                frame.insert(2, "y", meta.y.to_numpy()[ids]); frame.insert(3, "task", meta.task.to_numpy()[ids])
                frame.to_csv(out / f"{tag}_{scheme}_features_{split}.csv", index=False)
        for scheme in FEATURE_SCHEMES:
            classifier = winners[scheme]["classifier"]
            pipeline = make_pipeline(StandardScaler(), clone(CLASSIFIERS[classifier]))
            pipeline.fit(ft[scheme], yt)
            pred, score = pipeline.predict(fe[scheme]), pipeline.decision_function(fe[scheme])
            row = dict(fold=tag, scheme=scheme, classifier=classifier, validation_balanced_accuracy=winners[scheme]["balanced_accuracy"],
                       selected_by_validation=scheme == best_scheme, **metrics(ye, pred, score))
            comparisons.append(row)
            for task in [1, 2]:
                mask = meta.task.to_numpy()[test] == task
                comparisons.append(dict(fold=tag+f"_task{task}", scheme=scheme, classifier=classifier,
                     validation_balanced_accuracy=winners[scheme]["balanced_accuracy"], selected_by_validation=scheme == best_scheme,
                     **metrics(ye[mask], pred[mask], score[mask])))
            records = [dict(index=int(idx), fold=tag, scheme=scheme, classifier=classifier, task=int(meta.task.iloc[idx]),
                            trial=int(meta.trial.iloc[idx]), group=other, y=int(y), prediction=int(p), score=float(s))
                       for idx, y, p, s in zip(test, ye, pred, score)]
            all_predictions.extend(records)
            if scheme == best_scheme:
                selected_predictions.extend(records)
                coefficients = pipeline[-1].coef_[0]
                coefficient_rows += [dict(fold=tag, scheme=scheme, feature=n, coefficient=float(c)) for n, c in zip(names[scheme], coefficients)]
                with (out / f"{tag}_classifier.pkl").open("wb") as handle:
                    pickle.dump(dict(pipeline=pipeline, feature_scheme=scheme, names=names[scheme], model=final), handle)
            print(f"Stage 3 {tag}: {scheme}/{classifier}; validation={winners[scheme]['balanced_accuracy']:.3f}; test={row['balanced_accuracy']:.3f}", flush=True)
        manifest = dict(fold=tag, outer_train_indices=train.tolist(), final_training_indices=train_kept.tolist(), test_indices=test.tolist(),
                        inner_train_indices=inner_kept.tolist(), inner_validation_indices=valid.tolist(),
                        selected_scheme=best_scheme, selected_classifier=winners[best_scheme]["classifier"], selected_variant=selected["variant"],
                        candidate_schemes=FEATURE_SCHEMES, classifier_candidates=list(CLASSIFIERS),
                        total_test_recordings_trials=int((meta.group == other).sum()), test_hard_retained=len(test),
                        test_coverage=float(len(test)/(meta.group == other).sum()),
                        preprocessing="independent contiguous twenty-trial blocks; per-epoch wavelet thresholds",
                        test_soft_rejection=False, ground_truth_used_for_features=False)
        manifests.append(manifest)
        mechanisms += [dict(fold=tag, **r) for r in template_separation(final)]
    table = pd.DataFrame(comparisons)
    table.to_csv(out / "classification_metrics.csv", index=False)
    pd.DataFrame(tuning).to_csv(out / "validation_selection.csv", index=False)
    allpred = pd.DataFrame(all_predictions)
    allpred.to_csv(out / "all_test_predictions.csv", index=False)
    chosen = pd.DataFrame(selected_predictions)
    chosen.to_csv(out / "selected_test_predictions.csv", index=False)
    pd.DataFrame(coefficient_rows).to_csv(out / "classifier_coefficients.csv", index=False)
    pd.DataFrame(mechanisms).to_csv(out / "template_separation.csv", index=False)
    dump(out / "validation_manifest.json", manifests)
    primary = metrics(chosen.y, chosen.prediction, chosen.score)
    primary["roc_auc"] = float(np.mean([roc_auc_score(z.y, z.score) for _, z in chosen.groupby("fold")]))
    primary["roc_auc_note"] = "unweighted mean of outer-fold AUC; decision score scales are not pooled"
    primary["cross_recording_dependence_note"] = "two directions use different fitted models; pooled confusion is descriptive, not independent replication"
    dump(out / "primary_metrics.json", primary)
    cm = confusion_matrix(chosen.y, chosen.prediction, labels=[-1, 1])
    fig, ax = plt.subplots(figsize=(5.1, 4.6))
    fig.subplots_adjust(left=.18, right=.92, bottom=.17, top=.86)
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max()*1.25)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=20)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["左刺激", "右刺激"], yticklabels=["左刺激", "右刺激"],
           xlabel="预测类别", ylabel="真实类别", title="仅按内部验证选定：跨记录混淆矩阵")
    save(fig, out / "stage3_confusion")
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.5))
    fig.subplots_adjust(left=.08, right=.97, bottom=.17, top=.88, wspace=.32)
    roc_rows = []
    for ax, (fold, sub) in zip(axs, allpred.groupby("fold", sort=True)):
        for scheme, color in zip(FEATURE_SCHEMES, ["#357AA1", "#C98245", "#8F6B96", "#558878"]):
            z = sub[sub.scheme == scheme]
            fpr, tpr, _ = roc_curve(z.y, z.score)
            auc = roc_auc_score(z.y, z.score)
            ax.plot(fpr, tpr, label=f"{DISPLAY[scheme]}（{auc:.3f}）", color=color)
            roc_rows += [dict(fold=fold, scheme=scheme, false_positive_rate=float(f), true_positive_rate=float(t)) for f, t in zip(fpr, tpr)]
        ax.plot([0, 1], [0, 1], color=".6", ls="--")
        ax.set(title="甲训练乙测试" if fold == "A_to_B" else "乙训练甲测试", xlabel="假阳性率", ylabel="真阳性率", xlim=(0, 1), ylim=(0, 1))
        ax.legend(loc="lower right", fontsize=8)
    save(fig, out / "stage3_roc")
    pd.DataFrame(roc_rows).to_csv(out / "roc_curves.csv", index=False)
    fig, axs = plt.subplots(1, 2, figsize=(11, 5))
    fig.subplots_adjust(left=.19, right=.98, bottom=.16, top=.86, wspace=.90)
    coef = pd.DataFrame(coefficient_rows)
    for ax, (fold, z) in zip(axs, coef.groupby("fold", sort=True)):
        z = z.iloc[np.argsort(np.abs(z.coefficient))[-7:]]
        ax.barh(np.arange(len(z)), z.coefficient, color="#357AA1")
        ax.set_yticks(np.arange(len(z)), z.feature, fontsize=8)
        ax.axvline(0, color=".5", lw=.8)
        ax.set(xlabel="标准化特征的线性系数", title="甲训练乙测试" if fold == "A_to_B" else "乙训练甲测试")
    save(fig, out / "stage3_coefficients")
    print("Stage 3 completed:", json.dumps(primary), flush=True)
    return primary

