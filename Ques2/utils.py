"""Data audit, reused preprocessing, fold-local features, and plotting helpers."""
from pathlib import Path
import hashlib
import json
import re
import numpy as np
import pandas as pd
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
from Ques1.src.preprocess import filter_record, wavelet_high_only, baseline, TIME
from Ques1.src.plotting import setup, save
from Ques1.src.inference import choose_device

CHANNELS = ["F3", "Fz", "F4"]
CONDITIONS = [(1, -1), (1, 1), (2, -1), (2, 1)]
SEED = 20260924


def dump(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")




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
    np.savez_compressed(out / "epochs.npz", **arrays)
    np.savez_compressed(out / "validation_epochs.npz", X=validation, times=TIME)
    meta.to_csv(out / "trials.csv", index=False)
    pd.DataFrame(audit).drop(columns=["raw_shape"]).to_csv(out / "recording_counts.csv", index=False)
    erps = condition_means(clean, meta, np.flatnonzero(keep), device)
    rows = [dict(task=m, direction=d, channel=ch, time_s=t, amplitude=v)
            for (m, d), wave in zip(CONDITIONS, erps) for ch, curve in zip(CHANNELS, wave) for t, v in zip(TIME, curve)]
    pd.DataFrame(rows).to_csv(out / "erp_curves.csv", index=False)
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
    save(fig, out / "stage1_erp")
    dump(out / "data_audit.json", dict(recordings=audit, baseline_max_error=baseline_error,
          excluded_channels_zero_based=[3, 4, 5], validation_block_size=20, shape=list(arrays["X"].shape)))
    print(pd.DataFrame(audit).to_string(index=False), flush=True)
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
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix
    from Ques2.model import predict, mixing, model_features, all_features, PARAMETERS, LOW, HIGH, CLASSIFIERS, metrics
    checks = {}
    data = np.load(out / "epochs.npz")
    checks["shape_and_classes"] = bool(data["X"].shape[1:] == (3, 257) and set(data["y"]) == {-1, 1})
    checks["baseline"] = bool(np.abs(data["X"][..., TIME < 0].mean(-1)).max() < 1e-8)
    checks["finite_validation_epochs"] = bool(np.isfinite(X).all())
    checks["behavior_missing_not_fabricated"] = bool(meta.correct.isna().all() and meta.loc[meta.task == 1, "response"].isna().all())
    checks["features_no_label_argument"] = "y" not in inspect.signature(model_features).parameters
    folds = json.loads((out / "validation_manifest.json").read_text(encoding="utf-8"))
    selection = pd.read_csv(out / "validation_selection.csv")
    predictions = pd.read_csv(out / "selected_test_predictions.csv")
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
        with (out / f"{tag}_classifier.pkl").open("rb") as f:
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
        train_features = pd.read_csv(out / f"{tag}_{saved['feature_scheme']}_features_train.csv").iloc[:, 4:].to_numpy()
        checks[tag+"_scaler_train_only"] = bool(np.allclose(saved["pipeline"][0].mean_, train_features.mean(0)))
    primary = json.loads((out / "primary_metrics.json").read_text(encoding="utf-8"))
    cm = confusion_matrix(predictions.y, predictions.prediction, labels=[-1, 1])
    checks["reported_confusion"] = cm.ravel().tolist() == [primary[k] for k in ["left_left", "left_right", "right_left", "right_right"]]
    checks["reported_balanced_accuracy"] = bool(np.isclose(balanced_accuracy_score(predictions.y, predictions.prediction), primary["balanced_accuracy"]))
    checks["each_test_trial_once"] = bool(not predictions["index"].duplicated().any() and len(predictions) == meta.hard_keep.sum())
    dump(out / "verification.json", checks)
    if not all(checks.values()):
        raise AssertionError(f"Verification failed: {[k for k, v in checks.items() if not v]}")
    print(f"Verification passed: {len(checks)} checks", flush=True)



