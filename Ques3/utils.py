"""Reusable event audit, fold-isolated source inversion, and EEG windows."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
from Ques2 import utils as q2_utils, model as q2_model

SEED = 20260924
FS = 64
CHANNELS = ["F3", "Fz", "F4"]


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")




def stage1(data, cache, q2, out):
    events = pd.read_csv(cache / "events.csv")
    manifest = pd.read_csv(cache / "recordings.csv").set_index("record")
    prior = pd.read_csv(q2 / "trials.csv")
    rows, audits = [], []
    paths = sorted(data.glob("VisualCog*_Task-*.mat"))
    if len(paths) != 4:
        raise ValueError("Exactly four recordings are required")
    for path in paths:
        raw, starts, group, task, _ = q2_utils.load_record(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != manifest.loc[path.stem, "sha256"]:
            raise ValueError(f"Raw data and Question 1 cache differ: {path.name}")
        old = events[events.record == path.stem].sort_values("trial")
        if not np.array_equal(old["sample"], starts):
            raise ValueError(f"Cue alignment failed: {path.name}")
        with np.load(cache / f"{path.stem}_epochs.npz") as saved:
            if not np.array_equal(saved["samples"], starts) or list(saved["channels"]) != CHANNELS:
                raise ValueError(f"Cached trial identity mismatch: {path.name}")
        action = raw[8]
        changes = np.r_[True, action[1:] != action[:-1]]
        clicks = np.flatnonzero(np.isin(action, [-2, 2]) & changes)
        later = np.flatnonzero(np.isin(action, [-1, 1]) & changes)
        if not set(np.unique(action)).issubset({-2, -1, 0, 1, 2}):
            raise ValueError(f"Unknown action code: {path.name}")
        for k, start in enumerate(starts):
            stop = starts[k + 1] if k + 1 < len(starts) else raw.shape[1]
            ci = clicks[(clicks >= start) & (clicks < stop)]
            ti = later[(later >= start) & (later < stop)]
            if len(ci) > 1 or len(ti) > 1:
                raise ValueError(f"Ambiguous multiple events: {path.stem}, trial {k}")
            cue = int(raw[7, start])
            response = int(np.sign(action[ci[0]])) if len(ci) else 0
            latency = float((ci[0] - start) / 256) if len(ci) else np.nan
            prior_row = old.iloc[k]
            if len(ci) != int(prior_row.click_count) or (len(ci) and (response != prior_row.click_sign or not np.isclose(latency, prior_row.click_delay_s))):
                raise ValueError(f"Question 1 response mismatch: {path.stem}, trial {k}")
            if len(ci) and latency <= .9:
                raise ValueError("Response occurs before fixed evidence window plus motor exclusion")
            end = int(ci[0]) - 26 if len(ci) else min(start + 512, stop - 1)
            cue_time = float(raw[9, start])
            rows.append(dict(record=path.stem, record_group=group, task_id=task, trial_id=k,
                cue_direction=cue, response_direction=response, is_correct=(cue == response) if response else pd.NA,
                cue_time=cue_time, response_time=float(raw[9, ci[0]]) if len(ci) else np.nan,
                cue_to_response_latency=latency, analysis_end_time=cue_time + latency - .1 if len(ci) else float(raw[9, end]),
                response_status=("correct" if cue == response else "incorrect") if response else "no_response",
                cue_sample=int(start), analysis_end_sample=end, sampled_analysis_end_time=float(raw[9, end]),
                later_marker_time=float(raw[9, ti[0]]) if len(ti) else np.nan,
                later_marker_direction=int(action[ti[0]]) if len(ti) else 0,
                timeout_source="not_applicable" if len(ci) else "assumed_2s_capped_by_next_cue",
                behavior_available=bool(task == 2), no_response_is_missing_recording=bool(task == 1 and not len(ci))))
        audits.append(dict(record=path.stem, cues=len(starts), clicks=len(clicks), later_markers=len(later),
                           sha256=digest, unknown_action_codes=0))
    trials = pd.DataFrame(rows)
    if (trials[trials.task_id == 1].response_direction != 0).any():
        raise ValueError("Task 1 unexpectedly contains clicks; review behavioral scope")
    keys = ["record", "trial_id"]
    prior = prior.rename(columns={"trial": "trial_id"})
    trials = trials.merge(prior[keys + ["hard_keep", "keep"]], on=keys, how="left", validate="one_to_one")
    if trials.hard_keep.isna().any() or len(trials) != len(prior):
        raise ValueError("Question 2 trial alignment failed")
    trials.insert(0, "row_id", np.arange(len(trials)))
    trials.to_csv(out / "trial_table.csv", index=False)
    summaries = []
    for (task, group), x in trials.groupby(["task_id", "record_group"]):
        summaries.append(dict(task_id=task, record_group=group, trials=len(x), cue_left=int(sum(x.cue_direction == -1)),
            cue_right=int(sum(x.cue_direction == 1)), response_left=int(sum(x.response_direction == -1)),
            response_right=int(sum(x.response_direction == 1)), correct=int(sum(x.response_status == "correct")),
            incorrect=int(sum(x.response_status == "incorrect")), no_response=int(sum(x.response_status == "no_response"))))
    counts = pd.DataFrame(summaries)
    counts.to_csv(out / "behavior_counts.csv", index=False)
    latency = trials[trials.response_direction != 0].groupby("record_group").cue_to_response_latency.describe(percentiles=[.05, .25, .5, .75, .95]).reset_index()
    latency.to_csv(out / "latency_distribution.csv", index=False)
    dump(out / "stage1_audit.json", dict(records=audits, total=len(trials), unique_trial_keys=not trials.duplicated(keys).any(),
         q2_epoch_count=len(prior), timeout_s=2., behavior_task=2, no_response_is_not_task1_mechanism=True))
    print(counts.to_string(index=False), flush=True)
    print("Stage 1 passed: all events matched; Task 1 excluded from behavioral modeling", flush=True)
    return trials


def load_early(q2, trials):
    with np.load(q2 / "validation_epochs.npz") as saved:
        x, times = saved["X"].copy(), saved["times"].copy()
    old = pd.read_csv(q2 / "trials.csv")
    if len(x) != len(trials) or not np.array_equal(old.record, trials.record) or not np.array_equal(old.trial, trials.trial_id):
        raise ValueError("Early EEG cache order differs from trial table")
    if not np.allclose(times, q2_utils.TIME) or not np.isfinite(x).all():
        raise ValueError("Invalid fixed-window source inputs")
    return x


def observation_windows(data, trials, times):
    from Ques1.src.preprocess import filter_record
    y = np.full((len(trials), 3, len(times)), np.nan)
    mask = np.zeros((len(trials), len(times)), dtype=bool)
    for record, subset in trials.groupby("record", sort=False):
        raw, _, _, _, _ = q2_utils.load_record(data / f"{record}.mat")
        clean = filter_record(raw[[1, 0, 2]])
        for row in subset.itertuples():
            start, end = row.cue_sample, row.analysis_end_sample
            base = clean[:, start - 51:start].mean(1, keepdims=True)
            segment = clean[:, start:end + 1] - base
            reduced = resample_poly(segment, 1, 4, axis=-1)
            n = min(len(times), reduced.shape[1])
            y[row.row_id, :, :n] = reduced[:, :n]
            use = times[:n] <= (end - start) / 256
            hard = bool(row.hard_keep) and np.isfinite(segment).all() and not np.any(np.abs(raw[[1, 0, 2], start:end + 1]) >= 999.9)
            mask[row.row_id, :n] = use & hard
    return y, mask


def fold_sources(x, trials, q2, train_group, times, device):
    name = f"{train_group}_to_{'B' if train_group == 'A' else 'A'}"
    model = json.loads((q2 / f"{name}_final.json").read_text(encoding="utf-8"))
    ids = model.get("training_indices", [])
    if not ids or set(trials.iloc[ids].record_group) != {train_group}:
        raise ValueError("Question 2 model provenance does not prove fold isolation")
    features, _, _ = q2_model.model_features(x, trials.task_id.to_numpy(), model, device)
    _, basis = q2_model.predict(model, times)
    sources = np.zeros((len(x), 3, len(times)))
    for i, row in enumerate(trials.itertuples()):
        k = 2 * (row.task_id - 1)
        for source in range(3):
            waveform = basis[k, 0 if source < 2 else 2]
            sources[i, source] = features[i, source] * np.interp(times - features[i, -1], times, waveform, left=0, right=0)
    delta = sources[:, 1] - sources[:, 0]
    training = (trials.record_group == train_group).to_numpy() & (trials.task_id == 2).to_numpy()
    evidence_available = float(q2_utils.TIME[-1])
    scale = max(float(np.sqrt(np.mean(delta[training][:, times >= evidence_available] ** 2))), 1e-8)
    evidence = delta / scale
    evidence[:, times < evidence_available] = 0
    reconstruction = np.einsum("cs,nst->nct", q2_model.mixing(model["coef"]), sources)
    provenance = dict(training_group=train_group, q2_training_ids=list(map(int, ids)), source_scale=scale,
                      evidence_available_s=evidence_available, retrospective_early_filter=True,
                      evidence_definition="q_right - q_left; zero before fixed evidence acquisition ends",
                      q2_model=model, source_input_uses_response=False, source_input_uses_cue=False)
    return sources, evidence, reconstruction, provenance

