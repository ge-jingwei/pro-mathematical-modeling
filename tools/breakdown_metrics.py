"""Per-subject / per-task metric breakdown for Question 1 and Question 2.

The modelling choices are frozen: this script re-uses the project's own
preprocessing, statistics, simulation and cross-validation functions and only
re-slices the same trials by file (subject x task) and by subject.

Outputs go to outputs/breakdown/.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.special import ndtr
from scipy.stats import ttest_ind
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from Ques1.events import parse_events
from Ques1.fitting import fit_components
from Ques1.preprocess import prepare_epochs
from Ques1.statistics import (
    bootstrap_mean_ci,
    cluster_independent_test,
    cluster_sign_test,
    side_difference,
)
from Ques2.data import erp_summary
from Ques2.model import ModelParameters, simulate
from Ques2.validation import (
    _lag_design,
    _scales,
    apply_observation_kernel,
    calibrated_prediction,
    classify,
    feature_matrix,
    lateral_window,
    sign_agreement,
    validation_metrics,
)
from tools.io_mat import load_records

CHANNELS = ("Fz", "F3", "F4")
TIGHT_MAXIMUM = 60
TIGHT_ALPHA = 1.0
PARAMETERS = ModelParameters(delay=0.08, coupling=0.8, input_gain=2.0)


@dataclass(frozen=True)
class View:
    """Minimal object exposing what feature_matrix needs."""

    epochs: np.ndarray
    times: np.ndarray


def _fit_row(times: np.ndarray, mean: np.ndarray) -> dict:
    try:
        fitted = fit_components(times, mean)
    except Exception:  # pragma: no cover - optimisation failure is data dependent
        return {"r2_three_gauss": np.nan, "r2_two_gauss": np.nan, "p3_amplitude": np.nan, "p3_latency_ms": np.nan}
    params = fitted["params"]
    return {
        "r2_three_gauss": float(fitted["r2"]),
        "r2_two_gauss": float(fitted["r2_two"]),
        "p3_amplitude": float(params[7]),
        "p3_latency_ms": float(params[8] * 1000.0),
    }


def q1_records(root: Path, config: dict) -> tuple[list, dict, list]:
    """Run the frozen Q1 preprocessing once and collect per-file slices."""
def q1_channel_rows(meta: dict, values: np.ndarray, labels: np.ndarray, times: np.ndarray, config: dict, rng: np.random.Generator) -> list[dict]:
    """Per-channel Q1 indicators for one trial pool (one file, or one subject)."""
    primary = np.asarray(config["windows"]["primary"], dtype=float)
    label_permutations = int(config["statistics"]["label_permutations"])
    cluster_permutations = int(config["statistics"]["cluster_permutations"])
    post = (times >= 0.0) & (times <= 0.6)
    mask = (times >= primary[0]) & (times <= primary[1])
    rows = []
    for channel, name in enumerate(CHANNELS):
        left = values[labels == -1, channel]
        right = values[labels == 1, channel]
        effect, d, p_value = side_difference(left, right, mask, label_permutations, rng)
        lateral_clusters = cluster_independent_test(left[:, post], right[:, post], cluster_permutations, rng)
        response_clusters = 0
        for side_index in (-1, 1):
            selected = values[labels == side_index, channel][:, post]
            response_clusters += len(cluster_sign_test(selected, cluster_permutations, rng))
        fits = {}
        for side_index in (-1, 1):
            mean, _, _ = bootstrap_mean_ci(values[labels == side_index, channel], 250, rng)
            for key, value in _fit_row(times, mean).items():
                fits[f"{key}_{'left' if side_index == -1 else 'right'}"] = value
        rows.append(
            {
                **meta,
                "channel": name,
                "n_left": int((labels == -1).sum()),
                "n_right": int((labels == 1).sum()),
                "primary_window_mean": float(np.concatenate([left[:, mask].mean(axis=1), right[:, mask].mean(axis=1)]).mean()),
                "right_minus_left": effect,
                "cohen_d": d,
                "p_value": p_value,
                "response_clusters": response_clusters,
                "lateral_clusters": len(lateral_clusters),
                **fits,
            }
        )
    return rows


def q1_records(root: Path, config: dict) -> tuple[list, dict]:
    """Run the frozen Q1 preprocessing once and collect per-file slices."""
    rng = np.random.default_rng(config["random_seed"])
    rows = []
    bundles = {}
    for record in load_records(root / config["paths"]["data_dir"]):
        events = parse_events(record)
        prepared = prepare_epochs(record, events, config)
        frame = events.loc[prepared.target_keep].reset_index(drop=True)
        values = prepared.target_proposed[prepared.target_keep]
        times = prepared.target_times
        labels = frame["cue_side"].to_numpy()
        bundles[record.path.name] = {
            "subject": record.subject,
            "task": record.task,
            "cue": prepared.proposed[prepared.keep],
            "cue_labels": events["cue_side"].to_numpy()[prepared.keep],
            "target": values,
            "target_labels": labels,
            "target_side": events["target_side"].to_numpy()[prepared.target_keep],
            "times": prepared.times,
            "target_times": times,
        }
        meta = {
            "file": record.path.name,
            "subject": record.subject,
            "task": record.task,
            "n_trials_kept": int(prepared.target_keep.sum()),
            "retention_rate": float(prepared.target_keep.mean()),
            "cue_trials_kept": int(prepared.keep.sum()),
            "wavelet_alpha": float(prepared.alpha),
        }
        rows.extend(q1_channel_rows(meta, values, labels, times, config, rng))
    return rows, bundles


def q1_subject_rows(bundles: dict, config: dict, rng: np.random.Generator) -> list[dict]:
    """真正把同一被试两个项目的试次合并后重算同一套指标（不是两个项目数字的算术平均）。"""
    names = list(bundles)
    rows = []
    for subject in sorted({bundles[name]["subject"] for name in names}):
        chosen = [name for name in names if bundles[name]["subject"] == subject]
        values = np.concatenate([bundles[name]["target"] for name in chosen])
        labels = np.concatenate([bundles[name]["target_labels"] for name in chosen])
        times = bundles[chosen[0]]["target_times"]
        meta = {
            "file": "+".join(chosen),
            "subject": subject,
            "task": "1+2",
            "n_trials_kept": int(sum(len(bundles[name]["target"]) for name in chosen)),
            "retention_rate": float(np.mean([len(bundles[name]["target"]) for name in chosen]) / 100.0),
            "cue_trials_kept": int(sum(len(bundles[name]["cue_labels"]) for name in chosen)),
            "wavelet_alpha": float("nan"),
        }
        rows.extend(q1_channel_rows(meta, values, labels, times, config, rng))
    return rows


def fit_kernel(model: dict, reference: dict, keys: list, times: np.ndarray, maximum: int, alpha: float) -> tuple[dict, list[dict]]:
    """Same ridge lag-kernel fit as fit_observation_kernel, restricted to the
    (task, side) keys present in the current subset."""
    mask = (times >= 0) & (times <= 0.75)
    kernels = []
    for channel in range(3):
        design = np.vstack([_lag_design(model[side].eeg[channel], maximum)[mask] for task, side in keys])
        target = np.concatenate([reference[(task, side)]["mean"][channel, mask] for task, side in keys])
        scale = np.linalg.norm(design, axis=0)
        scale[scale == 0] = 1.0
        fitted = Ridge(alpha=alpha).fit(design / scale, target)
        kernels.append({"coef": fitted.coef_, "intercept": fitted.intercept_, "scale": scale})
    return apply_observation_kernel(model, kernels, times, maximum), kernels


def cv_metrics(epochs: np.ndarray, tasks: np.ndarray, labels: np.ndarray, times: np.ndarray, model: dict, rate: float, folds: int = 5, maximum: int = TIGHT_MAXIMUM, alpha: float = TIGHT_ALPHA) -> pd.DataFrame:
    """Same protocol as observation_cross_validation_folds, but tolerant of
    subsets that only contain one of the two projects."""
    splits = {}
    for task in np.unique(tasks):
        for side in (-1, 1):
            indices = np.flatnonzero((tasks == task) & (labels == side))
            if len(indices) < 2:
                continue
            count = min(folds, len(indices))
            splits[(int(task), side)] = [(indices[train], indices[test]) for train, test in KFold(count, shuffle=True, random_state=20260923).split(indices)]
    rows = []
    fold_count = max(len(values) for values in splits.values())
    for fold in range(fold_count):
        train, test = {}, {}
        for key, values in splits.items():
            if fold >= len(values):
                continue
            train_indices, test_indices = values[fold]
            train[key] = {"mean": epochs[train_indices].mean(axis=0)}
            test[key] = {"mean": epochs[test_indices].mean(axis=0)}
        prediction, _ = fit_kernel(model, train, sorted(train), times, maximum, alpha)
        window, _, _ = lateral_window(prediction, times)
        for task in np.unique(tasks):
            task = int(task)
            if not any(key[0] == task for key in test):
                continue
            rows.append(
                {
                    "task": task,
                    "fold": fold + 1,
                    **validation_metrics(prediction, test, times, task, rate),
                    "sign_agreement": sign_agreement(prediction, test, times, task, window),
                }
            )
    return pd.DataFrame(rows)


def aggregate(folds: pd.DataFrame, keys: list[str]) -> pd.DataFrame:    return folds.groupby(keys, as_index=False).agg(
        waveform_correlation=("waveform_correlation", "mean"),
        waveform_correlation_sd=("waveform_correlation", "std"),
        spectral_cosine=("spectral_cosine", "mean"),
        spectral_cosine_sd=("spectral_cosine", "std"),
        sign_agreement=("sign_agreement", "mean"),
        folds=("fold", "count"),
    )


def decodability(epochs: np.ndarray, labels: np.ndarray, tasks: np.ndarray, times: np.ndarray, lock: str) -> pd.DataFrame:
    rows = []
    for task in np.unique(tasks):
        selected = tasks == task
        values = epochs[selected, 2] - epochs[selected, 1]
        y = labels[selected]
        for start in np.arange(0, 0.6, 0.1):
            window = (times >= start) & (times < start + 0.1)
            per_trial = values[:, window].mean(axis=1)
            a, b = per_trial[y == -1], per_trial[y == 1]
            if len(a) < 2 or len(b) < 2:
                continue
            pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / max(len(a) + len(b) - 2, 1))
            d = (b.mean() - a.mean()) / max(pooled, np.finfo(float).eps)
            _, p_value = ttest_ind(a, b, equal_var=False)
            rows.append({"lock": lock, "subject": "ALL", "task": int(task), "window_start_s": float(start), "cohens_d": float(d), "t_p_value": float(p_value), "auc_upper_bound": float(ndtr(abs(d) / np.sqrt(2)))})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cti.yaml")
    parser.add_argument("--permutations", type=int, default=200)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    with (root / args.config).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    output = root / "outputs" / "breakdown"
    output.mkdir(parents=True, exist_ok=True)
    rate = float(config["sample_rate"])
    rng = np.random.default_rng(config["random_seed"] + 7)

    file_rows, bundles = q1_records(root, config)
    q1_file = pd.DataFrame(file_rows)
    q1_file.to_csv(output / "q1_by_file.csv", index=False)
    q1_subject = pd.DataFrame(q1_subject_rows(bundles, config, rng))
    q1_subject.to_csv(output / "q1_by_subject.csv", index=False)

    names = list(bundles)
    times = bundles[names[0]]["times"]
    target_times = bundles[names[0]]["target_times"]

    cue_epochs = np.concatenate([bundles[name]["cue"] for name in names])
    cue_labels = np.concatenate([bundles[name]["cue_labels"] for name in names])
    target_epochs = np.concatenate([bundles[name]["target"] for name in names])
    target_labels = np.concatenate([bundles[name]["target_side"] for name in names])
    tasks = np.concatenate([np.full(len(bundles[name]["cue"]), bundles[name]["task"]) for name in names])
    target_tasks = np.concatenate([np.full(len(bundles[name]["target"]), bundles[name]["task"]) for name in names])
    subjects = np.concatenate([np.full(len(bundles[name]["cue"]), bundles[name]["subject"]) for name in names])
    target_subjects = np.concatenate([np.full(len(bundles[name]["target"]), bundles[name]["subject"]) for name in names])
    groups = np.concatenate([np.full(len(bundles[name]["cue"]), index) for index, name in enumerate(names)])
    target_groups = np.concatenate([np.full(len(bundles[name]["target"]), index) for index, name in enumerate(names)])

    class Dataset:
        epochs: np.ndarray
        labels: np.ndarray
        tasks: np.ndarray
        groups: np.ndarray
        subjects: np.ndarray
        target_epochs: np.ndarray
        target_labels: np.ndarray
        target_tasks: np.ndarray
        target_groups: np.ndarray
        target_subjects: np.ndarray
        times: np.ndarray
        files: tuple

    dataset = Dataset()
    dataset.epochs, dataset.labels, dataset.tasks, dataset.groups, dataset.subjects = cue_epochs, cue_labels, tasks, groups, subjects
    dataset.target_epochs, dataset.target_labels, dataset.target_tasks = target_epochs, target_labels, target_tasks
    dataset.target_groups, dataset.target_subjects = target_groups, target_subjects
    dataset.times, dataset.files = times, tuple(names)

    erp = erp_summary(dataset, 500, rng)
    model = {side: simulate(side, times, rate, PARAMETERS, config["q2"]["grid_size"]) for side in (-1, 1)}
    scales = _scales(model, erp, times, 1)
    prediction = calibrated_prediction(model, scales)
    window, start, stop = lateral_window(prediction, times)

    subsets = {"ALL": np.ones(len(cue_labels), dtype=bool)}
    target_subsets = {"ALL": np.ones(len(target_labels), dtype=bool)}
    for subject in np.unique(subjects):
        subsets[f"subject_{subject}"] = subjects == subject
        target_subsets[f"subject_{subject}"] = target_subjects == subject
    for task in np.unique(tasks):
        subsets[f"task_{task}"] = tasks == task
        target_subsets[f"task_{task}"] = target_tasks == task
    for name in names:
        subsets[f"file_{name}"] = np.concatenate([np.full(len(bundles[item]["cue"]), item == name) for item in names])
        target_subsets[f"file_{name}"] = np.concatenate([np.full(len(bundles[item]["target"]), item == name) for item in names])

    validation_rows = []
    accuracy_rows = []
    bound_rows = []
    for label, mask in subsets.items():
        subject_tag = "ALL"
        task_tag = "ALL"
        if label.startswith("subject_"):
            subject_tag = label.split("_", 1)[1]
        elif label.startswith("task_"):
            task_tag = int(label.split("_", 1)[1])
        elif label.startswith("file_"):
            name = label.split("_", 1)[1]
            subject_tag = bundles[name]["subject"]
            task_tag = bundles[name]["task"]
        folds = cv_metrics(cue_epochs[mask], tasks[mask], cue_labels[mask], times, model, rate)
        if len(folds):
            summary = aggregate(folds, ["task"])
            for row in summary.itertuples():
                validation_rows.append(
                    {
                        "scope": label,
                        "subject": subject_tag,
                        "task": task_tag if task_tag == "ALL" else int(task_tag),
                        "validated_task": int(row.task),
                        "waveform_correlation": row.waveform_correlation,
                        "waveform_correlation_sd": row.waveform_correlation_sd,
                        "spectral_cosine": row.spectral_cosine,
                        "spectral_cosine_sd": row.spectral_cosine_sd,
                        "sign_agreement": row.sign_agreement,
                        "folds": int(row.folds),
                    }
                )
        view = View(epochs=cue_epochs[mask], times=times)
        features_all, features_template = feature_matrix(view, prediction, window)
        if len(np.unique(groups[mask])) >= 2:
            for feature_name, features in (("template", features_template), ("template+lateral", features_all)):
                result, _, _, _ = classify(features, cue_labels[mask], groups[mask], args.permutations, rng)
                accuracy_rows.append({"scope": label, "subject": subject_tag, "task": task_tag, "feature_set": feature_name, **result})
        bounds = decodability(cue_epochs[mask], cue_labels[mask], tasks[mask], times, "cue")
        bounds.insert(0, "scope", label)
        bounds["subject"] = subject_tag
        bound_rows.append(bounds)
        target_bounds = decodability(target_epochs[target_subsets[label]], target_labels[target_subsets[label]], target_tasks[target_subsets[label]], target_times, "target")
        target_bounds.insert(0, "scope", label)
        target_bounds["subject"] = subject_tag
        bound_rows.append(target_bounds)

    validation = pd.DataFrame(validation_rows)
    accuracy = pd.DataFrame(accuracy_rows)
    bounds = pd.concat(bound_rows, ignore_index=True)
    validation.to_csv(output / "q2_validation_by_scope.csv", index=False)
    accuracy.to_csv(output / "q2_accuracy_by_scope.csv", index=False)
    bounds.to_csv(output / "q2_bounds_by_scope.csv", index=False)

    def merged(prefix: str) -> pd.DataFrame:
        subset = validation[validation["scope"].str.startswith(prefix)].copy()
        columns = ["scope", "auc", "accuracy", "p_value", "n_trials", "folds"]
        extra = accuracy[(accuracy["scope"].str.startswith(prefix)) & (accuracy["feature_set"] == "template+lateral")][columns]
        if len(extra):
            subset = subset.merge(extra.rename(columns={"folds": "classifier_folds"}), on="scope", how="left")
        else:
            subset["classifier_note"] = "单文件只有一个分组，分类器交叉验证不可用"
        return subset

    q2_file = merged("file_")
    q2_file.to_csv(output / "q2_by_file.csv", index=False)
    q2_subject = merged("subject_")
    q2_subject.to_csv(output / "q2_by_subject.csv", index=False)
    q2_task = merged("task_")
    q2_task.to_csv(output / "q2_by_task.csv", index=False)
    print(f"side window: {start * 1000:.1f}-{stop * 1000:.1f} ms")
    print(validation.to_string(index=False))
    print(accuracy.to_string(index=False))


if __name__ == "__main__":
    main()
