from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from Ques1.baselines import bandpass_epochs, extract_epochs, hard_trial_mask, short_interpolation
from Ques1.denoise import fit_noise_scales, reliability_kalman
from Ques1.events import build_trials
from Ques1.reliability import fit_record_reliability
from tools.io_mat import Record, load_records


METHODS = ("B1", "B2", "B3", "B4")


@dataclass(frozen=True)
class ProcessedSplit:
    training_epochs: np.ndarray
    evaluation_epochs: np.ndarray
    training_weights: np.ndarray
    evaluation_weights: np.ndarray
    training_valid: np.ndarray
    evaluation_valid: np.ndarray


def continuous_blocks(count: int, block_size: int) -> list[np.ndarray]:
    if count % block_size:
        raise ValueError("Trial count is not divisible by block size")
    return [np.arange(start, start + block_size) for start in range(0, count, block_size)]


def _training_sample_mask(
    onsets: np.ndarray,
    indices: np.ndarray,
    length: int,
    sample_rate: float,
) -> np.ndarray:
    mask = np.zeros(length, dtype=bool)
    for onset in onsets[indices].astype(int):
        start = max(0, onset - int(round(0.5 * sample_rate)))
        stop = min(length, onset + int(round(sample_rate)))
        mask[start:stop] = True
    return mask


def _baseline_correct(epochs: np.ndarray, sample_rate: float) -> np.ndarray:
    stop = int(round(0.3 * sample_rate))
    return epochs - np.mean(epochs[:, :, :stop], axis=2, keepdims=True)


def process_split(
    record: Record,
    trials: pd.DataFrame,
    training_indices: np.ndarray,
    evaluation_indices: np.ndarray,
    method: str,
    highpass: float,
    config: dict,
) -> ProcessedSplit:
    onsets = trials["t_cue_on"].to_numpy()
    raw = extract_epochs(record.eeg, onsets, record.sample_rate, (-0.5, 1.0))
    train_raw = raw[training_indices]
    eval_raw = raw[evaluation_indices]
    unit_train = np.ones_like(train_raw)
    unit_eval = np.ones_like(eval_raw)
    valid_train = np.ones(len(train_raw), dtype=bool)
    valid_eval = np.ones(len(eval_raw), dtype=bool)
    settings = config["quality"]
    if method in {"B3", "B4"}:
        sample_mask = _training_sample_mask(
            onsets,
            training_indices,
            record.eeg.shape[1],
            record.sample_rate,
        )
        reliability, _ = fit_record_reliability(record, config, sample_mask)
        reliability_epochs = extract_epochs(reliability, onsets, record.sample_rate, (-0.5, 1.0))
        train_weights = reliability_epochs[training_indices]
        eval_weights = reliability_epochs[evaluation_indices]
    else:
        train_weights, eval_weights = unit_train, unit_eval
    if method == "B2":
        valid_train = hard_trial_mask(
            train_raw,
            settings["saturation"],
            settings["hard_trial_peak_to_peak"],
        )
        valid_eval = hard_trial_mask(
            eval_raw,
            settings["saturation"],
            settings["hard_trial_peak_to_peak"],
        )
    if method == "B3":
        maximum = int(round(settings["short_interpolation_seconds"] * record.sample_rate))
        train_raw = short_interpolation(train_raw, train_weights, maximum)
        eval_raw = short_interpolation(eval_raw, eval_weights, maximum)
    if method == "B4":
        observation, process = fit_noise_scales(train_raw)
        train_raw = reliability_kalman(
            train_raw,
            train_weights,
            observation,
            process,
            settings["reliability_floor"],
        )
        eval_raw = reliability_kalman(
            eval_raw,
            eval_weights,
            observation,
            process,
            settings["reliability_floor"],
        )
    filter_settings = config["filter"]
    train_epochs = bandpass_epochs(
        train_raw,
        record.sample_rate,
        highpass,
        filter_settings["lowpass"],
        filter_settings["notch"],
    )
    eval_epochs = bandpass_epochs(
        eval_raw,
        record.sample_rate,
        highpass,
        filter_settings["lowpass"],
        filter_settings["notch"],
    )
    return ProcessedSplit(
        _baseline_correct(train_epochs, record.sample_rate),
        _baseline_correct(eval_epochs, record.sample_rate),
        train_weights,
        eval_weights,
        valid_train,
        valid_eval,
    )


def _weighted_mean(values: np.ndarray, weights: np.ndarray, axis: tuple[int, ...]) -> np.ndarray:
    denominator = weights.sum(axis=axis)
    return np.divide(
        (values * weights).sum(axis=axis),
        denominator,
        out=np.full_like(denominator, np.nan, dtype=float),
        where=denominator > 0,
    )


def _features(epochs: np.ndarray, weights: np.ndarray, sample_rate: float) -> np.ndarray:
    start = int(round(0.75 * sample_rate))
    stop = int(round(sample_rate))
    difference = (epochs[:, 1] - epochs[:, 2]) / np.sqrt(2.0)
    difference_weights = np.minimum(weights[:, 1], weights[:, 2])
    return _weighted_mean(difference[:, start:stop], difference_weights[:, start:stop], (1,))


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    finite = np.isfinite(scores)
    labels = labels[finite]
    scores = scores[finite]
    positive = labels == 1
    negative = labels == -1
    if not positive.any() or not negative.any():
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[positive].sum() - positive.sum() * (positive.sum() + 1) / 2) / (positive.sum() * negative.sum()))


def _split_half(epochs: np.ndarray, valid: np.ndarray) -> float:
    selected = epochs[valid]
    if len(selected) < 4:
        return float("nan")
    first = selected[::2].mean(axis=0).ravel()
    second = selected[1::2].mean(axis=0).ravel()
    return float(np.corrcoef(first, second)[0, 1])


def split_metrics(
    split: ProcessedSplit,
    training_labels: np.ndarray,
    evaluation_labels: np.ndarray,
    sample_rate: float,
) -> dict[str, float]:
    train_feature = _features(split.training_epochs, split.training_weights, sample_rate)
    eval_feature = _features(split.evaluation_epochs, split.evaluation_weights, sample_rate)
    train_valid = split.training_valid & np.isfinite(train_feature)
    eval_valid = split.evaluation_valid & np.isfinite(eval_feature)
    left = train_feature[train_valid & (training_labels == -1)]
    right = train_feature[train_valid & (training_labels == 1)]
    direction = 1.0 if len(left) and len(right) and np.mean(right) >= np.mean(left) else -1.0
    auc = _auc(evaluation_labels[eval_valid], direction * eval_feature[eval_valid])
    reliability = _split_half(split.evaluation_epochs, eval_valid)
    baseline_stop = int(round(0.3 * sample_rate))
    residual = float(np.nanstd(split.evaluation_epochs[eval_valid, :, :baseline_stop]))
    response = split.evaluation_epochs[eval_valid, 0, int(0.75 * sample_rate) : int(sample_rate)].mean(axis=1)
    event_stat = float(np.mean(response) / (np.std(response, ddof=1) / np.sqrt(len(response)))) if len(response) > 1 else float("nan")
    return {
        "auc": auc,
        "split_half": reliability,
        "baseline_std": residual,
        "event_stat": event_stat,
        "valid_trials": int(eval_valid.sum()),
    }


def nested_validation(data_dir: str | Path, config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_trials = build_trials(data_dir)
    fold_rows = []
    selection_rows = []
    block_size = config["validation"]["trials_per_block"]
    highpasses = config["filter"]["highpass_candidates"]
    for record in load_records(data_dir):
        trials = all_trials[all_trials["file"] == record.path.name].reset_index(drop=True)
        labels = trials["side_cue"].to_numpy()
        blocks = continuous_blocks(len(trials), block_size)
        for outer_index, outer_test in enumerate(blocks):
            outer_train_blocks = [block for index, block in enumerate(blocks) if index != outer_index]
            candidate_scores = []
            for method in METHODS:
                for highpass in highpasses:
                    metrics = []
                    for inner_index, inner_test in enumerate(outer_train_blocks):
                        inner_train = np.concatenate(
                            [block for index, block in enumerate(outer_train_blocks) if index != inner_index]
                        )
                        split = process_split(
                            record,
                            trials,
                            inner_train,
                            inner_test,
                            method,
                            highpass,
                            config,
                        )
                        metrics.append(
                            split_metrics(split, labels[inner_train], labels[inner_test], record.sample_rate)
                        )
                    mean_auc = float(np.nanmean([item["auc"] for item in metrics]))
                    mean_reliability = float(np.nanmean([item["split_half"] for item in metrics]))
                    candidate_scores.append(
                        {
                            "method": method,
                            "highpass": highpass,
                            "auc": mean_auc,
                            "split_half": mean_reliability,
                            "score": mean_auc + mean_reliability,
                        }
                    )
            candidates = pd.DataFrame(candidate_scores)
            reference = candidates[candidates["method"] == "B1"].sort_values("score", ascending=False).iloc[0]
            eligible = candidates[
                (candidates["auc"] >= reference["auc"])
                & (candidates["split_half"] >= reference["split_half"])
            ].copy()
            eligible["order"] = eligible["method"].map({name: index for index, name in enumerate(METHODS)})
            selected = eligible.sort_values(["score", "order"], ascending=[False, True]).iloc[0]
            selection_rows.append(
                {
                    "file": record.path.name,
                    "outer_fold": outer_index + 1,
                    "method": selected["method"],
                    "highpass": selected["highpass"],
                }
            )
            outer_train = np.concatenate(outer_train_blocks)
            for method in METHODS:
                method_choice = candidates[candidates["method"] == method].sort_values("score", ascending=False).iloc[0]
                split = process_split(
                    record,
                    trials,
                    outer_train,
                    outer_test,
                    method,
                    float(method_choice["highpass"]),
                    config,
                )
                metrics = split_metrics(split, labels[outer_train], labels[outer_test], record.sample_rate)
                fold_rows.append(
                    {
                        "file": record.path.name,
                        "outer_fold": outer_index + 1,
                        "method": method,
                        "highpass": float(method_choice["highpass"]),
                        **metrics,
                    }
                )
    folds = pd.DataFrame(fold_rows)
    selections = pd.DataFrame(selection_rows)
    summary = (
        folds.groupby("method")
        .agg(
            outer_auc=("auc", "mean"),
            outer_split_half=("split_half", "mean"),
            baseline_residual_std=("baseline_std", "mean"),
            primary_statistic=("event_stat", "mean"),
            valid_trials=("valid_trials", "sum"),
        )
        .reset_index()
    )
    selected_counts = selections["method"].value_counts()
    summary["selected_folds"] = summary["method"].map(selected_counts).fillna(0).astype(int)
    best_conservative = summary[summary["method"] != "B4"].sort_values(
        ["outer_auc", "outer_split_half"], ascending=False
    ).iloc[0]
    b4 = summary[summary["method"] == "B4"].iloc[0]
    comparison = folds.pivot_table(index=["file", "outer_fold"], columns="method", values="event_stat")
    reproduced = int((comparison["B4"] > comparison[best_conservative["method"]]).sum())
    b4_passed = (
        b4["outer_auc"] >= best_conservative["outer_auc"]
        and b4["outer_split_half"] >= best_conservative["outer_split_half"]
        and reproduced >= len(comparison) - 1
    )
    summary["decision"] = "candidate"
    summary.loc[summary["method"] == "B4", "decision"] = "accepted" if b4_passed else "rejected_by_D1"
    summary.loc[summary["method"] == best_conservative["method"], "decision"] = (
        "comparison" if b4_passed else "selected_conservative"
    )
    return summary, folds, selections


def write_nested_validation(data_dir: str | Path, output_dir: str | Path, config: dict) -> pd.DataFrame:
    summary, folds, selections = nested_validation(data_dir, config)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "baseline_comparison.csv", index=False)
    folds.to_csv(output_dir / "baseline_folds.csv", index=False)
    selections.to_csv(output_dir / "selection_stability.csv", index=False)
    return summary
