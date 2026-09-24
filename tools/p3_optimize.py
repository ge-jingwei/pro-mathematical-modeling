from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from audit_panel_alignment import require_matplotlib_panel_alignment
from scipy.linalg import eigh
from scipy.optimize import curve_fit
from scipy.signal import butter, coherence, hilbert, savgol_filter, sosfiltfilt
from sklearn.decomposition import PCA
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

from p1_p300 import extract_file
from p3_model import build_dataset, template_feature, trial_features


SEED = 20260923
BASELINE_ACCURACY = 0.5315315315315315
EXTRA_COLUMNS = [
    "experiment_id",
    "optimization_direction",
    "classifier_variant",
    "outer_fold",
    "inner_score",
    "selected_parameters",
    "ci_low",
    "ci_high",
    "fit_success_rate",
    "fit_r2_median",
    "elapsed_seconds",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--limit-minutes", type=float, default=90.0)
    return parser.parse_args()


def configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"],
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def gamma_curve(t: np.ndarray, amplitude: float, onset: float, tau: float, shape: float, offset: float) -> np.ndarray:
    u = np.maximum((t - onset) / tau, 0.0)
    return amplitude * np.power(u, shape) * np.exp(-u) + offset


def morphology_features(epoch: np.ndarray, time_axis: np.ndarray) -> tuple[np.ndarray, list[str], list[float], list[bool]]:
    fit_mask = (time_axis >= 0.20) & (time_axis <= 0.60)
    peak_mask = (time_axis >= 0.25) & (time_axis <= 0.50)
    t_fit = time_axis[fit_mask]
    smooth = savgol_filter(epoch, 11, 3, axis=1)
    values, names, r2_values, successes = [], [], [], []
    for channel_index, channel in enumerate(("Fz", "F3", "F4")):
        y = smooth[channel_index, fit_mask]
        peak_local = int(np.argmax(smooth[channel_index, peak_mask]))
        peak_index = np.flatnonzero(peak_mask)[peak_local]
        peak_time = float(time_axis[peak_index])
        peak_value = float(smooth[channel_index, peak_index])
        offset0 = float(np.median(y[: max(3, y.size // 8)]))
        amplitude0 = max((peak_value - offset0) / max(3**3 * np.exp(-3), 1e-6), 1e-3)
        success = True
        try:
            parameters, _ = curve_fit(
                gamma_curve,
                t_fit,
                y,
                p0=[amplitude0, 0.20, 0.05, 3.0, offset0],
                bounds=([0.0, 0.12, 0.015, 1.0, -200.0], [500.0, 0.38, 0.15, 8.0, 200.0]),
                maxfev=3000,
            )
            fitted = gamma_curve(t_fit, *parameters)
            residual = float(np.sum((y - fitted) ** 2))
            total = float(np.sum((y - y.mean()) ** 2))
            r2 = 1.0 - residual / max(total, 1e-12)
        except (RuntimeError, ValueError, FloatingPointError):
            parameters = np.full(5, np.nan)
            r2 = np.nan
            success = False
        left = np.flatnonzero((time_axis >= 0.20) & (time_axis <= peak_time))
        right = np.flatnonzero((time_axis >= peak_time) & (time_axis <= 0.60))
        half = 0.5 * (peak_value + offset0)
        left_cross = left[np.argmin(np.abs(smooth[channel_index, left] - half))] if left.size else peak_index
        right_cross = right[np.argmin(np.abs(smooth[channel_index, right] - half))] if right.size else peak_index
        width = float((time_axis[right_cross] - time_axis[left_cross]) * 1000)
        rise = float((peak_value - smooth[channel_index, left_cross]) / max(peak_time - time_axis[left_cross], 1 / 256))
        fall = float((smooth[channel_index, right_cross] - peak_value) / max(time_axis[right_cross] - peak_time, 1 / 256))
        amplitude, onset, tau, shape, _ = parameters
        fitted_peak_time = onset + shape * tau if success else np.nan
        fitted_peak = amplitude * shape**shape * np.exp(-shape) if success else np.nan
        values.extend([amplitude, onset * 1000, tau * 1000, shape, r2, fitted_peak_time * 1000, fitted_peak, width, rise, fall])
        names.extend([f"{channel}_{name}" for name in ("gamma_A", "gamma_t0_ms", "gamma_tau_ms", "gamma_k", "gamma_R2", "gamma_peak_ms", "gamma_peak", "half_width_ms", "rise_slope", "fall_slope")])
        r2_values.append(r2)
        successes.append(success)
    return np.asarray(values, dtype=float), names, r2_values, successes


def band_hilbert(epoch: np.ndarray, low: float, high: float) -> np.ndarray:
    sos = butter(4, [low, high], btype="bandpass", fs=256, output="sos")
    return hilbert(sosfiltfilt(sos, epoch, axis=1), axis=1)


def expanded_features(epoch: np.ndarray, time_axis: np.ndarray) -> tuple[np.ndarray, list[str]]:
    smooth = savgol_filter(epoch, 11, 3, axis=1)
    values, names = [], []
    for label, low, high in (("N100", 0.08, 0.15), ("N200", 0.15, 0.25)):
        mask = (time_axis >= low) & (time_axis <= high)
        indices = np.argmin(smooth[:, mask], axis=1)
        for channel_index, channel in enumerate(("Fz", "F3", "F4")):
            values.extend([smooth[channel_index, mask][indices[channel_index]], time_axis[mask][indices[channel_index]] * 1000])
            names.extend([f"{channel}_{label}_amplitude", f"{channel}_{label}_latency_ms"])
    analytic = {}
    p300 = (time_axis >= 0.25) & (time_axis <= 0.50)
    for band, low, high in (("theta", 4.0, 8.0), ("alpha", 8.0, 13.0)):
        analytic[band] = band_hilbert(epoch, low, high)
        for channel_index, channel in enumerate(("Fz", "F3", "F4")):
            values.append(float(np.log(np.mean(np.abs(analytic[band][channel_index, p300]) ** 2) + 1e-12)))
            names.append(f"{channel}_{band}_instant_power")
    segment = (time_axis >= 0.10) & (time_axis <= 0.70)
    frequency, coh = coherence(epoch[1, segment], epoch[2, segment], fs=256, window="hann", nperseg=64, noverlap=32)
    for band, low, high in (("theta", 4.0, 8.0), ("alpha", 8.0, 13.0), ("theta_alpha", 4.0, 13.0)):
        mask = (frequency >= low) & (frequency <= high)
        values.append(float(np.mean(coh[mask])))
        names.append(f"F3-F4_coherence_{band}")
    for band in ("theta", "alpha"):
        phase = np.angle(analytic[band])
        for first, second, pair in ((0, 1, "Fz-F3"), (0, 2, "Fz-F4")):
            difference = float(np.angle(np.mean(np.exp(1j * (phase[first, p300] - phase[second, p300])))))
            values.extend([np.sin(difference), np.cos(difference)])
            names.extend([f"{pair}_{band}_phase_sin", f"{pair}_{band}_phase_cos"])
    return np.asarray(values, dtype=float), names


def spectro_spatial_features(epoch: np.ndarray, time_axis: np.ndarray) -> tuple[np.ndarray, list[str]]:
    values, names = [], []
    bands = (("delta", 1.0, 4.0), ("theta", 4.0, 8.0), ("alpha", 8.0, 13.0), ("beta", 13.0, 20.0))
    windows = ((50, 200), (200, 350), (350, 500), (500, 700))
    for band, low, high in bands:
        filtered = np.real(band_hilbert(epoch, low, high))
        for start, stop in windows:
            mask = (time_axis >= start / 1000) & (time_axis < stop / 1000)
            segment = filtered[:, mask]
            log_power = np.log(np.mean(segment**2, axis=1) + 1e-12)
            values.extend(log_power)
            names.extend([f"{channel}_{band}_{start}-{stop}ms_logpower" for channel in ("Fz", "F3", "F4")])
            values.append(log_power[1] - log_power[2])
            names.append(f"F3-F4_{band}_{start}-{stop}ms_logpower")
            correlation = np.corrcoef(segment)
            for first, second, pair in ((0, 1, "Fz-F3"), (0, 2, "Fz-F4"), (1, 2, "F3-F4")):
                values.append(float(np.nan_to_num(correlation[first, second])))
                names.append(f"{pair}_{band}_{start}-{stop}ms_corr")
    return np.asarray(values, dtype=float), names


def build_feature_matrices(dataset: dict) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict, list[str]]:
    expanded, morphology, spectro, mirror_base, mirror_enhanced, mirror_expanded, mirror_morphology, mirror_spectro = [], [], [], [], [], [], [], []
    r2_values, successes = [], []
    expanded_names, morphology_names, spectro_names = [], [], []
    for epoch in dataset["epochs"]:
        ext, expanded_names = expanded_features(epoch, dataset["time"])
        morph, morphology_names, r2, ok = morphology_features(epoch, dataset["time"])
        expanded.append(ext)
        morphology.append(morph)
        spectral, spectro_names = spectro_spatial_features(epoch, dataset["time"])
        spectro.append(spectral)
        r2_values.extend(r2)
        successes.extend(ok)
        mirrored = epoch[[0, 2, 1]]
        base, enhanced, _, _, _ = trial_features(mirrored, dataset["time"])
        ext_mirror, _ = expanded_features(mirrored, dataset["time"])
        morph_mirror, _, _, _ = morphology_features(mirrored, dataset["time"])
        spectral_mirror, _ = spectro_spatial_features(mirrored, dataset["time"])
        mirror_base.append(base)
        mirror_enhanced.append(enhanced)
        mirror_expanded.append(ext_mirror)
        mirror_morphology.append(morph_mirror)
        mirror_spectro.append(spectral_mirror)
    expanded = np.asarray(expanded)
    morphology = np.asarray(morphology)
    spectro = np.asarray(spectro)
    mirror_base = np.asarray(mirror_base)
    mirror_enhanced = np.asarray(mirror_enhanced)
    mirror_expanded = np.asarray(mirror_expanded)
    mirror_morphology = np.asarray(mirror_morphology)
    mirror_spectro = np.asarray(mirror_spectro)
    matrices = {
        "base": dataset["base"],
        "expanded_only": expanded,
        "base_expanded": np.c_[dataset["base"], expanded],
        "enhanced_expanded": np.c_[dataset["enhanced"], expanded],
        "morphology_only": morphology,
        "base_morphology": np.c_[dataset["base"], morphology],
        "all_features": np.c_[dataset["base"], expanded, morphology],
        "spectro_spatial": spectro,
        "base_spectro": np.c_[dataset["base"], spectro],
    }
    mirror = {
        "base": mirror_base,
        "expanded_only": mirror_expanded,
        "base_expanded": np.c_[mirror_base, mirror_expanded],
        "enhanced_expanded": np.c_[mirror_enhanced, mirror_expanded],
        "morphology_only": mirror_morphology,
        "base_morphology": np.c_[mirror_base, mirror_morphology],
        "all_features": np.c_[mirror_base, mirror_expanded, mirror_morphology],
        "spectro_spatial": mirror_spectro,
        "base_spectro": np.c_[mirror_base, mirror_spectro],
    }
    quality = {
        "fit_success_rate": float(np.mean(successes)),
        "fit_r2_median": float(np.nanmedian(r2_values)),
        "finite_fraction": float(np.mean(np.isfinite(morphology))),
    }
    names = dataset["names"]["base"] + expanded_names + morphology_names
    return matrices, mirror, quality, names


def csp_band_epochs(epochs: np.ndarray) -> dict[str, np.ndarray]:
    output = {}
    for name, low, high in (("broad", 0.5, 20.0), ("theta", 4.0, 8.0), ("alpha", 8.0, 13.0)):
        sos = butter(4, [low, high], btype="bandpass", fs=256, output="sos")
        output[name] = sosfiltfilt(sos, epochs, axis=2)
    return output


def csp_matrices(filtered: dict[str, np.ndarray], labels: np.ndarray, time_axis: np.ndarray, train: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = (time_axis >= 0.10) & (time_axis <= 0.70)
    train_features, target_features = [], []
    for band in ("broad", "theta", "alpha"):
        covariances = []
        for epoch in filtered[band][train][:, :, mask]:
            covariance = epoch @ epoch.T
            covariances.append(covariance / max(np.trace(covariance), 1e-12))
        covariances = np.asarray(covariances)
        negative = covariances[labels[train] == -1].mean(axis=0)
        positive = covariances[labels[train] == 1].mean(axis=0)
        _, vectors = eigh(positive, positive + negative + 1e-8 * np.eye(3))
        filters = vectors[:, [0, -1]]
        for source, destination in ((train, train_features), (target, target_features)):
            projected = np.einsum("ce,net->nct", filters.T, filtered[band][source][:, :, mask])
            destination.append(np.log(np.var(projected, axis=2) + 1e-12))
    return np.concatenate(train_features, axis=1), np.concatenate(target_features, axis=1)


def fill_and_scale(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    medians = np.nanmedian(np.where(np.isfinite(train), train, np.nan), axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    train_filled = np.where(np.isfinite(train), train, medians)
    test_filled = np.where(np.isfinite(test), test, medians)
    means = train_filled.mean(axis=0)
    scales = train_filled.std(axis=0, ddof=0)
    scales = np.where(scales > 1e-12, scales, 1.0)
    return (train_filled - means) / scales, (test_filled - means) / scales, means, scales


def prepared_split(
    dataset: dict,
    matrices: dict[str, np.ndarray],
    mirror: dict[str, np.ndarray],
    feature_set: str,
    train: np.ndarray,
    test: np.ndarray,
    augment: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels = dataset["labels"]
    x_train = matrices[feature_set][train]
    y_train = labels[train]
    epochs_train = dataset["epochs"][train]
    if augment:
        x_train = np.vstack([x_train, mirror[feature_set][train]])
        y_train = np.r_[y_train, -labels[train]]
        epochs_train = np.concatenate([epochs_train, dataset["epochs"][train][:, [0, 2, 1], :]], axis=0)
    train_template = template_feature(epochs_train, y_train, epochs_train, dataset["time"])
    test_template = template_feature(epochs_train, y_train, dataset["epochs"][test], dataset["time"])
    return np.c_[x_train, train_template], y_train, np.c_[matrices[feature_set][test], test_template], labels[test]


def model_scores(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, spec: dict) -> np.ndarray:
    x_train, x_test, _, _ = fill_and_scale(x_train, x_test)
    if spec["model"] == "bayes":
        model = BayesianRidge(compute_score=False, fit_intercept=True)
        model.fit(x_train, y_train)
        return model.predict(x_test)
    if spec["model"] == "pca_bayes":
        reducer = PCA(n_components=spec["variance"], svd_solver="full")
        reduced_train = reducer.fit_transform(x_train)
        reduced_test = reducer.transform(x_test)
        model = BayesianRidge(compute_score=False, fit_intercept=True)
        model.fit(reduced_train, y_train)
        return model.predict(reduced_test)
    means = {label: x_train[y_train == label].mean(axis=0) for label in (-1, 1)}
    centered = np.vstack([x_train[y_train == label] - means[label] for label in (-1, 1)])
    covariance = centered.T @ centered / max(centered.shape[0] - 2, 1)
    diagonal = np.diag(np.diag(covariance))
    trace_scale = np.trace(covariance) / max(covariance.shape[0], 1)
    regularized = (1 - spec["shrinkage"]) * covariance + spec["shrinkage"] * diagonal + spec["loading"] * trace_scale * np.eye(covariance.shape[0])
    weight = np.linalg.pinv(regularized, rcond=1e-8) @ (means[1] - means[-1])
    if spec["prior"] == "balanced":
        prior_positive = prior_negative = 0.5
    else:
        prior_positive = np.mean(y_train == 1)
        prior_negative = np.mean(y_train == -1)
    intercept = -0.5 * (means[1] + means[-1]) @ weight + np.log(prior_positive / prior_negative)
    return x_test @ weight + intercept


def outer_splits(dataset: dict) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    return list(splitter.split(np.zeros(dataset["labels"].size), dataset["labels"], dataset["groups"]))


def inner_splits(dataset: dict, outer_train: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    labels = dataset["labels"][outer_train]
    groups = dataset["groups"][outer_train]
    for count in (3, 2):
        try:
            splitter = StratifiedGroupKFold(n_splits=count, shuffle=True, random_state=SEED + count)
            splits = list(splitter.split(np.zeros(labels.size), labels, groups))
            if all(np.unique(labels[test]).size == 2 for _, test in splits):
                return splits
        except ValueError:
            continue
    raise RuntimeError("Unable to construct grouped inner folds")


def summarize_result(experiment_id: str, direction: str, variant: str, feature_set: str, labels: np.ndarray, predictions: np.ndarray, scores: np.ndarray, folds: list[dict], selections: list[dict], elapsed: float) -> dict:
    accuracy = float(accuracy_score(labels, predictions))
    balanced = float(balanced_accuracy_score(labels, predictions))
    return {
        "experiment_id": experiment_id,
        "direction": direction,
        "variant": variant,
        "feature_set": feature_set,
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "confusion": confusion_matrix(labels, predictions, labels=[-1, 1]),
        "predictions": predictions,
        "scores": scores,
        "folds": folds,
        "selections": selections,
        "elapsed_seconds": elapsed,
    }


def evaluate_fixed(dataset: dict, matrices: dict, mirror: dict, experiment_id: str, direction: str, feature_set: str, augment: bool, deadline: float) -> dict:
    start = time.monotonic()
    labels = dataset["labels"]
    predictions = np.zeros(labels.size, dtype=int)
    scores = np.zeros(labels.size)
    folds = []
    spec = {"model": "bayes"}
    for fold, (train, test) in enumerate(outer_splits(dataset), 1):
        if time.monotonic() >= deadline:
            raise TimeoutError
        x_train, y_train, x_test, y_test = prepared_split(dataset, matrices, mirror, feature_set, train, test, augment)
        fold_scores = model_scores(x_train, y_train, x_test, spec)
        fold_predictions = np.where(fold_scores >= 0, 1, -1)
        predictions[test], scores[test] = fold_predictions, fold_scores
        folds.append({"fold": fold, "accuracy": accuracy_score(y_test, fold_predictions), "balanced_accuracy": balanced_accuracy_score(y_test, fold_predictions), "n_test": test.size})
    variant = "BayesianRidge+mirror" if augment else "BayesianRidge"
    return summarize_result(experiment_id, direction, variant, feature_set, labels, predictions, scores, folds, [], time.monotonic() - start)


def stratum_values(dataset: dict, indices: np.ndarray, grouping: str) -> np.ndarray:
    if grouping == "global":
        return np.full(indices.size, "global", dtype=object)
    values = []
    for index in indices:
        row = dataset["metadata"][index]
        if grouping == "task":
            values.append(f"task_{row['task']}")
        elif grouping == "subject":
            values.append(f"subject_{row['subject']}")
        else:
            values.append(f"subject_{row['subject']}_task_{row['task']}")
    return np.asarray(values, dtype=object)


def grouped_scores(dataset: dict, matrices: dict, mirror: dict, train: np.ndarray, test: np.ndarray, spec: dict) -> np.ndarray:
    grouping = spec.get("grouping", "global")
    test_keys = stratum_values(dataset, test, grouping)
    train_keys = stratum_values(dataset, train, grouping)
    scores = np.zeros(test.size)
    for key in np.unique(test_keys):
        local_test = np.flatnonzero(test_keys == key)
        subgroup_train = train[train_keys == key]
        if subgroup_train.size < 20 or np.unique(dataset["labels"][subgroup_train]).size < 2:
            if grouping == "subject_task":
                task_grouping = np.array([f"task_{dataset['metadata'][index]['task']}" for index in train], dtype=object)
                test_task = dataset["metadata"][test[local_test[0]]]["task"]
                subgroup_train = train[task_grouping == f"task_{test_task}"]
            if subgroup_train.size < 20 or np.unique(dataset["labels"][subgroup_train]).size < 2:
                subgroup_train = train
        x_train, y_train, x_test, _ = prepared_split(dataset, matrices, mirror, spec["feature_set"], subgroup_train, test[local_test], spec.get("augment", False))
        scores[local_test] = model_scores(x_train, y_train, x_test, spec)
    return scores


def evaluate_stratified(dataset: dict, matrices: dict, mirror: dict, experiment_id: str, feature_set: str, grouping: str, deadline: float) -> dict:
    start = time.monotonic()
    labels = dataset["labels"]
    predictions = np.zeros(labels.size, dtype=int)
    scores = np.zeros(labels.size)
    folds = []
    spec = {"model": "bayes", "feature_set": feature_set, "grouping": grouping, "augment": False}
    for fold, (train, test) in enumerate(outer_splits(dataset), 1):
        if time.monotonic() >= deadline:
            raise TimeoutError
        fold_scores = grouped_scores(dataset, matrices, mirror, train, test, spec)
        fold_predictions = np.where(fold_scores >= 0, 1, -1)
        predictions[test], scores[test] = fold_predictions, fold_scores
        folds.append({"fold": fold, "accuracy": accuracy_score(labels[test], fold_predictions), "balanced_accuracy": balanced_accuracy_score(labels[test], fold_predictions), "n_test": test.size})
    return summarize_result(experiment_id, "direction2_stratified", f"BayesianRidge_{grouping}", feature_set, labels, predictions, scores, folds, [], time.monotonic() - start)


def nested_candidates(mode: str, feature_sets: list[str], augmentation_options: list[bool]) -> list[dict]:
    candidates = []
    for feature_set in feature_sets:
        for augment in augmentation_options:
            if mode in ("cov", "joint"):
                for shrinkage in (0.0, 0.25, 0.5, 0.75, 1.0):
                    for loading in (1e-4, 1e-2, 1e-1):
                        for prior in ("balanced", "empirical"):
                            candidates.append({"model": "cov", "feature_set": feature_set, "augment": augment, "shrinkage": shrinkage, "loading": loading, "prior": prior})
            if mode in ("pca", "joint"):
                for variance in (0.80, 0.90, 0.95):
                    candidates.append({"model": "pca_bayes", "feature_set": feature_set, "augment": augment, "variance": variance})
    return candidates


def evaluate_nested(dataset: dict, matrices: dict, mirror: dict, experiment_id: str, direction: str, mode: str, feature_sets: list[str], augmentation_options: list[bool], deadline: float) -> dict:
    start = time.monotonic()
    labels = dataset["labels"]
    predictions = np.zeros(labels.size, dtype=int)
    scores = np.zeros(labels.size)
    folds, selections = [], []
    candidates = nested_candidates(mode, feature_sets, augmentation_options)
    for fold, (outer_train, outer_test) in enumerate(outer_splits(dataset), 1):
        if time.monotonic() >= deadline:
            raise TimeoutError
        splits = inner_splits(dataset, outer_train)
        candidate_scores = []
        for candidate in candidates:
            inner_scores = []
            for local_train, local_test in splits:
                train = outer_train[local_train]
                test = outer_train[local_test]
                x_train, y_train, x_test, y_test = prepared_split(dataset, matrices, mirror, candidate["feature_set"], train, test, candidate["augment"])
                score = model_scores(x_train, y_train, x_test, candidate)
                prediction = np.where(score >= 0, 1, -1)
                inner_scores.append(balanced_accuracy_score(y_test, prediction))
            candidate_scores.append(float(np.mean(inner_scores)))
        best_index = int(np.argmax(candidate_scores))
        selected = candidates[best_index]
        x_train, y_train, x_test, y_test = prepared_split(dataset, matrices, mirror, selected["feature_set"], outer_train, outer_test, selected["augment"])
        fold_scores = model_scores(x_train, y_train, x_test, selected)
        fold_predictions = np.where(fold_scores >= 0, 1, -1)
        predictions[outer_test], scores[outer_test] = fold_predictions, fold_scores
        folds.append({"fold": fold, "accuracy": accuracy_score(y_test, fold_predictions), "balanced_accuracy": balanced_accuracy_score(y_test, fold_predictions), "n_test": outer_test.size})
        selections.append({"fold": fold, "inner_score": candidate_scores[best_index], **selected})
    variant = {"cov": "nested_covariance", "pca": "nested_PCA_BayesianRidge", "joint": "nested_joint_selector"}[mode]
    feature_label = feature_sets[0] if len(feature_sets) == 1 else "inner_selected"
    return summarize_result(experiment_id, direction, variant, feature_label, labels, predictions, scores, folds, selections, time.monotonic() - start)


def evaluate_nested_stratified(dataset: dict, matrices: dict, mirror: dict, deadline: float) -> dict:
    start = time.monotonic()
    labels = dataset["labels"]
    predictions = np.zeros(labels.size, dtype=int)
    scores = np.zeros(labels.size)
    folds, selections = [], []
    candidates = []
    for grouping in ("global", "task", "subject", "subject_task"):
        for feature_set in ("base", "base_expanded", "all_features"):
            candidates.append({"model": "bayes", "feature_set": feature_set, "grouping": grouping, "augment": False})
            for shrinkage, loading in ((0.25, 1e-2), (0.5, 1e-2), (0.75, 1e-1), (1.0, 1e-1)):
                candidates.append({"model": "cov", "feature_set": feature_set, "grouping": grouping, "augment": False, "shrinkage": shrinkage, "loading": loading, "prior": "balanced"})
    for fold, (outer_train, outer_test) in enumerate(outer_splits(dataset), 1):
        if time.monotonic() >= deadline:
            raise TimeoutError
        splits = inner_splits(dataset, outer_train)
        candidate_scores = []
        for candidate in candidates:
            values = []
            for local_train, local_test in splits:
                train, test = outer_train[local_train], outer_train[local_test]
                fold_scores = grouped_scores(dataset, matrices, mirror, train, test, candidate)
                values.append(balanced_accuracy_score(labels[test], np.where(fold_scores >= 0, 1, -1)))
            candidate_scores.append(float(np.mean(values)))
        best_index = int(np.argmax(candidate_scores))
        selected = candidates[best_index]
        fold_scores = grouped_scores(dataset, matrices, mirror, outer_train, outer_test, selected)
        fold_predictions = np.where(fold_scores >= 0, 1, -1)
        predictions[outer_test], scores[outer_test] = fold_predictions, fold_scores
        folds.append({"fold": fold, "accuracy": accuracy_score(labels[outer_test], fold_predictions), "balanced_accuracy": balanced_accuracy_score(labels[outer_test], fold_predictions), "n_test": outer_test.size})
        selections.append({"fold": fold, "inner_score": candidate_scores[best_index], **selected})
    return summarize_result("D2-13 训练折内分层选择", "direction2_stratified", "nested_stratified_selector", "inner_selected", labels, predictions, scores, folds, selections, time.monotonic() - start)


def csp_design(dataset: dict, filtered: dict[str, np.ndarray], train: np.ndarray, test: np.ndarray, include_base: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    csp_train, csp_test = csp_matrices(filtered, dataset["labels"], dataset["time"], train, test)
    x_train = np.c_[dataset["base"][train], csp_train] if include_base else csp_train
    x_test = np.c_[dataset["base"][test], csp_test] if include_base else csp_test
    train_template = template_feature(dataset["epochs"][train], dataset["labels"][train], dataset["epochs"][train], dataset["time"])
    test_template = template_feature(dataset["epochs"][train], dataset["labels"][train], dataset["epochs"][test], dataset["time"])
    return np.c_[x_train, train_template], dataset["labels"][train], np.c_[x_test, test_template], dataset["labels"][test]


def evaluate_csp(dataset: dict, filtered: dict[str, np.ndarray], experiment_id: str, include_base: bool, nested: bool, deadline: float) -> dict:
    start = time.monotonic()
    labels = dataset["labels"]
    predictions = np.zeros(labels.size, dtype=int)
    scores = np.zeros(labels.size)
    folds, selections = [], []
    parameters = [{"model": "cov", "shrinkage": shrinkage, "loading": loading, "prior": prior} for shrinkage in (0.0, 0.25, 0.5, 0.75, 1.0) for loading in (1e-4, 1e-2, 1e-1) for prior in ("balanced", "empirical")]
    for fold, (outer_train, outer_test) in enumerate(outer_splits(dataset), 1):
        if time.monotonic() >= deadline:
            raise TimeoutError
        if nested:
            cached = []
            for local_train, local_test in inner_splits(dataset, outer_train):
                cached.append(csp_design(dataset, filtered, outer_train[local_train], outer_train[local_test], include_base))
            candidate_scores = []
            for parameter in parameters:
                values = []
                for x_train, y_train, x_test, y_test in cached:
                    prediction = np.where(model_scores(x_train, y_train, x_test, parameter) >= 0, 1, -1)
                    values.append(balanced_accuracy_score(y_test, prediction))
                candidate_scores.append(float(np.mean(values)))
            best_index = int(np.argmax(candidate_scores))
            parameter = parameters[best_index]
            selections.append({"fold": fold, "inner_score": candidate_scores[best_index], "feature_set": "base_CSP" if include_base else "CSP", "augment": False, **parameter})
        else:
            parameter = {"model": "bayes"}
        x_train, y_train, x_test, y_test = csp_design(dataset, filtered, outer_train, outer_test, include_base)
        fold_scores = model_scores(x_train, y_train, x_test, parameter)
        fold_predictions = np.where(fold_scores >= 0, 1, -1)
        predictions[outer_test], scores[outer_test] = fold_predictions, fold_scores
        folds.append({"fold": fold, "accuracy": accuracy_score(y_test, fold_predictions), "balanced_accuracy": balanced_accuracy_score(y_test, fold_predictions), "n_test": outer_test.size})
    variant = "nested_covariance_CSP" if nested else "BayesianRidge_CSP"
    feature_set = "base_CSP" if include_base else "CSP"
    return summarize_result(experiment_id, "direction5_CSP", variant, feature_set, labels, predictions, scores, folds, selections, time.monotonic() - start)


def wilson_interval(successes: int, count: int) -> tuple[float, float]:
    z = 1.959963984540054
    proportion = successes / count
    denominator = 1 + z**2 / count
    center = (proportion + z**2 / (2 * count)) / denominator
    radius = z * np.sqrt(proportion * (1 - proportion) / count + z**2 / (4 * count**2)) / denominator
    return float(center - radius), float(center + radius)


def result_rows(results: list[dict], dataset: dict, quality: dict, total_elapsed: float) -> list[dict]:
    rows = [
        {
            "record_type": "optimization_feature_quality",
            "experiment_id": "gamma_quality",
            "optimization_direction": "direction3_morphology",
            "fit_success_rate": quality["fit_success_rate"],
            "fit_r2_median": quality["fit_r2_median"],
            "metric": "finite_fraction",
            "value": quality["finite_fraction"],
            "elapsed_seconds": total_elapsed,
        }
    ]
    labels = dataset["labels"]
    for result in results:
        successes = int(np.sum(result["predictions"] == labels))
        low, high = wilson_interval(successes, labels.size)
        common = {
            "model": result["variant"],
            "feature_set": result["feature_set"],
            "protocol": "nested_grouped5" if result["selections"] else "grouped5",
            "sample_unit": "single_trial",
            "n_trials": labels.size,
            "experiment_id": result["experiment_id"],
            "optimization_direction": result["direction"],
            "classifier_variant": result["variant"],
            "elapsed_seconds": result["elapsed_seconds"],
        }
        rows.append({"record_type": "optimization_summary", "metric": "accuracy", "value": result["accuracy"], "balanced_accuracy": result["balanced_accuracy"], "ci_low": low, "ci_high": high, **common})
        for fold in result["folds"]:
            rows.append({"record_type": "optimization_fold", "outer_fold": fold["fold"], **fold, **common})
        for true_index, true_side in enumerate(("left", "right")):
            for predicted_index, predicted_side in enumerate(("left", "right")):
                rows.append({"record_type": "optimization_confusion", "true_side": true_side, "predicted_side": predicted_side, "count": int(result["confusion"][true_index, predicted_index]), **common})
        for index, metadata in enumerate(dataset["metadata"]):
            rows.append(
                {
                    "record_type": "optimization_prediction",
                    "file": metadata["file"],
                    "subject": metadata["subject"],
                    "task": metadata["task"],
                    "trial": metadata["trial"],
                    "true_side": "left" if labels[index] == -1 else "right",
                    "predicted_side": "left" if result["predictions"][index] == -1 else "right",
                    "score": result["scores"][index],
                    **common,
                }
            )
        for selection in result["selections"]:
            parameter_text = json.dumps({key: value for key, value in selection.items() if key not in ("fold", "inner_score")}, ensure_ascii=False, sort_keys=True)
            rows.append({"record_type": "optimization_selection", "outer_fold": selection["fold"], "inner_score": selection["inner_score"], "selected_parameters": parameter_text, **common})
    return rows


def write_results(root: Path, optimization_rows: list[dict]) -> None:
    path = root / "P3_结果.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        old_fields = list(reader.fieldnames or [])
        old_rows = list(reader)
    baseline_rows = [row for row in old_rows if not row.get("record_type", "").startswith("optimization_")]
    fields = old_fields + [column for column in EXTRA_COLUMNS if column not in old_fields]
    for row in optimization_rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(baseline_rows + optimization_rows)
    temporary.replace(path)
    separate = root / "P3优化_结果.csv"
    with separate.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(optimization_rows)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reread = list(csv.DictReader(handle))
    if len(reread) != len(baseline_rows) + len(optimization_rows):
        raise RuntimeError("CSV row-count verification failed")
    for before, after in zip(baseline_rows, reread[: len(baseline_rows)]):
        if any(before.get(field, "") != after.get(field, "") for field in old_fields):
            raise RuntimeError("Baseline CSV rows changed")


def plot_comparison(root: Path, results: list[dict], count: int) -> None:
    labels = ["P3原始基线"] + [result["experiment_id"] for result in results]
    accuracies = [BASELINE_ACCURACY] + [result["accuracy"] for result in results]
    intervals = [wilson_interval(round(BASELINE_ACCURACY * count), count)] + [wilson_interval(int(round(result["accuracy"] * count)), count) for result in results]
    y = np.arange(len(labels))[::-1]
    height = max(4.2, 0.34 * len(labels) + 1.6)
    fig, axis = plt.subplots(figsize=(7.1, height))
    colors = ["#7A7A7A"] + ["#3B82B8" if value < 0.60 else "#D9822B" for value in accuracies[1:]]
    for index, (center, interval, color) in enumerate(zip(accuracies, intervals, colors)):
        axis.errorbar(center * 100, y[index], xerr=[[center * 100 - interval[0] * 100], [interval[1] * 100 - center * 100]], fmt="o", color=color, ecolor=color, capsize=2.5, markersize=4.5, lw=1.2)
    for index, result in enumerate(results, 1):
        fold_values = [fold["accuracy"] * 100 for fold in result["folds"]]
        axis.scatter(fold_values, np.full(len(fold_values), y[index]) + np.linspace(-0.10, 0.10, len(fold_values)), s=8, color="#9BB7CF", alpha=0.75, zorder=2)
    for value, style in ((BASELINE_ACCURACY * 100, "--"), (60, ":"), (70, "-.")):
        axis.axvline(value, color="#555555", lw=0.9, ls=style)
    display_labels = [f"{label}  {accuracy*100:.1f}%" for label, accuracy in zip(labels, accuracies)]
    axis.set_yticks(y, display_labels)
    axis.set_xlabel("单试次折外准确率（%）")
    axis.grid(axis="x", color="#D9D9D9", lw=0.6, alpha=0.8)
    axis.set_xlim(min(30, min(interval[0] for interval in intervals) * 100 - 2), max(80, max(interval[1] for interval in intervals) * 100 + 6))
    fig.suptitle("P3 单试次优化：严格分组五折下的方案比较", fontsize=10, weight="bold", x=0.31, y=0.985, ha="left")
    fig.text(0.31, 0.963, f"n={count}；汇总点为 Wilson 95% 区间，浅色小点为外层折；竖线依次为基线 53.15%、兜底 60%、期望 70%", fontsize=6.5, ha="left", va="top")
    fig.subplots_adjust(left=0.36, right=0.96, top=0.94, bottom=0.08)
    base = root / "P3优化_对比图"
    require_matplotlib_panel_alignment(fig, json_out=base.with_suffix(".alignment.json"), overlay_svg=base.with_suffix(".alignment.svg"), tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def update_plan(root: Path, results: list[dict], quality: dict, elapsed: float, timed_out: bool) -> None:
    path = root / "P3优化_方案.txt"
    text = path.read_text(encoding="utf-8")
    start_marker = "<!-- P3_OPT_AUTO_START -->"
    end_marker = "<!-- P3_OPT_AUTO_END -->"
    lines = [start_marker, "十一、实际优化结果（自动生成）"]
    for result in results:
        parameters = [selection.get("model", "") + "/" + selection.get("feature_set", "") + ("/mirror" if selection.get("augment") else "") for selection in result["selections"]]
        parameter_summary = ", ".join(sorted(set(parameters))) if parameters else "固定 BayesianRidge"
        lines.append(f"- {result['experiment_id']}（{result['direction']}，{result['feature_set']}）：准确率 {result['accuracy']*100:.3f}%，平衡准确率 {result['balanced_accuracy']*100:.3f}%；折准确率=" + "/".join(f"{fold['accuracy']*100:.1f}%" for fold in result["folds"]) + f"；内层选择={parameter_summary}。")
    best = max(results, key=lambda item: (item["balanced_accuracy"], item["accuracy"]))
    lines.extend(
        [
            f"- 单试次 Gamma 拟合成功率 {quality['fit_success_rate']*100:.2f}%，R² 中位数 {quality['fit_r2_median']:.3f}，形态矩阵有限值比例 {quality['finite_fraction']*100:.2f}%。",
            f"- 当前最优：{best['experiment_id']}，准确率 {best['accuracy']*100:.3f}%，平衡准确率 {best['balanced_accuracy']*100:.3f}%。",
            f"- 总运行时间 {elapsed/60:.2f} 分钟；90 分钟止损触发：{'是' if timed_out else '否'}。",
            "- 所有标准化、缺失填补、PCA、协方差参数、类别先验、模板和镜像增强均在外层训练折内完成；外层测试折仅用于一次折外评价。",
            end_marker,
        ]
    )
    block = "\n".join(lines)
    if start_marker in text and end_marker in text:
        prefix = text.split(start_marker, 1)[0].rstrip()
        suffix = text.split(end_marker, 1)[1].lstrip()
        text = prefix + "\n\n" + block + ("\n\n" + suffix if suffix else "\n")
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    configure_plotting()
    start = time.monotonic()
    deadline = start + args.limit_minutes * 60
    paths = sorted((root / "C题" / "dataset").glob("VisualCog?_Task-?.mat"))
    if len(paths) != 4:
        raise RuntimeError(f"Expected 4 MAT files, found {len(paths)}")
    dataset = build_dataset([extract_file(path) for path in paths])
    matrices, mirror, quality, _ = build_feature_matrices(dataset)
    filtered_csp = csp_band_epochs(dataset["epochs"])
    experiments = [
        ("D1-1 扩展特征", "direction1_features", "fixed", ["expanded_only"], [False]),
        ("D1-2 基础+扩展", "direction1_features", "fixed", ["base_expanded"], [False]),
        ("D1-3 增强+扩展", "direction1_features", "fixed", ["enhanced_expanded"], [False]),
        ("D2-1 正则协方差", "direction2_blda", "cov", ["base_expanded"], [False]),
        ("D2-2 正则协方差全特征", "direction2_blda", "cov", ["all_features"], [False]),
        ("D2-3 PCA+BLDA", "direction2_blda", "pca", ["base_expanded", "all_features"], [False]),
        ("D3-1 Gamma形态", "direction3_morphology", "fixed", ["morphology_only"], [False]),
        ("D3-2 基础+Gamma", "direction3_morphology", "fixed", ["base_morphology"], [False]),
        ("D3-3 全部特征", "direction3_morphology", "fixed", ["all_features"], [False]),
        ("D4-1 镜像正则协方差", "direction4_mirror", "cov", ["all_features"], [True]),
        ("D4-2 训练折内联合选择", "direction4_mirror", "joint", ["base", "base_expanded", "all_features"], [False, True]),
    ]
    results = []
    timed_out = False
    for experiment_id, direction, mode, feature_sets, augmentation in experiments:
        if time.monotonic() >= deadline:
            timed_out = True
            break
        try:
            if mode == "fixed":
                result = evaluate_fixed(dataset, matrices, mirror, experiment_id, direction, feature_sets[0], augmentation[0], deadline)
            else:
                result = evaluate_nested(dataset, matrices, mirror, experiment_id, direction, mode, feature_sets, augmentation, deadline)
        except TimeoutError:
            timed_out = True
            break
        results.append(result)
        print(f"{experiment_id}: accuracy={result['accuracy']:.4f}, balanced={result['balanced_accuracy']:.4f}, elapsed={result['elapsed_seconds']:.1f}s")
    stratified_experiments = []
    for grouping, label in (("task", "任务"), ("subject", "被试"), ("subject_task", "被试任务")):
        for feature_set, feature_label in (("base", "基础"), ("base_expanded", "基础扩展"), ("all_features", "全特征")):
            stratified_experiments.append((f"D2-{len(stratified_experiments)+4} {label}{feature_label}", feature_set, grouping))
    for experiment_id, feature_set, grouping in stratified_experiments:
        if time.monotonic() >= deadline:
            timed_out = True
            break
        try:
            result = evaluate_stratified(dataset, matrices, mirror, experiment_id, feature_set, grouping, deadline)
        except TimeoutError:
            timed_out = True
            break
        results.append(result)
        print(f"{experiment_id}: accuracy={result['accuracy']:.4f}, balanced={result['balanced_accuracy']:.4f}, elapsed={result['elapsed_seconds']:.1f}s")
    if not timed_out:
        try:
            result = evaluate_nested_stratified(dataset, matrices, mirror, deadline)
            results.append(result)
            print(f"{result['experiment_id']}: accuracy={result['accuracy']:.4f}, balanced={result['balanced_accuracy']:.4f}, elapsed={result['elapsed_seconds']:.1f}s")
        except TimeoutError:
            timed_out = True
    fallback_experiments = [
        ("D5-1 多频带时空", "fixed", "spectro_spatial"),
        ("D5-2 基础+多频带时空", "cov", "base_spectro"),
    ]
    for experiment_id, mode, feature_set in fallback_experiments:
        if timed_out or time.monotonic() >= deadline:
            timed_out = True
            break
        try:
            if mode == "fixed":
                result = evaluate_fixed(dataset, matrices, mirror, experiment_id, "direction5_spectro_spatial", feature_set, False, deadline)
            else:
                result = evaluate_nested(dataset, matrices, mirror, experiment_id, "direction5_spectro_spatial", mode, [feature_set], [False], deadline)
        except TimeoutError:
            timed_out = True
            break
        results.append(result)
        print(f"{experiment_id}: accuracy={result['accuracy']:.4f}, balanced={result['balanced_accuracy']:.4f}, elapsed={result['elapsed_seconds']:.1f}s")
    for experiment_id, include_base, nested in (("D5-3 CSP", False, False), ("D5-4 基础+CSP", True, False), ("D5-5 正则基础+CSP", True, True)):
        if timed_out or time.monotonic() >= deadline:
            timed_out = True
            break
        try:
            result = evaluate_csp(dataset, filtered_csp, experiment_id, include_base, nested, deadline)
        except TimeoutError:
            timed_out = True
            break
        results.append(result)
        print(f"{experiment_id}: accuracy={result['accuracy']:.4f}, balanced={result['balanced_accuracy']:.4f}, elapsed={result['elapsed_seconds']:.1f}s")
    if not results:
        raise RuntimeError("No optimization experiment completed before deadline")
    elapsed = time.monotonic() - start
    rows = result_rows(results, dataset, quality, elapsed)
    write_results(root, rows)
    plot_comparison(root, results, dataset["labels"].size)
    update_plan(root, results, quality, elapsed, timed_out)
    best = max(results, key=lambda item: (item["balanced_accuracy"], item["accuracy"]))
    print(f"best={best['experiment_id']}, accuracy={best['accuracy']:.4f}, balanced={best['balanced_accuracy']:.4f}")
    print(f"gamma_success={quality['fit_success_rate']:.4f}, gamma_r2_median={quality['fit_r2_median']:.4f}, total_minutes={elapsed/60:.2f}, timed_out={timed_out}")


if __name__ == "__main__":
    main()
