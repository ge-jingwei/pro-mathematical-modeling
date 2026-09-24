"""Verify saved states, provenance, metrics, and label-blind source extraction."""
from pathlib import Path
import argparse
import hashlib
import json
import tokenize
import io
import numpy as np
import pandas as pd
from Ques3.utils import FS, ROOT, dump, load_early, fold_sources, q2_utils
from Ques3.cognitive_model import simulate, crossings, behavior_metrics, eeg_metrics, gpu_candidates


def verify(out, q2, device):
    checks = []
    def check(name, condition):
        checks.append(dict(check=name, passed=bool(condition)))
        if not condition:
            dump(out / "verification.json", checks)
            raise AssertionError(name)
    trials = pd.read_csv(out / "trial_table.csv")
    predictions = pd.read_csv(out / "out_of_fold_predictions.csv")
    check("unique_raw_trials", not trials.duplicated(["record", "trial_id"]).any())
    check("all_400_trials_present", len(trials) == 400)
    check("task1_response_missing_not_behavioral", (trials.loc[trials.task_id == 1, "response_direction"] == 0).all())
    check("all_200_behavioral_trials_tested_once", len(predictions) == 200 and predictions.row_id.is_unique and set(predictions.row_id) == set(trials.loc[trials.task_id == 2, "row_id"]))
    answered = trials.response_direction != 0
    check("correctness_uses_separate_labels", (trials.loc[answered, "is_correct"].astype(bool) == (trials.loc[answered, "cue_direction"] == trials.loc[answered, "response_direction"])).all())
    check("no_response_correctness_missing", trials.loc[~answered, "is_correct"].isna().all())
    check("pre_motor_window", np.allclose(trials.loc[answered, "response_time"] - trials.loc[answered, "analysis_end_time"], .1))
    check("sampled_window_excludes_at_least_100ms", ((trials.loc[answered, "response_time"] - trials.loc[answered, "sampled_analysis_end_time"]) >= .1).all())
    check("no_fabricated_no_response", not ((trials.task_id == 2) & (trials.response_direction == 0)).any())
    x = load_early(q2, trials)
    recomputed_eeg = []
    for train_group, test_group in [("A", "B"), ("B", "A")]:
        fold = f"{train_group}_to_{test_group}"
        info = json.loads((out / f"{fold}_parameters.json").read_text(encoding="utf-8"))
        current_model = json.loads((q2 / f"{fold}_final.json").read_text(encoding="utf-8"))
        check(f"{fold}_q2_model_matches_current_input", info["q2_model"] == current_model)
        with np.load(out / f"{fold}_states.npz") as states:
            t, p = states["time_s"], states["parameters"]
            active = states["behavioral_model_available"]
            train, test = states["training"], states["testing"]
            check(f"{fold}_no_train_test_overlap", not np.any(train & test))
            check(f"{fold}_q2_train_group_only", set(trials.iloc[info["q2_training_ids"]].record_group) == {train_group})
            check(f"{fold}_initial_states_zero", np.all(states["memory"][active, 0] == 0) and np.all(states["decision"][active, 0] == 0))
            check(f"{fold}_task1_states_explicitly_missing", np.isnan(states["memory"][~active]).all() and np.isnan(states["decision"][~active]).all())
            check(f"{fold}_source_evidence_sign", np.allclose(states["source_evidence"], states["q_right"] - states["q_left"]))
            check(f"{fold}_no_evidence_before_acquisition", np.all(states["evidence"][:, t < info["evidence_available_s"]] == 0))
            memory, decision = simulate(states["evidence"][active], p, 1 / FS)
            check(f"{fold}_state_replay", np.allclose(memory, states["memory"][active]) and np.allclose(decision, states["decision"][active]))
            alone = simulate(states["evidence"][np.flatnonzero(active)[:1]], p, 1 / FS)
            check(f"{fold}_no_cross_trial_state_carry", np.allclose(alone[1][0], decision[0]))
            gpu = gpu_candidates(states["evidence"][active][:3, :130], p[None], device)
            cpu = simulate(states["evidence"][active][:3, :130], p, 1 / FS)[1]
            check(f"{fold}_gpu_cpu_match", np.allclose(gpu[0], cpu, atol=1e-10))
            pr, ct = crossings(states["decision"][test], p[-1], t, info["horizon"])
            stored = predictions[predictions.fold == fold].sort_values("row_id")
            check(f"{fold}_first_crossing_replay", np.array_equal(pr, stored.predicted_response) and np.allclose(ct, stored.predicted_crossing_time, equal_nan=True))
            check(f"{fold}_no_imputed_crossing_time", stored.loc[stored.predicted_response == 0, "predicted_crossing_time"].isna().all())
            altered = trials.copy()
            held = altered.record_group == test_group
            altered.loc[held, "cue_direction"] *= -1
            altered.loc[held, "response_direction"] *= -1
            altered.loc[held, "cue_to_response_latency"] = 999.
            _, altered_evidence, _, provenance = fold_sources(x, altered, q2, train_group, t, device)
            check(f"{fold}_heldout_labels_and_time_do_not_change_sources", np.allclose(altered_evidence, states["evidence"]) and provenance["source_scale"] == info["source_scale"])
            changed = x.copy()
            changed[held] *= 3
            _, _, _, rescaled = fold_sources(changed, trials, q2, train_group, t, device)
            check(f"{fold}_test_eeg_does_not_fit_scale", rescaled["source_scale"] == info["source_scale"])
            a = states["eeg_mask"] & train[:, None]
            design = np.stack([states["memory"], states["decision"]], axis=-1)[a]
            residual = np.moveaxis(states["observed_eeg"] - states["q2_eeg"], 1, 2)[a]
            w = np.linalg.solve(design.T @ design + info["mapping_ridge"] * np.eye(2), design.T @ residual)
            check(f"{fold}_mapping_uses_train_only", np.allclose(w, states["mapping"], atol=1e-8))
            for scope in ["full_pre_response", "after_evidence"]:
                mask = states["eeg_mask"] & test[:, None]
                if scope == "after_evidence":
                    mask &= t[None] > info["evidence_available_s"]
                for model, key in [("q2_only", "q2_eeg"), ("q2_plus_states", "extended_eeg")]:
                    for row in eeg_metrics(states["observed_eeg"], states[key], mask, fold, model):
                        row["scope"] = scope
                        recomputed_eeg.append(row)
    choice, timing, confusion = behavior_metrics(predictions)
    for filename, frame, key in [("choice_metrics.csv", choice, ["fold"]), ("time_metrics.csv", timing, ["fold", "model"]),
                                  ("confusion_matrix.csv", confusion, ["fold", "classifier", "actual", "predicted"]),
                                  ("eeg_metrics.csv", pd.DataFrame(recomputed_eeg), ["fold", "scope", "model", "channel"])]:
        stored = pd.read_csv(out / filename).sort_values(key).reset_index(drop=True)
        frame = frame.sort_values(key).reset_index(drop=True)
        numeric = frame.select_dtypes(include=np.number).columns
        check(f"reproduce_{filename}", np.allclose(stored[numeric], frame[numeric], equal_nan=True))
    hashes = {}
    for path in sorted((ROOT / "Ques3").glob("*.py")):
        source = path.read_text(encoding="utf-8-sig")
        compile(source, str(path), "exec")
        comments = [token.string for token in tokenize.generate_tokens(io.StringIO(source).readline) if token.type == tokenize.COMMENT]
        check(f"{path.name}_english_comments_only", not any("\u4e00" <= char <= "\u9fff" for text in comments for char in text))
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    dump(out / "source_hashes.json", hashes)
    dump(out / "verification.json", dict(passed=len(checks), failed=0, checks=checks))
    print(f"Verified {len(checks)} checks", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--q2", type=Path, required=True)
    args = parser.parse_args()
    verify(args.out, args.q2, q2_utils.choose_device()[0])
