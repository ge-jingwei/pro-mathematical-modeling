from pathlib import Path

import numpy as np
import pandas as pd

from tools.io_mat import Record, load_records


TRIAL_COLUMNS = [
    "file",
    "subject",
    "task",
    "trial_index",
    "t_cue_on",
    "cue_dur",
    "t_tgt_on",
    "cue2tgt_gap",
    "t_resp",
    "RT",
    "side_cue",
    "side_tgt",
    "side_resp",
    "congruent",
    "bad_frac_Fz",
    "bad_frac_F3",
    "bad_frac_F4",
    "n_sat",
    "warmup_flag",
    "side_conflict_flag",
]


def _unit_runs(signal: np.ndarray) -> list[tuple[int, int, int]]:
    indices = np.flatnonzero(np.isclose(np.abs(signal), 1.0))
    if not len(indices):
        return []
    breaks = np.flatnonzero(
        (np.diff(indices) != 1) | (signal[indices[1:]] != signal[indices[:-1]])
    )
    groups = np.split(indices, breaks + 1)
    return [(int(group[0]), int(group[-1] + 1), int(np.sign(signal[group[0]]))) for group in groups]


def parse_record(record: Record) -> pd.DataFrame:
    cue_runs = _unit_runs(record.cue)
    target_runs = _unit_runs(record.action)
    if len(cue_runs) != len(target_runs):
        raise AssertionError(
            f"Event count mismatch in {record.path.name}: {len(cue_runs)} cues, "
            f"{len(target_runs)} targets"
        )
    rows = []
    for trial_index, (cue_run, target_run) in enumerate(
        zip(cue_runs, target_runs, strict=True), start=1
    ):
        cue_on, cue_off, side_cue = cue_run
        target_on, target_stop, side_tgt = target_run
        response = target_stop
        if response >= len(record.action):
            raise AssertionError(f"Missing response in {record.path.name}, trial {trial_index}")
        if record.task == 2:
            response_value = record.action[response]
            if not np.isclose(abs(response_value), 2.0):
                raise AssertionError(
                    f"Missing task-2 response marker in {record.path.name}, trial {trial_index}"
                )
            side_resp = int(np.sign(response_value))
        else:
            side_resp = side_tgt
        gap = target_on - cue_off
        rows.append(
            {
                "file": record.path.name,
                "subject": record.subject,
                "task": record.task,
                "trial_index": trial_index,
                "t_cue_on": cue_on,
                "cue_dur": cue_off - cue_on,
                "t_tgt_on": target_on,
                "cue2tgt_gap": gap,
                "t_resp": response,
                "RT": (response - target_on) / record.sample_rate,
                "side_cue": side_cue,
                "side_tgt": side_tgt,
                "side_resp": side_resp,
                "congruent": side_cue == side_tgt,
                "bad_frac_Fz": np.nan,
                "bad_frac_F3": np.nan,
                "bad_frac_F4": np.nan,
                "n_sat": np.nan,
                "warmup_flag": gap not in range(515, 520),
                "side_conflict_flag": record.task == 1 and side_cue != side_tgt,
            }
        )
    return pd.DataFrame(rows, columns=TRIAL_COLUMNS)


def validate_record(record: Record, trials: pd.DataFrame) -> None:
    name = record.path.name
    assert len(trials) == 100, f"{name}: expected 100 cues"
    assert trials["cue_dur"].isin([52, 53]).all(), f"{name}: invalid cue duration"
    standard_gap = trials["cue2tgt_gap"].between(515, 519)
    if record.subject == "A" and record.task == 1:
        exceptional = (trials["trial_index"] == 1) & (trials["cue2tgt_gap"] == 544)
        assert (standard_gap | exceptional).all() and exceptional.sum() == 1, f"{name}: invalid gap"
    else:
        assert standard_gap.all(), f"{name}: invalid cue-to-target gap"
    response_markers = np.flatnonzero(np.isclose(np.abs(record.action), 2.0))
    if record.task == 2:
        assert len(response_markers) == 100, f"{name}: expected 100 response markers"
        assert np.array_equal(response_markers, trials["t_resp"].to_numpy()), f"{name}: response mismatch"
        marker_sides = np.sign(record.action[response_markers]).astype(int)
        assert np.array_equal(marker_sides, trials["side_tgt"].to_numpy()), f"{name}: side mismatch"
        rate = float(trials["congruent"].mean())
        assert 0.45 <= rate <= 0.52, f"{name}: invalid congruence rate {rate}"
    else:
        assert not len(response_markers), f"{name}: task 1 contains response markers"
    assert (trials["t_resp"] > trials["t_tgt_on"]).all(), f"{name}: missing response segment"
    assert np.allclose(np.diff(record.timestamps), 1.0 / record.sample_rate), f"{name}: timestamp gap"
    assert record.eeg_labels == ("Fz", "F3", "F4") and record.eeg.shape[0] == 3


def build_trials(data_dir: str | Path) -> pd.DataFrame:
    frames = []
    for record in load_records(data_dir):
        trials = parse_record(record)
        validate_record(record, trials)
        frames.append(trials)
    return pd.concat(frames, ignore_index=True)[TRIAL_COLUMNS]


def write_trials(data_dir: str | Path, output_path: str | Path) -> pd.DataFrame:
    trials = build_trials(data_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trials.to_csv(output_path, index=False)
    return trials
