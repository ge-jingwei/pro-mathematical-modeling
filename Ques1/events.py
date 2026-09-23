from pathlib import Path

import numpy as np
import pandas as pd

from tools.io_mat import Record, load_records


def _runs(signal: np.ndarray, magnitude: float) -> list[tuple[int, int, int]]:
    selected = np.isclose(np.abs(signal), magnitude)
    starts = np.flatnonzero(selected & ~np.r_[False, selected[:-1]])
    stops = np.flatnonzero(selected & ~np.r_[selected[1:], False]) + 1
    return [(int(a), int(b), int(np.sign(signal[a]))) for a, b in zip(starts, stops, strict=True)]


def parse_events(record: Record) -> pd.DataFrame:
    cues = _runs(record.cue, 1.0)
    targets = _runs(record.action, 1.0)
    responses = _runs(record.action, 2.0)
    if len(cues) != 100 or len(targets) != 100:
        raise ValueError(f"Unexpected event count in {record.path.name}")
    rows = []
    for i, (cue, target) in enumerate(zip(cues, targets, strict=True), 1):
        cue_on, cue_off, cue_side = cue
        target_on, target_off, target_side = target
        if record.task == 2:
            following = [item for item in responses if item[0] >= target_on and item[0] <= target_off + 1]
            response_on = following[0][0] if following else np.nan
            response_side = following[0][2] if following else 0
        else:
            response_on = target_off
            response_side = target_side
        rows.append(
            {
                "file": record.path.name,
                "subject": record.subject,
                "task": record.task,
                "trial": i,
                "cue_on": cue_on,
                "cue_duration_s": (cue_off - cue_on) / record.sample_rate,
                "target_on": target_on,
                "cue_target_s": (target_on - cue_on) / record.sample_rate,
                "response_on": response_on,
                "reaction_time_s": (response_on - target_on) / record.sample_rate,
                "cue_side": cue_side,
                "target_side": target_side,
                "response_side": response_side,
                "cue_target_congruent": cue_side == target_side,
            }
        )
    return pd.DataFrame(rows)


def build_events(data_dir: str | Path) -> pd.DataFrame:
    return pd.concat([parse_events(record) for record in load_records(data_dir)], ignore_index=True)
