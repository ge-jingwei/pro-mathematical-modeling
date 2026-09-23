from dataclasses import dataclass
from pathlib import Path

import numpy as np

from Ques1.events import parse_events
from Ques1.preprocess import prepare_epochs
from tools.io_mat import load_records


@dataclass(frozen=True)
class CueDataset:
    epochs: np.ndarray
    labels: np.ndarray
    tasks: np.ndarray
    groups: np.ndarray
    times: np.ndarray
    files: tuple[str, ...]


def load_cue_dataset(root: Path, config: dict) -> CueDataset:
    values = []
    labels = []
    tasks = []
    groups = []
    files = []
    times = None
    for group, record in enumerate(load_records(root / config["paths"]["data_dir"])):
        events = parse_events(record)
        prepared = prepare_epochs(record, events, config)
        keep = prepared.keep
        values.append(prepared.proposed[keep])
        labels.append(events["cue_side"].to_numpy()[keep])
        tasks.append(np.full(keep.sum(), record.task))
        groups.append(np.full(keep.sum(), group))
        files.append(record.path.name)
        times = prepared.times
    return CueDataset(np.concatenate(values), np.concatenate(labels), np.concatenate(tasks), np.concatenate(groups), times, tuple(files))


def erp_summary(dataset: CueDataset, bootstrap_samples: int, rng: np.random.Generator) -> dict:
    result = {}
    for task in (1, 2):
        for side in (-1, 1):
            selected = dataset.epochs[(dataset.tasks == task) & (dataset.labels == side)]
            mean = selected.mean(axis=0)
            indices = rng.integers(0, len(selected), size=(bootstrap_samples, len(selected)))
            boot = selected[indices].mean(axis=1)
            low, high = np.percentile(boot, [2.5, 97.5], axis=0)
            result[(task, side)] = {"mean": mean, "low": low, "high": high, "n": len(selected)}
    return result

