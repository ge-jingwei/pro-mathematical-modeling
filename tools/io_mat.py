from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
from scipy.io import loadmat


@dataclass(frozen=True)
class Record:
    path: Path
    subject: str
    task: int
    sample_rate: float
    eeg_labels: tuple[str, ...]
    eeg: np.ndarray
    ecg: np.ndarray
    cue: np.ndarray
    action: np.ndarray
    action_label: str
    timestamps: np.ndarray


def _label_index(labels: tuple[str, ...], name: str) -> int:
    try:
        return labels.index(name)
    except ValueError as exc:
        raise ValueError(f"Missing channel: {name}") from exc


def load_record(path: str | Path) -> Record:
    path = Path(path)
    match = re.fullmatch(r"VisualCog([AB])_Task-([12])\.mat", path.name)
    if not match:
        raise ValueError(f"Unexpected data filename: {path.name}")
    mat = loadmat(path, squeeze_me=True, simplify_cells=True)
    labels = tuple(str(value).strip() for value in np.atleast_1d(mat["DataLabel"]))
    data = np.asarray(mat["data"], dtype=float)
    sample_rate = float(np.asarray(mat["SampleRate"]).reshape(-1)[0])
    if data.ndim != 2 or data.shape[0] != len(labels):
        raise ValueError(f"Invalid data shape in {path.name}: {data.shape}")
    eeg_labels = ("Fz", "F3", "F4")
    action_label = next(
        (label for label in labels if label.startswith(("Action:", "TgtAct:"))),
        None,
    )
    if action_label is None:
        raise ValueError(f"Missing action channel in {path.name}")
    return Record(
        path=path,
        subject=match.group(1),
        task=int(match.group(2)),
        sample_rate=sample_rate,
        eeg_labels=eeg_labels,
        eeg=np.vstack([data[_label_index(labels, label)] for label in eeg_labels]),
        ecg=data[_label_index(labels, "ECG")],
        cue=data[_label_index(labels, "VisCue:L-1/R+1")],
        action=data[_label_index(labels, action_label)],
        action_label=action_label,
        timestamps=data[_label_index(labels, "TimeStamp")],
    )


def load_records(data_dir: str | Path) -> list[Record]:
    paths = sorted(Path(data_dir).glob("VisualCog[AB]_Task-[12].mat"))
    if len(paths) != 4:
        raise ValueError(f"Expected four data files, found {len(paths)}")
    return [load_record(path) for path in paths]
