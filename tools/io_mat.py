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
    labels: tuple[str, ...]
    eeg: np.ndarray
    decon: np.ndarray
    ecg: np.ndarray
    cue: np.ndarray
    action: np.ndarray
    timestamps: np.ndarray


def load_record(path: str | Path) -> Record:
    path = Path(path)
    match = re.fullmatch(r"VisualCog([AB])_Task-([12])\.mat", path.name)
    if not match:
        raise ValueError(f"Unexpected data filename: {path.name}")
    mat = loadmat(path, squeeze_me=True, simplify_cells=True)
    labels = tuple(str(value).strip() for value in np.atleast_1d(mat["DataLabel"]))
    data = np.asarray(mat["data"], dtype=float)
    index = {label: i for i, label in enumerate(labels)}
    action_label = next(label for label in labels if label.startswith(("Action:", "TgtAct:")))
    return Record(
        path=path,
        subject=match.group(1),
        task=int(match.group(2)),
        sample_rate=float(np.asarray(mat["SampleRate"]).reshape(-1)[0]),
        labels=labels,
        eeg=data[[index[name] for name in ("Fz", "F3", "F4")]],
        decon=data[[index[name] for name in ("FzDecon", "F3Decon", "F4Decon")]],
        ecg=data[index["ECG"]],
        cue=data[index["VisCue:L-1/R+1"]],
        action=data[index[action_label]],
        timestamps=data[index["TimeStamp"]],
    )


def load_records(data_dir: str | Path) -> list[Record]:
    paths = sorted(Path(data_dir).glob("VisualCog[AB]_Task-[12].mat"))
    if len(paths) != 4:
        raise ValueError(f"Expected four data files, found {len(paths)}")
    return [load_record(path) for path in paths]
