from pathlib import Path
import tempfile
import unittest

import pandas as pd

from Ques1.events import TRIAL_COLUMNS, build_trials, write_trials
from tools.io_mat import load_records


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "C题" / "dataset"


class EventTests(unittest.TestCase):
    def test_event_assertions(self) -> None:
        trials = build_trials(DATA_DIR)
        self.assertEqual(len(trials), 400)
        self.assertEqual(list(trials.columns), TRIAL_COLUMNS)
        self.assertTrue(trials.groupby("file").size().eq(100).all())
        self.assertEqual(int(trials["warmup_flag"].sum()), 1)
        conflicts = trials.groupby(["subject", "task"])["side_conflict_flag"].sum()
        self.assertEqual(int(conflicts.loc[("A", 1)]), 0)
        self.assertEqual(int(conflicts.loc[("B", 1)]), 2)

    def test_formal_eeg_channels(self) -> None:
        for record in load_records(DATA_DIR):
            self.assertEqual(record.eeg_labels, ("Fz", "F3", "F4"))
            self.assertEqual(record.eeg.shape[0], 3)

    def test_write_trials(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "trials.csv"
            expected = write_trials(DATA_DIR, output)
            actual = pd.read_csv(output)
            self.assertTrue(output.is_file())
            self.assertEqual(len(actual), len(expected))
            self.assertEqual(list(actual.columns), TRIAL_COLUMNS)


if __name__ == "__main__":
    unittest.main()
