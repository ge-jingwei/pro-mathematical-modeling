from pathlib import Path
import unittest

import numpy as np
import yaml

from Ques1.quality import fit_thresholds
from Ques1.reliability import compute_reliability, fit_record_reliability
from tools.io_mat import load_records


ROOT = Path(__file__).resolve().parents[1]


class ReliabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with (ROOT / "configs" / "cti.yaml").open(encoding="utf-8") as stream:
            cls.config = yaml.safe_load(stream)
        cls.record = load_records(ROOT / "C题" / "dataset")[0]

    def test_range_and_hard_factors(self) -> None:
        training = np.ones(self.record.eeg.shape[1], dtype=bool)
        reliability, _ = fit_record_reliability(self.record, self.config, training)
        hard = np.isclose(np.abs(self.record.eeg), self.config["quality"]["saturation"])
        self.assertEqual(reliability.shape, self.record.eeg.shape)
        self.assertTrue(((reliability >= 0.0) & (reliability <= 1.0)).all())
        self.assertTrue((reliability[hard] == 0.0).all())

    def test_formula(self) -> None:
        eeg = np.array([[0.0, 1.0, 2.0], [0.0, 0.0, 0.0], [0.0, -1.0, -2.0]])
        thresholds = fit_thresholds(eeg, 8.0, 6.0)
        reliability = compute_reliability(eeg, thresholds, 1000.0, 4)
        difference = np.diff(eeg, axis=1, prepend=eeg[:, :1])
        common = np.median(eeg, axis=0)
        expected = 1.0 / (1.0 + (np.abs(difference) / thresholds.transient[:, None]) ** 2)
        expected *= 1.0 / (
            1.0 + (np.abs(eeg - common) / thresholds.cross_channel[:, None]) ** 2
        )
        np.testing.assert_allclose(reliability, expected)

    def test_thresholds_ignore_held_out_samples(self) -> None:
        eeg = self.record.eeg.copy()
        training = np.zeros(eeg.shape[1], dtype=bool)
        training[: eeg.shape[1] // 2] = True
        before = fit_thresholds(eeg, 8.0, 6.0, training)
        eeg[:, ~training] = 1e9
        after = fit_thresholds(eeg, 8.0, 6.0, training)
        np.testing.assert_array_equal(before.transient, after.transient)
        np.testing.assert_array_equal(before.cross_channel, after.cross_channel)


if __name__ == "__main__":
    unittest.main()
