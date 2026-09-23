from pathlib import Path
import unittest

import yaml

from Ques1.quality import build_quality_outputs


ROOT = Path(__file__).resolve().parents[1]


class QualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with (ROOT / "configs" / "cti.yaml").open(encoding="utf-8") as stream:
            cls.config = yaml.safe_load(stream)
        cls.trials, cls.report = build_quality_outputs(ROOT / "C题" / "dataset", cls.config)

    def test_saturation_counts(self) -> None:
        expected = {
            ("VisualCogA_Task-1.mat", "Fz"): (0, 0),
            ("VisualCogA_Task-1.mat", "F3"): (0, 4573),
            ("VisualCogA_Task-1.mat", "F4"): (0, 6711),
            ("VisualCogA_Task-2.mat", "Fz"): (0, 0),
            ("VisualCogA_Task-2.mat", "F3"): (452, 2825),
            ("VisualCogA_Task-2.mat", "F4"): (410, 963),
            ("VisualCogB_Task-1.mat", "Fz"): (610, 3135),
            ("VisualCogB_Task-1.mat", "F3"): (0, 2246),
            ("VisualCogB_Task-1.mat", "F4"): (0, 2921),
            ("VisualCogB_Task-2.mat", "Fz"): (113, 237),
            ("VisualCogB_Task-2.mat", "F3"): (484, 2809),
            ("VisualCogB_Task-2.mat", "F4"): (678, 5158),
        }
        indexed = self.report.set_index(["file", "channel"])
        for key, counts in expected.items():
            actual = indexed.loc[key, ["sat_negative", "sat_positive"]]
            self.assertEqual(tuple(actual.astype(int)), counts)

    def test_usable_trial_counts(self) -> None:
        actual = self.report.groupby("file")["usable_trials"].first().to_dict()
        expected = {
            "VisualCogA_Task-1.mat": 67,
            "VisualCogA_Task-2.mat": 43,
            "VisualCogB_Task-1.mat": 51,
            "VisualCogB_Task-2.mat": 40,
        }
        self.assertEqual(actual, expected)

    def test_trial_quality_fields(self) -> None:
        fields = ["bad_frac_Fz", "bad_frac_F3", "bad_frac_F4", "n_sat"]
        self.assertFalse(self.trials[fields].isna().any().any())
        for field in fields[:3]:
            self.assertTrue(self.trials[field].between(0.0, 1.0).all())


if __name__ == "__main__":
    unittest.main()
