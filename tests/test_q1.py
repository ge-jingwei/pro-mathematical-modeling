from pathlib import Path
import unittest

import numpy as np

from Ques1.events import build_events
from Ques1.fitting import fit_components, gaussian3
from Ques1.preprocess import wavelet_shrink


ROOT = Path(__file__).resolve().parents[1]


class QuestionOneTests(unittest.TestCase):
    def test_events(self) -> None:
        events = build_events(ROOT / "C题" / "dataset")
        self.assertEqual(len(events), 400)
        self.assertTrue(events.groupby("file").size().eq(100).all())
        self.assertEqual(int(events.query("task == 2")["response_on"].isna().sum()), 0)

    def test_wavelet_shape(self) -> None:
        epochs = np.random.default_rng(1).normal(size=(4, 3, 256))
        result = wavelet_shrink(epochs, 0.5)
        self.assertEqual(result.shape, epochs.shape)
        self.assertTrue(np.isfinite(result).all())

    def test_constrained_fit(self) -> None:
        times = np.linspace(-0.2, 0.8, 256, endpoint=False)
        params = [0.0, -3.0, 0.12, 0.04, 4.0, 0.21, 0.05, 6.0, 0.36, 0.08]
        result = fit_components(times, gaussian3(times, *params))
        self.assertGreater(result["r2"], 0.99)
        self.assertTrue(250 <= result["params"][8] * 1000 <= 500)


if __name__ == "__main__":
    unittest.main()
