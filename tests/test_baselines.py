import unittest

import numpy as np

from Ques1.baselines import hard_trial_mask, short_interpolation
from Ques1.denoise import fit_noise_scales, reliability_kalman


class BaselineTests(unittest.TestCase):
    def test_hard_rejection(self) -> None:
        epochs = np.zeros((3, 3, 20))
        epochs[1, 0, 4] = 1000.0
        epochs[2, 1, 5] = 500.0
        np.testing.assert_array_equal(hard_trial_mask(epochs, 1000.0, 400.0), [True, False, False])

    def test_short_interpolation_only(self) -> None:
        epochs = np.tile(np.arange(10.0), (1, 3, 1))
        reliability = np.ones_like(epochs)
        reliability[0, 0, 3:5] = 0.0
        reliability[0, 1, 2:7] = 0.0
        result = short_interpolation(epochs, reliability, 2)
        np.testing.assert_allclose(result[0, 0], np.arange(10.0))
        np.testing.assert_allclose(result[0, 1], epochs[0, 1])

    def test_missing_observation_is_not_used(self) -> None:
        epochs = np.zeros((2, 3, 12))
        epochs[:, 0, 5] = 1000.0
        weights = np.ones_like(epochs)
        weights[:, 0, 5] = 0.0
        scale, process = fit_noise_scales(epochs[:, :, :4])
        result = reliability_kalman(epochs, weights, scale, process, 0.001)
        self.assertTrue(np.isfinite(result).all())
        self.assertLess(abs(result[0, 0, 5]), 1.0)


if __name__ == "__main__":
    unittest.main()
