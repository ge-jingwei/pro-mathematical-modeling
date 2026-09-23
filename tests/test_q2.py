import unittest

import numpy as np

from Ques2.model import ModelParameters, cortical_projection, encode_shape, lead_fields, simulate, triangle_image
from Ques2.validation import lateral_window


class QuestionTwoTests(unittest.TestCase):
    def test_shape_mirror(self) -> None:
        left = triangle_image(-1)
        right = triangle_image(1)
        self.assertLess(np.mean(np.abs(left - np.fliplr(right))), 1e-12)
        self.assertEqual(encode_shape(left).shape, (4, 64, 64))

    def test_model_output(self) -> None:
        times = np.arange(-0.2, 0.8, 1 / 256)
        model = {side: simulate(side, times, 256, ModelParameters(), 12) for side in (-1, 1)}
        self.assertEqual(model[-1].eeg.shape, (3, len(times)))
        self.assertTrue(np.isfinite(model[1].order_global).all())
        self.assertGreater(np.std(model[-1].eeg), 0)

    def test_lateral_window(self) -> None:
        times = np.arange(-0.2, 0.8, 1 / 256)
        model = {side: simulate(side, times, 256, ModelParameters(), 12).eeg for side in (-1, 1)}
        window, start, stop = lateral_window(model, times)
        self.assertGreater(window.sum(), 0)
        self.assertGreaterEqual(start, 0.02)
        self.assertLessEqual(stop, 0.56)

    def test_directional_projection_and_geometry(self) -> None:
        features = encode_shape(triangle_image(1))
        projected = cortical_projection(features, 12)
        self.assertEqual(projected.shape, (4, 12, 12))
        self.assertGreater(np.linalg.norm(projected[1] - projected[3]), 0)
        nominal = lead_fields(16)
        perturbed = lead_fields(16, 0.2, 1.2)
        self.assertTrue(np.allclose(nominal.sum(axis=(1, 2)), 1))
        self.assertFalse(np.allclose(nominal, perturbed))

    def test_sham_inputs_are_shape_independent(self) -> None:
        times = np.arange(-0.2, 0.4, 1 / 256)
        parameters = ModelParameters()
        left = simulate(-1, times, 256, parameters, 12, drive_mode="constant")
        right = simulate(1, times, 256, parameters, 12, drive_mode="constant")
        self.assertTrue(np.array_equal(left.cortical_input, right.cortical_input))


if __name__ == "__main__":
    unittest.main()
