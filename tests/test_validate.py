import unittest

import numpy as np

from Ques1.validate import continuous_blocks


class ValidationTests(unittest.TestCase):
    def test_continuous_blocks(self) -> None:
        blocks = continuous_blocks(100, 20)
        self.assertEqual(len(blocks), 5)
        np.testing.assert_array_equal(np.concatenate(blocks), np.arange(100))
        for block in blocks:
            np.testing.assert_array_equal(np.diff(block), np.ones(19))

    def test_invalid_block_size(self) -> None:
        with self.assertRaises(ValueError):
            continuous_blocks(99, 20)


if __name__ == "__main__":
    unittest.main()
