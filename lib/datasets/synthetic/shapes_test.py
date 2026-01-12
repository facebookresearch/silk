# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
from silk.datasets.synthetic.shapes import (
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    SyntheticShapes,
)


class _UnitTests(unittest.TestCase):
    def test_synthetic_shapes_dataset(self):
        size = 6
        height = DEFAULT_HEIGHT // 2
        width = DEFAULT_WIDTH // 8
        dataset = SyntheticShapes(
            height=height,
            width=width,
            seed=0,
            drawing_primitives="all",
        )

        iterator = iter(dataset)
        for _ in range(size):
            img, points = next(iterator)

            self.assertSequenceEqual(img.shape, (height, width, 1))
            self.assertEqual(img.dtype, np.float32)
            self.assertEqual(points.shape[1], 2)
            self.assertEqual(points.dtype, float)

            for i in range(points.shape[0]):
                # lower bound test
                self.assertGreaterEqual(points[i][0], 0)
                self.assertGreaterEqual(points[i][1], 0)

                # upper bound test
                self.assertLess(points[i][0], height)
                self.assertLess(points[i][1], width)


if __name__ == "__main__":
    unittest.main()
