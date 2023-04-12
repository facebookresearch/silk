# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from silk.backbones.superpoint.utils import prob_map_to_points_map, remove_border_points
from silk.profiler import timeit
from silk.test.util import OverwriteTensorEquality


class _UnitTests(unittest.TestCase):
    def test_remove_border_points(self):
        example_image = torch.rand((8, 8))
        border_dist = 4

        # remove points a distance of 4 pixels from border
        # should make entire image zeros
        img_no_border = remove_border_points(example_image, border_dist=border_dist)

        self.assertTrue(
            img_no_border.equal(
                torch.zeros(example_image.shape[0], example_image.shape[1])
            )
        )

    def test_nms_parity(self):
        torch.manual_seed(0)

        batch_size = 32
        height = 60
        width = 80

        input = torch.rand((batch_size, height, width))

        with timeit(message_template="original nms duration: {duration}"):
            original_result = prob_map_to_points_map(
                input,
                0.0,
                4,
                4,
                use_fast_nms=False,
            )
        with timeit(message_template="fast nms duration: {duration}"):
            fast_result = prob_map_to_points_map(
                input,
                0.0,
                4,
                4,
                use_fast_nms=True,
            )

        with OverwriteTensorEquality(
            torch,
            check_device=True,
            check_shape=True,
            check_dtype=True,
        ):
            self.assertEqual(original_result, fast_result)


if __name__ == "__main__":
    unittest.main()
