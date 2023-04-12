# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from silk.test.util import OverwriteTensorEquality
from silk.transforms.cv.image import CHWToHWC, HWCToCHW


class _UnitTests(unittest.TestCase):
    def test_hwc_to_chw_and_reverse(self):
        transf = HWCToCHW()
        reverse_transf = CHWToHWC()

        image = torch.rand((3, 5, 7))
        image_batch = torch.rand((3, 5, 7, 11))
        not_image_0 = torch.rand((3, 5))
        not_image_1 = torch.rand((3, 5, 7, 11, 13))

        with self.assertRaises(RuntimeError):
            transf(not_image_0)

        with self.assertRaises(RuntimeError):
            transf(not_image_1)

        out_image = transf(image)
        out_image_batch = transf(image_batch)

        self.assertEqual(image[1, 2, 3], out_image[3, 1, 2])
        self.assertEqual(image_batch[1, 2, 3, 5], out_image_batch[1, 5, 2, 3])

        with OverwriteTensorEquality(
            torch, check_shape=True, check_device=True, check_dtype=True
        ):
            self.assertEqual(image, reverse_transf(out_image))
            self.assertEqual(image_batch, reverse_transf(out_image_batch))


if __name__ == "__main__":
    unittest.main()
