# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from silk.cv.homography import HomographicSampler
from silk.test.util import max_tensor_diff, OverwriteTensorEquality, string_to_tensor


class _UnitTests(unittest.TestCase):
    def test_homographic_sampler(self):
        n = 1
        hc = HomographicSampler(n, "cpu")

        # 12 x 15 test image
        img = string_to_tensor(
            """
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            """,
        )
        img = img.unsqueeze(0)
        img_shape = img.shape[1:]

        epsilon = 0.01

        # check initial identity
        crop = hc.extract_crop(img, img_shape, mode="nearest")
        with OverwriteTensorEquality(torch):
            self.assertEqual(crop, img)

        # check scaling on center crop
        hc.scale(0.45)
        center_crop = img[0, 3 : 3 + 6, 4 : 4 + 7]  # noqa: E203

        crop = hc.extract_crop(img, (6, 7), mode="nearest")
        with OverwriteTensorEquality(torch):
            self.assertEqual(crop, center_crop)

        # check 90 rotated center crop
        hc.rotate(torch.pi / 2.0, clockwise=True)
        crop = hc.extract_crop(img, (7, 6), mode="nearest")
        with OverwriteTensorEquality(torch):
            self.assertEqual(crop, center_crop.rot90())

        # check 180 rotated center crop
        hc.rotate(torch.pi / 2.0, clockwise=True)
        crop = hc.extract_crop(img, (6, 7), mode="nearest")
        with OverwriteTensorEquality(torch):
            self.assertEqual(crop, center_crop.rot90().rot90())

        # check 270 rotated center crop
        hc.rotate(torch.pi / 2.0, clockwise=True)
        crop = hc.extract_crop(img, (7, 6), mode="nearest")
        with OverwriteTensorEquality(torch):
            self.assertEqual(crop, center_crop.rot90().rot90().rot90())

        # check reverting to identity
        hc.rotate(3 * torch.pi / 2.0, clockwise=False)
        hc.scale(2.0)
        crop = hc.extract_crop(img, img_shape, mode="nearest")
        with OverwriteTensorEquality(torch):
            self.assertEqual(crop, img)

        # check x anti-clockwise rotation
        hc.scale(0.5)
        hc.rotate(torch.pi / 4.0, axis="x")
        crop = hc.extract_crop(img, (6, 7), mode="nearest")
        with OverwriteTensorEquality(torch):
            # further away side should have 0 in corners
            self.assertEqual(crop[..., 0, 0].item(), 0.0)
            self.assertEqual(crop[..., 0, -1].item(), 0.0)

        # check x clockwise rotation
        hc.rotate(2.0 * torch.pi / 4.0, clockwise=True, axis="x")
        crop = hc.extract_crop(img, (6, 7), mode="nearest")
        with OverwriteTensorEquality(torch):
            # further away side should have 0 in corners
            self.assertEqual(crop[..., -1, 0].item(), 0.0)
            self.assertEqual(crop[..., -1, -1].item(), 0.0)

        # check reverting to identity
        hc.rotate(-torch.pi / 4.0, clockwise=True, axis="x")
        hc.scale(2.0)
        crop = hc.extract_crop(img, img_shape, mode="nearest")
        with OverwriteTensorEquality(torch):
            self.assertEqual(crop, img)

        # check y anti-clockwise rotation
        hc.scale(0.5)
        hc.rotate(torch.pi / 4.0, clockwise=False, axis="y")
        crop = hc.extract_crop(img, (6, 7), mode="nearest")
        with OverwriteTensorEquality(torch):
            # rotated axis is unchanged
            self.assertEqual(crop[:, :, :, 3], center_crop[:, 3])
            # further away side should have 0 in corners
            self.assertEqual(crop[..., 0, -1].item(), 0.0)
            self.assertEqual(crop[..., -1, -1].item(), 0.0)

        # check y clockwise rotation
        hc.rotate(2.0 * torch.pi / 4.0, clockwise=True, axis="y")
        crop = hc.extract_crop(img, (6, 7), mode="nearest")
        with OverwriteTensorEquality(torch):
            # rotated axis is unchanged
            self.assertEqual(crop[:, :, :, 3], center_crop[:, 3])
            # further away side should have 0 in corners
            self.assertEqual(crop[..., 0, 0].item(), 0.0)
            self.assertEqual(crop[..., -1, 0].item(), 0.0)

        # check reverting to identity
        hc.rotate(torch.pi / 4.0, clockwise=False, axis="y")
        hc.scale(2.0)
        crop = hc.extract_crop(img, img_shape, mode="nearest")
        with OverwriteTensorEquality(torch):
            self.assertEqual(crop, img)

        # check shift
        hc.scale(0.25)
        hc.shift((+0.25 - epsilon, -0.25 + epsilon))
        crop = hc.extract_crop(img, (3, 4), mode="nearest")
        with OverwriteTensorEquality(torch):
            self.assertEqual(crop, img[0, 3 : 3 + 3, 7 : 7 + 4])  # noqa: E203

        # check local rotation after shift
        hc.rotate(+torch.pi / 2.0, local_center=True, clockwise=True)
        crop = hc.extract_crop(img, (4, 3), mode="nearest")
        with OverwriteTensorEquality(torch):
            self.assertEqual(crop, img[0, 3 : 3 + 3, 7 : 7 + 4].rot90())  # noqa: E203

        # check reverting to identity
        hc.rotate(-torch.pi / 2.0, local_center=True, clockwise=True)
        hc.shift((-0.25 + epsilon, +0.25 - epsilon))
        hc.scale(4.0)
        crop = hc.extract_crop(img, img_shape, mode="nearest")
        with OverwriteTensorEquality(torch):
            self.assertEqual(crop, img)

        # check local scaling after shift
        hc.shift((-0.25 + epsilon, -0.25 + epsilon))
        hc.scale(0.25, local_center=True)

        crop = hc.extract_crop(img, (3, 4), mode="nearest")
        with OverwriteTensorEquality(torch):
            self.assertEqual(crop, img[0, 3 : 3 + 3, 4 : 4 + 4])  # noqa: E203

        # check reverting to identity
        hc.scale(4.0, local_center=True)
        hc.shift((+0.25, +0.25))
        crop = hc.extract_crop(img, img_shape, mode="nearest")
        with OverwriteTensorEquality(torch):
            self.assertEqual(crop, img)

        # check transform points
        hc.scale(0.5)
        hc.rotate(torch.pi / 4.0)
        hc.shift((+0.1, -0.2))

        self.assertLess(
            max_tensor_diff(
                hc.transform_points(hc.src_coords, direction="forward"),
                hc.dest_coords,
            ),
            1e-5,
        )

        self.assertLess(
            max_tensor_diff(
                hc.transform_points(hc.dest_coords, direction="backward"),
                hc.src_coords,
            ),
            1e-5,
        )


if __name__ == "__main__":
    unittest.main()
