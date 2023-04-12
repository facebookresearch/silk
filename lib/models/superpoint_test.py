# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
The unit tests for superpoint.py
"""

import unittest

import torch

from silk.backbones.superpoint.superpoint import SuperPoint


class _UnitTests(unittest.TestCase):
    def _build_mock_model(self, **kwargs):
        return SuperPoint(**kwargs)

    def test_forward_bn(self):
        """
        Test case for the forward function using batchnorm.

        Note that we can only test to compare output size, not
        actual outputs, with the SuperPoint paper's pretrained model
        because they do not use batchnorm.
        """
        model = self._build_mock_model(use_batchnorm=True)
        model.eval()

        # batch_size = 32, num_channels = 1, img_size = 128 x 128
        batch_size = 32
        sample_input_tensor = torch.rand(batch_size, 1, 128, 128)
        descriptor_output, detector_output = model.forward(sample_input_tensor)

        # the second dimension is the depth (equal to 256 in the model)
        # descriptor output size should be (batch_size, 256, H/8, W/8)
        self.assertEqual(descriptor_output.shape, torch.Size([batch_size, 256, 16, 16]))

        # detector output size should be (batch_size, 65, H/8, W/8)
        self.assertEqual(detector_output.shape, torch.Size([batch_size, 65, 16, 16]))

    def test_forward_bn_channels(self):
        """
        Test case for the forward function using batchnorm with non-1 number of channels.

        Note that we can only test to compare output size, not
        actual outputs, with the SuperPoint paper's pretrained model
        because they do not use batchnorm.
        """
        model = self._build_mock_model(num_channels=3, use_batchnorm=True)
        model.eval()

        # batch_size = 32, num_channels = 3, img_size = 128 x 128
        batch_size = 32
        sample_input_tensor = torch.rand(batch_size, 3, 128, 128)
        descriptor_output, detector_output = model.forward(sample_input_tensor)

        # the second dimension is the depth (equal to 256 in the model)
        # descriptor output size should be (batch_size, 256, H/8, W/8)
        self.assertEqual(descriptor_output.shape, torch.Size([batch_size, 256, 16, 16]))

        # detector output size should be (batch_size, 65, H/8, W/8)
        self.assertEqual(detector_output.shape, torch.Size([batch_size, 65, 16, 16]))


def main():
    unittest.main()


if __name__ == "__main__":
    main()
