# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
The unittests for the magicpoint model magicpoint.py
"""

import unittest

import torch

from silk.backbones.superpoint.vgg import VGG, vgg_block


class _UnitTests(unittest.TestCase):
    def _build_mock_backbone(self, **kwargs):
        return VGG(**kwargs)

    def test_vgg_block(self):
        """
        Test case for the function vgg_block.
        """
        in_channels = 1
        out_channels = 64
        kernel_size = 3
        vgg_blk = vgg_block(in_channels, out_channels, kernel_size)

        self.assertIsNotNone(vgg_blk)

        # test that running a tensor through the layer results in
        # the correct output shape
        input_ex_tensor = torch.rand(32, 1, 128, 128)
        sample_output = vgg_blk(input_ex_tensor)

        self.assertEqual(sample_output.shape, torch.Size([32, 64, 128, 128]))

    def test_vgg_block_channels(self):
        """
        Test case for the function vgg_block.
        """
        in_channels = 3
        out_channels = 64
        kernel_size = 3
        vgg_blk = vgg_block(in_channels, out_channels, kernel_size)

        self.assertIsNotNone(vgg_blk)

        # test that running a tensor through the layer results in
        # the correct output shape
        input_ex_tensor = torch.rand(32, 3, 128, 128)
        sample_output = vgg_blk(input_ex_tensor)

        self.assertEqual(sample_output.shape, torch.Size([32, 64, 128, 128]))

    def test_forward_bn(self):
        """
        Test case for the forward function using batchnorm.

        Note that we can only test to compare output size, not
        actual outputs, with the SuperPoint paper's pretrained model
        because they do not use batchnorm.
        """
        backbone = self._build_mock_backbone(use_batchnorm=True)

        # batch_size = 32, num_channels = 1, img_size = 128 x 128
        sample_input_tensor = torch.rand(32, 1, 128, 128)
        output = backbone.forward(sample_input_tensor)

        # output size should be (batch_size, 65, H/8, W/8)
        self.assertEqual(output.shape, torch.Size([32, 128, 16, 16]))

    def test_forward_bn_channels(self):
        """
        Test case for the forward function using batchnorm and
        a non-1 number of channels.

        Note that we can only test to compare output size, not
        actual outputs, with the SuperPoint paper's pretrained model
        because they do not use batchnorm.
        """
        # initialize model with num_channels=3
        backbone = self._build_mock_backbone(num_channels=3, use_batchnorm=True)

        # input must have the number of channels that the model was initialized to have
        sample_input_tensor = torch.rand(32, 3, 128, 128)
        output = backbone.forward(sample_input_tensor)

        # output size should be (batch_size, 65, H/8, W/8)
        self.assertEqual(output.shape, torch.Size([32, 128, 16, 16]))


def main():
    unittest.main()


if __name__ == "__main__":
    main()
