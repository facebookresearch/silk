# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
The unittests for the magicpoint model magicpoint.py
"""

import unittest

import torch

from silk.backbones.superpoint.magicpoint import MagicPoint


class _UnitTests(unittest.TestCase):
    def _build_mock_model(self, **kwargs):
        return MagicPoint(**kwargs)

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
        sample_input_tensor = torch.rand(32, 1, 128, 128)
        _, output = model.forward(sample_input_tensor)

        # output size should be (batch_size, 65, H/8, W/8)
        self.assertEqual(output.shape, torch.Size([32, 65, 16, 16]))

    def test_forward_bn_channels(self):
        """
        Test case for the forward function using batchnorm and
        a non-1 number of channels.

        Note that we can only test to compare output size, not
        actual outputs, with the SuperPoint paper's pretrained model
        because they do not use batchnorm.
        """
        # initialize model with num_channels=3
        model = self._build_mock_model(num_channels=3, use_batchnorm=True)
        model.eval()

        # input must have the number of channels that the model was initialized to have
        sample_input_tensor = torch.rand(32, 3, 128, 128)
        _, output = model.forward(sample_input_tensor)

        # output size should be (batch_size, 65, H/8, W/8)
        self.assertEqual(output.shape, torch.Size([32, 65, 16, 16]))

    def test_logits_to_prob_channels(self):
        """
        Test case for the function logits_to_prob.
        """
        model = self._build_mock_model(num_channels=3)
        sample_input_tensor = torch.rand(32, 3, 128, 128)
        prob = model.forward_flow("probability", sample_input_tensor)

        # output shape is batch_size, 65, H/8, W/8
        # each value is the probability that that pixel is a corner
        self.assertEqual(prob.shape, torch.Size([32, 65, 16, 16]))

        # test to see that probabilities sum to one
        self.assertTrue(torch.allclose(torch.sum(prob, 1), torch.tensor([1.0])))

    def test_logits_to_prob(self):
        """
        Test case for the function logits_to_prob.
        """
        model = self._build_mock_model()
        sample_input_tensor = torch.rand(32, 1, 128, 128)
        prob = model.forward_flow("probability", sample_input_tensor)

        # output shape is batch_size, 65, H/8, W/8
        # each value is the probability that that pixel is a corner
        self.assertEqual(prob.shape, torch.Size([32, 65, 16, 16]))

        # test to see that probabilities sum to one
        self.assertTrue(torch.allclose(torch.sum(prob, 1), torch.tensor([1.0])))

    def test_depth_to_space(self):
        """
        Test case for the function depth_to_space.
        """
        model = self._build_mock_model()
        sample_input_tensor = torch.rand(32, 1, 128, 128)
        score = model.forward_flow("score", sample_input_tensor)

        # output shape is batch_size, H, W
        # each value is the probability that that pixel is a corner
        self.assertEqual(score.shape, torch.Size([32, 1, 128, 128]))


def main():
    unittest.main()


if __name__ == "__main__":
    main()
