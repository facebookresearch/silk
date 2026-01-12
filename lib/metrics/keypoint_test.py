# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from silk.metrics.keypoint import KeypointDetectionAveragePrecision
from silk.test.util import string_to_tensor
from torchmetrics import AveragePrecision


class _UnitTests(unittest.TestCase):
    @staticmethod
    def _ap_result(preds, targets):
        assert len(preds) == len(targets)

        ap_metric = AveragePrecision(num_classes=1)
        preds = torch.tensor(preds)
        targets = torch.tensor(targets)
        return ap_metric(preds, targets)

    def test_keypoint_detection_average_precision(self):
        empty_tensor = (torch.empty(0),)

        mm_metric = KeypointDetectionAveragePrecision(
            distance_threshold=2.0, allow_multimatching=True, compute_on_step=True
        )
        no_mm_metric = KeypointDetectionAveragePrecision(
            distance_threshold=2.0, allow_multimatching=False, compute_on_step=True
        )

        # test multimatching difference
        preds = (
            string_to_tensor(
                """
            0.0, 0.0, 1.0
            2.0, 0.0, 1.0
            0.0, 2.0, 0.5
            0.0, 2.1, 1.0
            """
            ),
        )
        targets = (
            string_to_tensor(
                """
            0.0, 0.0
            """
            ),
        )

        # 1 match, 3 mis-match
        ap_preds = [1.0] * 3 + [0.5]
        ap_targets = [1] + [0] * 3
        self.assertEqual(
            no_mm_metric(preds, targets).item(),
            _UnitTests._ap_result(ap_preds, ap_targets).item(),
        )

        # 3 match, 1 mis-match
        ap_preds = [1.0] * 3 + [0.5]
        ap_targets = [0] + [1] * 3
        self.assertEqual(
            mm_metric(preds, targets).item(),
            _UnitTests._ap_result(ap_preds, ap_targets).item(),
        )

        # test with empty targets
        ap_preds = [1.0] * 3 + [0.5] + [1.0] * 3 + [0.5]
        ap_targets = [0] + [1] * 3 + [0] * 4
        self.assertEqual(
            mm_metric(preds + preds, targets + empty_tensor).item(),
            _UnitTests._ap_result(ap_preds, ap_targets).item(),
        )

        # test with empty predictions
        ap_preds = [1.0] * 3 + [0.5] + [0.0]
        ap_targets = [0] + [1] * 3 + [1]
        self.assertEqual(
            mm_metric(preds + empty_tensor, targets + targets).item(),
            _UnitTests._ap_result(ap_preds, ap_targets).item(),
        )


if __name__ == "__main__":
    unittest.main()
