# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest

from silk.datasets.scannet.loftr import scannet_all

DATASET_PATH = "/datasets01/scannet/082518/scans"
TRAIN_NPZ_PATH = "/private/home/gleize/code/external/LoFTR/train-data/scannet_indices/scene_data/train"
MODE = "train"
MIN_OVERLAP_SCORE_TRAIN = 0.4

# run test only if running on FAIR cluster
if os.path.exists(DATASET_PATH):

    class _UnitTests(unittest.TestCase):
        @staticmethod
        def _get_all(arr):
            return tuple(arr[i] for i in range(len(arr)))

        @staticmethod
        def _get_dataset():
            return scannet_all(
                DATASET_PATH, TRAIN_NPZ_PATH, MODE, MIN_OVERLAP_SCORE_TRAIN
            )

        def test_scannet_loftr(self):
            ds = _UnitTests._get_dataset()

            self.assertEqual(len(ds), 220519389)

            image0, image1, depth0, depth1, T_0to1, T_1to0, K_0, K_1 = ds[100]

            # check dims
            self.assertEqual(image0.ndim, 3)
            self.assertEqual(image1.ndim, 3)
            self.assertEqual(depth0.ndim, 2)
            self.assertEqual(depth1.ndim, 2)

            self.assertEqual(T_0to1.shape, (4, 4))
            self.assertEqual(T_1to0.shape, (4, 4))
            self.assertEqual(K_0.shape, (3, 3))
            self.assertEqual(K_1.shape, (3, 3))

    if __name__ == "__main__":
        unittest.main()
