# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
The unit tests for the hpatches_dataset.py file.
"""

import os
import unittest

from silk.datasets.hpatches.hpatches_dataset import (
    HPATCHES_DEVFAIR_DIR,
    HPatchesDataset,
)


class TestHPatchesDataset(unittest.TestCase):
    def test_load_images(self):
        # load the hpatches dataset from the faircluster location
        datasets = os.path.join(os.getcwd(), "..", "/datasets01")
        hpatches_dir = os.path.join("hpatches", "01042022")

        path = os.path.join(datasets, hpatches_dir)

        # load all images, images with perspective change (v), and images with lighting change (i)
        dataset_all = HPatchesDataset(path)
        dataset_i = HPatchesDataset(path, img_alteration="i")
        dataset_v = HPatchesDataset(path, img_alteration="v")

        self.assertEqual(len(dataset_all), 580)
        self.assertEqual(len(dataset_i), 285)
        self.assertEqual(len(dataset_v), 295)

    def test_hpatches_dataset_class(self):
        datasets = "/datasets01"
        path = os.path.join(datasets, HPATCHES_DEVFAIR_DIR)

        # only load 5 image sets (5 pairs per image set) for speed here
        num_to_load = 5
        dataset = HPatchesDataset(path, img_alteration="v", num_to_load=num_to_load)

        # there should be a total of 5 * 5 = 25 pairs of images
        num_image_pairs = 5 * num_to_load
        self.assertEqual(len(dataset), num_image_pairs)


def main():
    unittest.main()


if __name__ == "__main__":
    main()
