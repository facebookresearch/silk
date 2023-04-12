# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from silk.datasets.cached import CachedDataset
from silk.test.util import OverwriteTensorEquality, temporary_file
from silk.transforms.abstract import NamedContext


class _UnitTests(unittest.TestCase):
    def test_cached_dataset(self):
        iterable = [
            1,
            2.0,
            True,
            ("a", "b", "c"),
            {"a": 1, "b": 2, "c": 3},
            NamedContext(a=1, b=2, c=3),
            torch.rand(12, 12),
        ]

        with temporary_file() as filepath:
            dataset = CachedDataset.from_iterable(filepath, iterable)

            self.assertEqual(len(dataset), len(iterable))
            with OverwriteTensorEquality(
                torch, check_device=True, check_shape=True, check_dtype=True
            ):
                for i in range(len(dataset)):
                    self.assertEqual(dataset[i], iterable[i])


if __name__ == "__main__":
    unittest.main()
