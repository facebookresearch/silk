# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from itertools import islice

from silk.datasets.abstract import RandomizedIterable


class _MockRandomizedIterable(RandomizedIterable):
    def __init__(self, seed):
        super().__init__(seed=seed)

    def _generate_item(self, random_generator):
        return random_generator.integers(0, 1024)


class _UnitTests(unittest.TestCase):
    def test_randomized_iterable(self):
        seed_0 = 9842730473286469327042386846885
        seed_1 = seed_0
        seed_2 = 42
        sampling_size = 3 * 2 * 16

        dataset_0 = _MockRandomizedIterable(seed=seed_0)
        dataset_1 = _MockRandomizedIterable(seed=seed_1)
        dataset_2 = _MockRandomizedIterable(seed=seed_2)

        # 1. datasets with same seed should produce same data sequences
        results_0 = list(islice(dataset_0, sampling_size))
        results_1 = list(islice(dataset_1, sampling_size))
        results_2 = list(islice(dataset_2, sampling_size))

        self.assertListEqual(results_0, results_1)
        n_differents = sum(v0 != v2 for v0, v2 in zip(results_0, results_2))
        self.assertGreater(n_differents, 0)

        # 2. iterating again should produce same sequence as previous call
        results_0_again = list(islice(dataset_0, sampling_size))

        self.assertListEqual(results_0, results_0_again)

        # 3. re seeded dataset should produce same sequence as if it has been initialized with the seed
        dataset_1.seed = seed_2

        results_1 = list(islice(dataset_1, sampling_size))
        self.assertListEqual(results_1, results_2)

        # 4. simulate three workers setting seed, start, and step
        dataset_0.seed = seed_0
        dataset_1.seed = seed_0
        dataset_2.seed = seed_0

        dataset_0.slice(0, 3)
        dataset_1.slice(1, 3)
        dataset_2.slice(2, 3)

        results_01 = list(islice(dataset_0, sampling_size // 3))
        results_02 = list(islice(dataset_1, sampling_size // 3))
        results_03 = list(islice(dataset_2, sampling_size // 3))

        result_0_by_part = []
        for val in zip(results_01, results_02, results_03):
            result_0_by_part.extend(val)

        self.assertSequenceEqual(result_0_by_part, results_0)

        # 5. check slicing of already sliced dataset
        dataset_0.slice(1, 2)
        results_slice = list(islice(dataset_0, sampling_size // (3 * 2)))

        self.assertSequenceEqual(results_slice, results_01[1::2])


if __name__ == "__main__":
    unittest.main()
