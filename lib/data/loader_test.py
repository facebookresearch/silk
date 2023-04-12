# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import unittest

from silk.data.loader import StatefulDataLoader

from torch.utils.data import Dataset, IterableDataset


class _MockDataset(Dataset):
    def __init__(self, size) -> None:
        super().__init__()
        self._size = size

    def __len__(self):
        return self._size

    def __getitem__(self, index):
        return index + 1


class _MockIterableDataset(IterableDataset):
    def __iter__(self):
        n = 1
        while True:
            yield n
            n += 1


class _UnitTests(unittest.TestCase):
    def test_stateful_data_loader(self):
        M = 12
        dataset = _MockDataset(M)
        iterable_dataset = _MockIterableDataset()

        def collate_fn(x):
            return x

        # 1. check if stateful with dataset
        dataloader = StatefulDataLoader(
            dataset, cycle=False, batch_size=None, collate_fn=collate_fn
        )
        data_full = list(itertools.islice(dataloader, M))

        # should raise since cycle is disabled
        with self.assertRaises(StopIteration):
            next(iter(dataloader))

        dataloader = StatefulDataLoader(
            dataset, cycle=False, batch_size=None, collate_fn=collate_fn
        )
        data_0 = list(itertools.islice(dataloader, 5))
        data_1 = list(itertools.islice(dataloader, M - 5))

        # should raise since cycle is disabled
        with self.assertRaises(StopIteration):
            next(iter(dataloader))

        self.assertSequenceEqual(data_full, list(range(1, M + 1)))
        self.assertSequenceEqual(data_full, data_0 + data_1)

        # 2. check when cycle is enabled
        dataloader = StatefulDataLoader(
            dataset, cycle=True, batch_size=None, collate_fn=collate_fn
        )
        data_full = list(itertools.islice(dataloader, M))

        self.assertEqual(data_full[0], next(iter(dataloader)))

        # 3. check if stateful with iterable dataset
        dataloader = StatefulDataLoader(
            iterable_dataset, batch_size=None, collate_fn=collate_fn
        )
        data_full = list(itertools.islice(dataloader, M))

        dataloader = StatefulDataLoader(
            iterable_dataset, batch_size=None, collate_fn=collate_fn
        )
        data_0 = list(itertools.islice(dataloader, 5))
        data_1 = list(itertools.islice(dataloader, M - 5))

        self.assertSequenceEqual(data_full, list(range(1, M + 1)))
        self.assertSequenceEqual(data_full, data_0 + data_1)

        # 4. check reset method
        dataloader = StatefulDataLoader(
            dataset, cycle=False, batch_size=None, collate_fn=collate_fn
        )
        data_0 = list(itertools.islice(dataloader, 5))

        dataloader.reset()

        data_1 = list(itertools.islice(dataloader, 5))

        self.assertSequenceEqual(data_0, data_1)


def main():
    unittest.main()


if __name__ == "__main__":
    unittest.main()
