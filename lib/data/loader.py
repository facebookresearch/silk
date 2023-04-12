# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Iterator
from typing import Any, Dict, List, Union

import torch.utils.data
from torch.utils.data.dataset import Dataset, IterableDataset


class StatefulDataLoader(torch.utils.data.DataLoader):
    """Stateful version of a PyTorch DataLoader. Stateful in a sense that subsequent calls to `__iter__` will continue previously created iterator.
    The returned iterator is essentially shared.

    This is especially useful when dealing with infinite or very large datasets. In those cases, we have to limit the epoch size during training.
    However, that introduces a problem : Most training cycle implementation will reset the dataloader iterator by calling `iter(dataloader)`, which makes
    us train on the same initial data over and over again. A `StatefulDataLoader` solves that issue by making calls to `__iter__` return the same shared iterator.

    Examples
    --------

    ```python
    from torch.utils.data import IterableDataset
    from silk.data.loader import StatefulDataLoader
    import itertools

    class MyDataset(IterableDataset):
        def __iter__(self):
            n = 1
            while True:
                yield n
                n += 1

    dataset = MyDataset()
    dataloader = StatefulDataLoader(dataset, batch_size=None, collate_fn=lambda x: x)

    print(list(itertools.islice(dataloader, 5)))
    # >>> [1, 2, 3, 4, 5]
    print(list(itertools.islice(dataloader, 5)))
    # >>> [6, 7, 8, 9, 10]
    ```
    """

    class _Iterator(Iterator):
        """Iternal iterator.

        IMPORTANT : This class HAS to subclass the abstract class `Iterator` since PyTorch Lightning runs the `next()` function on iterators by filtering them by class.
        In case this super class is removed, the iterator will not be processed properly.
        """

        def __init__(self, data_loader, cycle) -> None:
            super().__init__()
            self._cycle = cycle
            self._dataloader = data_loader
            self._reset_iterator()

        def _reset_iterator(self):
            self._iterator = torch.utils.data.DataLoader.__iter__(self._dataloader)

        def __next__(self):
            if self._cycle:
                try:
                    return next(self._iterator)
                except StopIteration:
                    self._reset_iterator()
            return next(self._iterator)

    def __init__(
        self,
        dataset: Union[Dataset, IterableDataset],
        *args: List[Any],
        cycle: bool = False,
        **kwargs: Dict[str, Any],
    ):
        """

        Parameters
        ----------
        dataset : Union[Dataset, IterableDataset]
            Dataset provided to the dataloader (c.f. torch.utils.data.dataset.DataLoader).
        *args : List[Any]
            Positional arguments passed to the `DataLoader`.
        cycle : bool, optional
            Automatically reset the iterator once the end of the dataset is reached, by default False.
        **kwargs : Dict[str, Any]
            Keyword arguments passed to the `DataLoader`.
        """
        super().__init__(dataset, *args, **kwargs)
        self._cycle = cycle
        self._stateful_iterator = None

    def reset(self):
        """Resets the shared iterator to the beginning of the dataset."""
        self._stateful_iterator = StatefulDataLoader._Iterator(self, self._cycle)

    def __iter__(self):
        if self._stateful_iterator is None:
            self.reset()
        return self._stateful_iterator
