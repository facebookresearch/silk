# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Iterable

import torch
from numpy.random import default_rng, SeedSequence
from silk.logger import LOG
from torch.utils.data import Dataset, IterableDataset, random_split, Subset


def random_partition(
    dataset: Dataset,
    lengths: Iterable[int],
    partition_idx: int,
    seed: int = 0,
) -> Subset:
    """Randomly split the dataset and returns the specified partition.

    Parameters
    ----------
    dataset : Dataset
        Dataset to split.
    lengths : Iterable[int]
        Sizes of the partitions.
    partition_idx : int
        Index of the partition to return (0 <= partition_idx < len(lengths)).
    seed : int, optional
        Seed to use for random split, by default 0

    Returns
    -------
    Subset
        Randomly generated partition.
    """
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, lengths, generator)[partition_idx]


class RandomizedIterable(IterableDataset):
    r"""Abstract class for randomized iterable datasets.

    All datasets that represent a randomized iterable of data samples whose determinism can be controlled by a seed should subclass it.
    Such datasets are particularly useful to model randomly generated datasets in order to make them both reproducible and easy to deploy on multiple processes.

    When subclassed, the method `_generate_item` should be overwritten to return one data sample using `random_generator` as the random number generator.
    The default random generator is a `numpy.default_rng` object.
    If using another type of generator, the methods `_create_seed_sequence_generator` and `_get_next_random_generator` should be overwritten as well.

    Also, when using a randomized iterable dataset in a distributed setting, we do provide a convenient function to give to the pytorch data loader (see `RandomizedIterable.worker_init_fn`).

    Examples
    --------

    ```python
    from itertools import islice
    from silk.datasets.abstract import RandomizedIterable

    class CustomRandomizedIterable(RandomizedIterable):
        def __init__(self, seed):
            super().__init__(seed=seed)

        def _generate_item(self, random_generator):
            return random_generator.integers(0, 1024)


    dataset = CustomRandomizedIterable(seed=0)
    print([i for i in islice(dataset, 5)])
    # >>> [821, 677, 672, 913, 179]
    print([i for i in islice(dataset, 5)])
    # >>> [821, 677, 672, 913, 179]

    dataset.seed = 1
    print([i for i in islice(dataset, 5)])
    # >>> [15, 963, 436, 186, 768]
    ```

    """

    @staticmethod
    def worker_init_fn(worker_id: int) -> None:
        """Function to give to a data loader in order to handle the randomized iterable dataset in a distributed setting.

        Parameters
        ----------
        worker_id : int
            ID of the current worker iterating over this dataset.

        Raises
        ------
        RuntimeError
            Raised when the current worker dataset is not `RandomizedIterable`.

        Examples
        --------

        That function can be provided to a data loader as such : `torch.utils.data.DataLoader(..., worker_init_fn = RandomizedIterable.worker_init_fn)`.

        """

        worker_info = torch.utils.data.get_worker_info()

        if worker_info is not None:
            dataset = worker_info.dataset
            if not isinstance(dataset, RandomizedIterable):
                raise RuntimeError(
                    "Worker dataset is not compatible (should be `RandomizedIterable`)."
                )
            dataset.slice(start=worker_id, step=worker_info.num_workers)

    def __init__(self, seed: int = 0, start: int = 0, step: int = 1):
        """
        Parameters
        ----------
        seed : int, optional
            random seed used for the dataset sequence generation, by default 0
        start : int, optional
            index of first element of the dataset, useful for a distributed setting, by default 0
        step : int, optional
            number of elements to jump when iterating, useful for a distributed setting, by default 1
        """
        self.start = start
        self.step = step
        self.seed = seed

    def slice(self, start=0, step=1, reset=False):
        """Slices the dataset similar to `dataset[start::step]`, but for a randomized iterable dataset.

        Parameters
        ----------
        start : int, optional
            index of first element of the dataset, useful for a distributed setting, by default 0
        step : int, optional
            number of elements to jump when iterating, useful for a distributed setting, by default 1
        reset : bool, optional
            determines if the slicing should be combined with current slicing of the dataset, or reset to the provided values, by default False
        """
        if reset:
            self.start = start
            self.step = step
        else:
            self.start += self.step * start
            self.step *= step

    def __iter__(self):
        # check if the dataset isn't used properly in dataloader workers
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            if self.step < worker_info.num_workers:
                raise RuntimeError(
                    f"randomized iterable dataset has been instantiated on {worker_info.num_workers} workers with a step of {self.step}, this will make the dataloader build batches with redundant data, make sure you use the `RandomizedIterable.worker_init_fn` function in the dataloader"
                )

        seed_sequence_generator = self._create_seed_sequence_generator(self.seed)
        random_generator = self._get_next_random_generator(
            seed_sequence_generator, n=self.start + 1
        )
        # makes a copy in case `self.step` is changed after the iterator is created and still in use.
        step = self.step

        LOG.info(
            f"start iteration with parameter (seed={self.seed}, start={self.start}, step={step})"
        )

        while True:
            yield self._generate_item(random_generator)
            random_generator = self._get_next_random_generator(
                seed_sequence_generator, n=step
            )

    def _create_seed_sequence_generator(self, seed: int) -> Any:
        """Create initial generator of seeds passed to the `_get_next_random_generator` method.
        Use the numpy `SeedSequence` seed generator by default. Can be overloaded to use another type of seed generator.
        """
        return SeedSequence(entropy=seed)

    def _get_next_random_generator(
        self, seed_sequence_generator: Any, n: int = 1
    ) -> Any:
        """Get the name random generator object produced by `seed_sequence_generator`.
        Use numpy's `default_rng` by default. Can be overloaded to use another type of seed generator.
        """
        seeds = seed_sequence_generator.spawn(n)
        return default_rng(seeds[-1])

    def _generate_item(self, random_generator: Any):
        """Method to be overloaded by child class. Should return one randomly generated item using the provided random generator.
        The provided `random_generator` is unique to that item and will not be used subsequently for other items.
        """
        raise NotImplementedError(
            "This `_generate_item` method should be implemented in the child class",
        )


class ConcatDataset(torch.utils.data.ConcatDataset):
    def __init__(self, *argc, **kwargs) -> None:
        datasets = tuple(argc) + tuple(kwargs.values())
        super().__init__(datasets)
