# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""command : cache_dataset

The cache dataset command will read data from a loader (with CPU collate transform),
transform those data (can be done on GPU) and save the results as a cached dataset to disk.

This is very useful to :
    * Store expensive transform to speed-up future loading loops.
    * Transform existing datasets and store the result as a new dataset.
    * Simplify the collate function of loaders using already prepared dataset (no work done on the loaders except loading data).
"""

import os

from omegaconf import DictConfig
from silk.config.core import instantiate_and_ensure_is_instance
from silk.datasets.cached import CachedDataset
from silk.profiler import timeit
from silk.transforms.abstract import Transform
from torch.utils.data import DataLoader
from tqdm import tqdm


def _iterable(loader, transform):
    for batch in loader:
        batch = transform(batch)

        for item in batch:
            yield item


def main(config: DictConfig):
    loader = instantiate_and_ensure_is_instance(config.mode.loader, DataLoader)
    transform = instantiate_and_ensure_is_instance(config.mode.transform, Transform)

    # build cached dataset from iterable and save it to disk
    iterable = tqdm(
        _iterable(loader, transform),
        total=config.mode.output.take_n,
    )

    with timeit(
        level="SUCCESS",
        message_template="caching dataset duration : {duration}",
    ):
        dataset = CachedDataset.from_iterable(
            config.mode.output.path,
            iterable,
            config.mode.output.take_n,
        )

    return {
        "n": len(dataset),
        "path": os.path.abspath(config.mode.output.path),
    }
