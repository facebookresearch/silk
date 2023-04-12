# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys

from tqdm import tqdm

local_silk_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(local_silk_dir)

import silk
from silk.datasets.cached import CachedDataset

# patch module name for proper pickle file loading
sys.modules["pixenv"] = silk


def cache(dataset, output, transform, *args):
    # load input dataset
    dataset = CachedDataset(dataset)

    # save transformed dataset here
    CachedDataset.from_iterable(
        filepath=output,
        iterable=tqdm(transform(elem, *args) for elem in dataset),
    )

    print(f"{os.getcwd()}/{output}")
