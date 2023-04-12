# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from omegaconf import DictConfig

from silk.config.core import instantiate_and_ensure_is_instance
from silk.engine.loop import Benchmarker
from torch.utils.data import DataLoader


def main(config: DictConfig):
    model = instantiate_and_ensure_is_instance(config.mode.model, torch.nn.Module)

    loader = instantiate_and_ensure_is_instance(config.mode.loader, DataLoader)

    benchmarker = instantiate_and_ensure_is_instance(
        config.mode.benchmarker, Benchmarker
    )

    benchmarker.test(model, loader)

    return {
        "metrics": benchmarker.compute_metrics(),
    }
