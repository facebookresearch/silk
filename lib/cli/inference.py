# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from omegaconf import DictConfig, OmegaConf
from silk.logger import LOG


def main(config: DictConfig):
    LOG.error("inference [TODO(Pierre)]")
    return {"config": OmegaConf.to_object(config)}
