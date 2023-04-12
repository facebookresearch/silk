# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import importlib

from omegaconf import DictConfig

from silk.config.constants import PACKAGE_NAME
from silk.logger import LOG


def _check_dispatch(procedure, cfg):
    module_name = f"{PACKAGE_NAME}.cli.check.{procedure}"
    module = importlib.import_module(module_name)
    LOG.success(f"module `{module_name}` successfully imported")
    return module.main(cfg)


def main(config: DictConfig):
    return _check_dispatch(config.mode.procedure, config)
