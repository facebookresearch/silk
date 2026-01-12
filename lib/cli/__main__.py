# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import hydra
import silk.cli
from omegaconf import DictConfig
from silk.config.resolver import init_resolvers

init_resolvers()


@hydra.main(config_path="../../etc", config_name="config")
def main(cfg: DictConfig) -> Any:
    silk.cli.main(cfg)


main()
