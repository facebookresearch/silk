# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

r"""# silk - CLI

Highly configurable CLI tool for training, running inference and benchmarking.

## How to run a mode ?

A mode is a CLI command + parameters present in the configuration file `etc/mode/*.yaml`.
Running a mode without overriding any of its parameters can be done like this : `./bin/silk-cli mode=<mode-name>`

Examples
--------

```python
# start training the configured magicpoint model
./bin/silk-cli mode=train-magicpoint

# start benchmarking the configured magicpoint model
./bin/silk-cli mode=benchmark-magicpoint
```

Any parameter can be overridden since we are powered by [hydra](https://hydra.cc).

Examples
--------

```python
# start training the configured magicpoint model on GPU 0 and 1 and disable output print
./bin/silk-cli mode=train-magicpoint 'mode.trainer.gpus=[0,1]' formatter=none
```

"""

import importlib
import os
from typing import Any

import hydra
import hydra.utils
import silk.logger as logger
from omegaconf import DictConfig
from silk.config import PACKAGE_NAME
from silk.config.paths import ROOT as ROOT_PATH
from silk.logger import LOG

COMMANDS = {
    "training",
    "inference",
    "benchmark",
    "check",
    "cache_dataset",
    "visualization",
    "hpatches_tests",
    "viewreid_tests",
    "image_pair_visualization",
}


def _main_dispatch(command, cfg):
    module_name = f"{PACKAGE_NAME}.cli.{command}"
    module = importlib.import_module(module_name)
    LOG.success(f"module `{module_name}` successfully imported")
    return module.main(cfg)


def _init_logger(cfg):
    for name in cfg.handlers:
        kwargs = hydra.utils.instantiate(cfg.handlers[name])
        logger.set_handler_options(name, **kwargs)
    logger.enable_handlers_only(*cfg.handlers.keys())


@LOG.catch(reraise=True)
def _main(cfg: DictConfig, working_dir: str) -> Any:
    # check if command is correct
    if cfg.mode.command not in COMMANDS:
        raise RuntimeError(
            f'mode.command="{cfg.mode.command}" is not available (should be one of these {COMMANDS})'
        )

    LOG.info(f"run CLI in mode.command={cfg.mode.command}")

    formatter = hydra.utils.instantiate(cfg.formatter)

    LOG.success(f"formatter `{cfg.formatter.name}` successfully instantiated")

    # run main command
    output = _main_dispatch(cfg.mode.command, cfg)
    LOG.success("main dispatch successfully executed")

    if output is not None:
        output = formatter(output)
        LOG.success("formatter successfully converted output")

        print(output)

    LOG.success(f"ran successfully in working directory : {working_dir}")

    return output


def main(cfg: DictConfig) -> Any:
    working_dir = os.path.relpath(os.getcwd(), ROOT_PATH)

    _init_logger(cfg.logger)

    try:
        return _main(cfg, working_dir)
    except BaseException:
        LOG.error(f"run failed, `*.log` file might be found in : {working_dir}")
        if cfg.debug:
            raise
        exit(1)
