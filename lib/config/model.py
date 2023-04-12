# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, List, Optional, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.cloud_io import load as pl_load


def load_model_from_checkpoint(  # noqa: C901
    model: Union[pl.LightningModule, torch.nn.Module],
    checkpoint_path: str,
    strict: bool = True,
    device: Optional[str] = None,
    freeze: bool = False,
    eval: bool = False,
    map_name: Union[Dict[str, str], None] = None,
    remove_name: Union[List[str], None] = None,
    state_dict_key: Union[None, str] = "state_dict",
    state_dict_fn: Optional[Callable[[Any], Any]] = None,
):
    checkpoint = pl_load(checkpoint_path, device)

    if isinstance(model, pl.LightningModule):
        model.on_load_checkpoint(checkpoint)

    # get state dictionary
    if state_dict_key is not None:
        state_dict = checkpoint[state_dict_key]
    else:
        state_dict = checkpoint

    # remove names
    if remove_name is not None:
        for name in remove_name:
            del state_dict[name]

    # remap names
    if map_name is not None:
        for src, dst in map_name.items():
            if src not in state_dict:
                continue
            state_dict[dst] = state_dict[src]
            del state_dict[src]

    # apply custom changes to dict
    if state_dict_fn is not None:
        state_dict = state_dict_fn(state_dict)

    model.load_state_dict(state_dict, strict=strict)

    if device is not None:
        model = model.to(device)

    if freeze:
        for param in model.parameters():
            param.requires_grad = False
        eval = True

    if eval:
        model.eval()

    return model

    # line below causes issues with hyperparameters
    # return model.load_from_checkpoint(checkpoint_path)
