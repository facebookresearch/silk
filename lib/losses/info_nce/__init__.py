# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Optional

import jax

import silk.losses.info_nce.loss as jax_loss
import torch
from silk.utils.jax import jax2torch

positions_to_unidirectional_correspondence = jax2torch(
    jax.jit(
        jax.vmap(
            jax_loss.positions_to_unidirectional_correspondence,
            in_axes=(0, None, None, None, None),
            out_axes=0,
        ),
        static_argnames=["ordering"],
    ),
    backward_pass=False,
)

keep_mutual_correspondences_only = jax2torch(
    jax.jit(
        jax.vmap(
            jax_loss.keep_mutual_correspondences_only,
            in_axes=(0, 0),
            out_axes=(0, 0),
        )
    ),
    backward_pass=False,
)


def total_loss_reduction(
    desc_0,
    desc_1,
    corr_0,
    corr_1,
    logits_0,
    logits_1,
    ghost_sim=None,
    block_size=None,
):
    batched_total_loss = jax.vmap(
        jax_loss.total_loss,
        in_axes=(0, 0, 0, 0, 0, 0, None, None),
        out_axes=(0, 0, 0, 0),
    )

    loss_0, loss_1, precision, recall = batched_total_loss(
        desc_0,
        desc_1,
        corr_0,
        corr_1,
        logits_0,
        logits_1,
        ghost_sim,
        block_size,
    )

    return loss_0.mean(), loss_1.mean(), precision.mean(), recall.mean()


total_loss = jax2torch(
    jax.jit(
        total_loss_reduction,
        static_argnames=("block_size"),
    )
)


class Loss(torch.nn.Module):
    def __init__(
        self,
        block_size: Optional[int] = None,
        jax_device: str = "cuda:0",
        temperature: float = 0.1,
    ) -> None:
        super().__init__()

        self._block_size = block_size
        self._jax_device = jax_device
        self._temperature_sqrt_inv = 1.0 / math.sqrt(temperature)

    def __call__(
        self,
        desc_0,
        desc_1,
        corr_0,
        corr_1,
        logits_0,
        logits_1,
        ghost_sim=None,
    ):
        desc_0 = desc_0 * self._temperature_sqrt_inv
        desc_1 = desc_1 * self._temperature_sqrt_inv

        return total_loss(
            desc_0,
            desc_1,
            corr_0,
            corr_1,
            logits_0,
            logits_1,
            ghost_sim,
            block_size=self._block_size,
            jax_device=self._jax_device,
        )
