# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import torch
from torch.optim import Optimizer


class MultiOptimizer(Optimizer):
    """Simple optimizer container acting as one single optimizer."""

    def __init__(self, *optimizers: List[Optimizer]):
        self._optimizers = optimizers

    def __getstate__(self):
        return {"_optimizers": self._optimizers}

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __repr__(self):
        return f"{self.__class__.__name__}(*{repr(self._optimizers)})"

    def zero_grad(self):
        for op in self._optimizers:
            op.zero_grad()

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for op in self._optimizers:
            op.step(closure=None)

        return loss

    @property
    def state(self):
        return self.state_dict()

    @property
    def optimizers(self):
        return self._optimizers

    def state_dict(self):
        return {"_optimizers": [op.state_dict() for op in self._optimizers]}

    def load_state_dict(self, state_dict):
        for op, s in zip(self._optimizers, state_dict["_optimizers"]):
            op.load_state_dict(s)
