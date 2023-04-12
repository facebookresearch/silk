# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import pytorch_lightning as pl
import torch

from silk.config.optimizer import MultiSpec, Spec
from silk.models.abstract import OptimizersHandler
from silk.optimizers.multiple import MultiOptimizer


def _layer():
    return torch.nn.Linear(2, 4)


class _MockModelA(OptimizersHandler, pl.LightningModule):
    def __init__(self, optimizer_spec):
        OptimizersHandler.__init__(self, optimizer_spec)
        pl.LightningModule.__init__(self)

        self.submodel = _layer()


class _MockModelB(OptimizersHandler, pl.LightningModule):
    def __init__(self, optimizer_spec):
        pl.LightningModule.__init__(self)

        self.submodel_A = _layer()
        self.submodel_B = _layer()

        OptimizersHandler.__init__(
            self,
            optimizer_spec,
            self.submodel_A,
            self.submodel_B,
        )


class _UnitTests(unittest.TestCase):
    def _num_params(self, model):
        return sum(param.numel() for param in model.parameters())

    def test_optimizer_handler(self):
        optim_A = torch.optim.Adam
        optim_B = torch.optim.SGD

        optim_spec_A = Spec(optim_A, lr=0.1)
        optim_spec_B = Spec(optim_B, lr=0.1)
        optim_spec_A_and_B = MultiSpec(optim_spec_A, optim_spec_B)

        model_A = _MockModelA(optim_spec_A)
        model_B = _MockModelB(optim_spec_A_and_B)

        optimizers_A = model_A.configure_optimizers()
        optimizers_B = model_B.configure_optimizers()

        # check if the number of optimizers is consistent
        self.assertEqual(len(optimizers_A), 1)
        self.assertEqual(len(optimizers_B), 1)

        # check if the created optimizer classes are consistent
        self.assertIsInstance(optimizers_A[0], optim_A)
        self.assertIsInstance(optimizers_B[0], MultiOptimizer)
        self.assertIsInstance(optimizers_B[0].optimizers[0], optim_A)
        self.assertIsInstance(optimizers_B[0].optimizers[1], optim_B)

        # check if `OptimizersHandler` doesn't add references to layers
        layer = _layer()
        self.assertEqual(1 * self._num_params(layer), self._num_params(model_A))
        self.assertEqual(2 * self._num_params(layer), self._num_params(model_B))


def main():
    unittest.main()


if __name__ == "__main__":
    unittest.main()
