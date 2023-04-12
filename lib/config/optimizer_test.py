# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import silk.config.optimizer as optimizer

import torch


class _MockModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(2, 4)


class _UnitTests(unittest.TestCase):
    def _check_optim_spec(self, model, optim_class, **kwargs):
        optim_spec = optimizer.Spec(optim_class, **kwargs)
        optim_0 = optim_spec(model.parameters())
        optim_1 = optim_spec(model)

        # make sure the provided optimizer was used
        self.assertIsInstance(optim_0, optim_class)

        # make sure a new optimizer was created
        self.assertIsNot(optim_0, optim_1)

        # make sure it is equal to a manually created optimizer
        optim_manual = optim_class(model.parameters(), **kwargs)
        self.assertEqual(repr(optim_0), repr(optim_manual))
        self.assertEqual(repr(optim_1), repr(optim_manual))

    def test_optimizer_spec(self):
        model = _MockModel()

        # check two commonly used optimizers
        self._check_optim_spec(
            model, torch.optim.Adam, lr=0.001, eps=1e-9, weight_decay=0.01
        )
        self._check_optim_spec(model, torch.optim.SGD, lr=0.001, momentum=0.1)

        # check when the class doesn't subclass `torch.optim.Optimizer`
        with self.assertRaises(RuntimeError):
            self._check_optim_spec(model, dict)


if __name__ == "__main__":
    unittest.main()
