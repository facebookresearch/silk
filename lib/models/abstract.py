# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Iterable, List, Optional

import pytorch_lightning
import silk.config.optimizer as optimizer
import torch
from silk.config.core import ensure_is_instance


class OptimizersHandler:
    r"""Automate the most common pattern of optimizer creation.
    This pattern consists of one optimizer per model.

    Examples
    --------

    ```python
    class MyCustomModel(OptimizersHandler, pl.LightningModule):
        def __init__(self, optimizer_spec, **kwargs):
            OptimizersHandler.__init__(self, optimizer_spec)
            pl.LightningModule.__init__(self, **kwargs)
            ...
    ```

    This will automatically equip `MyCustomModel` with the `configure_optimizers` method required by Pytorch Lightning.
    Notice how `OptimizersHandler` is before `pl.LightningModule` in the list of base classes.
    This is necessary since `pl.LightningModule` checks if the current class has a method called `configure_optimizers`.

    ```python
    class MyCustomModel(OptimizersHandler, pl.LightningModule):
        def __init__(self, optimizer_spec_A, optimizer_spec_B, **kwargs):
            self.submodel_A = ModelA(...)
            self.submodel_B = ModelB(...)

            OptimizersHandler.__init__(self,
                MultiSpec(optimizer_spec_A, optimizer_spec_B),
                self.submodel_A, self.submodel_B
            )
            pl.LightningModule.__init__(self, **kwargs)
            ...
    ```

    In this case, two optimizers will be automatically created and attached to their relative model.

    """

    def __init__(
        self,
        optimizer_specs: Optional[optimizer.Spec] = None,
        *parameters: Iterable[optimizer.Spec.ParametersType],
    ) -> None:
        ensure_is_instance(self, pytorch_lightning.LightningModule)

        self._oh_optimizer_specs = optimizer_specs
        self._oh_parameters = parameters if len(parameters) > 0 else (self,)

    def configure_optimizers(self):
        if self._oh_optimizer_specs is not None:
            return (self._oh_optimizer_specs(*self._oh_parameters),)
        raise RuntimeError("no optimizer spec provided, cannot run in training mode")


class StateDictRedirect:
    def __init__(self, *module: List[torch.nn.Module]) -> None:
        if not len(module) > 0:
            raise RuntimeError(
                "At least one module should be provided to StateDictRedirect. None was provided."
            )

        if len(module) > 1:
            self._module = torch.nn.ModuleList(module)
        else:
            self._module = module[0]

    def state_dict(self, destination=None, prefix: str = "", keep_vars: bool = False):
        return self._module.state_dict(
            destination=destination,
            prefix=prefix,
            keep_vars=keep_vars,
        )

    def load_state_dict(self, state_dict, strict: bool = True):
        self._module.load_state_dict(state_dict, strict)
