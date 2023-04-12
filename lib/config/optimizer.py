# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, Dict, Iterable, Union

import torch

from silk.config.core import find_and_ensure_is_subclass
from silk.optimizers.multiple import MultiOptimizer


class Spec:
    """Optimizer Specification = (Optimizer Type + Optimizer Arguments - Model Parameters)

    This class is mostly used for creating PyTorch optimizers only using model parameters as arguments.
    This makes the configuration of optimizers easier since it decouples the optimizer's parameters from the model's parameters.

    Examples
    --------

    ```python
    # create the optimizer specification
    optim_spec = Spec(torch.optim.Adam, lr=0.001, eps=1e-9, weight_decay=0.01)

    # create the optimizer object and link it to the model's parameters
    optim = optim_spec(model.parameters())
    ```

    """

    ParametersType = Union[torch.nn.Module, Iterable[torch.nn.parameter.Parameter]]

    def __init__(
        self, optimizer_class: Union[str, type], **default_kwargs: Dict[str, Any]
    ) -> None:
        """

        Parameters
        ----------
        optimizer_class : Union[str, type]
            Optimizer class or module path to an optimizer class.

        default_kwargs : Dict[str, Any]
            Default arguments to pass to the optimizer during creation.
        """
        self._optimizer_class = find_and_ensure_is_subclass(
            optimizer_class, torch.optim.Optimizer
        )
        self._default_kwargs = default_kwargs

    def __call__(
        self, parameters: Spec.ParametersType, **override_kwargs
    ) -> torch.optim.Optimizer:
        """Create optimizer object and link it to a model's parameters.

        Parameters
        ----------
        parameters : ParametersType
            Parameters of the model to optimize (usually gotten using the `nn.Module.parameters()` method).

        Returns
        -------
        torch.optim.Optimizer
            Instantiated optimizer linked to specific model parameters.
        """
        kwargs = {**self._default_kwargs, **override_kwargs}
        parameters = (
            parameters.parameters()
            if isinstance(parameters, torch.nn.Module)
            else parameters
        )
        return self._optimizer_class(parameters, **kwargs)


class MultiSpec:
    """MultiSpec is a container of multiple Specs, generating one MultiOptimizer optimizer during training."""

    def __init__(self, *specs) -> None:
        self._specs = specs

    def __call__(
        self,
        *parameters: Iterable[Spec.ParametersType],
    ) -> torch.optim.Optimizer:
        if len(parameters) != len(self._specs):
            raise RuntimeError(
                f"the number of provided parameters ({len(parameters)}) should match the number of optimizer specs ({len(self._specs)})"
            )
        optimizers = (spec(params) for spec, params in zip(self._specs, parameters))
        return MultiOptimizer(*optimizers)
