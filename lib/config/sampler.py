# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

# TODO(Pierre): Add documentation once finalized.


class Sampler:
    def __call__(self, shape, dtype=None, device=None) -> torch.Tensor:
        raise NotImplementedError


class Uniform(Sampler):
    def __init__(self, min_value=0.0, max_value=1.0) -> None:
        super().__init__()

        self._min_value = min_value
        self._max_value = max_value

    def __call__(self, shape, dtype=None, device=None) -> torch.Tensor:
        tensor = torch.empty(size=shape, dtype=dtype, device=device)
        return torch.nn.init.uniform_(tensor, self._min_value, self._max_value)


class Normal(Sampler):
    def __init__(self, mean=0.0, std=1.0) -> None:
        super().__init__()

        self._mean = mean
        self._std = std

    def __call__(self, shape, dtype=None, device=None) -> torch.Tensor:
        tensor = torch.empty(size=shape, dtype=dtype, device=device)
        return torch.nn.init.normal_(tensor, self._mean, self._std)


class TruncatedNormal(Sampler):
    def __init__(self, mean=0.0, std=1.0, min_value=-2.0, max_value=2.0) -> None:
        super().__init__()

        self._mean = mean
        self._std = std
        self._min_value = min_value
        self._max_value = max_value

    def __call__(self, shape, dtype=None, device=None) -> torch.Tensor:
        tensor = torch.empty(size=shape, dtype=dtype, device=device)
        return torch.nn.init.trunc_normal_(
            tensor, self._mean, self._std, self._min_value, self._max_value
        )
