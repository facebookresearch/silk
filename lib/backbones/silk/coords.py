# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Iterable, Tuple

import torch


class CoordinateMapping:
    def apply(self, positions):
        raise NotImplementedError

    def reverse(self, positions):
        raise NotImplementedError

    def __add__(self, other):
        return SequentialCoordinateMapping((self, other))

    def __neg__(self):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError


class LinearCoordinateMapping(CoordinateMapping):
    def __init__(self, scale=1.0, bias=0.0) -> None:
        super().__init__()
        self.scale = scale
        self.bias = bias

    def apply(self, positions):
        device = (
            positions.device if isinstance(positions, torch.torch.Tensor) else "cpu"
        )
        return positions * self.scale.to(device) + self.bias.to(device)

    def reverse(self, positions):
        device = (
            positions.device if isinstance(positions, torch.torch.Tensor) else "cpu"
        )
        return (positions - self.bias.to(device)) / self.scale.to(device)

    def __add__(self, other):
        if isinstance(other, LinearCoordinateMapping):
            return LinearCoordinateMapping(
                self.scale * other.scale,
                self.bias * other.scale + other.bias,
            )
        elif isinstance(other, Identity):
            return self
        return CoordinateMapping.__add__(self, other)

    def __neg__(self):
        return LinearCoordinateMapping(
            scale=1.0 / self.scale,
            bias=-self.bias / self.scale,
        )

    def __str__(self):
        return f"x <- {self.scale} x + {self.bias}"


class Conv2dCoordinateMapping(LinearCoordinateMapping):
    @staticmethod
    def from_conv_module(module):
        assert (
            isinstance(module, torch.nn.Conv2d)
            or isinstance(module, torch.nn.MaxPool2d)
            or isinstance(module, torch.nn.ConvTranspose2d)
        )
        if isinstance(module, torch.nn.ConvTranspose2d):
            return -Conv2dCoordinateMapping(
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
            )
        return Conv2dCoordinateMapping(
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
        )

    def __init__(self, kernel_size, stride=1, padding=0, dilation=1) -> None:
        # TODO(Pierre) : Generalize later if necessary
        assert dilation == 1 or dilation == (1, 1)

        kernel_size = torch.tensor(kernel_size)
        stride = torch.tensor(stride)
        padding = torch.tensor(padding)

        output_coord_to_input_coord = LinearCoordinateMapping(
            scale=stride,
            bias=-0.5 * stride - padding + kernel_size / 2,
        )
        input_coord_to_output_coord = -output_coord_to_input_coord

        LinearCoordinateMapping.__init__(
            self,
            input_coord_to_output_coord.scale,
            input_coord_to_output_coord.bias,
        )


class Identity(CoordinateMapping):
    def apply(self, positions):
        return positions

    def reverse(self, positions):
        return positions

    def __add__(self, other):
        return other

    def __radd__(self, other):
        return other

    def __neg__(self):
        return self

    def __str__(self):
        return "x <- x"


class SequentialCoordinateMapping(CoordinateMapping):
    def __init__(self, mappings: Iterable[CoordinateMapping]) -> None:
        super().__init__()
        self.mappings = tuple(mappings)

    def apply(self, positions):
        for mapping in self.mappings:
            positions = mapping.apply(positions)
        return positions

    def reverse(self, positions):
        for mapping in reversed(self.mappings):
            positions = mapping.reverse(positions)
        return positions

    def __radd__(self, other):
        if isinstance(other, SequentialCoordinateMapping):
            return SequentialCoordinateMapping(other.mappings + self.mappings)
        return SequentialCoordinateMapping((other,) + self.mappings)

    def __neg__(self):
        return SequentialCoordinateMapping(reversed(self.mappings))

    def __str__(self):
        return " <- ".join(f"({str(mapping)})" for mapping in reversed(self.mappings))


class CoordinateMappingComposer:
    def __init__(self) -> None:
        self._mappings = {}
        self._arrows = set()

    def _set(self, id_from, id_to, mapping):
        if (id_to, id_from) in self._arrows:
            raise RuntimeError(f"the mapping '{id_to}' <- '{id_from}' already exist")

        m = self._mappings.setdefault(id_to, {})
        m[id_from] = mapping

        m = self._mappings.setdefault(id_from, {})
        m[id_to] = -mapping

        self._arrows.add((id_to, id_from))
        self._arrows.add((id_from, id_to))

    def set(self, id_from, id_to, mapping: CoordinateMapping):
        if not isinstance(mapping, CoordinateMapping):
            raise RuntimeError(
                f"the provided mapping should subclass `CoordinateMapping` to provide coordinate mapping between {id_from} and {id_to}"
            )

        for node_id in self._mappings.get(id_from, {}):
            self._set(node_id, id_to, self._mappings[id_from][node_id] + mapping)

        self._set(id_from, id_to, mapping)

    def get(self, id_from, id_to):
        return self._mappings[id_to][id_from]


class CoordinateMappingProvider:
    def mappings(self) -> Tuple[CoordinateMapping]:
        raise NotImplementedError


def function_coordinate_mapping_provider(mapping=None):
    mapping = Identity() if mapping is None else mapping

    def wrapper(fn):
        class AugFn(CoordinateMappingProvider):
            def __init__(self) -> None:
                super().__init__()

            def __call__(self, *args, **kwds):
                return fn(*args, **kwds)

            def mappings(self) -> Tuple[CoordinateMapping]:
                return mapping

        return AugFn()

    return wrapper


def mapping_from_torch_module(module) -> CoordinateMapping:
    if isinstance(module, CoordinateMappingProvider):
        return module.mappings()
    elif isinstance(module, torch.nn.Conv2d):
        return Conv2dCoordinateMapping.from_conv_module(module)
    elif isinstance(module, torch.nn.ConvTranspose2d):
        return Conv2dCoordinateMapping.from_conv_module(module)
    elif isinstance(module, torch.nn.modules.pooling.MaxPool2d):
        return Conv2dCoordinateMapping.from_conv_module(module)
    elif isinstance(module, torch.nn.Sequential):
        return sum((mapping_from_torch_module(mod) for mod in module), Identity())
    elif (
        isinstance(module, torch.nn.modules.activation.ReLU)
        or isinstance(module, torch.nn.modules.activation.LeakyReLU)
        or isinstance(module, torch.nn.Identity)
        or isinstance(module, torch.nn.BatchNorm2d)
        or isinstance(module, torch.nn.InstanceNorm2d)
        or isinstance(module, torch.nn.GroupNorm)
    ):
        return Identity()
    else:
        raise RuntimeError(
            f"cannot get the coordinate mappings of module of type {type(module)}"
        )
