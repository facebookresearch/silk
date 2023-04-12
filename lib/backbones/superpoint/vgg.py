# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from typing import Callable, Iterable, List, Union

import torch

from silk.backbones.silk.coords import (
    CoordinateMappingProvider,
    Identity,
    mapping_from_torch_module,
)


def vgg_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    use_batchnorm: bool = True,
    non_linearity: str = "relu",
    padding: int = 1,
) -> torch.nn.Module:
    """
    The VGG block for the model.
    This block contains a 2D convolution, a ReLU activation, and a
    2D batch normalization layer.
    Args:
        in_channels (int): the number of input channels to the Conv2d layer
        out_channels (int): the number of output channels
        kernel_size (int): the size of the kernel for the Conv2d layer
        use_batchnorm (bool): whether or not to include a batchnorm layer.
            Default is true (batchnorm will be used).
    Returns:
        vgg_blk (nn.Sequential): the vgg block layer of the model
    """

    if non_linearity == "relu":
        non_linearity = torch.nn.ReLU(inplace=True)
    else:
        raise NotImplementedError

    # the paper states that batchnorm is used after each convolution layer
    if use_batchnorm:
        vgg_blk = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            non_linearity,
            torch.nn.BatchNorm2d(out_channels),
        )
    # however, the official implementation does not include batchnorm
    else:
        vgg_blk = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            non_linearity,
        )

    return vgg_blk


class VGG(torch.nn.Module, CoordinateMappingProvider):
    """
    The VGG backbone.
    """

    def __init__(
        self,
        num_channels: int = 1,
        use_batchnorm: bool = False,
        use_max_pooling: bool = True,
        padding: int = 1,
    ):
        """
        Initialize the VGG backbone model.
        Can take an input image of any number of channels (e.g. grayscale, RGB).
        """
        torch.nn.Module.__init__(self)
        CoordinateMappingProvider.__init__(self)

        assert padding in {0, 1}

        self.padding = padding
        self.use_max_pooling = use_max_pooling

        if use_max_pooling:
            self.mp = torch.nn.MaxPool2d(2, stride=2)
        else:
            self.mp = torch.nn.Identity()

        # convolution layers (encoder)
        self.l1 = torch.nn.Sequential(
            vgg_block(
                num_channels,
                64,
                3,
                use_batchnorm=use_batchnorm,
                padding=padding,
            ),
            vgg_block(
                64,
                64,
                3,
                use_batchnorm=use_batchnorm,
                padding=padding,
            ),
        )

        self.l2 = torch.nn.Sequential(
            vgg_block(
                64,
                64,
                3,
                use_batchnorm=use_batchnorm,
                padding=padding,
            ),
            vgg_block(
                64,
                64,
                3,
                use_batchnorm=use_batchnorm,
                padding=padding,
            ),
        )

        self.l3 = torch.nn.Sequential(
            vgg_block(
                64,
                128,
                3,
                use_batchnorm=use_batchnorm,
                padding=padding,
            ),
            vgg_block(
                128,
                128,
                3,
                use_batchnorm=use_batchnorm,
                padding=padding,
            ),
        )

        self.l4 = torch.nn.Sequential(
            vgg_block(
                128,
                128,
                3,
                use_batchnorm=use_batchnorm,
                padding=padding,
            ),
            vgg_block(
                128,
                128,
                3,
                use_batchnorm=use_batchnorm,
                padding=padding,
            ),
        )

    def mappings(self):
        mapping = Identity()
        mapping = mapping + mapping_from_torch_module(self.l1)
        mapping = mapping + mapping_from_torch_module(self.mp)
        mapping = mapping + mapping_from_torch_module(self.l2)
        mapping = mapping + mapping_from_torch_module(self.mp)
        mapping = mapping + mapping_from_torch_module(self.l3)
        mapping = mapping + mapping_from_torch_module(self.mp)
        mapping = mapping + mapping_from_torch_module(self.l4)

        return mapping

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Goes through the layers of the VGG model as the forward pass.
        Computes the output.
        Args:
            images (tensor): image pytorch tensor with
                shape N x num_channels x H x W
        Returns:
            output (tensor): the output point pytorch tensor with
            shape N x cell_size^2+1 x H/8 x W/8.
        """
        o1 = self.l1(images)
        o1 = self.mp(o1)

        o2 = self.l2(o1)
        o2 = self.mp(o2)

        o3 = self.l3(o2)
        o3 = self.mp(o3)

        # features
        o4 = self.l4(o3)

        return o4


def parametric_vgg_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    normalization_fn,
    non_linearity: str = "relu",
    padding: int = 1,
) -> torch.nn.Module:
    if non_linearity == "relu":
        non_linearity = torch.nn.ReLU(inplace=True)
    else:
        raise NotImplementedError

    vgg_blk = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
        non_linearity,
        normalization_fn,
    )

    return vgg_blk


class ParametricVGG(torch.nn.Module, CoordinateMappingProvider):
    DEFAULT_NORMALIZATION_FN = torch.nn.Identity()

    def __init__(
        self,
        input_num_channels: int = 1,
        normalization_fn: Union[Callable, List[Callable]] = DEFAULT_NORMALIZATION_FN,
        use_max_pooling: bool = True,
        padding: int = 1,
        channels: List[int] = (64, 64, 128, 128),
    ):
        CoordinateMappingProvider.__init__(self)
        torch.nn.Module.__init__(self)

        assert padding in {0, 1}
        assert len(channels) >= 1

        self.padding = padding
        self.use_max_pooling = use_max_pooling
        if isinstance(normalization_fn, Iterable):
            normalization_fn = tuple(normalization_fn)
            assert len(normalization_fn) == len(channels)
        else:
            normalization_fn = tuple([normalization_fn] * len(channels))

        if use_max_pooling:
            self.mp = torch.nn.MaxPool2d(2, stride=2)
        else:
            self.mp = torch.nn.Identity()

        self.layers = []
        self.channels = (input_num_channels,) + channels
        for i in range(1, len(self.channels)):
            layer = torch.nn.Sequential(
                parametric_vgg_block(
                    self.channels[i - 1],
                    self.channels[i],
                    3,
                    deepcopy(normalization_fn[i - 1]),
                    "relu",
                    padding,
                ),
                parametric_vgg_block(
                    self.channels[i],
                    self.channels[i],
                    3,
                    deepcopy(normalization_fn[i - 1]),
                    "relu",
                    padding,
                ),
            )
            self.layers.append(layer)
        self.layers = torch.nn.ModuleList(self.layers)

    def mappings(self):
        mapping = Identity()
        for layer in self.layers[:-1]:
            mapping = mapping + mapping_from_torch_module(layer)
            mapping = mapping + mapping_from_torch_module(self.mp)
        mapping = mapping + mapping_from_torch_module(self.layers[-1])

        return mapping

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = images
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.mp(x)
        x = self.layers[-1](x)
        return x
