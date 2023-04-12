# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# source : https://github.com/milesial/Pytorch-UNet/tree/master/unet

import torch
import torch.nn as nn
import torch.nn.functional as F

from silk.backbones.silk.coords import (
    CoordinateMappingProvider,
    mapping_from_torch_module,
)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module, CoordinateMappingProvider):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def mappings(self):
        return mapping_from_torch_module(self.conv)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class MultiConv(nn.Module, CoordinateMappingProvider):
    def __init__(
        self,
        in_channels,
        out_channels,
        length=1,
        mid_channels=None,
        padding=1,
        kernel=3,
    ):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels

        self.channels = [in_channels] + [mid_channels] * (length - 1) + [out_channels]
        self.multi_conv = nn.Sequential(
            *sum(
                [
                    [
                        nn.Conv2d(
                            self.channels[i],
                            self.channels[i + 1],
                            kernel_size=kernel,
                            padding=padding,
                            bias=False,
                        ),
                        nn.BatchNorm2d(self.channels[i + 1]),
                        nn.ReLU(inplace=True),
                    ]
                    for i in range(length)
                ],
                [],
            )
        )

    def mappings(self):
        return mapping_from_torch_module(self.multi_conv)

    def forward(self, x):
        return self.multi_conv(x)


class ParametricDown(nn.Module, CoordinateMappingProvider):
    def __init__(
        self,
        in_channels,
        out_channels,
        length=1,
        use_max_pooling=True,
        padding=1,
        kernel=3,
    ):
        super().__init__()

        downscale_layer = (
            nn.MaxPool2d(2)
            if use_max_pooling
            else nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2)
        )

        self.maxpool_conv = nn.Sequential(
            downscale_layer,
            MultiConv(
                in_channels,
                out_channels,
                length,
                padding=padding,
                kernel=kernel,
            ),
        )

    def mappings(self):
        return mapping_from_torch_module(self.maxpool_conv)

    def forward(self, x):
        return self.maxpool_conv(x)


class ParametricUp(nn.Module, CoordinateMappingProvider):
    def __init__(
        self,
        in_channels,
        out_channels,
        hor_channels=None,
        length=1,
        bilinear=True,
        padding=1,
        kernel=3,
        hor_mapping=None,
        below_mapping=None,
    ):
        super().__init__()

        assert padding in {0, 1}
        self.padding = padding
        self.hor_channels = in_channels // 2 if hor_channels is None else hor_channels

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Sequential(
                nn.Conv2d(in_channels, self.hor_channels, kernel_size=1),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            )
        else:
            self.up = nn.ConvTranspose2d(
                in_channels,
                self.hor_channels,
                kernel_size=2,
                stride=2,
            )

        self.conv = MultiConv(
            2 * self.hor_channels,
            out_channels,
            length,
            padding=padding,
            kernel=kernel,
        )

        self._calculate_pad(hor_mapping, below_mapping)

    def _calculate_pad(self, hor_mapping=None, below_mapping=None):
        if (hor_mapping is None) or (below_mapping is None):
            self.top_pad = None
            self.left_pad = None
            return

        up_mapping = below_mapping + self.mappings()
        if not (hor_mapping.scale == up_mapping.scale).all():
            raise RuntimeError(
                f"only layer of same scale can be combine in upsampling layer : {hor_mapping.scale} != {up_mapping.scale}"
            )

        top_pad = (hor_mapping.bias[0] - up_mapping.bias[0]).item()
        left_pad = (hor_mapping.bias[1] - up_mapping.bias[1]).item()

        assert top_pad >= 0
        assert left_pad >= 0
        assert float(int(top_pad)) == top_pad
        assert float(int(left_pad)) == left_pad

        self.top_pad = int(top_pad)
        self.left_pad = int(left_pad)

    def mappings(self):
        return mapping_from_torch_module(self.up) + mapping_from_torch_module(self.conv)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        dy = x2.shape[2] - x1.shape[2]
        dx = x2.shape[3] - x1.shape[3]

        if self.top_pad is None:
            pad_y_top = dy // 2
        else:
            pad_y_top = self.top_pad
        pad_y_bottom = dy - pad_y_top

        if self.left_pad is None:
            pad_x_left = dx // 2
        else:
            pad_x_left = self.left_pad
        pad_x_right = dx - pad_x_left

        if self.padding:
            x1 = F.pad(x1, (pad_x_left, pad_x_right, pad_y_top, pad_y_bottom))
            # if you have padding issues, see
            # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
            # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        else:
            x2 = x2[..., pad_y_top:-pad_y_bottom, pad_x_left:-pad_x_right]

        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


class ParametricUNet(nn.Module, CoordinateMappingProvider):
    def __init__(
        self,
        n_channels,
        n_classes,
        bilinear=False,
        input_feature_channels=64,
        n_scales=4,
        length=1,
        use_max_pooling=True,
        padding=1,
        kernel=3,
        up_channels=None,
        down_channels=None,
    ):
        nn.Module.__init__(self)
        CoordinateMappingProvider.__init__(self)

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.input_feature_channels = input_feature_channels
        self.down_channels = (
            [input_feature_channels * (2**i) for i in reversed(range(n_scales + 1))]
            if down_channels is None
            else [input_feature_channels] + down_channels
        )
        self.up_channels = (
            [input_feature_channels * (2**i) for i in range(n_scales + 1)]
            if up_channels is None
            else [self.down_channels[-1]] + up_channels
        )

        assert len(self.up_channels) == n_scales + 1
        assert len(self.down_channels) == n_scales + 1

        self.padding = padding
        self.length = length
        self.n_scales = n_scales
        self.kernel = kernel

        self.up_mappings = [None] * (n_scales + 1)

        self.inc = MultiConv(
            n_channels,
            input_feature_channels,
            length,
            padding=padding,
            kernel=self.kernel,
        )

        self.down_mappings = [None] * (n_scales + 1)
        self.down_mappings[0] = mapping_from_torch_module(self.inc)

        down = []
        for i in range(n_scales):
            layer = ParametricDown(
                self.down_channels[i],
                self.down_channels[i + 1],
                length,
                use_max_pooling=use_max_pooling,
                padding=padding,
                kernel=self.kernel,
            )

            down.append(layer)
            self.down_mappings[i + 1] = self.down_mappings[
                i
            ] + mapping_from_torch_module(layer)

        up = []
        self.up_mappings[0] = self.down_mappings[-1]
        for i in range(n_scales):
            layer = ParametricUp(
                self.up_channels[i],
                self.up_channels[i + 1],
                self.down_channels[n_scales - 1 - i],
                length,
                bilinear=bilinear,
                padding=padding,
                kernel=self.kernel,
                hor_mapping=self.down_mappings[n_scales - i - 1],
                below_mapping=self.up_mappings[i],
            )

            up.append(layer)

            self.up_mappings[i + 1] = self.up_mappings[i] + mapping_from_torch_module(
                layer
            )

        self.down = nn.ModuleList(down)
        self.up = nn.ModuleList(up)

        self.outc = OutConv(self.up_channels[-1], n_classes)

    def total_pad(self):
        pad = (1 - self.padding) * (self.kernel // 2)
        return (
            sum(2 * pad * (2**i) * self.length for i in range(self.n_scales))
            + pad * (2**self.n_scales) * self.length
        )

    def mappings(self):
        mapping = mapping_from_torch_module(self.inc)
        for down in self.down:
            mapping = mapping + mapping_from_torch_module(down)
        for up in self.up:
            mapping = mapping + mapping_from_torch_module(up)
        mapping = mapping + mapping_from_torch_module(self.outc)
        return mapping

    def forward(self, x):
        layers = [self.inc(x)]

        # downscale
        for down in self.down:
            layers.append(down(layers[-1]))

        # upscale
        x = layers.pop(-1)
        for up in self.up:
            x = up(x, layers.pop(-1))

        logits = self.outc(x)
        return logits
