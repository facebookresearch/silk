# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from silk.backbones.silk.coords import (
    CoordinateMappingProvider,
    mapping_from_torch_module,
)
from silk.backbones.superpoint.magicpoint import MagicPoint, vgg_block
from silk.flow import AutoForward, Flow
from torchvision.transforms.functional import InterpolationMode, resize


class DescriptorHead(torch.nn.Module, CoordinateMappingProvider):
    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 256,
        use_batchnorm: bool = True,
        padding: int = 1,
    ) -> None:
        torch.nn.Module.__init__(self)
        CoordinateMappingProvider.__init__(self)

        assert padding in {0, 1}

        # descriptor head (decoder)
        self._desH1 = vgg_block(
            in_channels,
            out_channels,
            3,
            use_batchnorm=use_batchnorm,
            padding=padding,
        )

        if use_batchnorm:
            # no relu (bc last layer) - option to have batchnorm or not
            self._desH2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 1, padding=0),
                nn.BatchNorm2d(out_channels),
            )
        else:
            # if no batch norm - note that normailzation is calculated later
            self._desH2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 1, padding=0),
            )

    def mappings(self):
        mapping = mapping_from_torch_module(self._desH1)
        mapping = mapping + mapping_from_torch_module(self._desH2)
        return mapping

    def forward(self, x: torch.Tensor):
        x = self._desH1(x)
        x = self._desH2(x)
        return x


class SuperPoint(AutoForward, torch.nn.Module):
    """
    The SuperPoint model, as a subclass of the MagicPoint model.
    """

    def __init__(
        self,
        *,
        use_batchnorm: bool = True,
        descriptor_scale_factor: float = 1.0,
        input_name: str = "images",
        descriptor_head=None,
        descriptor_head_output_name="raw_descriptors",
        default_outputs=("coarse_descriptors", "logits"),
        **magicpoint_kwargs,
    ):
        """Initialize the SuperPoint model.

        Assumes an RGB image with 1 color channel (grayscale image).

        Parameters
        ----------
        use_batchnorm : bool, optional
            Specify if the model uses batch normalization, by default True
        """
        torch.nn.Module.__init__(self)

        self._descriptor_scale_factor = descriptor_scale_factor
        self.magicpoint = MagicPoint(
            use_batchnorm=use_batchnorm,
            input_name=input_name,
            **magicpoint_kwargs,
        )

        AutoForward.__init__(self, self.magicpoint.flow, default_outputs)

        self.magicpoint.backbone.add_head(
            descriptor_head_output_name,
            (
                DescriptorHead(
                    in_channels=128, out_channels=256, use_batchnorm=use_batchnorm
                )
                if descriptor_head is None
                else descriptor_head
            ),
        )

        SuperPoint.add_descriptor_head_post_processing(
            self.flow,
            input_name=input_name,
            descriptor_head_output_name=descriptor_head_output_name,
            prefix="",
            scale_factor=self._descriptor_scale_factor,
        )

    @staticmethod
    def add_descriptor_head_post_processing(
        flow: Flow,
        input_name: str = "images",
        descriptor_head_output_name: str = "raw_descriptors",
        prefix: str = "superpoint.",
        scale_factor: float = 1.0,
    ):
        flow.define_transition(
            f"{prefix}coarse_descriptors",
            partial(SuperPoint.normalize_descriptors, scale_factor=scale_factor),
            descriptor_head_output_name,
        )
        flow.define_transition(f"{prefix}image_size", SuperPoint.image_size, input_name)
        flow.define_transition(
            f"{prefix}sparse_descriptors",
            partial(SuperPoint.sparsify_descriptors, scale_factor=scale_factor),
            descriptor_head_output_name,
            f"{prefix}positions",
            f"{prefix}image_size",
        )
        flow.define_transition(
            f"{prefix}upsampled_descriptors",
            partial(SuperPoint.upsample_descriptors, scale_factor=scale_factor),
            descriptor_head_output_name,
            f"{prefix}image_size",
        )

    @staticmethod
    def image_size(images):
        return images.shape[-2:]

    @staticmethod
    def normalize_descriptors(raw_descriptors, scale_factor=1.0, normalize=True):
        if normalize:
            return scale_factor * F.normalize(
                raw_descriptors,
                p=2,
                dim=1,
            )  # L2 normalization
        return scale_factor * raw_descriptors

    @staticmethod
    def sparsify_descriptors(
        raw_descriptors,
        positions,
        image_size,
        scale_factor: float = 1.0,
    ):
        image_size = torch.tensor(
            image_size,
            dtype=torch.float,
            device=raw_descriptors.device,
        )
        sparse_descriptors = []

        for i, pos in enumerate(positions):
            pos = pos[:, :2]
            n = pos.shape[0]

            # handle edge case when no points has been detected
            if n == 0:
                desc = raw_descriptors[i]
                fdim = desc.shape[0]
                sparse_descriptors.append(
                    torch.zeros(
                        (n, fdim),
                        dtype=desc.dtype,
                        device=desc.device,
                    )
                )
                continue

            # revert pixel centering for grad sample
            pos = pos - 0.5

            # normalize to [-1. +1] & prepare for grid sample
            pos = 2.0 * (pos / (image_size - 1)) - 1.0
            pos = pos[:, [1, 0]]
            pos = pos[None, None, ...]

            # process descriptor output by interpolating into descriptor map using 2D point locations\
            # note that grid_sample takes coordinates in x, y order (col, then row)
            descriptors = raw_descriptors[i][None, ...]
            descriptors = F.grid_sample(
                descriptors,
                pos,
                mode="bilinear",
                align_corners=False,
            )
            descriptors = descriptors.view(-1, n).T

            # L2 normalize the descriptors
            descriptors = SuperPoint.normalize_descriptors(descriptors, scale_factor)

            sparse_descriptors.append(descriptors)
        return sparse_descriptors

    @staticmethod
    def upsample_descriptors(raw_descriptors, image_size, scale_factor: float = 1.0):
        upsampled_descriptors = resize(
            raw_descriptors,
            image_size,
            interpolation=InterpolationMode.BILINEAR,
        )
        return SuperPoint.normalize_descriptors(upsampled_descriptors, scale_factor)
