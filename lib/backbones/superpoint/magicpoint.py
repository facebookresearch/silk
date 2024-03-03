# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
The MagicPoint model of SuperPoint to be trained
on synthetic data. Based off of the official
PyTorch implementation from the MagicLeap paper.
# Checked Parity
## With Paper : https://arxiv.org/pdf/1712.07629.pdf
### Optimizer (page 6)
* [**done**] Type = Adam
* [**done**] Learning Rate = 0.001
* [**done**] β = (0.9, 0.999)
### Training (page 6)
* [**done**] Batch Size = 32
* [**diff**] Steps = 200,000 (ours : early stopping)
### Metrics (page 4)
* [**done**] mAP = 0.971 (ours : 0.999)
"""

from functools import partial

import torch
import torch.nn as nn

from silk.backbones.abstract.shared_backbone_multiple_heads import (
    SharedBackboneMultipleHeads,
)
from silk.backbones.silk.coords import (
    CoordinateMappingProvider,
    mapping_from_torch_module,
)
from silk.backbones.superpoint.utils import (
    depth_to_space,
    logits_to_prob,
    prob_map_to_points_map,
    prob_map_to_positions_with_prob,
)
from silk.backbones.superpoint.vgg import VGG, vgg_block
from silk.flow import AutoForward, Flow

Backbone = partial(
    VGG,
    num_channels=1,
    use_batchnorm=True,
    use_max_pooling=True,
)


class DetectorHead(torch.nn.Module, CoordinateMappingProvider):
    def __init__(
        self,
        in_channels: int = 128,
        lat_channels: int = 256,
        out_channels: int = 1,
        use_batchnorm: bool = True,
        padding: int = 1,
        detach: bool = False,
    ) -> None:
        torch.nn.Module.__init__(self)
        CoordinateMappingProvider.__init__(self)

        assert padding in {0, 1}

        self._detach = detach

        self._detH1 = vgg_block(
            in_channels,
            lat_channels,
            3,
            use_batchnorm=use_batchnorm,
            padding=padding,
        )

        if use_batchnorm:
            # no relu (bc last layer) - option to have batchnorm or not
            self._detH2 = nn.Sequential(
                nn.Conv2d(lat_channels, out_channels, 1, padding=0),
                nn.BatchNorm2d(out_channels),
            )
        else:
            # if no batch norm
            self._detH2 = nn.Sequential(
                nn.Conv2d(lat_channels, out_channels, 1, padding=0),
            )

    def mappings(self):
        mapping = mapping_from_torch_module(self._detH1)
        mapping = mapping + mapping_from_torch_module(self._detH2)
        return mapping

    def forward(self, x: torch.Tensor):
        if self._detach:
            x = x.detach()

        x = self._detH1(x)
        x = self._detH2(x)
        return x


class MagicPoint(AutoForward, torch.nn.Module):
    def __init__(
        self,
        *,
        use_batchnorm: bool = True,
        num_channels: int = 1,
        cell_size: int = 8,
        detection_threshold=0.015,
        detection_top_k=None,
        nms_dist=4,
        border_dist=4,
        use_max_pooling: bool = True,
        input_name="images",
        backbone=None,
        backbone_output_name: str = "features",
        detector_head=None,
        detector_head_output_name: str = "logits",
        default_outputs=None,
    ):
        torch.nn.Module.__init__(self)

        # architecture parameters
        self._num_channels = num_channels
        self._cell_size = cell_size  # depends on VGG's downsampling

        # detection parameters
        self._detection_threshold = detection_threshold
        self._detection_top_k = detection_top_k
        self._nms_dist = nms_dist
        self._border_dist = border_dist

        # add backbone
        self.backbone = SharedBackboneMultipleHeads(
            backbone=(
                Backbone(
                    num_channels=num_channels,
                    use_batchnorm=use_batchnorm,
                    use_max_pooling=use_max_pooling,
                )
                if backbone is None
                else backbone
            ),
            input_name=input_name,
            backbone_output_name=backbone_output_name,
        )

        if use_max_pooling:
            out_channels = cell_size * cell_size + 1
        else:
            out_channels = 1

        # add detector head
        self.backbone.add_head(
            detector_head_output_name,
            (
                DetectorHead(
                    in_channels=128,
                    lat_channels=256,
                    out_channels=out_channels,
                    use_batchnorm=use_batchnorm,
                )
                if detector_head is None
                else detector_head
            ),
        )

        # add the forward function
        default_outputs = (
            (backbone_output_name, detector_head_output_name)
            if default_outputs is None
            else default_outputs
        )
        AutoForward.__init__(self, self.backbone.flow, default_outputs)

        # add detector head post-processing
        MagicPoint.add_detector_head_post_processing(
            self.flow,
            detector_head_output_name=detector_head_output_name,
            prefix="",
            cell_size=self._cell_size,
            detection_threshold=self._detection_threshold,
            detection_top_k=self._detection_top_k,
            nms_dist=self._nms_dist,
            border_dist=self._border_dist,
        )

    @staticmethod
    def add_detector_head_post_processing(
        flow: Flow,
        detector_head_output_name: str = "logits",
        prefix: str = "magicpoint.",
        cell_size: int = 8,
        detection_threshold=0.015,
        detection_top_k=None,
        nms_dist=4,
        border_dist=4,
    ):
        flow.define_transition(
            f"{prefix}probability",
            logits_to_prob,
            detector_head_output_name,
        )
        flow.define_transition(
            f"{prefix}score",
            partial(depth_to_space, cell_size=cell_size),
            f"{prefix}probability",
        )
        flow.define_transition(
            f"{prefix}nms",
            partial(
                prob_map_to_points_map,
                prob_thresh=detection_threshold,
                nms_dist=nms_dist,
                border_dist=border_dist,
                top_k=detection_top_k,
            ),
            f"{prefix}score",
        )
        flow.define_transition(
            f"{prefix}positions",
            prob_map_to_positions_with_prob,
            f"{prefix}nms",
        )
