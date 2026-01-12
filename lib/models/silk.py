# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import Any, Dict, Optional, Union

import pytorch_lightning as pl
import torch
from silk.backbones.loftr.positional_encoding import PositionEncodingSine
from silk.backbones.silk.silk import SiLKBase as BackboneBase
from silk.config.core import ensure_is_instance
from silk.config.optimizer import Spec
from silk.cv.homography import HomographicSampler
from silk.flow import AutoForward, Flow
from silk.losses.info_nce import (
    keep_mutual_correspondences_only,
    positions_to_unidirectional_correspondence,
)
from silk.matching.mnn import (
    compute_dist,
    double_softmax_distance,
    match_descriptors,
    mutual_nearest_neighbor,
)
from silk.models.abstract import OptimizersHandler, StateDictRedirect
from silk.transforms.abstract import MixedModuleDict, NamedContext, Transform
from silk.transforms.cv.homography import RandomHomographicSampler
from silk.transforms.tensor import NormalizeRange
from torchvision.transforms import Grayscale

_DEBUG_MODE_ENABLED = True


def matcher(
    postprocessing="none",
    threshold=1.0,
    temperature=0.1,
    return_distances=False,
):
    if postprocessing == "none" or postprocessing == "mnn":
        return partial(mutual_nearest_neighbor, return_distances=return_distances)
    elif postprocessing == "ratio-test":
        return partial(
            mutual_nearest_neighbor,
            match_fn=partial(match_descriptors, max_ratio=threshold),
            distance_fn=partial(compute_dist, dist_type="cosine"),
            return_distances=return_distances,
        )
    elif postprocessing == "double-softmax":
        return partial(
            mutual_nearest_neighbor,
            match_fn=partial(match_descriptors, max_distance=threshold),
            distance_fn=partial(double_softmax_distance, temperature=temperature),
            return_distances=return_distances,
        )

    raise RuntimeError(f"postprocessing {postprocessing} is invalid")


class SiLKBase(
    OptimizersHandler,
    AutoForward,
    StateDictRedirect,
    pl.LightningModule,
):
    def __init__(
        self,
        model,
        loss,
        optimizer_spec: Optional[Spec] = None,
        image_aug_transform: Optional[Transform] = None,
        contextualizer: Optional[torch.nn.Module] = None,
        ghost_similarity: Optional[float] = None,
        learn_ghost_similarity: bool = False,
        feature_downsampling_mode: str = "scale",
        greyscale_input: bool = True,
        **kwargs,
    ):
        pl.LightningModule.__init__(self, **kwargs)
        OptimizersHandler.__init__(self, optimizer_spec)  # see below

        assert isinstance(model, BackboneBase)

        self._feature_downsampling_mode = feature_downsampling_mode
        self._greyscale_input = greyscale_input

        if ghost_similarity is not None:
            self._ghost_sim = torch.nn.parameter.Parameter(
                torch.tensor(ghost_similarity),
                requires_grad=learn_ghost_similarity,
            )
        else:
            self._ghost_sim = None

        ghost_sim_module = torch.nn.Module()
        ghost_sim_module.ghost_sim = self._ghost_sim

        state = MixedModuleDict(
            {
                "model": model,
                "contextualizer": contextualizer,
                "ghost_similarity": ghost_sim_module,
            }
        )

        StateDictRedirect.__init__(self, state)
        AutoForward.__init__(self, Flow("batch", "use_image_aug"), "loss")

        self._loss = loss
        self._model = model
        self._contextualizer = contextualizer
        if contextualizer:
            self._pe = PositionEncodingSine(256, max_shape=(512, 512))
        self._image_aug_transform = image_aug_transform

    @property
    def coordinate_mapping_composer(self):
        return self._model.coordinate_mapping_composer

    def _grayify(self, images):
        if self._greyscale_input:
            images = Grayscale(num_output_channels=1)(images)
            return NormalizeRange(0.0, 255.0, 0.0, 1.0)(images)
        return images

    def _init_loss_flow(
        self,
        images_input_name: str,
        corr_fn,
        *corr_args,
        **corr_kwargs,
    ):
        self.flow.define_transition(
            "augmented_images",
            self._aug_images,
            images_input_name,
            "use_image_aug",
        )
        self.flow.define_transition(
            "gray_images",
            self._grayify,
            "augmented_images",
        )
        self.flow.define_transition(
            ("descriptors", "logits"),
            self._model.forward_flow,
            outputs=Flow.Constant(("normalized_descriptors", "logits")),
            images="gray_images",
        )
        self.flow.define_transition(
            "descriptors_shape",
            lambda x: x.shape,
            "descriptors",
        )
        self.flow.define_transition(
            ("corr_forward", "corr_backward"),
            corr_fn,
            *corr_args,
            **corr_kwargs,
        )
        self.flow.define_transition(
            ("logits_0", "logits_1"),
            self._split_logits,
            "logits",
        )
        self.flow.define_transition(
            ("descriptors_0", "descriptors_1"),
            self._split_descriptors,
            "descriptors",
        )
        self.flow.define_transition(
            ("acontextual_descriptor_loss", "keypoint_loss", "precision", "recall"),
            self._loss,
            "descriptors_0",
            "descriptors_1",
            "corr_forward",
            "corr_backward",
            "logits_0",
            "logits_1",
            Flow.Constant(self._ghost_sim),
        )
        self.flow.define_transition(
            ("contextual_descriptor_0", "contextual_descriptor_1"),
            self._contextualize,
            "descriptors_0",
            "descriptors_1",
            "descriptors_shape",
        )
        self.flow.define_transition(
            "contextual_descriptor_loss",
            self._contextual_loss,
            "contextual_descriptor_0",
            "contextual_descriptor_1",
            "corr_forward",
            "corr_backward",
            "logits_0",
            "logits_1",
        )
        self._loss_fn = self.flow.with_outputs(
            (
                "contextual_descriptor_loss",
                "acontextual_descriptor_loss",
                "keypoint_loss",
                "precision",
                "recall",
            )
        )

    @property
    def model(self):
        return self._model

    def model_forward_flow(self, *args, **kwargs):
        return self._model.forward_flow(*args, **kwargs)

    def _apply_pe(self, descriptors_0, descriptors_1, descriptors_shape):
        if not self._pe:
            return descriptors_0, descriptors_1
        _0 = torch.zeros((1,) + descriptors_shape[1:], device=descriptors_0.device)
        pe = self._pe(_0)
        pe = self._img_to_flat(pe)
        pe = pe * self.model.descriptor_scale_factor

        return descriptors_0 + pe, descriptors_1 + pe

    def _contextualize(self, descriptors_0, descriptors_1, descriptors_shape=None):
        if self._contextualizer is None:
            return descriptors_0, descriptors_1

        spatial_shape = False
        if not descriptors_shape:
            spatial_shape = True
            assert descriptors_0.ndim == 4
            assert descriptors_1.ndim == 4

            descriptors_shape = descriptors_0.shape
            descriptors_0 = self._img_to_flat(descriptors_0)
            descriptors_1 = self._img_to_flat(descriptors_1)

        assert descriptors_0.ndim == 3
        assert descriptors_1.ndim == 3

        descriptors_0 = descriptors_0.detach()
        descriptors_1 = descriptors_1.detach()

        descriptors_0, descriptors_1 = self._apply_pe(
            descriptors_0, descriptors_1, descriptors_shape
        )

        descriptors_0, descriptors_1 = self._contextualizer(
            descriptors_0, descriptors_1
        )

        if spatial_shape:
            descriptors_0 = self._flat_to_img(descriptors_0, descriptors_shape)
            descriptors_1 = self._flat_to_img(descriptors_1, descriptors_shape)

        return descriptors_0, descriptors_1

    def _contextual_loss(
        self,
        descriptors_0,
        descriptors_1,
        corr_forward,
        corr_backward,
        logits_0,
        logits_1,
    ):
        if self._contextualizer is None:
            return 0.0

        logits_0 = logits_0.detach()
        logits_1 = logits_1.detach()

        desc_loss, _, _, _ = self._loss(
            descriptors_0,
            descriptors_1,
            corr_forward,
            corr_backward,
            logits_0,
            logits_1,
        )

        return desc_loss

    def _aug_images(self, images, use_image_aug):
        if use_image_aug:
            images = self._image_aug_transform(images)
        return images

    def _split_descriptors(self, descriptors):
        desc_0 = SiLKBase._img_to_flat(descriptors[0::2])
        desc_1 = SiLKBase._img_to_flat(descriptors[1::2])
        return desc_0, desc_1

    def _split_logits(self, logits):
        logits_0 = SiLKBase._img_to_flat(logits[0::2]).squeeze(-1)
        logits_1 = SiLKBase._img_to_flat(logits[1::2]).squeeze(-1)
        return logits_0, logits_1

    @staticmethod
    def _img_to_flat(x):
        # x : BxCxHxW
        batch_size = x.shape[0]
        channels = x.shape[1]
        x = x.reshape(batch_size, channels, -1)
        x = x.permute(0, 2, 1)
        return x

    @staticmethod
    def _flat_to_img(x, shape):
        # x : BxNxC
        assert len(shape) == 4
        assert shape[0] == x.shape[0]
        assert shape[1] == x.shape[2]

        x = x.permute(0, 2, 1)
        x = x.reshape(shape)
        return x

    def _total_loss(self, mode, batch, use_image_aug: bool):
        ctx_desc_loss, actx_desc_loss, keypt_loss, precision, recall = self._loss_fn(
            batch, use_image_aug
        )
        f1 = (2 * precision * recall) / (precision + recall)

        loss = ctx_desc_loss + actx_desc_loss + keypt_loss

        self.log(f"{mode}.total.loss", loss)
        self.log(f"{mode}.contextual.descriptors.loss", ctx_desc_loss)
        self.log(f"{mode}.acontextual.descriptors.loss", actx_desc_loss)
        self.log(f"{mode}.keypoints.loss", keypt_loss)
        self.log(f"{mode}.precision", precision)
        self.log(f"{mode}.recall", recall)
        self.log(f"{mode}.f1", f1)
        if (self._ghost_sim is not None) and (mode == "train"):
            self.log("ghost.sim", self._ghost_sim)

        return loss

    def training_step(self, batch, batch_idx):
        return self._total_loss(
            "train",
            batch,
            use_image_aug=True,
        )

    def validation_step(self, batch, batch_idx):
        return self._total_loss(
            "val",
            batch,
            use_image_aug=False,
        )


class SiLKRandomHomographies(SiLKBase):
    def __init__(
        self,
        model,
        loss,
        optimizer_spec: Union[Spec, None] = None,
        image_aug_transform: Union[Transform, None] = None,
        training_random_homography_kwargs: Union[Dict[str, Any], None] = None,
        **kwargs,
    ):
        SiLKBase.__init__(
            self,
            model,
            loss,
            optimizer_spec,
            image_aug_transform,
            **kwargs,
        )

        # homographic sampler arguments
        self._training_random_homography_kwargs = (
            {}
            if training_random_homography_kwargs is None
            else training_random_homography_kwargs
        )

        self.flow.define_transition("checked_batch", self._check_batch, "batch")
        self.flow.define_transition(
            ("images", "image_shape"),
            self._get_images,
            "checked_batch",
        )
        self.flow.define_transition(
            ("sampler", "warped_images"),
            self._warp_images,
            "images",
        )

        self._init_loss_flow(
            "warped_images",
            self._get_corr,
            "sampler",
            "descriptors",
            "image_shape",
        )

    def _check_batch(self, batch):
        # check batch
        ensure_is_instance(batch, NamedContext)
        batch.ensure_exists("image")

        # check data shape
        assert len(batch["image"].shape) == 4

        def to_device(el):
            if isinstance(el, torch.Tensor):
                return el.to(self.device)
            raise RuntimeError(f"type {type(el)} not handled")

        # send data to model's device
        batch = batch.map(to_device)

        return batch

    def _get_images(self, batch):
        assert isinstance(batch["image"], torch.Tensor)

        # check data shape
        shape = batch["image"].shape

        return batch["image"], shape

    def _warp_images(self, images):
        shape = images.shape
        images = images.to(torch.float32)

        # apply two homographic transforms to each input images
        sampler = RandomHomographicSampler(
            shape[0],
            shape[-2:],
            device=images.device,
            **self._training_random_homography_kwargs,
        )

        warped_images = sampler.forward_sampling(images)

        images = torch.stack((images, warped_images), dim=1)
        images = images.view((-1,) + shape[1:])

        return sampler, images

    def _get_corr(self, sampler, descriptors, image_shape):
        batch_size = image_shape[0]
        descriptors_height = descriptors.shape[2]
        descriptors_width = descriptors.shape[3]
        cell_size = 1.0

        # remove confidence value
        positions = HomographicSampler._create_meshgrid(
            descriptors_height,
            descriptors_width,
            device=descriptors.device,
            normalized=False,
        )
        positions = positions.expand(batch_size, -1, -1, -1)  # add batch dim
        positions = positions.reshape(batch_size, -1, 2)

        coord_mapping = self._model.coordinate_mapping_composer.get(
            "images",
            "raw_descriptors",
        )

        # send to image coordinates
        positions = coord_mapping.reverse(positions)

        # transform label positions to transformed image space
        warped_positions_forward = sampler.transform_points(
            positions,
            image_shape=image_shape[-2:],
            direction="forward",
            ordering="xy",
        )

        warped_positions_backward = sampler.transform_points(
            positions,
            image_shape=image_shape[-2:],
            direction="backward",
            ordering="xy",
        )

        # send back to descriptor coordinates
        warped_positions_forward = coord_mapping.apply(warped_positions_forward)
        warped_positions_backward = coord_mapping.apply(warped_positions_backward)

        corr_forward = positions_to_unidirectional_correspondence(
            warped_positions_forward,
            descriptors_width,
            descriptors_height,
            cell_size,
            ordering="xy",
        )

        corr_backward = positions_to_unidirectional_correspondence(
            warped_positions_backward,
            descriptors_width,
            descriptors_height,
            cell_size,
            ordering="xy",
        )

        corr_forward, corr_backward = keep_mutual_correspondences_only(
            corr_forward, corr_backward
        )

        return corr_forward, corr_backward
