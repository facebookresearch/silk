# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
The SuperPoint model, as subclassed from magicpoint.py.
SuperPoint adds a descriptor head to the MagicPoint model.
"""

from typing import Any, Dict, Iterable, Optional, Union

import pytorch_lightning as pl
import torch
from silk.backbones.superpoint.utils import positions_to_label_map, space_to_depth
from silk.config.core import ensure_is_instance
from silk.config.optimizer import Spec
from silk.flow import AutoForward, Flow
from silk.losses.superpoint import build_similarity_mask, DescriptorLoss, KeypointLoss
from silk.models.abstract import OptimizersHandler, StateDictRedirect
from silk.models.magicpoint import HomographyAdaptation
from silk.transforms.abstract import NamedContext, Transform
from silk.transforms.cv.homography import RandomHomographicSampler


class SuperPoint(
    OptimizersHandler,
    AutoForward,
    StateDictRedirect,
    pl.LightningModule,
    HomographyAdaptation,
):
    DEFAULT_DESCRIPTOR_LOSS = DescriptorLoss()
    DEFAULT_KEYPOINT_LOSS = KeypointLoss()

    def __init__(
        self,
        model,
        optimizer_spec: Union[Spec, None] = None,
        image_aug_transform: Union[Transform, None] = None,
        warp_original: bool = False,
        descriptor_loss=DEFAULT_DESCRIPTOR_LOSS,
        detection_loss=DEFAULT_KEYPOINT_LOSS,
        lamdba_descriptor_loss: float = 0.0001,
        training_random_homography_kwargs: Union[Dict[str, Any], None] = None,
        random_homographic_adaptation_kwargs: Union[Dict[str, Any], None] = None,
        default_outputs: Union[str, Iterable[str]] = ("coarse_descriptors", "logits"),
        **kwargs,
    ):
        """Initialize the SuperPoint model.

        Assumes an RGB image with 1 color channel (grayscale image).

        Parameters
        ----------
        optimizer_spec : Spec
            Optimizer spec to use for training.
        image_aug_transform : Union[Transform, None], optional
            Transform to apply to every warped images used during training.
        warp_original : bool, optional
            Warps original image during training, by default False
        lamdba_descriptor_loss : float, optional
            Descriptor loss weight, by default 0.0001
        random_homographic_adaptation_kwargs : Union[Dict[str, Any], None]
            Parameters passed to `RandomHomographicSampler` (used during homographic adaptation)
        training_random_homography_kwargs: Union[Dict[str, Any], None]
            Parameters passed to `RandomHomographicSampler` (used during training)
        """

        OptimizersHandler.__init__(self, optimizer_spec)
        pl.LightningModule.__init__(self, **kwargs)
        StateDictRedirect.__init__(self, model)
        AutoForward.__init__(self, Flow("batch", "use_image_aug"), default_outputs)
        HomographyAdaptation.__init__(
            self,
            random_homographic_adaptation_kwargs,
            self._get_scores,
            model.magicpoint._detection_threshold,
            model.magicpoint._nms_dist,
            model.magicpoint._border_dist,
        )

        self._model = model
        self._cell_size = self.model.magicpoint._cell_size
        self._detection_loss = detection_loss
        self._descriptor_loss = descriptor_loss

        self._lamdba_descriptor_loss = lamdba_descriptor_loss
        self._image_aug_transform = image_aug_transform

        # homographic sampler arguments
        self._training_random_homography_kwargs = (
            {}
            if training_random_homography_kwargs is None
            else training_random_homography_kwargs
        )

        self._warp_original = warp_original

        self.flow.define_transition("checked_batch", self._check_batch, "batch")
        self.flow.define_transition(
            ("images", "image_shape"),
            self._get_images,
            "checked_batch",
        )
        self.flow.define_transition("positions", self._get_positions, "checked_batch")
        self.flow.define_transition(
            ("sampler", "warped_images"),
            self._warp_images,
            "images",
        )
        self.flow.define_transition(
            "augmented_images",
            self._aug_images,
            "warped_images",
            "use_image_aug",
        )
        self.flow.define_transition(
            "warped_positions",
            self._warp_positions,
            "sampler",
            "positions",
            "image_shape",
        )
        self.flow.define_transition(
            "labels",
            self._get_labels,
            "warped_positions",
            "image_shape",
        )
        self.flow.define_transition(
            ("descriptors", "logits"),
            self._model.forward_flow,
            outputs=Flow.Constant(("coarse_descriptors", "logits")),
            images="augmented_images",
        )
        self.flow.define_transition(
            ("positions_0", "positions_1"),
            self._split_batch_dim,
            "warped_positions",
        )
        self.flow.define_transition(
            ("descriptors_0", "descriptors_1"),
            self._split_batch_dim,
            "descriptors",
        )
        self.flow.define_transition(
            "similarity_mask",
            build_similarity_mask,
            descriptors="descriptors",
            positions_0="positions_0",
            positions_1="positions_1",
            cell_size=Flow.Constant(self._cell_size),
        )

        # compute losses
        self.flow.define_transition(
            "detection_loss",
            self._detection_loss,
            "logits",
            "labels",
        )
        self.flow.define_transition(
            "descriptor_loss",
            self._descriptor_loss,
            "descriptors_0",
            "descriptors_1",
            "similarity_mask",
        )

        self._batch_to_both_losses = self.flow.with_outputs(
            ("detection_loss", "descriptor_loss")
        )

    def _get_scores(self, images):
        images = images.to(self.device)
        return self._model.forward_flow("score", images)

    @property
    def model(self):
        return self._model

    def model_forward_flow(self, *args, **kwargs):
        return self._model.forward_flow(*args, **kwargs)

    def _check_batch(self, batch):
        # check batch
        ensure_is_instance(batch, NamedContext)

        if self.training:
            batch.ensure_exists("image", "positions")
        else:
            batch.ensure_exists("image")

        def to_device(el):
            if isinstance(el, torch.Tensor):
                return el.to(self.device)
            elif isinstance(el, list):
                return [e.to(self.device) for e in el]
            raise RuntimeError(f"type {type(el)} not handled")

        # send data to model's device
        batch = batch.map(to_device)

        return batch

    def _split_batch_dim(self, tensor):
        return tensor[0::2], tensor[1::2]

    def _get_images(self, batch):
        assert isinstance(batch["image"], torch.Tensor)

        # check data shape
        shape = batch["image"].shape
        assert len(shape) == 4
        assert shape[1] == 1
        assert shape[2] % self._cell_size == 0
        assert shape[3] % self._cell_size == 0

        return batch["image"], shape

    def _get_labels(self, positions, shape):
        assert isinstance(positions, list)

        # get label map of sampled images
        label_map = positions_to_label_map(positions, shape[-2:])
        label_map = label_map.permute(0, 3, 1, 2)

        # get labels by adding dustbin and conver cells to depth
        label_map = space_to_depth(label_map, self._cell_size)

        return label_map

    def _get_positions(self, batch):
        # remove confidence value
        return [p[..., :2] for p in batch["positions"]]

    def _warp_images(self, images):
        shape = images.shape

        # apply two homographic transforms to each input images
        sampler = RandomHomographicSampler(
            (2 if self._warp_original else 1) * shape[0],
            shape[-2:],
            device=images.device,
            **self._training_random_homography_kwargs,
        )

        warped_images = sampler.forward_sampling(images)

        if self._warp_original:
            images = warped_images
        else:
            images = torch.stack((images, warped_images), dim=1)
            images = images.view((-1,) + shape[1:])

        return sampler, images

    def _warp_positions(self, sampler, positions, images_shape):
        # transform label positions to transformed image space
        warped_positions = sampler.transform_points(
            positions,
            image_shape=images_shape[-2:],
            direction="forward",
            ordering="yx",
        )

        if self._warp_original:
            positions = warped_positions
        else:
            new_positions = []
            for p0, p1 in zip(positions, warped_positions):
                new_positions.extend((p0, p1))
            positions = new_positions

        return positions

    def _aug_images(self, images, use_image_aug):
        if use_image_aug:
            images = self._image_aug_transform(images)
        return images

    def _total_loss(
        self,
        mode,
        batch,
        use_image_aug,
    ):
        detection_loss, descriptor_loss = self._batch_to_both_losses(
            batch, use_image_aug
        )

        total_loss = detection_loss + self._lamdba_descriptor_loss * descriptor_loss

        self.log(f"{mode}.detection.loss", detection_loss)
        self.log(f"{mode}.descriptor.loss", descriptor_loss)
        self.log(f"{mode}.total.loss", total_loss)

        return total_loss

    def training_step(self, batch, batch_idx):
        return self._total_loss("train", batch, use_image_aug=True)

    def validation_step(self, batch, batch_idx):
        return self._total_loss("val", batch, use_image_aug=False)

    def predict_step(
        self,
        batch: NamedContext,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
    ) -> Any:
        images = self.flow.with_outputs("images")(batch)
        points = self._model.flow.with_outputs("positions")(images)
        return batch.add("points", points)
