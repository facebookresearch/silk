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

from typing import Any, Dict, Optional, Union

import pytorch_lightning as pl
import torch

from silk.backbones.superpoint.utils import (
    prob_map_to_points_map,
    prob_map_to_positions_with_prob,
    space_to_depth,
)
from silk.config.core import ensure_is_instance
from silk.config.optimizer import Spec
from silk.flow import Flow
from silk.logger import LOG
from silk.models.abstract import OptimizersHandler, StateDictRedirect
from silk.tasks.training.supervised_keypoint import SupervisedKeypoint
from silk.transforms.abstract import NamedContext, Transform
from silk.transforms.cv.homography import RandomHomographicSampler

_DEBUG_MODE_ENABLED = False


# internal debug function to dump counts
# TODO(Pierre): clean
def _debug_dump_counts(counts, device):
    if not _DEBUG_MODE_ENABLED:
        return

    def _dump(image, path):
        from os import makedirs

        from silk.logger import LOG

        from skimage.io import imsave

        makedirs("./debug", exist_ok=True)

        image = image.permute((1, 2, 0))
        image = image.squeeze()
        image = image.detach().cpu().numpy()
        image /= image.max()

        LOG.warning(f"debug dump image to : {path}")
        imsave(path, image)

    for k in range(counts.shape[0]):
        _dump(counts[k], f"./debug/{device}-counts-{k}.png")

    LOG.warning(f'debug mode enabled on "{__file__}"')


class HomographyAdaptation:
    def __init__(
        self,
        random_homographic_adaptation_kwargs,
        score_fn,
        default_detection_threshold=None,
        default_nms_dist=None,
        default_border_dist=None,
    ) -> None:
        # will be initialized / used in homographic adaptation
        self._homographic_sampler = None
        self._random_homographic_adaptation_kwargs = (
            {}
            if random_homographic_adaptation_kwargs is None
            else random_homographic_adaptation_kwargs
        )
        self._default_detection_threshold = default_detection_threshold
        self._default_nms_dist = default_nms_dist
        self._default_border_dist = default_border_dist
        self._score_fn = score_fn

    def _check_homographic_sampler(self, images, n_samples=100):
        """Make sure the homographic sample is initialized for the proper input size."""
        reinit_homographic_sampler = False
        reinit_homographic_sampler |= self._homographic_sampler is None
        if self._homographic_sampler is not None:
            reinit_homographic_sampler |= (
                self._homographic_sampler.batch_size != images.shape[0] * n_samples
            )
            reinit_homographic_sampler |= (
                self._homographic_sampler._sampling_size != images.shape[-2:]
            )
            reinit_homographic_sampler |= (
                self._homographic_sampler.device != images.device
            )

        if reinit_homographic_sampler:
            self._homographic_sampler = RandomHomographicSampler(
                batch_size=images.shape[0] * n_samples,
                sampling_size=images.shape[-2:],
                auto_randomize=False,
                device=images.device,
                **self._random_homographic_adaptation_kwargs,
            )

    def homographic_adaptation_prediction(
        self,
        batch: NamedContext,
        detection_threshold=None,
        nms_dist=None,
        border_dist=None,
        n_samples: int = 100,
        add_identity: bool = False,
    ) -> NamedContext:
        """Prediction using homographic adaptation technique.

        Parameters
        ----------
        batch : NamedContext
            Input batch containing an "image" of shape :math:`(B,C,H,W)`.
        n_samples : int, optional
            Number of homographic samples to generate per image, by default 100.
        add_identity : bool, optional
            Include original image in the set random homographic samples, by default False.

        Returns
        -------
        NamedContext
            New context containing "points" tensor of shape :math:`(B,N,3)` (2D coordinates + probabilities).
        """
        # 1. prepare input and homographic sampler
        ensure_is_instance(batch, NamedContext)
        batch.ensure_exists("image")
        images = batch["image"]
        device = images.device
        detection_threshold = (
            self._default_detection_threshold
            if detection_threshold is None
            else detection_threshold
        )
        nms_dist = self._default_nms_dist if nms_dist is None else nms_dist
        border_dist = self._default_border_dist if border_dist is None else border_dist

        assert detection_threshold is not None, "detection_threshold should be provided"
        assert nms_dist is not None, "nms_dist should be provided"
        assert border_dist is not None, "border_dist should be provided"

        self._check_homographic_sampler(images, n_samples)

        # 2. run inference on initial image
        if add_identity:
            probs_map_identity = self._score_fn(images)
            probs_map_identity = probs_map_identity.to(device)

            counts_identity = torch.ones_like(images)

        # 3. run inference on random homographic crops
        images_samples = self._homographic_sampler.forward_sampling(
            images,
            randomize=True,
        )
        probs_map_samples = self._score_fn(images_samples)
        probs_map_samples = probs_map_samples.to(device)

        # 4. bring prob map to original image referential
        probs_map_samples = self._homographic_sampler.backward_sampling(
            probs_map_samples, randomize=False
        )

        # 5. sum reduction of all probs
        probs_map = probs_map_samples.view(images.shape[0], -1, *images.shape[1:]).sum(
            dim=1
        )
        if add_identity:
            probs_map += probs_map_identity

        # 6. update counts per pixel (to compute average per position)
        ones_crop = torch.ones(
            (1, 1, images.shape[2], images.shape[3]),
            dtype=probs_map.dtype,
            device=self._homographic_sampler.device,
        )
        counts_samples = self._homographic_sampler.backward_sampling(
            ones_crop, randomize=False
        )
        counts = counts_samples.view(images.shape[0], -1, *images.shape[1:]).sum(dim=1)
        if add_identity:
            counts += counts_identity

        _debug_dump_counts(counts, self.device)

        # 7. compute average prob per position
        zero_counts = counts == 0
        final_probs_map = probs_map / counts
        final_probs_map[zero_counts] = 0
        final_probs_map = final_probs_map.squeeze(1)

        # 8. convert to point coordinates (using NMS)
        prob_map = prob_map_to_points_map(
            final_probs_map,
            detection_threshold,
            nms_dist,
            border_dist,
        )
        points = prob_map_to_positions_with_prob(prob_map)

        return batch.add("points", list(points))


class MagicPoint(
    OptimizersHandler,
    StateDictRedirect,
    pl.LightningModule,
    HomographyAdaptation,
):
    def __init__(
        self,
        images_to_logits_fn,
        optimizer_spec: Spec = None,
        image_aug_transform: Union[Transform, None] = None,
        random_homographic_adaptation_kwargs: Union[Dict[str, Any], None] = None,
        **kwargs,
    ):
        """
        Initialize the model.
        Can take an input image of any number of channels (e.g. grayscale, RGB).
        """
        OptimizersHandler.__init__(self, optimizer_spec)
        pl.LightningModule.__init__(self, **kwargs)
        StateDictRedirect.__init__(self, images_to_logits_fn)
        HomographyAdaptation.__init__(
            self,
            random_homographic_adaptation_kwargs,
            self._get_scores,
            images_to_logits_fn._detection_threshold,
            images_to_logits_fn._nms_dist,
            images_to_logits_fn._border_dist,
        )

        self.flow = Flow("batch")
        self.flow.define_transition(
            "checked_batch",
            self._check_batch,
            "batch",
        )
        self.flow.define_transition("images", self._get_images, "checked_batch")
        self.flow.define_transition("labels", self._get_labels, "checked_batch")

        self._images_to_logits_fn = images_to_logits_fn
        self._batch_to_images_labels_fn = self.flow.with_outputs(("images", "labels"))
        self._training_task = SupervisedKeypoint(
            batch_to_images_and_labels_fn=self._batch_to_images_labels_fn,
            images_to_logits_fn=images_to_logits_fn,
            image_aug_transform=image_aug_transform,
        )

        self._cell_size = 8

    @property
    def model(self):
        return self._images_to_logits_fn

    def _get_scores(self, images):
        images = images.to(self.device)
        return self._images_to_logits_fn.forward_flow("score", images)

    def _named_context_to_device(self, batch: NamedContext) -> NamedContext:
        # send context tensors to model device
        def to_device(el):
            if isinstance(el, torch.Tensor):
                return el.to(self.device)
            elif isinstance(el, list):
                return [e.to(self.device) for e in el]
            elif isinstance(el, tuple):
                return tuple(e.to(self.device) for e in el)
            raise RuntimeError(f"type {type(el)} not handled")

        # send data to model's device
        return batch.map(to_device)

    def _check_batch(self, batch: NamedContext):
        # check batch
        ensure_is_instance(batch, NamedContext)

        if self.training:
            batch.ensure_exists("image", "label_map")
            assert batch["image"].shape == batch["label_map"].shape
        else:
            batch.ensure_exists("image")

        return self._named_context_to_device(batch)

    def predict_step(
        self,
        batch: NamedContext,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
    ) -> Any:
        images = self.flow.with_outputs("images")(batch)
        points = self._images_to_logits_fn.flow.with_outputs("positions")(images)
        return batch.add("points", points)

    def test_step(
        self, batch: NamedContext, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> NamedContext:
        images, labels = self._batch_to_images_labels_fn(batch)
        (
            class_probs,
            probs_map,
            nms_pred_positions_with_prob,
        ) = self._images_to_logits_fn.flow.with_outputs(
            (
                "probability",
                "score",
                "positions",
            )
        )(
            images
        )

        class_label = torch.argmax(labels, dim=1)

        pred_positions_with_prob = prob_map_to_positions_with_prob(
            probs_map, threshold=1e-4
        )

        batch = batch.add("one_hot_class_labels", labels)
        batch = batch.add("class_label", class_label)
        batch = batch.add("class_probs", class_probs)
        batch = batch.add("probs_map", probs_map)
        batch = batch.add("pred_positions_with_prob", pred_positions_with_prob)
        batch = batch.add("nms_pred_positions_with_prob", nms_pred_positions_with_prob)

        batch = self._named_context_to_device(batch)

        return batch

    def _get_images(self, batch: NamedContext):
        assert isinstance(batch["image"], torch.Tensor)

        # check data shape
        shape = batch["image"].shape
        assert len(shape) == 4
        assert shape[1] == 1
        assert shape[2] % self._cell_size == 0
        assert shape[3] % self._cell_size == 0

        return batch["image"]

    def _get_labels(
        self,
        batch: NamedContext,
    ):
        assert isinstance(batch["label_map"], torch.Tensor)

        # get labels by adding dustbin and conver cells to depth
        return space_to_depth(batch["label_map"], self._cell_size)

    def training_step(self, batch, batch_idx):
        loss = self._training_task.batch_to_training_loss_fn(batch)
        self.log("train.loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._training_task.batch_to_validation_loss_fn(batch)
        self.log("val.loss", loss)
        return loss
