# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from typing import Union

import torch
import torch.nn as nn

from silk.flow import Flow
from silk.transforms.abstract import Transform


class Training:
    def __init__(self, batch_to_training_loss_fn, batch_to_validation_loss_fn) -> None:
        self._batch_to_training_loss_fn = batch_to_training_loss_fn
        self._batch_to_validation_loss_fn = batch_to_validation_loss_fn

    @property
    def batch_to_training_loss_fn(self):
        return self._batch_to_training_loss_fn

    @property
    def batch_to_validation_loss_fn(self):
        return self._batch_to_validation_loss_fn


class SupervisedKeypoint(Training, torch.nn.Module):
    """Supervised Keypoint Learning
    Reponsibilities :
    - Data Augmentations
    - Loss

    Provide a map from batch to both validation and training losses.
    """

    def __init__(
        self,
        batch_to_images_and_labels_fn,
        images_to_logits_fn,
        image_aug_transform: Union[Transform, None] = None,
    ):
        # AutoForward.__init__(self, Flow("batch"), "validation_loss")
        torch.nn.Module.__init__(self)

        # loss function
        self._loss = nn.CrossEntropyLoss()
        self._image_aug_transform = image_aug_transform

        self._flow = Flow("batch")
        self._flow.define_transition(
            ("images", "labels"),
            batch_to_images_and_labels_fn,
            "batch",
        )
        self._flow.define_transition(
            "augmented_images",
            self.image_aug_transform,
            "images",
        )
        self._flow.define_transition("logits", images_to_logits_fn, "images")
        self._flow.define_transition(
            "augmented_logits",
            images_to_logits_fn,
            "augmented_images",
        )
        self._flow.define_transition("validation_loss", self._loss, "logits", "labels")
        self._flow.define_transition(
            "training_loss",
            self._loss,
            "augmented_logits",
            "labels",
        )

        # self._batch_to_validation_loss_fn = self._flow.with_outputs("validation_loss")
        # self._batch_to_training_loss_fn = self._flow.with_outputs("training_loss")

        Training.__init__(
            self,
            self._flow.with_outputs("training_loss"),
            self._flow.with_outputs("validation_loss"),
        )

    def forward(self, x):
        return x

    def image_aug_transform(self, images: torch.Tensor) -> torch.Tensor:
        if self._image_aug_transform is None:
            warnings.warn(
                "The Supervised Keypoint's training task is running without image augmentation. This could greatly reduce the model's performance.",
                UserWarning,
            )
            return images
        return self._image_aug_transform(images)
