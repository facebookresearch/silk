# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

import cv2 as cv
import torch
from silk.matching.mnn import compute_dist, mutual_nearest_neighbor

matcher = partial(
    mutual_nearest_neighbor, distance_fn=partial(compute_dist, dist_type="l2")
)


class SIFT:
    def __init__(self, device) -> None:
        self._sift = cv.SIFT_create()
        self._device = device

    def __call__(self, images: torch.Tensor):
        # convert image to numpy
        images = images * 255
        images = images.permute(0, 2, 3, 1)
        images = images.to(torch.uint8)
        images = images.numpy()

        keypoints = []
        descriptors = []

        for image in images:
            kp, desc = self._sift.detectAndCompute(image, None)

            # normalize
            kp = torch.tensor(tuple(k.pt for k in kp), device=self._device)
            desc = torch.tensor(desc, device=self._device)

            # xy to yx
            kp = kp[:, [1, 0]]

            keypoints.append(kp)
            descriptors.append(desc)

        return tuple(keypoints), tuple(descriptors)
