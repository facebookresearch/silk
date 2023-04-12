# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import cv2
import torch
from omegaconf import DictConfig

from silk.config.core import instantiate_and_ensure_is_instance
from silk.config.paths import ASSETS
from silk.models.superpoint import SuperPoint

ASSETS = ASSETS / "tests" / "magicpoint"


def _load_fb_logo():
    H = 40
    W = 120

    # open the fb logo as the test image (in grayscale)
    input_image = cv2.imread(str(ASSETS / "fb_logo.png"), 0)

    # resize the image (resize dimensions are horizontal, then vertical)
    input_image = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_AREA)
    input_image = input_image.astype("float32") / 255.0

    # convert to tensor with appropriate shape for input to the model
    input_image = torch.from_numpy(input_image)
    input_image = input_image.view(1, 1, H, W)

    return input_image


def _load_tensors(*names, device="cpu"):
    raw_tensors = (torch.load(ASSETS / name, map_location=device) for name in names)
    return (
        rt if isinstance(rt, torch.Tensor) else torch.tensor(rt, device=device)
        for rt in raw_tensors
    )


def _sort_sparse_descriptors(positions, sparse_descriptors):
    # NOTE: we sort the corner locations by probability to check consistency
    # with the SuperPoint paper's version (which is also sorted by probability)
    assert len(positions) == 1
    assert len(sparse_descriptors) == 1

    probs = positions[0][:, 2]

    # sort the list of probability values with most confident corners first
    prob_indices = torch.argsort(probs, dim=0, descending=True)

    return sparse_descriptors[0][prob_indices]


def main(config: DictConfig):
    """Check the homographic adaptation predition on images and saving them to disk."""

    # 1. load test image and SuperPoint model
    image = _load_fb_logo()
    model = instantiate_and_ensure_is_instance(config.mode.model, SuperPoint)
    model.eval()
    image = image.to(model.device)

    # 2. compute our model outputs
    logits, nms, descriptors, sparse_descriptors, positions = model.model_forward_flow(
        [
            "logits",
            "nms",
            "coarse_descriptors",
            "sparse_descriptors",
            "positions",
        ],
        image,
    )

    # 3. load outputs of pre-trained model
    (
        pretrained_logits,
        pretrained_nms,
        pretrained_descriptors,
        pretrained_sparse_descriptors,
    ) = _load_tensors(
        "pretrained_outputs.pt",
        "fb_logo_nms_paper.pt",
        "pretrained_outputs_super.pt",
        "fb_logo_descriptor_paper.pt",
        device=model.device,
    )

    # 4. check parity of results
    rtol = 0.0
    atol = 1e-4

    assert torch.allclose(
        logits, pretrained_logits, rtol=rtol, atol=atol
    ), "logits do not match"
    assert torch.all((nms > 0.0) == pretrained_nms), "nms do not match"
    assert torch.allclose(
        descriptors, pretrained_descriptors, rtol=rtol, atol=atol
    ), "descriptors do not match"

    assert torch.allclose(
        _sort_sparse_descriptors(positions, sparse_descriptors).T,
        pretrained_sparse_descriptors,
        rtol=rtol,
        atol=0.02,
    ), "sparse descriptors do not match"
