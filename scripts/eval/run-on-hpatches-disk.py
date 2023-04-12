# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys

import numpy as np
import torch

# Hacky; change to relative
sys.path.append("/checkpoint/weiyaowang/dev/disk/")

from common import cache
from disk import DISK
from match import match

DATASET = "assets/datasets/cached-hpatches/hpatches-full-size-480-rgb.h5"
OUTPUT_DATASET = "disk_all_points_480.h5"
MODEL_WEIGHT = "/checkpoint/weiyaowang/metapoint/pretrained_weights/depth-save.pth"


def transform(elem, model):
    # extract images (single image, no batch dimension)
    def preprocess_image(image):
        h, w = image.shape[2:]
        return image[..., : int(h / 16) * 16, : int(w / 16) * 16]

    original_image = preprocess_image(elem["original_img"][None].cuda() / 255.0)
    warped_image = preprocess_image(elem["warped_img"][None].cuda() / 255.0)

    batched_images = torch.cat([original_image, warped_image], dim=0)
    with torch.no_grad():
        features = model.features(
            batched_images,
            kind="nms",
            window_size=5,
            cutoff=0.0,
            n=None,
        )

    def postprocess_feature(feature):
        feature = feature.to(torch.device("cpu"))
        kps_crop_space = feature.kp.T
        x, y = kps_crop_space
        h, w = original_image.shape[2:]
        mask = (0 <= x) & (x < w) & (0 <= y) & (y < h)
        keypoints = kps_crop_space.numpy().T[mask]
        descriptors = feature.desc.numpy()[mask]
        scores = feature.kp_logp.numpy()[mask]
        order = np.argsort(scores)[::-1]
        keypoints = keypoints[order]
        descriptors = descriptors[order]
        scores = scores[order]
        return keypoints, descriptors, scores

    o_keypoint, o_descriptor, o_score = postprocess_feature(features[0])
    w_keypoint, w_descriptor, w_score = postprocess_feature(features[1])

    matches = match(torch.from_numpy(o_descriptor), torch.from_numpy(w_descriptor))

    # run model here
    original_points = torch.from_numpy(o_keypoint)[..., [1, 0]]
    warped_points = torch.from_numpy(w_keypoint)[..., [1, 0]]
    matched_original_points = torch.from_numpy(o_keypoint[matches[0]])[..., [1, 0]]
    matched_warped_points = torch.from_numpy(w_keypoint[matches[1]])[..., [1, 0]]

    # save keypoints
    elem = elem.add("original_points", original_points)
    elem = elem.add("warped_points", warped_points)
    elem = elem.add("matched_original_points", matched_original_points)
    elem = elem.add("matched_warped_points", matched_warped_points)

    return elem


def main():
    # load model
    state_dict = torch.load(MODEL_WEIGHT, map_location="cpu")

    if "extractor" in state_dict:
        weights = state_dict["extractor"]
    elif "disk" in state_dict:
        weights = state_dict["disk"]
    else:
        raise KeyError("Incompatible weight file!")

    model = DISK(window=8, desc_dim=128)
    model.load_state_dict(weights)
    model = model.cuda()
    model = model.eval()

    cache(DATASET, OUTPUT_DATASET, transform, model)


if __name__ == "__main__":
    main()
