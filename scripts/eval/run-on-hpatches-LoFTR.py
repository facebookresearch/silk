# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
from copy import deepcopy

import torch
from common import cache

sys.path.append("/checkpoint/weiyaowang/dev/LoFTR")

from src.loftr import default_cfg, LoFTR

DATASET = "assets/datasets/cached-hpatches/hpatches-full-size-480-grey.h5"
OUTPUT_DATASET = "LoFTR_indoor_new_480.h5"
MODEL_WEIGHT = "/private/home/gleize/code/external/LoFTR/weights/indoor_ds_new.ckpt"
# MODEL_WEIGHT = "/checkpoint/weiyaowang/metapoint/pretrained_weights/loftr_indoor.ckpt"


def transform(elem, matcher):
    # extract images (single image, no batch dimension)
    original_image = elem["original_img"]
    warped_image = elem["warped_img"]

    original_image = original_image[None].cuda()
    warped_image = warped_image[None].cuda()
    batch = {"image0": original_image, "image1": warped_image}

    with torch.no_grad():
        matcher(batch)
        mkpts0 = batch["mkpts0_f"].cpu()
        mkpts1 = batch["mkpts1_f"].cpu()
        mconf = batch["mconf"].cpu()  # noqa: F841

    # run model here
    original_points = mkpts0[..., [1, 0]]
    warped_points = mkpts1[..., [1, 0]]
    matched_original_points = mkpts0[..., [1, 0]]
    matched_warped_points = mkpts1[..., [1, 0]]

    # save keypoints
    elem = elem.add("original_points", original_points)
    elem = elem.add("warped_points", warped_points)
    elem = elem.add("matched_original_points", matched_original_points)
    elem = elem.add("matched_warped_points", matched_warped_points)

    return elem


def main():
    # init LoFTR model
    _default_cfg = deepcopy(default_cfg)
    _default_cfg["coarse"]["temp_bug_fix"] = "indoor" in os.path.basename(
        MODEL_WEIGHT
    )  # set to False when using the old ckpt
    matcher = LoFTR(config=_default_cfg)
    matcher.load_state_dict(torch.load(MODEL_WEIGHT)["state_dict"])
    matcher = matcher.eval().cuda()

    cache(DATASET, OUTPUT_DATASET, transform, matcher)


if __name__ == "__main__":
    main()
