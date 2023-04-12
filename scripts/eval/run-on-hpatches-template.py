# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from common import cache

DATASET = "assets/datasets/cached-hpatches/hpatches-full-size-480-grey.h5"
OUTPUT_DATASET = "dataset.h5"


def transform(elem):
    # extract images (single image, no batch dimension)
    original_image = elem["original_img"]  # noqa: F841
    warped_image = elem["warped_img"]  # noqa: F841

    # run model here
    original_points = None  # torch tensor [N, 2] : "yx" coordinates of N keypoints from original image
    warped_points = (
        None  # torch tensor [N, 2] : "yx" coordinates of N keypoints from warped image
    )
    matched_original_points = None  # torch tensor [N, 2] : "yx" coordinates of N keypoints from original image (after matching)
    matched_warped_points = None  # torch tensor [N, 2] : "yx" coordinates of N keypoints from warped image (before matching)

    # save keypoints
    elem = elem.add("original_points", original_points)
    elem = elem.add("warped_points", warped_points)
    elem = elem.add("matched_original_points", matched_original_points)
    elem = elem.add("matched_warped_points", matched_warped_points)

    return elem


def main():
    cache(DATASET, OUTPUT_DATASET, transform)


if __name__ == "__main__":
    main()
