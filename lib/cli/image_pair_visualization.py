# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Draw point correspondences between a pair of images.

1. Load input images.
2. Run SuperPoint on both images and get points output.
3. Get point correspondences with nearest-neighbor approach.
4. Prepare visualization of image pair:
    a. Draw predicted points on both images.
    b. Draw lines from one image to the other for matched points.
"""
import os

import cv2
import numpy as np
import pytorch_lightning as pl
from omegaconf import DictConfig

from silk.config.core import instantiate_and_ensure_is_instance
from silk.matching.mnn import compute_dist, match_descriptors
from silk.models.superpoint_utils import _process_output_new, load_image


def get_point_correspondences(image1, image2, img_height, img_width, model):
    image1 = load_image(image1, img_height, img_width).unsqueeze(dim=0)
    image2 = load_image(image2, img_height, img_width).unsqueeze(dim=0)

    keypoints, desc = _process_output_new(model, image1, sparse=True)
    warped_keypoints, warped_desc = _process_output_new(model, image2, sparse=False)

    distances = compute_dist(desc, warped_desc.T)
    matches = match_descriptors(distances)

    matched_keypoints = keypoints[matches[:, 0]].detach().numpy()
    matched_warped_keypoints = warped_keypoints[matches[:, 1]].detach().numpy()

    return matched_keypoints, matched_warped_keypoints


def create_img_pair_visual(
    image1,
    image2,
    img_height,
    img_width,
    matched_keypoints,
    matched_warped_keypoints,
):
    # load in images of shape (img_height, img_width)
    image1 = cv2.imread(image1)
    image2 = cv2.imread(image2)

    # resize if necessary
    if (img_height is not None) and (img_width is not None):
        image1 = cv2.resize(
            image1, (img_width, img_height), interpolation=cv2.INTER_AREA
        )
        image2 = cv2.resize(
            image2, (img_width, img_height), interpolation=cv2.INTER_AREA
        )

    return img_pair_visual(
        image1,
        image2,
        matched_keypoints,
        matched_warped_keypoints,
    )


def img_pair_visual(
    image1,
    image2,
    matched_keypoints,
    matched_warped_keypoints,
    good_matches_mask=None,
):
    img_width = image1.shape[1]

    height = max(image1.shape[0], image2.shape[0])

    if image1.shape[0] < height:
        image1 = np.pad(image1, ((0, height - image1.shape[0]), (0, 0), (0, 0)))

    if image2.shape[0] < height:
        image2 = np.pad(image2, ((0, height - image2.shape[0]), (0, 0), (0, 0)))

    image_pair = np.hstack((image1, image2))

    # convert keypoints to col, row (x, y) order
    matched_keypoints = matched_keypoints[:, [1, 0]]
    matched_warped_keypoints = matched_warped_keypoints[:, [1, 0]]

    matched_keypoints = matched_keypoints.astype(int)
    matched_warped_keypoints = matched_warped_keypoints.astype(int)

    # draw matched keypoint points and lines associating matched keypoints (point correspondences)
    for i in range(len(matched_keypoints)):
        img1_coords = matched_keypoints[i]
        img2_coords = matched_warped_keypoints[i]
        # add the width so the coordinates show up correctly on the second image
        img2_coords = (img2_coords[0] + img_width, img2_coords[1])

        radius = 1
        thickness = 2
        # points will be red (BGR color)
        image_pair = cv2.circle(image_pair, img1_coords, radius, (0, 0, 255), thickness)
        image_pair = cv2.circle(image_pair, img2_coords, radius, (0, 0, 255), thickness)

        thickness = 1

        if good_matches_mask is None:
            color = (
                np.random.randint(0, 255),
                np.random.randint(0, 255),
                np.random.randint(0, 255),
            )
        else:
            if good_matches_mask[i]:
                color = (0, 255, 0)
            else:
                color = (255, 0, 0)
        image_pair = cv2.line(image_pair, img1_coords, img2_coords, color, thickness)
    return image_pair


def save_image(img, output_location, output_img_name="output_img_pair.jpg"):
    file_to_save = os.path.join(output_location, output_img_name)

    # create the directory if it does not exist
    os.makedirs(output_location, exist_ok=True)

    cv2.imwrite(file_to_save, img)


def main(config: DictConfig):
    # load model
    model = instantiate_and_ensure_is_instance(config.mode.model, pl.LightningModule)

    # load images
    image1 = config.mode.image_1_file_path
    image2 = config.mode.image_2_file_path

    # define image sizes
    img_height = 480
    img_width = 640

    # compute the predicted points and point correspondences
    matched_keypoints, matched_warped_keypoints = get_point_correspondences(
        image1,
        image2,
        img_height,
        img_width,
        model,
    )

    # create the visualization images
    output_image_pair = create_img_pair_visual(
        image1,
        image2,
        img_height,
        img_width,
        matched_keypoints,
        matched_warped_keypoints,
    )

    # optionally save the output image pair
    if config.mode.save_output:
        save_image(
            output_image_pair,
            config.mode.save_location,
            output_img_name=config.mode.save_file_name,
        )

    return {
        "output_img_path": os.path.join(
            config.mode.save_location, config.mode.save_file_name
        )
    }
