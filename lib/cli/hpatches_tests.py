# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

import numpy as np
import torch
from omegaconf import DictConfig
from silk.cli.image_pair_visualization import img_pair_visual
from silk.config.core import instantiate_and_ensure_is_instance
from silk.datasets.synthetic.primitives import draw_interest_points
from silk.logger import LOG
from silk.metrics.hpatches_metrics import MeanMatchingAccuracy
from silk.transforms.abstract import Transform
from silk.transforms.tensor import Unbatch
from skimage.io import imsave
from torch.utils.data import DataLoader
from tqdm import tqdm


def _tensor_to_python(tensor):
    if len(tensor.shape) > 0:
        return [_tensor_to_python(t) for t in tensor]
    return tensor.item()


def compute_metrics(metric_updates):
    """Compute and return all provided metrics.

    Returns
    -------
    dict
        Mapped values of the provided metrics.
    """
    return {
        name: _tensor_to_python(metric_update.metric.compute().detach().cpu())
        for name, metric_update in metric_updates.items()
    }


def image_dump(name, image, viz_options):
    path = f"{viz_options.directory}/{name}.png"
    LOG.warning(f"debug dump image to : {path}")
    imsave(path, image)


def torch_image_to_numpy(image):
    # permute if channel is first
    if image.shape[0] == 1 or image.shape[0] == 3:
        image = image.permute((1, 2, 0))
    image = image.squeeze()
    image = image.detach().cpu().numpy()
    image = (image * 255.0).astype(np.uint8)
    return image.copy()


def gray_image_to_rgb(image):
    image = image.squeeze()
    if len(image.shape) == 3:  # already RGB
        return image
    image = torch.stack((image, image, image), dim=0)
    return image


def vizualize_keypoints(name, image, points, matched_points, viz_options):
    image = torch_image_to_numpy(image)
    points = points.detach().cpu().numpy()

    image = draw_interest_points(
        image,
        points,
        (0, 255, 0),
    )

    if matched_points is not None:
        matched_points = matched_points.detach().cpu().numpy()
        image = draw_interest_points(
            image,
            matched_points,
            (255, 0, 0),
        )

    image_dump(name, image, viz_options)


def visualize_keypoint_matches(
    name,
    image_0,
    image_1,
    matched_points_0,
    matched_points_1,
    good_matches_mask,
    viz_options,
):
    image_0 = torch_image_to_numpy(gray_image_to_rgb(image_0))
    image_1 = torch_image_to_numpy(gray_image_to_rgb(image_1))
    matched_points_0 = matched_points_0.detach().cpu().numpy()
    matched_points_1 = matched_points_1.detach().cpu().numpy()

    image_pair = img_pair_visual(
        image_0,
        image_1,
        matched_points_0,
        matched_points_1,
        good_matches_mask,
    )

    image_dump(name, image_pair, viz_options)


def visualized_probs(name, probs, viz_options):
    probs = torch_image_to_numpy(probs)
    probs = probs.astype(float)
    probs -= probs.min()
    probs /= probs.max()
    image_dump(name, probs, viz_options)


def visualize(idx, elem, viz_options):
    if not viz_options.enabled:
        return idx

    os.makedirs(viz_options.directory, exist_ok=True)

    for el in Unbatch(tuple_as_list=True)(elem):
        _visualize_one(idx, el, viz_options)
        idx += 1

    return idx


def _visualize_one(idx, elem, viz_options):
    vizualize_keypoints(
        f"{idx:04d}.keypoints_original",
        elem["original_img"],
        elem["original_points"],
        elem["matched_original_points"],
        viz_options,
    )
    vizualize_keypoints(
        f"{idx:04d}.keypoints_warped",
        elem["warped_img"],
        elem["warped_points"],
        elem["matched_warped_points"],
        viz_options,
    )
    visualize_keypoint_matches(
        f"{idx:04d}.keypoints_raw_matches",
        elem["original_img"],
        elem["warped_img"],
        elem["matched_original_points"],
        elem["matched_warped_points"],
        good_matches_mask=None,
        viz_options=viz_options,
    )
    visualize_keypoint_matches(
        f"{idx:04d}.keypoints_good_matches",
        elem["original_img"],
        elem["warped_img"],
        elem["matched_original_points"],
        elem["matched_warped_points"],
        good_matches_mask=MeanMatchingAccuracy.good_matches_mask(
            elem["matched_original_points"],
            elem["matched_warped_points"],
            elem["homography"].float().to(elem["matched_original_points"].device),
            threshold=3,
            ordering="yx",
        ),
        viz_options=viz_options,
    )

    if "original_probs" in elem.names():
        visualized_probs(
            f"{idx:04d}.original_probs",
            elem["original_probs"],
            viz_options=viz_options,
        )
    if "warped_probs" in elem.names():
        visualized_probs(
            f"{idx:04d}.warped_probs",
            elem["warped_probs"],
            viz_options=viz_options,
        )


def main(config: DictConfig):
    """
    Compute the repeatability and homography estimation metrics for the HPatches dataset.
    """
    loader = instantiate_and_ensure_is_instance(config.mode.loader, DataLoader)
    transform = instantiate_and_ensure_is_instance(config.mode.transform, Transform)
    metric_updates = instantiate_and_ensure_is_instance(
        config.mode.metric_updates, DictConfig
    )

    idx = 0
    for elem in tqdm(loader):
        elem = transform(elem)

        idx = visualize(idx, elem, config.mode.visualization)

        for _, metric in metric_updates.items():
            metric(elem)

    metrics = compute_metrics(metric_updates)
    return {
        "metrics": metrics,
    }
