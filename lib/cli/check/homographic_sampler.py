# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

import cv2 as cv
import numpy as np
from omegaconf import DictConfig
from silk.config.core import instantiate_and_ensure_is_instance
from silk.transforms.abstract import NamedContext
from silk.transforms.cv.homography import RandomHomographicSampler
from skimage import io
from torch.utils.data import DataLoader
from tqdm import tqdm


def draw_crops(image, coords):
    """Draw homographic crops in image."""
    if image.shape[2] == 1:
        image = image.repeat(3, axis=2)

    color = (255, 0, 0)
    thickness = 2
    coords[..., 0] = (coords[..., 0] + 1.0) * 0.5 * image.shape[1]
    coords[..., 1] = (coords[..., 1] + 1.0) * 0.5 * image.shape[0]
    coords = coords.detach().cpu().numpy().astype(np.uint16)

    for i in range(coords.shape[0]):
        line = [
            (coords[i][0], coords[i][1]),
            (coords[i][1], coords[i][3]),
            (coords[i][3], coords[i][2]),
            (coords[i][2], coords[i][0]),
        ]
        for j in range(len(line)):
            cv.line(image, line[j][0], line[j][1], color, thickness)
    return image


def main(config: DictConfig):
    """Check the homographic sampler on real images by randomly generating homographic crops and saving them to disk."""

    loader = instantiate_and_ensure_is_instance(config.mode.loader, DataLoader)

    sampler = instantiate_and_ensure_is_instance(
        config.mode.sampler, RandomHomographicSampler
    )

    it = iter(loader)
    for i in tqdm(range(config.mode.output.n_batches)):
        batch_output_dir = os.path.join(config.mode.output.directory, f"{i:04d}")
        os.makedirs(batch_output_dir)

        batch: NamedContext = next(it)
        batch.ensure_exists("image")

        image_samples = sampler(batch["image"])

        for j in range(batch["image"].shape[0]):
            # save images coming from loader
            image = batch["image"][j].permute(1, 2, 0)
            image = image.detach().cpu().numpy().astype(np.uint8)

            crop_per_image = sampler.batch_size // batch["image"].shape[0]
            s = j * crop_per_image
            e = (j + 1) * crop_per_image
            image = draw_crops(image, sampler.src_coords[s:e])

            image_path = os.path.join(batch_output_dir, f"{j:04d}_image.png")
            io.imsave(image_path, image)

            # save homographic crops
            for k in range(crop_per_image):
                image = image_samples[j * crop_per_image + k].permute(1, 2, 0)
                image = image.detach().cpu().numpy().astype(np.uint8)

                image_path = os.path.join(batch_output_dir, f"{j:04d}.{k:04d}_crop.png")
                io.imsave(image_path, image)
