# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

import numpy as np
import torch
from omegaconf import DictConfig
from silk.config.core import instantiate_and_ensure_is_instance
from silk.datasets.synthetic.primitives import draw_interest_points
from silk.models.magicpoint import HomographyAdaptation
from silk.profiler import timeit
from silk.transforms.abstract import NamedContext
from skimage import io
from torch.utils.data import DataLoader
from tqdm import tqdm


def _save_image(batch_output_dir, j, tag, image):
    image_path = os.path.join(batch_output_dir, f"{j:04d}_{tag}_image.png")
    io.imsave(image_path, image)


def main(config: DictConfig):
    """Check the homographic adaptation predition on images and saving them to disk."""

    loader = instantiate_and_ensure_is_instance(config.mode.loader, DataLoader)
    model = instantiate_and_ensure_is_instance(config.mode.model, HomographyAdaptation)

    model.eval()

    with torch.no_grad():
        it = iter(loader)
        for i in tqdm(range(config.mode.output.n_batches)):
            batch_output_dir = os.path.join(config.mode.output.directory, f"{i:04d}")
            os.makedirs(batch_output_dir)

            with timeit(message_template="getting next batch [duration] = {duration}"):
                batch: NamedContext = next(it)

            batch.ensure_exists("image")

            with timeit(
                message_template="homographic adaptation [duration] = {duration}"
            ):
                batch = model.homographic_adaptation_prediction(batch)
            batch = batch.rename("points", "ha_points")

            with timeit(message_template="prediction [duration] = {duration}"):
                batch = model.predict_step(batch)
            batch = batch.rename("points", "no_points")

            for j in range(len(batch["image"])):
                # save images coming from loader
                image = batch["image"][j].permute(1, 2, 0)
                image *= 255.0
                image = image.detach().cpu().numpy().astype(np.uint8)

                ha_image = draw_interest_points(image, batch["ha_points"][j])
                no_image = draw_interest_points(image, batch["no_points"][j])

                _save_image(batch_output_dir, j, "ha", ha_image)
                _save_image(batch_output_dir, j, "no", no_image)
