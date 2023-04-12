# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Setting up the HPatches dataset.

This file contains a function to load in images from hpatches directory
on faircluster and a PyTorch Dataset class for the HPatches dataset.
"""

import os
from pathlib import Path

import cv2
import numpy as np
import torch

from silk.cv.homography import resize_homography
from silk.models.superpoint_utils import load_image

HPATCHES_DEVFAIR_DIR = os.path.join("hpatches", "01042022")


class HPatchesDataset(torch.utils.data.Dataset):
    """
    The HPatches datset class.
    """

    def __init__(
        self,
        hpatches_path,
        img_alteration=None,
        num_to_load=None,
        img_size=None,  # none will have original size, or can give a scale factor as a float
        ignore_folders=(),
        max_img_size=None,
        min_img_size=None,
        force_min_max_resizing=False,
        grayscale=True,
    ):
        """
        Initialize the HPatches dataset from the hpatches_path (hpatches directory).
        """

        def _load_hpatches(
            hpatches_path,
            img_alteration,
            num_to_load,
            ignore_folders,
        ):
            """
            Load an image from hpatches and all of its homographies, plus the homography
            matrices linking the original image to the transformed images.

            Args:
                hpatches_path (str): the path to the directory containing the hpatches
                    dataset images
                img_alteration (optional str): i for illumination changes, v for change
                    in image perspective. Default is None for loading all (i and v) images.
                num_to_load (optional int): the number of image sets to load (will load
                    5 image pairs per image set because there are 5 pairs in each
                    directory of the hpatches dataset)

            Returns:
                files (dict): a dictionary containing the image paths, warped image paths,
                    and homographies connecting each pair of images
            """
            hpatches_path = Path(hpatches_path)

            folder_paths = [x for x in hpatches_path.iterdir() if x.is_dir()]
            image_paths = []
            warped_image_paths = []
            homographies = []

            count = 0

            # go through each image directory and load in images
            for path in folder_paths:
                if img_alteration == "i" and path.stem[0] != "i":
                    continue

                if img_alteration == "v" and path.stem[0] != "v":
                    continue

                if os.path.basename(path) in ignore_folders:
                    continue

                # NUM_IMAGES is always 5, the number of pairs of images per hpatches directory
                NUM_IMAGES = 5
                file_ext = ".ppm"

                for i in range(2, 2 + NUM_IMAGES):
                    # the original image is always 1.ppm
                    image_paths.append(str(Path(path, "1" + file_ext)))

                    # the warped images are named 2 through 6.ppm
                    warped_image_paths.append(str(Path(path, str(i) + file_ext)))

                    # there is one homography matrix for each warped image
                    homographies.append(np.loadtxt(str(Path(path, "H_1_" + str(i)))))

                count += 1

                if num_to_load is not None:
                    if count >= num_to_load:
                        break

            files = {
                "image_paths": image_paths,
                "warped_image_paths": warped_image_paths,
                "homography": homographies,
            }

            return files

        ignore_folders = set(ignore_folders)
        files = _load_hpatches(
            hpatches_path,
            img_alteration,
            num_to_load,
            ignore_folders,
        )

        self.img_size = img_size
        self.force_min_max_resizing = force_min_max_resizing
        self.max_img_size = max_img_size
        self.min_img_size = min_img_size
        self.image_paths = files["image_paths"]
        self.warped_image_paths = files["warped_image_paths"]
        self.homographies = files["homography"]
        self.grayscale = grayscale

    def __len__(self):
        """
        Get the length of the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, index):
        """
        Get an item from the dataset. Note that one item in the
        dataset is one image PAIR, and the label is the homography.
        """
        original_image_path = self.image_paths[index]
        warped_image_path = self.warped_image_paths[index]
        homography = torch.tensor(self.homographies[index])

        # resize images
        # TODO(Pierre): Avoid loading entire image just to get the shape (we re-loed it later)
        image = cv2.imread(original_image_path)
        wimage = cv2.imread(warped_image_path)

        original_height, original_width = image.shape[:2]

        assert (self.max_img_size is None) or (self.min_img_size is None)
        if self.max_img_size:
            if (
                original_height > self.max_img_size[0]
                or original_width > self.max_img_size[1]
                or self.force_min_max_resizing
            ):
                r = min(
                    self.max_img_size[0] / original_height,
                    self.max_img_size[1] / original_width,
                )
                original_height = int(original_height * r)
                original_width = int(original_width * r)
        elif self.min_img_size:
            if (
                original_height < self.min_img_size[0]
                or original_width < self.min_img_size[1]
                or self.force_min_max_resizing
            ):
                r = max(
                    self.min_img_size[0] / original_height,
                    self.min_img_size[1] / original_width,
                )
                original_height = int(original_height * r)
                original_width = int(original_width * r)

        # keep original image sizes, ensuring that height and width are divisible by 8
        if self.img_size is None:
            img_height = original_height + (8 - original_height % 8) % 8
            img_width = original_width + (8 - original_width % 8) % 8
        # if argument is a float, scale by the factor
        elif type(self.img_size) == float:
            img_height = int(original_height * self.img_size)
            img_width = int(original_width * self.img_size)
            img_height += (8 - img_height % 8) % 8
            img_width += (8 - img_width % 8) % 8
        elif len(self.img_size) == 2:
            img_height = self.img_size[0]
            img_width = self.img_size[1]
        else:
            raise ValueError("img_size must be of type None or Float")

        original_image = load_image(
            original_image_path, img_height, img_width, as_gray=self.grayscale
        )
        warped_image = load_image(
            warped_image_path, img_height, img_width, as_gray=self.grayscale
        )

        homography = resize_homography(
            homography,
            image.shape,
            (img_height, img_width),
            wimage.shape,
        )

        return original_image, warped_image, homography
