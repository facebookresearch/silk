# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import pickle as pkl
from typing import Any, Iterable, Tuple, Union

import h5py
import skimage.io as io
import torch.utils.data


class Frames(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        pickle_dir_list: Union[str, Iterable[str]],
        image_list_file_name: str = "imgs_MD.p",
        depth_list_file_name: str = "targets_MD.p",
    ) -> None:
        super().__init__()

        self.root = root

        pickle_dir_list = (
            [pickle_dir_list] if isinstance(pickle_dir_list, str) else pickle_dir_list
        )

        self._paths_list = []
        for pickle_dir in pickle_dir_list:
            image_list_file = os.path.join(pickle_dir, image_list_file_name)
            depth_list_file = os.path.join(pickle_dir, depth_list_file_name)

            with open(image_list_file, "br") as f:
                image_list = pkl.load(f)

            with open(depth_list_file, "br") as f:
                depth_list = pkl.load(f)

            self._paths_list.extend(zip(image_list, depth_list))

        self._paths_list = sorted(self._paths_list)

    def __len__(self):
        return len(self._paths_list)

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        image_path, depth_path = self._paths_list[index]

        image_path = os.path.join(self.root, image_path)
        depth_path = os.path.join(self.root, depth_path)

        image = io.imread(image_path)
        with h5py.File(depth_path) as f:
            depth = f["depth"][:]  # [:] converts h5py dataset to numpy

        return image, depth
