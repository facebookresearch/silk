# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

import skimage.io as io
import torch
from loguru import logger

DEVICE = "cuda"


def load_image(path, as_gray=True, device=DEVICE):
    image = io.imread(path, as_gray=as_gray)
    dtype = torch.float32 if as_gray else torch.uint8
    image = torch.tensor(image, device=device, dtype=dtype)
    if as_gray:
        image = image.unsqueeze(0)  # add channel dim
    else:
        image = image.permute(2, 0, 1)
    image = image.unsqueeze(0)  # add batch and channel dimensions
    return image


def canonical_file_path(path):
    path = path.strip()
    path = os.path.abspath(path)
    return path


def is_in(set):
    def fn(x):
        return x in set

    return fn


def load_data(*paths, ensure_type=None, raise_on_type_error=True):
    data = []
    for path in paths:
        dat = torch.load(path)
        if (ensure_type is not None) and (dat["type"] != ensure_type):
            error = f"data type expected to be '{ensure_type}', found '{dat['type']}' instead"
            if raise_on_type_error:
                raise RuntimeError(error)
            else:
                logger.warning(error)

        data.append(dat)
    if len(data) == 1:
        return data[0]
    return tuple(data)


def shape_to_str(tensor):
    return " x ".join(str(x) for x in tensor.shape)
