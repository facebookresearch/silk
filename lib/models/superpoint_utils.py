# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Utils functions for the SuperPoint model.
"""

import cv2
import torch
from silk.cv.homography import HomographicSampler
from silk.models.superpoint import SuperPoint


def warp_points(points: torch.Tensor, homography: torch.Tensor):
    """
    Warp the points with the given homography matrix.

    Args:
        points (tensor): the predicted points for an image in the format
            3 x num_pred_points, with a row of x coords, row of y coords, row of probs
        homography (tensor): the 3 x 3 homography matrix connecting two images

    Returns:
        cartesian_points (tensor): the points warped by the homography in the shape
            3 x num_pred_points, with a row of x coords, row of y coords, row of probs
    """
    num_points = points.shape[1]

    # convert to 2 x num_pred_points array with x coords row, y coords row
    points1 = points[:2]

    # add row of 1's for multiplication with the homography
    points1 = torch.vstack((points1, torch.ones(1, num_points, device=points1.device)))

    # calculate homogeneous coordinates by multiplying by the homography
    homogeneous_points = torch.mm(homography, points1)

    # get back to cartesian coordinates by dividing, (optional : KEEPING PROBS AS THIRD ROW)
    cartesian_points = torch.vstack(
        (
            homogeneous_points[0] / homogeneous_points[2],
            homogeneous_points[1] / homogeneous_points[2],
        )
    )
    if points.shape[0] > 2:
        cartesian_points = torch.vstack((cartesian_points, points[2]))

    return cartesian_points


def filter_points(points, img_shape):
    """
    Keep only the points whose coordinates are still inside the
    dimensions of img_shape.

    Args:
        points (tensor): the predicted points for an image
        img_shape (tensor): the image size

    Returns:
        points_to_keep (tensor): the points that are still inside
            the boundaries of the img_shape
    """
    # we want to get rid of any points that are not in the bounds of the second image
    # the mask will be a tensor of shape [num_points_to_keep]
    mask = (
        # ensure x coordinates are greater than 0 and less than image width
        (points[0] >= 0)
        & (points[0] < img_shape[1])
        # ensure y coordinates are greater than 0 and less than image height
        & (points[1] >= 0)
        & (points[1] < img_shape[0])
    )

    # apply the mask
    points_to_keep = points[:, mask]

    return points_to_keep, mask


def keep_true_points(
    points: torch.Tensor,
    homography: torch.Tensor,
    img_shape: torch.Tensor,
):
    """
    Keep only the points whose coordinates when warped by the
    homography are still inside the img_shape dimensions.

    Args:
        points (tensor): the predicted points for an image
        homography (tensor): the 3 x 3 homography matrix connecting
            two images
        img_shape (tensor): the image size (img_height, img_width)

    Returns:
        points_to_keep (tensor): the points that are still inside
            the boundaries of the img_shape after the homography is applied
    """

    # first warp the points by the homography
    warped_points = warp_points(points, homography)

    # we want to get rid of any points that are not in the bounds of the second image
    # the mask will be a tensor of shape [num_points_to_keep]
    points_to_keep, mask = filter_points(warped_points, img_shape)

    # need to warp by the inverse homography to get the original coordinates back
    points_to_keep = points[:, mask]

    return points_to_keep, mask


def select_k_best_points(points, k):
    """
    Select the k most probable points.

    Args:
        points (tensor): a 3 x num_pred_points tensor where the third row is the
            probabilities for each point
        k (int): the number of points to keep

    Returns:
        points (tensor): a 3 x k tensor with only the k best points selected in
            sorted order of the probabilities
    """
    points = points.T

    sorted_indices = torch.argsort(points[:, 2], descending=True)
    sorted_prob = points[sorted_indices]
    start = min(k, points.shape[0])

    sorted_points = sorted_prob[:start]
    sorted_indices = sorted_indices[:start]

    return sorted_points.T, sorted_indices


def max_image_size_downsampled_shape(h, w, max_h, max_w):
    downsampled = False
    if h > max_h or w > max_w:
        hr = max_h / h
        wr = max_w / w

        r = min(hr, wr)

        h = int(h * r)
        w = int(w * r)

        downsampled = True

    return h, w, downsampled


def load_image(file_path, H=None, W=None, max_H=None, max_W=None, as_gray=True):
    """
    Helper function to load image from file path and reshape for model input.
    NOTE: Loads the image in grayscale (with 1 input channel).

    Args:
        file_path (str): the image location
        H (int): the reshaped image height
        W (int): the reshaped image width
        max_H (int): maximum height of the loaded image (ignored if H is specified)
        max_W (int): maximum width of the loaded image (ignored if W is specified)

    Returns:
        input_image (tensor): a tensor of shape (1, H, W) for input into the
            SuperPoint model
    """
    import skimage.io as io  # this loads matplotlib, so we put it here instead of globally

    input_image = io.imread(file_path, as_gray=as_gray)

    if (W is not None) and (H is not None):
        input_image = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_AREA)
    elif (max_W is not None) or (max_H is not None):
        max_H = input_image.shape[0] if max_H is None else max_H
        max_W = input_image.shape[1] if max_W is None else max_W

        nh, nw, downsampled = max_image_size_downsampled_shape(
            input_image.shape[0],
            input_image.shape[1],
            max_H,
            max_W,
        )

        if downsampled:
            input_image = cv2.resize(
                input_image,
                (nw, nh),
                interpolation=cv2.INTER_AREA,
            )

    if not as_gray:
        return input_image.transpose((2, 0, 1))

    input_image = input_image.astype("float32")
    input_image = torch.from_numpy(input_image)
    if as_gray:
        input_image = input_image.view(1, input_image.shape[-2], input_image.shape[-1])

    return input_image


def _process_output_new(model, images, sparse=True):
    if sparse:
        if isinstance(model, SuperPoint):
            outputs = ("positions", "sparse_descriptors")
        else:
            outputs = ("sparse_positions", "sparse_descriptors")
    else:
        outputs = ("dense_positions", "dense_descriptors")

    positions, descriptors = model.forward_flow(
        images,
        outputs=outputs,
    )

    assert len(positions) == 1
    assert len(descriptors) == 1

    positions = positions[0]
    descriptors = descriptors[0]

    return positions, descriptors


def get_dense_positions(h, w, device, batch_size=None):
    dense_positions = HomographicSampler._create_meshgrid(
        w,
        h,
        device=device,
        normalized=False,
    )
    dense_positions = dense_positions.permute(0, 2, 1, 3)
    dense_positions = dense_positions.reshape(-1, 2)
    dense_positions = dense_positions.unsqueeze(0)

    if batch_size is not None:
        dense_positions = dense_positions.expand(batch_size, -1, -1)

    return dense_positions
