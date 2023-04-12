# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Utils functions for the magicpoint model.
"""

import math
from typing import Iterable, Tuple, Union

import torch
import torch.nn.functional as F


def logits_to_prob(logits: torch.Tensor, channel_dim: int = 1) -> torch.Tensor:
    """
    Get the probabilities given the logits output.

    The probabilities are the chances that each of the points
    in the image is a corner.

    Args:
        logits (tensor): the logits output from the model

    Returns:
        prob (tensor): the probabilities tensor (not reshaped) with shape
            (batch_size, 65, H/8, W/8) for cell_size = 8
    """

    # the logits tensor size is batch_size, 65, img_height, img_width
    # 65 is 8 * 8 + 1, where cell_size = 8
    channels = logits.shape[channel_dim]
    if channels == 1:
        prob = torch.sigmoid(logits)
    else:
        prob = torch.softmax(logits, dim=channel_dim)
    return prob


def depth_to_space(
    prob: torch.Tensor,
    cell_size: int = 8,
    channel_dim: int = 1,
) -> torch.Tensor:
    """
    Reorganizes output shape to be of size batch_size x img_height x img_width.

    Converts the structure of the outputs from a tensor consisting
    of a series of cells of size cell_size x cell_size, where
    cells correspond to groups of pixels in the image, to a tensor
    where the shape corresponds exactly to the shape of the image.

    Args:
        prob (tensor): the tensor comprising the corner probabilities for each
            pixel in a "depth" format as a tensor of cell_size x cell_size cells
        cell_size (int): the size of each cell (default is always 8)

    Returns:
        image_probs (tensor): the reshaped tensor where each image in the batch
            is shaped in a tensor of size 1 x H x W
    """
    if cell_size > 1:
        assert prob.shape[channel_dim] == cell_size * cell_size + 1

        # remove the last (dustbin) cell from the list
        prob, _ = prob.split(cell_size * cell_size, dim=channel_dim)

        # change the dimensions to get an output shape of (batch_size, H, W)
        image_probs = F.pixel_shuffle(prob, cell_size)
    else:
        assert prob.shape[channel_dim] == 1
        image_probs = prob

    return image_probs


def prob_map_to_points_map(
    prob_map: torch.Tensor,
    prob_thresh: float = 0.015,
    nms_dist: int = 4,
    border_dist: int = 4,
    use_fast_nms: bool = True,
    top_k: int = None,
):
    prob_map = remove_border_points(prob_map, border_dist=border_dist)

    prob_map = prob_map.squeeze(dim=1)

    prob_thresh = torch.tensor(prob_thresh, device=prob_map.device)
    prob_thresh = prob_thresh.unsqueeze(0)

    if use_fast_nms:
        # add missing channel
        prob_map = prob_map.unsqueeze(1)
        nms = fast_nms(prob_map, nms_dist=nms_dist)
        # remove added channel
        prob_map = nms.squeeze(1)
    else:
        # Original Implementation
        # NMS only runs one image at a time, so go through each elem in the batch
        prob_map = torch.stack(
            [original_nms(image, nms_dist=nms_dist) for image in prob_map]
        )

    if top_k:
        if top_k >= prob_map.shape[-1] * prob_map.shape[-2]:
            top_k_threshold = torch.zeros_like(prob_thresh)
        else:
            # infer top k threshold
            top_k = torch.tensor(top_k, device=prob_map.device)
            reshaped_prob_map = prob_map.reshape(prob_map.shape[0], -1)

            top_k_percentile = (
                reshaped_prob_map[0].size()[0] - top_k - 1
            ) / reshaped_prob_map[0].size()[0]

            top_k_threshold = reshaped_prob_map.quantile(
                top_k_percentile,
                dim=1,
                interpolation="midpoint",
            )
        prob_thresh = torch.minimum(top_k_threshold, prob_thresh)
        prob_thresh = prob_thresh.unsqueeze(-1).unsqueeze(-1)

    # only take points with probability above the probability threshold
    prob_map = torch.where(
        prob_map > prob_thresh,
        prob_map,
        torch.tensor(0.0, device=prob_map.device),
    )

    return prob_map  # batch_output


def remove_border_points(image_nms: torch.Tensor, border_dist: int = 4) -> torch.Tensor:
    """
    Remove predicted points within border_dist pixels of the image border.

    Args:
        image_nms (tensor): the output of the nms function, a tensor of shape
            (img_height, img_width) with corner probability values at each pixel location
        border_dist (int): the distance from the border to remove points

    Returns:
        image_nms (tensor): the image with all probability values equal to 0.0
            for pixel locations within border_dist of the image border
    """
    if border_dist > 0:
        # left columns
        image_nms[..., :, :border_dist] = 0.0

        # right columns
        image_nms[..., :, -border_dist:] = 0.0

        # top rows
        image_nms[..., :border_dist, :] = 0.0

        # bottom rows
        image_nms[..., -border_dist:, :] = 0.0

    return image_nms


def original_nms(image_probs: torch.Tensor, nms_dist: int = 4) -> torch.Tensor:
    """
    Run non-maximum suppression on the predicted corner points.

    NMS removes nearby points within a distance of nms_dist from the point
    with the highest probability value in that region. The algorithm is as
    follows:
        1. Order the predicted corner points from highest to lowest probability.
        2. Set up the input tensor with probability values for each pixel location
        to have padding of size nms_dist so points near the border can be suppressed.
        3. Go through each point in the list from step 1. If the point has not already
        been suppressed in the probability value tensor with padding from step 2
        (meaning the probability value has been changed to 0.0), suppress all
        points within nms_dist from that point by changing their probability values
        to 0.0. Keep the probability value of the current point as is.
        3. At the end, remove the padding from the tensor. Thus, the output is
        a tensor of size (img_height, img_width) with probability values for the remaining
        predicted corner pixels (those not suppressed) and 0.0 for non-corner pixels.

    Args:
        image_probs (tensor): a tensor of size (img_height, img_width) where each
            pixel location has value equal to the probability value of it being a corner,
            as predicted by the model.
        nms_dist (int): the minimum distance between two predicted corners after NMS

    Returns:
        image_probs_nms (tensor): a tensor of size (img_height, img_width) where each
            pixel location has value equal to the probability value of it being a corner,
            after running the non-maximum suppression algorithm. Thus, no two predicted
            corners will be within nms_dist pixels of each other.
    """
    # each elem in corners_list is (row, col) for predicted corners in image_probs
    corners_list = torch.nonzero(image_probs)

    # list of the probability values in the same order as the list of their locations
    list_of_prob = image_probs[torch.nonzero(image_probs, as_tuple=True)]

    # concatenate the probability values with their locations (prob, row, col) for each
    corners_list_with_prob = torch.cat(
        (list_of_prob.unsqueeze(dim=1), corners_list), dim=1
    )

    # sort the list of probability values with most confident corners first
    prob_indices = torch.argsort(list_of_prob, dim=0, descending=True)

    # sort the list of corner locations according to the order of the indices
    sorted_corners_list = corners_list_with_prob[prob_indices]

    # pad the border of the grid with zeros, so that we can NMS points near the border
    padding = (nms_dist, nms_dist, nms_dist, nms_dist)
    padded_image_probs = F.pad(image_probs, padding, "constant", 0)

    # go through each element in the sorted list of corners
    # suppress surrounding points by converting their probabilities to 0.0
    # TODO: Benchmark this to see if this loop is a bottleneck
    for prob, row, col in sorted_corners_list:
        row = int(row)
        col = int(col)

        # if the point hasn't already been suppressed
        if padded_image_probs[row + nms_dist][col + nms_dist] != 0.0:
            # suppress all points in the (2*nms_dist, 2*nms_dist) square around the point
            padded_image_probs[
                row : row + 2 * nms_dist + 1,  # noqa: E203
                col : col + 2 * nms_dist + 1,  # noqa: E203
            ] = 0.0

            # then add back in the one point not suppressed
            padded_image_probs[row + nms_dist][col + nms_dist] = prob

    # remove the image padding to get the actual image size
    image_probs_nms = padded_image_probs[nms_dist:-nms_dist, nms_dist:-nms_dist]

    return image_probs_nms


def fast_nms(
    image_probs: torch.Tensor,
    nms_dist: int = 4,
    max_iter: int = -1,
    min_value: float = 0.0,
) -> torch.Tensor:
    """Produce same result as `original_nms` (see documentation).
    The process is slightly different :
      1. Find any local maximum (and count them).
      2. Suppress their neighbors (by setting them to 0).
      3. Repeat 1. and 2. until the number of local maximum stays the same.

    Performance
    -----------
    The original implementation takes about 2-4 seconds on a batch of 32 images of resolution 240x320.
    This fast implementation takes about ~90ms on the same input.

    Parameters
    ----------
    image_probs : torch.Tensor
        Tensor of shape BxCxHxW.
    nms_dist : int, optional
        The minimum distance between two predicted corners after NMS, by default 4
    max_iter : int, optional
        Maximum number of iteration, by default -1.
        Setting this number to a positive integer guarantees execution speed, but not correctness (i.e. good approximation).
    min_value : float
        Minimum value used for suppression.

    Returns
    -------
    torch.Tensor
        Tensor of shape BxCxHxW containing NMS suppressed input.
    """
    if nms_dist == 0:
        return image_probs

    ks = 2 * nms_dist + 1
    midpoint = (ks * ks) // 2
    count = None
    batch_size = image_probs.shape[0]

    i = 0
    while True:
        if i == max_iter:
            break

        # get neighbor probs in last dimension
        unfold_image_probs = F.unfold(
            image_probs,
            kernel_size=(ks, ks),
            dilation=1,
            padding=nms_dist,
            stride=1,
        )
        unfold_image_probs = unfold_image_probs.reshape(
            batch_size,
            ks * ks,
            image_probs.shape[-2],
            image_probs.shape[-1],
        )

        # check if middle point is local maximum
        max_idx = unfold_image_probs.argmax(dim=1, keepdim=True)
        mask = max_idx == midpoint

        # count all local maximum that are found
        new_count = mask.sum()

        # we stop if we din't not find any additional local maximum
        if new_count == count:
            break
        count = new_count

        # propagate local-maximum information to local neighbors (to suppress them)
        mask = mask.float()
        mask = mask.expand(-1, ks * ks, -1, -1)
        mask = mask.view(batch_size, ks * ks, -1)
        mask = mask.contiguous()
        mask[:, midpoint] = 0.0  # make sure we don't suppress the local maximum itself
        fold_ = F.fold(
            mask,
            output_size=image_probs.shape[-2:],
            kernel_size=(ks, ks),
            dilation=1,
            padding=nms_dist,
            stride=1,
        )

        # suppress all points who have a local maximum in their neighboorhood
        image_probs = image_probs.masked_fill(fold_ > 0.0, min_value)

        i += 1

    return image_probs


def space_to_depth(prob: torch.Tensor, cell_size: int = 8) -> torch.Tensor:
    """
    Reshapes probabilities cell_size x cell_size cells to depth. Add dustbin
    corresponding to remaining probability.

    Args:
        prob (tensor): pobability tensor of shape (batch x 1 x height x width).
            The sum of probabilities per cell could be higher than one.
        cell_size (int): size of cell, by default 8

    Returns:
        prob (tensor): reshaped probability tensor with size
            (batch x (cell_size^2 + 1) x (height/cell_size) x (width/cell_size)).
            The sum of probabilities per cell could be higher than one.
    """
    prob = torch.nn.functional.pixel_unshuffle(prob, cell_size)
    prob_sum = torch.sum(prob, dim=1, keepdim=True)
    dustbin = torch.clamp_min(
        1.0 - prob_sum, torch.tensor(0.0, device=prob_sum.device)
    )  # make sure dustbin is not negative when `prob_sum[...]` > 1.
    prob = torch.concat((prob, dustbin), dim=1)
    return prob


def positions_to_label_map(
    positions: Union[torch.Tensor, Iterable[torch.Tensor]],
    image_or_shape: Union[torch.Tensor, Tuple[int, int]],
):
    """
    Create label map of same size as image where provided positions
    coordinates are set to 1., while everything else is set to 0.

    Args:
        positions (tensor): tensor of shape N x D or B x N x D or Iterable[Ni x D] (with D >= 2) containing the image
            coordinates (first two elements in the D dimension) that contain keypoints.
        image_or_shape (tensor or tuple): image tensor of shape H x W x 1 containing the keypoints or a tuple (H, W).

    Returns:
        label_map (tensor): label map of same size as image whose position
            coordinates are set to 1.
    """
    # run with added batch dimension if not present
    if isinstance(positions, torch.Tensor) and positions.ndim == 2:
        return positions_to_label_map((positions,), image_or_shape)[0]

    batch_size = len(positions)
    if isinstance(image_or_shape, torch.Tensor):
        shape = (batch_size,) + image_or_shape.shape
    else:
        shape = (batch_size,) + image_or_shape + (1,)

    device = positions[0].device

    label_map = torch.zeros(shape, dtype=torch.bool, device=device)

    batch_i = torch.cat(
        [torch.full((len(p),), i, device=device) for i, p in enumerate(positions)],
        dim=0,
    )

    if isinstance(positions, torch.Tensor):
        positions = positions.view(-1, positions.shape[-1])
    else:
        positions = torch.cat(positions, dim=0)

    # positions might also contain additional data like confidence
    positions = positions[:, :2]
    positions = torch.floor(positions).long()

    image_shape = torch.tensor(
        [label_map.shape[-3:-1]],
        device=device,
    )

    mask = torch.logical_and(positions >= 0, positions < image_shape).all(dim=1)
    positions = positions[mask]
    batch_i = batch_i[mask]

    indices = (batch_i,) + (positions[:, 0], positions[:, 1])
    label_map[indices] = 1

    return label_map


def float_positions_to_int(
    y: float,
    x: float,
    shape: Tuple[int, int],
) -> Union[Tuple[int, int], Tuple[None, None]]:
    """Convert floating positions y, x to integer positions

    Parameters
    ----------
    y : float
        Float value of y coordinate.
    x : float
        Float value of x coordinate.
    shape : Tuple[int, int]
        Shape of tensor those coordinates are used for.

    Returns
    -------
    Union[Tuple[int, int], Tuple[None, None]]
        Converted coordinates. Returns (None, None) if the coordinates are out of bound.
    """
    y, x = math.floor(y), math.floor(x)
    if 0 <= y < shape[0] and 0 <= x < shape[1]:
        return y, x
    return None, None


def prob_map_to_positions_with_prob(
    prob_map: torch.Tensor,
    threshold: float = 0.0,
) -> Tuple[torch.Tensor]:
    """Convert probability map to positions with probability associated with each position.

    Parameters
    ----------
    prob_map : torch.Tensor
        Probability map. Tensor of size N x 1 x H x W.
    threshold : float, optional
        Threshold used to discard positions with low probability, by default 0.0

    Returns
    -------
    Tuple[Tensor]
        Tuple of positions (with probability) tensors of size P x 3 (x, y and prob).
    """
    prob_map = prob_map.squeeze(dim=1)
    positions = tuple(
        torch.nonzero(prob_map[i] > threshold).float() + 0.5
        for i in range(prob_map.shape[0])
    )
    prob = tuple(
        prob_map[i][torch.nonzero(prob_map[i] > threshold, as_tuple=True)][:, None]
        for i in range(prob_map.shape[0])
    )
    positions_with_prob = tuple(
        torch.cat((pos, prob), dim=1) for pos, prob in zip(positions, prob)
    )
    return positions_with_prob
