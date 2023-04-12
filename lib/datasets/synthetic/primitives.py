# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

""" Module used to generate geometrical synthetic shapes """

# Modified code taken from here : https://github.com/eric-yyjau/pytorch-superpoint/blob/master/datasets/synthetic_shapes.py
# Modifications were done to control the determinism of the primitives.
# TODO(Pierre) : Might be re-implemented using PyTorch/OpenGL in the future.

import math

import cv2 as cv
import numpy as np


def get_random_color(random_state, background_color):
    """Output a random scalar in grayscale with a least a small
    contrast with the background color"""
    color = random_state.randint(256)
    if abs(color - background_color) < 30:  # not enough contrast
        color = (color + 128) % 256
    return color


def get_different_color(random_state, previous_colors, min_dist=50, max_count=20):
    """Output a color that contrasts with the previous colors
    Parameters:
      previous_colors: np.array of the previous colors
      min_dist: the difference between the new color and
                the previous colors must be at least min_dist
      max_count: maximal number of iterations
    """
    color = random_state.randint(256)
    count = 0
    while np.any(np.abs(previous_colors - color) < min_dist) and count < max_count:
        count += 1
        color = random_state.randint(256)
    return color


def generate_background(
    random_state,
    size=(960, 1280),
    num_blobs=100,
    min_rad_ratio=0.01,
    max_rad_ratio=0.05,
    min_kernel_size=50,
    max_kernel_size=300,
):
    """Generate a customized background image
    Parameters:
      size: size of the image
      num_blobs: number of circles to draw
      min_rad_ratio: the radius of blobs is at least min_rad_size * max(size)
      max_rad_ratio: the radius of blobs is at most max_rad_size * max(size)
      min_kernel_size: minimal size of the kernel
      max_kernel_size: maximal size of the kernel
    """
    dim = max(size)
    img = random_state.randint(0, 255, size=size, dtype=np.uint8)
    cv.threshold(img, random_state.randint(256), 255, cv.THRESH_BINARY, img)
    background_color = int(np.mean(img))
    blobs = np.concatenate(
        [
            random_state.randint(0, size[1], size=(num_blobs, 1)),
            random_state.randint(0, size[0], size=(num_blobs, 1)),
        ],
        axis=1,
    )
    for i in range(num_blobs):
        col = get_random_color(random_state, background_color)
        cv.circle(
            img,
            (blobs[i][0], blobs[i][1]),
            random_state.randint(int(dim * min_rad_ratio), int(dim * max_rad_ratio)),
            col,
            -1,
        )
    kernel_size = random_state.randint(min_kernel_size, max_kernel_size)
    cv.blur(img, (kernel_size, kernel_size), img)
    return img


def generate_custom_background(
    random_state,
    size,
    background_color,
    num_blobs=3000,
    kernel_boundaries=(50, 100),
):
    """Generate a customized background to fill the shapes
    Parameters:
      background_color: average color of the background image
      num_blobs: number of circles to draw
      kernel_boundaries: interval of the possible sizes of the kernel
    """
    img = np.zeros(size, dtype=np.uint8)
    img = img + get_random_color(random_state, background_color)
    blobs = np.concatenate(
        [
            random_state.randint(0, size[1], size=(num_blobs, 1)),
            random_state.randint(0, size[0], size=(num_blobs, 1)),
        ],
        axis=1,
    )
    for i in range(num_blobs):
        col = get_random_color(random_state, background_color)
        cv.circle(img, (blobs[i][0], blobs[i][1]), random_state.randint(20), col, -1)
    kernel_size = random_state.randint(kernel_boundaries[0], kernel_boundaries[1])
    cv.blur(img, (kernel_size, kernel_size), img)
    return img


def final_blur(img, kernel_size=(5, 5)):
    """Apply a final Gaussian blur to the image
    Parameters:
      kernel_size: size of the kernel
    """
    cv.GaussianBlur(img, kernel_size, 0, img)


def ccw(A, B, C, dim):
    """Check if the points are listed in counter-clockwise order"""
    if dim == 2:  # only 2 dimensions
        return (C[:, 1] - A[:, 1]) * (B[:, 0] - A[:, 0]) > (B[:, 1] - A[:, 1]) * (
            C[:, 0] - A[:, 0]
        )
    else:  # dim should be equal to 3
        assert dim == 3
        return (C[:, 1, :] - A[:, 1, :]) * (B[:, 0, :] - A[:, 0, :]) > (
            B[:, 1, :] - A[:, 1, :]
        ) * (C[:, 0, :] - A[:, 0, :])


def intersect(A, B, C, D, dim):
    """Return true if line segments AB and CD intersect

    Implements this : https://blog.devgenius.io/clockwise-and-counterclockwise-line-segment-intersection-b3317839534e
    """
    return np.any(
        (ccw(A, C, D, dim) != ccw(B, C, D, dim))
        & (ccw(A, B, C, dim) != ccw(A, B, D, dim))
    )


def keep_points_inside(points, size):
    """Keep only the points whose coordinates are inside the dimensions of
    the image of size 'size'"""
    mask = (
        (points[:, 0] >= 0)
        & (points[:, 0] < size[1])
        & (points[:, 1] >= 0)
        & (points[:, 1] < size[0])
    )
    return points[mask, :]


def draw_lines(random_state, img, num_lines=10):
    """Draw random lines and output the positions of the endpoints
    Parameters:
      num_lines: maximal number of lines
    """
    num_lines = random_state.randint(1, num_lines)
    segments = np.empty((0, 4), dtype=int)
    points = np.empty((0, 2), dtype=int)
    background_color = int(np.mean(img))
    min_dim = min(img.shape)
    for _ in range(num_lines):
        x1 = random_state.randint(img.shape[1])
        y1 = random_state.randint(img.shape[0])
        p1 = np.array([[x1, y1]])
        x2 = random_state.randint(img.shape[1])
        y2 = random_state.randint(img.shape[0])
        p2 = np.array([[x2, y2]])
        # Check that there is no overlap
        if intersect(segments[:, 0:2], segments[:, 2:4], p1, p2, 2):
            continue
        segments = np.concatenate([segments, np.array([[x1, y1, x2, y2]])], axis=0)
        col = get_random_color(random_state, background_color)
        thickness = random_state.randint(min_dim * 0.01, min_dim * 0.02)
        cv.line(img, (x1, y1), (x2, y2), col, thickness)
        points = np.concatenate([points, np.array([[x1, y1], [x2, y2]])], axis=0)
    return points


def draw_polygon(random_state, img, max_sides=8):
    """Draw a polygon with a random number of corners
    and return the corner points
    Parameters:
      max_sides: maximal number of sides + 1
    """
    num_corners = random_state.randint(3, max_sides)
    min_dim = min(img.shape[0], img.shape[1])
    rad = max(random_state.rand() * min_dim / 2, min_dim / 10)
    x = random_state.randint(rad, img.shape[1] - rad)  # Center of a circle
    y = random_state.randint(rad, img.shape[0] - rad)

    # Sample num_corners points inside the circle
    slices = np.linspace(0, 2 * math.pi, num_corners + 1)
    angles = [
        slices[i] + random_state.rand() * (slices[i + 1] - slices[i])
        for i in range(num_corners)
    ]
    points = np.array(
        [
            [
                int(x + max(random_state.rand(), 0.4) * rad * math.cos(a)),
                int(y + max(random_state.rand(), 0.4) * rad * math.sin(a)),
            ]
            for a in angles
        ]
    )

    # Filter the points that are too close or that have an angle too flat
    norms = [
        np.linalg.norm(points[(i - 1) % num_corners, :] - points[i, :])
        for i in range(num_corners)
    ]
    mask = np.array(norms) > 0.01
    points = points[mask, :]
    num_corners = points.shape[0]
    corner_angles = [
        angle_between_vectors(
            points[(i - 1) % num_corners, :] - points[i, :],
            points[(i + 1) % num_corners, :] - points[i, :],
        )
        for i in range(num_corners)
    ]
    mask = np.array(corner_angles) < (2 * math.pi / 3)
    points = points[mask, :]
    num_corners = points.shape[0]
    if num_corners < 3:  # not enough corners
        return draw_polygon(random_state, img, max_sides)

    corners = points.reshape((-1, 1, 2))
    col = get_random_color(random_state, int(np.mean(img)))
    cv.fillPoly(img, [corners], col)
    return points


def overlap(center, rad, centers, rads):
    """Check that the circle with (center, rad)
    doesn't overlap with the other circles"""
    flag = False
    for i in range(len(rads)):
        if np.linalg.norm(center - centers[i]) + min(rad, rads[i]) < max(rad, rads[i]):
            flag = True
            break
    return flag


def angle_between_vectors(v1, v2):
    """Compute the angle (in rad) between the two vectors v1 and v2."""
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def draw_multiple_polygons(random_state, img, max_sides=8, num_polygons=30, **extra):
    """Draw multiple polygons with a random number of corners
    and return the corner points
    Parameters:
      max_sides: maximal number of sides + 1
      num_polygons: maximal number of polygons
    """
    segments = np.empty((0, 4), dtype=int)
    centers = []
    rads = []
    points = np.empty((0, 2), dtype=int)
    background_color = int(np.mean(img))
    for i in range(num_polygons):
        num_corners = random_state.randint(3, max_sides)
        min_dim = min(img.shape[0], img.shape[1])
        rad = max(random_state.rand() * min_dim / 2, min_dim / 10)
        x = random_state.randint(rad, img.shape[1] - rad)  # Center of a circle
        y = random_state.randint(rad, img.shape[0] - rad)

        # Sample num_corners points inside the circle
        slices = np.linspace(0, 2 * math.pi, num_corners + 1)
        angles = [
            slices[i] + random_state.rand() * (slices[i + 1] - slices[i])
            for i in range(num_corners)
        ]
        new_points = [
            [
                int(x + max(random_state.rand(), 0.4) * rad * math.cos(a)),
                int(y + max(random_state.rand(), 0.4) * rad * math.sin(a)),
            ]
            for a in angles
        ]
        new_points = np.array(new_points)

        # Filter the points that are too close or that have an angle too flat
        norms = [
            np.linalg.norm(new_points[(i - 1) % num_corners, :] - new_points[i, :])
            for i in range(num_corners)
        ]
        mask = np.array(norms) > 0.01
        new_points = new_points[mask, :]
        num_corners = new_points.shape[0]
        corner_angles = [
            angle_between_vectors(
                new_points[(i - 1) % num_corners, :] - new_points[i, :],
                new_points[(i + 1) % num_corners, :] - new_points[i, :],
            )
            for i in range(num_corners)
        ]
        mask = np.array(corner_angles) < (2 * math.pi / 3)
        new_points = new_points[mask, :]
        num_corners = new_points.shape[0]
        if num_corners < 3:  # not enough corners
            continue

        new_segments = np.zeros((1, 4, num_corners))
        new_segments[:, 0, :] = [new_points[i][0] for i in range(num_corners)]
        new_segments[:, 1, :] = [new_points[i][1] for i in range(num_corners)]
        new_segments[:, 2, :] = [
            new_points[(i + 1) % num_corners][0] for i in range(num_corners)
        ]
        new_segments[:, 3, :] = [
            new_points[(i + 1) % num_corners][1] for i in range(num_corners)
        ]

        # Check that the polygon will not overlap with pre-existing shapes
        if intersect(
            segments[:, 0:2, None],
            segments[:, 2:4, None],
            new_segments[:, 0:2, :],
            new_segments[:, 2:4, :],
            3,
        ) or overlap(np.array([x, y]), rad, centers, rads):
            continue
        centers.append(np.array([x, y]))
        rads.append(rad)
        new_segments = np.reshape(np.swapaxes(new_segments, 0, 2), (-1, 4))
        segments = np.concatenate([segments, new_segments], axis=0)

        # Color the polygon with a custom background
        corners = new_points.reshape((-1, 1, 2))
        mask = np.zeros(img.shape, np.uint8)
        custom_background = generate_custom_background(
            random_state, img.shape, background_color, **extra
        )
        cv.fillPoly(mask, [corners], 255)
        locs = np.where(mask != 0)
        img[locs[0], locs[1]] = custom_background[locs[0], locs[1]]
        points = np.concatenate([points, new_points], axis=0)
    return points


def draw_ellipses(random_state, img, num_ellipses=20):
    """Draw several ellipses
    Parameters:
      num_ellipses: maximal number of ellipses
    """
    centers = np.empty((0, 2), dtype=int)
    rads = np.empty((0, 1), dtype=int)
    min_dim = min(img.shape[0], img.shape[1]) / 4
    background_color = int(np.mean(img))
    for _ in range(num_ellipses):
        ax = int(max(random_state.rand() * min_dim, min_dim / 5))
        ay = int(max(random_state.rand() * min_dim, min_dim / 5))
        max_rad = max(ax, ay)
        x = random_state.randint(max_rad, img.shape[1] - max_rad)  # center
        y = random_state.randint(max_rad, img.shape[0] - max_rad)
        new_center = np.array([[x, y]])

        # Check that the ellipsis will not overlap with pre-existing shapes
        diff = centers - new_center
        if np.any(max_rad > (np.sqrt(np.sum(diff * diff, axis=1)) - rads)):
            continue
        centers = np.concatenate([centers, new_center], axis=0)
        rads = np.concatenate([rads, np.array([[max_rad]])], axis=0)

        col = get_random_color(random_state, background_color)
        angle = random_state.rand() * 90
        cv.ellipse(img, (x, y), (ax, ay), angle, 0, 360, col, -1)
    return np.empty((0, 2), dtype=int)


def draw_star(random_state, img, num_branches=6):
    """Draw a star and output the interest points
    Parameters:
      num_branches: number of branches of the star
    """
    num_branches = random_state.randint(3, num_branches)
    min_dim = min(img.shape[0], img.shape[1])
    thickness = random_state.randint(min_dim * 0.01, min_dim * 0.02)
    rad = max(random_state.rand() * min_dim / 2, min_dim / 5)
    # select the center of a circle
    x = random_state.randint(rad, img.shape[1] - rad)
    y = random_state.randint(rad, img.shape[0] - rad)
    # Sample num_branches points inside the circle
    slices = np.linspace(0, 2 * math.pi, num_branches + 1)
    angles = [
        slices[i] + random_state.rand() * (slices[i + 1] - slices[i])
        for i in range(num_branches)
    ]
    points = np.array(
        [
            [
                int(x + max(random_state.rand(), 0.3) * rad * math.cos(a)),
                int(y + max(random_state.rand(), 0.3) * rad * math.sin(a)),
            ]
            for a in angles
        ]
    )
    points = np.concatenate(([[x, y]], points), axis=0)
    background_color = int(np.mean(img))
    for i in range(1, num_branches + 1):
        col = get_random_color(random_state, background_color)
        cv.line(
            img,
            (points[0][0], points[0][1]),
            (points[i][0], points[i][1]),
            col,
            thickness,
        )
    return points


def draw_checkerboard(
    random_state, img, max_rows=7, max_cols=7, transform_params=(0.05, 0.15)
):
    """Draw a checkerboard and output the interest points
    Parameters:
      max_rows: maximal number of rows + 1
      max_cols: maximal number of cols + 1
      transform_params: set the range of the parameters of the transformations"""
    background_color = int(np.mean(img))
    # Create the grid
    rows = random_state.randint(3, max_rows)  # number of rows
    cols = random_state.randint(3, max_cols)  # number of cols
    s = min((img.shape[1] - 1) // cols, (img.shape[0] - 1) // rows)  # size of a cell
    x_coord = np.tile(range(cols + 1), rows + 1).reshape(((rows + 1) * (cols + 1), 1))
    y_coord = np.repeat(range(rows + 1), cols + 1).reshape(((rows + 1) * (cols + 1), 1))
    points = s * np.concatenate([x_coord, y_coord], axis=1)

    # Warp the grid using an affine transformation and an homography
    # The parameters of the transformations are constrained
    # to get transformations not too far-fetched
    alpha_affine = np.max(img.shape) * (
        transform_params[0] + random_state.rand() * transform_params[1]
    )
    center_square = np.float32(img.shape) // 2
    min_dim = min(img.shape)
    square_size = min_dim // 3
    pts1 = np.float32(
        [
            center_square + square_size,
            [center_square[0] + square_size, center_square[1] - square_size],
            center_square - square_size,
            [center_square[0] - square_size, center_square[1] + square_size],
        ]
    )
    pts2 = pts1 + random_state.uniform(
        -alpha_affine, alpha_affine, size=pts1.shape
    ).astype(np.float32)
    affine_transform = cv.getAffineTransform(pts1[:3], pts2[:3])
    pts2 = pts1 + random_state.uniform(
        -alpha_affine / 2, alpha_affine / 2, size=pts1.shape
    ).astype(np.float32)
    perspective_transform = cv.getPerspectiveTransform(pts1, pts2)

    # Apply the affine transformation
    points = np.transpose(
        np.concatenate((points, np.ones(((rows + 1) * (cols + 1), 1))), axis=1)
    )
    warped_points = np.transpose(np.dot(affine_transform, points))

    # Apply the homography
    warped_col0 = np.add(
        np.sum(np.multiply(warped_points, perspective_transform[0, :2]), axis=1),
        perspective_transform[0, 2],
    )
    warped_col1 = np.add(
        np.sum(np.multiply(warped_points, perspective_transform[1, :2]), axis=1),
        perspective_transform[1, 2],
    )
    warped_col2 = np.add(
        np.sum(np.multiply(warped_points, perspective_transform[2, :2]), axis=1),
        perspective_transform[2, 2],
    )
    warped_col0 = np.divide(warped_col0, warped_col2)
    warped_col1 = np.divide(warped_col1, warped_col2)
    warped_points = np.concatenate([warped_col0[:, None], warped_col1[:, None]], axis=1)
    warped_points = warped_points.astype(int)

    # Fill the rectangles
    colors = np.zeros((rows * cols,), np.int32)
    for i in range(rows):
        for j in range(cols):
            # Get a color that contrast with the neighboring cells
            if i == 0 and j == 0:
                col = get_random_color(random_state, background_color)
            else:
                neighboring_colors = []
                if i != 0:
                    neighboring_colors.append(colors[(i - 1) * cols + j])
                if j != 0:
                    neighboring_colors.append(colors[i * cols + j - 1])
                col = get_different_color(random_state, np.array(neighboring_colors))
            colors[i * cols + j] = col
            # Fill the cell
            cv.fillConvexPoly(
                img,
                np.array(
                    [
                        (
                            warped_points[i * (cols + 1) + j, 0],
                            warped_points[i * (cols + 1) + j, 1],
                        ),
                        (
                            warped_points[i * (cols + 1) + j + 1, 0],
                            warped_points[i * (cols + 1) + j + 1, 1],
                        ),
                        (
                            warped_points[(i + 1) * (cols + 1) + j + 1, 0],
                            warped_points[(i + 1) * (cols + 1) + j + 1, 1],
                        ),
                        (
                            warped_points[(i + 1) * (cols + 1) + j, 0],
                            warped_points[(i + 1) * (cols + 1) + j, 1],
                        ),
                    ]
                ),
                col,
            )

    # Draw lines on the boundaries of the board at random
    num_rows = random_state.randint(2, rows + 2)
    num_cols = random_state.randint(2, cols + 2)
    thickness = random_state.randint(min_dim * 0.01, min_dim * 0.015)
    for _ in range(num_rows):
        row_idx = random_state.randint(rows + 1)
        col_idx1 = random_state.randint(cols + 1)
        col_idx2 = random_state.randint(cols + 1)
        col = get_random_color(random_state, background_color)
        cv.line(
            img,
            (
                warped_points[row_idx * (cols + 1) + col_idx1, 0],
                warped_points[row_idx * (cols + 1) + col_idx1, 1],
            ),
            (
                warped_points[row_idx * (cols + 1) + col_idx2, 0],
                warped_points[row_idx * (cols + 1) + col_idx2, 1],
            ),
            col,
            thickness,
        )
    for _ in range(num_cols):
        col_idx = random_state.randint(cols + 1)
        row_idx1 = random_state.randint(rows + 1)
        row_idx2 = random_state.randint(rows + 1)
        col = get_random_color(random_state, background_color)
        cv.line(
            img,
            (
                warped_points[row_idx1 * (cols + 1) + col_idx, 0],
                warped_points[row_idx1 * (cols + 1) + col_idx, 1],
            ),
            (
                warped_points[row_idx2 * (cols + 1) + col_idx, 0],
                warped_points[row_idx2 * (cols + 1) + col_idx, 1],
            ),
            col,
            thickness,
        )

    # Keep only the points inside the image
    points = keep_points_inside(warped_points, img.shape[:2])
    return points


def draw_stripes(
    random_state,
    img,
    max_num_cols=13,
    min_width_ratio=0.04,
    transform_params=(0.05, 0.15),
):
    """Draw stripes in a distorted rectangle and output the interest points
    Parameters:
      max_num_cols: maximal number of stripes to be drawn
      min_width_ratio: the minimal width of a stripe is
                       min_width_ratio * smallest dimension of the image
      transform_params: set the range of the parameters of the transformations
    """
    background_color = int(np.mean(img))
    # Create the grid
    board_size = (
        int(img.shape[0] * (1 + random_state.rand())),
        int(img.shape[1] * (1 + random_state.rand())),
    )
    col = random_state.randint(5, max_num_cols)  # number of cols
    cols = np.concatenate(
        [
            board_size[1] * random_state.rand(col - 1),
            np.array([0, board_size[1] - 1]),
        ],
        axis=0,
    )
    cols = np.unique(cols.astype(int))
    # Remove the indices that are too close
    min_dim = min(img.shape)
    min_width = min_dim * min_width_ratio
    cols = cols[
        (
            np.concatenate([cols[1:], np.array([board_size[1] + min_width])], axis=0)
            - cols
        )
        >= min_width
    ]
    col = cols.shape[0] - 1  # update the number of cols
    cols = np.reshape(cols, (col + 1, 1))
    cols1 = np.concatenate([cols, np.zeros((col + 1, 1), np.int32)], axis=1)
    cols2 = np.concatenate(
        [cols, (board_size[0] - 1) * np.ones((col + 1, 1), np.int32)], axis=1
    )
    points = np.concatenate([cols1, cols2], axis=0)

    # Warp the grid using an affine transformation and an homography
    # The parameters of the transformations are constrained
    # to get transformations not too far-fetched
    # Prepare the matrices
    alpha_affine = np.max(img.shape) * (
        transform_params[0] + random_state.rand() * transform_params[1]
    )
    center_square = np.float32(img.shape) // 2
    square_size = min(img.shape) // 3
    pts1 = np.float32(
        [
            center_square + square_size,
            [center_square[0] + square_size, center_square[1] - square_size],
            center_square - square_size,
            [center_square[0] - square_size, center_square[1] + square_size],
        ]
    )
    pts2 = pts1 + random_state.uniform(
        -alpha_affine, alpha_affine, size=pts1.shape
    ).astype(np.float32)
    affine_transform = cv.getAffineTransform(pts1[:3], pts2[:3])
    pts2 = pts1 + random_state.uniform(
        -alpha_affine / 2, alpha_affine / 2, size=pts1.shape
    ).astype(np.float32)
    perspective_transform = cv.getPerspectiveTransform(pts1, pts2)

    # Apply the affine transformation
    points = np.transpose(np.concatenate((points, np.ones((2 * (col + 1), 1))), axis=1))
    warped_points = np.transpose(np.dot(affine_transform, points))

    # Apply the homography
    warped_col0 = np.add(
        np.sum(np.multiply(warped_points, perspective_transform[0, :2]), axis=1),
        perspective_transform[0, 2],
    )
    warped_col1 = np.add(
        np.sum(np.multiply(warped_points, perspective_transform[1, :2]), axis=1),
        perspective_transform[1, 2],
    )
    warped_col2 = np.add(
        np.sum(np.multiply(warped_points, perspective_transform[2, :2]), axis=1),
        perspective_transform[2, 2],
    )
    warped_col0 = np.divide(warped_col0, warped_col2)
    warped_col1 = np.divide(warped_col1, warped_col2)
    warped_points = np.concatenate([warped_col0[:, None], warped_col1[:, None]], axis=1)
    warped_points = warped_points.astype(int)

    # Fill the rectangles
    color = get_random_color(random_state, background_color)
    for i in range(col):
        color = (color + 128 + random_state.randint(-30, 30)) % 256
        cv.fillConvexPoly(
            img,
            np.array(
                [
                    (warped_points[i, 0], warped_points[i, 1]),
                    (warped_points[i + 1, 0], warped_points[i + 1, 1]),
                    (warped_points[i + col + 2, 0], warped_points[i + col + 2, 1]),
                    (warped_points[i + col + 1, 0], warped_points[i + col + 1, 1]),
                ]
            ),
            color,
        )

    # Draw lines on the boundaries of the stripes at random
    num_rows = random_state.randint(2, 5)
    num_cols = random_state.randint(2, col + 2)
    thickness = random_state.randint(min_dim * 0.01, min_dim * 0.015)
    for _ in range(num_rows):
        row_idx = random_state.choice([0, col + 1])
        col_idx1 = random_state.randint(col + 1)
        col_idx2 = random_state.randint(col + 1)
        color = get_random_color(random_state, background_color)
        cv.line(
            img,
            (
                warped_points[row_idx + col_idx1, 0],
                warped_points[row_idx + col_idx1, 1],
            ),
            (
                warped_points[row_idx + col_idx2, 0],
                warped_points[row_idx + col_idx2, 1],
            ),
            color,
            thickness,
        )
    for _ in range(num_cols):
        col_idx = random_state.randint(col + 1)
        color = get_random_color(random_state, background_color)
        cv.line(
            img,
            (warped_points[col_idx, 0], warped_points[col_idx, 1]),
            (warped_points[col_idx + col + 1, 0], warped_points[col_idx + col + 1, 1]),
            color,
            thickness,
        )

    # Keep only the points inside the image
    points = keep_points_inside(warped_points, img.shape[:2])
    return points


def draw_cube(
    random_state,
    img,
    min_size_ratio=0.2,
    min_angle_rot=math.pi / 10,
    scale_interval=(0.4, 0.6),
    trans_interval=(0.5, 0.2),
):
    """Draw a 2D projection of a cube and output the corners that are visible
    Parameters:
      min_size_ratio: min(img.shape) * min_size_ratio is the smallest achievable
                      cube side size
      min_angle_rot: minimal angle of rotation
      scale_interval: the scale is between scale_interval[0] and
                      scale_interval[0]+scale_interval[1]
      trans_interval: the translation is between img.shape*trans_interval[0] and
                      img.shape*(trans_interval[0] + trans_interval[1])
    """
    # Generate a cube and apply to it an affine transformation
    # The order matters!
    # The indices of two adjacent vertices differ only of one bit (as in Gray codes)
    background_color = int(np.mean(img))
    min_dim = min(img.shape[:2])
    min_side = min_dim * min_size_ratio
    lx = min_side + random_state.rand() * 2 * min_dim / 3  # dimensions of the cube
    ly = min_side + random_state.rand() * 2 * min_dim / 3
    lz = min_side + random_state.rand() * 2 * min_dim / 3
    cube = np.array(
        [
            [0, 0, 0],
            [lx, 0, 0],
            [0, ly, 0],
            [lx, ly, 0],
            [0, 0, lz],
            [lx, 0, lz],
            [0, ly, lz],
            [lx, ly, lz],
        ]
    )
    rot_angles = random_state.rand(3) * 3 * math.pi / 10.0 + math.pi / 10.0
    rotation_1 = np.array(
        [
            [math.cos(rot_angles[0]), -math.sin(rot_angles[0]), 0],
            [math.sin(rot_angles[0]), math.cos(rot_angles[0]), 0],
            [0, 0, 1],
        ]
    )
    rotation_2 = np.array(
        [
            [1, 0, 0],
            [0, math.cos(rot_angles[1]), -math.sin(rot_angles[1])],
            [0, math.sin(rot_angles[1]), math.cos(rot_angles[1])],
        ]
    )
    rotation_3 = np.array(
        [
            [math.cos(rot_angles[2]), 0, -math.sin(rot_angles[2])],
            [0, 1, 0],
            [math.sin(rot_angles[2]), 0, math.cos(rot_angles[2])],
        ]
    )
    scaling = np.array(
        [
            [scale_interval[0] + random_state.rand() * scale_interval[1], 0, 0],
            [0, scale_interval[0] + random_state.rand() * scale_interval[1], 0],
            [0, 0, scale_interval[0] + random_state.rand() * scale_interval[1]],
        ]
    )
    trans = np.array(
        [
            img.shape[1] * trans_interval[0]
            + random_state.randint(
                -img.shape[1] * trans_interval[1], img.shape[1] * trans_interval[1]
            ),
            img.shape[0] * trans_interval[0]
            + random_state.randint(
                -img.shape[0] * trans_interval[1], img.shape[0] * trans_interval[1]
            ),
            0,
        ]
    )
    cube = trans + np.transpose(
        np.dot(
            scaling,
            np.dot(
                rotation_1, np.dot(rotation_2, np.dot(rotation_3, np.transpose(cube)))
            ),
        )
    )

    # The hidden corner is 0 by construction
    # The front one is 7
    cube = cube[:, :2]  # project on the plane z=0
    cube = cube.astype(int)
    points = cube[1:, :]  # get rid of the hidden corner

    # Get the three visible faces
    faces = np.array([[7, 3, 1, 5], [7, 5, 4, 6], [7, 6, 2, 3]])

    # Fill the faces and draw the contours
    col_face = get_random_color(random_state, background_color)
    for i in [0, 1, 2]:
        cv.fillPoly(img, [cube[faces[i]].reshape((-1, 1, 2))], col_face)
    thickness = random_state.randint(min_dim * 0.003, min_dim * 0.015)
    for i in [0, 1, 2]:
        for j in [0, 1, 2, 3]:
            col_edge = (
                col_face + 128 + random_state.randint(-64, 64)
            ) % 256  # color that constrats with the face color
            cv.line(
                img,
                (cube[faces[i][j], 0], cube[faces[i][j], 1]),
                (cube[faces[i][(j + 1) % 4], 0], cube[faces[i][(j + 1) % 4], 1]),
                col_edge,
                thickness,
            )

    # Keep only the points inside the image
    points = keep_points_inside(points, img.shape[:2])
    return points


def gaussian_noise(random_state, img):
    """Apply random noise to the image"""
    img[:] = random_state.randint(0, 255, size=img.shape, dtype=np.uint8)
    return np.empty((0, 2), dtype=int)


def draw_interest_points(img, points, color=(255, 0, 0)):
    """Convert img in RGB and draw in red the interest points"""

    # handles different shapes of image input (rgb, grayscale with or without channel dim)
    if len(img.shape) == 2:
        img_rgb = np.stack([img, img, img], axis=2)
    elif len(img.shape) == 3:
        if img.shape[2] == 1:
            img_rgb = np.concatenate([img, img, img], axis=2)
        elif img.shape[0] == 1:
            img_rgb = np.concatenate([img, img, img], axis=0)
            img_rgb = img_rgb.transpose([1, 2, 0])
        else:
            img_rgb = img
    else:
        raise RuntimeError(f"invalid image shape {img.shape}")

    radius = int(max(img_rgb.shape[0], img_rgb.shape[1]) * 0.01)

    for i in range(points.shape[0]):
        if radius > 1:
            cv.circle(img_rgb, (int(points[i][1]), int(points[i][0])), radius, color, 1)
        else:
            img_rgb[int(points[i][0]), int(points[i][1]), 0] = color[0]
            img_rgb[int(points[i][0]), int(points[i][1]), 1] = color[1]
            img_rgb[int(points[i][0]), int(points[i][1]), 2] = color[2]
    return img_rgb


def _gaussian_blur(img, kernel_size):
    """Apply in-place gaussian blur to image."""
    return cv.GaussianBlur(img, (kernel_size, kernel_size), 0)


def _resize_image_and_points(img, points, new_height, new_width):
    """Resize image and keypoint coordinates."""
    current_height = img.shape[0]
    current_width = img.shape[1]

    # rescale point coordinates
    points = (
        points
        * np.array((new_height, new_width), float)
        / np.array((current_height, current_width), float)
    )

    # resize image
    img = cv.resize(
        img,
        (new_width, new_height),  # notice how dimensions are reversed here
        interpolation=cv.INTER_LINEAR,
    )
    img = np.expand_dims(img, axis=-1)
    img = img.astype(np.float32)

    return img, points
