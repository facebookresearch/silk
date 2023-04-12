# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import itertools
from typing import List, Optional, Tuple, Union

import torch
from torch.nn.functional import grid_sample
from torch.nn.utils.rnn import pad_sequence


def resize_homography(
    homography: torch.Tensor,
    original_image_shape: Tuple[int, int],
    new_original_image_shape,
    warped_image_shape=None,
    new_warped_image_shape=None,
) -> torch.Tensor:
    """Change homography matrix when image sizes change.

    Parameters
    ----------
    homography : torch.Tensor
        Homography matrix as a 3x3 Tensor.
    original_image_shape : Tuple[int, int]
        Size of the original image the current homography applies to.
    new_original_image_shape : Tuple[int, int]
        Size of the new original image the resized homography should apply to.
    warped_image_shape : Tuple[int, int], optional
        Size of the warped image the current homography applies to, by default None. Set to `original_image_shape` when None.
    new_warped_image_shape : Tuple[int, int], optional
        Size of the new warped image the resized homography should apply to, by default None. Set to `new_original_image_shape` when None.

    Returns
    -------
    torch.Tensor
        New homography operating on provided image sizes.
    """
    warped_image_shape = (
        original_image_shape if warped_image_shape is None else warped_image_shape
    )
    new_warped_image_shape = (
        new_original_image_shape
        if new_warped_image_shape is None
        else new_warped_image_shape
    )

    # compute resizing factors
    oh_factor = original_image_shape[0] / new_original_image_shape[0]
    ow_factor = original_image_shape[1] / new_original_image_shape[1]

    wh_factor = new_warped_image_shape[0] / warped_image_shape[0]
    ww_factor = new_warped_image_shape[1] / warped_image_shape[1]

    # build resizing diagonal matrices
    up_scale = torch.diag(
        torch.tensor(
            [ow_factor, oh_factor, 1.0],
            dtype=homography.dtype,
            device=homography.device,
        )
    )
    down_scale = torch.diag(
        torch.tensor(
            [ww_factor, wh_factor, 1.0],
            dtype=homography.dtype,
            device=homography.device,
        )
    )

    # apply resizing to homography
    homography = down_scale @ homography @ up_scale

    return homography


class HomographicSampler:
    """Samples multiple homographic crops from multiples batched images.

    This sampler makes it very easy to sample homographic crops from multiple images by manipulating a virtual crop initially centered on the entire image.
    Applying successive simple transformations (xyz-rotation, shift, scale) will modify the position and shape of that virtual crop.
    Transformations operates on normalized coordinates independent of an image shape.
    The initial virtual crop has its top-left position at (-1, -1), and bottom-right position at (+1, +1).
    Thus the center being at position (0, 0).

    Examples
    --------

    ```python
    hc = HomographicSampler(2, "cpu") # homographic sampler with 2 virtual crops

    hc.scale(0.5) # reduce all virtual crops size by half
    hc.shift(((-0.25, -0.25), (+0.25, +0.25))) # shift first virtual crop to top-left part, second virtual crop to bottom-right part
    hc.rotate(3.14/4., axis="x", clockwise=True, local_center=True) # rotate both virtual crops locally by 45 degrees clockwise (around x-axis)

    crops = hc.extract_crop(image, (100, 100)) # extract two homographic crops defined earlier as (100, 100) images
    ```

    """

    _DEST_COORD = torch.tensor(
        [
            [-1.0, -1.0],  # top-left
            [+1.0, -1.0],  # top-right
            [-1.0, +1.0],  # bottom-left
            [+1.0, +1.0],  # bottom-right
        ],
        dtype=torch.double,
    )

    _VALID_AXIS = {"x", "y", "z"}
    _VALID_DIRECTIONS = {"forward", "backward"}
    _VALID_ORDERING = {"xy", "yx"}

    def __init__(self, batch_size: int, device: str):
        """

        Parameters
        ----------
        batch_size : int
            Number of virtual crops to handle.
        device : str
            Device on which operations will be done.
        """
        self.reset(batch_size, device)

    @staticmethod
    def _convert_points_from_homogeneous(
        points: torch.Tensor, eps: float = 1e-8
    ) -> torch.Tensor:
        """Function that converts points from homogeneous to Euclidean space."""

        # we check for points at max_val
        z_vec: torch.Tensor = points[..., -1:]

        # set the results of division by zeror/near-zero to 1.0
        # follow the convention of opencv:
        # https://github.com/opencv/opencv/pull/14411/files
        mask: torch.Tensor = torch.abs(z_vec) > eps
        scale = torch.where(mask, 1.0 / (z_vec + eps), torch.ones_like(z_vec))

        return scale * points[..., :-1]

    @staticmethod
    def _convert_points_to_homogeneous(points: torch.Tensor) -> torch.Tensor:
        """Function that converts points from Euclidean to homogeneous space."""

        return torch.nn.functional.pad(points, [0, 1], "constant", 1.0)

    @staticmethod
    def _transform_points(
        trans_01: torch.Tensor, points_1: torch.Tensor
    ) -> torch.Tensor:
        """Function that applies a linear transformations to a set of points."""

        points_1 = points_1.to(trans_01.device)
        points_1 = points_1.to(trans_01.dtype)

        # We reshape to BxNxD in case we get more dimensions, e.g., MxBxNxD
        shape_inp = points_1.shape
        points_1 = points_1.reshape(-1, points_1.shape[-2], points_1.shape[-1])
        trans_01 = trans_01.reshape(-1, trans_01.shape[-2], trans_01.shape[-1])
        # We expand trans_01 to match the dimensions needed for bmm
        trans_01 = torch.repeat_interleave(
            trans_01, repeats=points_1.shape[0] // trans_01.shape[0], dim=0
        )
        # to homogeneous
        points_1_h = HomographicSampler._convert_points_to_homogeneous(
            points_1
        )  # BxNxD+1
        # transform coordinates
        points_0_h = torch.bmm(points_1_h, trans_01.permute(0, 2, 1))
        points_0_h = torch.squeeze(points_0_h, dim=-1)
        # to euclidean
        points_0 = HomographicSampler._convert_points_from_homogeneous(
            points_0_h
        )  # BxNxD
        # reshape to the input shape
        points_0 = points_0.reshape(shape_inp)
        return points_0

    @staticmethod
    def _create_meshgrid(
        height: int,
        width: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        normalized: bool = True,
    ) -> torch.Tensor:
        """Generate a coordinate grid for an image."""
        if normalized:
            min_x = -1.0
            max_x = +1.0
            min_y = -1.0
            max_y = +1.0
        else:
            min_x = 0.5
            max_x = width - 0.5
            min_y = 0.5
            max_y = height - 0.5

        xs: torch.Tensor = torch.linspace(
            min_x,
            max_x,
            width,
            device=device,
            dtype=dtype,
        )
        ys: torch.Tensor = torch.linspace(
            min_y,
            max_y,
            height,
            device=device,
            dtype=dtype,
        )

        # generate grid by stacking coordinates
        base_grid: torch.Tensor = torch.stack(
            torch.meshgrid([xs, ys], indexing="ij"), dim=-1
        )  # WxHx2
        return base_grid.permute(1, 0, 2).unsqueeze(0)  # 1xHxWx2

    @staticmethod
    def _build_perspective_param(
        p: torch.Tensor, q: torch.Tensor, axis: str
    ) -> torch.Tensor:
        ones = torch.ones_like(p)[..., 0:1]
        zeros = torch.zeros_like(p)[..., 0:1]
        if axis == "x":
            return torch.cat(
                [
                    p[:, 0:1],
                    p[:, 1:2],
                    ones,
                    zeros,
                    zeros,
                    zeros,
                    -p[:, 0:1] * q[:, 0:1],
                    -p[:, 1:2] * q[:, 0:1],
                ],
                dim=1,
            )

        if axis == "y":
            return torch.cat(
                [
                    zeros,
                    zeros,
                    zeros,
                    p[:, 0:1],
                    p[:, 1:2],
                    ones,
                    -p[:, 0:1] * q[:, 1:2],
                    -p[:, 1:2] * q[:, 1:2],
                ],
                dim=1,
            )

        raise NotImplementedError(
            f"perspective params for axis `{axis}` is not implemented."
        )

    @staticmethod
    def _get_perspective_transform(src, dst):
        r"""Calculate a perspective transform from four pairs of the corresponding
        points.

        The function calculates the matrix of a perspective transform so that:

        .. math ::

            \begin{bmatrix}
            t_{i}x_{i}^{'} \\
            t_{i}y_{i}^{'} \\
            t_{i} \\
            \end{bmatrix}
            =
            \textbf{map_matrix} \cdot
            \begin{bmatrix}
            x_{i} \\
            y_{i} \\
            1 \\
            \end{bmatrix}

        where

        .. math ::
            dst(i) = (x_{i}^{'},y_{i}^{'}), src(i) = (x_{i}, y_{i}), i = 0,1,2,3

        Args:
            src: coordinates of quadrangle vertices in the source image with shape :math:`(B, 4, 2)`.
            dst: coordinates of the corresponding quadrangle vertices in
                the destination image with shape :math:`(B, 4, 2)`.

        Returns:
            the perspective transformation with shape :math:`(B, 3, 3)`.
        """

        # we build matrix A by using only 4 point correspondence. The linear
        # system is solved with the least square method, so here
        # we could even pass more correspondence
        p = []
        for i in [0, 1, 2, 3]:
            p.append(
                HomographicSampler._build_perspective_param(src[:, i], dst[:, i], "x")
            )
            p.append(
                HomographicSampler._build_perspective_param(src[:, i], dst[:, i], "y")
            )

        # A is Bx8x8
        A = torch.stack(p, dim=1)

        # b is a Bx8x1
        b = torch.stack(
            [
                dst[:, 0:1, 0],
                dst[:, 0:1, 1],
                dst[:, 1:2, 0],
                dst[:, 1:2, 1],
                dst[:, 2:3, 0],
                dst[:, 2:3, 1],
                dst[:, 3:4, 0],
                dst[:, 3:4, 1],
            ],
            dim=1,
        )

        # solve the system Ax = b
        X = torch.linalg.solve(A, b)

        # create variable to return
        batch_size = src.shape[0]
        M = torch.ones(batch_size, 9, device=src.device, dtype=src.dtype)
        M[..., :8] = torch.squeeze(X, dim=-1)

        return M.view(-1, 3, 3)  # Bx3x3

    def reset(self, batch_size: Optional[int] = None, device: Optional[str] = None):
        """Resets all the crops to their initial position and sizes.

        Parameters
        ----------
        batch_size : int, optional
            Number of virtual crops to handle, by default None.
        device : str, optional
            Device on which operations will be done, by default None.
        """
        batch_size = self.batch_size if batch_size is None else batch_size
        device = self.device if device is None else device

        self._dest_coords = HomographicSampler._DEST_COORD.to(device)
        self._dest_coords = self._dest_coords.unsqueeze(0)
        self._dest_coords = self._dest_coords.expand(batch_size, -1, -1)

        self._homog_src_coords = HomographicSampler._convert_points_to_homogeneous(
            self._dest_coords
        )

        self._clear_cache()

    def _clear_cache(self):
        """Intermediate data are cached such that the same homographic sampler can efficiently be called several times using the same homographic transforms."""
        self._src_coords = None
        self._forward_matrices = None
        self._backward_matrices = None

    def _to(self, device, name):
        attr = getattr(self, name)
        if attr is not None:
            setattr(self, name, attr.to(device))

    def to(self, device: str):
        """Moves all operations to new device.

        Parameters
        ----------
        device : str
            Pytorch device.
        """
        if device != self.device:
            self._to(device, "_dest_coords")
            self._to(device, "_src_coords")
            self._to(device, "_homog_src_coords")
            self._to(device, "_forward_matrices")
            self._to(device, "_backward_matrices")
        return self

    @property
    def batch_size(self):
        return self._homog_src_coords.shape[0]

    @property
    def device(self):
        return self._homog_src_coords.device

    @property
    def dtype(self):
        return self._homog_src_coords.dtype

    @property
    def src_coords(self) -> torch.Tensor:
        """Coordinates of the homographic crop corners in the virtual image coordinate reference system.
        Those four points are ordered as : (top-left, top-right, bottom-left, bottom-right)

        Returns
        -------
        torch.Tensor
            :math:`(B, 4, 2)` tensor containing the homographic crop foud corners coordinates.
        """
        if self._src_coords is None:
            self._src_coords = HomographicSampler._convert_points_from_homogeneous(
                self._homog_src_coords
            )
        return self._src_coords

    @property
    def dest_coords(self) -> torch.Tensor:
        return self._dest_coords

    def _auto_expand(self, input, outer_dim_size=None, **kwargs):
        """Auto-expand scalar or iterables to be batched."""
        if not isinstance(input, torch.Tensor):
            input = torch.tensor(input, **kwargs)

        # scalar
        if len(input.shape) == 0:
            input = input.unsqueeze(0)
            if outer_dim_size is None:
                outer_dim_size = 1
            else:
                input = input.expand(outer_dim_size)

        # vector
        if len(input.shape) == 1:
            if outer_dim_size is None:
                outer_dim_size = input.shape[0]
            elif outer_dim_size != input.shape[0]:
                raise RuntimeError(
                    f"provided outer dim size {outer_dim_size} doesn't match input shape {input.shape}"
                )

            input = input.unsqueeze(0)
            input = input.expand(self.batch_size, -1)

        if len(input.shape) != 2:
            raise RuntimeError(f"input should have size BxD (shape is {input.shape}")

        input = input.type(self.dtype)
        input = input.to(self.device)

        return input

    def rotate(
        self,
        angles: Union[float, torch.Tensor],
        clockwise: bool = False,
        axis: str = "z",
        local_center: bool = False,
    ):
        """Rotate virtual crops.

        Parameters
        ----------
        angles : Union[float, torch.Tensor]
            Angles of rotation. If scalar, applied to all crops. If :math:`(B, 1)` tensor, applied to each crop independently.
        clockwise : bool, optional
            Rotational direction, by default False
        axis : str, optional
            Axis of rotation, by default "z". Valid values are "x", "y" and "z". "z" is in-plane rotation. "x" and "y" are out-of-plane rotations.
        local_center : bool, optional
            Rotate on the center of the crop, by default False. If False, use global center of rotation (i.e. initial crop center). This option is only relevant after a shift has been used.

        Raises
        ------
        RuntimeError
            Raised if provided axis is invalid.
        """
        if axis not in HomographicSampler._VALID_AXIS:
            raise RuntimeError(
                f'provided axis "{axis}" isn\'t valid, should be one of {HomographicSampler._VALID_AXIS}'
            )

        angles = self._auto_expand(angles, outer_dim_size=1)

        if clockwise:
            angles = -angles

        cos_a = torch.cos(angles)
        sin_a = torch.sin(angles)

        _1 = torch.ones_like(cos_a)
        _0 = torch.zeros_like(cos_a)

        if axis == "z":
            flatmat = [+cos_a, -sin_a, _0, +sin_a, +cos_a, _0, _0, _0, _1]
        elif axis == "y":
            flatmat = [+cos_a, _0, -sin_a, _0, _1, _0, +sin_a, _0, +cos_a]
        elif axis == "x":
            flatmat = [_1, _0, _0, _0, +cos_a, +sin_a, _0, -sin_a, +cos_a]

        rot_matrix = torch.cat(flatmat, dim=-1)
        rot_matrix = rot_matrix.view(self.batch_size, 3, 3)

        self._clear_cache()

        if local_center:
            center = torch.mean(self._homog_src_coords, dim=1, keepdim=True)

            self._homog_src_coords -= center
            self._homog_src_coords = self._homog_src_coords @ rot_matrix
            self._homog_src_coords += center
        else:
            if axis != "z":
                self._homog_src_coords[..., -1] -= 1.0
            self._homog_src_coords = self._homog_src_coords @ rot_matrix
            if axis != "z":
                self._homog_src_coords[..., -1] += 1.0

    def shift(self, delta: Union[float, Tuple[float, float], torch.Tensor]):
        """Shift virtual crops.

        Parameters
        ----------
        delta : Union[float, Tuple[float, float], torch.Tensor]
            Shift values. Scalar or Tuple will be applied to all crops. :math:`(B, 2)` tensors will be applied to each crop independently.
        """

        delta = self._auto_expand(delta, outer_dim_size=2)
        delta = delta.unsqueeze(1)
        delta = delta * self._homog_src_coords[..., -1].unsqueeze(-1)

        self._clear_cache()
        self._homog_src_coords[..., :2] += delta

    def scale(
        self,
        factors: Union[float, Tuple[float, float], torch.Tensor],
        local_center: bool = False,
    ):
        """Scale the virtual crops.

        Parameters
        ----------
        factors : Union[float, Tuple[float, float], torch.Tensor]
            Scaling factors. Scalar or Tuple will be applied to all crops. :math:`(B, 2)` tensors will be applied to each crop independently.
        local_center : bool, optional
            Scale on the center of the crop, by default False. If False, use global center of rotation (i.e. initial crop center). This option is only relevant after a shift has been used.
        """
        factors = self._auto_expand(factors, outer_dim_size=2)
        factors = factors.unsqueeze(1)

        self._clear_cache()

        if local_center:
            center = torch.mean(self._homog_src_coords, dim=1, keepdim=True)

            self._homog_src_coords -= center
            self._homog_src_coords[..., :2] *= factors
            self._homog_src_coords += center
        else:
            self._homog_src_coords[..., :2] *= factors

    @property
    def forward_matrices(self):
        if self._forward_matrices is None:
            self._forward_matrices = HomographicSampler._get_perspective_transform(
                self.dest_coords,
                self.src_coords,
            )
        return self._forward_matrices

    @property
    def backward_matrices(self):
        if self._backward_matrices is None:
            self._backward_matrices = HomographicSampler._get_perspective_transform(
                self.src_coords,
                self.dest_coords,
            )
        return self._backward_matrices

    def extract_crop(
        self,
        images: torch.Tensor,
        sampling_size: Tuple[int, int],
        mode="bilinear",
        padding_mode="zeros",
        direction="forward",
    ) -> torch.Tensor:
        """Extract all crops from a set of images.

        It can handle one-image-to-many-crops and many-images-to-many-crops.
        If the number of images is smaller than the number of crops, a number of n crops will be asssigned to each image such that :math:`n_crops = n * n_images`.

        Parameters
        ----------
        images : torch.Tensor
            Tensor containing all images (valid shapes are :math:`(B,C,H,W)` and :math:`(C,H,W)`).
        sampling_size : Tuple[int, int]
            Spatial shape of the output crops.
        mode : str, optional
            Sampling mode passed to `grid_sample`, by default "bilinear".
        padding_mode : str, optional
            Padding mode passed to `grid_sample`, by default "zeros".
        direction : str, optional
            Direction of the crop sampling (`src -> dest` or `dest -> src`), by default "forward". Valid are "forward" and "backward".

        Returns
        -------
        torch.Tensor
            Sampled crops using transformed virtual crops.

        Raises
        ------
        RuntimeError
            Raised is `images` shape is invalid.
        RuntimeError
            Raised is `images` batch size isn't a multiple of the number of virtual crops.
        RuntimeError
            Raised is `direction` is invalid.
        """
        if images.dim() == 3:
            images = images.unsqueeze(0)
        elif images.dim() != 4:
            raise RuntimeError("provided image(s) should be of shape BxCxHxW or CxHxW")

        if self.batch_size % images.shape[0] != 0:
            raise RuntimeError(
                f"the sampler batch size ({self.batch_size}) should be a multiple of the image batch size (found {images.shape[0]})"
            )

        if direction not in HomographicSampler._VALID_DIRECTIONS:
            raise RuntimeError(
                f'invalid direction "{direction}" found, should be one of {self._VALID_DIRECTIONS}'
            )

        # reshape images to handle multiple crops
        crop_per_image = self.batch_size // images.shape[0]
        images = images.unsqueeze(1)
        images = images.expand(-1, crop_per_image, -1, -1, -1)
        images = images.reshape(self.batch_size, *images.shape[2:])

        # select homography matrix
        if direction == "forward":
            matrix = self.forward_matrices
        else:
            matrix = self.backward_matrices

        # create grid of coordinates used for image sampling
        grid = HomographicSampler._create_meshgrid(
            sampling_size[0],
            sampling_size[1],
            device=matrix.device,
            dtype=matrix.dtype,
        )
        grid = grid.expand(self.batch_size, -1, -1, -1)
        grid = HomographicSampler._transform_points(matrix[:, None, None], grid)
        grid = grid.type_as(images)

        # sample pixels using transformed grid coordinates
        return grid_sample(
            images,
            grid,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=False,
        )

    def transform_points(
        self,
        points: Union[torch.Tensor, List[torch.Tensor]],
        image_shape: Optional[Tuple[int, int]] = None,
        direction: str = "forward",
        ordering: str = "xy",
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Apply homography to a set of points.

        Parameters
        ----------
        points : Union[torch.Tensor, List[torch.Tensor]]
            BxNx2 tensor or list of Nx2 tensors containing the coordinates to transform.
        image_shape : Optional[Tuple[int, int]], optional
            Shape of the tensor the coordinates references, as in (height, width), by default None.
            If not provided, the coordinates are assumed to be already normalized between [-1, +1].
        direction : str, optional
            Direction of the homography, by default "forward".
        ordering : str, optional
            Specify the order in which the x,y coordinates are stored in "points", by default "xy".

        Returns
        -------
        Union[torch.Tensor, List[torch.Tensor]]
            Transformed coordinates.

        Raises
        ------
        RuntimeError
            If the provided direction is invalid.
        RuntimeError
            If the provided ordering is invalid.
        """
        # check arguments
        if direction not in HomographicSampler._VALID_DIRECTIONS:
            raise RuntimeError(
                f'invalid direction "{direction}" found, should be one of {self._VALID_DIRECTIONS}'
            )
        if ordering not in HomographicSampler._VALID_ORDERING:
            raise RuntimeError(
                f'invalid ordering "{ordering}" found, should be one of {self._VALID_ORDERING}'
            )

        # select homography matrices
        if direction == "forward":
            matrix = self.backward_matrices
        else:
            matrix = self.forward_matrices

        # pad input if using variable length
        lengths = None
        if not isinstance(points, torch.Tensor):
            lengths = [p.shape[0] for p in points]
            points = pad_sequence(points, batch_first=True)

        # convert to "xy" ordering
        if ordering == "yx":
            points = points[..., [1, 0]]

        # bring coordinates to [-1, +1] range
        if image_shape is not None:
            image_shape = torch.tensor(
                [image_shape[1], image_shape[0]],
                dtype=points.dtype,
                device=points.device,
            )
            image_shape = image_shape[None, None, ...]
            image_shape_half = image_shape / 2.0
            pixel_shift = 0.5 / image_shape
            points = (points - image_shape_half) / image_shape_half + pixel_shift

        # reshape points to handle multiple transforms
        transform_per_points = self.batch_size // points.shape[0]
        points = points.unsqueeze(1)
        points = points.expand(-1, transform_per_points, -1, -1)
        points = points.reshape(self.batch_size, *points.shape[2:])

        # change lengths size accordingly
        if transform_per_points != 1:
            lengths = list(
                itertools.chain.from_iterable(
                    itertools.repeat(s, transform_per_points) for s in lengths
                )
            )

        # apply homography to point coordinates
        transformed_points = HomographicSampler._transform_points(
            matrix[:, None, None], points
        )

        # bring coordinates to original range
        if image_shape is not None:
            transformed_points = (
                (transformed_points - pixel_shift) * image_shape_half
            ) + image_shape_half

        # convert back to initial ordering
        if ordering == "yx":
            transformed_points = transformed_points[..., [1, 0]]

        # remove padded results if input was variable length
        if lengths is not None:
            transformed_points = [
                transformed_points[i, :s] for i, s in enumerate(lengths)
            ]

        return transformed_points
