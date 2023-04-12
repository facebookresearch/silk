# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple, Union

import torch

from silk.config.sampler import Sampler
from silk.cv.homography import HomographicSampler
from silk.transforms.abstract import Transform


class RandomHomographicSampler(Transform, HomographicSampler):
    """`silk.cv.homography.HomographicSampler` with an easy way to randomize homographies into easy to control linear transforms."""

    def __init__(
        self,
        batch_size: int,
        sampling_size: Tuple[int, int],
        sampling_mode: str = "bilinear",
        scaling_sampler: Union[Sampler, None] = None,
        x_rotation_sampler: Union[Sampler, None] = None,
        y_rotation_sampler: Union[Sampler, None] = None,
        z_rotation_sampler: Union[Sampler, None] = None,
        x_translation_sampler: Union[Sampler, None] = None,
        y_translation_sampler: Union[Sampler, None] = None,
        auto_randomize: bool = True,
        device: Optional[str] = None,
    ) -> None:
        """
        Parameters
        ----------
        batch_size : int
            Number of virtual crops.
        sampling_size : Tuple[int, int]
            Spatial shape of generated output crops.
        sampling_mode : str, optional
            Sampling mode passed to `grid_sample`, by default "bilinear"
        scaling_sampler : Union[Sampler, None], optional
            Scaling factor sampler, by default None
        x_rotation_sampler : Union[Sampler, None], optional
            x-axis rotation (out-of-plane) sampler, by default None
        y_rotation_sampler : Union[Sampler, None], optional
            y-axis rotation (out-of-plane) sampler, by default None
        z_rotation_sampler : Union[Sampler, None], optional
            z-axis rotation (in-plane) sampler, by default None
        x_translation_sampler : Union[Sampler, None], optional
            horizontal translation sampler, by default None
        y_translation_sampler : Union[Sampler, None], optional
            vertical translation sampler, by default None
        auto_randomize : bool, optional
            Automatically call the `randomize` function when calling `forward_sampling` or `backward_sampling`, by default True
        device : Optional[str], optional
            Device used for sampling, by default None
        """
        Transform.__init__(self)
        HomographicSampler.__init__(self, batch_size, device)

        self._sampling_size = sampling_size
        self._sampling_mode = sampling_mode

        self._auto_randomize = auto_randomize

        self._scaling_sampler = scaling_sampler
        self._x_rotation_sampler = x_rotation_sampler
        self._y_rotation_sampler = y_rotation_sampler
        self._z_rotation_sampler = z_rotation_sampler
        self._x_translation_sampler = x_translation_sampler
        self._y_translation_sampler = y_translation_sampler

    @property
    def sampling_size(self):
        return self._sampling_size

    def randomize(self):
        """Generate random homographic transform parameters."""
        self.reset()

        # 1. rescale
        if self._scaling_sampler:
            scaling_factors = self._scaling_sampler(
                shape=(self.batch_size, 1),
                device=self.device,
                dtype=self.dtype,
            )
            self.scale(scaling_factors)

        # 2. rotations
        def _rot(rot_sampler, axis):
            if rot_sampler:
                angles = rot_sampler(
                    shape=(self.batch_size, 1),
                    device=self.device,
                    dtype=self.dtype,
                )
                self.rotate(angles, axis=axis)

        _rot(self._x_rotation_sampler, "x")
        _rot(self._y_rotation_sampler, "y")
        _rot(self._z_rotation_sampler, "z")

        # 3. shift crop
        if self._x_translation_sampler or self._y_translation_sampler:
            if self._x_translation_sampler:
                x_translations = self._x_translation_sampler(
                    shape=(self.batch_size, 1),
                    device=self.device,
                    dtype=self.dtype,
                )
            else:
                x_translations = torch.zeros(
                    size=(self.batch_size, 1),
                    device=self.device,
                    dtype=self.dtype,
                )

            if self._y_translation_sampler:
                y_translations = self._y_translation_sampler(
                    shape=(self.batch_size, 1),
                    device=self.device,
                    dtype=self.dtype,
                )
            else:
                y_translations = torch.zeros(
                    size=(self.batch_size, 1),
                    device=self.device,
                    dtype=self.dtype,
                )

            shift_deltas = torch.cat((x_translations, y_translations), dim=1)
            self.shift(shift_deltas)

        # TODO(Pierre): Add mechanism to make sure sampling stays inside the original image

    def _sample(
        self, images: torch.Tensor, randomize=None, direction="forward"
    ) -> torch.Tensor:
        if (randomize is None and self._auto_randomize) or randomize:
            self.randomize()

        return self.extract_crop(
            images, self._sampling_size, mode=self._sampling_mode, direction=direction
        )

    def forward_sampling(
        self, images: torch.Tensor, randomize: Optional[bool] = None
    ) -> torch.Tensor:
        """Sample crops from randomly generated homographies.

        Parameters
        ----------
        images : torch.Tensor
            Images to extract the crops from.
        randomize : bool, optional
            Randomize before sampling (otherwise use previous randomly generated homographies), by default None. Overwrites `auto_randomize` option.

        Returns
        -------
        torch.Tensor
            Generated crops.
        """
        return self._sample(images, randomize, "forward")

    def backward_sampling(self, images: torch.Tensor, randomize=None) -> torch.Tensor:
        """Reverse operation of `forward_sampling` (i.e. position provided crops into original image).

        Parameters
        ----------
        images : torch.Tensor
            Images to extract the crops from.
        randomize : bool, optional
            Randomize before sampling (otherwise use previous randomly generated homographies), by default None. Overwrites `auto_randomize` option.

        Returns
        -------
        torch.Tensor
            Generated crops.
        """
        return self._sample(images, randomize, "backward")

    __call__ = forward_sampling
