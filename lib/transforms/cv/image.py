# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple, Union

import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms

from silk.cv.homography import HomographicSampler
from silk.transforms.abstract import Transform
from silk.transforms.tensor import Clamp


class Albu(Transform):
    def __init__(self, augmentor) -> None:
        super().__init__()

        self._augmentor = augmentor

    def __call__(self, item: torch.Tensor) -> torch.Tensor:
        item = item.permute(0, 2, 3, 1)

        device = item.device
        item = item.cpu().numpy()

        item = np.stack(
            [self._augmentor(image=img.astype(np.uint8))["image"] for img in item],
            axis=0,
        )

        return torch.from_numpy(item).to(device).permute(0, 3, 1, 2)


class HWCToCHW(Transform):
    """Convert image format from HWC to CHW. Handles batch dimension when present."""

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, item: torch.Tensor) -> torch.Tensor:
        if len(item.shape) == 3:
            return item.permute((2, 0, 1))
        elif len(item.shape) == 4:
            return item.permute((0, 3, 1, 2))
        raise RuntimeError("invalid tensor shape, 3 or 4 dimensions are expected")


class CHWToHWC(Transform):
    """Reverse operation of `HWCToCHW`."""

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, item: torch.Tensor) -> torch.Tensor:
        if len(item.shape) == 3:
            return item.permute((1, 2, 0))
        elif len(item.shape) == 4:
            return item.permute((0, 2, 3, 1))
        raise RuntimeError("invalid tensor shape, 3 or 4 dimensions are expected")


class Resize(Transform):
    """Simple wrapper of `torchvision.transforms.Resize` that handles `size` that are not `int`, `list` or `tuple`."""

    def __init__(self, size, *args, **kwargs) -> None:
        super().__init__()

        if not isinstance(size, int):
            # convert iterable to tuple to avoid invalid size type in function below
            size = tuple(size)

        self._resizer = torchvision.transforms.Resize(size, *args, **kwargs)

    def __call__(self, item: torch.Tensor) -> torch.Tensor:
        return self._resizer(item)


class GaussianNoise(Transform):
    """Add Gaussian Noise to Images."""

    def __init__(
        self,
        std: Union[float, Tuple[float, float]],
        max_val: Union[float, None] = None,
        min_val: Union[float, None] = None,
    ) -> None:
        """
        Parameters
        ----------
        std : Union[float, Tuple[float, float]]
            Standard deviation.
            If float, the same standard deviation will be used accross all images.
            If tuple of floats (min_std, max_std), a standard deviation will be sampled per image in the provided range.
        max_val : Union[float, None], optional
            `Clamp`'s max_val, by default None
        min_val : Union[float, None], optional
            `Clamp`'s min_val, by default None
        """
        super().__init__()

        self._std = std
        self._clamp = Clamp(min_val, max_val)

    def __call__(self, item: torch.Tensor) -> torch.Tensor:
        """Apply the Gaussian Noise to Tensor.

        Parameters
        ----------
        item : torch.Tensor
            Tensor of shape ...xCxHxW

        Returns
        -------
        torch.Tensor
            Input item with added noise.
        """
        if isinstance(self._std, float):
            std = self._std
        else:
            # sample standard deviation per image
            batch_shape = item.shape[:-3] + (1, 1, 1)
            std = self._std[0] + torch.rand(batch_shape, device=item.device) * (
                self._std[1] - self._std[0]
            )

        # compute noise
        noise = torch.normal(
            0.0,
            1.0,
            size=item.shape,
            device=item.device,
            dtype=item.dtype,
        )
        noise *= std

        # apply noise & clamp
        item += noise
        item = self._clamp(item)
        return item


class SpeckleNoise(Transform):
    """Add Speckle Noise to Images."""

    def __init__(
        self,
        prob: Union[float, Tuple[float, float]],
        max_val: Union[float, None] = None,
        min_val: Union[float, None] = None,
    ) -> None:
        """
        Parameters
        ----------
        prob : Union[float, Tuple[float, float]]
            Probability of adding a min/max speckle noise.
            If float, the same probability will be used accross all images.
            If tuple of floats (min_prob, max_prob), a probability will be sampled per image in the provided range.
        max_val : Union[float, None], optional
            Max speckle noise value, by default None
        min_val : Union[float, None], optional
            Min speckle noise value, by default None
        """
        super().__init__()

        if (min_val is None) and (max_val is None):
            raise RuntimeError(
                "either `max_val` or `min_val` (or both) have to be provided"
            )

        self._prob = prob
        self._min_val = min_val
        self._max_val = max_val

    def __call__(self, item: torch.Tensor) -> torch.Tensor:
        """Apply the Speckle Noise to Tensor.

        Parameters
        ----------
        item : torch.Tensor
            Tensor of shape ...xCxHxW

        Returns
        -------
        torch.Tensor
            Input item with added noise.
        """
        if isinstance(self._prob, float):
            prob_range = self._prob
        else:
            # sample probability per image
            batch_shape = batch_shape = item.shape[:-3] + (1, 1, 1)
            prob_range = self._prob[0] + torch.rand(batch_shape, device=item.device) * (
                self._prob[1] - self._prob[0]
            )

        # sample probabilities of adding speckle noise
        prob_sampling = torch.rand(item.shape, device=item.device)

        # add speckle noise
        shape = (1,) * len(item.shape)

        if self._min_val is not None:
            min_val = torch.full(
                shape, self._min_val, device=item.device, dtype=item.dtype
            )
            item = torch.where(prob_sampling <= prob_range, min_val, item)

        if self._max_val is not None:
            max_val = torch.full(
                shape, self._max_val, device=item.device, dtype=item.dtype
            )
            item = torch.where((1.0 - prob_sampling) <= prob_range, max_val, item)

        return item


class RandomBrightness(Transform):
    """Randomly change the image brightness."""

    def __init__(
        self,
        max_delta: float,
        max_val: Union[float, None] = None,
        min_val: Union[float, None] = None,
    ) -> None:
        """
        Parameters
        ----------
        max_delta : float
            Maximum difference between current image intensity and new image intensity.
            A delta will be sampled in `[0, max_delta]` range for each image independently.
            Then, that delta is added to the current image intensity.
        max_val : Union[float, None], optional
            `Clamp`'s max_val, by default None
        min_val : Union[float, None], optional
            `Clamp`'s min_val, by default None
        """
        super().__init__()

        self._max_delta = max_delta
        self._clamp = Clamp(min_val, max_val)

    def __call__(self, item: torch.Tensor) -> torch.Tensor:
        """Apply random brightness to Tensor.

        Parameters
        ----------
        item : torch.Tensor
            Tensor of shape ...xCxHxW

        Returns
        -------
        torch.Tensor
            Input item with changed brightness.
        """
        # sample delta per image
        batch_shape = item.shape[:-3] + (1, 1, 1)
        delta = (
            -self._max_delta
            + 2 * torch.rand(batch_shape, device=item.device) * self._max_delta
        )

        # apply intensity change and clamp
        item = item.add_(delta)
        item = self._clamp(item)

        return item


class RandomContrast(Transform):
    """Randomly change contrast of images."""

    def __init__(
        self,
        max_factor_delta: float,
        max_val: Union[float, None] = None,
        min_val: Union[float, None] = None,
    ) -> None:
        """
        Parameters
        ----------
        max_factor_delta : float
            A factor will be sampled in `[1 - max_delta, 1 + max_delta]` range for each image independently.
            That factor will be used to change the contrast as in `(I - mean) * factor + mean`.
        max_val : Union[float, None], optional
            `Clamp`'s max_val, by default None
        min_val : Union[float, None], optional
            `Clamp`'s min_val, by default None
        """
        super().__init__()

        self._max_factor_delta = max_factor_delta
        self._clamp = Clamp(min_val, max_val)

    def __call__(self, item: torch.Tensor) -> torch.Tensor:
        """Apply contrast change to Tensor

        Parameters
        ----------
        item : torch.Tensor
            Tensor of shape ...xCxHxW

        Returns
        -------
        torch.Tensor
            Input item with changed contrast.
        """

        # compute mean per (batch, channel)
        mean_values = torch.mean(item, (-2, -1), keepdim=True)

        # sample factors per image
        batch_shape = item.shape[:-3] + (1, 1, 1)
        factors = (
            1.0
            - self._max_factor_delta
            + 2 * torch.rand(batch_shape, device=item.device) * self._max_factor_delta
        )

        # apply contrast change and clamp
        item = (item - mean_values) * factors + mean_values
        item = self._clamp(item)

        return item


class MotionBlur(Transform):
    """Blur 2D images using the motion filter."""

    def __init__(
        self,
        kernel_size: int,
        angle: Union[torch.Tensor, float],
        direction: Union[torch.Tensor, float],
    ) -> None:
        """
        Parameters
        ----------
        kernel_size : int
            Kernel size of motion filter.
        angle : Union[torch.Tensor, float]
            Angle of the motion.
            If float, the same angle will be used for every inputs.
            If tensor of size B, the `angle[i]` will be applied to `input[i]`.
        direction : Union[torch.Tensor, float]
            Direction of the motion :
              -1 for direction towards the back.
              +1 for direction towards the front.
              0  for uniform direction.
            If float, the same direction will be used for every inputs.
            If tensor of size B, the `direction[i]` will be applied to `input[i]`.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.angle: float = angle
        self.direction: float = direction

    @staticmethod
    def motion_blur(
        input: torch.Tensor,
        kernel_size: int,
        angle: Union[float, torch.Tensor],
        direction: Union[float, torch.Tensor],
        mode: str = "bilinear",
    ) -> torch.Tensor:
        """Apply motion blur to input.

        Parameters
        ----------
        input : torch.Tensor
            Input as tensor of shape BxCxHxW.
        kernel_size : int
            See `MotionBlur.__init__`.
        angle : Union[torch.Tensor, float]
           See `MotionBlur.__init__`.
        direction : Union[torch.Tensor, float]
            See `MotionBlur.__init__`.
        mode : str, optional
            Sampling mode to use for kernel rotation. "bilinear" or "nearest".
        """
        kernel: torch.Tensor = MotionBlur._get_motion_kernel2d(
            kernel_size, angle, direction, mode, input.device
        )

        if input.size(0) > 1 and kernel.size(0) == 1:
            kernel = kernel.expand((input.size(0), -1, -1, -1))

        return MotionBlur._filter2d(input, kernel)

    @staticmethod
    def _filter2d(
        input: torch.Tensor,
        kernel: torch.Tensor,
    ) -> torch.Tensor:
        """Convolve a tensor with a 2d kernel."""
        # get shapes
        bi, ci, hi, wi = input.shape
        bk, ck, hk, wk = kernel.shape

        # check kernel is of odd size and large enough
        assert hk > 2
        assert wk > 2
        assert hk % 2 == 1
        assert wk % 2 == 1
        assert bi == bk
        assert ck == 1

        # prepare kernel for group convolution
        # TODO(Pierre) : re-check if line below is indeed unnecessary
        # kernel = kernel.expand(-1, ci, -1, -1)
        kernel = kernel.reshape(-1, 1, hk, wk)

        # prepare input for group convolution
        input = input.permute((1, 0, 2, 3))

        # convolve the tensor with the kernel.
        output = F.conv2d(
            input,
            kernel,
            groups=bk,
            padding="same",
            stride=1,
        )

        # permute back to original shape
        output = output.permute((1, 0, 2, 3))

        return output

    @staticmethod
    def _get_motion_kernel2d(
        kernel_size: int,
        angle: Union[torch.Tensor, float],
        direction: Union[torch.Tensor, float] = 0.0,
        mode: str = "nearest",
        device: str = "cpu",
        dtype: torch.dtype = torch.float,
    ) -> torch.Tensor:
        """Return 2D motion blur kernel."""
        # check kernel size
        if not isinstance(kernel_size, int) or kernel_size % 2 == 0 or kernel_size < 3:
            raise TypeError("kernel_size must be an odd integer >= than 3")

        # check angle parameter
        if not isinstance(angle, torch.Tensor):
            angle = torch.tensor(angle, device=device, dtype=dtype)
        else:
            angle = angle.to(device=device, dtype=dtype)

        # check direction parameter
        if not isinstance(direction, torch.Tensor):
            direction = torch.tensor(direction, device=device, dtype=dtype)
        else:
            direction = direction.to(device=device, dtype=dtype)

        # add batch dimension if needed
        if angle.dim() == 0:
            angle = angle.unsqueeze(0)
        if direction.dim() == 0:
            direction = direction.unsqueeze(0)

        # build kernel with angle 0
        kernel_tuple: Tuple[int, int] = (kernel_size, kernel_size)
        direction = (torch.clamp(direction, -1.0, 1.0) + 1.0) / 2.0

        k = torch.stack(
            [
                (direction + ((1 - 2 * direction) / (kernel_size - 1)) * i)
                for i in range(kernel_size)
            ],
            dim=-1,
        )
        kernel = torch.nn.functional.pad(
            k[:, None], [0, 0, kernel_size // 2, kernel_size // 2, 0, 0]
        )

        assert kernel.shape == torch.Size([direction.size(0), *kernel_tuple])

        # add channel
        kernel = kernel.unsqueeze(1)

        # rotate (counter-clockwise) kernel by given angle
        kernel = MotionBlur._rotate_kernels(
            kernel,
            angle,
            mode=mode,
            padding_mode="zeros",
        )

        # normalize
        kernel = kernel / kernel.sum(dim=(2, 3), keepdim=True)

        return kernel

    @staticmethod
    def _rotate_kernels(
        kernels: torch.Tensor,
        angles: torch.Tensor,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
    ) -> torch.Tensor:
        """Rotate kernels by provided angles."""

        sampler = HomographicSampler(angles.size(0), device=kernels.device)
        sampler.rotate(angles.unsqueeze(-1))
        return sampler.extract_crop(
            kernels,
            kernels.shape[-2:],
            mode=mode,
            padding_mode=padding_mode,
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply motion blur.

        Parameters
        ----------
        x : torch.Tensor
            Images as BxCxHxW tensor.

        Returns
        -------
        torch.Tensor
            Motion blurred images.
        """
        return MotionBlur.motion_blur(
            x,
            self.kernel_size,
            self.angle,
            self.direction,
        )


class RandomMotionBlur(Transform):
    """Apply random motion blur filter."""

    def __init__(
        self,
        kernel_size: int,
        angle: Union[float, Tuple[float, float]] = (0, 2 * torch.pi),
        direction: Union[float, Tuple[float, float]] = 0.0,
    ) -> None:
        """
        Parameters
        ----------
        kernel_size : int
            Size of motion blur kernel.
        angle : Union[float, Tuple[float, float]], optional
            Angle to use, or range to uniformely sample from, by default (0, 2 * torch.pi)
        direction : Union[float, Tuple[float, float]], optional
            Direction to use, or range to uniformely sample from, by default 0.0
        """
        super().__init__()

        self._kernel_size = kernel_size
        self._angle = angle
        self._direction = direction

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply randomly generated motion blur to images.

        Parameters
        ----------
        x : torch.Tensor
            Image as BxCxHxW tensor.

        Returns
        -------
        torch.Tensor
            Motion blurred images.
        """
        batch_size = x.size(0)

        # sample angles
        if isinstance(self._angle, float):
            angle = self._angle
        else:
            angle = self._angle[0] + torch.rand(batch_size, device=x.device) * (
                self._angle[1] - self._angle[0]
            )

        # sample positions
        if isinstance(self._direction, float):
            direction = self._direction
        else:
            direction = self._direction[0] + torch.rand(batch_size, device=x.device) * (
                self._direction[1] - self._direction[0]
            )

        return MotionBlur.motion_blur(
            x,
            self._kernel_size,
            angle,
            direction,
        )
