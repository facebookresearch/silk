# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import random
from typing import Any, Dict, Iterable, List, Union

import torch
from pytorchvideo.data.video import Video
from silk.transforms.abstract import Transform


class VideoToImageBatch(Transform):
    """Transform PyTorchVideo video clip data as a batch of images."""

    def __call__(self, video_data: Dict[str, Any]) -> torch.Tensor:
        """

        Parameters
        ----------
        video_data : Dict[str, Any]
            Dictionay which contains a key "video" mapping to a CxTxHxW tensor.

        Returns
        -------
        torch.Tensor
            Batch of images as a TxCxHxW tensor.
        """
        video = video_data["video"]
        video = video.permute(1, 0, 2, 3)  # CTHW -> TCHW
        return video


class Stream(Iterable):
    """Create an iterable streaming clips of video data."""

    def __init__(
        self,
        video: Video,
        clip_duration: float,
        clip_transform: Transform = None,
        **get_clip_kwargs: Dict[str, Any],
    ) -> None:
        """
        Parameters
        ----------
        video : Video
            PyTorchVideo video instance to stream.
        clip_duration : float
            Maximum duration (in seconds) of the returned clip at every iteration.
        clip_transform : Transform, optional
            Optional transform to apply to each clip, by default None
        get_clip_kwargs : Dict[str, Any]
            Arguments to pass to the underlying video `get_clip` method.
        """
        super().__init__()
        self._clip_duration = clip_duration
        self._clip_transform = clip_transform
        self._video = video
        self._get_clip_kwargs = get_clip_kwargs

    def __iter__(self):
        current_time = 0.0
        while current_time < self._video.duration:
            next_time = min(
                self._video.duration,
                current_time + self._clip_duration,
            )
            video_data = self._video.get_clip(
                current_time,
                next_time,
                **self._get_clip_kwargs,
            )
            current_time = next_time

            if self._clip_transform:
                video_data = self._clip_transform(video_data)

            yield video_data

    @property
    def video(self):
        return self._video


class Streamed(Transform):
    """Apply a video transform in a streamed fashion (useful for large videos)."""

    def __init__(
        self,
        clip_duration: float,
        clip_transform: Transform = None,
        return_iterable: bool = False,
        **get_clip_kwargs,
    ) -> None:
        """
        Parameters
        ----------
        clip_duration : float
            Maximum duration (in seconds) of the transformed clip at every iteration.
        clip_transform : Transform, optional
            Optional transform to apply to each clip, by default None.
        return_iterable : bool, optional
            Decides if transform should return an iterable (more control over looping) or the iterated result, by default False.
        """
        super().__init__()
        self._clip_transform = clip_transform
        self._clip_duration = clip_duration
        self._return_iterable = return_iterable
        self._get_clip_kwargs = get_clip_kwargs

    def __call__(self, video: Video) -> Union[Stream, List[Any]]:
        stream = Stream(
            video,
            self._clip_duration,
            self._clip_transform,
            **self._get_clip_kwargs,
        )
        if self._return_iterable:
            return stream
        return list(stream)


class GetClipVideoWrapper(Transform):
    """Transform that extracts a clip from a video."""

    class WrappedVideo(Video):
        def __init__(self, video: Video, **get_clip_kwargs) -> None:
            self._video = video
            self._get_clip_kwargs = get_clip_kwargs

        @property
        def video(self):
            return self._video

        @property
        def duration(self):
            return self._video.duration

        def get_clip(
            self,
            start_sec,
            end_sec,
            **get_clip_kwargs,
        ):
            kwargs = {**self._get_clip_kwargs, **get_clip_kwargs}
            return self._video.get_clip(start_sec, end_sec, **kwargs)

    def __init__(self, **get_clip_kwargs) -> None:
        super().__init__()
        self._get_clip_kwargs = get_clip_kwargs

    def __call__(self, video: Video) -> Video:
        return GetClipVideoWrapper.WrappedVideo(video, **self._get_clip_kwargs)


class GetAllClip(Transform):
    """Get entire video tensor."""

    def __call__(self, video: Video):
        return video.get_clip(0, video.duration)


class UniformTemporalSubsampleFrameFilter:
    """Subsample frames uniformly."""

    def __init__(self, num_samples) -> None:
        self._num_samples: int = num_samples

    def __call__(self, idx: List[int]) -> List[int]:
        t = len(idx)
        indices = torch.linspace(0, t - 1, self._num_samples)
        indices = indices.long()
        return [idx[i] for i in indices]


class RandomTemporalShuffleFrameFilter:
    """Random subsampling of frames."""

    def __init__(self, num_samples) -> None:
        self._num_samples: int = num_samples

    def __call__(self, idx: List[int]) -> List[int]:
        random.shuffle(idx)
        return idx[: self._num_samples]
