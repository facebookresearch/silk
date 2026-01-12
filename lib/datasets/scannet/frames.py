# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, List, Set, Tuple, Union

from silk.datasets.scannet.helper import Frame, Scan, ScanNet
from torch.utils.data import Dataset


def _check_extractors(
    valid_extractors: Set[str],
    extractor: Union[List[str], str, None] = None,
):
    if extractor is None:
        extractors_to_check = set()
    elif isinstance(extractor, str):
        extractors_to_check = set((extractor,))  # noqa: C405
    else:
        extractors_to_check = set(extractor)

    invalid_extractors = extractors_to_check - valid_extractors
    if len(invalid_extractors) > 0:
        raise RuntimeError(
            f"invalid extractors found : {invalid_extractors}, valid extractors are : {valid_extractors}"
        )


# TODO(Pierre): Add documentation & tests.
class FramesDataset(Dataset):
    """PyTorch dataset at the Frame level."""

    EXTRACTORS = {
        "color": lambda x: x.color,
        "rgb": lambda x: x.color,
        "depth": lambda x: x.depth,
        "room": lambda x: x.scan.txt["sceneType"],
    }
    VALID_EXTRACTORS = set(EXTRACTORS.keys())

    def __init__(
        self,
        path: str,
        extractor: Union[List[str], str, None] = None,
        cache_path: Union[str, None] = None,
        scan_filter: Callable[[Scan], bool] = None,
    ) -> None:
        super().__init__()

        _check_extractors(FramesDataset.VALID_EXTRACTORS, extractor)

        self._extractor = extractor
        self._scannet = ScanNet(path, cache_path=cache_path, scan_filter=scan_filter)

    def __len__(self) -> int:
        return self._scannet.n_frames

    def _extract(self, frame) -> Tuple[Any]:
        if self._extractor is None:
            return frame
        elif isinstance(self._extractor, str):
            return FramesDataset.EXTRACTORS[self._extractor](frame)
        return tuple(
            FramesDataset.EXTRACTORS[extractor](frame) for extractor in self._extractor
        )

    def __getitem__(self, idx: int) -> Frame:
        frame = self._scannet.frame(idx)
        return self._extract(frame)
