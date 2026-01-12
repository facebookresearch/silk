# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, List, Tuple, Union

from silk.datasets.scannet.helper import Scan, ScanNet
from torch.utils.data import Dataset

from .frames import _check_extractors


class ScansDataset(Dataset):
    """PyTorch dataset at the Scan level."""

    EXTRACTORS = {
        "frames": lambda x: x.frames,
        "room": lambda x: x.txt["sceneType"],
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
        _check_extractors(ScansDataset.VALID_EXTRACTORS, extractor)

        self._extractor = extractor
        self._scannet = ScanNet(path, cache_path=cache_path, scan_filter=scan_filter)

    def __len__(self) -> int:
        return self._scannet.n_scans

    def _extract(self, scan) -> Tuple[Any]:
        if self._extractor is None:
            return scan
        elif isinstance(self._extractor, str):
            return ScansDataset.EXTRACTORS[self._extractor](scan)
        return tuple(
            ScansDataset.EXTRACTORS[extractor](scan) for extractor in self._extractor
        )

    def __getitem__(self, idx: int) -> Scan:
        scan = self._scannet.scan(idx)
        return self._extract(scan)
