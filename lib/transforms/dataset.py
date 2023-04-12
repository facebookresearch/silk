# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Iterable

from silk.transforms.abstract import Transform


class LabelStringToClassId(Transform):
    """Converts string representation of a label to its integer representation between 0 and N-1."""

    def __init__(self, labels: Iterable[str]) -> None:
        """
        Parameters
        ----------
        labels : Iterable[str]
            Exhaustive list of string representation of labels.
        """
        super().__init__()
        self._label_to_id = {label: i for i, label in enumerate(labels)}

    def __call__(self, item: str) -> int:
        return self._label_to_id[item]
