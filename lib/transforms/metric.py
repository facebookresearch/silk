# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List

from silk.transforms.abstract import MethodCall

from torchmetrics import Metric


class MetricUpdateBase(MethodCall):
    """Transform that runs a `torchmetrics.Metric` on a `NamedContext`"""

    def __init__(
        self,
        name: str,
        metric: Metric,
        update_only: bool,
        *args_keys: List[Any],
        **kwargs_keys: Dict[str, Any],
    ) -> None:
        metric_class = metric.__class__
        method = metric_class.update if update_only else metric_class.forward
        super().__init__(name, metric, method, *args_keys, **kwargs_keys)
        if not isinstance(metric, Metric):
            raise RuntimeError("metric should be an instance of `torchmetrics.Metric`")

    @property
    def metric(self):
        return self._args_keys[0]


class MetricUpdate(MetricUpdateBase):
    def __init__(
        self,
        name: str,
        metric: Metric,
        *args_keys: List[Any],
        **kwargs_keys: Dict[str, Any],
    ) -> None:
        super().__init__(name, metric, True, *args_keys, **kwargs_keys)


class MetricUpdateAndCompute(MetricUpdateBase):
    def __init__(
        self,
        name: str,
        metric: Metric,
        *args_keys: List[Any],
        **kwargs_keys: Dict[str, Any],
    ) -> None:
        super().__init__(name, metric, False, *args_keys, **kwargs_keys)
