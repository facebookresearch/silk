# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Iterable

import torch
from pytorch_lightning.trainer.trainer import Trainer as PLTrainer
from silk.engine.callbacks import BatchTransform
from silk.transforms.metric import MetricUpdate


Trainer = PLTrainer


Predictor = PLTrainer


class Benchmarker(PLTrainer):
    """Perform benchmarking loop by plugin metric callbacks in the `pytorch_lightning.trainer.trainer.Trainer` class.

    Examples
    --------

    ```python
    benchmarker = Benchmarker(metric_updates, params...)

    # metric updates are automatically run in the test loop
    benchmarker.test(model, dataloader)

    # get all the metric results (as a dictionary)
    results = benchmarker.compute_metrics()
    ```

    """

    def __init__(
        self, metric_updates: Dict[str, MetricUpdate], *args, **kwargs
    ) -> None:
        """
        Parameters
        ----------
        metric_updates : Dict[str, MetricUpdate]
            Dictionary of metric updates to apply during the test loop.
        """
        super().__init__(*args, **kwargs)
        self._metrics = {}

        # create & add callbacks to update the metrics
        for name, metric_update in metric_updates.items():
            transform_callback = BatchTransform(
                metric_update, "test", end=True, log_as=f"metric.{name}"
            )
            self.callbacks.append(transform_callback)
            self._metrics[name] = metric_update.metric

    @staticmethod
    def _to_python(result):
        if isinstance(result, torch.Tensor):
            if result.ndim == 0:
                return result.detach().cpu().item()
            else:
                return tuple(
                    Benchmarker._to_python(result[i]) for i in range(len(result))
                )
        elif isinstance(result, Dict):
            return {k: Benchmarker._to_python(result[k]) for k in result}
        elif isinstance(result, Iterable):
            return tuple(Benchmarker._to_python(r) for r in result)
        return result

    def compute_metrics(self):
        """Compute and return all provided metrics.

        Returns
        -------
        dict
            Mapped values of the provided metrics.
        """
        return Benchmarker._to_python(
            {name: metric.compute() for name, metric in self._metrics.items()}
        )
