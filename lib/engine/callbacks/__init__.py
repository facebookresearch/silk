# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import pytorch_lightning as pl
from silk.transforms.abstract import NamedContext, Transform


class BatchTransform(pl.Callback):
    """Callback that applies a transform to `on_{train,validation,test,predict}_batch_{start,end}` hooks."""

    VALID_LOOPS = {"train", "validation", "test", "predict"}

    def __init__(
        self,
        transform: Transform,
        loop: str,
        end: bool,
        log_as: Optional[str] = None,
    ) -> None:
        """

        Parameters
        ----------
        transform : Transform
            Transform to apply to the batch.
        loop : str
            Type of loop on which the transform will be used. Valid values are "train", "validation", "test" or "predict".
        end : bool
            Specifiy if the transform will be applied before or after the step function.
        log_as : Optional[str], optional
            Name of transformed value to log, by default None. Do not log if name is None.

        Raises
        ------
        RuntimeError
            Raised if invalid loop name is provided.
        """
        super().__init__()
        self._transform = transform
        self._log_as = log_as

        if loop not in BatchTransform.VALID_LOOPS:
            raise RuntimeError(
                f'invalid loop name found "{loop}", valid ones are {BatchTransform.VALID_LOOPS}'
            )

        end = "end" if end else "start"

        hook_method_name = f"on_{loop}_batch_{end}"
        process_method = getattr(self, f"_process_{end}")
        setattr(self, hook_method_name, process_method)

    def _process_start(self, trainer, pl_module, batch, batch_idx, unused=0):
        self._process(pl_module, batch)

    def _process_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        self._process(pl_module, outputs)

    def _process(self, pl_module, batch):
        if batch is not None:
            transform = self._transform.to(pl_module.device)
            result = transform(batch)
            if (result is not None) and (self._log_as is not None):
                if isinstance(result, NamedContext):
                    pl_module.log(
                        self._log_as, result[self._log_as], on_step=True, on_epoch=False
                    )
                else:
                    pl_module.log(self._log_as, result, on_step=True, on_epoch=False)
