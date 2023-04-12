# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Iterable, Optional

import torch
from torchmetrics import AveragePrecision


class KeypointDetectionAveragePrecision(AveragePrecision):
    """Average Precision computed using keypoint position and probabilities.

    This metric implementation has been inspired from :

    * SuperPoint Paper (Appendix A) : https://arxiv.org/pdf/1712.07629.pdf
    * Tensorflow Implementation : https://github.com/rpautrat/SuperPoint (`superpoint/evaluations/detector_evaluation.py`).
    """

    def __init__(
        self,
        distance_threshold: float = 2.0,
        allow_multimatching: bool = True,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
    ) -> None:
        """

        Parameters
        ----------
        distance_threshold : float, optional
            Distance from target under which a prediction would be considered valid, by default 2.0
        allow_multimatching : bool, optional
            Allow multiple prediction to match to the same target or not, by default True
        compute_on_step : bool, optional
            Option passed to parent class (c.f. `torchmetrics.Metric`), by default True
        dist_sync_on_step : bool, optional
            Option passed to parent class (c.f. `torchmetrics.Metric`), by default False
        process_group : Optional[Any], optional
            Option passed to parent class (c.f. `torchmetrics.Metric`), by default None
        """
        super().__init__(
            num_classes=1,
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
        )
        self._distance_threshold = distance_threshold
        self._allow_multimatching = allow_multimatching

    def update(
        self, preds: Iterable[torch.Tensor], target: Iterable[torch.Tensor]
    ) -> None:
        """Update the metric providing predictions and targets.

        Parameters
        ----------
        preds : Iterable[torch.Tensor]
            Iterable set of predictions. Each prediction should be a N x 3 tensor where `preds[:, 2:]` are the positions and `preds[:, 2]` are the probabilities.
        target : Iterable[torch.Tensor]
            Iterable set of targets. Each target should be a N x 2 tensor containing the targets positions.
        """
        ap_target = []
        ap_preds = []

        for p, t in zip(preds, target):
            # skip if no target and no prediction
            if t.numel() == 0 and p.numel() == 0:
                continue
            elif t.numel() == 0:
                # false positives (only predictions, no targets)
                pred_prob = p[:, 2]
                ap_preds.extend(pred_prob.detach().cpu().numpy())
                ap_target.extend([0] * p.shape[0])

            elif p.numel() == 0:
                # false negatives (only targets, no preditions)
                ap_preds.extend([0.0] * t.shape[0])
                ap_target.extend([1] * t.shape[0])
            else:
                # get probability
                pred_prob = p[:, 2]

                # get positions
                pred_posi = p[:, :2].unsqueeze(1)
                targ_posi = t.unsqueeze(0)

                # sort preds in descencing order
                pred_idx = torch.argsort(pred_prob, descending=True)
                pred_prob = pred_prob[pred_idx]
                pred_posi = pred_posi[pred_idx]

                # keep track of unmatched targets
                targ_not_matched = torch.ones(t.shape[0], dtype=bool, device=t.device)

                # compute pairwise distances and potential matches
                delta = pred_posi - targ_posi
                dist = torch.norm(delta, dim=2)
                matches = dist <= self._distance_threshold

                # assign predictions
                for i, m in enumerate(matches):
                    # in non-simplified version, we remove previously matched targets from potential matches
                    if not self._allow_multimatching:
                        m = torch.logical_and(m, targ_not_matched)

                    correct = torch.any(m)
                    if correct:
                        # get closest target among potential matches
                        ap_target_idx = torch.argmin(dist[i] * m)
                        ap_target.append(1)
                        targ_not_matched[ap_target_idx] = False
                    else:
                        ap_target.append(0)
                    ap_preds.append(pred_prob[i])

                # add false negative (unmatched targets)
                for _, not_matched in enumerate(targ_not_matched):
                    if not_matched:
                        ap_preds.append(0.0)
                        ap_target.append(1)

        super().update(
            torch.tensor(ap_preds, device=p.device),
            torch.tensor(ap_target, device=p.device),
        )
