# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch


def build_similarity_mask(descriptors, positions_0, positions_1, cell_size):
    device = descriptors.device

    similarity_mask = torch.zeros(
        (
            descriptors.shape[0] // 2,
            descriptors.shape[2],
            descriptors.shape[3],
            descriptors.shape[2],
            descriptors.shape[3],
        ),
        device=device,
        dtype=torch.bool,
    )

    # convert shape to use to check bounds
    image_shape = torch.tensor(
        [similarity_mask.shape[-2:]],
        device=similarity_mask.device,
    )
    # zip pair of positions correspondences
    positions = [torch.cat((p0, p1), dim=1) for p0, p1 in zip(positions_0, positions_1)]

    # gather batch index
    batch_i = torch.cat(
        [torch.full((len(p), 1), i, device=device) for i, p in enumerate(positions)],
        dim=0,
    )

    # filter out out-of-bound positions
    positions = torch.cat(positions, dim=0)
    positions = torch.floor(positions / cell_size).int()
    positions_N_2 = positions.view(-1, 2)
    mask = torch.logical_and(positions_N_2 >= 0, positions_N_2 < image_shape)
    mask = mask.reshape(-1, 4).all(dim=1)
    positions = positions[mask]
    batch_i = batch_i[mask]
    positions = torch.cat((batch_i, positions), dim=1)

    # convert to tuples for fast index filling
    positions = tuple(p for p in positions.T)
    similarity_mask[positions] = True

    return similarity_mask


class DescriptorLoss(torch.nn.Module):
    def __init__(
        self, margin_pos: float = 1.0, margin_neg: float = 0.2, lambda_d: float = 250.0
    ) -> None:
        # margin_neg : float, optional
        #     Margin threshold for negative pairs, by default 0.2
        # margin_pos : float, optional
        #     Margin threshold for positive pairs, by default 1.0
        # lambda_d : float, optional
        #     Positive pair relative weighting, by default 250.0
        super().__init__()
        self._margin_pos = margin_pos
        self._margin_neg = margin_neg
        self._lambda_d = lambda_d

    def forward(self, descriptors_0, descriptors_1, similarity_mask):
        dotprod = torch.einsum("bdij,bdkl->bijkl", descriptors_0, descriptors_1)
        val0 = torch.tensor(0, dtype=dotprod.dtype, device=dotprod.device)
        pos_loss = torch.maximum(val0, self._margin_pos - dotprod) * self._lambda_d
        neg_loss = torch.maximum(val0, dotprod - self._margin_neg)
        return torch.where(
            similarity_mask,
            pos_loss,
            neg_loss,
        ).mean()


class KeypointLoss(torch.nn.CrossEntropyLoss):
    pass
