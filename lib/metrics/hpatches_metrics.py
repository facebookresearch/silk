# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from silk.logger import LOG
from silk.models.superpoint_utils import keep_true_points, warp_points
from torchmetrics import Metric

DEFAULT_DEVICE = "cuda:0"


class AvgShapeCount(Metric):
    def __init__(self, dim=0, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self._sum = 0.0
        self._n = 0
        self._dim = (dim,) if isinstance(dim, int) else tuple(dim)

    def update(self, *tensors):
        for tensor in tensors:
            for ten in tensor:
                self.update_one(ten)

    def update_one(self, tensor):
        v = 1
        for d in self._dim:
            v *= tensor.shape[d]
        self._sum += v
        self._n += 1

    def compute(self):
        return torch.tensor(self._sum / self._n)


class Repeatability(Metric):
    def __init__(self, distance_thresh=3, dist_sync_on_step=False, ordering="xy"):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        # TODO(Pierre): Make metric's device inferred from inputs.
        self.to_device = torch.device(
            DEFAULT_DEVICE if torch.cuda.is_available() else "cpu"
        )
        self.add_state("repeatability", default=[])
        self.distance_thresh = distance_thresh
        self.ordering = ordering

        assert self.ordering in {"xy", "yx"}

    def update(
        self,
        output_points,
        output_warped_points,
        img_shape,
        warped_img_shape,
        true_homography,
    ):
        for args in zip(
            output_points,
            output_warped_points,
            img_shape,
            warped_img_shape,
            true_homography,
        ):
            self.update_one(*args)

    def update_one(
        self,
        output_points,
        output_warped_points,
        img_shape,
        warped_img_shape,
        true_homography,
    ):
        assert (img_shape == warped_img_shape).all()

        output_points = output_points.to(self.to_device)
        output_warped_points = output_warped_points.to(self.to_device)

        # clean
        if self.ordering == "xy":
            output_points = output_points.T[:2]
            output_warped_points = output_warped_points.T[:2]
        else:
            output_points = output_points.T[[1, 0]]
            output_warped_points = output_warped_points.T[[1, 0]]

        # convert homography from 1 x 3 x 3 to 3 x 3 matrix
        true_homography = true_homography.to(self.to_device).float()

        # only keep points from warped image if they would be present in original image
        warped_pred_points, _ = keep_true_points(
            output_warped_points,
            torch.linalg.inv(true_homography),
            img_shape,
        )

        # only keep points from original image if they would be present in warped image
        output_points, _ = keep_true_points(
            output_points,
            true_homography,
            warped_img_shape,
        )

        # warp the original image output_points with the true homography
        true_warped_pred_points = warp_points(output_points, true_homography)

        # need to transpose to properly compute repeatability
        warped_pred_points = warped_pred_points.T
        true_warped_pred_points = true_warped_pred_points.T

        # compute the repeatability

        # get the number of predicted points in both images in the pair
        original_num = true_warped_pred_points.shape[0]
        warped_num = warped_pred_points.shape[0]

        # compute the norm
        assert true_warped_pred_points.shape[1] == 2
        assert warped_pred_points.shape[1] == 2

        norm = torch.linalg.norm(
            true_warped_pred_points[:, :2].unsqueeze(1)
            - warped_pred_points[:, :2].unsqueeze(0),
            dim=2,
        )

        # count the number of points with norm distance less than distance_thresh
        count1 = 0
        count2 = 0

        if original_num != 0:
            min1 = torch.min(norm, 0).values
            count1 = torch.sum(min1 <= self.distance_thresh)
        if warped_num != 0:
            min2 = torch.min(norm, 1).values
            count2 = torch.sum(min2 <= self.distance_thresh)
        if original_num + warped_num > 0:
            repeatability = (count1 + count2) / (original_num + warped_num)
            self.repeatability.append(repeatability)

    def compute(self):
        repeatability = torch.tensor(
            self.repeatability,
            dtype=torch.float32,
            device=self.to_device,
        )
        return torch.mean(repeatability)


class HomographyEstimation(Metric):
    def __init__(
        self,
        correctness_thresh=3,
        dist_sync_on_step=False,
        auc=False,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("homography_estimation", default=[], dist_reduce_fx="cat")
        self.add_state("errors", default=[], dist_reduce_fx="cat")

        # TODO(Pierre): Make metric's device inferred from inputs.
        self.to_device = torch.device(
            DEFAULT_DEVICE if torch.cuda.is_available() else "cpu"
        )
        self.correctness_thresh = correctness_thresh
        self._auc = auc

    @staticmethod
    def compute_auc(errors, thresholds):
        sort_idx = np.argsort(errors)
        errors = np.array(errors.copy())[sort_idx]
        recall = (np.arange(len(errors)) + 1) / len(errors)
        errors = np.r_[0.0, errors]
        recall = np.r_[0.0, recall]

        aucs = []
        for thres in thresholds:
            last_index = np.searchsorted(errors, thres)
            rec = np.r_[recall[:last_index], recall[last_index - 1]]
            err = np.r_[errors[:last_index], thres]
            aucs.append(np.trapz(rec, x=err) / thres)
        return aucs

    def update(
        self,
        img_shape,
        estimated_homography,
        true_homography,
    ):
        for args in zip(
            img_shape,
            estimated_homography,
            true_homography,
        ):
            self.update_one(*args)

    # TODO(Pierre): Make homography estimation only work on two homographies.
    def update_one(
        self,
        img_shape,
        estimated_homography,
        true_homography,
    ):
        if estimated_homography is None:
            correctness = 0.0
            self.homography_estimation.append(correctness)
            self.errors.append(self.correctness_thresh + 1.0)
            LOG.warning("invalid matrix provided")
            return

        true_homography = true_homography.to(self.to_device).float()
        estimated_homography = estimated_homography.to(self.to_device).float()

        # img_shape = img_shape - 64

        corners = torch.tensor(
            [
                [0, 0, 1],
                [img_shape[1] - 1, 0, 1],
                [0, img_shape[0] - 1, 1],
                [img_shape[1] - 1, img_shape[0] - 1, 1],
            ],
            dtype=torch.float32,
            device=self.to_device,
        )

        # apply the true homography and the estimated homography to the corners
        real_warped_corners = torch.mm(corners, torch.transpose(true_homography, 0, 1))
        real_warped_corners = real_warped_corners[:, :2] / real_warped_corners[:, 2:]

        warped_corners = torch.mm(corners, torch.transpose(estimated_homography, 0, 1))
        warped_corners = warped_corners[:, :2] / warped_corners[:, 2:]

        mean_dist = torch.mean(
            torch.linalg.norm(real_warped_corners - warped_corners, dim=1)
        )

        # considered correct if mean distance is below a given threshold
        correctness = mean_dist <= self.correctness_thresh

        self.homography_estimation.append(correctness)
        self.errors.append(mean_dist)

    def compute(self):
        # return the average homography across all pairs of images
        if self._auc:
            errors = (
                torch.tensor(
                    self.errors,
                    dtype=torch.float32,
                    device=self.to_device,
                )
                .detach()
                .cpu()
                .numpy()
            )
            auc = HomographyEstimation.compute_auc(errors, [self.correctness_thresh])[0]
            return torch.tensor(
                auc,
                dtype=torch.float32,
                device=self.to_device,
            )

        homography_estimation = torch.tensor(
            self.homography_estimation,
            dtype=torch.float32,
            device=self.to_device,
        )
        homography_avg = torch.mean(homography_estimation)

        return homography_avg


class MeanMatchingAccuracy(Metric):
    def __init__(
        self,
        threshold=3,
        dist_sync_on_step=False,
        ordering="xy",
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self._threshold = threshold
        self._cum_acc = 0
        self._n_matches = 0
        self._n_points = 0
        self._ordering = ordering

        assert ordering in {"xy", "yx"}

        # TODO(Pierre): Make metric's device inferred from inputs.
        self.to_device = torch.device(
            DEFAULT_DEVICE if torch.cuda.is_available() else "cpu"
        )

    @staticmethod
    def good_matches_mask(
        matched_keypoints, warped_matched_keypoints, homography, threshold, ordering
    ):
        assert ordering in {"yx", "xy"}

        if ordering == "xy":
            matched_keypoints = matched_keypoints[:, :2]
            warped_matched_keypoints = warped_matched_keypoints[:, :2]
        else:
            matched_keypoints = matched_keypoints[:, [1, 0]]
            warped_matched_keypoints = warped_matched_keypoints[:, [1, 0]]

        true_warped_keypoints = warp_points(
            matched_keypoints.T,
            homography,
        ).T

        mask_good = ((true_warped_keypoints - warped_matched_keypoints) ** 2).sum(
            dim=1
        ).sqrt() <= threshold

        return mask_good

    def update(
        self,
        matched_keypoints,
        warped_matched_keypoints,
        true_homography,
    ):
        for args in zip(
            matched_keypoints,
            warped_matched_keypoints,
            true_homography,
        ):
            self.update_one(*args)

    # TODO(Pierre): Make homography estimation only work on two homographies.
    def update_one(
        self,
        matched_keypoints,
        warped_matched_keypoints,
        true_homography,
    ):
        assert len(matched_keypoints) == len(warped_matched_keypoints)

        true_homography = true_homography.to(self.to_device).float()

        mask_good = MeanMatchingAccuracy.good_matches_mask(
            matched_keypoints,
            warped_matched_keypoints,
            true_homography,
            self._threshold,
            self._ordering,
        )

        if mask_good.numel() > 0:
            acc = mask_good.float().mean().item()
        else:
            acc = 0.0

        self._cum_acc += acc
        self._n_matches += 1
        self._n_points += 2 * mask_good.shape[0]

    def compute(self):
        print(f"n_points : {self._n_points / (2 * self._n_matches)}")

        # return the mean matching accuracy
        return torch.tensor(
            self._cum_acc / self._n_matches,
            dtype=torch.float32,
            device=self._device,
        )
