# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# original source : https://github.com/zju3dv/LoFTR

import os

import h5py
import numpy as np
import torch
import torch.utils as utils
from einops import repeat
from kornia.utils import create_meshgrid

from silk.datasets.scannet.helper import ScanNet
from torch.utils.data.dataset import ConcatDataset


@torch.no_grad()
def mask_pts_at_padded_regions(grid_pt, mask):
    """For megadepth dataset, zero-padding exists in images"""
    mask = repeat(mask, "n h w -> n (h w) c", c=2)
    grid_pt[~mask.bool()] = 0
    return grid_pt


@torch.no_grad()
def warp_kpts(kpts0, depth0, depth1, T_0to1, K0, K1):
    """Warp kpts0 from I0 to I1 with depth, K and Rt
    Also check covisibility and depth consistency.
    Depth is consistent if relative error < 0.2 (hard-coded).

    Args:
        kpts0 (torch.Tensor): [N, L, 2] - <x, y>,
        depth0 (torch.Tensor): [N, H, W],
        depth1 (torch.Tensor): [N, H, W],
        T_0to1 (torch.Tensor): [N, 3, 4],
        K0 (torch.Tensor): [N, 3, 3],
        K1 (torch.Tensor): [N, 3, 3],
    Returns:
        calculable_mask (torch.Tensor): [N, L]
        warped_keypoints0 (torch.Tensor): [N, L, 2] <x0_hat, y1_hat>
    """
    kpts0_long = kpts0.round().long()

    # Sample depth, get calculable_mask on depth != 0
    kpts0_depth = torch.stack(
        [
            depth0[i, kpts0_long[i, :, 1], kpts0_long[i, :, 0]]
            for i in range(kpts0.shape[0])
        ],
        dim=0,
    )  # (N, L)
    nonzero_mask = kpts0_depth != 0

    # Unproject
    kpts0_h = (
        torch.cat([kpts0, torch.ones_like(kpts0[:, :, [0]])], dim=-1)
        * kpts0_depth[..., None]
    )  # (N, L, 3)
    kpts0_cam = K0.inverse() @ kpts0_h.transpose(2, 1)  # (N, 3, L)

    # Rigid Transform
    w_kpts0_cam = T_0to1[:, :3, :3] @ kpts0_cam + T_0to1[:, :3, [3]]  # (N, 3, L)
    w_kpts0_depth_computed = w_kpts0_cam[:, 2, :]

    # Project
    w_kpts0_h = (K1 @ w_kpts0_cam).transpose(2, 1)  # (N, L, 3)
    w_kpts0 = w_kpts0_h[:, :, :2] / (
        w_kpts0_h[:, :, [2]] + 1e-4
    )  # (N, L, 2), +1e-4 to avoid zero depth

    # Covisible Check
    h, w = depth1.shape[1:3]
    covisible_mask = (
        (w_kpts0[:, :, 0] > 0)
        * (w_kpts0[:, :, 0] < w - 1)
        * (w_kpts0[:, :, 1] > 0)
        * (w_kpts0[:, :, 1] < h - 1)
    )
    w_kpts0_long = w_kpts0.long()
    w_kpts0_long[~covisible_mask, :] = 0

    w_kpts0_depth = torch.stack(
        [
            depth1[i, w_kpts0_long[i, :, 1], w_kpts0_long[i, :, 0]]
            for i in range(w_kpts0_long.shape[0])
        ],
        dim=0,
    )  # (N, L)
    consistent_mask = (
        (w_kpts0_depth - w_kpts0_depth_computed) / w_kpts0_depth
    ).abs() < 0.2
    valid_mask = nonzero_mask * covisible_mask * consistent_mask

    return valid_mask, w_kpts0


@torch.no_grad()
def spvs_coarse(data, descriptors):
    """
    Update:
        data (dict): {
            "conf_matrix_gt": [N, hw0, hw1],
            'spv_b_ids': [M]
            'spv_i_ids': [M]
            'spv_j_ids': [M]
            'spv_w_pt0_i': [N, hw0, 2], in original image resolution
            'spv_pt1_i': [N, hw1, 2], in original image resolution
        }

    NOTE:
        - for scannet dataset, there're 3 kinds of resolution {i, c, f}
        - for megadepth dataset, there're 4 kinds of resolution {i, i_resize, c, f}
    """
    # 1. misc
    device = data["image_0"].device
    N, _, H0, W0 = data["image_0"].shape
    _, _, H1, W1 = data["image_1"].shape

    assert H0 == H1
    assert W0 == W1

    DN, DH0, DW0 = data["depth_0"].shape
    _, DH1, DW1 = data["depth_1"].shape

    assert N == DN
    assert DH0 == DH1
    assert DW0 == DW1

    h, w = descriptors.shape[-2:]

    scale_h = DH0 / h
    scale_w = DW0 / w

    assert scale_h == scale_w
    scale = scale_w

    # 2. warp grids
    # create kpts in meshgrid and resize them to image resolution
    grid_pt0_c = (
        create_meshgrid(h, w, False, device).reshape(1, h * w, 2).repeat(N, 1, 1)
    )  # [N, hw, 2]
    # grid_pt0_c += 0.5
    grid_pt0_i = scale * grid_pt0_c

    grid_pt1_c = (
        create_meshgrid(h, w, False, device).reshape(1, h * w, 2).repeat(N, 1, 1)
    )
    # grid_pt1_c += 0.5
    grid_pt1_i = scale * grid_pt1_c

    # mask padded region to (0, 0), so no need to manually mask conf_matrix_gt
    if "mask0" in data:
        grid_pt0_i = mask_pts_at_padded_regions(grid_pt0_i, data["mask0"])
        grid_pt1_i = mask_pts_at_padded_regions(grid_pt1_i, data["mask1"])

    # warp kpts bi-directionally and resize them to coarse-level resolution
    # (no depth consistency check, since it leads to worse results experimentally)
    # (unhandled edge case: points with 0-depth will be warped to the left-up corner)
    _, w_pt0_i = warp_kpts(
        grid_pt0_i,
        data["depth_0"],
        data["depth_1"],
        data["T_0to1"],
        data["K_0"],
        data["K_1"],
    )
    _, w_pt1_i = warp_kpts(
        grid_pt1_i,
        data["depth_1"],
        data["depth_0"],
        data["T_1to0"],
        data["K_1"],
        data["K_0"],
    )
    w_pt0_c = w_pt0_i / scale
    w_pt1_c = w_pt1_i / scale

    return w_pt0_c, w_pt1_c


def read_scannet_depth(frame):
    depth = frame.depth
    depth = torch.from_numpy(depth).float()  # (h, w)
    return depth


def scannet_all(dataset_path, npz_path, mode, min_overlap_score):
    npz_names = os.listdir(npz_path)
    helper = ScanNet(dataset_path)
    return ConcatDataset(
        ScanNetDataset(
            helper,
            os.path.join(npz_path, name),
            mode=mode,
            min_overlap_score=min_overlap_score,
            augment_fn=None,
        )
        for name in npz_names
        if name.endswith(".npz")
    )


class ScanNetDataset(utils.data.Dataset):
    def __init__(
        self,
        helper,
        npz_path,
        mode="train",
        min_overlap_score=0.4,
        augment_fn=None,
    ):
        """Manage one scene of ScanNet Dataset.
        Args:
            root_dir (str): ScanNet root directory that contains scene folders.
            npz_path (str): {scene_id}.npz path. This contains image pair information of a scene.
            intrinsic_path (str): path to depth-camera intrinsic file.
            mode (str): options are ['train', 'val', 'test'].
            augment_fn (callable, optional): augments images with pre-defined visual effects.
            pose_dir (str): ScanNet root directory that contains all poses.
                (we use a separate (optional) pose_dir since we store images and poses separately.)
        """
        super().__init__()
        self.mode = mode
        self._helper = helper

        assert npz_path.endswith(".npz")
        h5_path = npz_path[:-4] + ".h5"

        if not os.path.exists(h5_path):
            # cache numpy original data to h5py
            with h5py.File(h5_path, "w") as f:
                with np.load(npz_path) as data:
                    for name in data.keys():
                        f[name] = data[name]

        self._h5file = h5py.File(h5_path, "r")
        self.data_names = self._h5file["name"]

        # filter out pairs with low overlap score
        if "score" in self._h5file.keys() and mode not in ["val" or "test"]:
            kept_mask = self._h5file["score"][()] > min_overlap_score
            self.kept_indices = np.nonzero(kept_mask)[0]
        else:
            self.kept_indices = np.arange(len(self.data_names), dtype=np.uint64)

        self.augment_fn = augment_fn if mode == "train" else None

    def __len__(self):
        return len(self.kept_indices)

    def __getitem__(self, idx):
        data_name = self.data_names[self.kept_indices[idx]]
        space_id, scene_id, frame_index_0, frame_index_1 = data_name

        # get frames
        scan = self._helper.space(space_id)[scene_id]
        frame0 = scan.frames[frame_index_0]
        frame1 = scan.frames[frame_index_1]

        # get depth and images
        depth0 = read_scannet_depth(frame0)
        depth1 = read_scannet_depth(frame1)
        image0 = frame0.color
        image1 = frame1.color

        # get intrinsics
        K_0 = K_1 = torch.tensor(
            scan.frames.header["intrinsic_depth"][:3, :3],
            dtype=torch.float,
        )

        # get extrinsics
        pose0 = torch.tensor(
            frame0.header["camera_to_world"],
            dtype=torch.float,
        )
        pose1 = torch.tensor(
            frame1.header["camera_to_world"],
            dtype=torch.float,
        )

        T_0to1 = pose1.inverse() @ pose0
        T_1to0 = pose0.inverse() @ pose1

        return image0, image1, depth0, depth1, T_0to1, T_1to0, K_0, K_1
