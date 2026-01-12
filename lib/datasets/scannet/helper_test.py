# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
import torch
from silk.datasets.scannet.helper import Frame, Frames, Scan, ScanNet, Space
from silk.test.util import temporary_directory

DATASET_PATH = "/datasets01/scannet/082518/scans"


class _UnitTests(unittest.TestCase):
    @staticmethod
    def _get_all(arr):
        return tuple(arr[i] for i in range(len(arr)))

    @staticmethod
    def _get_dataset(cache=None):
        return ScanNet(DATASET_PATH, cache_path=cache)

    def _test_scannet_hierarchy(self, ds):
        self.assertEqual(ds.n_spaces, 707)
        self.assertEqual(ds.n_scans, 1513)
        self.assertEqual(ds.n_frames, 2477378)

        # check if parents are consistent
        idx = ds.n_frames // 2
        frame = ds.frame(idx)
        frames = frame.parent
        scan = frames.parent
        space = scan.parent

        self.assertIsInstance(frame, Frame)
        self.assertIsInstance(frames, Frames)
        self.assertIsInstance(scan, Scan)
        self.assertIsInstance(space, Space)

        self.assertIn(frame, _UnitTests._get_all(frames))
        self.assertIn(frame, _UnitTests._get_all(scan))
        self.assertIn(scan, _UnitTests._get_all(space))

    def _test_scannet_frame(self, ds):
        # get last frame
        frame = ds.frame(-1)
        color = frame.color
        depth = frame.depth

        # check rank
        self.assertEqual(len(color.shape), 3)
        self.assertEqual(len(depth.shape), 2)

        # check channel size
        self.assertEqual(color.shape[2], 3)

        # check dtype
        self.assertEqual(color.dtype, np.uint8)
        self.assertEqual(depth.dtype, np.float64)

    def _test_scannet_txt(self, ds):
        # get last scan
        scan = ds.scan(-1)
        txt = scan.txt

        self.assertSetEqual(
            {
                "axisAlignment",
                "colorHeight",
                "colorWidth",
                "depthHeight",
                "depthWidth",
                "fx_color",
                "fx_depth",
                "fy_color",
                "fy_depth",
                "mx_color",
                "mx_depth",
                "my_color",
                "my_depth",
                "numColorFrames",
                "numDepthFrames",
                "numIMUmeasurements",
                "sceneType",
            },
            set(txt.keys()),
        )

    def _test_scannet_camera_pose(self, ds):
        # get R, T from Scannet
        for i in range(3):
            for j in range(0, 50, 10):
                frame = ds.scan(i)._frames[j]
                cam_to_world = frame.header["camera_to_world"]
                world_to_cam = torch.tensor(np.linalg.inv(cam_to_world))
                self.assertEqual(world_to_cam.size(), torch.Size([4, 4]))

    def _test_all(self, ds):
        self._test_scannet_hierarchy(ds)
        self._test_scannet_frame(ds)
        self._test_scannet_txt(ds)
        self._test_scannet_camera_pose(ds)

    def test_uncached_scannet(self):
        ds = _UnitTests._get_dataset()
        self._test_all(ds)

    def test_cached_scannet(self):
        with temporary_directory() as dirpath:
            ds = _UnitTests._get_dataset(dirpath)
            self._test_all(ds)


if __name__ == "__main__":
    unittest.main()
