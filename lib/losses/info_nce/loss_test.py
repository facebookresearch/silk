# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import jax
import jax.numpy as jnp

from silk.losses.info_nce.loss import (
    _scan_reduce,
    keep_mutual_correspondences_only,
    positions_to_unidirectional_correspondence,
)


class _UnitTests(unittest.TestCase):
    def assertJaxArrayEqual(self, arr_0, arr_1, precision_tolerance=None):
        if precision_tolerance:
            self.assertTrue(jnp.abs(arr_0 - arr_1).max() < precision_tolerance)
        else:
            self.assertTrue((arr_0 == arr_1).all())

    def test_positions_to_unidirectional_correspondence(self):
        cell_size = 3
        width = 44
        height = 21
        sub_width = int(width / cell_size) + int(width % cell_size > 0)
        sub_height = int(height / cell_size) + int(height % cell_size > 0)
        positions = jnp.array(
            [
                [0.0, 0.0],  # valid min
                [20.5, 43.5],  # valid max
                [10.5, 22.0],  # valid center
                [3.14, 38.1],  # valid random point
                [-1.0, 0.0],  # invalid min y
                [0.0, -1.0],  # invalid min x
                [21.0, 0.0],  # invalid max y
                [0.0, 45.0],  # invalid max x
                [-3.14, 91.2],  # invalid random point
            ]
        )

        corr = positions_to_unidirectional_correspondence(
            positions, sub_width, sub_height, cell_size
        )

        expected_corr = jnp.array(
            [
                0,
                sub_width * sub_height - 1,
                sub_width * int(10.5 / cell_size) + int(22.0 / cell_size),
                sub_width * int(3.14 / cell_size) + int(38.1 / cell_size),
                -1,
                -1,
                -1,
                -1,
                -1,
            ],
        )

        self.assertJaxArrayEqual(corr, expected_corr)

    def test_keep_mutual_correspondences_only(self):
        corr = jnp.array(
            [
                [1, -1],  # keep
                [-1, 0],  # keep
                [2, 2],  # keep
                [4, -1],  # keep
                [4, 3],  # suppress 0
                [5, 5],  # keep
                [-1, 5],  # suppress 1
                [8, -1],  # suppress 0
                [-1, 8],  # suppress 1
                [-1, -1],  # keep
            ]
        )

        corr_0, corr_1 = corr[:, 0], corr[:, 1]

        corr_0, corr_1 = keep_mutual_correspondences_only(corr_0, corr_1)

        expected_corr_0 = jnp.array([-1] * corr.shape[0])
        expected_corr_1 = jnp.array([-1] * corr.shape[0])

        expected_corr_0 = expected_corr_0.at[0].set(1)
        expected_corr_0 = expected_corr_0.at[2].set(2)
        expected_corr_0 = expected_corr_0.at[3].set(4)
        expected_corr_0 = expected_corr_0.at[5].set(5)

        expected_corr_1 = expected_corr_1.at[1].set(0)
        expected_corr_1 = expected_corr_1.at[2].set(2)
        expected_corr_1 = expected_corr_1.at[4].set(3)
        expected_corr_1 = expected_corr_1.at[5].set(5)

        self.assertJaxArrayEqual(corr_0, expected_corr_0)
        self.assertJaxArrayEqual(corr_1, expected_corr_1)

    def test_scan_reduce(self):
        def reducer(x0, x1):
            x0x1 = x0 @ x1.T
            output = (
                jnp.mean(x0x1, axis=1),
                jnp.min(x0x1, axis=1),
                jnp.max(x0x1, axis=1),
                jnp.median(x0x1, axis=1),
            )
            return output

        key = jax.random.PRNGKey(42)
        x0 = jax.random.uniform(key, (100, 16))
        x1 = jax.random.uniform(key, (100, 16))

        result_0 = _scan_reduce(
            x0, x1, reducer, block_size=10
        )  # multiple of block_size
        result_1 = _scan_reduce(
            x0, x1, reducer, block_size=11
        )  # not multiple of block_size

        x0x1 = x0 @ x1.T
        result_expected = (
            jnp.mean(x0x1, axis=1),
            jnp.min(x0x1, axis=1),
            jnp.max(x0x1, axis=1),
            jnp.median(x0x1, axis=1),
        )

        for i, r in enumerate(result_expected):
            self.assertEqual(r.shape, result_0[i].shape)
            self.assertEqual(r.shape, result_1[i].shape)
            self.assertJaxArrayEqual(r, result_0[i], precision_tolerance=1e-5)
            self.assertJaxArrayEqual(r, result_1[i], precision_tolerance=1e-5)


def main():
    unittest.main()


if __name__ == "__main__":
    main()
