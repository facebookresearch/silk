# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
import PIL.Image
import torch
from silk.test.util import OverwriteTensorEquality, string_to_tensor
from silk.transforms.abstract import Lambda, Map, NamedContext
from silk.transforms.tensor import AutoBatch, NormalizeRange, ToTensor, Unbatch


class _UnitTests(unittest.TestCase):
    def test_to_tensor(self):
        transf = Map(ToTensor())
        ctx = NamedContext(
            a=[0, 1, 2],
            b=(0.0, 1.0, 2),
            c=np.arange(3),
            d=[[0, 1], (2, 3)],
            e=PIL.Image.fromarray(np.array([[0, 1], (2, 3)], dtype=np.uint8)),
        )

        ctx = transf(ctx)

        with OverwriteTensorEquality(torch, check_shape=True, check_dtype=True):
            self.assertEqual(
                ctx,
                NamedContext(
                    {
                        "a": torch.tensor([0, 1, 2]),
                        "b": torch.tensor([0.0, 1.0, 2.0]),
                        "c": torch.tensor([0, 1, 2]),
                        "d": torch.tensor([[0, 1], [2, 3]]),
                        "e": torch.tensor([[0, 1], [2, 3]], dtype=torch.uint8),
                    }
                ),
            )

    def _auto_batch(self, fn_or_batch, n, expected, transform=None):
        transf = AutoBatch(transform)
        auto_batch = AutoBatch()
        unbatch = Unbatch()

        if callable(fn_or_batch):
            batch = [fn_or_batch(i) for i in range(n)]
        else:
            batch = fn_or_batch

        output = transf(batch)
        self.assertEqual(output, expected)

        # test unbatch consistency as a reverse operator
        self.assertEqual(output, auto_batch(unbatch(output)))

    def test_auto_batch_and_unbatch(self):
        with OverwriteTensorEquality(
            torch,
            check_dtype=True,
            check_shape=True,
            check_device=True,
        ):
            tensor_ints = torch.tensor([0, 1, 2])
            tensor_floats = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
            tuple_strings = ["0", "1", "2"]

            # test tensor scalar
            self._auto_batch(
                lambda i: torch.tensor(int(i)),
                3,
                tensor_ints,
            )

            # test ints
            self._auto_batch(
                lambda i: int(i),
                3,
                tensor_ints,
            )
            # test floats
            self._auto_batch(
                lambda i: float(i),
                3,
                tensor_floats,
            )
            # test strings
            self._auto_batch(
                lambda i: str(i),
                3,
                tuple_strings,
            )
            # test dicts
            self._auto_batch(
                lambda i: {"a": int(i), "b": str(i), "c": {"1": float(i), "2": str(i)}},
                3,
                {
                    "a": tensor_ints,
                    "b": tuple_strings,
                    "c": {
                        "1": tensor_floats,
                        "2": tuple_strings,
                    },
                },
            )
            # test tuples
            self._auto_batch(
                lambda i: (int(i), str(i)),
                3,
                (
                    tensor_ints,
                    tuple_strings,
                ),
            )
            # test named context
            self._auto_batch(
                lambda i: NamedContext(
                    a=int(i), b=str(i), c=NamedContext(_1=float(i), _2=str(i))
                ),
                3,
                NamedContext(
                    a=tensor_ints,
                    b=tuple_strings,
                    c=NamedContext(
                        _1=tensor_floats,
                        _2=tuple_strings,
                    ),
                ),
            )
            # test single input
            self._auto_batch(
                NamedContext(
                    a=0,
                    b="0",
                ),
                1,
                NamedContext(
                    a=torch.tensor([0]),
                    b=["0"],
                ),
            )
            # test with simple transform
            self._auto_batch(
                lambda i: NamedContext(a=int(i), b=str(i)),
                3,
                NamedContext(
                    a=tensor_ints,
                    b=tuple_strings,
                    c=tensor_floats + 1,
                ),
                transform=Lambda("c", lambda x: float(x + 1), "@a"),
            )

    def test_normalize_range(self):
        transf = NormalizeRange(
            ilow=-20,
            ihigh=+20,
            olow=0.0,
            ohigh=1.0,
        )

        input_tensor = string_to_tensor(
            """
            -20, 20
            0, 10
            """
        )

        expected_tensor = string_to_tensor(
            """
            0., 1.
            0.5, 0.75
            """
        )

        output_tensor = transf(input_tensor)

        with OverwriteTensorEquality(torch):
            self.assertEqual(output_tensor, expected_tensor)


def main():
    unittest.main()


if __name__ == "__main__":
    unittest.main()
