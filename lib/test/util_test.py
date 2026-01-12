# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest

import torch
from silk.test.util import OverwriteTensorEquality, temporary_directory, temporary_file


class _UnitTests(unittest.TestCase):
    def _check_equalities(self, arr0, arr1, arr2):
        # arr0 = arr1 and arr0 != arr2
        # test simple (in)equality
        self.assertEqual(arr0, arr1)
        self.assertNotEqual(arr0, arr2)

        # test recursive (in)equality
        self.assertEqual((arr0, arr2), (arr1, arr2))
        self.assertEqual([arr0, arr2], [arr1, arr2])
        self.assertEqual({"a": arr0, "b": arr2}, {"a": arr1, "b": arr2})

        self.assertNotEqual((arr0, arr1), (arr1, arr2))
        self.assertNotEqual([arr0, arr1], [arr1, arr2])
        self.assertNotEqual({"a": arr0, "b": arr1}, {"a": arr1, "b": arr2})

    def test_overwrite_tensor_equality(self):
        arr0 = torch.tensor([1, 2, 3], dtype=torch.uint8)
        arr1 = torch.tensor([1, 2, 3], dtype=torch.float)
        arr2 = torch.tensor([3, 2, 1], dtype=torch.uint8)

        # test simpled overwriting
        with OverwriteTensorEquality(torch):
            self._check_equalities(arr0, arr1, arr2)

        # should go back to using original (in)equality functions
        with self.assertRaises(RuntimeError):
            self.assertEqual(arr0, arr1)

        # test nested overwriting
        with OverwriteTensorEquality(torch):
            with OverwriteTensorEquality(torch):
                self._check_equalities(arr0, arr1, arr2)

        # should go back to using original (in)equality functions
        with self.assertRaises(RuntimeError):
            self.assertEqual(arr0, arr1)

        # check same content, different dtype
        with OverwriteTensorEquality(torch, check_dtype=True):
            self.assertNotEqual(arr0, arr1)

        # check same content, different shape
        with OverwriteTensorEquality(torch, check_shape=True):
            self.assertNotEqual(arr0, arr1.unsqueeze(0))

    def test_temporary_file(self):
        # test common use
        with temporary_file() as filepath:
            self.assertTrue(os.path.isfile(filepath))
        self.assertFalse(os.path.exists(filepath))

        # test when exception is raised
        class CustomException(BaseException):
            pass

        try:
            with temporary_file() as filepath:
                raise CustomException()
        except CustomException:
            self.assertFalse(os.path.exists(filepath))

    def test_temporary_directory(self):
        # test common use
        with temporary_directory() as dirpath:
            self.assertTrue(os.path.isdir(dirpath))
        self.assertFalse(os.path.exists(dirpath))

        # test when exception is raised
        class CustomException(BaseException):
            pass

        try:
            with temporary_directory() as dirpath:
                raise CustomException()
        except CustomException:
            self.assertFalse(os.path.exists(dirpath))


def main():
    unittest.main()


if __name__ == "__main__":
    unittest.main()
