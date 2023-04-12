# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import tempfile

import torch


class OverwriteTensorEquality:
    """Convenient tool for overwriting torch.Tensor's (in)equality functions.

    Examples
    --------

    ```python
    import torch
    from silk.test.util import OverwriteTensorEquality

    tensor = torch.tensor([1, 2, 3])

    if tensor == tensor: # raise exception
        print("not executed")
    # >>> RuntimeError: Boolean value of Tensor with more than one value is ambiguous

    with OverwriteTensorEquality(torch):
        if tensor == tensor: # doesn't raise exception
            print("executed")
    # >>> executed
    ```

    """

    _OVERWRITEN = set()

    def __init__(
        self, torch_module, check_shape=False, check_dtype=False, check_device=False
    ) -> None:
        """

        Parameters
        ----------
        torch_module : module
            Torch module to overwrite the torch.Tensor functions from.
        """

        # do not overwrite methods twice
        if torch_module.Tensor in OverwriteTensorEquality._OVERWRITEN:
            self._torch_module = None
        else:
            self._torch_module = torch_module

            OverwriteTensorEquality._OVERWRITEN.add(self._torch_module.Tensor)

            self._pytorch_tensor_eq = self._torch_module.Tensor.__eq__
            self._pytorch_tensor_ne = self._torch_module.Tensor.__ne__

        self._check_shape = check_shape
        self._check_dtype = check_dtype
        self._check_device = check_device

    def _test_tensor_eq(self):
        def eq(t0, t1):
            if self._check_shape and t0.shape != t1.shape:
                return False
            if self._check_dtype and t0.dtype != t1.dtype:
                return False
            if self._check_device and t0.device != t1.device:
                return False

            return self._pytorch_tensor_eq(t0, t1).all()

        return eq

    def _test_tensor_ne(self):
        eq = self._test_tensor_eq()

        def ne(t0, t1):
            return not eq(t0, t1)

        return ne

    def __enter__(self):
        if self._torch_module is not None:
            self._torch_module.Tensor.__eq__ = self._test_tensor_eq()
            self._torch_module.Tensor.__ne__ = self._test_tensor_ne()
        return self

    def __exit__(self, type, value, traceback):
        if self._torch_module is not None:
            OverwriteTensorEquality._OVERWRITEN.remove(self._torch_module.Tensor)

            self._torch_module.Tensor.__eq__ = self._pytorch_tensor_eq
            self._torch_module.Tensor.__ne__ = self._pytorch_tensor_ne


def string_to_tensor(string: str, *args, **kwargs):
    array = []
    for line in string.split("\n"):
        elements = line.split(",")
        if len(elements) > 1:
            array.append([float(el) for el in line.split(",") if len(el.strip()) > 0])
    return torch.tensor(array, *args, **kwargs)


def temporary_file():
    """Automatically creates a temporary file (for testing purposes) and cleans it once it gets out of scope.

    Examples
    --------

    ```python
    from silk.test.util import temporary_file

    with temporary_file() as filepath:
        ... # do something with file at "filepath"

    # here the file "filepath" has been removed
    ```

    """
    return _TemporaryFile()


class _TemporaryFile:
    def __enter__(self):
        fd, self.filepath = tempfile.mkstemp()
        os.close(fd)
        return self.filepath

    def __exit__(self, type, value, traceback):
        if os.path.isfile(self.filepath):
            os.remove(self.filepath)


def temporary_directory():
    """Automatically creates a temporary directory (for testing purposes) and cleans it once it gets out of scope.

    Examples
    --------

    ```python
    from silk.test.util import temporary_directory

    with temporary_directory() as dirpath:
        ... # do something with directory at "dirpath"

    # here the directory "dirpath" has been removed
    ```

    """
    return _TemporaryDirectory()


class _TemporaryDirectory:
    def __enter__(self):
        self.dirpath = tempfile.mkdtemp()
        return self.dirpath

    def __exit__(self, type, value, traceback):
        if os.path.isdir(self.dirpath):
            shutil.rmtree(self.dirpath)


def max_tensor_diff(tensor0, tensor1):
    return torch.max(torch.abs(tensor0 - tensor1)).detach().cpu().item()
