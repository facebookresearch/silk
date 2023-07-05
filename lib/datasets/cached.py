# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import io
import pickle
from typing import Iterable, Optional

import h5py
import numpy
from torch.utils.data import Dataset


# fix data saved using old pixenv codebase
class PixenvFixUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("pixenv"):
            module = module.replace("pixenv", "silk", 1)
        return super().find_class(module, name)


class CachedDataset(Dataset):
    """Cached dataset of serialized python objects."""

    DEFAULT_UNPICKLER = PixenvFixUnpickler

    @staticmethod
    def from_iterable(
        filepath: str, iterable: Iterable, take_n: Optional[int] = None
    ) -> "CachedDataset":
        """Creates a pytorch dataset from an iterable and serialize it to disk for easy loading.

        Parameters
        ----------
        filepath : str
            Path of file to save the cached dataset to.
        iterable : Iterable
            Iterable to traverse to store in cached dataset.
        take_n : Optional[int], optional
            Number of elements to store from iterable, by default None. When None is specified, the parameter will be inferred from iterable.

        Returns
        -------
        CachedDataset
            Newly created dataset.
        """
        db = h5py.File(filepath, mode="w")

        for i, item in enumerate(iterable):
            # early stopping if specified
            if i == take_n:
                break

            # converts python object to bytes
            obj_bytes = pickle.dumps(item)
            obj_bytes = numpy.void(obj_bytes)

            # save bytes as one h5py dataset
            key = str(i)
            dtype = h5py.opaque_dtype(obj_bytes.dtype)
            dset = db.create_dataset(key, (1,), dtype=dtype, maxshape=(1,))

            dset[0] = obj_bytes

        return CachedDataset(filepath)

    def __init__(self, filepath: str) -> None:
        """Load cached dataset from file.

        Parameters
        ----------
        filepath : str
            Path of the file to load.
        """
        super().__init__()

        self._filepath = filepath
        self._db = h5py.File(self._filepath, mode="r")

    # iteration doesn't work without this
    # TODO(Pierre) : investigate why
    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    def __len__(self):
        return len(self._db)

    def __getitem__(self, index):
        obj_bytes = self._db[str(index)][0]
        obj_bytes = obj_bytes.tobytes()
        obj_bytes = io.BytesIO(obj_bytes)
        # TODO(Pierre): Fix use of pickle to solve `python_pickle_is_bad`
        # reference : https://fburl.com/pickle_is_bad
        return CachedDataset.DEFAULT_UNPICKLER(obj_bytes).load()

    def __del__(self):
        self._db.close()

    def __getstate__(self):
        return {"filepath": self._filepath}

    def __setstate__(self, newstate):
        self._filepath = newstate["filepath"]
        self._db = h5py.File(self._filepath, mode="r")
