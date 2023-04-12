# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL.Image import Image as PILImage

from silk.transforms.abstract import NamedContext, Transform


class ToTensor(Transform):
    """Transform item to pytorch tensor."""

    def __init__(
        self,
        dtype: Union[str, None] = None,
        device: Union[str, None] = None,
        requires_grad: bool = False,
    ) -> None:
        super().__init__()
        self._dtype = dtype
        self._device = device
        self._requires_grad = requires_grad

    def __call__(self, item: Any) -> Any:
        # if PIL image, convert to numpy array
        if isinstance(item, PILImage):
            item = np.array(item)
        elif isinstance(item, torch.Tensor):
            item = item.to(self._device)
            item = item.to(self._dtype)
            item.requires_grad_(self._requires_grad)

        if isinstance(item, torch.Tensor):
            return item
        return torch.tensor(
            item,
            dtype=self._dtype,
            device=self._device,
            requires_grad=self._requires_grad,
        )


class ToDevice(Transform):
    """Sends tensor(s) to specified device. Handle batch dimension as tuple or list."""

    def __init__(self, device: str) -> None:
        super().__init__()
        self.device = device

    def __call__(
        self, item: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]]
    ) -> Any:
        if isinstance(item, list):
            return [it.to(self.device) for it in item]
        if isinstance(item, tuple):
            return tuple(it.to(self.device) for it in item)
        return item.to(self.device)


class Shape(Transform):
    def __init__(self, start=None, end=None) -> None:
        super().__init__()
        self._start = start
        self._end = end

    def __call__(self, item: torch.Tensor) -> Any:
        start = 0 if self._start is None else self._start
        end = len(item.shape) if self._end is None else self._end
        return torch.tensor(item.shape[start:end])


class AutoBatch(Transform):
    """Transform useful to batch data coming from a data loader.

    * Input are either single item or list of items.
    * Single item are processed as being a list of one item.
    * Numerical items (int, float) are batched as a torch tensor.
    * Torch tensor items are batch as a new torch tensor having a new batch dimension (dimension 0).
    * Strings items are batched as a tuple of strings.
    * Dictionary items are recursively batched accross all of their keys. All dictionary items should have same keys.
    * Tuple items are recursively batched accross all of their positions. All tuple items should have same size.
    * NamedContext items are recursively batched accross all of their named variables. All named context item should have same variable names.
    * Lists are not handled since they represent the batch dimension.

    Examples
    --------

    ```python
    from silk.transforms.abstract import AutoBatch, NamedContext

    transf = AutoBatch()

    print(transf([1, 2, 3]))
    # >>> tensor([1, 2, 3])

    print(transf(1))
    # >>> tensor([1])

    batch = [ NamedContext(a=float(i), b=str(i)) for i in range(4) ]
    print(batch)
    # >>> [NamedContext({'a': 0.0, 'b': '0'}), NamedContext({'a': 1.0, 'b': '1'}), NamedContext({'a': 2.0, 'b': '2'}), NamedContext({'a': 3.0, 'b': '3'})]
    print(transf(batch))
    # >>> NamedContext({'a': tensor([0., 1., 2., 3.], dtype=torch.float64), 'b': ('0', '1', '2', '3')})
    ```
    """

    def __init__(self, transform: Optional[Transform] = None) -> None:
        """

        Parameters
        ----------
        transform : Optional[Transform], optional
            Optional transform to apply to each item in batch, by default None
        """
        super().__init__()

        def do_nothing(x):
            return x

        self._transform = do_nothing if transform is None else transform

    def __call__(self, item: Any) -> Any:
        """Run the optional transform and auto-batch the input item.

        Parameters
        ----------
        item : Any
            Input to auto-batch.

        Returns
        -------
        Any
            Batched output.

        Raises
        ------
        RuntimeError
            When provided item is an empty list.
        """
        if isinstance(item, list) or isinstance(item, tuple):
            if not len(item) > 0:
                raise RuntimeError("provided item is empty")

            transformed = [self._transform(it) for it in item]
        else:
            transformed = [self._transform(item)]

        return AutoBatch._collate(transformed)

    @staticmethod
    def _collate(batch):
        r"""Puts each data field into a tensor with outer dimension batch size"""

        elem = batch[0]
        elem_type = type(elem)
        if isinstance(elem, torch.Tensor):
            if all(el.shape == elem.shape for el in batch):
                # returns tensor with first dimension as batch dimension
                return torch.stack(batch, 0)
            # returns batch of tensors having different shapes
            return list(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float64)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, str):
            return list(batch)
        elif isinstance(elem, dict):
            return {key: AutoBatch._collate([d[key] for d in batch]) for key in elem}
        elif isinstance(elem, tuple):
            # check to make sure that the elements in batch have consistent size
            elem_size = len(elem)
            if not all(len(el) == elem_size for el in batch):
                raise RuntimeError(
                    "each element in list of batch should be of equal size"
                )
            return tuple(AutoBatch._collate(samples) for samples in zip(*batch))
        elif isinstance(elem, NamedContext):
            ctx = NamedContext.batching(batch)
            return ctx.map(AutoBatch._collate)
        elif isinstance(elem, Iterable):
            return list(batch)
        elif isinstance(elem, type(None)):
            return list(batch)

        raise TypeError(
            f"batch must contain tensors, numbers, dicts or lists; found {elem_type}"
        )


class Unbatch(Transform):
    """Reverse operator of `AutoBatch`."""

    def __init__(self, tuple_as_list: bool = False) -> None:
        super().__init__()
        self._tuple_as_list = tuple_as_list

    def __call__(self, item: Any) -> Any:
        return Unbatch._uncollate(item, self._tuple_as_list)

    @staticmethod
    def _uncollate(batched_item, tuple_as_list):
        if torch.is_tensor(batched_item):
            return [batched_item[i] for i in range(batched_item.shape[0])]
        elif isinstance(batched_item, list):
            return batched_item
        elif isinstance(batched_item, tuple):
            if tuple_as_list:
                return batched_item
            items = tuple(
                Unbatch._uncollate(batched_item[i], tuple_as_list)
                for i in range(len(batched_item))
            )

            if not all(len(items[i]) == len(items[0]) for i in range(len(items))):
                raise RuntimeError(
                    "each element in list of batch should be of equal size"
                )

            return [
                tuple(items[i][j] for i in range(len(items)))
                for j in range(len(items[0]))
            ]

        elif isinstance(batched_item, dict):
            items = {
                key: Unbatch._uncollate(batched_item[key], tuple_as_list)
                for key in batched_item
            }
            batch_size = len(items[next(iter(items.keys()))])
            return [
                {key: items[key][i] for key in batched_item} for i in range(batch_size)
            ]
        elif isinstance(batched_item, NamedContext):
            items = {
                key: Unbatch._uncollate(batched_item[key], tuple_as_list)
                for key in batched_item.names()
            }
            batch_size = len(items[next(iter(items.keys()))])
            return [
                NamedContext({key: items[key][i] for key in items})
                for i in range(batch_size)
            ]
        elif (
            isinstance(batched_item, float)
            or isinstance(batched_item, int)
            or isinstance(batched_item, str)
        ):
            return batched_item

        raise TypeError(
            f"batched item must either be a tensor, number, dict or list; found {type(batched_item)}"
        )


class NormalizeRange(Transform):
    """Normalize input range to new range."""

    def __init__(self, ilow, ihigh, olow=0.0, ohigh=1.0) -> None:
        super().__init__()

        self._ilow = ilow
        self._ihigh = ihigh
        self._olow = olow
        self._ohigh = ohigh

        self._alpha = (ohigh - olow) / (ihigh - ilow)
        self._beta = olow - ilow * self._alpha

    def __call__(self, item: Any) -> Any:
        return item * self._alpha + self._beta


class Clamp(Transform):
    """Clamp values to make sure they get out of range."""

    def __init__(
        self,
        min_val: Union[float, None] = None,
        max_val: Union[float, None] = None,
    ) -> None:
        super().__init__()

        self._min_val = min_val
        self._max_val = max_val

    def __call__(self, item: torch.Tensor) -> Any:
        if (self._max_val is not None) or (self._min_val is not None):
            item = item.clamp_(self._min_val, self._max_val)
        return item
