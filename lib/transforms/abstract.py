# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import inspect
from collections import OrderedDict
from random import random
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple as TupleType,
    Union,
)

import torch
from silk.config.core import ensure_is_instance


class Transform(torch.nn.Module):
    """Abstract representation of a transform, which is essentially a parametrized function taking one input."""

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, item: Any) -> Any:
        raise NotImplementedError


# TODO(Pierre): Overload list specific methods as we need them.
class MixedModuleList(torch.nn.Module):
    """Works the same as `torch.nn.ModuleList`, but allows to have non-module items."""

    def __init__(self, items: Iterable[Any]) -> None:
        super().__init__()

        self._mods = torch.nn.ModuleList(
            [mod for mod in items if isinstance(mod, torch.nn.Module)]
        )
        self._items = items

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx: int):
        return self._items[idx]


# TODO(Pierre): Overload dict specific methods as we need them.
class MixedModuleDict(torch.nn.Module):
    """Works the same as `torch.nn.ModuleDict`, but allows to have non-module items."""

    def __init__(self, items: Dict[Any, Any] = None) -> None:
        super().__init__()

        items = OrderedDict() if items is None else items
        self._mods = torch.nn.ModuleDict(
            {key: mod for key, mod in items.items() if isinstance(mod, torch.nn.Module)}
        )
        self._items = items

    def __len__(self):
        return len(self._items)

    def __getitem__(self, key: Any):
        return self._items[key]

    def __setitem__(self, key: str, item: Any) -> None:
        if key in self._mods:
            del self._mods[key]
        if isinstance(item, torch.nn.Module):
            self._mods[key] = item
        self._items[key] = item

    def __contains__(self, key: str) -> bool:
        return key in self._items

    def items(self):
        return self._items.items()

    def keys(self):
        return self._items.keys()


class NamedContext:
    """Container of named data. This helps applying and tracking data transformation in complex data pipelines.
    Used mostly to structure and transform data loader outputs.
    """

    @staticmethod
    def batching(contexts: List[NamedContext]) -> NamedContext:
        """Batch context variables.

        Parameters
        ----------
        contexts : List[NamedContext]
            List of contexts to batch the variables for. Each context should possess the exact same variable names.

        Returns
        -------
        NamedContext
            Named context having each variable being batched (as a list).

        Raises
        ------
        RuntimeError
            When provided context names do not exactly match.
        """
        data = {}

        if not len(contexts) > 0:
            return NamedContext(data)

        names = contexts[0].names()

        all_same_name_error_msg = (
            "each named context in list of batch should all have same names"
        )
        if not all(len(ctx.names()) == len(names) for ctx in contexts):
            raise RuntimeError(all_same_name_error_msg)

        try:
            data = {name: [ctx[name] for ctx in contexts] for name in names}
        except KeyError:
            raise RuntimeError(all_same_name_error_msg)

        return NamedContext(data)

    def __init__(
        self, data: Optional[Dict[str, Any]] = None, **kwargs: Dict[str, Any]
    ) -> None:
        """

        Parameters
        ----------
        data : Optional[Dict[str, Any]], optional
            named variabled to put in context, by default None
        """
        self._data = {}
        if data is not None:
            self._data.update(data)
        self._data.update(kwargs)

    def exists(self, name: str) -> bool:
        """Return wether a variable name is in context or not.

        Parameters
        ----------
        name : str
            Name of variable to check for.

        Returns
        -------
        bool
            True if `name` is in context, False otherwise.
        """
        return name in self._data

    __contains__ = exists

    def ensure_exists(self, *names: List[str]) -> None:
        """Make sure variable names exist in context. Raise exception otherwise.

        Parameters
        ----------
        names : str
            Names of variables to check for.

        Raises
        ------
        RuntimeError
            When at least one of the names is not in context.
        """
        for name in names:
            if not self.exists(name):
                raise RuntimeError(f'"{name}" should be present in named context')

    def ensure_not_exists(self, *names: List[str]) -> None:
        """Make sure variable names doesn't exist in context. Raise exception otherwise.

        Parameters
        ----------
        names : str
            Names of variables to check for.

        Raises
        ------
        RuntimeError
            When at least one of the names is not in context.
        """
        for name in names:
            if self.exists(name):
                raise RuntimeError(f'"{name}" should not be present in named context')

    def rename(self, old_name: str, new_name: str) -> NamedContext:
        """Rename variable in context.

        Parameters
        ----------
        old_name : str
            Name of variable to rename.
        new_name : str
            New name.

        Returns
        -------
        NamedContext
            New context with rename variable.
        """
        self.ensure_exists(old_name)
        if old_name == new_name:
            return self
        data = dict(self._data)
        data[new_name] = data[old_name]
        del data[old_name]
        return NamedContext(data)

    def add(self, name: str, value: Any, allow_exist=False) -> NamedContext:
        """Add new variable with associated value to context.

        Parameters
        ----------
        name : str
            Name of variable to add.
        value : Any
            Value of variable to add.
        allow_exist : bool, optional
            Determine if overwriting existing variable is ok, will raise exception otherwise, by default False.

        Returns
        -------
        NamedContext
            New context with added variable.
        """
        if not allow_exist:
            self.ensure_not_exists(name)
        data = dict(self._data)
        data[name] = value
        return NamedContext(data)

    def remove(self, *names: List[str], allow_not_exist=False) -> NamedContext:
        """Remove existing variables in context.

        Parameters
        ----------
        names : List[str]
            Names of variables to remove.
        allow_not_exist : bool, optional
            Allow removal of non-existing variables, by default False

        Returns
        -------
        NamedContext
            New context minus the removed variables.
        """
        if len(names) == 0:
            return self

        data = dict(self._data)

        for name in names:
            if not allow_not_exist:
                self.ensure_exists(name)
                del data[name]
            else:
                if name in data:
                    del data[name]
        return NamedContext(data)

    def map(
        self, fn: Callable[..., Any], *args: List[Any], **kwargs: Dict[str, Any]
    ) -> NamedContext:
        """Map-execute function for every variable in context.

        Parameters
        ----------
        fn : Callable[..., Any]
            Function to execute per variable `fn(var, *args, **kwargs)`.

        Returns
        -------
        NamedContext
            New context with variable transformed by provided function.
        """
        data = {name: fn(el, *args, **kwargs) for name, el in self._data.items()}
        return NamedContext(data)

    def map_only(
        self,
        names: List[str],
        fn: Callable[..., Any],
        *args: List[Any],
        **kwargs: Dict[str, Any],
    ):
        """Map-execute function only for every provided variable present in context.

        Parameters
        ----------
        names: List[str]
            List of context names to apply the map to.
        fn : Callable[..., Any]
            Function to execute per variable `fn(var, *args, **kwargs)`.

        Returns
        -------
        NamedContext
            New context with variable transformed by provided function.
        """
        data = dict(self._data)
        data.update({name: fn(data[name], *args, **kwargs) for name in names})
        return NamedContext(data)

    def __getitem__(self, name: str) -> Any:
        """Get value of variable.

        Parameters
        ----------
        name : str
            Name of variable.

        Returns
        -------
        Any
            Value of variable.
        """
        self.ensure_exists(name)
        return self._data[name]

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(self._data)})"

    def __eq__(self, other: NamedContext) -> bool:
        if not isinstance(other, NamedContext):
            return False
        return self._data == other._data

    def names(self) -> TupleType[str]:
        """Return names of all variables in context.

        Returns
        -------
        Tuple[str]
            Names of variables.
        """
        return tuple(self._data.keys())


class ToNamedContext(Transform):
    """Transform tuples into named contexts. This operator is very useful as a first step in transforming data coming from a data loader since data loaders output tuples of data.

    Examples
    --------

    ```python
    from silk.transforms.abstract import ToNamedContext

    transf = ToNamedContext("a", "b", None, "c")
    ctx = transf((0, 1, 2, 3))

    print(ctx)
    # >>> NamedContext({'a': 0, 'b': 1, 'c': 3})
    ```

    """

    def __init__(self, *names: List[Optional[str]], single_item: bool = False) -> None:
        """

        Parameters
        ----------
        names : List[Optional[str]]
            Names to associate to each tuple position. `None` can be used to exclude tuple values.
            The ordering of names has to match the ordering of tuples.

        single_item : bool
            Specify if we should handle input as a single item (=True), or as a tuple (=False), default is False.

        """
        super().__init__()
        self._names = names
        self._single_item = single_item

        if not (len(self._names) > 0):
            raise RuntimeError(
                "empty name list provided, should at least contain one element"
            )

        if self._single_item and len(self._names) != 1:
            raise RuntimeError(
                "only one name should be provided when `single_item` option is enabled"
            )

    def __call__(self, item: Union[TupleType[Any], Any]) -> NamedContext:
        """Convert a single item or tuple to named context.

        Parameters
        ----------
        item : Union[TupleType[Any], Any]
            Tuple or item to convert.

        Returns
        -------
        NamedContext
            Named context created from tuple or item.

        Raises
        ------
        RuntimeError
            When tuple's size doesn't match the size of the provided names.
        """
        if not self._single_item:
            ensure_is_instance(item, tuple)
        else:
            item = (item,)

        if len(self._names) != len(item):
            raise RuntimeError(
                f"tuple item size (={len(item)}) doesn't match the number of provided names (={len(self._names)})"
            )
        return NamedContext(
            {name: item[i] for i, name in enumerate(self._names) if name is not None}
        )


Name = ToNamedContext  # shorter alias


class Compose(Transform):
    """Transform that is composed of multiple transform that will run sequentially."""

    def __init__(self, *transforms: List[Transform]) -> None:
        """
        Parameters
        ----------
        transforms : List[Transform]
            List of transforms to apply.
        """
        super().__init__()

        self._transforms = MixedModuleList(transforms)

    def __call__(self, item: Any) -> Any:
        """Apply sequence of transform to item.

        Parameters
        ----------
        item : Any
            Input item of the sequence of transform.

        Returns
        -------
        Any
            Output of sequence of transform.
        """
        for transform in self._transforms:
            item = transform(item)
        return item


class Rename(Transform):
    """Transform that renames a variable from a named context."""

    def __init__(self, old_name: str, new_name: str) -> None:
        """

        Parameters
        ----------
        old_name : str
            Name of variable to rename.
        new_name : str
            New name of variable.
        """
        super().__init__()
        self._old_name = old_name
        self._new_name = new_name

    def __call__(self, item: NamedContext) -> NamedContext:
        """Apply the renaming transform to named context.

        Parameters
        ----------
        item : NamedContext
            Input named context to apply the renaming to.

        Returns
        -------
        NamedContext
            New named context with renamed variable.
        """
        ensure_is_instance(item, NamedContext)
        return item.rename(self._old_name, self._new_name)


class Lambda(Transform):
    """Transform that runs a python function to a named context and put the result in that same context.

    Examples
    --------

    ```python
    from silk.transforms.abstract import Lambda, NamedContext

    def sum4(a, b, c, d=0):
        return a + b + c + d

    transf = Lambda(
        "c",     # name of variable that will store the output
        sum4,    # function to run
        "@a",    # value of first argument will be variable "a" extracted from named context
        10,      # value of second argument will be value 10
        d="@b",  # value of argument d will be variable "b" extracted from named context
        c=0,     # value of argument c will be value 0
    )

    ctx = NamedContext({"a": 1, "b": 2})
    ctx = transf(ctx)

    print(ctx)
    # >>> NamedContext({'a': 1, 'b': 2, 'c': 13})
    ```
    """

    def __init__(
        self,
        name: Union[str, Iterable[str], None],
        function: Callable[..., Any],
        *args_keys: List[Any],
        **kwargs_keys: Dict[str, Any],
    ) -> None:
        """

        Parameters
        ----------
        name : Union[str, Iterable[str], None]
            Name of the variable where the output will be stored.
            If None, the result is directly returned, discarding the NamedContext.
            If iterable of strings, the items in the returned tuple will be placed in NamedContext at provided names.
        function : Callable[..., Any]
            Function to apply.
        args_keys: List[Any]
            Arguments to pass to the function. String values starting with a "@" will be replaced by corresponding named context variable value.
        kwargs_keys: Dict[str, Any]
            Named arguments to pass to the function. String values starting with a "@" will be replaced by corresponding named context variable value.
        """
        super().__init__()
        self._name = name
        self._function = function
        self._args_keys = MixedModuleList(args_keys)
        self._kwargs_keys = MixedModuleDict(kwargs_keys)

    @property
    def name(self):
        return self._name

    @staticmethod
    def _is_context(name) -> bool:
        return isinstance(name, str) and name == "@"

    @staticmethod
    def _is_context_name(name: Any) -> bool:
        return isinstance(name, str) and len(name) > 1 and name[0] == "@"

    @staticmethod
    def _get_context_name(name: str) -> str:
        return name[1:]

    @staticmethod
    def _get_value(name: Any, item: NamedContext):
        if Lambda._is_context_name(name):
            name = Lambda._get_context_name(name)
            return item[name]
        elif Lambda._is_context(name):
            return item
        return name

    def __call__(self, item: NamedContext) -> NamedContext:
        """Execute function on item and store result in context.

        Parameters
        ----------
        item : NamedContext
            Input named context.

        Returns
        -------
        NamedContext
            New named context containing the function's result.
        """
        ensure_is_instance(item, NamedContext)

        args = [Lambda._get_value(name, item) for name in self._args_keys]
        kwargs = {
            fname: Lambda._get_value(cname, item)
            for fname, cname in self._kwargs_keys.items()
        }
        output = self._function(*args, **kwargs)
        if self._name is None:
            return output
        elif isinstance(self._name, str):
            return item.add(self._name, output, allow_exist=True)

        for i, name in enumerate(self._name):
            if name is None:
                continue
            item = item.add(name, output[i], allow_exist=True)

        return item


class MethodCall(Lambda):
    """Similar to `Lambda`, but using a method instead of a function."""

    def __init__(
        self,
        name: Union[str, Iterable[str], None],
        self_: Any,
        method: Union[Callable[..., Any], str],
        *args_keys: List[Any],
        **kwargs_keys: Dict[str, Any],
    ) -> None:
        """

        Parameters
        ----------
        name : Union[str, Iterable[str], None]
            Name of the variable where the output will be stored.
            If None, the result is directly returned, discarding the NamedContext.
            If iterable of strings, the items in the returned tuple will be placed in NamedContext at provided names.
        self_ : Any
            Instance that the method will be applied on. String values starting with a "@" will be replaced by corresponding named context variable value.
        method : Union[Callable[..., Any], str]
            Method to apply.
        args_keys: List[Any]
            Arguments to pass to the method. String values starting with a "@" will be replaced by corresponding named context variable value.
        kwargs_keys: Dict[str, Any]
            Named arguments to pass to the method. String values starting with a "@" will be replaced by corresponding named context variable value.
        """
        if isinstance(method, str):
            super().__init__(
                name,
                MethodCall._find_and_call_method_by_name,
                method,
                self_,
                *args_keys,
                **kwargs_keys,
            )
        else:
            super().__init__(name, method, self_, *args_keys, **kwargs_keys)

    @staticmethod
    def _find_and_call_method_by_name(method, self_, *args_keys, **kwargs_keys):
        if not hasattr(self_, method):
            raise RuntimeError(f"no method named '{method}' has been found in self")
        method = getattr(self_, method)
        if (not inspect.ismethod(method)) and (not inspect.isbuiltin(method)):
            raise RuntimeError(
                f"method named '{method}' was found, but is not a method"
            )
        return method(*args_keys, **kwargs_keys)


class Tuple(Lambda):
    """Apply tuple operator to named context."""

    def __init__(
        self,
        name: Union[str, None],
        *args_keys: List[Any],
    ) -> None:
        """

        Parameters
        ----------
        name : Union[str, None]
            Name of the variable where the output will be stored.
            If None, the result is directly returned, discarding the NamedContext.
        args_keys: List[Any]
            Arguments to pass to the method. String values starting with a "@" will be replaced by corresponding named context variable value.
        """

        def make_tuple(*args):
            return tuple(args)

        super().__init__(name, make_tuple, *args_keys)


class Map(Transform):
    """Apply map operator to named context. See `NamedContext.map` for more details."""

    def __init__(self, function, *args, **kwargs) -> None:
        super().__init__()
        self._function = function
        self._args = args
        self._kwargs = kwargs

    def __call__(self, item: NamedContext) -> NamedContext:
        return item.map(self._function, *self._args, **self._kwargs)


class MapOnly(Transform):
    """Apply map only operator to named context. See `NamedContext.map_only` for more details."""

    def __init__(self, names, function, *args, **kwargs) -> None:
        super().__init__()
        self._names = names
        self._function = function
        self._args = args
        self._kwargs = kwargs

    def __call__(self, item: NamedContext) -> NamedContext:
        return item.map_only(self._names, self._function, *self._args, **self._kwargs)


class Remove(Transform):
    """Apply remove operator to named context. See `NamedContext.remove` for more details."""

    def __init__(self, *names: List[str], allow_not_exist=False) -> None:
        super().__init__()
        self._names = names
        self._allow_not_exist = allow_not_exist

    def __call__(self, item: NamedContext) -> NamedContext:
        return item.remove(*self._names, allow_not_exist=self._allow_not_exist)


class Add(Transform):
    """Apply add operator to named context. See `NamedContext.add` for more details."""

    def __init__(self, name: str, value: Any, allow_exist=False) -> None:
        super().__init__()
        self._name = name
        self._value = value
        self._allow_exist = allow_exist

    def __call__(self, item: NamedContext) -> NamedContext:
        return item.add(self._name, self._value, allow_exist=self._allow_exist)


class Stochastic(Transform):
    """Apply provided transform with given probability."""

    def __init__(self, transform: Transform, probability: float = 0.5) -> None:
        super().__init__()

        self._transform = transform
        self._probability = probability

    def __call__(self, item: Any) -> Any:
        if random() < self._probability:
            return self._transform(item)
        return item
