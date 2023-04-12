# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Union

import hydra.utils
from omegaconf import DictConfig, ListConfig

TargetType = Union[str, type, Callable[..., Any]]
ClassOrCallableType = Union[type, Callable[..., Any]]


def locate(path: str) -> Any:
    """
    Locate an object by name or dotted path, importing as necessary.
    This is similar to the pydoc function `locate`, except that it checks for
    the module from the given path from back to front.

    Parameters
    ----------
    path : str
        `module.path` path of the target to locate.

    Returns
    -------
    Any
        The value found at `path`.

    Raises
    ------
    ImportError
        Raised when empty path is provided.
    ImportError
        Raised when loading a module subpath has errors.
    """

    if path == "":
        raise ImportError("Empty path")

    import builtins
    from importlib import import_module

    parts = [part for part in path.split(".") if part]

    # load module part
    module = None
    for n in reversed(range(len(parts))):
        try:
            mod = ".".join(parts[:n])
            module = import_module(mod)
        except Exception as e:
            if n == 0:
                raise ImportError(f"Error loading module '{path}'") from e
            continue
        if module:
            break

    if module:
        obj = module
    else:
        obj = builtins

    # load object path in module
    for part in parts[n:]:
        mod = mod + "." + part
        if not hasattr(obj, part):
            try:
                import_module(mod)
            except Exception as e:
                raise ImportError(
                    f"Encountered error: `{e}` when loading module '{path}'"
                ) from e
        obj = getattr(obj, part)

    return obj


def find_class_or_callable_from_target(
    target: TargetType,
) -> ClassOrCallableType:
    """Finds class or callable from its full module path. Do nothing if the input is already a class or callable.

    Parameters
    ----------
    target : TargetType
        Full module path.

    Returns
    -------
    ClassOrCallableType
        Class or callable found at target path.
    """
    if isinstance(target, str):
        obj = locate(target)
    else:
        obj = target

    if (not isinstance(obj, type)) and (not callable(obj)):
        raise ValueError(f"Invalid type ({type(obj)}) found for {target}")

    return obj


def find_and_ensure_is_subclass(target: TargetType, type_: type) -> ClassOrCallableType:
    """Find class from its full module path. Then checks if it is a subclass of a specific type.

    Parameters
    ----------
    target : TargetType
        Full module path.
    type_ : type
        Type to check for.

    Returns
    -------
    ClassOrCallableType
        Class or callable found at target path.
    """
    klass = find_class_or_callable_from_target(target)
    ensure_is_subclass(klass, type_)
    return klass


def find_and_ensure_is_instance(target: TargetType, type_: type):
    """Find item from its full module path. Then checks if it implements a specific type.

    Parameters
    ----------
    target : TargetType
        Full module path.
    type_ : type
        Type to check for.

    Returns
    -------
    ClassOrCallableType
        Class or callable found at target path.
    """
    instance = find_class_or_callable_from_target(target)
    ensure_is_instance(instance, type_)
    return instance


def instantiate_and_ensure_is_instance(
    cfg: Union[DictConfig, ListConfig], type_: type
) -> Any:
    """Instantiate item from its config specification (hydra _target_ field). Then checks if it implements a specific type.

    Parameters
    ----------
    cfg : DictConfig
        Hydra dictionary container a `_target_` field.
    type_ : type
        Type to check for.

    Returns
    -------
    ClassOrCallableType
        Class or callable found at target path.
    """
    instance = instantiate(cfg)
    ensure_is_instance(instance, type_)
    return instance


def instantiate(cfg: Union[DictConfig, ListConfig]):
    if isinstance(cfg, ListConfig):
        return [instantiate(item) for item in cfg]
    return hydra.utils.instantiate(cfg)


def full_instance_name(instance: Any) -> str:
    """Get full module path name of instance.

    Parameters
    ----------
    instance : Any
        Any object.

    Returns
    -------
    str
        Full module path name (e.g. `<module>.<name>`).
    """
    return full_class_name(instance.__class__)


def full_class_name(klass: Any) -> str:
    """Get full module path name of class.

    Parameters
    ----------
    klass : Any
        Any class.

    Returns
    -------
    str
        Full module path name (e.g. `<module>.<name>`).
    """
    module = klass.__module__
    if module == "builtins":
        return klass.__qualname__  # avoid outputs like 'builtins.str'
    return module + "." + klass.__qualname__


def ensure_is_subclass(child_class: type, parent_class: type) -> None:
    """Make sure a class is a subclass of another. Raise exception otherwise.

    Parameters
    ----------
    child_class : type
        Child class
    parent_class : type
        Parent class

    Raises
    ------
    RuntimeError
        Raised when `child_class` is not a subclass of `parent_class`.
    """
    if not issubclass(child_class, parent_class):
        raise RuntimeError(
            f"class {full_class_name(child_class)} should be a subclass of {full_class_name(parent_class)}"
        )


def ensure_is_instance(instance: Any, type_: type) -> None:
    """Make sure an object is an instance of a specifc class. Raise exception otherwise.

    Parameters
    ----------
    instance : Any
        Object to check for
    type_ : type
        Type to check for

    Raises
    ------
    RuntimeError
        Raised when `instance` is not an instance of `type_`.
    """
    if not isinstance(instance, type_):
        raise RuntimeError(
            f"instance should be of type {full_class_name(type_)}, not {full_instance_name(instance)}"
        )
