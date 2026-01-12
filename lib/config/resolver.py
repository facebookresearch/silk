# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from copy import copy
from typing import Any

from omegaconf import Node, OmegaConf
from silk.config.core import instantiate

INITIALIZED = False


def init_resolvers():
    """Initializes all the OmegaConf resolvers (https://omegaconf.readthedocs.io/en/latest/custom_resolvers.html).
    IMPORTANT : Should be executed before `hydra.main` is called.
    """
    global INITIALIZED, OMEGACONF_RESOLVE
    if not INITIALIZED:
        OmegaConf.register_new_resolver("self_instantiate", self_instantiate)
        OmegaConf.register_new_resolver("ref", ref)
        INITIALIZED = True


def ref(path: str, *, _node_: Node, _parent_: Node, _root_: Node) -> Any:
    """References an existing config path.
    Unlike the regular interpolation ${path...}, the `ref` resolver can handle self-instantiated fields.

    Examples
    --------

    ```yaml
    value: ${ref:config.path}
    ```

    Parameters
    ----------
    path : str
        Config path to use to retrieve value.

    Returns
    -------
    Any
        Value found at `path`.
    """
    keys = path.split(".")
    node = _root_
    for key in keys:
        node = node[key]

    try:
        obj = node._value_
    except AttributeError:
        obj = node

    _parent_[_node_._metadata.key] = obj
    return _parent_[_node_._metadata.key]


def self_instantiate(*, _node_: Node, _parent_: Node) -> Any:
    """Automatically called the instantiate method on specific field.

    Examples
    --------

    ```yaml
    value:
        _value_: ${self_instantiate:}
        _target_: module.function.path
        _args_:
            - arg0
            - arg1
            - arg2
        kwarg0: val0
        kwarg1: val1
    ```

    Returns
    -------
    Any
        Instantiated value.

    Raises
    ------
    RuntimeError
        Raised if resolver is not associated with key `_value_`.
    RuntimeError
        Raised if resolver's parent doesn't have a `_target_` field.
    """
    if _node_._metadata.key != "_value_":
        raise RuntimeError(
            "`self_instantiate` resolver needs to be associated with key `_value_`"
        )

    try:
        _parent_._target_
    except AttributeError:
        raise RuntimeError(
            "`self_instantiate` resolver's parent node needs to have a `_target_` field"
        )

    parent = copy(_parent_)
    del parent._value_

    obj = instantiate(parent)
    _parent_._parent[_parent_._metadata.key] = obj

    return _parent_._parent[_parent_._metadata.key]
