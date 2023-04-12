# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, Optional

FORMATTERS = {"none", "python", "json", "yaml"}

FormatterType = Callable[[Any], Optional[str]]


def _none_formatter(**kwargs) -> FormatterType:
    r"""Returns a function that converts any object to nothing."""

    def formatter(x):
        return None

    return formatter


def _python_formatter(**kwargs) -> FormatterType:
    r"""Returns a function that converts any object to is string."""

    def formatter(x):
        return str(x)

    return formatter


def _json_formatter(**kwargs) -> FormatterType:
    r"""Returns a function that converts any object to a json sring. Additional arguments can be passed to the json dumper."""
    import json

    def formatter(x):
        return json.dumps(x, **kwargs)

    return formatter


def _yaml_formatter(**kwargs) -> FormatterType:
    r"""Returns a function that converts any object to a yaml sring. Additional arguments can be passed to the yaml dumper."""
    from omegaconf import OmegaConf

    def formatter(x):
        return OmegaConf.to_yaml(x, **kwargs)

    return formatter


def get_formatter(name: str, **kwargs: Dict[str, Any]) -> FormatterType:
    r"""Create and configure a formatter based on its name and pass optional arguments (formatter dependent).

    Parameters
    ----------
    name : str
        Name of the formatter. See `silk.config.formatter.FORMATTERS` for available formatters.

    Returns
    -------
    FormatterType
        Formatter function that converts object to strings.

    Raises
    ------
    RuntimeError
        Raised when provided name doesn't match an existing formatter.
    """
    if name not in FORMATTERS:
        raise RuntimeError(
            f'formatter.name="{name}" is not available (should be one of these {FORMATTERS})'
        )
    return globals()[f"_{name}_formatter"](**kwargs)
