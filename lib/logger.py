# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Logger Module

How to use in all modules :

    1. Import the logger in your module : `from silk.logger import LOG`
    2. Use logging primitive to log messages :
      - LOG.trace(message : str)     > For exhaustive logging of all details.
      - LOG.debug(message : str)     > For logging information useful for debugging purposes.
      - LOG.info(message : str)      > For logging general information about the execution flow.
      - LOG.success(message : str)   > For logging major successful milestones.
      - LOG.warning(message : str)   > For logging possibly unwanted behavior.
      - LOG.error(message : str)     > For logging non-critical error (do not stop the execution).
      - LOG.critical(message : str)  > For logging critical errors that requires a full stop of the process.

How to handle and print exceptions with stacktrace :

    * Use `LOG.catch` as a function decorator.
        @LOG.catch
        def my_func(x):
            raise RuntimeError("Oh no ! I'm going to get caught !")

    * Use the `opt(exception=True)` option.
        try:
            raise RuntimeError("I'm going to roam free !")
        except:
            LOG.opt(exception=True).error(
                "Some exception tried to escape. We caught it.",
            )

When using in the main module, you can reconfigure the default handlers :

    * Enable three handlers.
        logger.enable_handlers(
            "<handler_name_0>",
            "<handler_name_1>",
            "<handler_name_2>",
        )

    * Enable ONLY those three handlers (will disable all others).
        logger.enable_handlers_only(
            "<handler_name_0>",
            "<handler_name_1>",
            "<handler_name_2>",
        )

    * Changing the sink of the "<handler_name>" handler.
        logger.set_handler_options("<handler_name>", sink="err.log")

    * Changing the level of the "<handler_name>" handler.
        logger.set_handler_options("<handler_name>", level="INFO")


This module is built on top of loguru : https://github.com/Delgan/loguru

"""

import sys
from dataclasses import dataclass

from loguru import logger as LOG

# TODO(Pierre) : Set our own default format below
from loguru._defaults import LOGURU_FORMAT

# TODO(Pierre) : Add proper documentation

# useful handlers here
HANDLER_OPTIONS = {
    "default": {
        "sink": sys.stderr,
    },
    "stderr.common": {
        "parent": "default",
        "format": LOGURU_FORMAT,
    },
    "stderr.dev": {
        "parent": "stderr.common",
        "level": "DEBUG",
    },
    "stderr.prod": {
        "parent": "stderr.common",
        "level": "WARNING",
    },
}
HANDLER_IDS = {}
DEFAULT_HANDLERS = ["stderr.dev"]


@dataclass
class HandlerId:
    hid: int = -1
    enabled: bool = False


def _split_handler_add_args(kwargs):
    args = (kwargs["sink"],)
    del kwargs["sink"]
    return args, kwargs


def stderr():
    return sys.stderr


def get_handler_id(name):
    return HANDLER_IDS.setdefault(name, HandlerId())


def set_handler_options(name, **options):
    handler = HANDLER_OPTIONS.setdefault(name, {"parent": "default"})
    handler.update(options)
    if get_handler_id(name).enabled:
        reload_handlers(name)


def get_handler_options(name):
    return HANDLER_OPTIONS.setdefault(name, {"parent": "default"})


def get_handler_names():
    return tuple(HANDLER_OPTIONS.keys())


def get_valid_levels():
    return ("TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


def get_finalized_handler_options(name):
    parent = get_handler_options(name)
    handler_list = [parent]
    while "parent" in handler_list[-1]:
        parent = handler_list[-1]["parent"]
        handler_list.append(get_handler_options(parent))

    final_handler = {}
    while len(handler_list) > 0:
        current = handler_list.pop(-1)
        final_handler.update(**current)

    if "parent" in final_handler:
        del final_handler["parent"]

    return final_handler


def enable_handlers(*handler_names):
    for name in handler_names:
        handler_id = get_handler_id(name)

        if handler_id.enabled:
            LOG.remove(handler_id.hid)

        args, kwargs = _split_handler_add_args(
            get_finalized_handler_options(name),
        )

        handler_id.hid = LOG.add(*args, **kwargs)
        handler_id.enabled = True


reload_handlers = enable_handlers


def enable_handlers_only(*handler_names):
    disable_all_handlers()
    enable_handlers(*handler_names)


def disable_handlers(*handler_names):
    for name in handler_names:
        handler_id = get_handler_id(name)

        if handler_id.enabled:
            LOG.remove(handler_id.hid)
            handler_id.hid = -1
            handler_id.enabled = False


def disable_all_handlers():
    disable_handlers(*HANDLER_IDS.keys())


def _intercept_logging_messages():
    # https://github.com/Delgan/loguru#entirely-compatible-with-standard-logging
    # TODO(Pierre) : Intercepts hydra, but not pytorch_lightning. Figure out why.
    import logging

    class InterceptHandler(logging.Handler):
        def emit(self, record):
            # Get corresponding Loguru level if it exists
            try:
                level = LOG.level(record.levelname).name
            except ValueError:
                level = record.levelno

            # Find caller from where originated the logged message
            frame, depth = logging.currentframe(), 2
            while frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1

            LOG.opt(depth=depth, exception=record.exc_info).log(
                level, record.getMessage()
            )

    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)


LOG.remove()  # remove all loguru default handlers
enable_handlers(*DEFAULT_HANDLERS)
_intercept_logging_messages()
