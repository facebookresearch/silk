# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import List, Union

import pynvml

from silk.logger import LOG


def get_cpus(percentage: float = 100.0, max_cpus: Union[int, None] = None) -> int:
    """Returns number of CPUs as a function of the total number of available CPUs on the current machine.

    Parameters
    ----------
    percentage : float, optional
        Percentage of CPUs to use, by default 100.
    max_cpus : Union[int, None], optional
        Maximum limit to never exceed, by default None. None means there is no limit.

    Returns
    -------
    int
        Number of CPUs.
    """
    n_cpus = os.cpu_count()
    n_cpus = int((percentage * n_cpus) / 100.0)
    if max_cpus is not None:
        n_cpus = min(max_cpus, n_cpus)
    return n_cpus


def get_gpus(
    selection: str = "all",
    max_gpus: Union[int, None] = None,
    min_available_memory: Union[float, None] = None,
) -> Union[List[int], None]:
    """Returns a list of GPUs as a function of the available GPUs on the current machine.

    Parameters
    ----------
    selection : str, optional
        Type of selection, by default "all". Available selections are "all", "min_Mb", , "min_Gb", "min_%".
    max_gpus : Union[int, None], optional
        Maximum limit to never exceed, by default None. None means there is no limit.
    min_available_memory : Union[float, None], optional
        Minimum available memory required to accept the GPU in our list, by default None.
        Ignored when using `selection` = "all".
        Required when using `selection` = "min_*".
            * When `selection` = "min_Mb", `min_available_memory` should be the minimum required memory in megabytes.
            * When `selection` = "min_Gb", `min_available_memory` should be the minimum required memory in gigabytes.
            * When `selection` =  "min_%", `min_available_memory` should be the minimum required memory in percentage of total memory.

    Returns
    -------
    Union[List[int], None]
        List of acceptable GPUs to use. Returns None if no valid GPU was found.
    """
    pynvml.nvmlInit()

    # get all GPUs
    n_gpus = pynvml.nvmlDeviceGetCount()
    gpus = list(range(n_gpus))

    def get_mem(i):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        return pynvml.nvmlDeviceGetMemoryInfo(handle)

    # filter out GPUs
    if selection == "all":
        pass
    elif selection.startswith("min_"):
        if selection.endswith("Mb"):
            removal_condition = (
                lambda mem: (mem.free / (1024**2)) < min_available_memory
            )
        elif selection.endswith("Gb"):
            removal_condition = (
                lambda mem: (mem.free / (1024**3)) < min_available_memory
            )
        elif selection.endswith("%"):
            removal_condition = (
                lambda mem: ((mem.free * 100.0) / mem.total) < min_available_memory
            )
        else:
            raise RuntimeError(f'invalid selection "{selection}"')

        for i in range(n_gpus):
            mem = get_mem(i)
            if removal_condition(mem):
                gpus[i] = None
    else:
        raise RuntimeError(f'invalid selection "{selection}"')

    # remove filtered out gpus
    gpus = [gpu for gpu in gpus if gpu is not None]

    # no gpus found
    if len(gpus) is None:
        LOG.warning("no available gpu was found")
        return None

    # no gpu limit, return all
    if max_gpus is None:
        return gpus

    # limit returned list
    return gpus[:max_gpus]
