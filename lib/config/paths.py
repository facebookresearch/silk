# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import List


def join(dirname: str, *names: List[str]) -> str:
    """Convenient function to join paths in hyda config files.

    Parameters
    ----------
    dirname : str
        Name of the constant base path (see constants below).
    names : List[str]
        Names of the subfolders / files to join.

    Returns
    -------
    str
        Full joined path as string.
    """
    path: Path = globals()[dirname]
    path = path.joinpath(*names)
    return str(path)


ROOT = Path(__file__).parent.parent.parent
ASSETS = ROOT / "assets"
MODELS = ASSETS / "models"
TMP = Path("/tmp") / "silk"
CACHE = TMP / "cache"
OWNER_USER_PATH = Path("/private/home/gleize")
DATA = OWNER_USER_PATH / "data"
DATASETS = DATA / "datasets"
