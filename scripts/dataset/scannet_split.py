# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json

import numpy.random as rd
from silk.datasets.scannet.helper import Txt
from silk.datasets.scannet.scans import ScansDataset

VALIDATION_SPLIT_PERCENT = 0.02
SEED = 0

ds = ScansDataset(
    path="/datasets01/scannet/082518/scans",
    cache_path="/tmp/silk/cache/scannet",
)

class_to_scans = {key: [] for key in Txt.SCENE_TYPE_LIST}

for scan in ds:
    class_to_scans[scan.txt["sceneType"]].append(scan.uid)

rd.seed(SEED)
val_ids = []
train_ids = []
for key, scans in class_to_scans.items():
    val_n = max(int(VALIDATION_SPLIT_PERCENT * len(scans)), 1)
    print(f"[{key}] - split {val_n}/{len(scans) - val_n}")
    rd.shuffle(scans)
    val_ids.extend(scans[:val_n])
    train_ids.extend(scans[val_n:])

with open("validation_split.json", "w") as f:
    json.dump(val_ids, f, indent=2)

with open("train_split.json", "w") as f:
    json.dump(train_ids, f, indent=2)

assert len(set(val_ids).intersection(set(train_ids))) == 0
