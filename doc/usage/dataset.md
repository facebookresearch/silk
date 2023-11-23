# Setup Datasets

The several datasets are configured in `./etc/datasets/`.
Below, we list each dataset, how to download them, and how to change their configuration to point to your local files.

## Evaluation

### HPatches

__Download__ : https://hpatches.github.io/ ([test](http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz))

Once downloaded, update the path `hpatches_path` in the configuration file `./etc/datasets/hpatches/defaults.yaml`.

## Training

### COCO Dataset

__Download__ : https://cocodataset.org/#download ([train](http://images.cocodataset.org/zips/train2017.zip), [val](http://images.cocodataset.org/zips/val2017.zip), [test](http://images.cocodataset.org/zips/test2017.zip))

Once downloaded, update the path `root` in the configuration files `./etc/datasets/coco/{training,validation,test}.yaml`. The field `annFile` parameter **should** point to the proper annotation file (even though annotations aren't used), otherwise the resulting dataset will be considered **empty**.

### ImageNet

__Download__ : https://www.image-net.org/download.php

Once downloaded, update the path `root` in the configuration files `./etc/datasets/image-net/{training,validation}.yaml`.

### ScanNet

__Download__ : https://github.com/ScanNet/ScanNet#scannet-data

Once downloaded, update the path `path` in the configuration files `./etc/datasets/scannet-frames/{training-all,test}.yaml`.

<u>Remark #1</u> : We've designed our own dataset class for ScanNet that loads directly from the original raw source. There is no need to run a frame extractor (suggested by ScanNet).

<u>Remark #2</u> : Additionally, our class does cache a few things like frame offsets in order to speed-up random access to frames. As a consequence, the first access to a scan will be slow, but should be faster afterwards. The cache path can be changed by changing the `cache_path` field.

### MegaDepth

__Download__ : https://www.cs.cornell.edu/projects/megadepth/ ([all](https://www.cs.cornell.edu/projects/megadepth/dataset/Megadepth_v1/MegaDepth_v1.tar.gz))

Once downloaded, update the path `root` in the configuration files `./etc/datasets/megadepth/defaults.yaml`.
