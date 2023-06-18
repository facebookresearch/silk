# Inference

## Script Example

An example of using SiLK for inference is provided in `./scripts/examples/silk-inference.py`. The pre-trained models and inference parameters are specified in `./scripts/examples/common.py`.

A very important parameter is the `SILK_DEFAULT_OUTPUT`. It specifies the output(s) that is required from the model. When running the model, it will output a tuple of the same size as `SILK_DEFAULT_OUTPUT`, with the corresponding output at each position.

Valid output names for `SILK_DEFAULT_OUTPUT` are :
* `features` : shared feature map being fed to the keypoint head and descriptor head.
* `logits` : dense output of the keypoint head (pre-sigmoid)
* `probability` : dense output of the keypoint head (post-sigmoid)
* `raw_descriptors` : unnormalized dense descriptors
* `sparse_positions` : positions of keypoints (with confidence value)
* `sparse_descriptors` : descriptors of each keypoints
* `dense_positions` : positions of all pixels
* `dense_descriptors` : descriptors of all pixels

<u>Remark</u> : SiLK will **ONLY** run the minimum required computation to obtain the specified output. Any irrelevant computation will be avoided, in order to keep the computation effcient. For example, if one specifies to output "dense_descriptors" only, the keypoint scoring branch of the model will never be computed.

## CLI Tools

Additional command line tools are provided for convenient keypoint extraction, matching and visualization.
Here is an example of exhaustive matching from an image folder `$MY_IMAGES`.

```bash
#!/bin/bash
set -e

# folders param
TAG=example
IMAGES=$MY_IMAGES
IMAGES_EXTENSION=jpg
OUT=var/cli/$TAG

# model / matching params
CHECKPOINT=assets/models/silk/coco-rgb-aug.ckpt
TOPK=10000
SIZE=480

mkdir -p $OUT

# extract features
./bin/silk-features -o -m pvgg-4 -c $CHECKPOINT -k $TOPK -d $OUT/features -s $SIZE $IMAGES/*.$IMAGES_EXTENSION
# generate image pairs to match
ls $OUT/features/*.pt | sort -V | xargs ./bin/generate-matches exhaustive -s > $OUT/matches.txt
# match keypoints from generated image pairs
./bin/silk-matching -o -m double-softmax -t 0.9 $OUT/matches.txt $OUT/matches/
# visualize matches
./bin/silk-viz image -o $OUT/viz $OUT/matches/*.pt

```

Features, matches and visualizations can be found in `var/cli/example` after running this script at the root of this repository.
