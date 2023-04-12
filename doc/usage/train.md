# Training

## Default

The default training can be run using the command `./bin/silk-cli mode=train-silk`.

<u>__Important__</u> : Please be aware that the training requires a minimum of __two GPUs__. Since the descriptor loss is computationally expensive and memory heavy, it is delegated to a jax optimized piece of code running on GPU 0. That code handles the computation of the large similarity matrix in a scanning fashion, and is also re-computed during back-propagation (instead of being stored). Everything else (model computation and back-propagation) is handled by PyTorch on GPU 1.

### Training Set

The default training set is COCO, and can be changed in file `./etc/mode/train-silk.yaml`.
Simply comment/uncomment the appropriate lines in the `### Datasets` section at the beginning of the file.

### Backbone

The default backbone is VGG-4, and can be changed in file `./etc/models/silk-vgg.yaml`. This file contains a list of backbones mentioned in the paper.

Adding a new custom backbone is also possible, but requires a bit more work. It is explained [here](backbone.md).

<u>Remark</u> : Changing the backbone will likely change the descriptor spatial resolution (because of unpadded convolutions and downsampling layers). Therefore, it might make sense to change the image input size in order to avoid having too few descriptors when computing the loss. The training input image size can be changed in `./etc/mode/train-silk.yaml`. It is a parameter of the `RandomCrop` transform.
