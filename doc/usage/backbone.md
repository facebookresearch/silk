# Backbone

## Overview

Adding a backbone should be a fairly straightforward process. The main backbone we use for SiLK is the `ParametricVGG` found in `lib/backbones/superpoint/vgg.py`.

A backbone is a __PyTorch Module__ with a few constraints :
* It should take images as input
* It should output dense feature maps per image
* It should subclass `CoordinateMappingProvider` and implement a `mappings()` method that provides the bijective coordinate map between the feature coordinate space and the input coordinate space (in order to know where an output feature is located in the image). Convenient functions are provided to automate the process, and one can get inspiration from how `ParametricVGG.mappings()` is implemented.
* Once the new backbone is implemented, its configuration file should be added to `etc/backbones/`.

## We can help

We make heavy use of [hydra](https://hydra.cc/docs/intro/) to handle compositionality of our configurations. This can feel a bit daunting at first.
So we can help you with the integration of a new backbone, by following the steps below:
1. Implement the backbone and add the code to `lib/backbones`, with simple unit tests.
1. Create a pull request with the title `Integrate <name> backbone.`
1. At that point, we can help and iterate on the pull request to make sure the backbone is properly integrated.
