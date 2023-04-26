# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
from common import get_model, load_images
from silk.backbones.silk.silk import from_feature_coords_to_image_coords

IMAGE_0_PATH = "/datasets01/hpatches/01042022/v_adam/1.ppm"
IMAGE_1_PATH = "/datasets01/hpatches/01042022/v_adam/2.ppm"
OUTPUT_MODEL = "script_model.pt"


def test_on_image_pair(model, script_model, images):
    # run model
    positions_0, descriptors_0, _ = model(images)
    positions_1, descriptors_1, _ = script_model(images)

    # check result consistency
    assert len(positions_0) == len(positions_1) == 2
    assert torch.allclose(positions_0[0], positions_1[0])
    assert torch.allclose(positions_0[1], positions_1[1])

    assert len(descriptors_0) == len(descriptors_1) == 2
    assert torch.allclose(descriptors_0[0], descriptors_1[0])
    assert torch.allclose(descriptors_0[1], descriptors_1[1])


def model_with_corrected_positions(model):
    def fn(images):
        results = model(images)
        assert type(results) is tuple
        positions = results[
            0
        ]  # IMPORTANT : only works when positions are in first place
        positions = from_feature_coords_to_image_coords(model, positions)
        return (positions,) + results[1:]

    return fn


def main():
    # load image
    images = load_images(IMAGE_0_PATH, IMAGE_1_PATH)

    # load model
    model = get_model()
    model = model_with_corrected_positions(model)

    # trace model to torch script
    script_model = torch.jit.trace(model, images)

    # save model to disk
    torch.jit.save(script_model, OUTPUT_MODEL)
    # load model from disk (to test it below)
    script_model = torch.jit.load(OUTPUT_MODEL)

    # test on same size images
    test_on_image_pair(model, script_model, images)

    # test on downsampled images (to check shapes are not frozen during tracing)
    downsampled_images = torch.nn.functional.interpolate(images, scale_factor=0.5)
    test_on_image_pair(model, script_model, downsampled_images)

    print(f'torch script model "{OUTPUT_MODEL}" created and tested')
    print("done")


if __name__ == "__main__":
    main()
