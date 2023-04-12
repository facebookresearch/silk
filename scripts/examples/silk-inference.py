# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os

from common import get_model, load_images, SILK_MATCHER
from silk.backbones.silk.silk import from_feature_coords_to_image_coords
from silk.cli.image_pair_visualization import create_img_pair_visual, save_image

IMAGE_0_PATH = "/datasets01/hpatches/01042022/v_adam/1.ppm"
IMAGE_1_PATH = "/datasets01/hpatches/01042022/v_adam/2.ppm"
OUTPUT_IMAGE_PATH = "./img.png"


def main():
    # load image
    images_0 = load_images(IMAGE_0_PATH)
    images_1 = load_images(IMAGE_1_PATH)

    # load model
    model = get_model(default_outputs=("sparse_positions", "sparse_descriptors"))

    # run model
    sparse_positions_0, sparse_descriptors_0 = model(images_0)
    sparse_positions_1, sparse_descriptors_1 = model(images_1)

    sparse_positions_0 = from_feature_coords_to_image_coords(model, sparse_positions_0)
    sparse_positions_1 = from_feature_coords_to_image_coords(model, sparse_positions_1)

    # get matches
    matches = SILK_MATCHER(sparse_descriptors_0[0], sparse_descriptors_1[0])

    # create output image
    image_pair = create_img_pair_visual(
        IMAGE_0_PATH,
        IMAGE_1_PATH,
        None,
        None,
        sparse_positions_0[0][matches[:, 0]].detach().cpu().numpy(),
        sparse_positions_1[0][matches[:, 1]].detach().cpu().numpy(),
    )

    save_image(
        image_pair,
        os.path.dirname(OUTPUT_IMAGE_PATH),
        os.path.basename(OUTPUT_IMAGE_PATH),
    )

    print(f"result saved in {OUTPUT_IMAGE_PATH}")
    print("done")


if __name__ == "__main__":
    main()
