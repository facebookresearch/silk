# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys

import torch
import torchvision.transforms as tvf
from common import cache

sys.path.append("/checkpoint/weiyaowang/dev/r2d2/")

from extract import extract_multiscale, load_network, NonMaxSuppression

DATASET = "assets/datasets/cached-hpatches/hpatches-full-size-480-rgb.h5"
OUTPUT_DATASET = "r2d2_all_points_480.h5"
MODEL_WEIGHT = "/checkpoint/weiyaowang/metapoint/pretrained_weights/r2d2_WASF_N16.pt"


def mnn_matcher(descriptors_a, descriptors_b):
    device = descriptors_a.device
    sim = descriptors_a @ descriptors_b.t()
    nn12 = torch.max(sim, dim=1)[1]
    nn21 = torch.max(sim, dim=0)[1]
    ids1 = torch.arange(0, sim.shape[0], device=device)
    mask = ids1 == nn21[nn12]
    matches = torch.stack([ids1[mask], nn12[mask]])
    return matches.t().data.cpu().numpy()


def transform(elem, model, detector):
    # extract images (single image, no batch dimension)

    original_image = elem["original_img"][None].cuda() / 255.0
    warped_image = elem["warped_img"][None].cuda() / 255.0

    RGB_mean = [0.485, 0.456, 0.406]
    RGB_std = [0.229, 0.224, 0.225]

    norm_RGB = tvf.Normalize(mean=RGB_mean, std=RGB_std)

    with torch.no_grad():
        o_xys, o_desc, o_scores = extract_multiscale(
            model, norm_RGB(original_image), detector
        )
        w_xys, w_desc, w_scores = extract_multiscale(
            model, norm_RGB(warped_image), detector
        )

    def postprocess_results(xys, desc, scores):
        xys = xys.cpu().numpy()
        desc = desc.cpu().numpy()
        scores = scores.cpu().numpy()
        idxs = scores.argsort()
        return xys[idxs], desc[idxs], scores[idxs]

    o_xys, o_desc, o_scores = postprocess_results(o_xys, o_desc, o_scores)
    w_xys, w_desc, w_scores = postprocess_results(w_xys, w_desc, w_scores)

    with torch.no_grad():
        matches = mnn_matcher(
            torch.from_numpy(o_desc).cuda(), torch.from_numpy(w_desc).cuda()
        )

    # run model here
    original_points = torch.from_numpy(o_xys[:, :2])[..., [1, 0]]
    warped_points = torch.from_numpy(w_xys[:, :2])[..., [1, 0]]
    matched_original_points = torch.from_numpy(o_xys[matches[:, 0], :2])[..., [1, 0]]
    matched_warped_points = torch.from_numpy(w_xys[matches[:, 1], :2])[..., [1, 0]]

    # save keypoints
    elem = elem.add("original_points", original_points)
    elem = elem.add("warped_points", warped_points)
    elem = elem.add("matched_original_points", matched_original_points)
    elem = elem.add("matched_warped_points", matched_warped_points)

    return elem


def dense_transform(elem, model, detector):
    # extract images (single image, no batch dimension)

    original_image = elem["original_img"][None].cuda() / 255.0
    warped_image = elem["warped_img"][None].cuda() / 255.0

    RGB_mean = [0.485, 0.456, 0.406]
    RGB_std = [0.229, 0.224, 0.225]

    norm_RGB = tvf.Normalize(mean=RGB_mean, std=RGB_std)

    with torch.no_grad():
        o_xys, o_desc, o_scores = extract_multiscale(
            model, norm_RGB(original_image), detector
        )
        w_xys, w_desc, w_scores = extract_multiscale(
            model, norm_RGB(warped_image), detector
        )

    def postprocess_results(xys, desc, scores):
        xys = xys.cpu().numpy()
        desc = desc.cpu().numpy()
        scores = scores.cpu().numpy()
        idxs = scores.argsort()
        return xys[idxs], desc[idxs], scores[idxs]

    o_xys, o_desc, o_scores = postprocess_results(o_xys, o_desc, o_scores)
    w_xys, w_desc, w_scores = postprocess_results(w_xys, w_desc, w_scores)

    with torch.no_grad():
        matches = mnn_matcher(
            torch.from_numpy(o_desc).cuda(), torch.from_numpy(w_desc).cuda()
        )

    # run model here
    original_points = torch.from_numpy(o_xys[:, :2])[..., [1, 0]]
    warped_points = torch.from_numpy(w_xys[:, :2])[..., [1, 0]]
    matched_original_points = torch.from_numpy(o_xys[matches[:, 0], :2])[..., [1, 0]]
    matched_warped_points = torch.from_numpy(w_xys[matches[:, 1], :2])[..., [1, 0]]

    # save keypoints
    elem = elem.add("original_points", original_points)
    elem = elem.add("warped_points", warped_points)
    elem = elem.add("matched_original_points", matched_original_points)
    elem = elem.add("matched_warped_points", matched_warped_points)

    return elem


def main():
    model = load_network(MODEL_WEIGHT)
    model = model.cuda()
    model = model.eval()

    detector = NonMaxSuppression(rel_thr=0.7, rep_thr=0.7)

    cache(DATASET, OUTPUT_DATASET, transform, model, detector)


if __name__ == "__main__":
    main()
