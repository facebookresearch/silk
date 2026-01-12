# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
CLI function for PointTracking video visualization tool.
"""

import os

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from silk.config.core import instantiate_and_ensure_is_instance
from silk.models.pointtracker import PointTracker
from silk.models.superpoint_utils import _process_output_new
from tqdm import tqdm


def load_video_frames_opencv(video_file_path, H, W):
    """
    Load a list of frames from a video and prepare each frame in the list
    for input to the SuperPoint model.

    Args:
        video_file_path (str): the path to the video file
        H (int): the height of the reshaped video
        W (int): the width of the reshaped video

    Returns:
        model_input_frames (list): a list of tensors for each frame
            where each tensor has shape (1, 1, H, W) for input to the model
    """
    video = cv2.VideoCapture(video_file_path)

    frame_list = []
    success, frame = video.read()

    while success:
        frame_list.append(frame)
        success, frame = video.read()

    model_input_frames = []
    for frame in frame_list:
        input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        input_image = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_AREA)

        input_image = input_image.astype("float32") / 255.0
        input_image = torch.from_numpy(input_image)
        input_image = input_image.view(1, 1, H, W)

        model_input_frames.append(input_image)

    return model_input_frames


def draw_tracks(image, tracks, all_points, offsets):
    """
    Draw the tracks from the point tracker on an image.

    Args:
        image (numpy): an image array with size H x W x 3 on which
            the tracks are drawn
        tracks (numpy): tracks matrix of size num_tracks x (2+max_length)
            containing track data
        all_points (list): list of most recently tracked points
        offsets (tensor): num_tracks length array with integer offset locations

    Returns:
        image (numpy): an image with the drawn tracks
    """
    # store the number of points
    all_points = [elem.detach().numpy() for elem in all_points]
    num_points = len(all_points)

    # width of track and point circles to be drawn
    stroke = 1

    # colormap for visualization
    colormap = np.array(
        [
            [0.0, 0.0, 0.5],
            [0.0, 0.0, 0.99910873],
            [0.0, 0.37843137, 1.0],
            [0.0, 0.83333333, 1.0],
            [0.30044276, 1.0, 0.66729918],
            [0.66729918, 1.0, 0.30044276],
            [1.0, 0.90123457, 0.0],
            [1.0, 0.48002905, 0.0],
            [0.99910873, 0.07334786, 0.0],
            [0.5, 0.0, 0.0],
        ]
    )

    # iterate through each track and draw the point and track line
    for track in tracks:
        # choose one of 10 possible colors corresponding to the value of the
        # avg_desc_score for each track (with greater scores closer to red and lower
        # scores closer to blue)
        color = colormap[int(np.clip(np.floor(track[1] * 10), 0, 9)), :] * 255

        # iterate through each of the points predicted on the image
        for i in range(num_points - 1):
            # skip tracks if there was no tracked point
            if track[i + 2] == -1 or track[i + 3] == -1:
                continue

            offset1 = offsets[i]
            offset2 = offsets[i + 1]

            idx1 = int(track[i + 2] - offset1)
            idx2 = int(track[i + 3] - offset2)

            pt1 = all_points[i][:2, idx1]
            pt2 = all_points[i + 1][:2, idx2]

            p1 = (int(round(pt1[0])), int(round(pt1[1])))
            p2 = (int(round(pt2[0])), int(round(pt2[1])))

            cv2.line(image, p1, p2, color, thickness=stroke, lineType=16)

            # draw end points of each track
            if i == num_points - 2:
                clr2 = (255, 0, 0)
                cv2.circle(image, p2, stroke, clr2, -1, lineType=16)

    return image


def display_point_tracks(
    image, tracks, dist_thresh, all_points, offsets, display_scale=2
):
    """
    Displays the predicted points and tracks on one image.

    Args:
        image (tensor): the image from the frame list as output by the
            model on which to draw tracks
        tracks (tensor): the tracks from the pointtracker
        dist_thresh (int): the distance threshold from the point tracker
        all_points (list): most recently tracked points list from the point tracker
        offsets (tensor): tensor of offset locations from the point tracker
        display_scale (int): the factor to scale output visualization (paper's
            default is 2)

    Returns:
        output_image (cv2 image): the image with the points and tracks
    """
    # convert image and tracks to numpy for drawing
    image = image.squeeze(dim=0).squeeze(dim=0).detach().numpy()
    tracks = tracks.detach().numpy()

    H = image.shape[0]
    W = image.shape[1]

    output_img_tracks = (np.dstack((image, image, image)) * 255.0).astype("uint8")

    # normalize track scores to be in the range [0, 1]
    tracks[:, 1] /= float(dist_thresh)

    # update the output_img_tracks image with the tracks
    tracks_image = draw_tracks(output_img_tracks, tracks, all_points, offsets)

    # resize final output image
    output_image = cv2.resize(tracks_image, (display_scale * W, display_scale * H))

    return output_image


def create_visualizations(
    video, model, pointtracker, min_length=2, height=112, width=152
):
    """
    Create the video with visualized point tracks.

    This function does the following:
        1. Load in the video file
        2. Run the SuperPoint model on each frame
        3. Get the tracks from the point tracker for each frame
        4. Draw the point tracks on frame
        5. Get an output list containing the visualized point tracks
        for all frames in the input video

    Args:
        video (cv2 video): the input video
        model (SuperPoint): the superpoint model
        pointtracker (PointTracker): a pointtracker object
        min_length (int): minimum track length
        height (int): resized height dimension
        width (int): resized width dimension
    """
    # get the video frames
    frame_list = load_video_frames_opencv(video, height, width)

    demo_output = []

    for img in tqdm(frame_list):
        # run the model on the image
        output_points, output_desc = _process_output_new(model, img)

        # add points and descriptors to the pointtracker
        pointtracker.update(output_points[:, [1, 0]].T, output_desc.T)

        # get tracks for points which were matched successfully across all frames
        tracks = pointtracker.get_tracks(min_length=min_length)

        # display point tracks overlayed on top of input image
        output_image = display_point_tracks(
            img,
            tracks,
            pointtracker.dist_thresh,
            pointtracker.all_points,
            pointtracker.get_offsets(),
        )

        demo_output.append(output_image)

    return demo_output


def save_video(frame_list, output_location, output_vid_name="output_vid.mp4"):
    """
    Takes a list of images and converts them to an mp4 file saved
    at location output_dir.

    Args:
        frame_list (list): a list of cv2 images to be converted to a video
        output_location (str): the file path for the output video
        output_vid_name (optional str): the name for the output video

    Returns:
        None
    """
    img = frame_list[0]
    height, width, layers = img.shape
    frames_per_second = 15  # slower output videos

    # create the directory if it does not exist
    os.makedirs(output_location, exist_ok=True)

    output_file = cv2.VideoWriter(
        os.path.join(output_location, output_vid_name),
        fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
        fps=float(frames_per_second),
        frameSize=(width, height),
    )

    # write the frames to the video writer
    for frame in frame_list:
        output_file.write(frame)

    output_file.release()


def main(config: DictConfig):
    # load model
    model = instantiate_and_ensure_is_instance(config.mode.model, pl.LightningModule)

    # create the point tracker
    tracker = PointTracker(
        max_length=config.mode.max_length, dist_thresh=config.mode.dist_thresh
    )

    # create the visualization video
    demo_output = create_visualizations(config.mode.video_file_path, model, tracker)

    # optionally save the output video
    if config.mode.save_output:
        save_video(
            demo_output,
            config.mode.save_location,
            output_vid_name=config.mode.save_file_name,
        )

    return {"config": OmegaConf.to_object(config.mode.model)}
