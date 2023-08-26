# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
The PointTracker class to track predicted corner points (as predicted by
SuperPoint descriptors) across multiple frames/images of the same scene
(for instance in a video, or across homography changes for an image).
"""

import torch

from silk.matching.mnn import compute_dist, match_descriptors

MAX_SCORE = 9999


class PointTracker:
    def __init__(self, max_length: int = 5, dist_thresh: float = 0.7):
        """
        Initialize the PointTracker class.

        Args:
            max_length (int): maximum length of point tracks, corresponding to the
                maximum number of points (predicted corners) to include in a track
                (paper's default is 5)
            dist_thresh (float): descriptor matching threshold (paper's default is 0.7)
        """
        self.max_length = max_length
        self.dist_thresh = dist_thresh

        # keep a list of most recently tracked points
        # used for visualization purposes
        self.all_points = [torch.zeros((2, 0)) for i in range(self.max_length)]

        self.last_desc = None

        # initialize the tracks tensor which will have size num_tracks x (2+max_length)
        # each row corresponds to a track: [track_id, avg_desc_score, point_0_id, ..., point_(max_length-1)_id]
        self.tracks = torch.zeros((0, self.max_length + 2))

        self.track_count = 0

    def get_offsets(self):
        """
        Iterate through list of points and accumulate an offset value. Used to
        index the global point IDs into the list of points.

        Args:
            None

        Returns:
            offsets (tensor): N length array with integer offset locations
        """
        # compute id offsets
        offsets = torch.zeros(1)

        for i in range(len(self.all_points) - 1):
            offsets = torch.cat(
                (offsets, torch.tensor([int(self.all_points[i].shape[1])]))
            )

        offsets = torch.cumsum(offsets, dim=0)

        return offsets

    def update(self, output_points, output_desc):
        """
        Adds a new set of points and descriptors to the tracker.

        Updates the PointTracker object with the output points and descriptor
        from running the SuperPoint model on the next image.

        Args:
            output_points (tensor): the points tensor of shape 3 x num_predicted_corners
                where each predicted corner has value (col_location, row_location, prob) as
                output by process_output in superpoint_utils.py
            output_desc (tensor): the descriptor tensor of shape 256 x num_predicted_corners
                where each predicted corner has a 256-dimension descriptor vector (as output
                by process_output in superpoint_utils.py)

        Returns:
            None
        """
        # initialize last_desc with the proper size
        if self.last_desc is None:
            self.last_desc = torch.zeros((output_desc.shape[0], 0))

        remove_size = self.all_points[0].shape[1]

        # update all_points by removing oldest point and adding newest
        self.all_points.pop(0)
        self.all_points.append(output_points)

        # remove oldest point in tracks, which is at index 2
        self.tracks = torch.hstack((self.tracks[:, :2], self.tracks[:, 3:]))

        # update track offsets
        for i in range(2, self.tracks.shape[1]):
            self.tracks[:, i] -= remove_size

        self.tracks[:, 2:][self.tracks[:, 2:] < -1] = -1

        # add a new -1 column
        self.tracks = torch.hstack(
            (self.tracks, -1 * torch.ones((self.tracks.shape[0], 1)))
        )

        offsets = self.get_offsets()

        # try to append to existing tracks
        matched = torch.zeros((output_points.shape[1]), dtype=torch.bool)

        if self.last_desc.shape[1] == 0 or output_desc.shape[1] == 0:
            matches = torch.zeros((3, 0))
        else:
            distances = compute_dist(self.last_desc.T, output_desc.T)
            matches = match_descriptors(distances)
            mdist = distances[matches[:, 0], matches[:, 1]].unsqueeze(-1)
            matches = torch.cat((matches, mdist), dim=-1)
            matches = matches.T

        for match in matches.T:
            # add a new point to its matched track
            id1 = int(match[0]) + offsets[-2]
            id2 = int(match[1]) + offsets[-1]
            found = torch.nonzero(self.tracks[:, -2] == id1)

            if found.shape[0] > 0:
                matched[int(match[1])] = True
                row = int(found)
                self.tracks[row, -1] = id2

                # initialize track score
                if self.tracks[row, 1] == MAX_SCORE:
                    self.tracks[row, 1] = match[2]

                # update track score with running average of descriptor distances
                else:
                    # NOTE: this running average can contain scores from old matches
                    # not contained in last max_length track points
                    # (consistent with the paper's implementation)
                    track_len = (self.tracks[row, 2:] != -1).sum() - 1.0
                    frac = 1.0 / float(track_len)
                    self.tracks[row, 1] = (1.0 - frac) * self.tracks[
                        row, 1
                    ] + frac * match[2]

        # add unmatched tracks
        new_ids = torch.arange(output_points.shape[1]) + offsets[-1]
        new_ids = new_ids[~matched]

        new_tracks = -1 * torch.ones((new_ids.shape[0], self.max_length + 2))
        new_tracks[:, -1] = new_ids

        new_num = new_ids.shape[0]

        new_track_ids = self.track_count + torch.arange(new_num)

        # update the tracks tensor with new track ids and placeholder scores
        new_tracks[:, 0] = new_track_ids
        new_tracks[:, 1] = MAX_SCORE * torch.ones(new_ids.shape[0])

        self.tracks = torch.vstack((self.tracks, new_tracks))

        self.track_count += new_num

        # remove any empty tracks (those that aren't storing any points)
        non_empty = torch.any(self.tracks[:, 2:] >= 0, dim=1)
        self.tracks = self.tracks[non_empty, :]

        # update the last descriptor
        self.last_desc = output_desc

    def get_tracks(self, min_length: int = 2):
        """
        Retrieve point tracks of a given minimum length.

        Args:
            min_length (int): the minimum track length, must be >= 1

        Returns:
            returned_tracks (tensor): a tensor with shape (num_tracks, 2 + max_track_length)
                with each track containing the track id, the average descriptor score for that
                track, and the max_track_length point ids for each point in the track
        """
        good_len = torch.sum(self.tracks[:, 2:] != -1, axis=1) >= min_length

        # remove the tracks that do not have a point in most recent frame
        not_recent = self.tracks[:, -1] != -1

        to_keep = torch.logical_and(good_len, not_recent)
        returned_tracks = self.tracks[to_keep, :]

        return returned_tracks


def nearest_neighbor_match(desc1, desc2, dist_thresh):
    """
    Match two sets of descriptors with nearest-neighbor approach.

    Takes the descriptors from the SuperPoint model's output on two images
    and uses a nearest-neighbor matching approach to find matched points
    across the two images. The nearest-neighbor matching is performed two
    ways such that the nearest-neighbor match from descriptor A to B is equal
    to the nearest-neighbor match from descriptor B to A.

    Args:
        desc1 (tensor): the first image's descriptor matrix with size
            (D, img_1_num_pred_corners), where D = 256, the length of the descriptor vector
        desc2 (tensor): the second image's descriptor matrix with size
            (D, img_2_num_pred_corners), where D = 256, the length of the descriptor vector
        dist_thresh (tensor): descriptor distance below which two descriptors are a match

    Returns:
        descriptor_matches (tensor): a tensor of size (3, num_matched_descriptors)
            where num_matched_descriptors will be less than or equal to the smaller of
            img_1_num_pred_corners and img_2_num_pred_corners. The descriptor_matches
            tensor contains matched pairs of descriptors and their distances, with each column
            equal to (index of descriptor in image 1, index of descriptor in image 2, distance)^T
    """
    # if no descriptors for an image, return no matches
    if desc1.shape[1] == 0 or desc2.shape[1] == 0:
        return torch.zeros((3, 0))

    # compute L2 distance between each pair of descriptors
    desc_dist = torch.cdist(desc1.T, desc2.T)

    # the col indices for the smallest value in each row (the matches)
    col_indices = torch.argmin(desc_dist, dim=1)

    # the row indices for the matches are simply 0 through the number of rows
    row_indices = torch.arange(desc_dist.shape[0])

    # the distances are at each row, col location in the desc_dist tensor
    distances = desc_dist[row_indices, col_indices]

    # keep only the descriptor matches with distance below threshold
    keep = distances < dist_thresh

    # only keep matches if the nearest-neighbor match goes both directions
    dim_0_row_indices = torch.argmin(desc_dist, dim=0)
    keep_dim_0 = torch.arange(len(col_indices)) == dim_0_row_indices[col_indices]

    keep = torch.logical_and(keep, keep_dim_0)

    # get the descriptor indices and the distances at the kept descriptor locations
    desc1_indices = row_indices[keep]
    desc2_indices = col_indices[keep]
    distances = distances[keep]

    # output is a (3, num_matched_descriptors) tensor
    descriptor_matches = torch.vstack((desc1_indices, desc2_indices, distances))

    return descriptor_matches
