#!/usr/bin/python
"""
@author: Christian Forster
"""

import os
import numpy as np
import transformations as tf


def compute_comparison_indices_time(t_gt, dist, max_error):
    max_idx = len(t_gt)
    comparisons = []
    for idx, d in enumerate(t_gt):
        j = -1
        error = max_error
        for i in range(idx, max_idx):
            if np.abs(t_gt[i]-(d+dist)) < error:
                j = i
                error = np.abs(t_gt[i]-(d+dist))
        comparisons.append(j)
    return comparisons


def get_distance_from_start(gt_translation):
    distances = np.diff(gt_translation[:, 0:3], axis=0)
    distances = np.sqrt(np.sum(np.multiply(distances, distances), 1))
    distances = np.cumsum(distances)
    distances = np.concatenate(([0], distances))
    return distances


def distances_along_trajectory(traj):
    """
    Compute the translational distances along a trajectory.
    """

    motion = [np.linalg.norm(traj[i, :]-traj[i+1, :])
              for i in range(len(traj)-1)]
    distances = [0]
    sum = 0
    for t in motion:
        sum += t
        distances.append(sum)
    return distances
