#!/usr/bin/python
"""
@author: Christian Forster
"""

import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Cardo']})
rc('text', usetex=True)

FORMAT = '.pdf'


def plot_pointcloud_3d(results_dir, p_gt, p_es, scale, rot, trans):

    m = np.loadtxt(os.path.join(results_dir, 'pointcloud.txt'))
    m_aligned = np.zeros(np.shape(m))
    for i in range(len(m)):
        m_aligned[i, :] = scale * np.dot(rot, m[i, :]) + trans

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(m_aligned[:, 0], m_aligned[:, 1],
            m_aligned[:, 2], '.', ms=1, color='green')
    ax.plot(p_es[:, 0], p_es[:, 1], p_es[:, 2], linewidth=3,
            color='blue', label='SVO Bundle Adjust')
    ax.plot(p_gt[:, 0], p_gt[:, 1], p_gt[:, 2],
            linewidth=3, color='k', label='Groundtruth')
    ax.legend()
    fig.savefig(results_dir+'/trajectory_3d'+FORMAT)


def plot_trajectory(results_dir, p_gt, p_es, n_align_frames):

    # plot trajectory
    fig = plt.figure(figsize=(6, 5.5))
    ax = fig.add_subplot(111, aspect='equal', xlabel='x [m]', ylabel='y [m]')
    ax.grid(ls='--', color='0.7')
    p_es_0 = p_es - p_gt[0, :]
    p_gt_0 = p_gt - p_gt[0, :]
    ax.plot(p_es_0[:, 0], p_es_0[:, 1], 'b-', label='Estimate')
    ax.plot(p_gt_0[:, 0], p_gt_0[:, 1], 'r-', label='Groundtruth')
    if n_align_frames > 0:
        ax.plot(p_es_0[0:n_align_frames, 0],
                p_es_0[0:n_align_frames, 1],
                'g-', linewidth=2, label='aligned')
        for (x1, y1, z1), (x2, y2, z2) in zip(
                p_es_0[:n_align_frames:10, :], p_gt_0[:n_align_frames:10, :]):
            ax.plot([x1, x2], [y1, y2], '-', color="gray")

    ax.legend(loc='upper right')
    # ax.set_ylim([-0.5, 5])
    fig.tight_layout()
    fig.savefig(results_dir+'/trajectory_top'+FORMAT)

    # plot trajectory side
    fig = plt.figure(figsize=(6, 2.2))
    ax = fig.add_subplot(111, aspect='equal', xlabel='x [m]', ylabel='z [m]')
    ax.grid(ls='--', color='0.7')
    ax.plot(p_es[:, 0]-p_gt[0, 0], p_es[:, 2] -
            p_gt[0, 1], 'b-', label='Estimate')
    ax.plot(p_gt[:, 0]-p_gt[0, 0], p_gt[:, 2] -
            p_gt[0, 1], 'r-', label='Groundtruth')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, 'trajectory_side'+FORMAT))

    # write aligned trajectory to file
    file_out = open(os.path.join(results_dir, 'trajectory.csv'), 'w')
    file_out.write(
        '# estimate-x [m], estimate-y [m], estimate-z [m],
        groundtruth-x [m], groundtruth-y [m], groundtruth-z [m]\n')
    for i in range(len(p_es)):
        file_out.write(
            '%.6f, %.6f, %.6f, %.6f, %.6f, %.6f\n' %
            (p_es[i, 0]-p_gt[0, 0], p_es[i, 1]-p_gt[0, 1],
             p_es[i, 2]-p_gt[0, 2],
             p_gt[i, 0]-p_gt[0, 0], p_gt[i, 1]-p_gt[0, 1],
             p_gt[i, 2]-p_gt[0, 2]))
    file_out.close()
