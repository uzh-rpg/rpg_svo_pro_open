#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['Cardo']})
rc('text', usetex=True)

FORMAT = '.pdf'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Analyze the timing for loop closing''')
    parser.add_argument(
        'result_dir', type=str,
        help="Folder containing the groundtruth and the estimate.")
    args = parser.parse_args()

    abs_res_dir = os.path.abspath(args.result_dir)
    lc_fn = os.path.join(abs_res_dir, 'lc_timing.txt')
    pg_fn = os.path.join(abs_res_dir, 'pg_timing.txt')
    nqueris_fn = os.path.join(abs_res_dir, 'num_queries.txt')
    assert os.path.exists(lc_fn)
    assert os.path.exists(pg_fn)
    assert os.path.exists(nqueris_fn)

    print("Going to analyze:")
    print('- {0}'.format(lc_fn))
    print('- {0}'.format(pg_fn))
    print('- {0}'.format(nqueris_fn))

    lc_data = np.loadtxt(lc_fn)
    pg_data = np.loadtxt(pg_fn)
    nqueris_data = np.loadtxt(nqueris_fn, dtype=int)

    total_queries = np.sum(nqueris_data)
    n_frames = nqueris_data.size
    assert total_queries == lc_data.shape[0]
    assert lc_data.shape[1] == 4
    print("Find query data for {0} keyframes.".format(n_frames))

    t_bow = lc_data[:, 0]
    t_gveri = lc_data[:, 1]
    t_ume = lc_data[:, 2]
    t_trans = lc_data[:, 3]

    t_sum1 = t_bow + t_gveri
    t_sum2 = t_sum1 + t_ume
    t_total = t_sum2 + t_trans

    t_bow_kf = np.zeros(n_frames)
    t_gveri_kf = np.zeros(n_frames)
    t_ume_kf = np.zeros(n_frames)
    t_trans_kf = np.zeros(n_frames)
    t_sum1_kf = np.zeros(n_frames)
    t_sum2_kf = np.zeros(n_frames)
    t_total_kf = np.zeros(n_frames)
    s = 0
    for idx in range(n_frames):
        n = nqueris_data[idx]
        t_bow_kf[idx] = np.sum(t_bow[s:s+n])
        t_gveri_kf[idx] = np.sum(t_gveri[s:s+n])
        t_ume_kf[idx] = np.sum(t_ume[s:s+n])
        t_trans_kf[idx] = np.sum(t_trans[s:s+n])

        t_sum1_kf[idx] = np.sum(t_sum1[s:s+n])
        t_sum2_kf[idx] = np.sum(t_sum2[s:s+n])
        t_total_kf[idx] = np.sum(t_total[s:s+n])
        s = s + n
    assert s == total_queries

    suc_detect_idx = np.nonzero(t_ume_kf)[0]
    suc_closure_idx = np.nonzero(t_trans_kf)[0]
    print("{0} success detections and {1} closures.".format(
        suc_detect_idx.size, suc_closure_idx.size))
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(411, ylabel='Time (s)')
    ax.plot(t_bow_kf, label='BoW Check')
    plt.legend()
    ax = fig.add_subplot(412, ylabel='Time (s)')
    ax.plot(t_gveri_kf, label='2D Geo. Veri.')
    plt.legend()
    ax = fig.add_subplot(413, ylabel='Time (s)')
    ax.plot(t_ume_kf, label='3D Veri.')
    plt.legend()
    ax = fig.add_subplot(414, xlabel='Keyframe Index', ylabel='Time (s)')
    ax.plot(t_trans_kf, label='Transform VIO')
    plt.legend()
    fig.savefig(abs_res_dir+'/'+'lc_timing_parts'+FORMAT, bbox_inches="tight")
    plt.close(fig)

    t_suc_bow_kf = t_bow_kf[suc_detect_idx]
    t_suc_gveri_kf = t_gveri_kf[suc_detect_idx]
    t_suc_ume_kf = t_ume_kf[suc_detect_idx]
    t_suc_trans_kf = t_trans_kf[suc_closure_idx]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(311, ylabel='Time (s)')
    ax.plot(t_suc_bow_kf, label='BoW check')
    plt.legend()
    ax = fig.add_subplot(312, ylabel='Time (s)')
    ax.plot(t_suc_gveri_kf, label='2D Geo. Veri.')
    plt.legend()
    ax = fig.add_subplot(313, xlabel='Keyframe Index', ylabel='Time (s)')
    ax.plot(t_suc_ume_kf, label='3D Veri.')
    plt.legend()
    fig.savefig(abs_res_dir+'/'+'lc_timing_suc'+FORMAT, bbox_inches="tight")
    plt.close(fig)

    stats_trans_vio = {}
    stats_trans_vio['mean'] = np.mean(t_suc_trans_kf)
    stats_trans_vio['std'] = np.std(t_suc_trans_kf)
    print("VIO transformation timing stats:")
    print(t_suc_trans_kf)
    print(stats_trans_vio)
    print("Average total time of all selected keyframes: {0}".format(
        np.mean(t_total_kf)))
    print("Average detection time of all selected keyframes: {0}".format(
        np.mean(t_sum2_kf)))
