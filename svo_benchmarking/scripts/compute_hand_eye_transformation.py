#!/usr/bin/env python2

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from colorama import init, Fore

import svo_benchmarking.transformations as tfs
import svo_benchmarking.tfs_utils as tu
import svo_benchmarking.hand_eye_calib as hand_eye
import svo_benchmarking.results_loading as res_load

rc('font', **{'family': 'serif', 'serif': ['Cardo']})
rc('text', usetex=True)
FORMAT = '.png'

init(autoreset=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''
    analyze the pure IMU integration when the visual tracking fails
    ''')
    parser.add_argument(
        'trace_dir', help='directory containing the trace files')
    parser.add_argument(
        'start_id', type=int, default=0, help='start of analysis')
    parser.add_argument(
        'end_id', type=int, default=-1, help='end of analysis (-1: end)')
    parser.add_argument(
        '--delta', type=int, default=10, help='interval for relative pose')
    args = parser.parse_args()

    assert os.path.exists(args.trace_dir)
    print("Going to calculate hand-eye in {0} from {1} to {2}".format(
        args.trace_dir, args.start_id, args.end_id))
    unique_name = '_'.join(
        ['s'+str(args.start_id), 'e'+str(args.end_id), 'd'+str(args.delta)])
    save_dir = os.path.join(args.trace_dir, 'hand_eye')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print("The results/plots will be written to {0}.".format(save_dir))

    print(Fore.GREEN + "> Loading estimate...")
    # extract state estimate
    # we assume the extracted poses are of the same order
    es_id, es_t_sec, es_p, es_q, _, gt_p, gt_q =\
        res_load.load_traj_est_gt_matched(args.trace_dir, 0)
    _, _, _, tracking_quality, _ = res_load.load_status(args.trace_dir, 0)

    valid_es_entries = []
    for i, v in enumerate(es_id):
        if v < args.start_id:
            continue
        if args.end_id > 0 and v >= args.end_id:
            break
        assert tracking_quality[v] == 'Good', "Bad tracking encountered."
        valid_es_entries.append(i)

    es_id = es_id[valid_es_entries]
    es_t_sec = es_t_sec[valid_es_entries]

    es_p = es_p[valid_es_entries]
    es_q = es_q[valid_es_entries]
    es_T = [tu.get_rigid_body_trafo(q, t) for q, t in zip(es_q, es_p)]

    gt_p = gt_p[valid_es_entries]
    gt_q = gt_q[valid_es_entries]
    gt_T = [tu.get_rigid_body_trafo(q, t) for q, t in zip(gt_q, gt_p)]

    print("Extact estimate range from {0} to {1}".format(es_id[0],
                                                         es_id[-1]))

    N = es_p.shape[0]
    indices = np.arange(N - args.delta)
    print("Going to use {0} pairs from {1} to {2}.".format(
        indices.shape[0], indices[0], indices[-1]))
    R, t = hand_eye.hand_eye_calib(gt_q, es_q, gt_p, es_p, indices,
                                   args.delta)
    T_EH = np.identity(4)
    T_EH[0:3, 0:3] = R
    T_EH[0:3, 3] = np.ravel(t)
    T_HE = np.linalg.inv(T_EH)
    print("Computed hand-eye transformation is:\n{0}.".format(T_HE))

    gt_T = [np.dot(v, T_HE) for v in gt_T]
    gt_p = np.array([T[0:3, 3] for T in gt_T])
    gt_q = np.array([T[0:3, 0:3] for T in gt_T])

    es_T0 = es_T[0]
    gt_T0 = gt_T[0]
    align_T = np.dot(gt_T0, np.linalg.inv(es_T0))
    aligned_es_T = [np.dot(align_T, v) for v in es_T]

    aligned_es_p = np.array([T[0:3, 3] for T in aligned_es_T])

    fig = plt.figure()
    ax = fig.add_subplot(311, xlabel='t (sec)', ylabel='x (m)}')
    ax.plot(es_t_sec, aligned_es_p[:, 0], label='est')
    ax.plot(es_t_sec, gt_p[:, 0], label='gt')
    ax = fig.add_subplot(312, xlabel='t (sec)', ylabel='y (m)')
    ax.plot(es_t_sec, aligned_es_p[:, 1], label='est')
    ax.plot(es_t_sec, gt_p[:, 1], label='gt')
    ax = fig.add_subplot(313, xlabel='t (sec)', ylabel='z (m)')
    ax.plot(es_t_sec, aligned_es_p[:, 2], label='est')
    ax.plot(es_t_sec, gt_p[:, 2], label='gt')
    plt.legend()
    fig.savefig(save_dir+'/'+unique_name+'aligned_position_est_gt'+FORMAT)

    np.savetxt(save_dir+'/'+unique_name+'T_HE.txt', T_HE, header='# T_HE')
