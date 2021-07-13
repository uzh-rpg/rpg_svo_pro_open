#!/usr/bin/env python2

import os
import argparse
from shutil import copy2


kNsToEstFnMapping = {'traj_est': 'stamped_traj_estimate',
                     'pose_graph': 'stamped_pose_graph_estimate',
                     'ba_estimate': 'stamped_ba_estimate'}
kNsToAlgoMapping = {'traj_est': 'svo_ceres_rp',
                    'pose_graph': 'svo_ceres_pg',
                    'ba_estimate': 'svo_ceres_ba'}
kFnExt = '.txt'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Organize reuslts for analysis.''')
    parser.add_argument(
        'result_dir', type=str,
        help="Folder containing the groundtruth and the estimate.")
    parser.add_argument(
        '--save_dir', type=str,
        help="Folder to save, None to use result_dir", default=None)
    parser.add_argument(
        '--platform', type=str, help="platform folder", default='laptop')
    parser.add_argument(
        '--est_types', nargs="*", type=str,
        default=['traj_est', 'pose_graph'])
    parser.add_argument(
        '--mul_trials', type=int,
        help='number of trials, None for single run', default=None)
    args = parser.parse_args()

    if not args.save_dir:
        args.save_dir = os.path.realpath(args.result_dir) + "_organized"

    print("Going to organize results in {0}...".format(args.result_dir))

    top_save_dir = os.path.join(args.save_dir, args.platform)
    print("Will save into {0}.".format(top_save_dir))

    est_fn_base = []
    est_save_dir = []
    for et in args.est_types:
        est_fn_base.append(kNsToEstFnMapping[et])
        est_save_dir.append(os.path.join(top_save_dir, kNsToAlgoMapping[et]))
        es_d = est_save_dir[-1]
        if not os.path.exists(es_d):
            os.makedirs(es_d)
            print("Making directory {0}".format(es_d))

    print("Will get files for:")
    for et, etb, etd in zip(args.est_types, est_fn_base, est_save_dir):
        print("- {0}: {1} --> {2}".format(et, etb, etd))

    datasets = [v for v in os.listdir(args.result_dir)
                if os.path.isdir(os.path.join(args.result_dir, v))]
    datasets.sort()
    print("Found datasets {0}".format(datasets))

    n_trials = 1
    if args.mul_trials:
        n_trials = args.mul_trials
    print("Will copy {0} trials.".format(n_trials))

    for d in datasets:
        res_dir_i = os.path.join(args.result_dir, d)
        print(">>> Processing {0}...".format(res_dir_i))
        for et, etb, etd in zip(args.est_types, est_fn_base, est_save_dir):
            tar_dir_i = os.path.join(top_save_dir, etd,
                                     "_".join([args.platform,
                                               kNsToAlgoMapping[et], d]))
            if not os.path.exists(tar_dir_i):
                os.makedirs(tar_dir_i)

            gt_fn = os.path.join(res_dir_i, 'stamped_groundtruth.txt')
            assert os.path.exists(gt_fn), "{0} not found.".format(gt_fn)
            copy2(gt_fn, tar_dir_i)
            print("{0} ---> {1}".format(gt_fn, tar_dir_i))
            cfg_fn = os.path.join(res_dir_i, 'eval_cfg.yaml')
            if os.path.exists(cfg_fn):
                copy2(cfg_fn, tar_dir_i)
                print("{0} ---> {1}".format(cfg_fn, tar_dir_i))
            for trial_i in range(n_trials):
                suffix = str(trial_i)
                if trial_i == 0 and not args.mul_trials:
                    suffix = ''
                fn = os.path.join(res_dir_i, etb+suffix+kFnExt)
                if not os.path.exists(fn):
                    print("{0} not found.".format(fn))
                else:
                    copy2(fn, tar_dir_i)
                    print("{0} ---> {1}".format(fn, tar_dir_i))
