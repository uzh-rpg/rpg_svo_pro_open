#!/usr/bin/env python2

import os
import argparse
import subprocess


kNsToEstFnMapping = {'traj_est': 'stamped_traj_estimate',
                     'pose_graph': 'stamped_pose_graph_estimate'}
kNsToAlgoMapping = {'traj_est': 'svo_ceres_lc',
                    'pose_graph': 'svo_ceres_lc_pg'}
# kNsToAlgoMapping = {'traj_est': 'svo_ceres_nolc'}
kFnExt = '.txt'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Organize reuslts for analysis.''')
    parser.add_argument(
        'result_dir', type=str,
        help="Folder containing the groundtruth and the estimate.")
    args = parser.parse_args()

    abs_res_dir = os.path.abspath(args.result_dir)
    assert os.path.exists(abs_res_dir), "Result folder does not exist."

    res_dirs = [v for v in os.listdir(abs_res_dir)
                if os.path.isdir(os.path.join(abs_res_dir, v))]
    res_dirs.sort()
    print("Found directories {0}".format(res_dirs))

    for d in res_dirs:
        print("Processing {0}...".format(d))
        abs_dir = os.path.join(abs_res_dir, d)
        subprocess.call(
            ['rosrun', 'rpg_trajectory_evaluation',
             'stamp_state_est_using_matches.py', 'traj_estimate.txt'],
            cwd=abs_dir)
        subprocess.call(
            ['rosrun', 'rpg_trajectory_evaluation', 'strip_gt_id.py',
             'groundtruth.txt'], cwd=abs_dir)
        subprocess.call(
            ['mv', 'stamped_traj_estimate.txt', 'stamped_traj_estimate0.txt'],
            cwd=abs_dir)

