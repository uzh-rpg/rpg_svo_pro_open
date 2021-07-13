#!/bin/bash

rosrun rpg_trajectory_evaluation analyze_trajectory_single.py ./MH_01 --recalculate_errors --mul_trials 5 --est_types traj_est pose_graph
rosrun rpg_trajectory_evaluation analyze_trajectory_single.py ./MH_02 --recalculate_errors --mul_trials 5 --est_types traj_est pose_graph
rosrun rpg_trajectory_evaluation analyze_trajectory_single.py ./MH_03 --recalculate_errors --mul_trials 5 --est_types traj_est pose_graph
rosrun rpg_trajectory_evaluation analyze_trajectory_single.py ./MH_04 --recalculate_errors --mul_trials 5 --est_types traj_est pose_graph
rosrun rpg_trajectory_evaluation analyze_trajectory_single.py ./MH_05 --recalculate_errors --mul_trials 5 --est_types traj_est pose_graph


rosrun rpg_trajectory_evaluation analyze_trajectory_single.py ./V1_01 --recalculate_errors --mul_trials 5 --est_types traj_est pose_graph
rosrun rpg_trajectory_evaluation analyze_trajectory_single.py ./V1_02 --recalculate_errors --mul_trials 5 --est_types traj_est pose_graph
rosrun rpg_trajectory_evaluation analyze_trajectory_single.py ./V1_03 --recalculate_errors --mul_trials 5 --est_types traj_est pose_graph

rosrun rpg_trajectory_evaluation analyze_trajectory_single.py ./V2_01 --recalculate_errors --mul_trials 5 --est_types traj_est pose_graph
rosrun rpg_trajectory_evaluation analyze_trajectory_single.py ./V2_02 --recalculate_errors --mul_trials 5 --est_types traj_est pose_graph
rosrun rpg_trajectory_evaluation analyze_trajectory_single.py ./V2_03 --recalculate_errors --mul_trials 5 --est_types traj_est pose_graph
