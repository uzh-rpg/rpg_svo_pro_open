This document describes the benchmark tools and detailed instructions to use it.

1. [Overview](#overview)
2. [Dataset generation](#dataset-generation)
3. [Run a configuration on multiple datasets](#run-a-configuration-on-multiple-datasets)
4. [Analyze the Results](#analyze-the-results)


## Overview
The benchmark tool is designed to evaluate and compare the performance of different configurations on multiple datasets. Each configuration can be different parameters, sensor combination (stereo/mono), etc.

To use it, one needs to do the following

* Dataset generation
* Running one configuration / multiple configurations
* Analyze the results

## Dataset generation
Each dataset is organized in a self-contained folder.
We provide tools to generate the desired folder from a ROS bag file.
See detailed instructions in `dataset_tools/README.md`.

## Run a configuration on multiple datasets
To run a configuration on multiple datasets, one first needs to specify a configuration file and put all the datasets in the correct folder.

### Prepare a configuration file and datasets
Each configuration is described in a `yaml` file under `experiments` folder.
The structure of a configuration file is:
```
# meta information
experiment_label: 'svoceresstereo'
ros_node: svo_ros
ros_node_name: svo_benchmark_with_ceres
flags:
  v: 0
  logtostderr: 1

# parameters for the pipeline
settings:
......

# list of datasets to run with the settings above
# the name is the folder contains the dataset in the data folder
datasets:
  - name: euroc/stereo/MH_01
    settings:
      dataset_first_frame: 0
  - name: euroc/stereo/MH_04
    settings:
      dataset_first_frame: 0
......
```
You can find multiple examples under `experiments` folder already.

All the datasets are put in the `data` folder (you can put a soft link to your actual data folder). The `name` field of each dataset in the configuration file is the path to the dataset folder. For example, for the above configuration, the folder structure inside `data` will look like:
```
data
├── euroc
│   └── stereo
│       ├── MH_01
│       └── MH_04

```
Each dataset folder is what we get from using `dataset_tools` in the previous step. Monocular setup can also be run on a stereo dataset, and only the first camera will be used in that case.

### Run a configuration
After the preparation is done, you can run one configuration via
```
rosrun svo_benchmarking benchmark.py <config_name>.yaml
```
The results are written to `results/<time>_<config_name>`,
and the results for each dataset are put into a corresponding sub-folder.
For example, running the above configuration will generate a folder structure like:
```
results
├── 20180425_153857_euroc_stereo
│   ├── MH_01
│   └── MH_04
```
Note:
* There are also some configurations that allow to run each time several time (`--n_trials`), customize the evaluation configuraiton (`--align_type` and `--n_aligned`). Note that the default setting is to use `posyaw` for alignment, which may give lower accuracy then `se3` or `sim3`.
* There will be different estimate files in the folder, depending on the pipeline you executed. For example, if you execute the global map experiment, you will see the following output

  * `stamped_traj_estimate.txt`: real-time estimate after processing each frame

  * `stamped_ba_estimate.txt`: bundle adjusted poses from iSAM2 after everything is done
    Please take care to compare the same type of estimate using the same analysis configuration in your accuracy evaluation. You can use the script in `rpg_trajectory_evaluation` to change the evaluation configuration files easily

    ```sh
    rosrun rpg_trajectory_evaluation change_eval_cfg_recursive.py <folder> <align_type> <align_frame>
    ```

    

## Analyze the Results
### Trajectory accuracy
We recommend to use the [evaluation toolbox](https://github.com/uzh-rpg/rpg_trajectory_evaluation) to analyze the trajectory accuracy. We have already written a `eval_cfg.yaml` to the result folder, which will be used by the toolbox to determine the alignment parameters for evaluation (default: `posyaw` and all frames). 
#### Single trajectory accuracy
Basically, you can put the toolbox in your workspace and run (after compiling and sourcing)
```sh
rosrun rpg_trajectory_evaluation analyze_trajectory_single.py <result_folder>
```

#### Batch evaluation
We provide a script `scripts/organize_results_mul.py` to organize the results of mutiple estimation types, multiple trials into the folder structure that can be processed by `rpg_trajectory_evaluation` toolbox. For example
```sh
# in results folder
rosrun svo_ceres_benchmarking organize_results_mul.py ./<results> --est_types traj_est ba_estimate
```
will generates a folder structure like
```
└── laptop
    ├── svo_ceres_ba
    │   ├── laptop_svo_ceres_ba_MH_01
    │   ├── laptop_svo_ceres_ba_MH_02
 ....
    └── svo_ceres_rp
        ├── laptop_svo_ceres_rp_MH_01
        ├── laptop_svo_ceres_rp_MH_02
....
```
Then you can use the example analysis configration `analyze_traj_cfg/exp_gm_euroc_all.yaml` with `rpg_trajectory_evaluation` to analyze the accuracy of both bundle adjusted pose and real-time poses conveniently. Please see the documentation in [rpg_trajectory_evaluation](https://github.com/uzh-rpg/rpg_trajectory_evaluation/tree/dev) for details. There are also some other configurations that work for different experiment types (e.g., VIO only, pose graph).

### Other evaluation

Different evaluations can be done using the result using the script under `scripts`. See `evaluation_tools.md` for details.