#!/usr/bin/env python2
"""
@author: Christian Forster
"""

import os
import argparse
import yaml

from colorama import init, Fore

init(autoreset=True)


def run_evaluation_script(trace_dir, script):
    command = 'rosrun ' + script['ros_node'] + ' ' + script['ros_executable']
    # position parameters
    if 'params' in script:
        for param in script['params']:
            if param == 'trace_dir':
                command += " " + trace_dir
            else:
                command += " " + param
    # flags
    if 'flags' in script:
        for key, option in script['flags'].items():
            command += ' --'+str(key)+'='+str(option)
    print(Fore.RED+'==> Staring: '+command)
    os.system(command)
    print(Fore.GREEN+'<== Finished')


def evaluate_dataset(trace_dir):
    if os.path.exists(os.path.join(trace_dir, 'evaluation_scripts.yaml')):
        eval_scripts = yaml.load(
            open(os.path.join(trace_dir, 'evaluation_scripts.yaml'), 'r'))
        for script in eval_scripts:
            run_evaluation_script(trace_dir, script)
    else:
        print('Folder "'+trace_dir +
              '" does not contain a file "evaluation_scripts.yaml".')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''
    Evaluates tracefiles of SVO and generates the plots.
    The experiment folder should contain an "evaluation_scripts.yaml" that 
    specifies scripts to run.
    ''')
    parser.add_argument(
        'experiment_dir', help='directory name of the tracefiles')
    args = parser.parse_args()

    evaluate_dataset(args.experiment_dir)
