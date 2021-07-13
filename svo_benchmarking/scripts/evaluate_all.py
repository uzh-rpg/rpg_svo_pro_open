#!/usr/bin/env python2

import argparse
import os

from evaluate import evaluate_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''
    run the evaluation for all subfolders
    ''')
    parser.add_argument(
        'top_dir', help='directory name of the tracefiles')
    args = parser.parse_args()

    subfolders = [v for v in os.listdir(args.top_dir) 
                  if os.path.join(args.top_dir, v)]
    print("Going to evaluate the following folders:{0}".format(subfolders))
    subfolders = [os.path.join(args.top_dir, v) for v in subfolders]

    for s in subfolders:
        if not os.path.exists(os.path.join(s, 'evaluation_scripts.yaml')):
            print("skip {0}".format(s))
            continue
        evaluate_dataset(s)
