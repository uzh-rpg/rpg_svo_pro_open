#!/usr/bin/python

import argparse
import numpy as np

def extract(gt, out_filename):
    n = 0
    fout = open(out_filename, 'w')
    fout.write('# id timestamp tx ty tz qx qy qz qz\n')    

    with open(gt,'rb') as fin:
        data = np.genfromtxt(fin, delimiter=",")
        for l in data:
            fout.write('%d %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f\n' % 
                      (n, l[0]/1e9, l[1], l[2], l[3], l[5], l[6], l[7], l[4]))
            n += 1
    print('wrote ' + str(n) + ' poses to the file: ' + out_filename)
    fout.close()

          
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''
    Extracts IMU messages from a csv file in ASL format.
    ''')
    parser.add_argument('gt', help='Ground truth csv')
    parser.add_argument('--output', default='groundtruth.txt')
    args = parser.parse_args()    
    out_filename = args.output
    print('Extract ground truth pose from file '+args.gt)
    print('Saving to file '+out_filename)
    extract(args.gt, out_filename)
