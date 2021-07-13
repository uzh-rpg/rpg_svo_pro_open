#!/usr/bin/python2
"""
@author: Jonathan Huber

@brief: combine two image lists of stereo pair to format, which can be used
by benchmark_node_with_ceres. Will append each image in images_1.txt to 
corresponding line in images_0.txt
"""

f = open('images_1.txt', "r")
lines = f.readlines()
images_1 = []
timestamps_1 = []
for x in lines:
    images_1.append(x.split(' ')[2])
    timestamps_1.append(x.split(' ')[1])
f.close()
    
file_lines = []
with open('images_0.txt', 'r') as f:
    i = 0
    for x in f.readlines():
        timestamp_0 = x.split(' ')[1]
        while(timestamps_1[i] < timestamp_0):
            i = i+1
            if(i == len(timestamps_1)):
                break
        if(i == len(timestamps_1)):
            break
        if timestamps_1[i] == timestamp_0:
            file_lines.append(''.join([x.strip(), ' ', images_1[i], '\n']))
            i = i+1
            if(i == len(timestamps_1)):
                break
        elif timestamps_1[i] > timestamp_0:
            continue


with open('images_combined.txt', 'w') as f:
    f.writelines(file_lines)
    
print('Created file images_combined.txt')
