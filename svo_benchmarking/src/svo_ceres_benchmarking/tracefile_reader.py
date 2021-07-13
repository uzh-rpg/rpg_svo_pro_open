#!/usr/bin/python
"""
@author: Christian Forster
"""

import os
import csv
import numpy as np


def read(trace_dir, trace_name):
    # read tracefile
    data = csv.reader(open(os.path.join(trace_dir, trace_name)))
    fields = data.next()
    D = dict()
    for field in fields:
        D[field] = list()

    # fill dictionary with column values
    for row in data:
        for (field, value) in zip(fields, row):
            D[field].append(float(value))

    # change dictionary values from list to numpy array for easier manipulation
    for field, value in D.items():
        D[field] = np.array(D[field])

    return D
