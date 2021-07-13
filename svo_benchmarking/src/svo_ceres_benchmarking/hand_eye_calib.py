#!/usr/bin/env python2

import numpy as np
import transformations as tf

import tfs_utils as tu


def hand_eye_calib(q_gt, q_es, p_gt, p_es, I, delta=10, verbose=True):
    """Implementation of the least squares solution described in the paper:
    Robot Sensor Calibration: Solving AX=XB on the Euclidean Group
    by Frank C. Park and Bryan J. Martin
   """
    n = np.shape(I)[0]
    M = np.zeros([3, 3])
    C = np.zeros([3*n, 3])
    b_A = np.zeros([3*n, 1])
    b_B = np.zeros([3*n, 1])
    for ix, i in enumerate(I):
        A1 = tu.quat2dcm(q_es[i, :])
        A2 = tu.quat2dcm(q_es[i+delta, :])
        A = np.dot(A1.transpose(), A2)
        B1 = tu.quat2dcm(q_gt[i, :])
        B2 = tu.quat2dcm(q_gt[i+delta, :])
        B = np.dot(B1.transpose(), B2)
        alpha = tf.logmap_so3(A)
        beta = tf.logmap_so3(B)
        M = M + np.dot(np.matrix(beta).transpose(), np.matrix(alpha))
        C[3*ix:3*ix+3, :] = np.eye(3) - A
        b_A[3*ix:3*ix+3, 0] = np.dot(np.transpose(A1),
                                     p_es[i+delta, :]-p_es[i, :])
        b_B[3*ix:3*ix+3, 0] = np.dot(np.transpose(B1),
                                     p_gt[i+delta, :]-p_gt[i, :])
    # compute rotation
    D, V = np.linalg.linalg.eig(np.dot(M.transpose(), M))
    Lambda = np.diag([np.sqrt(1.0/D[0]), np.sqrt(1.0/D[1]), np.sqrt(1.0/D[2])])
    Vinv = np.linalg.linalg.inv(V)
    X = np.dot(V, np.dot(Lambda, np.dot(Vinv, M.transpose())))
    # compute translation
    d = np.zeros([3*n, 1])
    for i in range(n):
        d[3*i:3*i+3, :] = b_A[3*i:3*i+3, :] - np.dot(X, b_B[3*i:3*i+3, :])
    b = np.dot(np.linalg.inv(np.dot(np.transpose(C), C)),
               np.dot(np.transpose(C), d))
    return np.array(X), b
