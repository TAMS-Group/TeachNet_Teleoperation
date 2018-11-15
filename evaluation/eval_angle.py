#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# File Name : eval.py
# Creation Date : 30-07-2018
# Created By : Jeasine Ma [jeasinema[at]gmail[dot]com]

import sys

import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def cal_acc(res, gt, threshold, strategy='avg'): # max
    # predict: (N, X)
    # label: (N, X)
    if strategy == 'avg':
        return np.sum(np.mean(np.abs(res - gt), axis=-1) < threshold)/len(res)
    elif strategy == 'max':
        return np.sum(np.max(np.abs(res - gt), axis=-1) < threshold)/len(res)
    else:
        raise NotImplementedError


def main():
    print('load predict:{}, label:{}'.format(sys.argv[1], sys.argv[1:]))
    label = pd.read_csv(sys.argv[1]).values
    predict = pd.read_csv(sys.argv[2]).values
    predict_2 = pd.read_csv(sys.argv[3]).values
    predict_3 = pd.read_csv(sys.argv[4]).values

    index = {}
    for ind, i in enumerate(label):
        index[i[0]] = ind

    res, res2, res3 = [], [], []
    gt, gt2, gt3 = [], [], []
    for i in predict:
        try:
            gt.append(label[index[i[0]]][1:].reshape(-1).astype(np.float32))
            res.append(i[1:].reshape(-1).astype(np.float32))
        except:
            pass
    for i in predict_2:
        try:
            gt2.append(label[index[i[0]]][1:].reshape(-1).astype(np.float32))
            res2.append(i[1:].reshape(-1).astype(np.float32))
        except:
            pass
    for i in predict_3:
        try:
            gt3.append(label[index[i[0]]][1:].reshape(-1).astype(np.float32))
            res3.append(i[1:].reshape(-1).astype(np.float32))
        except:
            pass

    res = np.array(res)
    res2 = np.array(res2)
    res3 = np.array(res3)
    gt = np.array(gt)
    gt2 = np.array(gt2)
    gt3 = np.array(gt3)
    assert(res.shape == gt.shape == res2.shape == gt.shape)
    print('total: {} samples'.format(len(gt)))

    acc = []
    acc2 = []
    acc3 = []
    threshold = []
    for thresh in np.arange(0., 0.5, 0.0025):
        acc.append(cal_acc(res, gt, thresh, 'avg'))
        acc2.append(cal_acc(res2, gt2, thresh, 'avg'))
        acc3.append(cal_acc(res3, gt3, thresh, 'avg'))
        threshold.append(thresh)
    fig = plt.figure(1)
    fig.set_size_inches(5, 10)
    plt.subplot(211)
    plt.xlim(0, 0.5)
    plt.ylim(0, 1)
    plt.xticks(np.arange(0., 0.51, 0.05))
    plt.yticks(np.arange(0., 1.01, 0.1))
    plt.plot(threshold, acc, 'r-', label='Single Branch')
    plt.plot(threshold, acc2, 'g-', label='Double Branch')
    plt.plot(threshold, acc3, 'b-', label='Keypoint IK')
    plt.grid(True)
    # plt.title('Avg')
    plt.xlabel('Average Angle Threshold/rad')
    plt.ylabel('acc')
    plt.legend()

    acc = []
    acc2 = []
    acc3 = []
    threshold = []
    for thresh in np.arange(0., 2.0, 0.01):
        acc.append(cal_acc(res, gt, thresh, 'max'))
        acc2.append(cal_acc(res2, gt2, thresh, 'max'))
        acc3.append(cal_acc(res3, gt3, thresh, 'max'))
        threshold.append(thresh)
    plt.subplot(212)
    plt.xlim(0, 2.0)
    plt.ylim(0, 1)
    plt.xticks(np.arange(0., 2.01, 0.2))
    plt.yticks(np.arange(0., 1.01, 0.1))
    plt.plot(threshold, acc, 'r-', label='Single Branch')
    plt.plot(threshold, acc2, 'g-', label='Double Branch')
    plt.plot(threshold, acc3, 'b-', label='Keypoint IK')
    plt.grid(True)
    # plt.title('Max')
    plt.xlabel('Maxium Angle Threshold/rad')
    plt.ylabel('acc')
    plt.legend()
    # plt.show()
    fig.savefig('1.pdf')


if __name__ == '__main__':
    main()
