#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# File Name : eval.py
# Creation Date : 30-07-2018
# Created By : Jeasine Ma [jeasinema[at]gmail[dot]com]

import sys

import pickle
import numpy as np
import matplotlib.pyplot as plt

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
    print('load predict:{}, label:{}'.format(sys.argv[1], sys.argv[2]))
    label = pickle.load(open(sys.argv[1], 'rb'))
    predict = pickle.load(open(sys.argv[2], 'rb'))
    predict_2 = pickle.load(open(sys.argv[3], 'rb'))
    predict_3 = pickle.load(open(sys.argv[4], 'rb'))

    index = {}
    for ind, i in enumerate(label):
        index[i[0]] = ind

    res, res2, res3 = [], [], []
    gt, gt2, gt3 = [], [], []
    for i in predict:
        try:
            gt.append(label[index[i[0]]][1][:, :2].reshape(-1))
            res.append(i[1][:, :2].reshape(-1))
        except:
            pass
    for i in predict_2:
        try:
            gt2.append(label[index[i[0]]][1][:, :2].reshape(-1))
            res2.append(i[1][:, :2].reshape(-1))
        except:
            pass
    for i in predict_3:
        try:
            gt3.append(label[index[i[0]]][1][:, :2].reshape(-1))
            res3.append(i[1][:, :2].reshape(-1))
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
    for thresh in np.arange(0., 20., 0.25):
        acc.append(cal_acc(res, gt, thresh, 'avg'))
        acc2.append(cal_acc(res2, gt2, thresh, 'avg'))
        acc3.append(cal_acc(res3, gt3, thresh, 'avg'))
        threshold.append(thresh)
    fig = plt.figure(1)
    fig.set_size_inches(5, 10)
    plt.subplot(211)
    plt.xlim(0, 20)
    plt.ylim(0, 1)
    plt.xticks(np.arange(0., 20.1, 2.))
    plt.yticks(np.arange(0., 1.01, 0.1))
    plt.plot(threshold, acc, 'r-')
    plt.plot(threshold, acc2, 'g-')
    plt.plot(threshold, acc3, 'b-')
    plt.grid(True)
    # plt.title('Avg')
    plt.xlabel('Average Distance Threshold/mm')
    plt.ylabel('acc')

    acc = []
    acc2 = []
    acc3 = []
    threshold = []
    for thresh in np.arange(0., 80., 0.1):
        acc.append(cal_acc(res, gt, thresh, 'max'))
        acc2.append(cal_acc(res2, gt2, thresh, 'max'))
        acc3.append(cal_acc(res3, gt3, thresh, 'max'))
        threshold.append(thresh)
    plt.subplot(212)
    plt.xlim(0, 80)
    plt.ylim(0, 1)
    plt.xticks(np.arange(0., 80.1, 10.))
    plt.yticks(np.arange(0., 1.01, 0.1))
    plt.plot(threshold, acc, 'r-')
    plt.plot(threshold, acc2, 'g-')
    plt.plot(threshold, acc3, 'b-')
    plt.grid(True)
    # plt.title('Max')
    plt.xlabel('Maxium Distance Threshold/mm')
    plt.ylabel('acc')
    # plt.show()
    fig.savefig('1.pdf')


if __name__ == '__main__':
    main()
