#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# File Name : seg_depth.py
# Purpose :
# Creation Date : 19-07-2018
# Last Modified : 2018年08月02日 星期四 06时37分13秒
# Created By : Jeasine Ma [jeasinema[at]gmail[dot]com]

import cv2
import glob
import numpy as np
import multiprocessing as mp
from shutil import copyfile

from utils import seg_hand_depth

##
# This is for crop human and shadow image into 100*100 (normalized to [0,255))
##

def func(f):
    img = cv2.imread(f[0], cv2.IMREAD_ANYDEPTH).astype(np.float32)
    output = seg_hand_depth(img)
    cv2.imwrite(f[1], output)
    # copyfile(f[0], f[2])
    print(f[1])


def func2(f):
    copyfile(f[0], f[1])
    img = cv2.imread(f[1], cv2.IMREAD_ANYDEPTH).astype(np.float32)
    if np.max(img) == np.min(img) == 0:
        print(f[1], ' bad!')
        return
    try:
        output = seg_hand_depth(img, 500, 1000, 10, 100, 4, 4, 250, True, 300)
        cv2.imwrite(f[1], output)
    except:
        print(f[1])
    # print(f[1])


def main():
    # crop shadow depth images
    # f = np.load('./data/joint_all.npy')[..., 0]
    # fl = []
    # for i in range(9):
    #     for j in f:
    #         fl.append([
    #             '/home1/Dataset/shadow_depth/train/depth_shadow{}/{}'.format(i, j),
    #             './data/crop/shadow{}/{}'.format(i, j),
    #             # './data/origin/shadow{}/{}'.format(i, j),
    #         ])
    # crop human depth images
    cores = mp.cpu_count()
    pool = mp.Pool(processes=cores) 
    # pool.map(func, fl)
    f = np.load('./data/joint_all.npy')[..., 0]
    fl = []
    for j in f:
        fl.append([
            '/home1/Dataset/shadow_depth/train/human_crop/{}'.format(j),
            './data/human/{}'.format(j),
        ])
    
    pool.map(func2, fl)


if __name__ == '__main__':
    main()
