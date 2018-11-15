#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# File Name : seg_depth.py
# Purpose :
# Creation Date : 19-07-2018
# Last Modified : 2018年08月07日 星期二 04时06分51秒
# Created By : Jeasine Ma [jeasinema[at]gmail[dot]com]

import cv2
import glob
import pickle
import numpy as np
import multiprocessing as mp
from shutil import copyfile

from utils import seg_hand_depth, seg_hand_with_label

##
# This is for crop human depth, used by keypoint based method
##

def func2(f):
    copyfile(f[0], f[1])
    img = cv2.imread(f[1], cv2.IMREAD_ANYDEPTH).astype(np.float32)
    if np.max(img) == np.min(img) == 0:
        print(f[1], ' bad!')
        return
    try:
        output, label, crop_data1 = seg_hand_with_label(img, f[2])
        output, label, crop_data2 = seg_hand_depth(output, 500, 1000, 10, 100, 4, 4, 250, True, 300, label=label)
        cv2.imwrite(f[1], output)
        np.save(f[1][:-4]+'.npy', label)
        np.save(f[1][:-4]+'_crop1.npy', crop_data1)
        np.save(f[1][:-4]+'_crop2.npy', crop_data2)
    except:
        print(f[1])
    # print(f[1])


def main():
    cores = mp.cpu_count()
    pool = mp.Pool(processes=cores) 
    # pool.map(func, fl)
    f = pickle.load(open('./data/human_uvd.pkl', 'rb'))
    fl = []
    for j in f:
        name = j[0]
        label = j[1]
        fl.append([
            '/home1/lishuang/sli/handdata/training/images/{}'.format(name),
            './data/human_for_keypoint/{}'.format(name),
            label,
        ])
    
    pool.map(func2, fl)


if __name__ == '__main__':
    main()
