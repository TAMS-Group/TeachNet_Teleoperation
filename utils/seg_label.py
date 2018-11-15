#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# File Name : seg_depth.py
# Purpose :
# Creation Date : 19-07-2018
# Last Modified : 2018年07月29日 星期日 15时06分00秒
# Created By : Jeasine Ma [jeasinema[at]gmail[dot]com]

import cv2
import glob
import pickle
import numpy as np
import multiprocessing as mp
from shutil import copyfile

from utils import seg_hand_depth

##
# This is for crop shadow uv label
##

def func(f):
    img = cv2.imread(f[0], cv2.IMREAD_ANYDEPTH).astype(np.float32)
    output, label, _ = seg_hand_depth(img, label=f[1])
    cv2.imwrite(f[2], output)
    np.save('{}.npy'.format(f[2][:-4]), label)
    # copyfile(f[0], f[2])
    print(f[2])


def main():
    cores = mp.cpu_count()
    pool = mp.Pool(processes=cores) 
    
    pkls = glob.glob('./data/*.pkl')
    pkls.sort()
    for ind, pkl in enumerate(pkls):
        f = pickle.load(open(pkl, 'rb'))
        fl = []
        for j in f:
            name = j[0]
            label = j[1]
            fl.append([
                './data/origin/shadow{}/{}'.format(ind, name),
                label,
                './data/crop/shadow{}/{}'.format(ind, name),
            ])
        
        pool.map(func, fl)


if __name__ == '__main__':
    main()
