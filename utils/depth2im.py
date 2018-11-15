#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# File Name : depth2im.py
# Creation Date : 05-07-2018
# Created By : Jeasine Ma [jeasinema[at]gmail[dot]com]

import sys 

import cv2
import numpy as np 

def main():
    f = sys.argv[1]
    img = cv2.imread(f, cv2.IMREAD_ANYDEPTH)
    # img[img == 700] = 0
    img = img.astype(np.float32)
    max = 60000
    min = np.min(img)

    img = np.clip((img - min)/(max - min)*255, a_min=0, a_max=255)
    img = img[..., np.newaxis]
    img = np.dstack([img, img, img])

    cv2.namedWindow('0')
    cv2.imshow('0', img)
    cv2.waitKey(-1)

if __name__ == '__main__':
    main()	
