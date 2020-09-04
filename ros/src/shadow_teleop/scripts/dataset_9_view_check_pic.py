#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# File Name : check pic py
# Purpose :
# Creation Date : 01-08-2018
# Created By : Hongzhuo Liang

import cv2
import numpy as np
import glob
import os
from IPython import embed

path = "../data/depth_shadow"

file_list = glob.glob(os.path.join(path + '1', '*.png'))
file_number = len(file_list)
file_list.sort()
# selected_file = file_list[np.random.randint(file_number)][-20:]
# selected_file = '/image_D00000001.png'

for j in range(file_number):
    selected_file = file_list[j][-20:]

    a = []
    for i in range(1,10):
            path_tmp = path + str(i)
            img = cv2.imread(path_tmp + selected_file, cv2.IMREAD_ANYDEPTH)
            # embed()
            img = img[50:img.shape[0] - 50, 120:img.shape[1]-90]
            # img = cv2.resize(img, (200, 200))
            a.append(img)

    m1 = np.hstack([a[1], a[7]])
    m1 = np.hstack([m1, a[2]])
    m2 = np.hstack([a[5], a[0]])
    m2 = np.hstack([m2, a[6]])
    m3 = np.hstack([a[3], a[8]])
    m3 = np.hstack([m3, a[4]])

    m = np.vstack([m1, m2])
    m = np.vstack([m, m3])
    n = cv2.normalize(m, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    cv2.imshow(selected_file, n)
    key = cv2.waitKey()

    if key == 27:    # Esc key to stop
        break
    elif key==ord('a'):  # normally -1 returned,so don't print it
        cv2.destroyAllWindows()
        continue
