#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# File Name : shadow cartesian2uvd
# Purpose : eval shadow catesian position and corresponding uvd in images acquired from gazebo.
# Creation Date : 01-08-2018
# Created By : Shuang Li

import numpy as np
import math
import csv
from numpy.linalg import inv
from pyquaternion import Quaternion

def main():
    base_path = "../data/paper_img/"
    DataFile = open(base_path + "output.csv", "r")
    lines = DataFile.read().splitlines()

    # gazebo camera center coordinates and focal length
    mat = np.array([[554.255, 0, 320.5], [0, 554.25, 240.5], [0, 0, 1]])
    uv = np.zeros([15, 2])

    for ln in lines:
        frame = ln.split(',')[0]
        print(frame)
        label_source = ln.split(',')[1:46]
        keypoints = np.array([float(ll) for ll in label_source]).reshape(15, 3)

        camera_theta = np.array([[0, 0, 1.57], [0, 0.52, 0.84001], [0, 0.52, 2.4],
                                [0, -0.52, 0.872664626],[0, -0.52, 2.2689280276],
                                [0, 0, 1.04721975512],[0, 0, 2.0943951024], [0, 0.6, 1.57],
                                [0, -0.52, 1.57]])
        # orientation: w, x,y,z in /gazebo/model_state
        camera_qt = np.array([[0.707388269167, 0, 0, 0.706825181105], [0.882398030774, -0.104828455998, 0.234736884563, 0.394060027311],
                              [0.350178902426, -0.239609122606, 0.0931551315033, 0.900713231908],
                              [0.875846686123, 0.108646979538, -0.232994085758, 0.408414216507],
                              [0.408413188959, 0.232994213223, -0.108646706188, 0.875847165277],
                              [0.866019791529, 0, 0, 0.500009720586],
                              [0.499997879273, 0, 0, 0.866026628184],
                              [0.675793825515, -0.208881123594, 0.209047527494, 0.675255886943],
                              [0.683612933973, 0.18171100765, -0.18185576664, 0.683068771313],
                              ])
        camera_tran = np.array([[0, -0.5, 0.35], [-0.35, -0.3, 0.65], [0.35, -0.3, 0.65],
                                [-0.3, -0.38, 0.11], [0.3, -0.35, 0.12],[-0.25, -0.4330127, 0.35],
                                [0.25, -0.4330127, 0.35], [0, -0.3, 0.5], [0, -0.4, 0.1]])

        for i in range(0, 9):
            csvSum = open(base_path + "uv46.csv", "a")
            writer = csv.writer(csvSum)
            w, x, y, z = camera_qt[i]
            quat = Quaternion(w, x, y, z)

            R = quat.rotation_matrix
            R = R.T
            camera_t = camera_tran[i]

            result = [frame]
            for j in range(0, len(keypoints)):
                key = keypoints[j] - camera_t
                cam_world = np.dot(R, key)
                cam_world = np.array([-cam_world[1], -cam_world[2], cam_world[0]])
                uv[j] = ((1 / cam_world[2]) * np.dot(mat, cam_world))[0:2]
                result.append(uv[j][0])
                result.append(uv[j][1])
                result.append(cam_world[2]*1000)

            writer.writerow(result)


if __name__ == '__main__':
    main()
