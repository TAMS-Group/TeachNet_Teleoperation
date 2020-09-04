#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# Creation Date : 20-07-2018
# Created By : Shuang Li [sli[at]informatik.uni-hamburg.de]
# Last Modified : Sa 21 Jul 2018 22:17:56 CST

import numpy as np
import glob
import cv2
import csv
import os
from mayavi import mlab

def npy_index_joint_select():
    index = np.load("data/human_keypoint_index_test.npy")
    index.sort()
    joints_file = open("data/joints_file.csv", "r")
    lines = joints_file.read().splitlines()

    f_index = {}
    for ind, line in enumerate(lines):
        frame = line.split(',')[0]
        f_index[frame] = ind

    for i in index:
        try:
            line = lines[f_index[i]]
            print(i)
            label_source = line.split(',')[1:]
            label = []
            label.append([float(l) for l in label_source])
            label = np.array(label)
            np.save('data/output_shadow_joint/' + i[0:15] + '.npy', label)
        except:
            pass


def csv_index_joint_select():
    index_file = open("./joints.csv", "r")
    index = index_file.read().splitlines()
    j_index = []
    for line in index:
      j_index.append(line.split(',')[0])

    joints_file = open("./data/joints_file.csv", "r")
    lines = joints_file.read().splitlines()
    output = open("./data/teach_label.csv", "w")

    f_index = {}
    for ind, line in enumerate(lines):
        frame = line.split(',')[0]
        f_index[frame] = ind

    for i in j_index:
     try:
         line = lines[f_index[i]][20:]
         print(i)
         output.write(line + '\n')
     except:
         pass


def annotation_index_derive_xyz():
    # from human_keypoint_index derive xyz
    DataFile = open("./data/Human_label/Training_Annotation.txt", "r")
    lines = DataFile.read().splitlines()
    f_index = {}
    for ind, line in enumerate(lines):
         frame = line.split(' ')[0].replace("\t", "")
         f_index[frame] = ind

    output = open("./data/Human_label/annotation_all.csv", "w")
    a =np.load('./data/Human_label/human_keypoint_index.npy')
    for aa in a:
        try:
            # line = lines[f_index[aa[-19:]]]
            # print(f_index[aa[-19:]])
            print(lines[f_index[aa[-19:]]])
            line = lines[f_index[aa[-19:]]][20:]
            output.write(line + '\n')
        except:
            pass
    a = np.loadtxt(open("./data/Human_label/annotation_all.csv", "rb"),
                      dtype='S30', delimiter=",", skiprows=0)
    np.save('./data/Human_label/human_index_xyz.npy', a)


def filename2csv():
    f=open("./file.csv",'r+')
    w=csv.writer(f)
    for path, dirs, files in os.walk("./4/"):
        for filename in files:
            w.writerow([filename])


def readable():
    file_list = glob.glob(os.path.join('data/depth', '*.png'))
    # file_list = glob.glob(os.path.join('*.png'))
    file_number = len(file_list)
    #from IPython import embed;embed()
    for i in range(0, file_number):
        img = cv2.imread(file_list[i], cv2.IMREAD_ANYDEPTH)
        if img is None:
            continue
        n = cv2.normalize(img, img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        cv2.imwrite( file_list[i][:-4] + ".jpg", n)


def xyz2uvd():
    """ test code : convert world coordinates to uvd """
    base_path = "./data/"
    DataFile = open(base_path + "uv.csv", "r")
    lines = DataFile.read().splitlines()

    frame = lines[0].split(',')[0]
    print(frame)
    label_source = lines[0].split(',')[1:46]
    keypoints = np.array(label_source).reshape(15, 3)

    img = cv2.imread("./data/image_D00000001.png", cv2.IMREAD_ANYDEPTH)
    img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    print(img.shape)
    uv = np.random.randn(15, 2)
    for i in range(0, len(keypoints)):
        uv[i] = keypoints[i][0:2]
        print(uv[i])
        cv2.circle(img, (int(uv[i][0]), int(uv[i][1])), 5, (255, 255, 0), -1)
    cv2.imwrite("../data/uv0/1_.jpg", img)
    left = uv.min(axis=0)[0]
    top = uv.min(axis=0)[1]
    right = uv.max(axis=0)[0]
    bottom = uv.max(axis=0)[1]

    print(left)
    print(top)
    print(right)
    print(bottom)


def uvd2xyz():
    """ test code : convert uvd to world coordinates """
    base_path = "./data/"
    index = np.load('index.npy')
    uvd = np.load('uvd_test.npy')

    mat = np.array([[475.065948, 0, 315.944855], [0, 475.065857, 245.287079], [0, 0, 1]])

    joint_human = uvd.reshape(-1, 21, 3)

    for i in range(joint_human.shape[0]):
        key = (np.dot(np.linalg.inv(mat), joint_human[i].T)).T
        np.save('data/output_xyz/' + str(i) + '.npy', key)


def animation():
    """ visualization: show animation of shadow hand skeleton in mayavi """
    file_list = glob.glob(os.path.join('/homeL/demo/test/', '*.png'))
    file_list.sort()
    DataFile = open("/data/shadow_output.csv", "r")
    lines = DataFile.read().splitlines()
    pos = np.zeros([1,48])
    for ln in lines:
        label_source = ln.split(',')
        label = [float(ll) for ll in label_source]
        keypoints = np.array(label)
        pos = np.vstack([pos, keypoints])
        # print(keypoints)
        # from IPython import embed; embed()
    DataFile.close()

    pos = np.delete(pos, 0, 0)

    points = np.array(pos).reshape(-1, 16, 3)
    # print(points)

    p1 = mlab.points3d(points[0][:, 0], points[0][:, 1], points[0][:, 2], color=(1, 0, 0), scale_factor=0.009)

    plt11 = mlab.plot3d([points[0][0][0], points[0][11][0]], [points[0][0][1], points[0][11][1]], [points[0][0][2], points[0][11][2]], color=(0.3, 0.2, 0.7), tube_radius=0.003)
    plt6 = mlab.plot3d([points[0][6][0], points[0][11][0]], [points[0][6][1], points[0][11][1]], [points[0][6][2], points[0][11][2]], color=(0.3, 0.2, 0.7), tube_radius=0.003)
    plt1 = mlab.plot3d([points[0][6][0], points[0][1][0]], [points[0][6][1], points[0][1][1]], [points[0][6][2], points[0][1][2]], color=(0.3, 0.2, 0.7), tube_radius=0.003)

    plt12 = mlab.plot3d([points[0][0][0], points[0][12][0]], [points[0][0][1], points[0][12][1]], [points[0][0][2], points[0][12][2]], color=(0.5, 1, 1), tube_radius=0.003)
    plt7 = mlab.plot3d([points[0][7][0], points[0][12][0]], [points[0][7][1], points[0][12][1]], [points[0][7][2], points[0][12][2]], color=(0.5, 1, 1), tube_radius=0.003)
    plt2 = mlab.plot3d([points[0][7][0], points[0][2][0]], [points[0][7][1], points[0][2][1]], [points[0][7][2], points[0][2][2]], color=(0.5, 1, 1), tube_radius=0.003)

    plt13 = mlab.plot3d([points[0][0][0], points[0][13][0]], [points[0][0][1], points[0][13][1]], [points[0][0][2], points[0][13][2]], color=(1, 1, 0), tube_radius=0.003)
    plt8 = mlab.plot3d([points[0][8][0], points[0][13][0]], [points[0][8][1], points[0][13][1]], [points[0][8][2], points[0][13][2]], color=(1, 1, 0), tube_radius=0.003)
    plt3 = mlab.plot3d([points[0][8][0], points[0][3][0]], [points[0][8][1], points[0][3][1]], [points[0][8][2], points[0][3][2]], color=(1, 1, 0), tube_radius=0.003)

    plt14 = mlab.plot3d([points[0][0][0], points[0][14][0]], [points[0][0][1], points[0][14][1]], [points[0][0][2], points[0][14][2]], color=(0.8, 0, 0.9), tube_radius=0.003)
    plt9 = mlab.plot3d([points[0][9][0], points[0][14][0]], [points[0][9][1], points[0][14][1]], [points[0][9][2], points[0][14][2]], color=(0.8, 0, 0.9), tube_radius=0.003)
    plt4 = mlab.plot3d([points[0][9][0], points[0][4][0]], [points[0][9][1], points[0][4][1]], [points[0][9][2], points[0][4][2]], color=(0.8, 0, 0.9), tube_radius=0.003)

    plt15 = mlab.plot3d([points[0][0][0], points[0][15][0]], [points[0][0][1], points[0][15][1]], [points[0][0][2], points[0][15][2]], color=(0.1, 1, 0.8), tube_radius=0.003)
    plt10 = mlab.plot3d([points[0][10][0], points[0][15][0]], [points[0][10][1], points[0][15][1]], [points[0][10][2], points[0][15][2]], color=(0.1, 1, 0.8), tube_radius=0.003)
    plt5 = mlab.plot3d([points[0][10][0], points[0][5][0]], [points[0][10][1], points[0][5][1]], [points[0][10][2], points[0][5][2]], color=(0.1, 1, 0.8), tube_radius=0.003)

    @mlab.animate(delay=200)
    def anim():
            i = 0
            for p in points:
                img = cv2.imread(file_list[i],
                                 cv2.IMREAD_ANYDEPTH)
                i =i+1
                n = cv2.normalize(img, img, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                cv2.imshow("crop human hand", n)
                cv2.waitKey(40)

                print(p)
                print('Updating scene...')
                p1.mlab_source.reset(x=p[:, 0], y=p[:, 1], z=p[:, 2])
                plt11.mlab_source.reset(x=[p[0][0], p[11][0]], y=[p[0][1], p[11][1]], z=[p[0][2], p[11][2]])
                plt6.mlab_source.reset(x=[p[6][0], p[11][0]], y=[p[6][1], p[11][1]], z=[p[6][2], p[11][2]])
                plt1.mlab_source.reset(x=[p[6][0], p[1][0]], y=[p[6][1], p[1][1]], z=[p[6][2], p[1][2]])

                plt12.mlab_source.reset(x=[p[0][0], p[12][0]], y=[p[0][1], p[12][1]], z=[p[0][2], p[12][2]])
                plt7.mlab_source.reset(x=[p[7][0], p[12][0]], y=[p[7][1], p[12][1]], z=[p[7][2], p[12][2]])
                plt2.mlab_source.reset(x=[p[7][0], p[2][0]], y=[p[7][1], p[2][1]], z=[p[7][2], p[2][2]])

                plt13.mlab_source.reset(x=[p[0][0], p[13][0]], y=[p[0][1], p[13][1]], z=[p[0][2], p[13][2]])
                plt8.mlab_source.reset(x=[p[8][0], p[13][0]], y=[p[8][1], p[13][1]], z=[p[8][2], p[13][2]])
                plt3.mlab_source.reset(x=[p[8][0], p[3][0]], y=[p[8][1], p[3][1]], z=[p[8][2], p[3][2]])

                plt14.mlab_source.reset(x=[p[0][0], p[14][0]], y=[p[0][1], p[14][1]], z=[p[0][2], p[14][2]])
                plt9.mlab_source.reset(x=[p[9][0], p[14][0]], y=[p[9][1], p[14][1]], z=[p[9][2], p[14][2]])
                plt4.mlab_source.reset(x=[p[9][0], p[4][0]], y=[p[9][1], p[4][1]], z=[p[9][2], p[4][2]])

                plt15.mlab_source.reset(x=[p[0][0], p[15][0]], y=[p[0][1], p[15][1]], z=[p[0][2], p[15][2]])
                plt10.mlab_source.reset(x=[p[10][0], p[15][0]], y=[p[10][1], p[15][1]], z=[p[10][2], p[15][2]])
                plt5.mlab_source.reset(x=[p[10][0], p[5][0]], y=[p[10][1], p[5][1]], z=[p[10][2], p[5][2]])
                yield
                # from IPython import embed;
                # embed()
    anim()
    mlab.show()


if __name__ == '__main__':
    readable()
