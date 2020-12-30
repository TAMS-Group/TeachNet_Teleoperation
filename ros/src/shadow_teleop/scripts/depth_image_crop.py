#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# File Name :depth_image_crop
# Purpose : crop dataset to 100*100
# Creation Date : 01-08-2018
# Created By : Shuang Li

import numpy as np
from PIL import Image


def main():
    """crop human hand images to 100*100"""
    base_path = "../data/"
    DataFile = open(base_path + "Human_label/Training_Annotation.txt", "r")

    lines = DataFile.read().splitlines()
    for line in lines:
        frame = line.split(' ')[0].replace("\t", "")
        label_source = line.split('\t')[1:]
        label = []
        label.append([float(l.replace(" ", "")) for l in label_source[0:63]])

        keypoints = np.array(label)
        # image path depends on the location of your training dataset
        h_img = Image.open(base_path + "training/" + str(frame))  # 640*320
        # h_img.show()
        keypoints = keypoints.reshape(21, 3)

        # camera center coordinates and focal length
        mat = np.array([[475.065948, 0, 315.944855], [0, 475.065857, 245.287079], [0, 0, 1]])
        uv = np.random.randn(21, 2)

        for i in range(0, len(keypoints)):
            uv[i] = ((1 / keypoints[i][2]) * mat @ keypoints[i])[0:2]

        # from IPython import embed;embed()
        # Image coordinates: origin at the top-left corner, u axis going right and v axis going down
        left = uv.min(axis=0)[0]
        top = uv.min(axis=0)[1]
        right = uv.max(axis=0)[0]
        bottom = uv.max(axis=0)[1]
        padding = 10
        h_img_ = h_img.crop((left, top, right, bottom))
        width, height = h_img_.size  # Get dimensions
        # print(width)
        # print(height)
        if height > width:
            left_padding = float(height - width)
            top_padding = 0
        else:
            left_padding = 0
            top_padding = float(width - height)
        # h_img_.show()
        h_img = h_img.crop((left - padding / 2 - left_padding / 2, top - padding / 2 - top_padding / 2,
                            right + padding / 2 + left_padding / 2, bottom + padding / 2 + top_padding / 2))
        print(frame)
        width, height = h_img.size  # Get dimensions
        print(width)
        print(height)
        # h_img.show()

        # resized to 100*100 pixels or you can choose not
        # if resize, must choose Image.NEAREST.
        h_img = h_img.resize((100, 100), resample=Image.NEAREST)

        # a = cv2.imread("../data/training/crop100/" + frame,cv2.IMREAD_ANYDEPTH)
        # print(a.shape)
        # print(a.dtype) #int16
        # img = transforms.ToPILImage()(h_img)
        # h_img.show()
        width, height = h_img.size  # Get dimensions
        print(width)
        print(height)

        # change the path of cropped images
        h_img.save("../data/" + frame)
        DataFile.close()


if __name__ == '__main__':
    main()
        
