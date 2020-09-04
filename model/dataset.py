#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# File Name : dataset.py
# Purpose :
# Creation Date : 23-06-2018
# Created By : Jeasine Ma [jeasinema[at]gmail[dot]com]

import os
import glob
import pickle

import cv2
import torch
import torch.utils.data
import torch.nn as nn
import torchvision.transforms as trans
import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import *

class ShadowPairedDataset(torch.utils.data.Dataset):
    def __init__(self, path, input_size, input_viewpoint, is_train=False, with_name=False):
        self.input_size = input_size
        self.input_viewpoint = np.array(input_viewpoint)
        self.path = path
        self.with_name = with_name
        self.is_train = is_train

        if is_train:
            self.label = np.load(os.path.join(path, 'joint_train.npy'))
        else:
            # self.label = np.load(os.path.join(path, 'joint_test.npy'))
            self.label = np.load(os.path.join(path, 'test.npy'))


        self.length = len(self.label)

    def __getitem__(self, index):
        tag = self.label[index]
        fname = tag[0]
        # critical
        target = tag[1:].astype(np.float32)[-22:]

        human = cv2.imread(os.path.join(self.path, 'human', fname), cv2.IMREAD_ANYDEPTH).astype(np.float32)
        viewpoint = self.input_viewpoint[np.random.choice(len(self.input_viewpoint), 1)][0]
        shadow = cv2.imread(os.path.join(self.path, 'crop', 'shadow{}'.format(viewpoint), fname), 
                            cv2.IMREAD_ANYDEPTH).astype(np.float32)
        assert(human.shape[0] == human.shape[1] == self.input_size) 
        assert(shadow.shape[0] == shadow.shape[1] == self.input_size) 

        if self.is_train:
            # Augmented(if train)
            # 1. random rotated
            angle = np.random.randint(-180, 180)
            M = cv2.getRotationMatrix2D(((self.input_size-1)/2.0, (self.input_size-1)/2.0), angle, 1)
            human = cv2.warpAffine(human, M, (self.input_size, self.input_size))
            # shadow = cv2.warpAffine(shadow, M, (self.input_size, self.input_size))

            # 2. jittering
            min_human = np.min(human[human != 255.])
            max_human = np.max(human[human != 255.])
            delta = np.random.rand()*(255. - max_human + min_human) - min_human
            human[human != 255.] += delta
            human = human.clip(max=255., min=0.)

            # min_shadow = np.min(shadow[shadow != 255.])
            # max_shadow = np.max(shadow[shadow != 255.])
            # delta = np.random.rand()*(255. - max_shadow + min_shadow) - min_shadow
            # shadow[shadow != 255.] += delta
            # shadow = shadow.clip(max=255., min=0.)

        # Normalized
        human = human / 255. * 2. - 1
        shadow = shadow / 255. * 2. - 1

        human = human[np.newaxis, ...]
        shadow = shadow[np.newaxis, ...]

        if self.with_name:
            return shadow, human, target, fname
        else:
            return shadow, human, target

    def __len__(self):
        return self.length


if __name__ == '__main__':
    pass	
