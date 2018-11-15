#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# File Name : model.py
# Purpose :
# Creation Date : 25-06-2018
# Created By : Jeasine Ma [jeasinema[at]gmail[dot]com]

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
import numpy as np


class Resnet18Conv(nn.Module):
    def __init__(self, input_chann=3, pretrained=True):
        super(Resnet18Conv, self).__init__()
        self.resnet = resnet18(pretrained=pretrained)
        self.resnet.conv1 = nn.Conv2d(input_chann, 64, kernel_size=7, stride=2, padding=3,bias=False)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.resnet.avgpool(x)

        return x


class BaseDeepConv(nn.Module):
    def __init__(self, input_chann=3):
        super(BaseDeepConv, self).__init__()
        # /2
        self.conv0 = nn.Conv2d(input_chann, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn0 = nn.BatchNorm2d(64)
        self.pool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # /2
        self.conv1 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 1, stride=2, bias=False)
        self.bn4 = nn.BatchNorm2d(128)

        # /3
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 256, 1, stride=2, bias=False)
        self.bn8 = nn.BatchNorm2d(256)

        # /4
        self.conv9 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn9 = nn.BatchNorm2d(512)
        self.conv10 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn10 = nn.BatchNorm2d(512)
        self.conv11 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn11 = nn.BatchNorm2d(512)
        self.conv12 = nn.Conv2d(512, 512, 1, stride=2, bias=False)
        self.bn12 = nn.BatchNorm2d(512)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.pool0(self.relu(self.bn0(self.conv0(x))))

        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.relu(self.bn3(self.conv3(self.relu(self.bn2(self.conv2(x1))))))
        x3 = self.relu(self.bn4(self.conv4(x1+x2)))

        x4 = self.relu(self.bn5(self.conv5(x3)))
        x5 = self.relu(self.bn7(self.conv7(self.relu(self.bn6(self.conv6(x4))))))
        x6 = self.relu(self.bn8(self.conv8(x4+x5)))

        x7 = self.relu(self.bn9(self.conv9(x6)))
        x8 = self.relu(self.bn11(self.conv11(self.relu(self.bn10(self.conv10(x7))))))
        x9 = self.relu(self.bn12(self.conv12(x7+x8)))

        return x9


class BaseConv(nn.Module):
    def __init__(self, input_chann=3):
        super(BaseConv, self).__init__()
        # /2
        self.conv0 = nn.Conv2d(input_chann, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn0 = nn.BatchNorm2d(64)
        self.pool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # /2
        self.conv1 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 1, stride=2, bias=False)
        self.bn4 = nn.BatchNorm2d(128)

        # /3
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 256, 1, stride=2, bias=False)
        self.bn8 = nn.BatchNorm2d(256)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.pool0(self.relu(self.bn0(self.conv0(x))))

        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.relu(self.bn3(self.conv3(self.relu(self.bn2(self.conv2(x1))))))
        x3 = self.relu(self.bn4(self.conv4(x1+x2)))

        x4 = self.relu(self.bn5(self.conv5(x3)))
        x5 = self.relu(self.bn7(self.conv7(self.relu(self.bn6(self.conv6(x4))))))
        x6 = self.relu(self.bn8(self.conv8(x4+x5)))

        return x6


class BaseDeepDeconv(nn.Module):
    def __init__(self, input_chann=512):
        super(BaseDeepDeconv, self).__init__()

        self.deconv1 = nn.ConvTranspose2d(input_chann, 512, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 512, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(512)
        self.deconv3 = nn.ConvTranspose2d(512, 256, 1, stride=2, bias=False)
        self.bn3 = nn.BatchNorm2d(256)

        self.deconv4 = nn.ConvTranspose2d(256, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.deconv5 = nn.ConvTranspose2d(256, 256, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.deconv6 = nn.ConvTranspose2d(256, 128, 1, stride=2, bias=False)
        self.bn6 = nn.BatchNorm2d(128)

        self.deconv7 = nn.ConvTranspose2d(128, 128, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(128)
        self.deconv8 = nn.ConvTranspose2d(128, 128, 3, padding=1)
        self.bn8 = nn.BatchNorm2d(128)
        self.deconv9 = nn.ConvTranspose2d(128, 64, 1, stride=2, bias=False)
        self.bn9 = nn.BatchNorm2d(64)

        self.deconv10 = nn.ConvTranspose2d(64, 64, 3, padding=1)
        self.bn10 = nn.BatchNorm2d(64)
        self.deconv11 = nn.ConvTranspose2d(64, 64, 3, padding=1)
        self.bn11 = nn.BatchNorm2d(64)
        self.deconv12 = nn.ConvTranspose2d(64, 3, 1, stride=2, bias=False)
        self.bn12 = nn.BatchNorm2d(3)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.relu(self.bn2(self.deconv2(self.relu(self.bn1(self.deconv1(x))))))
        x2 = self.relu(self.bn3(self.deconv3(x+x1)))

        x3 = self.relu(self.bn5(self.deconv5(self.relu(self.bn4(self.deconv4(x2))))))
        x4 = self.relu(self.bn6(self.deconv6(x2+x3)))

        x5 = self.relu(self.bn8(self.deconv8(self.relu(self.bn7(self.deconv7(x4))))))
        x6 = self.relu(self.bn9(self.deconv9(x4+x5)))

        x7 = self.relu(self.bn11(self.deconv11(self.relu(self.bn10(self.deconv10(x6))))))
        x8 = self.relu(self.bn12(self.deconv12(x6+x7)))

        return x8


class BaseDeconv(nn.Module):
    def __init__(self, input_chann=128):
        super(BaseDeconv, self).__init__()

        self.deconv1 = nn.ConvTranspose2d(input_chann, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(128, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 64, 1, stride=2, bias=False)
        self.bn3 = nn.BatchNorm2d(64)

        self.deconv4 = nn.ConvTranspose2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 64, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.deconv6 = nn.ConvTranspose2d(64, 3, 1, stride=2, bias=False)
        self.bn6 = nn.BatchNorm2d(3)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.relu(self.bn2(self.deconv2(self.relu(self.bn1(self.deconv1(x))))))
        x2 = self.relu(self.bn3(self.deconv3(x+x1)))

        x3 = self.relu(self.bn5(self.deconv5(self.relu(self.bn4(self.deconv4(x2))))))
        x4 = self.relu(self.bn6(self.deconv6(x2+x3)))

        return x4


class JointRegression(nn.Module):
    ''' joint angle regression from hand embedding space'''
    def __init__(self, input_size=128, output_size=24):
        super(JointRegression, self).__init__()

        hidden_size = input_size // 2
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.reg = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        angle = self.reg(x)

        return angle


class NewJointRegression(nn.Module):
    def __init__(self, input_size=128, output_size=24):
        super(NewJointRegression, self).__init__()
        self.reg = nn.Linear(input_size, output_size)

    def forward(self, x):
        angle = self.reg(x)

        return angle

class Discriminator_Embedding(nn.Module):
    ''' hand embedding space discriminator'''
    def __init__(self, input_size=128):
        super(Discriminator_Embedding, self).__init__()

        hidden_size = input_size // 2
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        output = F.sigmoid(self.fc3(x))

        return output


class TeachingTeleModel(nn.Module):
    ''' model with joint loss, predict loss and consistancy loss '''
    def __init__(self, input_size=100, embedding_size=128, joint_size=24):
        super(TeachingTeleModel, self).__init__()
        self.base_human = BaseDeepConv(input_chann=1)
        self.base_shadow = BaseDeepConv(input_chann=1)
        # self.base_human = Resnet18Conv(input_chann=1)
        # self.base_shadow = Resnet18Conv(input_chann=1)
        # self.feature_size = 512*((input_size//16)**2)
        self.feature_size = 512*16
        self.embedding_size = embedding_size
        self.joint_size = joint_size

        # human branch
        self.encoder_human = nn.Sequential(
            nn.Linear(self.feature_size, self.embedding_size*4),
            nn.BatchNorm1d(self.embedding_size*4),
            nn.ReLU(),
            nn.Linear(self.embedding_size*4, self.embedding_size)
        )
        # shadow branch
        self.encoder_shadow = nn.Sequential(
            nn.Linear(self.feature_size, self.embedding_size*4),
            nn.BatchNorm1d(self.embedding_size*4),
            nn.ReLU(),
            nn.Linear(self.embedding_size*4, self.embedding_size)
        )

        self.human_reg = JointRegression(input_size=self.embedding_size, output_size=self.joint_size)
        self.shadow_reg = JointRegression(input_size=self.embedding_size, output_size=self.joint_size)


    def forward(self, x, is_human=True):
        if is_human:
            x = self.base_human(x).view(-1, self.feature_size)
            embedding = self.encoder_human(x)
            joint = self.human_reg(embedding)
        else:
            x = self.base_shadow(x).view(-1, self.feature_size)
            embedding = self.encoder_shadow(x)
            joint = self.shadow_reg(embedding)

        return embedding, joint


class NewTeachingTeleModel(nn.Module):
    ''' New model with joint loss, predict loss and consistancy loss '''
    def __init__(self, input_size=100, embedding_size=128, joint_size=24):
        super(NewTeachingTeleModel, self).__init__()
        self.base_human = BaseDeepConv(input_chann=1)
        self.base_shadow = BaseDeepConv(input_chann=1)
        # self.base_human = Resnet18Conv(input_chann=1)
        # self.base_shadow = Resnet18Conv(input_chann=1)
        # self.feature_size = 512*((input_size//16)**2)
        self.feature_size = 512*16
        self.embedding_size = embedding_size
        self.joint_size = joint_size

        # human branch
        self.encoder_human = nn.Sequential(
            nn.Linear(self.feature_size, self.embedding_size*4),
            nn.BatchNorm1d(self.embedding_size*4),
            nn.ReLU(),
            nn.Linear(self.embedding_size*4, self.embedding_size*2),
            nn.BatchNorm1d(self.embedding_size*2),
            nn.ReLU(),
            nn.Linear(self.embedding_size*2, self.embedding_size),
            nn.BatchNorm1d(self.embedding_size),
            nn.ReLU(),
            nn.Linear(self.embedding_size, self.embedding_size)
        )
        # shadow branch
        self.encoder_shadow = nn.Sequential(
            nn.Linear(self.feature_size, self.embedding_size*4),
            nn.BatchNorm1d(self.embedding_size*4),
            nn.ReLU(),
            nn.Linear(self.embedding_size*4, self.embedding_size*2),
            nn.BatchNorm1d(self.embedding_size*2),
            nn.ReLU(),
            nn.Linear(self.embedding_size*2, self.embedding_size),
            nn.BatchNorm1d(self.embedding_size),
            nn.ReLU(),
            nn.Linear(self.embedding_size, self.embedding_size)
        )

        self.human_reg = NewJointRegression(input_size=self.embedding_size, output_size=self.joint_size)
        self.shadow_reg = NewJointRegression(input_size=self.embedding_size, output_size=self.joint_size)


    def forward(self, x, is_human=True):
        if is_human:
            x = self.base_human(x).view(-1, self.feature_size)
            embedding = self.encoder_human(x)
            joint = self.human_reg(embedding)
        else:
            x = self.base_shadow(x).view(-1, self.feature_size)
            embedding = self.encoder_shadow(x)
            joint = self.shadow_reg(embedding)

        return embedding, joint


class TeachingTeleGANModel(nn.Module):
    ''' model with joint loss, predict loss and soft consistancy loss '''
    def __init__(self, input_size=100, embedding_size=128, joint_size=24):
        super(TeachingTeleGANModel, self).__init__()
        self.base_human = BaseDeepConv(input_chann=1)
        self.base_shadow = BaseDeepConv(input_chann=1)
        # self.base_human = Resnet18Conv(input_chann=1)
        # self.base_shadow = Resnet18Conv(input_chann=1)
        # self.feature_size = 512*((input_size//16)**2)
        self.feature_size = 512*16
        self.embedding_size = embedding_size
        self.joint_size = joint_size

        # human branch
        self.encoder_human = nn.Sequential(
            nn.Linear(self.feature_size, self.embedding_size*4),
            nn.BatchNorm1d(self.embedding_size*4),
            nn.ReLU(),
            nn.Linear(self.embedding_size*4, self.embedding_size)
        )
        # shadow branch
        self.encoder_shadow = nn.Sequential(
            nn.Linear(self.feature_size, self.embedding_size*4),
            nn.BatchNorm1d(self.embedding_size*4),
            nn.ReLU(),
            nn.Linear(self.embedding_size*4, self.embedding_size)
        )

        self.human_reg = JointRegression(input_size=self.embedding_size, output_size=self.joint_size)
        self.shadow_reg = JointRegression(input_size=self.embedding_size, output_size=self.joint_size)
        self.discriminator = Discriminator_Embedding(self.embedding_size)


    def forward(self, x, is_human=True):
        if is_human:
            x = self.base_human(x).view(-1, self.feature_size)
            embedding = self.encoder_human(x)
            joint = self.human_reg(embedding)
        else:
            x = self.base_shadow(x).view(-1, self.feature_size)
            embedding = self.encoder_shadow(x)
            joint = self.shadow_reg(embedding)

        return embedding, joint


class NewTeachingTeleGANModel(nn.Module):
    ''' new model with joint loss, predict loss and soft consistancy loss '''
    def __init__(self, input_size=100, embedding_size=128, joint_size=24):
        super(NewTeachingTeleGANModel, self).__init__()
        self.base_human = BaseDeepConv(input_chann=1)
        self.base_shadow = BaseDeepConv(input_chann=1)
        # self.base_human = Resnet18Conv(input_chann=1)
        # self.base_shadow = Resnet18Conv(input_chann=1)
        # self.feature_size = 512*((input_size//16)**2)
        self.feature_size = 512*16
        self.embedding_size = embedding_size
        self.joint_size = joint_size

        # human branch
        self.encoder_human = nn.Sequential(
            nn.Linear(self.feature_size, self.embedding_size*4),
            nn.BatchNorm1d(self.embedding_size*4),
            nn.ReLU(),
            nn.Linear(self.embedding_size*4, self.embedding_size*2),
            nn.BatchNorm1d(self.embedding_size*2),
            nn.ReLU(),
            nn.Linear(self.embedding_size*2, self.embedding_size),
            nn.BatchNorm1d(self.embedding_size),
            nn.ReLU(),
            nn.Linear(self.embedding_size, self.embedding_size)
        )
        # shadow branch
        self.encoder_shadow = nn.Sequential(
            nn.Linear(self.feature_size, self.embedding_size*4),
            nn.BatchNorm1d(self.embedding_size*4),
            nn.ReLU(),
            nn.Linear(self.embedding_size*4, self.embedding_size*2),
            nn.BatchNorm1d(self.embedding_size*2),
            nn.ReLU(),
            nn.Linear(self.embedding_size*2, self.embedding_size),
            nn.BatchNorm1d(self.embedding_size),
            nn.ReLU(),
            nn.Linear(self.embedding_size, self.embedding_size)
        )

        self.human_reg = NewJointRegression(input_size=self.embedding_size, output_size=self.joint_size)
        self.shadow_reg = NewJointRegression(input_size=self.embedding_size, output_size=self.joint_size)
        self.discriminator = Discriminator_Embedding(self.embedding_size)

    def forward(self, x, is_human=True):
        if is_human:
            x = self.base_human(x).view(-1, self.feature_size)
            embedding = self.encoder_human(x)
            joint = self.human_reg(embedding)
        else:
            x = self.base_shadow(x).view(-1, self.feature_size)
            embedding = self.encoder_shadow(x)
            joint = self.shadow_reg(embedding)

        return embedding, joint


class NaiveTeleModel(nn.Module):
    ''' model with predict loss only '''
    def __init__(self, input_size=100, embedding_size=64, joint_size=24):
        super(NaiveTeleModel, self).__init__()
        self.base = BaseDeepConv(input_chann=1)
        # self.base = Resnet18Conv(input_chann=1)
        # self.feature_size = 512*((input_size//16)**2)
        self.feature_size = 512*16
        self.embedding_size = embedding_size
        self.joint_size = joint_size

        # human branch
        self.encoder_human = nn.Sequential(
            nn.Linear(self.feature_size, self.embedding_size*4),
            nn.BatchNorm1d(self.embedding_size*4),
            nn.ReLU(),
            nn.Linear(self.embedding_size*4, self.embedding_size)
        )

        self.human_reg = JointRegression(input_size=self.embedding_size, output_size=self.joint_size)


    def forward(self, x, is_human):
        x = self.base(x).view(-1, self.feature_size)
        embedding = self.encoder_human(x)
        joint = self.human_reg(embedding)

        return embedding, joint


class NewNaiveTeleModel(nn.Module):
    ''' new model with predict loss only '''
    def __init__(self, input_size=100, embedding_size=64, joint_size=24):
        super(NaiveTeleModel, self).__init__()
        self.base = BaseDeepConv(input_chann=1)
        # self.base = Resnet18Conv(input_chann=1)
        # self.feature_size = 512*((input_size//16)**2)
        self.feature_size = 512*16
        self.embedding_size = embedding_size
        self.joint_size = joint_size

        # human branch
        self.encoder_human = nn.Sequential(
            nn.Linear(self.feature_size, self.embedding_size*4),
            nn.BatchNorm1d(self.embedding_size*4),
            nn.ReLU(),
            nn.Linear(self.embedding_size*4, self.embedding_size*2),
            nn.BatchNorm1d(self.embedding_size*2),
            nn.ReLU(),
            nn.Linear(self.embedding_size*2, self.embedding_size),
            nn.BatchNorm1d(self.embedding_size),
            nn.ReLU(),
            nn.Linear(self.embedding_size, self.embedding_size)
        )

        self.human_reg = NewJointRegression(input_size=self.embedding_size, output_size=self.joint_size)

    def forward(self, x, is_human):
        x = self.base(x).view(-1, self.feature_size)
        embedding = self.encoder_human(x)
        joint = self.human_reg(embedding)

        return embedding, joint


if __name__ == '__main__':
    x=torch.ones(2, 1, 100, 100)
    t= TeachingRENModel(100, 128, 22)
    a,b = t.forward(x, 1)
    print(b.shape)
    print(a.shape)
