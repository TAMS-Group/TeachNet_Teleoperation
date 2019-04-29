#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# File Name : eval_error_bar.py
# Creation Date : 10-09-2018
# Created By : Jeasine Ma [jeasinema[at]gmail[dot]com]

#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# Purpose : draw histogram of joint error
# Creation Date : 17-08-2018
# Created By : sli

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


target = np.loadtxt(open("./test_label.csv", "rb"),
        dtype='S30', delimiter=",")
target = target[np.argsort(target[:, 0])][:, 3:].astype(float)
target = np.delete(target, (3, 8, 12, 16, 21), axis=1)
#tams_target = np.delete(target, (5,10,14,18,23), axis=1)[:,2:]

# handik
handik_output = np.loadtxt(open("./predict_handik/input.csv", "rb"),
         dtype='S30', delimiter=",")
handik_output = handik_output[np.argsort(handik_output[:, 0])][:, 3:].astype(float)
handik_output = np.delete(handik_output, (3, 8, 12, 16, 21), axis=1)
# average joint loss
ave_loss = np.sqrt((handik_output -target) ** 2).mean()
from IPython import embed; embed()
# error loss for each joint
joint_loss = np.sqrt((handik_output - target) ** 2).mean(axis=0)
handik_loss=np.hstack([joint_loss, ave_loss]).tolist()
# handik_loss[13]=0.36

# single Shadow
shadow_output = np.loadtxt(open("./predict_single_shadow/input.csv", "rb"),
         dtype='S30', delimiter=",")
shadow_output = shadow_output[np.argsort(shadow_output[:, 0])][:, 3:].astype(float)
shadow_output = np.delete(shadow_output, (3, 8, 12, 16, 21), axis=1)
shadow_ave_loss = np.sqrt((shadow_output - target) ** 2).mean()
shadow_joint_loss = np.sqrt((shadow_output - target) ** 2).mean(axis=0)
shadow_loss = np.hstack([shadow_joint_loss, shadow_ave_loss]).tolist()

# single human
human_output = np.loadtxt(open("./predict_single_human/input.csv", "rb"),
         dtype='S30', delimiter=",")
human_output = human_output[np.argsort(human_output[:, 0])][:, 3:].astype(float)
human_output = np.delete(human_output, (3, 8, 12, 16, 21), axis=1)
human_ave_loss = np.sqrt((human_output -target) ** 2).mean()
human_joint_loss = np.sqrt((human_output - target) ** 2).mean(axis=0)
human_loss=np.hstack([human_joint_loss, human_ave_loss]).tolist()

# teach mse early
mse_early_output = np.loadtxt(open("./predict_teach_mse_early/input.csv", "rb"),
         dtype='S30', delimiter=",")
mse_early_output = mse_early_output[np.argsort(mse_early_output[:, 0])][:, 3:].astype(float)
mse_early_output = np.delete(mse_early_output, (3, 8, 12, 16, 21), axis=1)
mse_early_ave_loss = np.sqrt((mse_early_output -target) ** 2).mean()
mse_early_joint_loss = np.sqrt((mse_early_output - target) ** 2).mean(axis=0)
mse_early_loss=np.hstack([mse_early_joint_loss, mse_early_ave_loss]).tolist()

# teach mse late
mse_late_output = np.loadtxt(open("./predict_teach_mse_late/input.csv", "rb"),
        dtype='S30', delimiter=",")
mse_late_output = mse_late_output[np.argsort(mse_late_output[:, 0])][:, 3:].astype(float)
mse_late_output = np.delete(mse_late_output, (3, 8, 12, 16, 21), axis=1)
mse_late_ave_loss = np.sqrt((mse_late_output - target) ** 2).mean()
mse_late_joint_loss = np.sqrt((mse_late_output - target) ** 2).mean(axis=0)
mse_late_loss=np.hstack([mse_late_joint_loss, mse_late_ave_loss]).tolist()

# teach gan early
gan_early_output = np.loadtxt(open("./predict_teach_gan_early/input.csv", "rb"),
         dtype='S30', delimiter=",")
gan_early_output = gan_early_output[np.argsort(gan_early_output[:, 0])][:, 3:].astype(float)
gan_early_output = np.delete(gan_early_output, (3, 8, 12, 16, 21), axis=1)
gan_early_ave_loss = np.sqrt((gan_early_output -target) ** 2).mean()
gan_early_joint_loss = np.sqrt((gan_early_output - target) ** 2).mean(axis=0)
gan_early_loss = np.hstack([gan_early_joint_loss, gan_early_ave_loss]).tolist()
#
# # teach gan late
gan_late_output = np.loadtxt(open("./predict_teach_gan_late/input.csv", "rb"),
        dtype='S30', delimiter=",")
gan_late_output = gan_late_output[np.argsort(gan_late_output[:, 0])][:, 3:].astype(float)
gan_late_output = np.delete(gan_late_output, (3, 8, 12, 16, 21), axis=1)
gan_late_ave_loss = np.sqrt((gan_late_output - target) ** 2).mean()
gan_late_joint_loss = np.sqrt((gan_late_output - target) ** 2).mean(axis=0)
gan_late_loss=np.hstack([gan_late_joint_loss, gan_late_ave_loss]).tolist()

name_list = ['F4', 'F3', 'F2',
             'L5', 'L4', 'L3', 'L2',
             'M4', 'M3', 'M2',
             'R4', 'R3', 'R2',
             'T5', 'T4', 'T3', 'T2', 'Ave']

x = list(range(len(name_list)))
total_width, n = 1.5, 10
width = total_width / n

clrs = sns.color_palette("husl", 6)
with sns.axes_style('darkgrid'):
    fig = plt.figure()
    fig.set_size_inches(15, 7)

    plt.bar(x, shadow_loss, width=width,  label='single_shadow', fc=clrs[0])
    for i in range(len(x)):
        x[i] = x[i] + width
    plt.bar(x, human_loss, width=width, label='single_human', fc=clrs[1])
    for i in range(len(x)):
        x[i] = x[i] + width
    plt.bar(x, gan_early_loss, width=width,  label='teach_gan_early', fc=clrs[2])
    for i in range(len(x)):
        x[i] = x[i] + width
    plt.bar(x, gan_late_loss, width=width,  label='teach_gan_late', fc=clrs[3], tick_label=name_list)
    for i in range(len(x)):
        x[i] = x[i] + width
    plt.bar(x, mse_early_loss, width=width,  label='teach_mse_early', fc=clrs[4])
    for i in range(len(x)):
        x[i] = x[i] + width
    plt.bar(x, mse_late_loss, width=width,  label='teach_mse_late', fc=clrs[5])
    # for i in range(len(x)):
    #     x[i] = x[i] + width
    # plt.bar(x, handik_loss, width=width,  label='handIK', fc='wheat')
    ax = plt.gca()
    ax.tick_params(axis='x', which='major', labelsize=16)
    ax.tick_params(axis='y', which='major', labelsize=16)
    plt.yticks(np.arange(0, 0.04, 0.03))
    plt.margins(x=0)
    plt.ylabel('Mean Error (rad)', fontsize=16)
    plt.legend(prop={'size': 16})
    fig.savefig('error_bar.pdf')
