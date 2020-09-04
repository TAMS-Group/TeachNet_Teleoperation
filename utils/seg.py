#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# File Name : seg.py
# Creation Date : 10-07-2018
# Created By : Jeasine Ma [jeasinema[at]gmail[dot]com]

from __future__ import division

import sys
import cv2
import pickle
import numpy as np
from numba import jit
from IPython import embed


@jit(nopython=True)
def surround(i, j, xl, yl, add=1):
    sur = []
    if i - add >= 0:
        sur.append([i - add, j])
    if j - add >= 0:
        sur.append([i, j - add])
    if i + add < xl:
        sur.append([i + add, j])
    if j + add < yl:
        sur.append([i, j + add])
    return sur


def seg_hand_with_label(img, label, output_size=100):
    label = label.astype(np.float32)
    top = np.min(label[:, 1])
    bottom = np.max(label[:, 1])
    left = np.min(label[:, 0])
    right = np.max(label[:, 0])
    width, height = right - left, bottom - top
    padding = 10
    if height > width:
        left_padding = float(height - width)
        top_padding = 0
    else:
        left_padding = 0
        top_padding = float(width - height)
    x_min = int(max(0, top - top_padding / 2 - padding / 2))
    x_max = int(min(img.shape[0] - 1, bottom + top_padding / 2 + padding / 2))
    y_min = int(max(0, left - left_padding / 2 - padding / 2))
    y_max = int(min(img.shape[1] - 1, right + left_padding / 2 + padding / 2))
    img = img[x_min:x_max, y_min:y_max]
    img = cv2.resize(img, (output_size, output_size))
    label[:, 0] = label[:, 0] - float(y_min)
    label[:, 1] = label[:, 1] - float(x_min)
    label[:, 0] *= (float(output_size) / float(y_max - y_min + 1))
    label[:, 1] *= (float(output_size) / float(x_max - x_min + 1))
    label = label.round().astype(np.int32)

    return img, label, np.array([x_max, x_min, y_max, y_min])


@jit(nopython=True)
def inner(inner_edge, img, zero_as_infty, fore_thresh, mask, gap, thresh, x, y, w, l, add):
    for i, j in zip(x, y):
        sur = surround(i, j, w, l, add)
        for s in sur:
            xx, yy = s
            if gap < abs(img[xx, yy] - img[i, j]):
                if zero_as_infty or abs(img[xx, yy] - img[i, j]) < thresh:
                    if img[xx, yy] > img[i, j]:
                        if img[i, j] <= fore_thresh:
                            mask[xx, yy] = 0
                            inner_edge.append((i, j))
                    else:
                        if img[xx, yy] <= fore_thresh:
                            mask[i, j] = 0
                            inner_edge.append((xx, yy))
    return inner_edge, mask


def seg_hand_depth(img, gap=100, thresh=500, padding=10, output_size=100, scale=10, add=5, box_z=250,
                   zero_as_infty=False, fore_p_thresh=300, label=None):
    img = img.astype(np.float32)
    img[np.where(img > 450)] = 0
    if zero_as_infty:
        # TODO: for some sensor that maps infty as 0, we should override them
        thresh = np.inf
        his = np.histogram(img[img != 0])
        sum_p = 0
        for i in range(len(his[0])):
            sum_p += his[0][i]
            if his[0][i] == 0 and sum_p > fore_p_thresh:
                fore_thresh = his[1][i]
                break
        else:
            fore_thresh = np.inf
    else:
        fore_thresh = np.inf
    mask = np.ones_like(img)
    w, l = img.shape
    x = np.linspace(0, w - 1, w // scale)
    y = np.linspace(0, l - 1, l // scale)
    grid = np.meshgrid(x, y)
    x = grid[0].reshape(-1).astype(np.int32)
    y = grid[1].reshape(-1).astype(np.int32)
    inner_edge = []
    if zero_as_infty:
        img[img == 0] = np.iinfo(np.uint16).max

    # morphlogy
    open_mask = np.zeros_like(img)
    open_mask[img != np.iinfo(np.uint16).max] = 1
    tmp = open_mask.copy()
    tmp = cv2.morphologyEx(tmp, cv2.MORPH_OPEN, np.ones((3, 3)))
    open_mask -= tmp
    img[open_mask.astype(np.bool)] = np.iinfo(np.uint16).max

    inner_edge = [(1, 1)]
    inner_edge, mask = inner(inner_edge, img, zero_as_infty, fore_thresh, mask, gap, thresh, x, y, w, l, add)
    inner_edge = inner_edge[1:]
    # for i, j in zip(x, y):
    #     sur = surround(i, j, w, l, add)
    #     for s in sur:
    #         xx, yy = s
    #         if gap < abs(img[xx, yy] - img[i, j]):
    #             if zero_as_infty or abs(img[xx, yy] - img[i, j]) < thresh:
    #                 if img[xx, yy] > img[i, j]:
    #                     if img[i, j] <= fore_thresh:
    #                         mask[xx, yy] = 0
    #                         inner_edge.append((i, j))
    #                 else:
    #                     if img[xx, yy] <= fore_thresh:
    #                         mask[i, j] = 0
    #                         inner_edge.append((xx, yy))
    mask = mask.astype(np.bool)
    edge_x, edge_y = np.where(mask == 0)
    x_min, x_max = np.min(edge_x), np.max(edge_x)
    y_min, y_max = np.min(edge_y), np.max(edge_y)

    x_min = max(0, x_min - padding)
    x_max = min(x_max + padding, w - 1)
    y_min = max(0, y_min - padding)
    y_max = min(y_max + padding, l - 1)
    if x_max - x_min > y_max - y_min:
        delta = (x_max - x_min) - (y_max - y_min)
        y_min -= delta / 2
        y_max += delta / 2
    else:
        delta = (y_max - y_min) - (x_max - x_min)
        x_min -= delta / 2
        x_max += delta / 2
    x_min = int(max(0, x_min))
    x_max = int(min(x_max, w - 1))
    y_min = int(max(0, y_min))
    y_max = int(min(y_max, l - 1))

    edge_depth = []
    for (x, y) in inner_edge:
        edge_depth.append(img[x, y])
    avg_depth = np.sum(edge_depth) / float(len(edge_depth))
    # avg_depth = 40
    depth_min = max(avg_depth - box_z / 2, 0)
    depth_max = avg_depth + box_z / 2
    seg_area = img.copy()
    seg_area[seg_area < depth_min] = depth_min
    seg_area[seg_area > depth_max] = depth_max
    # normalized
    seg_area = ((seg_area - avg_depth) / (box_z / 2))  # [-1, 1]
    seg_area = ((seg_area + 1) / 2.) * 255.  # [0, 255]

    output = seg_area[x_min:x_max, y_min:y_max]
    output = cv2.resize(output, (output_size, output_size)).astype(np.uint16)
    # rgb = rgb.copy()
    # rgb = rgb[x_min-10:x_max+100, y_min-10:y_max+100]
    # rgb = cv2.resize(rgb, (output_size*2, output_size*2))
    if label is not None:
        label = label.astype(np.float32)
        label[:, 0] = label[:, 0] - y_min
        label[:, 1] = label[:, 1] - x_min
        label[:, 0] *= (float(output_size) / (y_max - y_min + 1))
        label[:, 1] *= (float(output_size) / (x_max - x_min + 1))
        label = label.round().astype(np.int32)
        return output, label, np.array([x_max, x_min, y_max, y_min])
    else:
        # return output, rgb
        return output


def main():
    # params:
    # zero_as_infty, fore_p_thresh, gap, thresh, fore_thresh, scale, add, box_z
    # for simulated shadow:
    #       False, X(useless), 100, 500, X(useless), 10, 5, 250
    # for cropped human hand(bigHand dataset):
    #       True, 300, 500, X(useless), histogram(with depth==0 removed), 4, 4, 250
    # for real shadow:
    # for real human hand:
    #       True, 300, 500, X(useless), histogram(with depth==0 removed), 4, 4, 250
    img = cv2.imread(sys.argv[1], cv2.IMREAD_ANYDEPTH).astype(np.float32)
    zero_as_infty = False
    fore_p_thresh = 300
    if zero_as_infty:
        # TODO: for some sensor that maps infty as 0, we should override them
        gap = int(sys.argv[2])  # edge detection
        thresh = np.inf
        his = np.histogram(img[img != 0])
        sum_p = 0
        for i in range(len(his[0])):
            sum_p += his[0][i]
            if his[0][i] == 0 and sum_p > fore_p_thresh:
                fore_thresh = his[1][i]
                break
        else:
            fore_thresh = np.inf
    else:
        gap = int(sys.argv[2])  # edge detection
        thresh = int(sys.argv[3])  # prevent background edge
        fore_thresh = np.inf
    scale = int(sys.argv[4])
    add = int(sys.argv[5])
    box_z = int(sys.argv[6])
    w, l = 480, 640
    # w, l = 100, 100
    mask = np.ones_like(img)
    x = np.linspace(0, w - 1, w // scale)
    y = np.linspace(0, l - 1, l // scale)
    grid = np.meshgrid(x, y)
    x = grid[0].reshape(-1).astype(np.int32)
    y = grid[1].reshape(-1).astype(np.int32)
    inner_edge = []
    # TODO: for some sensor that maps infty as 0, we should let 0 be a large value then compute edge
    if zero_as_infty:
        img[img == 0] = np.iinfo(np.uint16).max

    # morphlogy
    open_mask = np.zeros_like(img)
    open_mask[img != np.iinfo(np.uint16).max] = 1
    tmp = open_mask.copy()
    tmp = cv2.morphologyEx(tmp, cv2.MORPH_OPEN, np.ones((3, 3)))
    open_mask -= tmp
    img[open_mask.astype(np.bool)] = np.iinfo(np.uint16).max

    for i, j in zip(x, y):
        sur = surround(i, j, w, l, add)
        for s in sur:
            xx, yy = s
            if gap < abs(img[xx, yy] - img[i, j]):
                if zero_as_infty or abs(img[xx, yy] - img[i, j]) < thresh:
                    if img[xx, yy] > img[i, j]:
                        if img[i, j] <= fore_thresh:
                            mask[xx, yy] = 0
                            inner_edge.append((i, j))
                    else:
                        if img[xx, yy] <= fore_thresh:
                            mask[i, j] = 0
                            inner_edge.append((xx, yy))
    mask = mask.astype(np.bool)
    seg_area = img.copy()
    tmp = seg_area[..., np.newaxis]
    tmp = np.dstack([tmp, tmp, tmp])
    n_max, n_min = np.max(tmp), np.min(tmp)
    tmp = (tmp - n_min) / (n_max - n_min) * 255
    # cv2.imwrite('{}_output.png'.format(sys.argv[1][:-4]), tmp)
    edge_x, edge_y = np.where(mask == 0)
    for x, y in zip(edge_x, edge_y):
        tmp = cv2.circle(tmp, (y, x), 2, (255, 0, 0))
    for (x, y) in inner_edge:
        tmp = cv2.circle(tmp, (y, x), 2, (0, 255, 0))
    cv2.imwrite('{}_seg_area.png'.format(sys.argv[1][:-4]), tmp)
    x_min, x_max = np.min(edge_x), np.max(edge_x)
    y_min, y_max = np.min(edge_y), np.max(edge_y)

    x_min = max(0, x_min - 10)
    x_max = min(x_max + 10, w - 1)
    y_min = max(0, y_min - 10)
    y_max = min(y_max + 10, l - 1)
    if x_max - x_min > y_max - y_min:
        delta = (x_max - x_min) - (y_max - y_min)
        y_min -= delta / 2
        y_max += delta / 2
    else:
        delta = (y_max - y_min) - (x_max - x_min)
        x_min -= delta / 2
        x_max += delta / 2
    x_min = int(max(0, x_min))
    x_max = int(min(x_max, w - 1))
    y_min = int(max(0, y_min))
    y_max = int(min(y_max, l - 1))

    # bbox and depth normalization
    # after that, depth can be normalized by
    # (depth-np.min(depth))/(np.max(depth)-np.min(depth))
    edge_depth = []
    for (x, y) in inner_edge:
        edge_depth.append(img[x, y])
    avg_depth = np.sum(edge_depth) / float(len(edge_depth))
    depth_min = max(avg_depth - box_z / 2, 0)
    depth_max = avg_depth + box_z / 2
    print(avg_depth, depth_min, depth_max)
    seg_area = img.copy()
    # TODO: for some sensor that maps infty as 0, we should let 0 be depth_max
    # already done before
    seg_area[seg_area < depth_min] = depth_min
    seg_area[seg_area > depth_max] = depth_max
    # normalized
    seg_area = ((seg_area - avg_depth) / (box_z / 2))  # [-1, 1]
    seg_area = ((seg_area + 1) / 2.) * 255.  # [0, 255]

    tmp = seg_area[..., np.newaxis]
    tmp = np.dstack([tmp, tmp, tmp])
    n_max, n_min = np.max(tmp), np.min(tmp)
    tmp = (tmp - n_min) / (n_max - n_min) * 255

    ind = int(sys.argv[1][:-4].split('/')[-1])
    f = pickle.load(open('./uv{}.pkl'.format(ind), 'rb'))
    label = f[0][1]
    label = label[:, :2]
    label_crop = label.copy()
    label_crop[:, 0] = label_crop[:, 0] - y_min
    label_crop[:, 1] = label_crop[:, 1] - x_min

    output = tmp[x_min:x_max, y_min:y_max, :]
    label_crop[:, 0] *= (100. / (y_max - y_min))
    label_crop[:, 1] *= (100. / (x_max - x_min))
    label_crop = label_crop.round().astype(np.int32)
    output = cv2.resize(output, (100, 100))
    for l in label_crop:
        cv2.circle(output, (l[0], l[1]), 1, (255, 0, 0))
    cv2.imwrite('{}_crop.png'.format(sys.argv[1][:-4]), output)

    output = seg_area[x_min:x_max, y_min:y_max]
    output = cv2.resize(output, (100, 100))
    cv2.imwrite('{}_crop_depth.png'.format(sys.argv[1][:-4]), output.astype(np.uint16))
    print('seg_area done')


if __name__ == '__main__':
    main()
