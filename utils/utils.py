#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# File Name : utils.py
# Purpose :
# Creation Date : 09-04-2018
# Last Modified : Fri 20 Apr 2018 07:02:31 PM CST
# Created By : Jeasine Ma [jeasinema[at]gmail[dot]com]
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
from math import cos, sin, floor, sqrt, pi, ceil


def bar(current, total, prefix="", suffix="", bar_sz=25, end_string=None):
    sp = ""
    print("\x1b[2K\r", end='')
    for i in range(bar_sz):
        if current * bar_sz // total > i:
            sp += '='
        elif current * bar_sz // total == i:
            sp += '>'
        else:
            sp += ' '
    if current == total:
        if end_string is None:
            print("\r%s[%s]%s" % (prefix, sp, suffix))
        else:
            if end_string != "":
                print("\r%s" % end_string)
            else:
                print("\r", end='')
    else:
        print("\r%s[%s]%s" % (prefix, sp, suffix), end='')


class EnvDataset(data.Dataset):
    def __init__(self, config, train=True):
        self.cfg = config
        self.train = train
        self.env = self.cfg.env(self.cfg)

    def __getitem__(self, index):
        return self.env.sample()

    def __len__(self):
        return self.cfg.batches_train*self.cfg.batch_size_train if self.train else self.cfg.batch_size_test


def collate_fn_env(batch):
    ret = []
    for sample in batch:
        ret.append(tuple(sample))
    return ret


def worker_init_fn_env(pid):
    np.random.seed(torch.initial_seed() % (2**31-1))


# generator: (traj, task, image) x batch_size
def build_loader(config, train=True):
    return torch.utils.data.DataLoader(
        EnvDataset(
            config=config,
            train=train
        ),
        batch_size=config.batch_size_train if train else config.batch_size_test,
        num_workers=config.multi_threads if train else 1,
        pin_memory=True if train else False,
        shuffle=True,
        collate_fn=collate_fn_env,
        worker_init_fn=worker_init_fn_env
    )


def euclidean_distance(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return sqrt(dx * dx + dy * dy)


def poisson_disc_samples(width, height, r, k=5, distance=euclidean_distance, random=np.random.rand):
    tau = 2 * pi
    cellsize = r / sqrt(2)

    grid_width = int(ceil(width / cellsize))
    grid_height = int(ceil(height / cellsize))
    grid = [None] * (grid_width * grid_height)

    def grid_coords(p):
        return int(floor(p[0] / cellsize)), int(floor(p[1] / cellsize))

    def fits(p, gx, gy):
        yrange = list(range(max(gy - 2, 0), min(gy + 3, grid_height)))
        for x in range(max(gx - 2, 0), min(gx + 3, grid_width)):
            for y in yrange:
                g = grid[x + y * grid_width]
                if g is None:
                    continue
                if distance(p, g) <= r:
                    return False
        return True

    p = width * random(), height * random()
    queue = [p]
    grid_x, grid_y = grid_coords(p)
    grid[grid_x + grid_y * grid_width] = p

    while queue:
        qi = int(random() * len(queue))
        qx, qy = queue[qi]
        queue[qi] = queue[-1]
        queue.pop()
        for _ in range(k):
            alpha = tau * random()
            d = r * sqrt(3 * random() + 1)
            px = qx + d * cos(alpha)
            py = qy + d * sin(alpha)
            if not (0 <= px < width and 0 <= py < height):
                continue
            p = (px, py)
            grid_x, grid_y = grid_coords(p)
            if not fits(p, grid_x, grid_y):
                continue
            queue.append(p)
            grid[grid_x + grid_y * grid_width] = p
    return [p for p in grid if p is not None]


if __name__ == '__main__':
    pass
