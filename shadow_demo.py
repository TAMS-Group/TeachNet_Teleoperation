from __future__ import print_function
import argparse
import os
import time
import pickle
import glob
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import csv
from model.model import *
from utils import seg_hand_depth

import rospy
import moveit_commander
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from shadow_teleop.srv import *

parser = argparse.ArgumentParser(description='deepShadowTeleop')
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--model-path', type=str, default='./weights/naive_human_baseline_embed128.model',
                   help='pre-trained model path')
# add robot lated args here

args = parser.parse_args()

args.cuda = args.cuda if torch.cuda.is_available else False

if args.cuda:
    torch.cuda.manual_seed(1)

np.random.seed(int(time.time()))

input_size=100
embedding_size=128
joint_size=22

joint_upper_range = torch.tensor([0.349, 1.571, 1.571, 1.571, 0.785, 0.349, 1.571, 1.571,
                                  1.571, 0.349, 1.571, 1.571, 1.571, 0.349, 1.571, 1.571,
                                  1.571, 1.047, 1.222, 0.209, 0.524, 1.571])
joint_lower_range = torch.tensor([-0.349, 0, 0, 0, 0, -0.349, 0, 0, 0, -0.349, 0, 0, 0,
                                  -0.349, 0, 0, 0, -1.047, 0, -0.209, -0.524, 0])

model = torch.load(args.model_path, map_location='cpu')
model.device_ids = [args.gpu]
print('load model {}'.format(args.model_path))

if args.cuda:
    if args.gpu != -1:
        torch.cuda.set_device(args.gpu)
        model = model.cuda()
    else:
        device_id = [0]
        torch.cuda.set_device(device_id[0])
        model = nn.DataParallel(model, device_ids=device_id).cuda()
    joint_upper_range = joint_upper_range.cuda()
    joint_lower_range = joint_lower_range.cuda()


def test(model, img):
    model.eval()
    torch.set_grad_enabled(False)

    assert(img.shape == (input_size, input_size))
    img = img[np.newaxis, np.newaxis, ...]
    img = torch.Tensor(img)
    if args.cuda:
        img = img.cuda()

    # human part
    embedding_human, joint_human = model(img, is_human=True)
    joint_human = joint_human * (joint_upper_range - joint_lower_range) + joint_lower_range

    return joint_human.cpu().data.numpy()[0]


class Teleoperation():
    def __init__(self):
        self.mgi = moveit_commander.MoveGroupCommander("right_hand")
        self.bridge = CvBridge()
        self.mgi.set_named_target("open")
        self.mgi.go()

        # get next image after finish one pose
        self.online_once()

        rospy.spin()

    def online_once(self):
        while True:            
            img_data = rospy.wait_for_message("/camera/depth/image_raw", Image)
            rospy.loginfo("Got an image ^_^")
            try:
                img = self.bridge.imgmsg_to_cv2(img_data, desired_encoding="passthrough")
            except CvBridgeError as e:
                rospy.logerr(e)

            try:
                # preproces
                img = seg_hand_depth(img, 500, 1000, 10, 100, 4, 4, 250, True, 300)
                img = img.astype(np.float32)
                img = img / 255. * 2. - 1

                n = cv2.resize(img, (0, 0), fx=2, fy=2)
                n1 = cv2.normalize(n, n, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                cv2.imshow("segmented human hand", n1)
                cv2.waitKey(1)

                # get the clipped joints
                goal = self.joint_cal(img, isbio=True)
                start = self.mgi.get_current_joint_values()
                
                # collision check and manipulate
                csl_client = rospy.ServiceProxy('CheckSelfCollision', checkSelfCollision)
                try:
                    shadow_pos = csl_client(start, tuple(goal))
                    if shadow_pos.result:
                        rospy.loginfo("Move Done!")
                    else:
                       rospy.logwarn("Failed to move!")
                except rospy.ServiceException as exc:
                   rospy.logwarn("Service did not process request: " + str(exc))

                rospy.loginfo("Next one please ---->")
            except:
                rospy.loginfo("no images")


    def joint_cal(self, img, isbio=False):
        # start = rospy.Time.now().to_sec()

        # run the model
        feature = test(model, img)
        # network_time = rospy.Time.now().to_sec() - start
        # print("network_time is ", network_time)

        joint = [0.0, 0.0]
        joint += feature.tolist()
        if isbio:
            joint[5] = 0.3498509706185152
            joint[10] = 0.3498509706185152
            joint[14] = 0.3498509706185152
            joint[18] = 0.3498509706185152
            joint[23] = 0.3498509706185152

        # joints crop
        joint[2] = self.clip(joint[2], 0.349, -0.349)
        joint[3] = self.clip(joint[3], 1.57, 0)
        joint[4] = self.clip(joint[4], 1.57, 0)
        joint[5] = self.clip(joint[5], 1.57, 0)

        joint[6] = self.clip(joint[6], 0.785, 0)

        joint[7] = self.clip(joint[7], 0.349, -0.349)
        joint[8] = self.clip(joint[8], 1.57, 0)
        joint[9] = self.clip(joint[9], 1.57, 0)
        joint[10] = self.clip(joint[10], 1.57, 0)

        joint[11] = self.clip(joint[11], 0.349, -0.349)
        joint[12] = self.clip(joint[12], 1.57, 0)
        joint[13] = self.clip(joint[13], 1.57, 0)
        joint[14] = self.clip(joint[14], 1.57, 0)

        joint[15] = self.clip(joint[15], 0.349, -0.349)
        joint[16] = self.clip(joint[16], 1.57, 0)
        joint[17] = self.clip(joint[17], 1.57, 0)
        joint[18] = self.clip(joint[18], 1.57, 0)

        joint[19] = self.clip(joint[19], 1.047, -1.047)
        joint[20] = self.clip(joint[20], 1.222, 0)
        joint[21] = self.clip(joint[21], 0.209, -0.209)
        joint[22] = self.clip(joint[22], 0.524, -0.524)
        joint[23] = self.clip(joint[23], 1.57, 0)

        return joint


    def clip(self, x, maxv=None, minv=None):
        if maxv is not None and x > maxv:
            x = maxv
        if minv is not None and x < minv:
            x = minv
        return x


def main():
    rospy.init_node('human_teleop_shadow')
    while not rospy.is_shutdown():
        Teleoperation()
    rospy.spin()


if __name__ == "__main__":
    main()
