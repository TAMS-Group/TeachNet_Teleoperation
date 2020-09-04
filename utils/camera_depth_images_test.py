from __future__ import print_function
import os
import time
import pickle
import glob
import numpy as np
import cv2
import csv
import copy
from utils import seg_hand_depth

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

np.random.seed(int(time.time()))


def main():
    rospy.init_node('human_teleop_shadow')
    bridge = CvBridge()
    while not rospy.is_shutdown():
        while True:
            #       /camera/aligned_depth_to_color/image_raw
            img_data = rospy.wait_for_message("/camera/aligned_depth_to_color/image_raw", Image)
            rospy.loginfo("Got an image ^_^")
            try:
                img = bridge.imgmsg_to_cv2(img_data, desired_encoding="passthrough")
            except CvBridgeError as e:
                rospy.logerr(e)
            # preproces
            img = seg_hand_depth(img, 500, 1000, 10, 100, 4, 4, 250, True, 300)
            img = img.astype(np.float32)
            img = img / 255. * 2. - 1

            n = cv2.resize(img, (0, 0), fx=2, fy=2)
            n1 = cv2.normalize(n, n, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            cv2.imshow("segmented human hand", n1)
            cv2.waitKey(1)


if __name__ == "__main__":
    main()
