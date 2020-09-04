from __future__ import print_function
import numpy as np
import rospy
import moveit_commander
from IPython import embed

if __name__ == "__main__":
    joint_upper_range = np.array([0.349, 1.571, 1.571, 1.571, 0.785, 0.349, 1.571, 1.571,
                                  1.571, 0.349, 1.571, 1.571, 1.571, 0.349, 1.571, 1.571,
                                  1.571, 1.047, 1.222, 0.209, 0.524, 1.571])
    joint_lower_range = np.array([-0.349, 0, 0, 0, 0, -0.349, 0, 0, 0, -0.349, 0, 0, 0,
                                  -0.349, 0, 0, 0, -1.047, 0, -0.209, -0.524, 0])
    rospy.init_node('human_teleop_shadow')
    mgi = moveit_commander.MoveGroupCommander("right_hand")
    DataFile = open("tams_joints.csv", "r")
    lines = DataFile.read().splitlines()
    for i in range(100):
        choice = np.random.randint(0, len(lines))
        ln = lines[choice]
        frame = ln.split(',')[0]
        print(frame)
        label_source = ln.split(',')[1:46]
        joint_human = np.array([float(ll) for ll in label_source])
        mgi.set_joint_value_target(joint_human)
        mgi.go()
        print("Enter to continue...")
