import rospy
import moveit_commander
from sr_robot_commander.sr_hand_commander import SrHandCommander
from std_msgs.msg import Float64MultiArray

def main():
    rospy.init_node('shadow_unsafe_mode_teleop')
    hand_commander = SrHandCommander(name="right_hand")

    #  start pos
    start_play_pos = {"rh_THJ1": 20, "rh_THJ2": 10, "rh_THJ3": 0, "rh_THJ4": 0, "rh_THJ5":0,
			  "rh_FFJ1": 45, "rh_FFJ2": 80, "rh_FFJ3": 0, "rh_FFJ4": 0,
			  "rh_MFJ1": 45, "rh_MFJ2": 80, "rh_MFJ3": 0, "rh_MFJ4": 0,
			  "rh_RFJ1": 45, "rh_RFJ2": 80, "rh_RFJ3": 0, "rh_RFJ4": 0,
			  "rh_LFJ1": 45, "rh_LFJ2": 80, "rh_LFJ3": 0, "rh_LFJ4": 0, "rh_LFJ5": 0,
			 "rh_WRJ1": -30, "rh_WRJ2": 0}

    hand_commander.move_to_joint_value_target_unsafe(start_play_pos, 1.5, False, angle_degrees=True)
    rospy.sleep(1)

    while not rospy.is_shutdown():
        joints_msg = rospy.wait_for_message("/teleop_outputs_joints", Float64MultiArray)
        goal = joints_msg.data

        # bug:get_joints_position() return radian joints
        hand_pos = hand_commander.get_joints_position()

        # first finger
        hand_pos.update({"rh_FFJ3": goal[3]})
        hand_pos.update({"rh_FFJ2": goal[4]})
        hand_pos.update({"rh_FFJ4": goal[2]})

        # middle finger
        hand_pos.update({"rh_MFJ3": goal[12]})
        hand_pos.update({"rh_MFJ2": goal[13]})

        # ring finger
        hand_pos.update({"rh_RFJ3": goal[16]})
        hand_pos.update({"rh_RFJ2": goal[17]})

        # little finger
        hand_pos.update({"rh_LFJ3": goal[8]})
        hand_pos.update({"rh_LFJ2": goal[9]})

        # thumb
        hand_pos.update({"rh_THJ5": goal[19]})
        hand_pos.update({"rh_THJ4": goal[20]})
        hand_pos.update({"rh_THJ3": goal[21]})
        hand_pos.update({"rh_THJ2": goal[22]})

        self.hand_commander.move_to_joint_value_target_unsafe(hand_pos, 0.3, False, angle_degrees=False)
        rospy.loginfo("Next one please ---->")


if __name__ == "__main__":
    main()
