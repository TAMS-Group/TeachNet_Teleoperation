import rospy
import moveit_commander
from std_msgs.msg import Float64MultiArray
from shadow_teleop.srv import *

def main():
    rospy.init_node('shadow_safe_mode_teleop')

    mgi = moveit_commander.MoveGroupCommander("right_hand")
    rospy.sleep(1)

    while not rospy.is_shutdown():
        joints_msg = rospy.wait_for_message("/teleop_outputs_joints", Float64MultiArray)
        goal = joints_msg.data
        start = mgi.get_current_joint_values()

        # collision check and manipulate
        csl_client = rospy.ServiceProxy('CheckSelfCollision', checkSelfCollision)
        try:
            shadow_pos = csl_client(start, goal)
            if shadow_pos.result:
                rospy.loginfo("Move Done!")
            else:
               rospy.logwarn("Failed to move!")
        except rospy.ServiceException as exc:
           rospy.logwarn("Service did not process request: " + str(exc))

        rospy.loginfo("Next one please ---->")

if __name__ == "__main__":
    main()
