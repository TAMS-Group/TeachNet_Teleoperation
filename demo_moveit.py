import rospy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
import copy

def main():
    rospy.init_node('shadow_moveit_demo')
    joint_pub = rospy.Publisher('joint_states', JointState, queue_size=10)

    while not rospy.is_shutdown():
        joints_msg = rospy.wait_for_message("/teleop_outputs_joints", Float64MultiArray)
        goal = list(joints_msg.data)
        a = copy.deepcopy(goal)
        a[6:10] = goal[11:15]
        a[10:14] = goal[15:19]
        a[14:19] = goal[6:11]
        print(a[-5], a[-4], a[-3], a[-2])
        msg = JointState()
        msg.name = ["rh_WRJ2", "rh_WRJ1", "rh_FFJ4", "rh_FFJ3", "rh_FFJ2", "rh_FFJ1", "rh_MFJ4", "rh_MFJ3", "rh_MFJ2",
                  "rh_MFJ1", "rh_RFJ4", "rh_RFJ3", "rh_RFJ2", "rh_RFJ1", "rh_LFJ5", "rh_LFJ4", "rh_LFJ3", "rh_LFJ2",
                  "rh_LFJ1", "rh_THJ5", "rh_THJ4", "rh_THJ3", "rh_THJ2", "rh_THJ1"]
        msg.position = a
        joint_pub.publish(msg)
        rospy.loginfo("Next one please ---->")

if __name__ == "__main__":
    main()
