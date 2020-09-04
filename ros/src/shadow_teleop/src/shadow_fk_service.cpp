#include <ros/ros.h>

#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_state/robot_state.h>

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <functional>
#include <memory>
#include <sstream>

#include <shadow_teleop/fk.h>

std::vector<std::string> MapPositionlinks {
    "rh_wrist",
    "rh_thtip",
    "rh_fftip",
    "rh_mftip",
    "rh_rftip",
    "rh_lftip",
    "rh_thmiddle",
    "rh_ffmiddle",
    "rh_mfmiddle",
    "rh_rfmiddle",
    "rh_lfmiddle",
    "rh_thproximal",
    "rh_ffproximal",
    "rh_mfproximal",
    "rh_rfproximal",
    "rh_lfproximal"
};

bool shadow_fk(shadow_teleop::fk::Request &req, shadow_teleop::fk::Response &res)
{
    robot_model_loader::RobotModelLoader robot_model_loader("robot_description");
    robot_model::RobotModelPtr kinematic_model = robot_model_loader.getModel();
    robot_state::RobotStatePtr kinematic_state(new robot_state::RobotState(kinematic_model));

    std::vector<double> current_joint;
    current_joint = req.joints;
    kinematic_state->setVariablePositions(current_joint);
    kinematic_state->update();
    
    std::vector<double> current_pos;
    for( auto& link : MapPositionlinks )
    {
        const Eigen::Affine3d &link_state = kinematic_state->getGlobalLinkTransform(link);
        ROS_INFO_STREAM("Translation: " << link_state.translation());
        current_pos.push_back(link_state.translation().x());
        current_pos.push_back(link_state.translation().y());
        current_pos.push_back(link_state.translation().z());
    }
    res.pos = current_pos;
    return true;
}


int main(int argc, char** argv)
{
    ros::init(argc, argv, "shadow_forward_kinematics", 1);
    ros::NodeHandle nh;

    ros::ServiceServer service = nh.advertiseService("FK", shadow_fk);
    ros::spin();
    return 0;
}


