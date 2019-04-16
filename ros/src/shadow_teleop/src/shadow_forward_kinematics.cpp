#include <ros/ros.h>
#include <tf2_ros/transform_listener.h>
#include "tf2/transform_datatypes.h"

// important for tf2::Transform
#include "tf2_geometry_msgs/tf2_geometry_msgs.h"

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

int main(int argc, char** argv)
{
    ros::init(argc, argv, "shadow_forward_kinematics", 1);
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");
    ros::AsyncSpinner spinner(1);
    spinner.start();
    tf2_ros::Buffer tfbuffer;
    tf2_ros::TransformListener tf_listener(tfbuffer);

    std::string jointsfile_;
    std::string cartesian_pos_file_;
    std::string location_frame_;
    pnh.getParam("jointsfile", jointsfile_);
    pnh.getParam("cartesian_pos_file", cartesian_pos_file_);
    pnh.getParam("location_frame", location_frame_);

    robot_model_loader::RobotModelLoader robot_model_loader("robot_description");
    robot_model::RobotModelPtr kinematic_model = robot_model_loader.getModel();
    std::string base_frame = kinematic_model->getModelFrame();
    ROS_INFO("Model frame: %s", kinematic_model->getModelFrame().c_str());
    robot_state::RobotStatePtr kinematic_state(new robot_state::RobotState(kinematic_model));

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

    std::ifstream jointsfile(jointsfile_);
    std::ofstream cartesian_pos_file;
    cartesian_pos_file.open(cartesian_pos_file_,std::ios::app);
    std::string line,items,item;
    while(std::getline(jointsfile, line))
    {
        std::istringstream myline(line);
        std::vector<double> current_pos;

        while(std::getline(myline, items, ','))
        {
            if (items[0]=='i')
            {
                item = items;
                std::cout<< item <<std::endl;
                continue;
            }
            current_pos.push_back(std::stof(items));
        }
        kinematic_state->setVariablePositions(current_pos);
        kinematic_state->update();
        cartesian_pos_file << item << ',';

        for( auto& link : MapPositionlinks )
        {
            // std::cout<< link << ":" <<std::endl;
            const Eigen::Affine3d &link_state = kinematic_state->getGlobalLinkTransform(link);
            // ROS_INFO_STREAM("Translation: " << link_state.translation());
            // ROS_INFO_STREAM("Rotation: " << link_state.rotation()); //3*3
            tf2::Vector3 link_position(link_state.translation().x(), link_state.translation().y(), link_state.translation().z());

            if (location_frame_ == "wrist")
            {
                // transform position from base_frame into rh_wrist
                // use for cartesian_rhwrist_pos_file.csv  [0.000, -0.010, 0.213]
                tf2::Stamped<tf2::Vector3> stamped_in(link_position, ros::Time::now(), base_frame);
                tf2::Stamped<tf2::Vector3> stamped_out;
                geometry_msgs::TransformStamped wrist_base_transform = tfbuffer.lookupTransform("rh_wrist", base_frame, ros::Time::now(), ros::Duration(5.0));
                tf2::Stamped<tf2::Transform> trans;
                tf2::convert(wrist_base_transform, trans);
                stamped_out.setData(trans * stamped_in);
                stamped_out.stamp_ = trans.stamp_;
                stamped_out.frame_id_ = base_frame;
                link_position = stamped_out;
            }

            cartesian_pos_file << std::to_string( link_position.x() ) << ','
            << std::to_string( link_position.y() ) <<','
            << std::to_string( link_position.z() ) <<',';
            // << std::to_string( link_state.rotation()(0,0)) <<','
            // << std::to_string( link_state.rotation()(0,1)) <<','
            // << std::to_string( link_state.rotation()(0,2)) <<','
            // << std::to_string( link_state.rotation()(1,0)) <<','
            // << std::to_string( link_state.rotation()(1,1)) <<','
            // << std::to_string( link_state.rotation()(1,2)) <<','
            // << std::to_string( link_state.rotation()(2,0)) <<','
            // << std::to_string( link_state.rotation()(2,1)) <<','
            // << std::to_string( link_state.rotation()(2,2)) <<',';
        }
        cartesian_pos_file << std::endl;

    }
    cartesian_pos_file.close();
    ros::shutdown();
    return 0;
}
