#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/LinearMath/Vector3.h>
#include <tf2/LinearMath/Scalar.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <sensor_msgs/Image.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/planning_pipeline/planning_pipeline.h>
#include <moveit_msgs/RobotState.h>
#include <moveit/kinematics_base/kinematics_base.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/robot_state/conversions.h>

#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit/planning_scene/planning_scene.h>

#include <moveit/collision_detection/collision_matrix.h>

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <functional>
#include <memory>
#include <sstream>
#include <sys/stat.h>

#include <bio_ik/bio_ik.h>
#include "collision_free_goal.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

bool take_photo = false;
std::string item;
std::vector<std::string> depth_img_path_;


void callback(const sensor_msgs::Image::ConstPtr &image_data1, const sensor_msgs::Image::ConstPtr &image_data2,
              const sensor_msgs::Image::ConstPtr &image_data3, const sensor_msgs::Image::ConstPtr &image_data4,
              const sensor_msgs::Image::ConstPtr &image_data5, const sensor_msgs::Image::ConstPtr &image_data6,
              const sensor_msgs::Image::ConstPtr &image_data7, const sensor_msgs::Image::ConstPtr &image_data8,
              const sensor_msgs::Image::ConstPtr &image_data9)
{
    if (take_photo)
    {
        cv_bridge::CvImagePtr cv_ptr;
        try
        {
            std::vector<sensor_msgs::Image::ConstPtr> images = {image_data1, image_data2, image_data3, image_data4,
                    image_data5, image_data6, image_data7, image_data8, image_data9};
            //cv_ptr = cv_bridge::toCvCopy(image_data,sensor_msgs::image_encodings::RGB8);
            if (image_data1->encoding == "32FC1")
            {
                for (int i=0; i<9; i++)
                {
                    cv_ptr = cv_bridge::toCvCopy(images[i],sensor_msgs::image_encodings::TYPE_32FC1);
                    cv::Mat image = cv_ptr->image;
                    image.convertTo(image, CV_16UC1, 1000);
                    cv::imwrite(depth_img_path_[i] + item, image);
                }
                take_photo = false;
                std::cout << "==============> save images finish"<< std::endl;
            }
        }
        catch (cv_bridge::Exception& e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }
    }
}


int main(int argc, char** argv)
{
    ros::init(argc, argv, "bio_ik_human_robot_mapping", 1);
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");
    ros::AsyncSpinner spinner(5);
    spinner.start();
    tf2_ros::Buffer tfBuffer;
    tf2_ros::TransformListener tfListener(tfBuffer);

    std::string mapfile_;
    std::string jointsfile_;
    pnh.getParam("mapfile", mapfile_);
    pnh.getParam("jointsfile", jointsfile_);

    std::string depth_img_path;
    for (int i=1; i<10; i++){
        pnh.getParam("depth_img_path" + std::to_string(i), depth_img_path);
        depth_img_path_.push_back(depth_img_path);
    }

    std::string depth_topic1 = "/camera1/depth/image_raw";
    std::string depth_topic2 = "/camera2/depth/image_raw";
    std::string depth_topic3 = "/camera3/depth/image_raw";
    std::string depth_topic4 = "/camera4/depth/image_raw";
    std::string depth_topic5 = "/camera5/depth/image_raw";
    std::string depth_topic6 = "/camera6/depth/image_raw";
    std::string depth_topic7 = "/camera7/depth/image_raw";
    std::string depth_topic8 = "/camera8/depth/image_raw";
    std::string depth_topic9 = "/camera9/depth/image_raw";

    message_filters::Subscriber<sensor_msgs::Image> image1_sub(nh, depth_topic1, 1);
    message_filters::Subscriber<sensor_msgs::Image> image2_sub(nh, depth_topic2, 1);
    message_filters::Subscriber<sensor_msgs::Image> image3_sub(nh, depth_topic3, 1);
    message_filters::Subscriber<sensor_msgs::Image> image4_sub(nh, depth_topic4, 1);
    message_filters::Subscriber<sensor_msgs::Image> image5_sub(nh, depth_topic5, 1);
    message_filters::Subscriber<sensor_msgs::Image> image6_sub(nh, depth_topic6, 1);
    message_filters::Subscriber<sensor_msgs::Image> image7_sub(nh, depth_topic7, 1);
    message_filters::Subscriber<sensor_msgs::Image> image8_sub(nh, depth_topic8, 1);
    message_filters::Subscriber<sensor_msgs::Image> image9_sub(nh, depth_topic9, 1);

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image,
                                            sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image> MySyncPolicy;

    message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), image1_sub, image2_sub, image3_sub, image4_sub, image5_sub, image6_sub, image7_sub, image8_sub, image9_sub);
    sync.registerCallback(boost::bind(&callback, _1, _2, _3, _4, _5, _6, _7, _8, _9));

    std::string group_name = "right_hand";
    moveit::planning_interface::MoveGroupInterface mgi(group_name);
    std::string base_frame = mgi.getPoseReferenceFrame();

    mgi.setGoalTolerance(0.01);
    mgi.setPlanningTime(25);
    mgi.setPlannerId("RRTConnectkConfigDefault");
    auto robot_model = mgi.getCurrentState()->getRobotModel();
    auto joint_model_group = robot_model->getJointModelGroup(group_name);
    moveit::core::RobotState robot_state(robot_model);
    planning_scene::PlanningScene planning_scene(robot_model);
    robot_state::RobotState& current_state = planning_scene.getCurrentStateNonConst();

    collision_detection::CollisionRequest collision_request;
    collision_detection::CollisionResult collision_result;
    std::vector<std::string> collision_pairs;

//    ROS_WARN_STREAM("move to 'open' pose");
//    mgi.setNamedTarget("open");
//    mgi.move();

    double timeout = 0.2;
    std::vector<std::string> MapPositionlinks {
      "rh_thtip",
      "rh_fftip",
      "rh_mftip",
      "rh_rftip",
      "rh_lftip",
      "rh_thmiddle",
      "rh_ffmiddle",
      "rh_mfmiddle",
      "rh_rfmiddle",
      "rh_lfmiddle"
    };
    std::vector<std::string> MapDirectionlinks {
      "rh_thproximal",
      "rh_ffproximal",
      "rh_mfproximal",
      "rh_rfproximal",
      "rh_lfproximal",
      "rh_thmiddle"
    };

    std::vector <float> MapPositionweights {1,1,1,1,1,0.2,0.2,0.2,0.2,0.2};
    std::vector <float> MapDirectionweights{0.1,0.1,0.1,0.1,0.1, 0.1};

    std::ifstream mapfile(mapfile_);
    std::string line, items;
    while(std::getline(mapfile, line)){
        ros::Time begin = ros::Time::now();
        // track goals using bio ik
        bio_ik::BioIKKinematicsQueryOptions ik_options;
        ik_options.replace = true;
        ik_options.return_approximate_solution = true;

        std::istringstream myline(line);
        std::vector<double> csvItem;
        while(std::getline(myline, items, ','))
        {
            if (items[0]=='i')
            {
                item = items;
                std::cout<< item <<std::endl;
                continue;
            }
            csvItem.push_back(std::stof(items));
        }

        for (int j = 0; j< MapPositionlinks.size(); j++)
        {
            int t = j * 3;

            // transform position from current rh_wrist into base_frame
            geometry_msgs::PointStamped stamped_in;
            stamped_in.header.frame_id = "rh_wrist";
            stamped_in.point.x = csvItem[t];
            stamped_in.point.y = csvItem[t+1];
            stamped_in.point.z = csvItem[t+2];

            geometry_msgs::PointStamped stamped_out;
            tfBuffer.transform(stamped_in, stamped_out, base_frame);
            tf2::Vector3 Mapposition (stamped_out.point.x, stamped_out.point.y, stamped_out.point.z);

            ik_options.goals.emplace_back(new bio_ik::PositionGoal(MapPositionlinks[j], Mapposition, MapPositionweights[j]));
        }

        for (int j = 0; j< MapDirectionlinks.size(); j++)
        {
            int t = 30 + j * 3;

            geometry_msgs::PointStamped stamped_in;
            stamped_in.header.frame_id = "rh_wrist";
            stamped_in.point.x = csvItem[t];
            stamped_in.point.y = csvItem[t+1];
            stamped_in.point.z = csvItem[t+2];

            geometry_msgs::PointStamped stamped_out;
            tfBuffer.transform(stamped_in, stamped_out, base_frame);
            tf2::Vector3 Mapdirection (stamped_out.point.x, stamped_out.point.y, stamped_out.point.z);

            ik_options.goals.emplace_back(new bio_ik::DirectionGoal(MapDirectionlinks[j], tf2::Vector3(0,0,1), Mapdirection.normalized(), MapDirectionweights[j]));
        }

        robot_state = current_state;
        // set ik solver
        bool found_ik =robot_state.setFromIK(
                          joint_model_group,           // active Shadow joints
                          EigenSTL::vector_Isometry3d(), // no explicit poses here
                          std::vector<std::string>(),
                          timeout,
                          moveit::core::GroupStateValidityCallbackFn(),
                          ik_options
                        );
        // move to the solution position
        std::vector<double> joint_values;
        moveit::planning_interface::MoveGroupInterface::Plan shadow_plan;
        if (found_ik)
        {
            robot_state.copyJointGroupPositions(joint_model_group, joint_values);
            // set the angle of two wrist joint zero
            joint_values[0] = 0;
            joint_values[1] = 0;
            mgi.setJointValueTarget(joint_values);

            // get collision pairs then use collision free ik
            collision_request.contacts = true;
            collision_request.max_contacts = 1000;
            collision_result.clear();

            current_state = mgi.getJointValueTarget();
            planning_scene.checkSelfCollision(collision_request, collision_result);
            collision_detection::CollisionResult::ContactMap::const_iterator it;
            collision_pairs.clear();
            for(it = collision_result.contacts.begin();	it != collision_result.contacts.end(); ++it)
            {
                collision_pairs.push_back(it->first.first.c_str());
                collision_pairs.push_back(it->first.second.c_str());
                ROS_WARN("Collision between: %s and %s, need to reIK", it->first.first.c_str(), it->first.second.c_str());
            }

            if (collision_pairs.size() > 0 )
            {
                 // self_collision_free goal
                 double collision_weight = 1;
                 ik_options.goals.emplace_back(new Collision_freeGoal(collision_pairs, collision_weight));

                 // set ik solver again
                 bool refound_ik =robot_state.setFromIK(
                                   joint_model_group,           // active Shadow joints
                                   EigenSTL::vector_Isometry3d(), // no explicit poses here
                                   std::vector<std::string>(),
                                   timeout+0.1,
                                   moveit::core::GroupStateValidityCallbackFn(),
                                   ik_options
                                 );

                 if (refound_ik)
                 {
                    robot_state.copyJointGroupPositions(joint_model_group, joint_values);
                    // set the angle of two wrist joint zero
                    joint_values[0] = 0;
                    joint_values[1] = 0;
                    mgi.setJointValueTarget(joint_values);

                    // get collision pairs then use collision free ik
                    collision_request.contacts = true;
                    collision_request.max_contacts = 1000;
                    collision_result.clear();

                    current_state = mgi.getJointValueTarget();
                    planning_scene.checkSelfCollision(collision_request, collision_result);
                    if (collision_result.contacts.size() > 0)
                    {
                        ROS_ERROR("Failed to get collision_free result, skip to next one");
                        continue;
                    }
                }
                else
                {
                    std::cout << "Did not find reIK solution" << std::endl;
                    continue;
                }
            }

            if (!(static_cast<bool>(mgi.plan(shadow_plan))))
            {
                std::cout<< "Failed to plan pose " << item << std::endl;
                continue;
            }

            if(!(static_cast<bool>(mgi.execute(shadow_plan))))
            {
                std::cout << "Failed to execute pose " << item<< std::endl;
                continue;
            }

            std::cout << "Moved to " << item <<". Take photo now. " << std::endl;
            // ros::Duration(1).sleep();
            take_photo = true;

            // can not move robot when taking photoes.
            while (take_photo)
                ros::Duration(0.1).sleep();

            // save joint angles
            std::ofstream joints_file;
            joints_file.open(jointsfile_,std::ios::app);
            joints_file << item << ',' << std::to_string( joint_values[0]) << ',' << std::to_string( joint_values[1]) <<','
            << std::to_string( joint_values[2]) <<',' << std::to_string( joint_values[3]) <<',' << std::to_string( joint_values[4]) <<','
            << std::to_string( joint_values[5]) <<',' << std::to_string( joint_values[6]) <<',' << std::to_string( joint_values[7]) <<','
            << std::to_string( joint_values[8]) <<',' << std::to_string( joint_values[9]) <<',' << std::to_string( joint_values[10]) <<','
            << std::to_string( joint_values[11]) <<',' << std::to_string( joint_values[12]) <<',' << std::to_string( joint_values[13]) <<','
            << std::to_string( joint_values[14]) <<',' << std::to_string( joint_values[15]) <<',' << std::to_string( joint_values[16]) <<','
            << std::to_string( joint_values[17]) <<',' << std::to_string( joint_values[18]) <<',' << std::to_string( joint_values[19]) <<','
            << std::to_string( joint_values[20]) <<',' << std::to_string( joint_values[21]) <<',' << std::to_string( joint_values[22]) <<','
            << std::to_string( joint_values[23]) << std::endl;
            joints_file.close();

            ros::Duration dur = ros::Time::now() - begin;
            std::cout << "running time is "  << dur <<  std::endl;
        }
        else
        {
            std::cout << "Did not find IK solution" << std::endl;
        }

        for (int j = 0; j <ik_options.goals.size();j++)
            ik_options.goals[j].reset();
    }

    ros::shutdown();
    return 0;
}
