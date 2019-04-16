#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <tf2_ros/transform_listener.h>
#include "tf2/transform_datatypes.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.h"
#include <sensor_msgs/Image.h>

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
bool take_photo1 = false;
bool take_photo2 = false;
bool take_photo3 = false;
bool take_photo4 = false;
bool take_photo5 = false;
bool take_photo6 = false;
bool take_photo7 = false;
bool take_photo8 = false;
std::string item;
std::string depth_img_path_;
std::string depth_img_path1_;
std::string depth_img_path2_;
std::string depth_img_path3_;
std::string depth_img_path4_;
std::string depth_img_path5_;
std::string depth_img_path6_;
std::string depth_img_path7_;
std::string depth_img_path8_;
void depth_Callback(const sensor_msgs::Image::ConstPtr &image_data)
{
    if (take_photo)
    {
        cv_bridge::CvImagePtr cv_ptr;
        try
        {
            //cv_ptr = cv_bridge::toCvCopy(image_data,sensor_msgs::image_encodings::RGB8);
            if (image_data->encoding == "32FC1")
            {
                cv_ptr = cv_bridge::toCvCopy(image_data,sensor_msgs::image_encodings::TYPE_32FC1);
                cv::Mat image = cv_ptr->image;
                image.convertTo(image, CV_16UC1, 1000);
                cv::imwrite(depth_img_path_ + item, image);
                take_photo = false;
            }
        }
        catch (cv_bridge::Exception& e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }
    }
}

void depth_Callback1(const sensor_msgs::Image::ConstPtr &image_data)
{
    if (take_photo1)
    {
        cv_bridge::CvImagePtr cv_ptr;
        try
        {
            //cv_ptr = cv_bridge::toCvCopy(image_data,sensor_msgs::image_encodings::RGB8);
            if (image_data->encoding == "32FC1")
            {
                 cv_ptr = cv_bridge::toCvCopy(image_data,sensor_msgs::image_encodings::TYPE_32FC1);
                 cv::Mat image = cv_ptr->image;
            	 image.convertTo(image, CV_16UC1, 1000);
            	 cv::imwrite(depth_img_path1_ + item, image);
                 take_photo1 = false;
            }
        }
        catch (cv_bridge::Exception& e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }
    }
}

void depth_Callback2(const sensor_msgs::Image::ConstPtr &image_data)
{
    if (take_photo2)
    {
        cv_bridge::CvImagePtr cv_ptr;
        try
        {
            //cv_ptr = cv_bridge::toCvCopy(image_data,sensor_msgs::image_encodings::RGB8);
            if (image_data->encoding == "32FC1")
            {
                  cv_ptr = cv_bridge::toCvCopy(image_data,sensor_msgs::image_encodings::TYPE_32FC1);
                  cv::Mat image = cv_ptr->image;
                  image.convertTo(image, CV_16UC1, 1000);
                  cv::imwrite(depth_img_path2_ + item , image);
                  take_photo2 = false;
            }
        }
        catch (cv_bridge::Exception& e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }
    }
}

void depth_Callback3(const sensor_msgs::Image::ConstPtr &image_data)
{
    if (take_photo3)
    {
        cv_bridge::CvImagePtr cv_ptr;
        try
        {
            //cv_ptr = cv_bridge::toCvCopy(image_data,sensor_msgs::image_encodings::RGB8);
            if (image_data->encoding == "32FC1")
            {
                cv_ptr = cv_bridge::toCvCopy(image_data,sensor_msgs::image_encodings::TYPE_32FC1);
                cv::Mat image = cv_ptr->image;
                image.convertTo(image, CV_16UC1, 1000);
                cv::imwrite(depth_img_path3_ + item , image);
                take_photo3 = false;
            }
        }
        catch (cv_bridge::Exception& e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }
    }
}

void depth_Callback4(const sensor_msgs::Image::ConstPtr &image_data)
{
    if (take_photo4)
    {
        cv_bridge::CvImagePtr cv_ptr;
        try
        {
            //cv_ptr = cv_bridge::toCvCopy(image_data,sensor_msgs::image_encodings::RGB8);
            if (image_data->encoding == "32FC1")
            {
                cv_ptr = cv_bridge::toCvCopy(image_data,sensor_msgs::image_encodings::TYPE_32FC1);
                cv::Mat image = cv_ptr->image;
                image.convertTo(image, CV_16UC1, 1000);
                cv::imwrite(depth_img_path4_ + item , image);
                take_photo4 = false;
            }
        }
        catch (cv_bridge::Exception& e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }
    }
}

void depth_Callback5(const sensor_msgs::Image::ConstPtr &image_data)
{
    if (take_photo5)
    {
        cv_bridge::CvImagePtr cv_ptr;
        try
        {
            //cv_ptr = cv_bridge::toCvCopy(image_data,sensor_msgs::image_encodings::RGB8);
            if (image_data->encoding == "32FC1")
            {
                cv_ptr = cv_bridge::toCvCopy(image_data,sensor_msgs::image_encodings::TYPE_32FC1);
                cv::Mat image = cv_ptr->image;
                image.convertTo(image, CV_16UC1, 1000);
                cv::imwrite(depth_img_path5_ + item , image);
                take_photo5 = false;
            }
        }
        catch (cv_bridge::Exception& e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }
    }
}

void depth_Callback6(const sensor_msgs::Image::ConstPtr &image_data)
{
    if (take_photo6)
    {
        cv_bridge::CvImagePtr cv_ptr;
        try
        {
            //cv_ptr = cv_bridge::toCvCopy(image_data,sensor_msgs::image_encodings::RGB8);
            if (image_data->encoding == "32FC1")
            {
                cv_ptr = cv_bridge::toCvCopy(image_data,sensor_msgs::image_encodings::TYPE_32FC1);
                cv::Mat image = cv_ptr->image;
                image.convertTo(image, CV_16UC1, 1000);
                cv::imwrite(depth_img_path6_ + item , image);
                take_photo6 = false;
            }
        }
        catch (cv_bridge::Exception& e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }
    }
}

void depth_Callback7(const sensor_msgs::Image::ConstPtr &image_data)
{
    if (take_photo7)
    {
        cv_bridge::CvImagePtr cv_ptr;
        try
        {
            //cv_ptr = cv_bridge::toCvCopy(image_data,sensor_msgs::image_encodings::RGB8);
            if (image_data->encoding == "32FC1")
            {
                cv_ptr = cv_bridge::toCvCopy(image_data,sensor_msgs::image_encodings::TYPE_32FC1);
                cv::Mat image = cv_ptr->image;
                image.convertTo(image, CV_16UC1, 1000);
                cv::imwrite(depth_img_path7_ + item , image);
                take_photo7 = false;
            }
        }
        catch (cv_bridge::Exception& e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }
    }
}

void depth_Callback8(const sensor_msgs::Image::ConstPtr &image_data)
{
    if (take_photo8)
    {
        cv_bridge::CvImagePtr cv_ptr;
        try
        {
            //cv_ptr = cv_bridge::toCvCopy(image_data,sensor_msgs::image_encodings::RGB8);
            if (image_data->encoding == "32FC1")
            {
                cv_ptr = cv_bridge::toCvCopy(image_data,sensor_msgs::image_encodings::TYPE_32FC1);
                cv::Mat image = cv_ptr->image;
                image.convertTo(image, CV_16UC1, 1000);
                cv::imwrite(depth_img_path8_ + item , image);
                take_photo8 = false;
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
    ros::AsyncSpinner spinner(1);
    spinner.start();
    tf2_ros::Buffer tfbuffer;
    tf2_ros::TransformListener tf_listener(tfbuffer);

    std::string mapfile_;
    std::string jointsfile_;
    pnh.getParam("mapfile", mapfile_);
    pnh.getParam("depth_img_path", depth_img_path_);
    pnh.getParam("depth_img_path1", depth_img_path1_);
    pnh.getParam("depth_img_path2", depth_img_path2_);
    pnh.getParam("depth_img_path3", depth_img_path3_);
    pnh.getParam("depth_img_path4", depth_img_path4_);
    pnh.getParam("depth_img_path5", depth_img_path5_);
    pnh.getParam("depth_img_path6", depth_img_path6_);
    pnh.getParam("depth_img_path7", depth_img_path7_);
    pnh.getParam("depth_img_path8", depth_img_path8_);
    pnh.getParam("jointsfile", jointsfile_);

    ros::Subscriber sub_depth = nh.subscribe("/camera/depth/image_raw", 1, depth_Callback);
    ros::Subscriber sub_depth1 = nh.subscribe("/camera1/depth/image_raw", 1, depth_Callback1);
    ros::Subscriber sub_depth2 = nh.subscribe("/camera2/depth/image_raw", 1, depth_Callback2);
    ros::Subscriber sub_depth3 = nh.subscribe("/camera3/depth/image_raw", 1, depth_Callback3);
    ros::Subscriber sub_depth4 = nh.subscribe("/camera4/depth/image_raw", 1, depth_Callback4);
    ros::Subscriber sub_depth5 = nh.subscribe("/camera5/depth/image_raw", 1, depth_Callback5);
    ros::Subscriber sub_depth6 = nh.subscribe("/camera6/depth/image_raw", 1, depth_Callback6);
    ros::Subscriber sub_depth7 = nh.subscribe("/camera7/depth/image_raw", 1, depth_Callback7);
    ros::Subscriber sub_depth8 = nh.subscribe("/camera8/depth/image_raw", 1, depth_Callback8);

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

    ROS_WARN_STREAM("move to 'open' pose");
    mgi.setNamedTarget("open");
    mgi.move();

    double timeout = 0.2;
    int attempts = 5;
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
    std::vector<std::string> MapDirectionlinks1 {
      "rh_thproximal",
      "rh_ffproximal",
      "rh_mfproximal",
      "rh_rfproximal",
      "rh_lfproximal",
    };
    std::vector<std::string> MapDirectionlinks2 {
      "rh_thmiddle"
    };
    std::vector <float> MapPositionweights {1,1,1,1,1,0.2,0.2,0.2,0.2,0.2};
    std::vector <float> MapDirectionweights1{0.1,0.1,0.1,0.1,0.1};
    std::vector <float> MapDirectionweights2{0.1};

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
        geometry_msgs::TransformStamped base_wrist_transform = tfbuffer.lookupTransform(base_frame, "rh_wrist", ros::Time::now(), ros::Duration(5.0));
        tf2::Stamped<tf2::Transform> trans;
        tf2::convert(base_wrist_transform, trans);

        for (int j = 0; j< MapPositionlinks.size(); j++)
        {
            int t = j * 3;
            tf2::Vector3 position = tf2::Vector3(csvItem[t], csvItem[t+1], csvItem[t+2]);
            // transform position from current rh_wrist into base_frame
            tf2::Stamped<tf2::Vector3> stamped_in(position, ros::Time::now(), "rh_wrist");
            tf2::Stamped<tf2::Vector3>  stamped_out;
            stamped_out.setData(trans * stamped_in);
            stamped_out.stamp_ = trans.stamp_;
            stamped_out.frame_id_ = "rh_wrist";
            tf2::Vector3 Mapposition = stamped_out;
            Mapposition.setZ(Mapposition.z() + 0.04);

            ik_options.goals.emplace_back(new bio_ik::PositionGoal(MapPositionlinks[j], Mapposition, MapPositionweights[j]));
        }

        for (int j = 0; j< MapDirectionlinks1.size(); j++)
        {
            int t = 30 + j * 3;
            tf2::Vector3 proximal_direction = (tf2::Vector3(csvItem[t], csvItem[t+1], csvItem[t+2])).normalized();

            // transform position from current rh_wrist into base_frame
            tf2::Stamped<tf2::Vector3> stamped_in(proximal_direction, ros::Time::now(), "rh_wrist");
            tf2::Stamped<tf2::Vector3> stamped_out;
            tf2::Vector3 end = stamped_in;
            tf2::Vector3 origin = tf2::Vector3(0,0,0);
            tf2::Vector3 output = (trans * end) - (trans * origin);
            stamped_out.setData( output);
            stamped_out.stamp_ = trans.stamp_;
            stamped_out.frame_id_ = "rh_wrist";

            tf2::Vector3 Mapdirection = stamped_out;
            ik_options.goals.emplace_back(new bio_ik::DirectionGoal(MapDirectionlinks1[j], tf2::Vector3(0,0,1), Mapdirection.normalized(), MapDirectionweights1[j]));
        }

        for (int j = 0; j< MapDirectionlinks2.size(); j++)
        {
            int t = 45 + j*3;
            tf2::Vector3 dummy_direction = (tf2::Vector3(csvItem[t], csvItem[t+1], csvItem[t+2])).normalized();

            // transform position from current rh_wrist into base_frame
            tf2::Stamped<tf2::Vector3> stamped_in(dummy_direction, ros::Time::now(), "rh_wrist");
            tf2::Stamped<tf2::Vector3> stamped_out;
            tf2::Vector3 end = stamped_in;
            tf2::Vector3 origin = tf2::Vector3(0,0,0);
            tf2::Vector3 output = (trans * end) - (trans * origin);
            stamped_out.setData(output);
            stamped_out.stamp_ = trans.stamp_;
            stamped_out.frame_id_ = "rh_wrist";

            tf2::Vector3 Mapdirection = stamped_out;

            ik_options.goals.emplace_back(new bio_ik::DirectionGoal(MapDirectionlinks2[j], tf2::Vector3(0,0,1), Mapdirection.normalized(), MapDirectionweights2[j]));
        }

        robot_state = current_state;
        // set ik solver
        bool found_ik =robot_state.setFromIK(
                          joint_model_group,           // active Shadow joints
                          EigenSTL::vector_Affine3d(), // no explicit poses here
                          std::vector<std::string>(),
                          attempts, timeout,
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
                                   EigenSTL::vector_Affine3d(), // no explicit poses here
                                   std::vector<std::string>(),
                                   5, timeout + 0.1,
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

            std::cout << "Moved to " << item <<". Take photo now. ";
            // ros::Duration(1).sleep();
            take_photo = true;
            take_photo1 = true;
            take_photo2 = true;
            take_photo3 = true;
            take_photo4 = true;
            take_photo5 = true;
            take_photo6 = true;
            take_photo7 = true;
            take_photo8 = true;

            // can not move robot when taking photoes.
            while (take_photo || take_photo1 || take_photo2 || take_photo3 || take_photo4 || take_photo5 || take_photo6 || take_photo7 || take_photo8 )
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
