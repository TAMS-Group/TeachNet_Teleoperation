#include <ros/ros.h>

#include <moveit/planning_scene/planning_scene.h>
#include <moveit/robot_model_loader/robot_model_loader.h>

#include <moveit/robot_trajectory/robot_trajectory.h>
#include <moveit/trajectory_processing/iterative_time_parameterization.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/collision_detection/collision_common.h>

#include <shadow_teleop/checkSelfCollision.h>

#include <moveit/collision_detection_fcl/collision_common.h>


class CheckSelfCollision{
private:
  int STEPS;
  ros::NodeHandle nh;
  ros::ServiceServer service;
  robot_model_loader::RobotModelLoader rml;
  robot_model::RobotModelPtr robot_model;
  moveit::planning_interface::MoveGroupInterface mgi;

public:
  CheckSelfCollision(): mgi("right_hand")
  {
      service = nh.advertiseService("CheckSelfCollision", &CheckSelfCollision::check_service, this);
      robot_model = rml.getModel();
  }

  bool check_service(shadow_teleop::checkSelfCollision::Request &req, shadow_teleop::checkSelfCollision::Response &res)
  {
      planning_scene::PlanningScene scene(robot_model);
      moveit::core::RobotState last_robot_state(scene.getCurrentState());
      last_robot_state.setVariablePositions(req.start);

      moveit::core::RobotState robot_state(scene.getCurrentState());
      robot_state.setVariablePositions(req.goal);
      robot_trajectory::RobotTrajectory trajectory(robot_model, "right_hand");
      moveit::core::RobotState interpolated_robot_state(scene.getCurrentState());

      // ros::Time begin = ros::Time::now();
      // std::cout << "start check collision"  << std::endl;
      STEPS = robot_state.distance(last_robot_state) / 0.2;
      for(int t= 0; t < STEPS + 1; ++t){
          if (STEPS == 0)
              interpolated_robot_state = robot_state;
          else
              last_robot_state.interpolate(robot_state, t*(1.0/STEPS), interpolated_robot_state);

          interpolated_robot_state.update();

          collision_detection::CollisionRequest collision_request;
          collision_request.contacts = true;
          collision_request.verbose = false;
          collision_detection::CollisionResult collision_result;
          interpolated_robot_state.updateCollisionBodyTransforms();
          scene.checkSelfCollision(collision_request, collision_result,
                                             interpolated_robot_state);
          if (collision_result.collision) {
            ROS_ERROR("collision, cannot manipulate this pose");
            return 0;
          }
          trajectory.addSuffixWayPoint(interpolated_robot_state, 0.01);
      }
      // ros::Duration dur = ros::Time::now() - begin;
      ROS_INFO("No Collsion ^_^");

      trajectory_processing::IterativeParabolicTimeParameterization iptp;
      iptp.computeTimeStamps(trajectory);

      moveit_msgs::RobotTrajectory trajectory_msg;
      trajectory.getRobotTrajectoryMsg(trajectory_msg);
//      ROS_INFO_STREAM("joint space interpolation produces a valid trajectory: " << trajectory_msg);

      moveit::planning_interface::MoveGroupInterface::Plan plan;
      plan.trajectory_= trajectory_msg;
      res.result = static_cast<bool>(mgi.execute(plan));

      return true;
  }

  ~CheckSelfCollision(){}
};

int main(int argc, char** argv){
  	ros::init(argc, argv, "checkSelfCollision_node");
    ros::AsyncSpinner spinner(2);
    spinner.start();

    CheckSelfCollision csc;
    ros::waitForShutdown();
  	return 0;
}
