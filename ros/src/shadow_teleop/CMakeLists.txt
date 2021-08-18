cmake_minimum_required(VERSION 2.8.12)
project(shadow_teleop)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  moveit_core
  moveit_ros_planning
  roscpp
  moveit_ros_move_group
  bio_ik
  moveit_ros_planning_interface
  cv_bridge
  gazebo_msgs
  trajectory_msgs
  message_generation
  bio_ik_msgs
)
find_package( OpenCV REQUIRED )

add_service_files(
  FILES
  fk.srv
  checkSelfCollision.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(CATKIN_DEPENDS bio_ik_msgs moveit_core moveit_ros_planning trajectory_msgs bio_ik moveit_ros_planning roscpp gazebo_msgs message_runtime)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(moveit_multi_bioik_dataset src/moveit_multi_bioik_dataset.cpp)
add_dependencies(moveit_multi_bioik_dataset ${catkin_EXPORTED_TARGETS})
target_link_libraries(moveit_multi_bioik_dataset ${catkin_LIBRARIES} ${OpenCV_LIBS} ${GAZEBO_LIBRARIES})

add_executable(shadow_forward_kinematics src/shadow_forward_kinematics.cpp)
add_dependencies(shadow_forward_kinematics ${catkin_EXPORTED_TARGETS})
target_link_libraries(shadow_forward_kinematics ${catkin_LIBRARIES} ${OpenCV_LIBS})

add_executable(shadow_fk_service src/shadow_fk_service.cpp)
add_dependencies(shadow_fk_service ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_gencpp)
target_link_libraries(shadow_fk_service ${catkin_LIBRARIES} ${OpenCV_LIBS})

add_executable(interpolate_traj_service src/interpolate_traj_service.cpp)
add_dependencies(interpolate_traj_service ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_gencpp)
target_link_libraries(interpolate_traj_service ${catkin_LIBRARIES} ${OpenCV_LIBS})

