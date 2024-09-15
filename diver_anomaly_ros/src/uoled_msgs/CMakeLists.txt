cmake_minimum_required(VERSION 2.8.3)
project(uoled_msgs)

set(CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS} -g -Wall -O2 -Wno-write-strings") # TODO: remove -Wno-write-string

find_package(catkin REQUIRED COMPONENTS roscpp std_msgs sensor_msgs message_generation rospy)

add_message_files(
  FILES
  TagFeedback.msg
  BBox.msg
  StatusIndicator.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

#catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES aquatest
#  CATKIN_DEPENDS roscpp std_msgs sensor_msgs image_transport cv_bridge message_runtime
#  DEPENDS system_lib
#)
