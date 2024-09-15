#!/bin/bash

# This is for running diver_anomaly_ros specifically

# Aqua
#export ROS_MASTER_URI=http://vision:11311
#export ROS_HOSTNAME=192.168.2.73

sudo chmod +777 /home/dtkutzke/.local
pip3 install onnxruntime_gpu
pip3 install aeon

cd /home/dtkutzke/ros_workspaces/diver_anomaly_ros/
source ./devel/setup.bash
python3 /home/dtkutzke/ros_workspaces/diver_anomaly_ros/src/diver_anomaly/src/verify_working_environment.py
