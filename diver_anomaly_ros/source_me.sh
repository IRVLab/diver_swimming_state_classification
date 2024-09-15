#!/bin/bash

./post_install.sh

source /opt/ros/noetic/setup.bash
source /data/dtkutzke/diver_anomaly_ros/devel/setup.bash

export PYTHONPATH=$PYTHONPATH:/data/dtkutzke/diver_anomaly_ros/src/diver_anomaly/src/anomaly_detection_hack

# These are for Aqua AUV
# export ROS_MASTER_URI=http://192.168.210.44:11311
# export ROS_HOSTNAME=192.168.210.44
export ROS_MASTER_URI=http://127.0.0.1:11311
export ROS_HOSTNAME=127.0.0.1

python3 verify_working_install.py


