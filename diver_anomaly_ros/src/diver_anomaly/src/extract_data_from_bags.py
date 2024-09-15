#!/usr/bin/env python3

import pandas as pd
import rosbag

bag = rosbag.Bag('test.bag')
for topic, msg, t in bag.read_messages(topics=['chatter', 'numbers']):
    print(msg)
bag.close()

# replace the topic name as per your need
camera = b.message_by_topic('/camera_front_left/image_raw')
pose_2d = b.message_by_topic('/pose/pose_2d_vis_topic')
pose_3d = b.message_by_topic('/pose/pose_3d_vis_topic')
anomaly = b.message_by_topic('/diver_anomaly/anomaly_img')


# LASER_MSG
df_laser = pd.read_csv(LASER_MSG)
df_laser # prints laser data in the form of pandas dataframe