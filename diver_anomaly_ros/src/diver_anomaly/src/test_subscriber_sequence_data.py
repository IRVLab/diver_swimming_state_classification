#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float64MultiArray
import numpy as np
from utils import get_numpy_from_2d_multiarray

def cb(msg):
    as_np = get_numpy_from_2d_multiarray(msg)

    print(f"The data after conversion is {as_np}")
    print(f"The data shape is {as_np.shape}")

rospy.Subscriber('/diver_anomaly_test_sequence/sequence', Float64MultiArray, cb)
rospy.init_node('test_subscriber')
while not rospy.is_shutdown():
    rospy.spin()
