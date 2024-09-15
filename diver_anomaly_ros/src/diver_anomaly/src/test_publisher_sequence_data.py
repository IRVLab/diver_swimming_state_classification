#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from utils import numpy_2d_to_msg
import cv2

#img = cv2.imread('/home/dtkutzke/ros_workspaces/diver_anomaly_ros/src/diver_anomaly/src/test_image_publication.png')
img = cv2.imread('/mnt/sdcard/dtkutzke/diver_anomaly_ros/src/diver_anomaly/src/test_image_publication.png')
# cv2.imshow("image", img)
# cv2.waitKey(0)

bridge = CvBridge()

N_frames = 4
N_features = 3

test_pub = rospy.Publisher('/diver_anomaly_test/sequence', Float64MultiArray, queue_size=15)
test_img_pub = rospy.Publisher('/diver_anomaly_test/img', Image, queue_size=10)
rospy.init_node('test_publisher', anonymous=True)
rate = rospy.Rate(2)
while not rospy.is_shutdown():
    test_data = np.random.rand(N_frames, N_features)
    msg = numpy_2d_to_msg(test_data)
    test_pub.publish(msg)
    rospy.loginfo('Publishing test sequence data...')
    test_img_pub.publish(bridge.cv2_to_imgmsg(img, encoding="passthrough"))
