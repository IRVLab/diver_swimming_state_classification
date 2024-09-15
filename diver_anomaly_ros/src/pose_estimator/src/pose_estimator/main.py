#!/usr/bin/env python3
import os
import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import ctypes
libgcc_s = ctypes.CDLL('libgcc_s.so.1')

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, Header
from estimator_msgs.msg import InferenceListMsg

from .pose_estimate import Pose3DEstimator
from .features_transform import FeaturesTransform
from .tools.utils import numpy_2d_to_msg


class PoseSequence:
    def __init__(self):
        # super(PoseSequence, self).__init__()
        estimator_weight_path = rospy.get_param("estimator_weight_path", "")
        liftor_weight_path = rospy.get_param("liftor_weight_path", "")

        calib_left_file_path = rospy.get_param("calib_left_file_path", "")
        calib_right_file_path = rospy.get_param("calib_right_file_path", "")

        camera_side = rospy.get_param("camera_side", "left")

        display_pose_2d = rospy.get_param("display_pose_2d", False)
        display_pose_3d = rospy.get_param("display_pose_3d", False)

        window_size = rospy.get_param("window_size", 5)
        sample_frequency = rospy.get_param("sample_frequency", 10)

        self.sequence_pub = rospy.Publisher(
            '/sequence_topic', Float64MultiArray, queue_size=10)
        self.combined_inference_pub = rospy.Publisher(
            '/combined_inference', InferenceListMsg, queue_size=10)        

        self.estimator = Pose3DEstimator(
            estimator_weight_path=estimator_weight_path,
            liftor_weight_path=liftor_weight_path,
            calib_left_file_path=calib_left_file_path,
            calib_right_file_path=calib_right_file_path,
            camera_side=camera_side,
            display_pose_2d=display_pose_2d,
            display_pose_3d=display_pose_3d)

        self.image_sub = rospy.Subscriber(
            "/image_topic", Image, callback=self.run)

        self.image_seq = []
        self.pose3d_list = []

        # This class is used to accumulate pose features and calculate angular
        # acceleration from the time series data
        self.feats_trans = FeaturesTransform(
            window_size=window_size,
            sample_frequency=sample_frequency)

        # Create a CvBridge instance
        self.bridge = CvBridge()

        # Log the variables being read
        rospy.loginfo("estimator_weight_path: %s", estimator_weight_path)
        rospy.loginfo("liftor_weight_path: %s", liftor_weight_path)
        rospy.loginfo("calib_left_file_path: %s", calib_left_file_path)
        rospy.loginfo("calib_right_file_path: %s", calib_right_file_path)
        rospy.loginfo("camera_side: %s", camera_side)
        rospy.loginfo("display_pose_2d: %s", display_pose_2d)
        rospy.loginfo("display_pose_3d: %s", display_pose_3d)
        rospy.loginfo("window_size: %s", window_size)
        rospy.loginfo("sample_frequency: %s", sample_frequency)
        rospy.loginfo("PoseSequence node is ready")

    def run(self, msg):
        # Apply the pose estimator
        pose3d = self.estimator.apply(msg)

        # Accumulate the pose features
        if pose3d is not None:
            # Accumulate the images that we're actually running inference on
            self.image_seq.append(msg)

            # Accumulate the pose3d data
            self.pose3d_list.append(pose3d)

            sequence_feats = None
            if len(self.pose3d_list) == 50:
                sequence_feats = self.feats_trans.get_imu_features(
                    np.array(self.pose3d_list))
                sequence_feats = sequence_feats.astype(np.float32)
                self.pose3d_list.pop(0)

            if sequence_feats is not None:
                # Publish the imu features with shape:
                # (window_size * sample_frequency, 3)
                sequence_msg = numpy_2d_to_msg(sequence_feats)
                self.sequence_pub.publish(sequence_msg)

                inflistmsg = InferenceListMsg()
                h = Header()
                h.stamp = rospy.Time.now()
                inflistmsg.header = h

                inflistmsg.image_list = self.image_seq
                inflistmsg.feature_vec = sequence_msg

                self.image_seq = []
                self.combined_inference_pub.publish(inflistmsg)
                rospy.loginfo("Pose Sequences are published!")
