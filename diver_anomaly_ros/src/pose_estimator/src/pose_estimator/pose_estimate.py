#!/usr/bin/env python3

import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import numpy as np
import rospy
from sensor_msgs.msg import Image
from .not_cv_bridge import imgmsg_to_cv2, cv2_to_imgmsg
# import ctypes
# libgcc_s = ctypes.CDLL('libgcc_s.so.1')
from .models import YOLO, VideoPose
from .tools.utils import get_uncalibrated
from .tools.rectify import Rectificator
from .tools.undistort import Undistortor
from .tools.helper import compute_diver_body_frame
from .tools.plot import plot_pose_2d, plot_pose_3d


class Pose3DEstimator:
    def __init__(self,
                 estimator_weight_path, liftor_weight_path,
                 calib_left_file_path, calib_right_file_path,
                 camera_side="left",
                 display_pose_2d=False, display_pose_3d=False):
        super(Pose3DEstimator, self).__init__()

        self.is_tx2 = True

        # cam_left = Rectificator.load_camera_parameters(calib_left_file_path)
        # cam_right = Rectificator.load_camera_parameters(calib_right_file_path)
        # cam_left, cam_right = Rectificator.parse_calibration_data(calib_left_file_path)
        # self.rectify = Rectificator(cam_left, cam_right)
        # self.K = self.rectify.get_cam_params()['K1']

        cam = Undistortor.load_camera_parameters(calib_left_file_path)
        self.undistortor = Undistortor(cam)
        self.K = cam.cam_matrix

        self.pose_estimator = YOLO(estimator_weight_path)
        self.lifting_network = VideoPose(liftor_weight_path)

        assert camera_side in ["left", "right"], \
            "camera_side should be either 'left' or 'right', " \
            f"but got {camera_side}"

        self.camera_side = camera_side
        self.display_pose_2d = display_pose_2d
        self.display_pose_3d = display_pose_3d

        self.batch_kpts = []

        self.num_frame = 27
        self.base_joint = 6
        self.base_pose = 13
        self.t_start = rospy.Time.now()
        self.pose_2d_vis_pub = rospy.Publisher(
            '/pose/pose_2d_vis_topic', Image, queue_size=10)
        self.pose_3d_vis_pub = rospy.Publisher(
            '/pose/pose_3d_vis_topic', Image, queue_size=10)

        # Create a CvBridge instance
        # self.bridge = CvBridge()

    def apply(self, msg):
        # Hack for dealing with broken cv_brdige on Jetson Tx2
        if self.is_tx2:
            img = imgmsg_to_cv2(msg)
        else:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        # img_rectified = self.rectify.rectify_images(self.camera_side, img)
        img_rectified = self.undistortor.undistort_image(img)

        # extract human pose features
        det_bbox, det_kpt = self.pose_estimator.inference(
            img_rectified.copy())

        rospy.loginfo(f"Detected {len(det_bbox)} humans")

        out = None
        if len(det_bbox) > 0:
            # only consider one person for now
            det_bbox = det_bbox[0]
            det_kpt = det_kpt[0]

            kpts = det_kpt.reshape(-1, 3)[:12, :2]
            self.batch_kpts.append(kpts)

            # plot 2D detection results
            if self.display_pose_2d:
                t_now = rospy.Time.now()
                t_elapsed = np.round((t_now - self.t_start).to_sec(),2)
                pose2d_res = plot_pose_2d(img_rectified, det_bbox, det_kpt, t_elapsed, t_now)
                if self.is_tx2:
                    pose2d_res_msg = cv2_to_imgmsg(pose2d_res)
                else:
                    pose2d_res_msg = self.bridge.cv2_to_imgmsg(pose2d_res, "bgr8")

                self.pose_2d_vis_pub.publish(pose2d_res_msg)

        if len(self.batch_kpts) == self.num_frame:
            out = self.inference()
            self.batch_kpts.pop(0)

        return out

    def inference(self):
        batch_kpts = np.stack(self.batch_kpts)

        # 2D pose (uncalibrated)
        pose2d = batch_kpts[self.base_pose]
        # all joints are relative to the base joint
        pose2d = pose2d - pose2d[[self.base_joint]]

        # lift 2D keypoints to 3D
        pose3d = self.lifting_network.inference(batch_kpts)
        # all joints are relative to the base joint
        pose3d = pose3d - pose3d[[self.base_joint]]

        # plot 3D pose
        if self.display_pose_3d:
            c = np.mean(pose3d[[0, 1, 6, 7]], axis=0)
            frame_axis = compute_diver_body_frame(pose3d)
            pose3d_res = plot_pose_3d(pose3d, (c, *frame_axis))
            if self.is_tx2:
                pose3d_res_msg = cv2_to_imgmsg(pose3d_res)
            else:
                pose3d_res_msg = self.bridge.cv2_to_imgmsg(pose3d_res, "bgr8")
            self.pose_3d_vis_pub.publish(pose3d_res_msg)

        return pose3d
