import cv2
import yaml
import numpy as np

from .rectify import Camera


class Undistortor:
    def __init__(self, camera):
        self.camera = camera

    @staticmethod
    def load_camera_parameters(yaml_file):
        with open(yaml_file, 'r') as file:
            camera_params = yaml.safe_load(file)

        width = camera_params['cam0']['resolution'][0]
        height = camera_params['cam0']['resolution'][1]

        def gen_intrinsic(coeff):
            return np.array([
                [coeff[0], 0, coeff[2]],
                [0, coeff[1], coeff[3]],
                [0, 0, 1]
            ])

        camera_matrix = gen_intrinsic(
            camera_params['cam0']['intrinsics'])
        dist_coeffs = np.array(
            camera_params['cam0']['distortion_coeffs']).reshape(1, -1)

        camera = Camera(
            width=width,
            height=height,
            cam_matrix=camera_matrix,
            dist=dist_coeffs)

        return camera

    def undistort_image(self, img):
        """
        Undistort the input image

        Args:
            img (np.ndarray): input image

        Returns:
            np.ndarray: undistorted image
        """
        camera_intrinsic = self.camera.cam_matrix
        dist_coeffs = self.camera.dist
        img_undistorted = cv2.undistort(img, camera_intrinsic, dist_coeffs)

        return img_undistorted
