import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm
import torch
import glob
import json
from pathlib import Path

from models.yolo import YOLO
from models.videopose import VideoPose
from models.dino import Dino
from tools.rectify import Rectificator
from tools.utils import get_uncalibrated
from tools.plot import plot_pose_2d, plot_pose_3d, plot_feats


class ImageLoader:
    def __init__(self, path):
        self.files = sorted(glob.glob(
            os.path.join(path, '*left.png')))
        self.nf = len(self.files)

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        self.count += 1
        img = cv2.imread(path)  # BGR
        assert img is not None, 'Image Not Found ' + path

        return path, img

    def __len__(self):
        return self.nf  # number of files


class Pose3DEstimator:
    def __init__(self,
                 pose_estimator, lifting_network, feature_extractor,
                 calib_file_path, output_path):
        self.pose_estimator = pose_estimator
        self.lifting_network = lifting_network
        self.feature_extractor = feature_extractor

        cam_left, cam_right = \
            Rectificator.parse_calibration_data(calib_file_path)
        self.rectify = Rectificator(cam_left, cam_right)
        self.K = self.rectify.get_cam_params()['K1']

        self.feat_output_path = os.path.join(output_path, "features")
        self.pose_output_path = os.path.join(output_path, "pose")
        self.crop_output_path = os.path.join(output_path, "crop")
        self.vis_output_path = os.path.join(output_path, "vis")
        os.makedirs(self.feat_output_path, exist_ok=True)
        os.makedirs(self.pose_output_path, exist_ok=True)
        os.makedirs(self.crop_output_path, exist_ok=True)
        os.makedirs(self.vis_output_path, exist_ok=True)

        self.batch_kpts = []
        self.batch_features = []
        self.batch_images = []

        self.base_joint = 6
        self.base_pose = 13

        self.pad = 13

    def apply(self, idx, img):
        img_rectified = self.rectify.rectify_images("left", img)

        # extract human pose features
        det_bbox, det_kpt = self.pose_estimator.inference(
            img_rectified.copy())

        if len(det_bbox) == 0:
            raise AssertionError("No person detected in the image")

        kpts = det_kpt.reshape(-1, 3)[:, :2]
        kpts = get_uncalibrated(kpts, self.K)  # convert to uncalibrated coords

        kpts = kpts[[0, 1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14]]
        kpts = kpts[[3, 2, 4, 1, 5, 0, 7, 6, 9, 8, 11, 10]]

        self.batch_kpts.append(kpts)

        # extract image features
        img_crop, features = self.feature_extractor.inference(
            img_rectified.copy(), det_bbox)
        self.batch_features.append(features)
        self.batch_images.append(img_crop)

        # plot 2D pose and features
        feat = plot_feats(torch.from_numpy(img_crop),
                          torch.from_numpy(features).permute(2, 0, 1))
        pose2d_res = plot_pose_2d(img_rectified, det_bbox, det_kpt)
        cv2.imwrite(os.path.join(
            self.vis_output_path, f"feat_{idx:06d}.png"), feat)
        cv2.imwrite(os.path.join(
            self.vis_output_path, f"pose2d_{idx:06d}.png"), pose2d_res)

    def inference(self):
        for i in range(len(self.batch_kpts)):
            # decide the interval to load
            # ex: if we want to predict the 3D pose of index 10, we need to
            # load the 2D data from index 10 - pad to 10 + pad
            start_3d, end_3d = i, i + 1
            start_2d = start_3d - self.pad
            end_2d = end_3d + self.pad
            low_2d = max(start_2d, 0)
            high_2d = min(end_2d, len(self.batch_kpts))
            pad_left_2d = low_2d - start_2d
            pad_right_2d = end_2d - high_2d

            batch = np.array(self.batch_kpts[low_2d:high_2d], dtype=np.float32)

            if pad_left_2d != 0 or pad_right_2d != 0:
                batch = np.pad(
                    batch,
                    ((pad_left_2d, pad_right_2d), (0, 0), (0, 0)),
                    'edge')

            # Dino features
            feature = self.batch_features[i]

            # image crop
            img_crop = self.batch_images[i]

            # 2D pose (uncalibrated)
            pose2d = batch[self.base_pose]
            # all joints are relative to the base joint
            pose2d = pose2d - pose2d[[self.base_joint]]

            # 3D pose
            pose3d = self.lifting_network.inference(batch)
            # all joints are relative to the base joint
            pose3d = pose3d - pose3d[[self.base_joint]]

            # save cropped image
            cv2.imwrite(os.path.join(
                self.crop_output_path, f"crop_{i:06d}.png"), img_crop)

            # Save features into .npy file
            np.save(os.path.join(
                self.feat_output_path, f"feat_{i:06d}.npy"), feature)

            # Write JSON data to file
            data = {
                "pose2d": pose2d.tolist(),
                "pose3d": pose3d.tolist(),
            }
            json_data = json.dumps(data)
            output_file = os.path.join(
                self.pose_output_path, f"pose_{i:06d}.json")
            with open(output_file, 'w') as file:
                file.write(json_data)

            # plot 3D pose
            pose3d_res = plot_pose_3d(pose3d)
            cv2.imwrite(os.path.join(
                self.vis_output_path, f"pose3d_{i:06d}.png"), pose3d_res)


def process_image(args, networks, image_path):
    # image loader
    image_loader = ImageLoader(image_path)

    p = Path(image_path)

    # pose estimator
    pose_3d_estimator = Pose3DEstimator(
        *networks,
        args.calib_path,
        os.path.join(args.dst_path,
                     p.parts[-3], p.parts[-2], p.parts[-1]))

    pbar = tqdm(enumerate(image_loader), total=len(image_loader))
    for i, (img_file, img) in pbar:
        try:
            pose_3d_estimator.apply(i, img)
        except AssertionError as e:
            print(f"Error raised at image file: {img_file}")
            print(e)

    pose_3d_estimator.inference()


def main(args):
    os.makedirs(args.dst_path, exist_ok=True)
    data_path_list = sorted(glob.glob(
        os.path.join(args.data_path, "**/**/**")))

    # Feed models to the process_image function instead of loading them
    # inside the function to avoid loading the models multiple times
    pose_estimator = YOLO(args.estimator_model_path)
    lifting_network = VideoPose(args.liftor_model_path)
    feature_extractor = Dino('dinov2_vitb14')

    networks = (pose_estimator, lifting_network, feature_extractor)

    for data_path in data_path_list:
        process_image(args, networks, data_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--estimator-model-path", type=str, required=True,
        default="")
    parser.add_argument(
        "--liftor-model-path", type=str, required=True,
        default="")
    parser.add_argument(
        "--calib-path", type=str, required=True,
        default="")
    parser.add_argument(
        "--data-path", type=str, required=True,
        default="")
    parser.add_argument(
        "--dst-path", type=str,
        default="./output")
    args = parser.parse_args()

    main(args)
