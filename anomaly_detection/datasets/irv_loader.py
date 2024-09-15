import glob
import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from pathlib import Path

from tools.helper import (
    compute_diver_body_frame, compute_diver_part_frame,
    compute_rotation_difference,
    second_derivative_of_translation, second_derivative_of_rotation,
)


class BaseFeaturesDataset(Dataset):
    def __init__(self, data_path, window_size=3, sample_frequency=10,
                 window_overlap=0.5, transform=None, crop_image=False):
        self.window = window_size * sample_frequency
        self.sample_interval = 1 / sample_frequency

        self.class_dict = {"notmoving": 0, "moving": 1}

        self.window_overlap = window_overlap

        # Initialize the dataset
        self.metadata = self._load_data(data_path, crop_image)
        self.seq_pairs, self.labels = self._get_pairs(self.metadata)

        self.transform = transform

    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.seq_pairs)

    def __getitem__(self, idx):
        raise NotImplementedError

    def _load_data(self, data_path, crop_image):
        metadata_path = sorted(glob.glob(os.path.join(data_path, "**/**")))

        metadata = {}
        for path in metadata_path:
            target = Path(path).parent.stem
            subject = f"{target}_{Path(path).stem}"

            pose_path = sorted(glob.glob(
                os.path.join(path, "pose/*.json")))
            image_path = sorted(glob.glob(
                os.path.join(path, "crop/*.png")))

            assert len(pose_path) == len(image_path), \
                ("Number of data files do not match, "
                 "pose: {}, image: {}".format(len(pose_path), len(image_path)))

            metadata[subject] = {
                "data": [],
                "target": target
            }

            for i in range(len(pose_path)):
                if crop_image:
                    image = cv2.imread(image_path[i])
                    info = {'image': image}
                else:
                    with open(pose_path[i], 'r') as f:
                        data = json.load(f)

                    info = {
                        'pose_2d': data['pose2d'],
                        'pose_3d': data['pose3d']
                    }

                metadata[subject]["data"].append(info)

        return metadata

    def _get_pairs(self, metadata):
        # Build lineage info
        # ex: [(subject1, 0, 30), (subject1, 1, 31), (subject1, 2, 32), ...
        seq_pairs = []  # (seq_subj, start_frame, end_frame) tuples
        labels = []
        for subject, db in metadata.items():
            n_chunks = len(db["data"])
            bounds = np.arange(n_chunks + 1)
            # determines degree of overlap
            stride = int((1-self.window_overlap)*self.window)
            pairs = list(zip(np.repeat(subject, len(bounds - self.window)),
                             bounds[:-self.window:stride],
                             bounds[self.window::stride]))
            seq_pairs.extend(pairs)

            for _ in range(len(pairs)):
                labels.append(self.class_dict[db["target"]])

        return seq_pairs, labels


class PoseFeaturesDataset(BaseFeaturesDataset):
    def __init__(self, data_fp, window_size=3, sample_frequency=10,
                 window_overlap=0.5, part_trans_acc=False, part_rot_acc=False,
                 transform=None):
        super().__init__(data_fp, window_size, sample_frequency,
                         window_overlap, transform)

        self.include_part_trans_acc = part_trans_acc
        self.include_part_rot_acc = part_rot_acc

        self._add_imu_features()

    def __getitem__(self, idx):
        seq_subject, start_3d, end_3d = self.seq_pairs[idx]
        db = self.metadata[seq_subject]

        pose_feats = []
        target = db['target']
        for d in db["data"][start_3d:end_3d]:
            feat = []

            frame_acc = d['frame_acc'][np.newaxis, :]
            feat.append(frame_acc)

            if self.include_part_trans_acc:
                part_trans_acc = d['part_trans_acc']
                feat.append(part_trans_acc)

            if self.include_part_rot_acc:
                part_rot_acc = d['part_rot_acc']
                feat.append(part_rot_acc)

            feat = np.vstack(feat)
            pose_feats.append(feat.flatten())

        pose_feats = np.array(pose_feats, dtype=np.float32)
        target = np.array(self.labels[idx], dtype=np.int64)

        if self.transform:
            pose_feats = self.transform(pose_feats)

        return pose_feats, target

    def _add_imu_features(self):
        for values in self.metadata.values():
            pose_3d = np.array([data['pose_3d'] for data in values["data"]])

            frame_acc = self._cal_body_frame_acc(pose_3d.copy())
            part_trans_acc = self._cal_part_trans_acc(pose_3d.copy())
            part_rot_acc = self._cal_part_rot_acc(pose_3d.copy())

            for i, d in enumerate(values["data"]):
                d.update({"frame_acc": frame_acc[i]})
                d.update({"part_trans_acc": part_trans_acc[i]})
                d.update({"part_rot_acc": part_rot_acc[i]})

    def _cal_body_frame_acc(self, pose_feats):
        # Get the coordinate of the diver's body frame
        frame_axis = np.array([
            np.vstack(compute_diver_body_frame(pose)).T for
            pose in pose_feats
        ])

        frame_acc = second_derivative_of_rotation(
            frame_axis, self.sample_interval)

        return frame_acc

    def _cal_part_trans_acc(self, pose_feats):
        part_trans_acc = np.zeros_like(pose_feats)

        # convert the unit of the pose features to meters
        pose_feats = pose_feats / 1000

        num_joints = pose_feats.shape[1]
        for i in range(num_joints):
            part_trans_acc[:, i] = second_derivative_of_translation(
                pose_feats[:, i], self.sample_interval)

        # get rid of the sixth joint (idx=6) as it is the origin of the pose
        part_trans_acc = np.delete(part_trans_acc, 6, axis=1)

        return part_trans_acc

    def _cal_part_rot_acc(self, pose_feats):
        part_frame_axis_relative = []

        # Get the coordinate of the diver's part frame
        for pose in pose_feats:
            part_frame_axis = compute_diver_part_frame(pose)
            body_frame_axis = np.vstack(compute_diver_body_frame(pose)).T

            for i in range(part_frame_axis.shape[0]):
                part_frame_axis[i] = compute_rotation_difference(
                    body_frame_axis, part_frame_axis[i])

            part_frame_axis_relative.append(part_frame_axis)

        part_frame_axis_relative = np.array(part_frame_axis_relative)

        num_parts = part_frame_axis_relative.shape[1]
        part_rot_acc = np.zeros((pose_feats.shape[0], num_parts, 3))

        for i in range(num_parts):
            part_rot_acc[:, i] = second_derivative_of_rotation(
                part_frame_axis_relative[:, i], self.sample_interval)

        return part_rot_acc


class ImageDataset(BaseFeaturesDataset):
    def __init__(self, data_fp, window_size=3, sample_frequency=10,
                 window_overlap=0.5):
        super().__init__(data_fp, window_size, sample_frequency,
                         window_overlap, crop_image=True)

        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, idx):
        seq_subject, start_3d, end_3d = self.seq_pairs[idx]
        db = self.metadata[seq_subject]

        imgs = []
        for d in db["data"][start_3d:end_3d]:
            img = d['image']

            img = img[:, :, ::-1]  # BGR to RGB
            img = np.ascontiguousarray(img)

            img = self.transform(img)

            imgs.append(img)

        imgs = torch.stack(imgs)
        target = np.array(self.labels[idx], dtype=np.int64)

        return imgs, target
