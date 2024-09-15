import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter

from .tools.helper import (
    compute_diver_body_frame, compute_diver_part_frame,
    compute_rotation_difference,
    second_derivative_of_translation, second_derivative_of_rotation,
)


class Standardize:
    def __init__(self):
        self.scaler = StandardScaler()

    def __call__(self, series):
        '''
        input: series - (n_samples, n_feats)
            The data used to compute the mean and standard deviation used for
            later scaling along the features axis.
        returns: series - (n_samples, n_features_new)
        '''
        return self.scaler.fit_transform(series)


class Savgol_Filter:
    def __init__(self, filter_size=5, polyorder=3):
        self.filter_size = filter_size
        self.polyorder = polyorder

    def __call__(self, ts):
        _, dim = ts.shape
        filtered_ts = np.zeros_like(ts)
        for i in range(dim):
            filtered_ts[:, i] = savgol_filter(
                ts[:, i], self.filter_size, self.polyorder)

        return np.array(filtered_ts, dtype="float32")


class FeaturesTransform:
    def __init__(self, window_size, sample_frequency,
                 body_rot_acc=False, part_trans_acc=True, part_rot_acc=False,
                 standardize=True, savgol=True):
        self.window_size = window_size
        self.sample_freq = sample_frequency
        self.sample_interval = 1 / sample_frequency

        self.include_body_rot_acc = body_rot_acc
        self.include_part_trans_acc = part_trans_acc
        self.include_part_rot_acc = part_rot_acc

        self.standardize = None
        if standardize:
            self.standardize = Standardize()
        self.savgol_filter = None
        if savgol:
            self.savgol_filter = Savgol_Filter()

    def get_imu_features(self, pose3d):
        assert isinstance(pose3d, np.ndarray), \
            'pose3d should be a numpy array'
        assert pose3d.ndim == 3 and pose3d.shape[2] == 3, \
            'pose3d should be a sequence of 3D poses'
        assert pose3d.shape[0] == self.window_size * self.sample_freq, \
            'length of pose3d should be equal to window_size * sample_freq'

        accs = self._cal_acceleration(pose3d)

        if self.standardize:
            accs = self.standardize(accs)
        if self.savgol_filter:
            accs = self.savgol_filter(accs)
        return accs

    def _cal_acceleration(self, pose3d):
        pose_feats = []

        # Calculate the angular acceleration of the body frame
        if self.include_body_rot_acc:
            frame_acc = self._cal_body_frame_acc(pose3d.copy())
            pose_feats.append(frame_acc[:, np.newaxis, :])

        if self.include_part_trans_acc:
            # Calculate the translation acceleration of the body parts
            part_trans_acc = self._cal_part_trans_acc(pose3d.copy())
            pose_feats.append(part_trans_acc)

        if self.include_part_rot_acc:
            # Calculate the rotational acceleration of the body parts
            part_rot_acc = self._cal_part_rot_acc(pose3d.copy())
            pose_feats.append(part_rot_acc)

        pose_feats = np.concatenate(pose_feats, axis=1)
        pose_feats = pose_feats.reshape(pose_feats.shape[0], -1)

        return pose_feats

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
