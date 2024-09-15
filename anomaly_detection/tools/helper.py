import numpy as np
from scipy.spatial.transform import Rotation


def compute_diver_body_frame(pose_3d):
    """
    Compute the diver body frame given the 3D pose.

    Args:
        pose_3d : numpy.ndarray (12, 3)
            The 3D pose of the diver.

    Returns:
        x_hat : numpy.ndarray (3,)
            The x-axis unit vector of the diver's body frame.
        y_hat : numpy.ndarray (3,)
            The y-axis unit vector of the diver's body frame.
        z_hat : numpy.ndarray (3,)
            The z-axis unit vector of the diver's body frame.
    """
    # COCO format:
    # [r_shoulder, l_shoulder, r_elbow, l_elbow, r_wrist, l_wrist,
    #  r_hip, l_hip, r_knee, l_knee, r_ankle, l_ankle]
    r_shoulder, l_shoulder = pose_3d[1], pose_3d[0]
    r_hip, l_hip = pose_3d[7], pose_3d[6]

    center_mass = np.mean(
        [r_shoulder, l_shoulder, r_hip, l_hip], axis=0)

    rhs = r_shoulder - r_hip
    rhls = l_shoulder - r_hip
    lhs = l_shoulder - l_hip
    lhrs = r_shoulder - l_hip

    r_cross = np.cross(rhls, rhs)
    l_cross = np.cross(lhs, lhrs)

    z = (r_cross + l_cross) / 2
    z_hat = z / (np.linalg.norm(z) + 1e-5)

    hip_midpt = (r_hip + l_hip) / 2
    y = hip_midpt - center_mass
    y_hat = y / (np.linalg.norm(y) + 1e-5)

    # x is the cross product of y_hat and z_hat
    x_hat = np.cross(y_hat, z_hat)

    return x_hat, y_hat, z_hat


def compute_diver_part_frame(pose_3d):
    """
    Compute the diver part frame given the 3D pose.

    Args:
        pose_3d : numpy.ndarray (12, 3)
            The 3D pose of the diver.

    Returns:
        frame_axis : numpy.ndarray (8, 3, 3)
            The frame axis of the diver's body parts.
    """
    # arm_up -> vector from shoulder to elbow
    # arm_down -> vector from elbow to wrist
    r_arm_up = pose_3d[3] - pose_3d[1]
    r_arm_up = r_arm_up / np.linalg.norm(r_arm_up)

    r_arm_down = pose_3d[5] - pose_3d[3]
    r_arm_down = r_arm_down / np.linalg.norm(r_arm_down)

    l_arm_up = pose_3d[2] - pose_3d[0]
    l_arm_up = l_arm_up / np.linalg.norm(l_arm_up)

    l_arm_down = pose_3d[4] - pose_3d[2]
    l_arm_down = l_arm_down / np.linalg.norm(l_arm_down)

    # chest -> vector from right shoulder to left shoulder
    # stomach -> vector from right hip to left hip
    chest = pose_3d[0] - pose_3d[1]
    chest = chest / np.linalg.norm(chest)

    stomach = pose_3d[6] - pose_3d[7]
    stomach = stomach / np.linalg.norm(stomach)

    # thigh -> vector from hip to knee
    # calf -> vector from knee to ankle
    r_thigh = pose_3d[9] - pose_3d[7]
    r_thigh = r_thigh / np.linalg.norm(r_thigh)

    r_calf = pose_3d[11] - pose_3d[9]
    r_calf = r_calf / np.linalg.norm(r_calf)

    l_thigh = pose_3d[8] - pose_3d[6]
    l_thigh = l_thigh / np.linalg.norm(l_thigh)

    l_calf = pose_3d[10] - pose_3d[8]
    l_calf = l_calf / np.linalg.norm(l_calf)

    # right arm frame
    r_arm_x_hat = r_arm_down
    r_arm_y_hat = np.cross(r_arm_x_hat, r_arm_up)
    r_arm_z_hat = np.cross(r_arm_x_hat, r_arm_y_hat)
    r_arm_frame = np.vstack([r_arm_x_hat, r_arm_y_hat, r_arm_z_hat]).T

    # right shoulder frame
    r_shoulder_x_hat = r_arm_up
    r_shoulder_y_hat = np.cross(r_shoulder_x_hat, chest)
    r_shoulder_z_hat = np.cross(r_shoulder_x_hat, r_shoulder_y_hat)
    r_shoulder_frame = np.vstack(
        [r_shoulder_x_hat, r_shoulder_y_hat, r_shoulder_z_hat]).T

    # left arm frame
    l_arm_x_hat = l_arm_down
    l_arm_y_hat = np.cross(l_arm_x_hat, l_arm_up)
    l_arm_z_hat = np.cross(l_arm_x_hat, l_arm_y_hat)
    l_arm_frame = np.vstack([l_arm_x_hat, l_arm_y_hat, l_arm_z_hat]).T

    # left shoulder frame
    l_shoulder_x_hat = l_arm_up
    l_shoulder_y_hat = np.cross(-chest, l_shoulder_x_hat)
    l_shoulder_z_hat = np.cross(l_shoulder_x_hat, l_shoulder_y_hat)
    l_shoulder_frame = np.vstack(
        [l_shoulder_x_hat, l_shoulder_y_hat, l_shoulder_z_hat]).T

    # right hip frame
    r_hip_x_hat = r_thigh
    r_hip_y_hat = np.cross(r_hip_x_hat, stomach)
    r_hip_z_hat = np.cross(r_hip_x_hat, r_hip_y_hat)
    r_hip_frame = np.vstack([r_hip_x_hat, r_hip_y_hat, r_hip_z_hat]).T

    # right knee frame
    r_knee_x_hat = r_calf
    r_knee_y_hat = np.cross(r_knee_x_hat, r_thigh)
    r_knee_z_hat = np.cross(r_knee_x_hat, r_knee_y_hat)
    r_knee_frame = np.vstack([r_knee_x_hat, r_knee_y_hat, r_knee_z_hat]).T

    # left hip frame
    l_hip_x_hat = l_thigh
    l_hip_y_hat = np.cross(-stomach, l_hip_x_hat)
    l_hip_z_hat = np.cross(l_hip_x_hat, l_hip_y_hat)
    l_hip_frame = np.vstack([l_hip_x_hat, l_hip_y_hat, l_hip_z_hat]).T

    # left knee frame
    l_knee_x_hat = l_calf
    l_knee_y_hat = np.cross(l_knee_x_hat, l_thigh)
    l_knee_z_hat = np.cross(l_knee_x_hat, l_knee_y_hat)
    l_knee_frame = np.vstack([l_knee_x_hat, l_knee_y_hat, l_knee_z_hat]).T

    frame_axis = np.array([
        r_arm_frame, r_shoulder_frame, r_hip_frame, r_knee_frame,
        l_arm_frame, l_shoulder_frame, l_hip_frame, l_knee_frame
    ])

    return frame_axis


def compute_rotation_difference(T1, T2):
    """
    Compute the rotation transformation between two coordinates.

    Args:
        T1 : numpy.ndarray (3, 3)
            The first coordinate.
        T2 : numpy.ndarray (3, 3)
            The second coordinate.

    Returns:
        numpy.ndarray (3, 3)
    """
    r1 = Rotation.from_matrix(T1)
    r2 = Rotation.from_matrix(T2)

    R = (r2 * r1.inv()).as_matrix()
    return R


def second_derivative_of_translation(x, h):
    """
    Calculate the second derivative of x using the finite difference method.

    Parameters:
    x (np.ndarray: [num_timesteps, 3]): Pose positions.
    h (float): The step size.

    Returns:
    np.ndarray ([num_timesteps, 3]): The second derivative of x.
    """
    assert x.ndim == 2, "Input must be a 2D array."
    assert x.shape[1] == 3, "Input must have 3 columns (x, y, z)."

    # Pad the edge of the array with nearest neighbors
    x = np.pad(x, ((1, 1), (0, 0)), mode='edge')

    return (x[2:] - 2 * x[1:-1] + x[:-2]) / h**2


def second_derivative_of_rotation(x, h):
    """
    Calculate the second derivative of x using the finite difference method.

    Parameters:
    x (np.ndarray: [num_timesteps, 3, 3]): Coordinates of the frame
    h (float): The step size.

    Returns:
    np.ndarray ([num_timesteps, 3]): The second derivative of x.
    """
    assert x.ndim == 3, "Input must be a 3D array."
    assert x.shape[1:] == (3, 3), \
        "Input must have shape (3, 3), which represents frame coordinate."

    num_timesteps = x.shape[0]
    acceleration = np.zeros((num_timesteps, 3))

    # Pad the edge of the array with nearest neighbors
    x = np.pad(x, ((1, 1), (0, 0), (0, 0)), mode='edge')

    for i in range(num_timesteps):
        r0 = Rotation.from_matrix(x[i - 1])
        r1 = Rotation.from_matrix(x[i])
        r2 = Rotation.from_matrix(x[i + 1])

        angle_diff = (r2 * r1.inv()) * (r1 * r0.inv()).inv()
        acceleration[i] = angle_diff.as_euler('xyz', degrees=False) / h**2

    return acceleration
