import os
import math
import cv2
import numpy as np
import onnxruntime as ort
from std_msgs.msg import (Float64MultiArray, MultiArrayDimension,
                          MultiArrayLayout)


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True,
              scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    # wh padding
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        # width, height ratios
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def check_img_size(img_size, s=32):
    def make_divisible(x, divisor):
        # Returns x evenly divisible by divisor
        return math.ceil(x / divisor) * divisor
    # Verify img_size is a multiple of stride s
    new_size = make_divisible(img_size, int(s))  # ceil gs-multiple
    if new_size != img_size:
        print('WARNING: --img-size %g must be multiple of " \
              "max stride %g, updating to %g' % (img_size, s, new_size))
    return new_size


def scale_coords(img1_shape, coords, img0_shape,
                 ratio_pad=None, kpt_label=False, step=2):
    def clip_coords(boxes, img_shape, step=2):
        # Clip bounding xyxy bounding boxes to image shape (height, width)
        boxes[:, 0::step].clip(0, img_shape[1])  # x1
        boxes[:, 1::step].clip(0, img_shape[0])  # y1
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0],
                   img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = ((img1_shape[1] - img0_shape[1] * gain) / 2,
               (img1_shape[0] - img0_shape[0] * gain) / 2)  # wh padding
    else:
        gain = ratio_pad[0]
        pad = ratio_pad[1]
    if isinstance(gain, (list, tuple)):
        gain = gain[0]

    if not kpt_label:
        coords[:, [0, 2]] -= pad[0]  # x padding
        coords[:, [1, 3]] -= pad[1]  # y padding
        coords[:, [0, 2]] /= gain
        coords[:, [1, 3]] /= gain
        clip_coords(coords[0:4], img0_shape)
    else:
        coords[:, 0::step] -= pad[0]  # x padding
        coords[:, 1::step] -= pad[1]  # y padding
        coords[:, 0::step] /= gain
        coords[:, 1::step] /= gain
        clip_coords(coords, img0_shape, step=step)
    return coords


def load_onnx_model(weight_path):
    if not os.path.exists(weight_path):
        assert False, "Model is not exist in {}".format(weight_path)

    session = ort.InferenceSession(
        weight_path,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    print("Weight from ONNX model: {} is loaded with {}"
          .format(weight_path, ort.get_device()))

    return session


def get_uncalibrated(pts, K):
    """
    Generate the uncalibrated 3D points.

    Args:
        pts : numpy.ndarray (12, 2)
            Points at the image plane.
        K : numpy.ndarray (3, 3)
            The indices of the base pose for each sample.

    Returns:
        numpy.ndarray (12, 3):
            Points at the camera coordinate system.
    """
    pts_homo = np.concatenate((pts, np.ones_like(pts[..., :1])), axis=-1)
    pts_homo = pts_homo.reshape(-1, 3)
    pts_cam = (np.linalg.inv(K) @ pts_homo.T).T
    pts_cam = pts_cam.reshape(-1, 3)

    return pts_cam[:, :2]


def numpy_2d_to_msg(data_in):

    rows, cols = data_in.shape
    multi_array_dimension_list = []

    tmp = MultiArrayDimension()
    tmp.label = 'rows'
    tmp.size = int(rows)
    multi_array_dimension_list.append(tmp)

    tmp = MultiArrayDimension()
    tmp.label = 'cols'
    tmp.size = int(cols)
    multi_array_dimension_list.append(tmp)

    multi_array_layout = MultiArrayLayout()
    multi_array_layout.dim = multi_array_dimension_list

    data_array_msg = Float64MultiArray()
    data_array_msg.data = data_in.flatten()
    data_array_msg.layout = multi_array_layout

    return data_array_msg
