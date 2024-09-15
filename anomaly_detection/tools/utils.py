import os
import math
import tqdm
import cv2
import numpy as np
import onnxruntime as ort
import torch

from tools.loss import compute_loss


def is_supervised(task_name):
    supervised = ["classification"]
    return task_name in supervised


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

    print("ONNX is using {}".format(ort.get_device()))

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


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_affine_transform(center,
                         scale,
                         rot,
                         origin_size,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])

    scale_tmp = scale * origin_size
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def get_preds(batch, model, device, task):
    if task == "classification":
        x, lbl = batch
        return model(x.to(device)), lbl.to(device), None
    elif task == "imputation":
        x, target, masks = batch
        return model(x.to(device)), target.to(device), masks.to(device)


def train(epoch, num_epochs, model, optimizer, train_loader, val_loader,
          device, task="classification"):
    train_loss, val_loss = 0, 0
    train_class_acc, val_class_acc = 0, 0
    train_num, val_num = 0, 0

    print(('\n' + '%10s' * 3) % ('Epoch', 'lr', 'loss'))

    # -------------------
    # ------ Train ------
    # -------------------

    model.train()
    pbar = enumerate(train_loader)
    pbar = tqdm.tqdm(pbar, total=len(train_loader))
    for i, batch in pbar:
        optimizer.zero_grad()

        # if task is classifcation, masks will be null
        pred, target, masks = get_preds(batch, model, device, task)
        loss = compute_loss(pred, target, masks)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if is_supervised(task):
            pred = torch.argmax(pred, 1)
            train_class_acc += (pred == target).float().sum().item()
            train_num += len(target)

        s = ('%10s' + '%10.4g' * 2) \
            % ('%g/%g' % (epoch + 1, num_epochs),
                optimizer.param_groups[0]["lr"], loss)
        pbar.set_description(s)
        pbar.update(0)

    # --------------------
    # ---- Validation ----
    # --------------------
    epoch_results = {'preds': [], 'targets': [], 'masks': []}

    model.eval()
    with torch.no_grad():
        for i, batch in tqdm.tqdm(
                enumerate(val_loader), total=len(val_loader)):
            pred, target, masks = get_preds(batch, model, device, task)
            loss = compute_loss(pred, target, masks)

            if not is_supervised(task):
                epoch_results['preds'].append(pred.cpu().numpy())
                epoch_results['targets'].append(target.cpu().numpy())
                epoch_results['masks'].append(masks.cpu().numpy())

            val_loss += loss.item()

            if is_supervised(task):
                pred = torch.argmax(pred, 1)
                val_class_acc += (pred == target).float().sum().item()
                val_num += len(target)

    train_mean_loss = train_loss / train_loader.__len__()
    val_mean_loss = val_loss / val_loader.__len__()
    train_mean_acc = train_class_acc / train_num if train_num > 0 else 0
    val_mean_acc = val_class_acc / val_num if val_num > 0 else 0

    return train_mean_loss, val_mean_loss, train_mean_acc, val_mean_acc, \
        epoch_results


def train_epoch(model, optimizer, dl, device, task="classification"):
    epoch_loss = 0.0
    epoch_count = 0

    for _, batch in enumerate(dl):
        pred, target, masks = get_preds(batch, model, device, task)

        # if task is classifcation, masks will be null
        loss = compute_loss(pred, target, masks)

        epoch_loss += loss.item()
        epoch_count += 1

        optimizer.zero_grad()

        loss.backward()

        # helps deal with exploding gradients
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)

        optimizer.step()

    # return the average loss for the epoch
    return epoch_loss / epoch_count


def evaluate_epoch(model, dl, device, task="classification"):
    # test or validate
    model.eval()

    epoch_loss = 0.0
    epoch_count = 0
    epoch_results = {'preds': [], 'targets': [], 'masks': []}

    with torch.no_grad():
        for _, batch in enumerate(dl):
            preds, target, mask = get_preds(batch, model, device, task)

            loss = compute_loss(preds, target, mask)
            if task == "imputation":
                epoch_results['preds'].append(preds.cpu().numpy())
                epoch_results['targets'].append(target.cpu().numpy())
                epoch_results['masks'].append(mask.cpu().numpy())

            epoch_loss += loss.item()
            epoch_count += 1

    loss = epoch_loss / epoch_count

    return loss, epoch_results


def test(model, dl, device, task, plots=True):
    model.eval()
    acc = 0
    count = 0
    epoch_results = {'preds': [], 'targets': []}

    with torch.no_grad():
        for _, batch in enumerate(dl):
            pred, target, mask = get_preds(batch, model, device, task)

            if task == "imputation":
                pred = pred[mask].reshape(-1)
                target = target[mask].reshape(-1)
                acc += torch.abs(pred - target).float().sum().item()
                count += mask.sum()

            elif task == "classification":
                pred = torch.argmax(pred, 1)
                acc += (pred == target).float().sum().item()
                count += len(target)

            else:
                raise Exception("Unrecognized task")

            if plots:
                epoch_results['preds'].extend(
                    pred.cpu().numpy().tolist())
                epoch_results['targets'].extend(
                    target.cpu().numpy().tolist())

    return acc / count, epoch_results
