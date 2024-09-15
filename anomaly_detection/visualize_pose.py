import os
import tqdm
import argparse
import cv2
import numpy as np

from tools.plot import plot_pose_3d, plot_series
from tools.helper import compute_diver_body_frame
from datasets import PoseFeaturesDataset


def vis(value, save_path, filename):
    acc_x, acc_y, acc_z = [], [], []
    frames = []
    for d in tqdm.tqdm(value['data']):
        pose = np.array(d['pose_3d'])
        acc = d['frame_acc']
        x_hat, y_hat, z_hat = compute_diver_body_frame(pose)

        c = np.mean(pose[[0, 1, 6, 7]], axis=0)
        pose_plot = plot_pose_3d(pose, (c, x_hat, y_hat, z_hat))

        acc_x.append(acc[0])
        acc_y.append(acc[1])
        acc_z.append(acc[2])

        acc_plot = plot_series(acc_x, acc_y, acc_z, len(value['data']))

        # Calculate the resize ratio
        resize_ratio = acc_plot.shape[0] / pose_plot.shape[0]
        # Resize pose_plot with the calculated ratio
        pose_plot = cv2.resize(
            pose_plot,
            (int(pose_plot.shape[1] * resize_ratio), acc_plot.shape[0]))

        plot = np.hstack((pose_plot, acc_plot))
        plot = cv2.cvtColor(plot, cv2.COLOR_RGB2BGR)
        frames.append(plot)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        os.path.join(save_path, f'{filename}.mp4'), fourcc, 10.0,
        (plot.shape[1], plot.shape[0]))

    # Write each frame to the video file
    for frame in frames:
        out.write(frame)

    print(f"Video saved to {os.path.join(save_path, f'{filename}.mp4')}")

    # Release the video writer and close the video file
    out.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path", type=str, required=True,
        default="")
    parser.add_argument(
        "--dst-path", type=str, required=True,
        default="")
    parser.add_argument(
        "--subject", type=str, required=True,
        default="")
    args = parser.parse_args()

    os.makedirs(args.dst_path, exist_ok=True)
    dataset = PoseFeaturesDataset(args.data_path)

    try:
        value = dataset.metadata[args.subject]
    except KeyError:
        raise ValueError(
            f"Subject {args.subject} not found in the dataset.\n"
            f"List of available subjects: {list(dataset.metadata.keys())}")
    vis(value, args.dst_path, args.subject)
