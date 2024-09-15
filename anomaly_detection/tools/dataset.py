import os
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, Subset

from datasets.utils import (
    Standardize, Savgol_Filter, train_val_split, collate_unsuperv
)
from datasets import (
    ClassificationDataset, PoseFeaturesDataset, ImageDataset
)
from tools.plot import visualize_dataset


def get_UEA_dataset(data_cfg, transform=None):
    '''
    load a dataset from the uea multivariate time series archive

    '''
    train_dataset = ClassificationDataset(
        data_fp=data_cfg.train_data_path,
        num_classes=data_cfg.num_classes,
        num_features=data_cfg.num_features,
        sample_frequency=data_cfg.sample_frequency,
        window_size=data_cfg.preproc.window_size,
        window_overlap=data_cfg.preproc.window_overlap,
        transform=transform)

    test_dataset = ClassificationDataset(
        data_fp=data_cfg.test_data_path,
        num_classes=data_cfg.num_classes,
        num_features=data_cfg.num_features,
        sample_frequency=data_cfg.sample_frequency,
        window_size=data_cfg.preproc.window_size,
        window_overlap=data_cfg.preproc.window_overlap,
        transform=transform)

    return train_dataset, test_dataset


def get_pose_dataset(data_cfg, transform=None):
    train_dataset = PoseFeaturesDataset(
        data_fp=data_cfg.train_data_path,
        sample_frequency=data_cfg.sample_frequency,
        window_size=data_cfg.preproc.window_size,
        window_overlap=data_cfg.preproc.window_overlap,
        part_trans_acc=data_cfg.preproc.part_trans_acc,
        part_rot_acc=data_cfg.preproc.part_rot_acc,
        transform=transform)

    test_dataset = PoseFeaturesDataset(
        data_fp=data_cfg.test_data_path,
        sample_frequency=data_cfg.sample_frequency,
        window_size=data_cfg.preproc.window_size,
        window_overlap=data_cfg.preproc.window_overlap,
        part_trans_acc=data_cfg.preproc.part_trans_acc,
        part_rot_acc=data_cfg.preproc.part_rot_acc,
        transform=transform)

    return train_dataset, test_dataset


def get_image_dataset(data_cfg):
    train_dataset = ImageDataset(
        data_fp=data_cfg.train_data_path,
        sample_frequency=data_cfg.sample_frequency,
        window_size=data_cfg.preproc.window_size,
        window_overlap=data_cfg.preproc.window_overlap)

    test_dataset = ImageDataset(
        data_fp=data_cfg.test_data_path,
        sample_frequency=data_cfg.sample_frequency,
        window_size=data_cfg.preproc.window_size,
        window_overlap=data_cfg.preproc.window_overlap)

    return train_dataset, test_dataset


def get_dataloader(data_cfg, batch_size, num_workers, model_name,
                   task="classification", plot=False):
    assert os.path.exists(data_cfg.train_data_path), \
        f"Data file path '{data_cfg.train_data_path}' does not exist."
    assert os.path.exists(data_cfg.test_data_path), \
        f"Data file path '{data_cfg.test_data_path}' does not exist."

    preprocessing_piepline = []
    if data_cfg.preproc.standardize:
        preprocessing_piepline.append(Standardize())
    if data_cfg.preproc.savgol:  # Savitzky-Golay filter
        preprocessing_piepline.append(Savgol_Filter())
    transform = transforms.Compose(preprocessing_piepline)

    # =========================
    # ===== Load dataset ======
    # =========================
    IRV = ["PoolData"]
    UEA = ["BasicMotions", "Epilepsy", "WalkingSittingStanding",
           "ChestMntdAcl"]
    if data_cfg.name in IRV:
        if model_name == "vision":
            train_set, test_set = get_image_dataset(data_cfg)
        else:
            train_set, test_set = get_pose_dataset(data_cfg, transform)

        if data_cfg.undersample:
            index_moving = np.where(np.array(train_set.labels) == 1)[0]
            index_notmoving = np.where(np.array(train_set.labels) == 0)[0]

            # Undersample the moving class
            index_moving = sorted(np.random.choice(
                index_moving, size=len(index_notmoving), replace=False))

            indices = index_moving
            indices.extend(index_notmoving)

            # Create a subset of the dataset
            train_set = Subset(train_set, indices)

    elif data_cfg.name in UEA:
        train_set, test_set = get_UEA_dataset(data_cfg, transform)
    else:
        raise Exception("unrecognized dataset.")

    train_set, val_set = train_val_split(train_set)
    print(f"Train set: {len(train_set)} samples")
    print(f"Validation set: {len(val_set)} samples")
    print(f"Test set: {len(test_set)} samples")

    if plot and model_name != "vision":
        save_path = os.path.join("./plots", data_cfg.name, "dataset_summary")
        os.makedirs(save_path, exist_ok=True)
        visualize_dataset(train_set, os.path.join(save_path, "train_data"))
        visualize_dataset(val_set, os.path.join(save_path, "val_data"))
        visualize_dataset(test_set, os.path.join(save_path, "test_data"))

    # ==========================
    # ===== Create loaders =====
    # ==========================
    if task == "classification":
        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True,
            num_workers=num_workers)
        val_loader = DataLoader(
            val_set, batch_size=batch_size, shuffle=False,
            num_workers=num_workers)
        test_loader = DataLoader(
            test_set, batch_size=batch_size, shuffle=False,
            num_workers=num_workers)
    elif task == "imputation":
        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, collate_fn=collate_unsuperv)
        val_loader = DataLoader(
            val_set, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, collate_fn=collate_unsuperv)
        test_loader = DataLoader(
            test_set, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, collate_fn=collate_unsuperv)
    else:
        raise Exception("unrecognized task.")

    assert len(train_loader) > 0 and len(val_loader) > 0 \
        and len(test_loader) > 0, \
        "batch size too large for dataset length"

    return train_loader, val_loader, test_loader
