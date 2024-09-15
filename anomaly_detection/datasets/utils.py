import copy
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter
import torch
from torch.utils.data import Dataset, random_split


'''
DATASET PREPROCESSING
'''


class Standardize(object):
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


class Savgol_Filter(object):
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


def feat_noise_mask(L, lm, masking_ratio):
    mask = np.ones(L, dtype=bool)
    p_m = 1 / lm
    p_u = p_m * masking_ratio / (1 - masking_ratio)
    p = [p_m, p_u]

    state = int(np.random.rand() > masking_ratio)
    for i in range(L):
        mask[i] = state
        if np.random.rand() < p[state]:
            state = 1 - state

    return mask


def noise_mask(ts, lm, r):
    L = ts.shape[0]

    mask = np.ones(ts.shape, dtype=bool)
    for feature in range(ts.shape[1]):
        mask[:, feature] = feat_noise_mask(L, lm, r)
    return mask


def train_val_split(train_dataset: Dataset, val_frac=0.2):
    train_size = len(train_dataset)
    val_size = int(val_frac * train_size)
    train, val = random_split(train_dataset, [train_size - val_size, val_size])
    return train, val


def collate_unsuperv(batch, mean_mask_length=3, masking_ratio=0.15):
    """Build targets for imputation task.
    Args:
        batch: list of tuples (X, label).
            - X: numpy array of shape (seq_length, feat_dim).
            - label: int, class label.

    Returns:
        ts: (batch_size, seq_length, feat_dim),
            torch tensor of masked features.
        lbl: (batch_size, seq_length, feat_dim),
            torch tensor of unmasked features.
        masks: (batch_size, seq_length, feat_dim),
            boolean torch tensor, 1 indicates masked values to be predicted,
            0 indicates unaffected/"active" feature values.
    """
    X, _ = zip(*batch)

    ts = torch.zeros((len(X), *(X[0].shape)), dtype=torch.float32)
    lbl = torch.zeros((len(X), *(X[0].shape)), dtype=torch.float32)
    masks = torch.zeros((len(X), *(X[0].shape)), dtype=torch.bool)

    for i, x in enumerate(X):
        mask = noise_mask(x, lm=mean_mask_length, r=masking_ratio)

        target = copy.deepcopy(x)
        masked_input = x * mask

        target_masks = ~mask

        ts[i] = torch.from_numpy(masked_input)
        lbl[i] = torch.from_numpy(target)
        masks[i] = torch.from_numpy(target_masks)

    return ts, lbl, masks
