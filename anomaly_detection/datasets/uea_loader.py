import os
import numpy as np
from torch.utils.data import Dataset
from aeon.datasets import load_from_tsfile


class ClassificationDataset(Dataset):
    def __init__(self, data_fp, num_classes, num_features, sample_frequency,
                 window_size=5, window_overlap=0.5, transform=None):

        assert os.path.exists(data_fp), \
            f"Data file path '{data_fp}' does not exist."

        # ts file needs thte equal lenght hyperparam
        x, y = load_from_tsfile(data_fp)
        x = x.astype('float32')

        x = x.transpose(0, 2, 1)
        # shape of x: [num_samples, num_timesteps, num_features]
        # shape of y: [num_samples]

        self.num_classes = num_classes
        self.num_features = num_features
        self.sample_frequency = sample_frequency
        self.window_size = window_size
        self.window_overlap = window_overlap
        self.transform = transform

        classes = np.unique(y)
        assert len(classes) == num_classes, "Number of classes do not match"
        class_dict = {classes[i]: i for i in range(num_classes)}

        if x.shape[1] > self.window_size * self.sample_frequency:
            self.clips = []
            for i, ts in enumerate(x):
                self.clips.extend(self._clip(ts, np.array(class_dict[y[i]])))
        elif x.shape[1] == self.window_size * self.sample_frequency:
            self.clips = [(ts, np.array(class_dict[lbl]))
                          for ts, lbl in zip(x, y)]
        else:
            raise ValueError(
                "Time series length is less than window size * "
                "sample frequency")

    def _clip(self, series, act):

        # Create clips from the time series data, with each clip
        # being of length window_size * sample_frequency,
        # with overlap of window_overlap * 100 percent. For example:
        # data   -> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        # freq -> 1 Hz
        # window -> 5
        # window_overlap -> 0.2
        # clips  -> [[0, 1, 2, 3, 4], [4, 5, 6, 7, 8]]

        window = self.window_size * self.sample_frequency
        # number of steps to slide window on each iter
        step = int(window - window*self.window_overlap)

        curr = 0
        series_list = []
        while curr <= len(series) - window:
            s = series[
                curr:curr + window:
            ]  # need to check dimensions
            curr += step
            series_list.append((s, act))
        return series_list

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        ts, lbl = self.clips[idx]

        # shape of series sample: (channels, timesteps)
        if self.transform:
            ts = self.transform(ts)

        return ts, lbl
