# training/dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np

class TimeSeriesDataset(Dataset):
    def __init__(self, series, lookback, horizons):
        """
        series: raw prices (1D numpy array)
        """
        self.raw_series = series.astype(np.float32)
        self.lookback = lookback
        self.horizons = horizons

        # log-diff transform
        log_series = np.log(self.raw_series + 1e-8)
        self.diff_series = np.diff(log_series)

    def __len__(self):
        return len(self.diff_series) - self.lookback - max(self.horizons)

    def __getitem__(self, idx):
        x = self.diff_series[idx:idx + self.lookback]

        y = [
            self.diff_series[idx + self.lookback + h - 1]
            for h in self.horizons
        ]

        # last known raw price (needed for reconstruction)
        last_price = self.raw_series[idx + self.lookback]

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(last_price, dtype=torch.float32),
        )