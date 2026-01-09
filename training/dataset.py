
import torch
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, series, lookback, horizons):
        self.series = series
        self.lookback = lookback
        self.horizons = horizons

    def __len__(self):
        return len(self.series) - self.lookback - max(self.horizons)

    def __getitem__(self, idx):
        x = self.series[idx:idx+self.lookback]
        y = [self.series[idx+self.lookback+h-1] for h in self.horizons]
        return torch.tensor(x), torch.tensor(y)
