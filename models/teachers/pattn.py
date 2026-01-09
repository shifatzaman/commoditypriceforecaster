
import torch
import torch.nn as nn

class PAttn(nn.Module):
    def __init__(self, lookback, horizons):
        super().__init__()
        self.fc = nn.Linear(lookback, len(horizons))

    def forward(self, x):
        return self.fc(x)
