import torch
import torch.nn as nn

class TimeMixer(nn.Module):
    def __init__(self, lookback, horizons):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(lookback, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, len(horizons)),
        )

    def forward(self, x):
        return self.net(x)