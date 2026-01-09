import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, lookback, horizons, hidden=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(lookback, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden//2, len(horizons)),
        )

    def forward(self, x):
        return self.net(x)
