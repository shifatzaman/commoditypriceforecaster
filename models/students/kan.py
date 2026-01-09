import torch
import torch.nn as nn

# NOTE: This is a lightweight placeholder KAN-style student.
# Swap with a full KAN implementation later without changing the training pipeline.
class KAN(nn.Module):
    def __init__(self, lookback, horizons, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(lookback, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, len(horizons)),
        )

    def forward(self, x):
        return self.net(x)
