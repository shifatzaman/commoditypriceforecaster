
import torch
import torch.nn as nn

class PatchTST(nn.Module):
    def __init__(self, lookback, horizons):
        super().__init__()
        self.fc = nn.Linear(lookback, len(horizons))

    def forward(self, x):
        return self.fc(x)
