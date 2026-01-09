import torch
import torch.nn as nn

class PatchTST(nn.Module):
    def __init__(self, lookback, horizons):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=lookback,
                nhead=4,
                batch_first=True,
            ),
            num_layers=2
        )
        self.fc = nn.Linear(lookback, len(horizons))

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, L)
        z = self.encoder(x)
        return self.fc(z.mean(dim=1))