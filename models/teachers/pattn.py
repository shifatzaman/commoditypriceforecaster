# models/teachers/pattn.py
import torch
import torch.nn as nn

class PAttn(nn.Module):
    def __init__(self, lookback, horizons, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=lookback,
            num_heads=heads,
            batch_first=True
        )
        self.fc = nn.Linear(lookback, len(horizons))

    def forward(self, x):
        # x: (B, L)
        x = x.unsqueeze(1)          # (B, 1, L)
        z, _ = self.attn(x, x, x)   # self-attention
        return self.fc(z.squeeze(1))