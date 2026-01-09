import torch.nn as nn
class MLP(nn.Module):
    def __init__(self,l,h): super().__init__(); self.fc=nn.Linear(l,len(h))
    def forward(self,x): return self.fc(x.float())