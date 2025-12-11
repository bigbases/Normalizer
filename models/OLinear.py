import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):
    """
    Normalization-Linear
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        slice_len = 48
        n_slice = self.pred_len // slice_len
        self.Linears = nn.ModuleList([nn.Linear(self.seq_len + i*slice_len, slice_len) for i in range(n_slice)])

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        for i, linear in enumerate(self.Linears):
            seq_last = x[:,-1:,:].detach()
            x = x - seq_last
            x_slice = linear(x.permute(0,2,1)).permute(0,2,1)
            x_slice = x_slice + seq_last
            if i == 0:
                x_out = x_slice
            else:
                x_out = torch.cat([x_out, x_slice], dim=1)
            x = torch.cat([x, x_slice], dim=1)
        return x # [Batch, Output length, Channel]