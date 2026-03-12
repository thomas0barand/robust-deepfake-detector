import torch
import torch.nn as nn

import torch.nn.functional as F

class LinearProj(nn.Module):
    def __init__(
        self,
        feature_dim,
        use_norm=True,
        use_bias=False,
        init_std=0.02,
    ):
        super(LinearProj, self).__init__()

        self.feature_dim = feature_dim
        self.use_norm = use_norm
        self.weights = nn.Parameter(torch.randn(1, feature_dim) * init_std)
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter('bias', None)

    def forward(self, x, convolve=False):
        
        if self.use_norm:
            x = F.normalize(torch.clip(x, min=None, max=8), dim=-1)  # (B, F)
            weights = F.normalize(self.weights, dim=-1)  # (1, F)

        if convolve:
            x_conv = x.unsqueeze(1)  # (B, 1, F)
            w_conv = weights.unsqueeze(1)  # (1, 1, F)
            cross_corr = F.conv1d(x_conv, w_conv, padding="same").squeeze(1) # (B, 1, F) x (1, 1, F) -> (B, F)
            logits, _ = torch.max(cross_corr, dim=1)  # (B, F) -> (B, 1)
        else:
            logits = torch.matmul(x, self.weights.T) # (B, F) x (F, 1) -> (B, 1)
            cross_corr = None
        
        if self.bias is not None:
            logits = logits + self.bias # (B, 1)

        return logits, cross_corr