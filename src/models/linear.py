import torch
import torch.nn as nn

import torch.nn.functional as F

class LinearProj(nn.Module):
    def __init__(
        self,
        feature_dim,
        use_bias=True,
        use_norm=True,
        init_std=0.02,
    ):
        super(LinearProj, self).__init__()

        self.feature_dim = feature_dim
        self.use_norm = use_norm
        self.weights = nn.Parameter(torch.randn(1, feature_dim) * init_std)
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(1))

        self.pooling = nn.AdaptiveMaxPool1d(1)

    def forward(self, x, convolve=False):

        x = F.rms_norm(x, normalized_shape=[self.feature_dim], eps=1e-8) if self.use_norm else x # (B, F)
        if convolve:
            x = F.conv1d(x.unsqueeze(1), self.weights.unsqueeze(1), padding=self.feature_dim-1).squeeze(1) # (B, F) x (1, F) -> (B, 2F-1)
            x = self.pooling(x) # (B, 1)
        else:
            x = torch.matmul(x, self.weights.T) # (B, F) x (F, 1) -> (B, 1)

        if hasattr(self, 'bias'):
            x += self.bias # (B, 1)

        return x