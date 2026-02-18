import torch
import torch.nn as nn

import torch.nn.functional as F

class LinearProj(nn.Module):
    def __init__(
        self,
        feature_dim,
        use_bias=False,
        init_std=0.02,
    ):
        super(LinearProj, self).__init__()

        self.feature_dim = feature_dim
        self.weights = nn.Parameter(torch.randn(1, feature_dim) * init_std)
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(1))

        self.pooling = nn.AdaptiveMaxPool1d(1)

    def forward(self, x, convolve=False):
        
        x = F.rms_norm(x, normalized_shape=[self.feature_dim], eps=1e-8)
        if convolve:
            x = F.conv1d(x.unsqueeze(1), self.weights.unsqueeze(1), padding=self.feature_dim-1).squeeze(1)
            x = self.pooling(x)
        else:
            x = torch.matmul(x, self.weights.T)

        if hasattr(self, 'bias'):
            x += self.bias

        return x