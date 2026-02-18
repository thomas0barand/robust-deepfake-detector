import torch

import lightning as L
import torch.nn as nn

from src.models.linear import LinearProj

class RobustDetector(L.LightningModule):
    def __init__(
        self,
        feature_dim,
        use_bias=False,
        init_std=0.02,
        use_convolution=False,
    ):
        super().__init__()
        self.use_convolution = use_convolution
        self.linear_proj = LinearProj(feature_dim, use_bias=use_bias, init_std=init_std)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()

        self.save_hyperparameters()

    def forward(self, x, convolve=False):
        return self.linear_proj(x, convolve=convolve)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x, convolve=self.use_convolution)
        loss = self.loss_fn(logits, y)

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x, convolve=self.use_convolution)
        loss = self.loss_fn(logits, y)

        self.log('val_loss', loss)