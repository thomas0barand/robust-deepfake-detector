import torch

import numpy as np
import torch.nn as nn
import lightning as L

from nnAudio.features import CQT
from torchmetrics import AUROC, F1Score, Accuracy

from src.models.linear import LinearProj
from src.models.utils import get_cqt, get_freqs, get_fakeprints

class RobustDetector(L.LightningModule):
    def __init__(
        self,
        # Model params
        feature_dim,
        use_bias=True,
        use_norm=True,
        init_std=0.02,
        use_convolution=False,
        # CQT params
        n_fft=16384,  # 2**14
        sampling_rate=48000,
        bins_per_octave=96,
        freq_range=[200, 6000],
        f_min=32.7,
        # Loss params
        pos_weight=None,
        # Optimizer params
        lr=1e-3,
        weight_decay=1e-5,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.use_bias = use_bias
        self.use_norm = use_norm
        self.use_convolution = use_convolution
        self.n_fft = n_fft
        self.sampling_rate = sampling_rate
        self.bins_per_octave = bins_per_octave
        self.freq_range = freq_range
        self.f_min = f_min
        self.lr = lr
        self.weight_decay = weight_decay

        hop_length = n_fft // 2
        nyquist = sampling_rate / 2  # Maximum frequency that can be represented
        n_octaves = np.log2(nyquist / f_min) - 0.1  # Subtract a small margin to ensure we don't exceed Nyquist
        nbins = int(n_octaves * bins_per_octave)  # Total number of CQT bins to cover the desired frequency range

        self.cqt_layer = CQT(
            sr=sampling_rate,
            hop_length=hop_length,
            fmin=f_min,
            n_bins=nbins,
            bins_per_octave=bins_per_octave,
            output_format="Magnitude",
            verbose=False,
        )

        self.freqs, self.freq_mask = get_freqs(nbins, sampling_rate, bins_per_octave, freq_range=freq_range, f_min=f_min)

        self.linear_proj = LinearProj(
            feature_dim=feature_dim,
            use_bias=use_bias,
            use_norm=use_norm,
            init_std=init_std,
        )

        # Handle class imbalance via pos_weight
        pw = torch.tensor([pos_weight]) if pos_weight is not None else None
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

        self.auroc = AUROC(task="binary")
        self.f1 = F1Score(task="binary")
        self.accuracy = Accuracy(task="binary")

        self.save_hyperparameters()


    def extract_features(self, waveform):
        """
        waveforms: (channels, T)
        Returns: (1, feature_dim)
        """
        waveform = waveform.mean(dim=0, keepdim=True)  # Convert to mono
        cqt = get_cqt(self.cqt_layer, waveform) # (1, n_bins, T')
        spec = cqt.mean(dim=-1).squeeze(0)  # (n_bins,)
        
        spec_crop = spec[self.freq_mask]
        fp = get_fakeprints(spec_crop, self.freqs)
        return fp.unsqueeze(0)  # (1, feature_dim)


    def forward(self, x, convolve=False):
        return self.linear_proj(x, convolve=convolve)
    

    def predict(self, waveform, convolve=False):
        self.eval()
        with torch.inference_mode():
            features = self.extract_features(waveform.to(self.linear_proj.weights.device))
            logits = self(features, convolve=convolve)
            probs = torch.sigmoid(logits)
        return probs.item()


    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x, convolve=self.use_convolution)
        loss = self.loss_fn(logits.squeeze(-1), y)

        self.log('train_loss', loss)
        return loss


    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x, convolve=self.use_convolution)
        loss = self.loss_fn(logits.squeeze(-1), y)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_auroc', self.auroc(logits.squeeze(-1), y), on_epoch=True, prog_bar=True)
        self.log('val_f1', self.f1(logits.squeeze(-1), y), on_epoch=True, prog_bar=True)
        self.log('val_accuracy', self.accuracy(logits.squeeze(-1), y), on_epoch=True, prog_bar=True)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
    

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        # Remove CQT layer from state dict
        keys_to_remove = [k for k in checkpoint["state_dict"] if k.startswith("cqt_layer")]
        for k in keys_to_remove:
            del checkpoint["state_dict"][k]

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        cqt_state = {k: v for k, v in self.cqt_layer.state_dict().items()}
        for k, v in cqt_state.items():
            checkpoint["state_dict"][f"cqt_layer.{k}"] = v