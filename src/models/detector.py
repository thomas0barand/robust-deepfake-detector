import torch

import numpy as np
import torch.nn as nn
import lightning as L

from torchaudio.transforms import Spectrogram
from nnAudio.features import CQT
from torchmetrics import AUROC, F1Score, Accuracy

from src.models.linear import LinearProj
from src.models.utils import get_spectrum, get_freqs, get_fakeprints

class RobustDetector(L.LightningModule):
    def __init__(
        self,
        # Model params
        feature_dim,
        use_cqt=True,
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
        # Equivariance params
        alpha_loss_weight=0.0,
        alpha_speed_range=[0.8, 1.25],
        alpha_std=None,
        softmax_temperature=1.0,
        equi_warmup_epochs=0,
        use_adaptive_weights=False,
        # Optimizer params
        lr=1e-3,
        weight_decay=1e-5,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.use_cqt = use_cqt
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
        self.alpha_loss_weight = alpha_loss_weight
        self.softmax_temperature = softmax_temperature
        self.equi_warmup_epochs = equi_warmup_epochs
        self.use_adaptive_weights = use_adaptive_weights

        # Alpha0 range in CQT bins from speed range
        self.alpha_min = int(np.ceil(np.log2(alpha_speed_range[0]) * bins_per_octave))
        self.alpha_max = int(np.floor(np.log2(alpha_speed_range[1]) * bins_per_octave))
        self.alpha_std = alpha_std if alpha_std is not None else (self.alpha_max - self.alpha_min) / 4.0

        hop_length = n_fft // 2
        nyquist = sampling_rate / 2
        n_octaves = np.log2(nyquist / f_min) - 0.1
        nbins = int(n_octaves * bins_per_octave)

        self.cqt_transform = CQT(
            sr=sampling_rate,
            hop_length=hop_length,
            fmin=f_min,
            n_bins=nbins,
            bins_per_octave=bins_per_octave,
            output_format="Magnitude",
            verbose=False,
        )

        self.stft_transform = Spectrogram(n_fft=n_fft, power=2, hop_length=hop_length)

        self.freqs, self.freq_mask = get_freqs(
            n_fft=n_fft,
            sr=sampling_rate,
            transform="cqt" if use_cqt else "stft",
            bins_per_octave=bins_per_octave,
            freq_range=freq_range,
            f_min=f_min
        )

        self.linear_proj = LinearProj(
            feature_dim=feature_dim,
            use_bias=use_bias,
            use_norm=use_norm,
            init_std=init_std,
        )

        # Handle class imbalance via pos_weight
        pw = torch.tensor([pos_weight]) if pos_weight is not None else None
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
        self.equi_loss_fn = nn.SmoothL1Loss()

        # Adaptive multi-task weighting (Kendall et al.)
        if use_adaptive_weights:
            self.log_var_bce = nn.Parameter(torch.zeros(1))
            self.log_var_equi = nn.Parameter(torch.zeros(1))

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
        transform = self.cqt_transform if self.use_cqt else self.stft_transform
        spec = get_spectrum(transform, waveform) # (1, n_bins, T')
        spec = spec.mean(dim=-1).squeeze(0)  # (n_bins,)
        
        spec_crop = spec[self.freq_mask]
        fp = get_fakeprints(spec_crop, self.freqs)
        return fp.unsqueeze(0)  # (1, feature_dim)


    def sample_alpha0(self, batch_size, device):
        """Truncated normal via rejection sampling — no edge spikes."""
        alpha0 = torch.empty(batch_size)
        remaining = torch.arange(batch_size)
        while len(remaining) > 0:
            candidates = torch.normal(0, self.alpha_std, size=(len(remaining),))
            valid = (candidates >= self.alpha_min) & (candidates <= self.alpha_max)
            alpha0[remaining[valid]] = candidates[valid]
            remaining = remaining[~valid]
        return alpha0.round().long().to(device)

    @staticmethod
    def shift_fakeprints(x, alpha0):
        """Roll fakeprints by alpha0 bins with zero-fill. x: (B, F), alpha0: (B,) long tensor."""
        B, F = x.shape
        idx = torch.arange(F, device=x.device).unsqueeze(0)
        src_idx = idx - alpha0.unsqueeze(1)
        valid = (src_idx >= 0) & (src_idx < F)
        src_idx = src_idx.clamp(0, F - 1)
        return torch.gather(x, 1, src_idx) * valid.float()

    def soft_argmax(self, x):
        """Soft argmax over last dim. x: (B, L) -> (B,)"""
        indices = torch.arange(x.shape[-1], device=x.device, dtype=x.dtype)
        weights = torch.softmax(x / self.softmax_temperature, dim=-1)
        return (weights * indices).sum(dim=-1)

    def forward(self, x, convolve=False, return_conv=False):
        return self.linear_proj(x, convolve=convolve, return_conv=return_conv)
    

    def predict(self, waveform, convolve=False):
        self.eval()
        with torch.inference_mode():
            features = self.extract_features(waveform.to(self.linear_proj.weights.device))
            logits = self(features, convolve=convolve)
            probs = torch.sigmoid(logits)
        return probs.item()


    def _compute_equivariance_loss(self, x, y, conv_out):
        """Invariance loss: shifted samples should keep their original label.
        Applied to AI-only or all samples based on equi_all_classes flag.
        Also monitors lag shift via hard argmax (non-differentiable indicator)."""
        if not (conv_out is not None and self.alpha_loss_weight > 0):
            return torch.tensor(0.0, device=x.device), torch.tensor(0.0, device=x.device)

        alpha0 = self.sample_alpha0(x.shape[0], x.device)
        x_shifted = self.shift_fakeprints(x, alpha0)
        logits_shifted, conv_shifted = self(x_shifted, convolve=True, return_conv=True)

        equi_loss = self.loss_fn(logits_shifted.squeeze(-1), y)

        with torch.no_grad():
            ai_mask = (y == 1)
            if ai_mask.any():
                lag_orig = conv_out[ai_mask].argmax(dim=-1).float()
                lag_shifted = conv_shifted[ai_mask].argmax(dim=-1).float()
                mean_lag_error = ((lag_shifted - lag_orig) - alpha0[ai_mask].float()).abs().mean()
            else:
                mean_lag_error = torch.tensor(0.0, device=x.device)

        return equi_loss, mean_lag_error

    def _get_equi_weight(self):
        if self.alpha_loss_weight <= 0:
            return 0.0
        if self.equi_warmup_epochs <= 0:
            return self.alpha_loss_weight
        progress = min(self.current_epoch / self.equi_warmup_epochs, 1.0)
        return self.alpha_loss_weight * progress

    def _combine_losses(self, bce_loss, equi_loss):
        w = self._get_equi_weight()
        if self.use_adaptive_weights:
            prec_bce = torch.exp(-self.log_var_bce)
            prec_equi = torch.exp(-self.log_var_equi)
            return 0.5 * (prec_bce * bce_loss + self.log_var_bce + prec_equi * equi_loss + self.log_var_equi)
        return bce_loss + w * equi_loss

    def training_step(self, batch, batch_idx):
        x, y = batch

        if self.use_convolution:
            logits, conv_out = self(x, convolve=True, return_conv=True)
        else:
            logits = self(x, convolve=False)
            conv_out = None

        bce_loss = self.loss_fn(logits.squeeze(-1), y)
        equi_loss, mean_lag_err = self._compute_equivariance_loss(x, y, conv_out)
        loss = self._combine_losses(bce_loss, equi_loss)

        self.log('train_loss', loss)
        self.log('train_bce', bce_loss)
        self.log('train_equi', equi_loss)
        self.log('train_lag_mae', mean_lag_err)
        if self.use_adaptive_weights:
            self.log('train_w_bce', torch.exp(-self.log_var_bce))
            self.log('train_w_equi', torch.exp(-self.log_var_equi))
        return loss


    def validation_step(self, batch, batch_idx):
        x, y = batch

        if self.use_convolution:
            logits, conv_out = self(x, convolve=True, return_conv=True)
        else:
            logits = self(x, convolve=False)
            conv_out = None

        bce_loss = self.loss_fn(logits.squeeze(-1), y)
        equi_loss, mean_lag_err = self._compute_equivariance_loss(x, y, conv_out)
        loss = self._combine_losses(bce_loss, equi_loss)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_bce', bce_loss)
        self.log('val_equi', equi_loss)
        self.log('val_lag_mae', mean_lag_err)
        self.log('val_auroc', self.auroc(logits.squeeze(-1), y), on_epoch=True, prog_bar=True)
        self.log('val_f1', self.f1(logits.squeeze(-1), y), on_epoch=True, prog_bar=True)
        self.log('val_accuracy', self.accuracy(logits.squeeze(-1), y), on_epoch=True, prog_bar=True)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
    

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        # Remove CQT transform from state dict
        keys_to_remove = [k for k in checkpoint["state_dict"] if k.startswith("cqt_transform")]
        for k in keys_to_remove:
            del checkpoint["state_dict"][k]

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        cqt_state = {k: v for k, v in self.cqt_transform.state_dict().items()}
        for k, v in cqt_state.items():
            checkpoint["state_dict"][f"cqt_transform.{k}"] = v