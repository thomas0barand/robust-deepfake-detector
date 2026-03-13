import torch

import torch.nn as nn
import lightning as L
import torch.nn.functional as F

from nnAudio.features import STFT, CQT
from torchmetrics import AUROC, F1Score, Precision, Accuracy

from src.models import LinearProj
from src.utils import get_spectrum, get_freqs, get_freqs_mask, get_fakeprints

class RobustDetector(L.LightningModule):
    def __init__(
        self,
        # Model params
        transform_type="stft",
        use_norm=True,
        use_bias=False,
        init_std=0.02,
        use_convolution=False,
        # Transform params
        freq_range=[500, 16000],
        n_fft=16384,  # 2**14
        sampling_rate=44100,
        bins_per_octave=96,
        hull_area=20,
        fmin=32.7,
        # Loss params
        pos_weight=None,
        lamb=0.1,
        # Optimizer params
        lr=1e-3,
        weight_decay=1e-5,
    ):
        super().__init__()

        self.transform_type = transform_type
        self.use_bias = use_bias
        self.use_norm = use_norm
        self.use_convolution = use_convolution
        self.freq_range = freq_range
        self.n_fft = n_fft
        self.sampling_rate = sampling_rate
        self.bins_per_octave = bins_per_octave
        self.hull_area = hull_area
        self.fmin = fmin
        self.lamb = lamb
        self.lr = lr
        self.weight_decay = weight_decay

        hop_length = n_fft // 2
        fmax = sampling_rate / 2  # Maximum frequency that can be represented

        if transform_type == "stft":
            self.transform = STFT(
                n_fft=n_fft,
                sr=sampling_rate,
                hop_length=hop_length,
                fmin=fmin,
                fmax=fmax,
                output_format="Magnitude",
                verbose=False,
            )
        elif transform_type == "cqt":
            self.transform = CQT(
                sr=sampling_rate,
                hop_length=hop_length,
                fmin=fmin,
                fmax=fmax,
                bins_per_octave=bins_per_octave,
                output_format="Magnitude",
                verbose=False,
            )
        else:
            raise ValueError(f"Unsupported transform: {transform_type}")
        

        log = (transform_type == "cqt")
        self.freqs = get_freqs(
            n_fft=n_fft,
            sr=sampling_rate,
            log=log, 
            bins_per_octave=bins_per_octave,
            fmin=fmin
        )

        self.mask = get_freqs_mask(self.freqs, sampling_rate, freq_range)

        self.feature_dim = len(self.freqs[self.mask])

        self.linear_proj = LinearProj(
            feature_dim=self.feature_dim,
            use_norm=use_norm,
            use_bias=use_bias,
            init_std=init_std,
        )

        # Handle class imbalance via pos_weight
        pw = torch.tensor([pos_weight]) if pos_weight is not None else None
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pw)

        self.auroc = AUROC(task="binary")
        self.f1 = F1Score(task="binary")
        self.precision = Precision(task="binary")
        self.accuracy = Accuracy(task="binary")

        self.save_hyperparameters()


    def extract_features(self, waveform):
        """
        waveforms: (channels, T)
        Returns: (1, feature_dim)
        """
        waveform = waveform.mean(dim=0, keepdim=True)  # Convert to mono
        spec = get_spectrum(self.transform, waveform) # (1, n_bins, T')
        spec = spec.mean(dim=-1) # (1, n_bins)
        
        spec_crop = spec[:, self.mask]
        fp = get_fakeprints(spec_crop, area=self.hull_area)  # (1, feature_dim)
        return fp


    def forward(self, x, convolve=False):
        return self.linear_proj(x, convolve=convolve)
    

    def predict(self, waveform, convolve=False):
        self.eval()
        with torch.inference_mode():
            features = self.extract_features(waveform.to(self.linear_proj.weights.device))
            features = features[:, self.mask]  # Apply same frequency mask as during training
            logits, _ = self(features, convolve=convolve)
            probs = torch.sigmoid(logits)
        return probs.item()


    def training_step(self, batch, batch_idx):
        fp, label, lag_idx = batch
        logits, cross_corr = self(fp, convolve=self.use_convolution)

        class_loss = self.bce_loss(logits.squeeze(-1), label)

        if self.use_convolution:
            mask = label == 1
            reg_loss = F.cross_entropy(cross_corr[mask], lag_idx[mask].long())
            loss = class_loss + self.lamb * reg_loss
            self.log('train_class_loss', class_loss)
            self.log('train_reg_loss', reg_loss)
        else:
            loss = class_loss

        self.log('train_loss', loss)
        return loss


    def validation_step(self, batch, batch_idx):
        fp, label, lag_idx = batch
        logits, cross_corr = self(fp, convolve=self.use_convolution)
        probs = torch.sigmoid(logits).squeeze(-1)
        
        class_loss = self.bce_loss(logits.squeeze(-1), label)

        if self.use_convolution:
            mask = label == 1
            reg_loss = F.cross_entropy(cross_corr[mask], lag_idx[mask].long())
            loss = class_loss + self.lamb * reg_loss
            self.log('val_class_loss', class_loss)
            self.log('val_reg_loss', reg_loss)
        else:
            loss = class_loss

        self.auroc.update(probs, label.long())
        self.f1.update(probs, label.long())
        self.precision.update(probs, label.long())
        self.accuracy.update(probs, label.long())

        self.log('val_loss', loss, prog_bar=True)
        return loss


    def on_validation_epoch_end(self):
        self.log('val_auroc', self.auroc.compute(), prog_bar=True)
        self.log('val_f1', self.f1.compute(), prog_bar=True)
        self.log('val_precision', self.precision.compute(), prog_bar=True)
        self.log('val_accuracy', self.accuracy.compute(), prog_bar=True)

        self.auroc.reset()
        self.f1.reset()
        self.precision.reset()
        self.accuracy.reset()
 

    def test_step(self, batch, batch_idx):
        fp, label, lag_idx = batch
        logits, cross_corr = self(fp, convolve=self.use_convolution)
        probs = torch.sigmoid(logits).squeeze(-1)

        self.auroc.update(probs, label.long())
        self.f1.update(probs, label.long())
        self.precision.update(probs, label.long())
        self.accuracy.update(probs, label.long())


    def on_test_epoch_end(self):
        self.log('test_auroc', self.auroc.compute())
        self.log('test_f1', self.f1.compute())
        self.log('test_precision', self.precision.compute())
        self.log('test_accuracy', self.accuracy.compute())

        self.auroc.reset()
        self.f1.reset()
        self.precision.reset()
        self.accuracy.reset()


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
    

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        # Remove transform from state dict
        keys_to_remove = [k for k in checkpoint["state_dict"] if k.startswith("transform")]
        for k in keys_to_remove:
            del checkpoint["state_dict"][k]

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        # Restore transform state
        cqt_state = {k: v for k, v in self.transform.state_dict().items()}
        for k, v in cqt_state.items():
            checkpoint["state_dict"][f"transform.{k}"] = v