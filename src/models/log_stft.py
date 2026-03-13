import math
import torch
import torch.nn.functional as F
from nnAudio.features import STFT

class LogSTFT(STFT):
    def __init__(
        self,
        n_fft=16384,
        sr=44100,
        hop_length=512,
        fmin=32.7,
        fmax=None,
        bins_per_octave=1920,
        output_format="Magnitude",
        verbose=False,
    ):
        fmax = fmax or sr / 2
        super().__init__(
            n_fft=n_fft,
            sr=sr,
            hop_length=hop_length,
            fmin=fmin,
            fmax=fmax,
            output_format=output_format,
            verbose=verbose,
        )
        self.bins_per_octave = bins_per_octave

        # Linear STFT frequencies
        stft_freqs = torch.linspace(0, sr / 2, n_fft // 2 + 1)  # (n_fft//2 + 1,)

        # Log-spaced target frequencies
        n_log_bins = int(math.ceil(bins_per_octave * math.log2(fmax / fmin)))
        log_freqs = torch.logspace(
            math.log10(fmin),
            math.log10(fmax),
            steps=n_log_bins,
        )  # (n_log_bins,)

        # Precompute interpolation indices and weights
        freq_indices = (log_freqs - stft_freqs[0]) / (stft_freqs[1] - stft_freqs[0])
        freq_indices = freq_indices.clamp(0, len(stft_freqs) - 1)
        idx_low = freq_indices.long()
        idx_high = (idx_low + 1).clamp(max=len(stft_freqs) - 1)
        alpha = freq_indices - idx_low.float()

        self.register_buffer('idx_low', idx_low)
        self.register_buffer('idx_high', idx_high)
        self.register_buffer('alpha', alpha)
        self.n_log_bins = n_log_bins

    def forward(self, waveform):
        spec = super().forward(waveform)  # (B, n_bins, T)

        # Log-resample along frequency axis
        spec_log = (1 - self.alpha) * spec[:, self.idx_low, :] + self.alpha * spec[:, self.idx_high, :]  # (B, n_log_bins, T)
        return spec_log