"""
Compare fakeprints of two audio files in log-frequency space with cross-correlation.

Usage:
python scripts/attack/soxr/visualise/fakeprints_log_correlation.py data/signals/originals/signal1.mp3 data/signals/spedup/signal1_s.mp3
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchaudio
from scipy import interpolate

N_FFT = 1 << 14


def load_audio(filepath, target_sr=44100):
    audio, sr = torchaudio.load(filepath)
    audio = audio.mean(dim=0).numpy()
    return audio, sr


def compute_fakeprint(audio, sr, f_range=(100, 20000)):
    # Step 1: Average spectrum
    spec = torchaudio.transforms.Spectrogram(n_fft=N_FFT, power=2)(torch.Tensor(audio))
    spectrum = 10 * np.log10(np.clip(spec.mean(dim=1).numpy(), 1e-10, None))
    freqs = np.linspace(0, sr / 2, len(spectrum))
    
    # Crop to frequency range
    mask = (freqs > f_range[0]) & (freqs < f_range[1])
    freqs, spectrum = freqs[mask], spectrum[mask]
    
    # Step 2: Compute lower envelope
    hull_idx, hull_vals = [0], [spectrum[0]]
    for i in range(len(spectrum) - 10):
        idx = i + np.argmin(spectrum[i:i+10])
        if idx != hull_idx[-1]:
            hull_idx.append(idx)
            hull_vals.append(spectrum[idx])
    hull_idx.append(len(spectrum) - 1)
    hull_vals.append(spectrum[-1])
    
    # Step 3: Subtract baseline and normalize
    baseline = interpolate.interp1d(freqs[hull_idx], hull_vals, kind='quadratic')(freqs)
    residual = np.clip(spectrum - baseline, 0, None)
    fingerprint = residual / (residual.max() + 1e-6)
    
    return freqs, fingerprint


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_fakeprints_log.py <audio1> <audio2>")
        sys.exit(1)
    
    audio1, sr = load_audio(sys.argv[1])
    audio2, sr = load_audio(sys.argv[2])
    
    freqs1, fp1 = compute_fakeprint(audio1, sr)
    freqs2, fp2 = compute_fakeprint(audio2, sr)
    
    # Interpolate to uniform log-frequency grid
    log_freqs1 = np.log10(freqs1)
    log_freqs2 = np.log10(freqs2)
    
    # Create uniform log-frequency axis
    log_freq_uniform = np.linspace(log_freqs1.min(), log_freqs1.max(), len(fp1))
    bin_width = log_freq_uniform[1] - log_freq_uniform[0]
    
    # Interpolate fakeprints to uniform log grid
    fp1_uniform = np.interp(log_freq_uniform, log_freqs1, fp1)
    fp2_uniform = np.interp(log_freq_uniform, log_freqs2, fp2)
    
    # Keep only peaks (threshold out noise)
    threshold = 0
    fp1_peaks = np.where(fp1_uniform > threshold, fp1_uniform, 0)
    fp2_peaks = np.where(fp2_uniform > threshold, fp2_uniform, 0)
    
    # Cross-correlation
    correlation = np.correlate(fp1_peaks, fp2_peaks, mode='full')
    
    # Compute the shift axis (in log10 units)
    lags = np.arange(-len(fp2_peaks) + 1, len(fp1_peaks)) * bin_width
    
    # Find peak
    peak_idx = np.argmax(correlation)
    peak_lag = lags[peak_idx]
    estimated_alpha = 10 ** (-peak_lag)  # flip sign: "how much was signal2 sped up relative to signal1"
    corr_coef = correlation[peak_idx] / (np.linalg.norm(fp1_peaks) * np.linalg.norm(fp2_peaks))

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Top: fakeprints
    axes[0].plot(log_freq_uniform, fp1_uniform, label=sys.argv[1])
    axes[0].plot(log_freq_uniform, fp2_uniform, label=sys.argv[2], alpha=0.8)
    axes[0].set_xlabel('log10(Frequency)')
    axes[0].set_ylabel('Normalized Residual')
    axes[0].set_title('Fakeprint Comparison (log-frequency)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Bottom: cross-correlation
    axes[1].plot(lags, correlation)
    axes[1].axvline(peak_lag, color='r', linestyle='--', label=f'Peak at {peak_lag:.4f}')
    axes[1].set_xlabel('Lag (log10 units)')
    axes[1].set_ylabel('Correlation')
    axes[1].set_title(f'Cross-correlation → estimated speed factor α = {estimated_alpha:.4f}, correlation coef = {corr_coef:.3f}, threshold = {threshold}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()