"""
Compare fakeprints of two audio files.

Usage:
    python scripts/attack/fakeprints_comparaison.py data/ai/fake_00001_suno_0.mp3 data/signals/resampled/fake_00001_suno_0_rs.mp3 
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchaudio
import soxr
from scipy import interpolate

N_FFT = 1 << 14


def load_audio(filepath, target_sr=44100):
    audio, sr = torchaudio.load(filepath)
    audio = audio.mean(dim=0).numpy()
    if sr != target_sr:
        audio = soxr.resample(audio, sr, target_sr)
    return audio, target_sr


def compute_fakeprint(audio, sr, f_range=(5000, 16000)):
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
        print("Usage: python compare_fakeprints.py <audio1> <audio2>")
        sys.exit(1)
    
    audio1, sr = load_audio(sys.argv[1])
    audio2, sr = load_audio(sys.argv[2])
    
    freqs1, fp1 = compute_fakeprint(audio1, sr)
    freqs2, fp2 = compute_fakeprint(audio2, sr)
    
    plt.figure(figsize=(12, 5))
    plt.plot(freqs1 / 1000, fp1, label=sys.argv[1])
    plt.plot(freqs2 / 1000, fp2, label=sys.argv[2], alpha=0.8)
    plt.xlabel('Frequency (kHz)')
    plt.ylabel('Normalized Residual')
    plt.title('Fakeprint Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()