"""
Compare the fakeprint of an original audio file against a sped-up version.

Produces two plots:
  1. Overlay of both fakeprints.
  2. Cross-correlation between them (measures shape similarity & lag).
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torchaudio
from scipy.signal import correlate

from utils import get_stft, get_cqt, get_fakeprint, refine_peak

# ── Configuration ────────────────────────────────────────────────────────────

N_FFT = 1 << 14            # 16384
SAMPLE_RATE = 16000
HOP_LENGTH = N_FFT // 2
F_MIN = 32.7                # C1 note frequency
BINS_PER_OCTAVE = 96
F_RANGE = (500, 5000)       # Hz — frequency band of interest
MODE = "cqt"               # "stft" or "cqt"

# CQT bin count (spans from F_MIN up to just below Nyquist)
_nyquist = SAMPLE_RATE / 2
_n_octaves = np.log2(_nyquist / F_MIN) - 0.1
N_BINS = int(_n_octaves * BINS_PER_OCTAVE)


# ── Plotting helpers ─────────────────────────────────────────────────────────

def plot_fakeprints(
    freqs_orig: np.ndarray,
    fp_orig: np.ndarray,
    freqs_mod: np.ndarray,
    fp_mod: np.ndarray,
    mode: str = "stft",
) -> None:
    """Overlay two fakeprints on the same axes."""
    sns.set_theme()
    plt.figure(figsize=(12, 5))

    sns.lineplot(x=freqs_orig, y=fp_orig, label="Original")
    sns.lineplot(x=freqs_mod, y=fp_mod, label="Sped-up", alpha=0.8)

    if mode == "stft":
        plt.xlabel("Log Frequency")
    else:
        plt.xscale("log")
        plt.xlabel("Frequency (Hz)")

    plt.ylabel("Normalized Residue")
    plt.title("Fakeprint Comparison")
    plt.legend()
    plt.show()


def plot_cross_correlation(
    fp_orig: np.ndarray,
    fp_mod: np.ndarray,
    label: str = "STFT",
) -> None:
    """Compute and plot the normalized cross-correlation of two fakeprints."""
    a = fp_orig.flatten()
    b = fp_mod.flatten()

    # Normalize (Pearson-style)
    a_norm = (a - np.mean(a)) / (np.std(a) * len(a))
    b_norm = (b - np.mean(b)) / np.std(b)

    corr = correlate(a_norm, b_norm, mode="same")
    lags = np.arange(-len(corr) // 2, len(corr) // 2)

    plt.figure(figsize=(12, 5))
    plt.plot(lags, corr, alpha=0.8, label="Original vs Sped-up")
    plt.title(f"Cross-Correlation of {label} Fakeprints")
    plt.xlabel("Lag")
    plt.ylabel("Correlation Coefficient")
    plt.legend()
    plt.grid(True)
    plt.show()

    lag, peak = refine_peak(corr)
    print(f"[{label}] Max correlation: {peak:.4f} (at lag {lag:.2f})")

# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    original_path = "data/signals/originals/signal2.mp3"
    spedup_path = "data/signals/spedup/signal2_s.mp3"

    waveform, sr = torchaudio.load(original_path)
    su_waveform, su_sr = torchaudio.load(spedup_path)
    print(f"Original : {waveform.shape}, sr={sr}")
    print(f"Sped-up  : {su_waveform.shape}, sr={su_sr}")

    # Compute spectrograms (both variants, pick one via MODE)
    if MODE == "cqt":
        spectrum = get_cqt(waveform, sr, N_BINS, BINS_PER_OCTAVE, HOP_LENGTH)
        su_spectrum = get_cqt(su_waveform, su_sr, N_BINS, BINS_PER_OCTAVE, HOP_LENGTH)
    else:
        spectrum = get_stft(waveform, N_FFT, HOP_LENGTH)
        su_spectrum = get_stft(su_waveform, N_FFT, HOP_LENGTH)

    # Extract fakeprints
    freqs, fp = get_fakeprint(spectrum, sr, f_range=F_RANGE, mode=MODE, log=True, raw=True)
    su_freqs, su_fp = get_fakeprint(su_spectrum, su_sr, f_range=F_RANGE, mode=MODE, log=True, raw=True)

    # Visualize
    plot_fakeprints(freqs, fp, su_freqs, su_fp, mode=MODE)
    plot_cross_correlation(fp, su_fp, label=MODE.upper())


if __name__ == "__main__":
    main()