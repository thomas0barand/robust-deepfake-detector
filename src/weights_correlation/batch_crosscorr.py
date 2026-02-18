"""
Batch fakeprint comparison.

Compares a reference audio file against a list of other audio files:
  1. Overlays all fakeprints (reference in bold, others in pale colors).
  2. Computes cross-correlation between reference and each file,
     then plots the average correlation curve.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import torchaudio
from scipy.signal import correlate
from pathlib import Path

from utils import get_stft, get_cqt, get_fakeprint, refine_peak

# ── Configuration ────────────────────────────────────────────────────────────

N_FFT = 1 << 14            # 16384
SAMPLE_RATE = 16000
HOP_LENGTH = N_FFT // 2
F_MIN = 32.7                # C1 note frequency
BINS_PER_OCTAVE = 96
F_RANGE = (500, 5000)       # Hz — frequency band of interest
MODE = "cqt"

_nyquist = SAMPLE_RATE / 2
_n_octaves = np.log2(_nyquist / F_MIN) - 0.1
N_BINS = int(_n_octaves * BINS_PER_OCTAVE)

# ── File lists ───────────────────────────────────────────────────────────────
# Edit these paths to match your setup.

REFERENCE = "data/signals/spedup/fake_suno_0_su.mp3"

COMPARE_FILES = [
    "data/ai/fake_00001_suno_1.mp3",
    "data/ai/fake_00006_suno_0.mp3",
    "data/ai/fake_00006_suno_1.mp3",
    "data/ai/fake_00007_suno_0.mp3",
    "data/ai/fake_00007_suno_1.mp3",
    "data/ai/fake_00011_suno_0.mp3",
    "data/ai/fake_00011_suno_1.mp3",
    "data/ai/fake_00014_suno_0.mp3",
    "data/ai/fake_00014_suno_1.mp3",
    "data/ai/fake_00018_suno_0.mp3",
    "data/ai/fake_00018_suno_1.mp3",
    "data/ai/fake_00025_suno_0.mp3",
    "data/ai/fake_00025_suno_1.mp3",
    "data/ai/fake_00031_suno_0.mp3",
    "data/ai/fake_00031_suno_1.mp3",
    "data/ai/fake_00034_suno_0.mp3",
    "data/ai/fake_00034_suno_1.mp3",
]

# ── Helpers ──────────────────────────────────────────────────────────────────

def compute_spectrum(filepath: str):
    """Load audio and compute spectrogram."""
    waveform, sr = torchaudio.load(filepath)
    print(f"  Loaded {Path(filepath).name}: {waveform.shape}, sr={sr}")
    if MODE == "cqt":
        spectrum = get_cqt(waveform, sr, N_BINS, BINS_PER_OCTAVE, HOP_LENGTH)
    else:
        spectrum = get_stft(waveform, N_FFT, HOP_LENGTH)
    return spectrum, sr


def normalized_cross_correlation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson-style normalized cross-correlation (same as crosscorr.py)."""
    a = a.flatten()
    b = b.flatten()
    a_norm = (a - np.mean(a)) / (np.std(a) * len(a))
    b_norm = (b - np.mean(b)) / np.std(b)
    return correlate(a_norm, b_norm, mode="same")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    # --- Reference ---
    print("Loading reference...")
    ref_spectrum, ref_sr = compute_spectrum(REFERENCE)
    ref_freqs, ref_fp = get_fakeprint(
        ref_spectrum, ref_sr, f_range=F_RANGE, mode=MODE, log=True, raw=False
    )

    # --- Comparison files ---
    comp_data = []  # list of (name, freqs, fp)
    for path in COMPARE_FILES:
        print(f"Loading comparison file...")
        spectrum, sr = compute_spectrum(path)
        freqs, fp = get_fakeprint(
            spectrum, sr, f_range=F_RANGE, mode=MODE, log=True, raw=False
        )
        comp_data.append((Path(path).stem, freqs, fp))

    # ── Plot 1: Fakeprint overlay ────────────────────────────────────────────
    sns.set_theme()
    fig, ax = plt.subplots(figsize=(14, 6))

    # Pale colors for comparison files
    palette = cm.get_cmap("tab10", max(len(comp_data), 1))
    for i, (name, freqs, fp) in enumerate(comp_data):
        color = palette(i)
        ax.plot(freqs, fp, color=color, alpha=0.30, linewidth=0.8, label=name)

    # Reference on top, bold
    ax.plot(ref_freqs, ref_fp, color="black", linewidth=1.8,
            label=f"REF: {Path(REFERENCE).stem}")

    ax.set_xscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Mean Power (dB)")
    ax.set_title("CQT Fakeprint Overlay")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    plt.tight_layout()
    plt.savefig("fakeprint_overlay.png", dpi=150)
    plt.show()
    print("Saved fakeprint_overlay.png")

    # ── Plot 2: Cross-correlations + average ─────────────────────────────────
    all_corrs = []
    print("\nCross-correlation results:")
    print("-" * 50)

    for name, _, fp in comp_data:
        # Ensure same length (trim to shorter)
        min_len = min(len(ref_fp), len(fp))
        corr = normalized_cross_correlation(ref_fp[:min_len], fp[:min_len])
        all_corrs.append(corr)

        lag, peak = refine_peak(corr)
        print(f"  {name:40s}  peak={peak:.4f}  lag={lag:.2f}")

    if not all_corrs:
        print("No comparison files — nothing to correlate.")
        return

    # Pad/truncate all correlations to the same length for averaging
    max_len = max(len(c) for c in all_corrs)
    padded = np.zeros((len(all_corrs), max_len))
    for i, c in enumerate(all_corrs):
        padded[i, :len(c)] = c

    avg_corr = np.mean(padded, axis=0)
    avg_lag, avg_peak = refine_peak(avg_corr)
    print("-" * 50)
    print(f"  {'AVERAGE':40s}  peak={avg_peak:.4f}  lag={avg_lag:.2f}")

    fig, ax = plt.subplots(figsize=(14, 6))

    # Individual correlations — fully visible
    for i, (name, _, _) in enumerate(comp_data):
        c = padded[i]
        lags_i = np.arange(-len(c) // 2, len(c) // 2)
        ax.plot(lags_i, c, color=palette(i), alpha=0.85, linewidth=1.2, label=name)

    ax.set_title(f"Cross-Correlation with {Path(REFERENCE).stem}")
    ax.set_xlabel("Lag (bins)")
    ax.set_ylabel("Correlation Coefficient")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig("crosscorr_average.png", dpi=150)
    plt.show()
    print("Saved crosscorr_average.png")


if __name__ == "__main__":
    main()
