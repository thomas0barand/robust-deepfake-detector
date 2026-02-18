"""
Utility functions for computing spectrograms and extracting fakeprints.

A fakeprint is the normalized spectral residue above the lower hull of
the mean power spectrum — designed to capture characteristic spectral
artifacts in AI-generated audio.
"""

import numpy as np
import torch
import torchaudio
import librosa
from scipy import interpolate

F_MIN = 32.7  # C1 note frequency (Hz)


def get_stft(waveform: torch.Tensor, n_fft: int, hop_length: int) -> np.ndarray:
    """Compute a power spectrogram (in dB) via STFT.

    Returns:
        np.ndarray of shape (channels, freq_bins, time_frames).
    """
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    transform = torchaudio.transforms.Spectrogram(
        n_fft=n_fft, power=2, hop_length=hop_length
    )
    power = transform(waveform).numpy()
    return 10 * np.log10(np.clip(power, 1e-10, 1e6))


def get_cqt(
    waveform: torch.Tensor,
    sr: int,
    n_bins: int,
    bins_per_octave: int,
    hop_length: int,
) -> np.ndarray:
    """Compute a Constant-Q Transform spectrogram (in dB).

    Returns:
        np.ndarray of shape (channels, freq_bins, time_frames).
    """
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    cqt = librosa.cqt(
        y=waveform.numpy(),
        sr=sr,
        hop_length=hop_length,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
    )
    cqt = np.abs(cqt)
    cqt = 10 * np.log10(np.clip(cqt, 1e-10, 1e6))

    if cqt.ndim == 2:
        cqt = cqt[np.newaxis, :, :]
    return cqt


def lower_hull(signal: np.ndarray, window: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Trace the lower envelope of *signal* using a sliding-window minimum.

    Args:
        signal:  1-D array.
        window:  Size of the sliding window.

    Returns:
        (indices, values) — sparse samples along the lower envelope,
        guaranteed to include the first and last points.
    """
    indices, values = [], []

    for i in range(len(signal) - window + 1):
        patch = signal[i : i + window]
        local_min = np.argmin(patch) + i
        if local_min not in indices:
            indices.append(local_min)
            values.append(signal[local_min])

    # Fallback if too few points were found
    if len(indices) < 2:
        return (
            np.array([0, len(signal) - 1]),
            np.array([signal[0], signal[-1]]),
        )

    # Ensure endpoints are included
    if indices[0] != 0:
        indices.insert(0, 0)
        values.insert(0, signal[0])
    if indices[-1] != len(signal) - 1:
        indices.append(len(signal) - 1)
        values.append(signal[-1])

    return np.array(indices), np.array(values)


def get_fakeprint(
    spectrum: np.ndarray | torch.Tensor,
    sr: int,
    bins_per_octave: int = 96,
    f_range: tuple[float, float] = (100, 44000),
    mode: str = "stft",
    log: bool = False,
    raw: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a fakeprint from a spectrogram.

    Steps:
        1. Average the spectrogram over channels and time → mean spectrum.
        2. Map bins to real frequencies (linear for STFT, log for CQT).
        3. Crop to *f_range*.
        4. (unless raw=True) Subtract an interpolated lower-hull baseline.
        5. (unless raw=True) Normalize the residue to [0, 1].

    Args:
        spectrum:        (channels, freq_bins, time_frames) spectrogram.
        sr:              Sample rate of the original audio.
        bins_per_octave: Only used when mode="cqt".
        f_range:         (f_min, f_max) in Hz.
        mode:            "stft" or "cqt".
        log:             If True and mode="stft", return log-scaled freqs.
        raw:             If True, skip hull subtraction and normalization.
                         Useful for testing shift-invariance directly.

    Returns:
        (freqs, fakeprint) — both 1-D arrays of the same length.
    """
    if isinstance(spectrum, torch.Tensor):
        spectrum = spectrum.detach().cpu().numpy()

    mean_spectrum = np.mean(spectrum, axis=(0, 2))

    # Map frequency-bin indices to Hz
    if mode == "cqt":
        bin_indices = np.arange(len(mean_spectrum))
        freqs_hz = F_MIN * (2 ** (bin_indices / bins_per_octave))
    else:
        freqs_hz = np.linspace(0, sr / 2, num=len(mean_spectrum))

    # Crop to requested range
    f_max = min(f_range[1], sr // 2)
    mask = (freqs_hz >= f_range[0]) & (freqs_hz <= f_max)
    freqs = freqs_hz[mask]
    cropped = mean_spectrum[mask]

    if raw:
        # Return the cropped mean spectrum as-is
        if mode == "stft" and log:
            freqs = np.log10(np.clip(freqs, 1e-6, None))
        return freqs, cropped

    # Subtract lower-hull baseline
    hull_idx, hull_vals = lower_hull(cropped, window=10)
    baseline = interpolate.interp1d(freqs[hull_idx], hull_vals, kind="quadratic")(freqs)
    residue = np.clip(cropped - baseline, 0, None)
    residue /= np.max(residue) + 1e-6  # normalize to [0, 1]

    if mode == "stft" and log:
        freqs = np.log10(np.clip(freqs, 1e-6, None))

    return freqs, residue

def refine_peak(corr: np.ndarray) -> tuple[float, float]:
    """Parabolic interpolation around the max of a correlation array.
    
    Fits a parabola through the peak and its two neighbors to estimate
    the true (sub-sample) peak location and value.

    Returns:
        (refined_lag, refined_max) — lag relative to center, and peak value.
    """
    center = len(corr) // 2
    k = np.argmax(corr)

    # Can't interpolate if peak is at the edge
    if k == 0 or k == len(corr) - 1:
        return float(k - center), float(corr[k])

    y0, y1, y2 = corr[k - 1], corr[k], corr[k + 1]

    # Vertex of parabola through (-1,y0), (0,y1), (1,y2)
    delta = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
    refined_lag = (k + delta) - center
    refined_max = y1 - 0.25 * (y0 - y2) * delta

    return refined_lag, refined_max