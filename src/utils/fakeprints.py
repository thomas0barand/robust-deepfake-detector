import torch
import soxr
import librosa
import numpy as np
from warnings import deprecated

def load_audio(file_path, max_duration=None):
    try:
        waveform, sr = librosa.load(file_path, sr=None, mono=True, duration=max_duration)
        waveform = torch.from_numpy(waveform).unsqueeze(0)  # (1, num_samples)
        return waveform, sr
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None
    
@deprecated("This method has been deprecated on 08-04. Please use: from src.data.preprocessing import resample")
def speed_up(waveform, sr, speed_factor):
    if speed_factor == 1.0:
        return waveform
    
    device = waveform.device
    new_sr = int(sr / speed_factor)
    resampled_waveform = soxr.resample(waveform.cpu().numpy().T, sr, new_sr, quality="VHQ").T

    return torch.from_numpy(resampled_waveform).to(device)


def get_spectrum(transform, waveform):
    with torch.no_grad():
        spec = transform(waveform) # (B, 1, L) -> (B, n_bins, T')

    spec = 10 * torch.log10(torch.clamp(spec, min=1e-10, max=1e6))
    return spec


def get_freqs(n_fft, sr, log=True, bins_per_octave=192, fmin=32.7):
    if log:
        nyquist = sr / 2
        n_octaves = np.log2(nyquist / fmin)
        nbins = int(n_octaves * bins_per_octave)
        bin_indices = torch.arange(nbins + 1)
        freqs = fmin * (2 ** (bin_indices / bins_per_octave))
    else:
        freqs = torch.linspace(0, sr / 2, steps=n_fft // 2 + 1)
    return freqs


def get_freqs_mask(freqs, sr, freq_range):
    actual_fmax = min(freq_range[1], sr // 2)
    mask = (freqs >= freq_range[0]) & (freqs <= actual_fmax)
    return mask


def get_low_hull_curve(x, area=20):
    """
    For each sample, compute the lower hull envelope evaluated at every bin.
    x: (B, n_bins)
    Returns: (B, n_bins) lower hull curve
    """
    B, n = x.shape

    # Find anchor indices per sample
    windows = x.unfold(1, area, 1)   # (B, num_windows, area)
    rel_idx = windows.argmin(dim=2)  # (B, num_windows)
    offsets = torch.arange(rel_idx.shape[1], device=x.device)
    abs_idx = rel_idx + offsets      # (B, num_windows)

    # Mark selected bins per sample, always include endpoints
    anchor_mask = torch.zeros(B, n, dtype=torch.bool, device=x.device)
    anchor_mask.scatter_(1, abs_idx, True)
    anchor_mask[:, 0] = True
    anchor_mask[:, n - 1] = True  # (B, n_bins) bool

    # Interpolate hull at every bin position
    # For each bin j, find the nearest anchor to the left and right *per sample*
    # We encode anchor positions as their index, non-anchors as 0/inf, then
    # do a forward/backward cummax to propagate neighbour indices.
    bin_pos = torch.arange(n, device=x.device).unsqueeze(0).expand(B, -1)  # (B, n)

    # Left anchor index: propagate forward
    left_idx = torch.where(anchor_mask, bin_pos, torch.zeros_like(bin_pos))
    left_idx, _ = left_idx.cummax(dim=1)

    # Right anchor index: propagate backward
    right_idx = torch.where(anchor_mask, bin_pos, torch.full_like(bin_pos, n - 1))
    right_idx = torch.flip(
        torch.flip(right_idx, dims=[1]).cummin(dim=1)[0], dims=[1]
    )                                       

    # Gather y-values at left/right anchors
    y_left  = x.gather(1, left_idx)   # (B, n)
    y_right = x.gather(1, right_idx)  # (B, n)

    # Linear interpolation weight
    span = (right_idx - left_idx).float().clamp_min(1)
    weight = (bin_pos - left_idx).float() / span       # (B, n)

    hull_curve = y_left + weight * (y_right - y_left)  # (B, n)
    return hull_curve


def get_fakeprints(spectrums, area=20):
    if spectrums.dim() == 1:
        spectrums = spectrums.unsqueeze(0)
    low_hull_curves = get_low_hull_curve(spectrums, area=area)
    residues = torch.clamp(spectrums - low_hull_curves, min=0)
    return residues