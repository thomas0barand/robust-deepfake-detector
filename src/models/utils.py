import torch
import numpy as np
import scipy.interpolate as interpolate


def get_cqt(cqt_layer, waveform):
    with torch.no_grad():
        cqt = cqt_layer(waveform)  # (1, n_bins, T')

    cqt = 10 * torch.log10(torch.clamp(cqt, min=1e-10, max=1e6))
    return cqt


def lower_hull(x, area=10):
    """
    Compute the lower hull of a 1D tensor using a sliding window approach.
    """
    assert x.ndim == 1, "Input must be a 1D tensor"
    n = len(x)
    # Unfold into sliding windows: (n - area + 1, area)
    windows = x.unfold(0, area, 1)  # (num_windows, area)
    
    rel_idx = windows.argmin(dim=1)  # (num_windows,)
    abs_idx = rel_idx + torch.arange(len(rel_idx), device=x.device)  # (num_windows,)
    
    # Deduplicate while preserving first occurrence order
    unique_mask = torch.zeros(n, dtype=torch.bool, device=x.device)
    unique_mask[abs_idx] = True
    idx = torch.where(unique_mask)[0]
    hull = x[idx]
    
    # Ensure endpoints are included
    if idx[0] != 0:
        idx = torch.cat([torch.tensor([0], device=x.device), idx])
        hull = torch.cat([x[0:1], hull])
    if idx[-1] != n - 1:
        idx = torch.cat([idx, torch.tensor([n - 1], device=x.device)])
        hull = torch.cat([hull, x[n-1:n]])
    
    return idx, hull


def get_freqs(n_bins, sr, bins_per_octave=96, freq_range=[1000, 22000], f_min=32.7):
    bin_indices = torch.arange(n_bins)
    x_real = f_min * (2 ** (bin_indices / bins_per_octave))

    actual_fmax = min(freq_range[1], sr // 2)
    mask = (x_real >= freq_range[0]) & (x_real <= actual_fmax)
    
    freqs = x_real[mask]
    return freqs, mask


def get_feature_dim(sr, bins_per_octave=96, freq_range=[1000, 22000], f_min=32.7):
    nyquist = sr / 2
    n_octaves = np.log2(nyquist / f_min) - 0.1
    nbins = int(n_octaves * bins_per_octave)
    bin_indices = torch.arange(nbins)
    x_real = f_min * (2 ** (bin_indices / bins_per_octave))

    actual_fmax = min(freq_range[1], sr // 2)
    mask = (x_real >= freq_range[0]) & (x_real <= actual_fmax)
    feature_dim = len(x_real[mask])
    return feature_dim



def get_fakeprints(spectrum, freqs, db_range=[-80, 5]):
    lower_x, lower_c = lower_hull(spectrum, area=10)

    low_hull_curve = torch.from_numpy(
        interpolate.interp1d(freqs[lower_x].cpu(), lower_c.cpu(), kind='quadratic')(freqs.cpu().numpy())
    ).to(dtype=spectrum.dtype, device=spectrum.device)
    
    low_hull_curve = torch.clip(low_hull_curve, min=db_range[0])
    residue = torch.clip(spectrum - low_hull_curve, min=0, max=db_range[1])
    return residue