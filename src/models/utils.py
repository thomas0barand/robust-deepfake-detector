import torch
import torchaudio
import soxr

import numpy as np
import scipy.interpolate as interpolate

from nnAudio.features import CQT
from torchaudio.transforms import Spectrogram
from tqdm import tqdm


def get_spectrum(transform, waveform):
    with torch.no_grad():
        spec = transform(waveform)  # (1, n_bins, T')

    spec = 10 * torch.log10(torch.clamp(spec, min=1e-10, max=1e6))
    return spec


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


def get_freqs(n_fft, sr, transform="cqt", bins_per_octave=96, freq_range=[1000, 22000], f_min=32.7):
    assert transform in ["cqt", "stft"], "Transform must be 'cqt' or 'stft'"
    
    if transform == "cqt":
        nyquist = sr / 2
        n_octaves = np.log2(nyquist / f_min) - 0.1
        nbins = int(n_octaves * bins_per_octave)
        bin_indices = torch.arange(nbins)
        x_real = f_min * (2 ** (bin_indices / bins_per_octave))
    else:
        x_real = torch.linspace(0, sr / 2, steps=n_fft // 2 + 1)

    actual_fmax = min(freq_range[1], sr // 2)
    mask = (x_real >= freq_range[0]) & (x_real <= actual_fmax)
    
    freqs = x_real[mask]
    return freqs, mask


def get_feature_dim(n_fft, sr, transform="cqt", bins_per_octave=96, freq_range=[1000, 22000], f_min=32.7):
    if transform == "cqt":
        nyquist = sr / 2
        n_octaves = np.log2(nyquist / f_min) - 0.1
        nbins = int(n_octaves * bins_per_octave)
        bin_indices = torch.arange(nbins)
        x_real = f_min * (2 ** (bin_indices / bins_per_octave))
    else:
        x_real = torch.linspace(0, sr / 2, steps=n_fft // 2 + 1)

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


def preprocess_fakeprints(
    file_paths,
    n_fft=16384,
    sampling_rate=48000,
    bins_per_octave=96,
    freq_range=[1000, 22000],
    db_range=[-80, 5],
    f_min=32.7, # C1 note frequency
    device=torch.device("cpu"),
):
    assert device.type != "mps", "MPS device is not supported for this preprocessing pipeline. Please use CPU or CUDA."

    hop_length = n_fft // 2
    nyquist = sampling_rate / 2  # Maximum frequency that can be represented
    n_octaves = np.log2(nyquist / f_min) - 0.1  # Subtract a small margin to ensure we don't exceed Nyquist
    nbins = int(n_octaves * bins_per_octave)  # Total number of CQT bins to cover the desired frequency range

    cqt_transform = CQT(
        sr=sampling_rate,
        hop_length=hop_length,
        fmin=f_min,
        n_bins=nbins,
        bins_per_octave=bins_per_octave,
        output_format="Magnitude",
        verbose=False,
    ).to(device)

    stft_transform = Spectrogram(n_fft=n_fft, power=2, hop_length=hop_length).to(device)

    cqt_freqs, cqt_mask = get_freqs(
        n_fft=n_fft,
        sr=sampling_rate,
        transform="cqt",
        bins_per_octave=bins_per_octave,
        freq_range=freq_range,
        f_min=f_min
    )

    stft_freqs, stft_mask = get_freqs(
        n_fft=n_fft,
        sr=sampling_rate,
        transform="stft",
        bins_per_octave=bins_per_octave,
        freq_range=freq_range,
        f_min=f_min
    )

    cqt_fakeprints = []
    stft_fakeprints = []
    for path in tqdm(file_paths, leave=True):
        try:
            waveform, sr = torchaudio.load(path, channels_first=True)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            continue
        
        if sr != sampling_rate:
            waveform = soxr.resample(waveform.T, sr, sampling_rate).T
            waveform = torch.from_numpy(waveform).to(device)

        waveform = waveform.mean(dim=0, keepdim=True).to(device)  # Convert to mono

        cqt = get_spectrum(cqt_transform, waveform) # (1, n_bins, T')
        cqt = cqt.mean(dim=-1).squeeze(0)  # (n_bins,)

        stft = get_spectrum(stft_transform, waveform) # (1, n_bins, T')
        stft = stft.mean(dim=-1).squeeze(0)  # (n_bins,)
        
        cqt_spec_crop = cqt[cqt_mask]
        cqt_fp = get_fakeprints(cqt_spec_crop, cqt_freqs, db_range=db_range)
        cqt_fakeprints.append(cqt_fp)

        stft_spec_crop = stft[stft_mask]
        stft_fp = get_fakeprints(stft_spec_crop, stft_freqs, db_range=db_range)
        stft_fakeprints.append(stft_fp)

    cqt_fakeprints = torch.stack(cqt_fakeprints, dim=0)  # (N, freqs)
    stft_fakeprints = torch.stack(stft_fakeprints, dim=0)  # (N, freqs)

    return {
        "cqt": cqt_fakeprints.cpu().numpy(),
        "stft": stft_fakeprints.cpu().numpy()
    }