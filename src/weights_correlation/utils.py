import torch
import torchaudio
import librosa
import numpy as np
from scipy import interpolate



def get_stft(waveform, n_fft, hop_length):
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    stft_transformer = torchaudio.transforms.Spectrogram(n_fft=n_fft, power=2, hop_length=hop_length)
    stft = stft_transformer(waveform).numpy()
    stft = 10 * np.log10(np.clip(stft, 1e-10, 1e6))
    return stft


def get_cqt(waveform, sr, n_bins, bins_per_octave, hop_length):
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


def lower_hull(x, area=10):
    idx, hull = [], []
    for i in range(len(x)-area+1):
        patch = x[i:i+area]
        rel_idx = np.argmin(patch)
        abs_idx = rel_idx + i
        if abs_idx not in idx:
            idx.append(abs_idx)
            hull.append(patch[rel_idx])
    
    if len(idx) < 2:
        return np.array([0, len(x)-1]), np.array([x[0], x[-1]])

    if idx[0] != 0:
        idx.insert(0, 0); hull.insert(0, x[0])
    if idx[-1] != len(x)-1:
        idx.append(len(x)-1); hull.append(x[-1])
    return np.array(idx), np.array(hull)


def get_fakeprint(
    spectrum,
    sr,
    f_min=27.5,
    bins_per_octave=96,
    f_range=[5000, 16000],
    mode="stft",
    log=False,
):

    if isinstance(spectrum, torch.Tensor):
        spectrum = spectrum.detach().cpu().numpy()

    fp = np.mean(spectrum, axis=(0, 2))

    if mode == "cqt":
        bin_indices = np.arange(len(fp))
        x_real = f_min * (2 ** (bin_indices / bins_per_octave))
    else:
        x_real = np.linspace(0, sr / 2, num=len(fp))

    actual_fmax = min(f_range[1], sr // 2)
    mask = (x_real >= f_range[0]) & (x_real <= actual_fmax)
    
    freqs = x_real[mask]
    fp_crop = fp[mask]
    
    lower_x, lower_c = lower_hull(fp_crop, area=10)
    low_hull_curve = interpolate.interp1d(freqs[lower_x], lower_c, kind="quadratic")(freqs)
    
    residue = np.clip(fp_crop - low_hull_curve, 0, None)
    residue = residue / (1e-6 + np.max(residue))

    if mode == "stft" and log:
        freqs = np.log10(np.clip(freqs, 1e-6, None))
    
    return freqs, residue


