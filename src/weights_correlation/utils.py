import torch
import torchaudio
import librosa
import numpy as np
import matplotlib.pyplot as plt

from scipy import interpolate

F_MIN = 32.7

def get_stft(waveform, n_fft, hop_length):
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    stft_transformer = torchaudio.transforms.Spectrogram(n_fft=n_fft, power=2, hop_length=hop_length)
    stft = stft_transformer(torch.Tensor(waveform.T)).numpy()
    stft = 10 * np.log10(np.clip(stft, 1e-10, 1e6))
    return stft


def get_cqt(waveform, sr, n_bins, bins_per_octave, hop_length):
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    cqt = librosa.cqt(
        y=torch.Tensor(waveform.T).numpy(),
        sr=sr,
        hop_length=hop_length,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        fmin=F_MIN,
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
    bins_per_octave=96,
    f_range=[5000, 16000],
    dB_range=[-80, 5],
    mode="stft",
):

    if isinstance(spectrum, torch.Tensor):
        spectrum = spectrum.detach().cpu().numpy()

    fp = np.mean(spectrum, axis=(0, 2))

    if mode == "cqt":
        bin_indices = np.arange(len(fp))
        x_real = F_MIN * (2 ** (bin_indices / bins_per_octave))
    else:
        x_real = np.linspace(0, sr / 2, num=len(fp))

    actual_fmax = min(f_range[1], sr // 2)
    mask = (x_real >= f_range[0]) & (x_real <= actual_fmax)
    
    freqs = x_real[mask]
    fp_crop = fp[mask]
    
    lower_x, lower_c = lower_hull(fp_crop, area=10)
    low_hull_curve = interpolate.interp1d(freqs[lower_x], lower_c, kind="quadratic")(freqs)
    low_hull_curve = np.clip(low_hull_curve, dB_range[0], None)
    
    residue = np.clip(fp_crop - low_hull_curve, 0, dB_range[1])
    residue = residue / (1e-6 + np.max(residue))
    
    return freqs, residue


def get_correlation(fp1, fp2):
    fp1 = (fp1 - np.mean(fp1)) / (np.std(fp1) + 1e-10)
    fp2 = (fp2 - np.mean(fp2)) / (np.std(fp2) + 1e-10)
    print(fp1.shape, fp2.shape)
    corr = np.convolve(fp1, fp2[::-1], mode='full') / len(fp1)
    print(corr.shape)
    lags = np.arange(-len(fp2) + 1, len(fp1))
    return lags, corr


def get_refined_peak(lags, corr):
    # 1. Find the discrete maximum
    idx = np.argmax(corr)
    
    # Check if peak is at the very edge (cannot interpolate)
    if idx <= 0 or idx >= len(corr) - 1:
        return lags[idx], corr[idx]
    
    # 2. Get the three points
    y1 = corr[idx - 1]
    y2 = corr[idx]
    y3 = corr[idx + 1]
    
    # 3. Calculate the fractional offset
    # Formula: 0.5 * (left - right) / (left - 2*mid + right)
    denom = (y1 - 2 * y2 + y3)
    if abs(denom) < 1e-10: # Avoid division by zero
        return lags[idx], corr[idx]
        
    delta = 0.5 * (y1 - y3) / denom
    
    # 4. Calculate refined lag and refined magnitude
    refined_lag = lags[idx] + delta
    refined_val = y2 - 0.25 * (y1 - y3) * delta
    
    return refined_lag, refined_val


def plot_fp(freqs1, fp1, freqs2, fp2, log_scale=True):
    plt.figure(figsize=(12, 5))
    plt.plot(freqs1, fp1, label="Original")
    plt.plot(freqs2, fp2, label="Pitch-Shifted", alpha=0.8)
    if log_scale:
        plt.xscale('log')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Normalized Residue')
    plt.title('Fakeprint Comparison')
    plt.legend()
    plt.grid()
    plt.show()


def plot_correlation(fp1, fp2):

    lags, corr = get_correlation(fp1, fp2)
    idx = np.argmax(corr)
    mx_lag = lags[idx]
    mx_val = corr[idx]

    refined_lag, refined_val = get_refined_peak(lags, corr)
    
    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(lags, corr, label='Cross-Correlation', color='royalblue', alpha=0.7)

    # Add horizontal bar for refined value
    plt.axhline(y=refined_val, color='red', linestyle='--', linewidth=1, 
                alpha=0.6, label=f'Refined Max: {refined_val:.4f}')
    
    # Add vertical bar for refined lag
    plt.axvline(x=refined_lag, color='red', linestyle=':', linewidth=1, alpha=0.4)
    
    # Plot discrete peak
    plt.scatter(mx_lag, mx_val, color='black', s=30, label=f'Discrete Peak (Lag: {mx_lag})', zorder=3)
    
    # Plot interpolated refined peak
    plt.scatter(refined_lag, refined_val, color='red', s=60, edgecolors='white', 
                label=f'Refined Peak (Lag: {refined_lag:.3f})', zorder=4)
    
    plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
    plt.title("Fingerprint Correlation with Sub-sample Peak Estimation")
    plt.xlabel("Lag (Samples)")
    plt.ylabel("Correlation Magnitude")
    plt.legend()
    plt.grid(True, which='both', linestyle=':', alpha=0.6)
    plt.show()

    return refined_lag, refined_val