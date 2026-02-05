import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import soxr
import numpy as np
import torchaudio
import torch

from deezer.compute_fakeprints import open_audio_slice, fakeprint

# Parameters to change:
# - FREQ: frequency of the sinusoid (Hz)
# - DURATION: duration of the signal (seconds)
# - SR: original sampling rate (Hz)
# - NEW_SR: target sampling rate (Hz)
# - N_FFT: FFT window size (power of 2)

FREQ = 1000  # 1 kHz sinusoid
DURATION = 10.0  # 1 second
SR = 44100  # Original sampling rate
FMIN = 5000
FMAX = 16000
N_FFT = 4096

def get_fft(mp3_path, sr):
    audio, orig_sr = torchaudio.load(mp3_path, channels_first=True, normalize=True)
    
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    if orig_sr != sr:
        # Note: torchaudio has a resampler, but sticking to your soxr usage:
        audio_np = audio.numpy()
        audio_np = soxr.resample(audio_np.T, orig_sr, sr).T
        audio = torch.from_numpy(audio_np)
    
    audio = audio.flatten()
    
    n = len(audio)
    fft_complex = torch.fft.rfft(audio)
    fft_mag = torch.abs(fft_complex).numpy()
    freqs = torch.fft.rfftfreq(n, 1/sr).numpy()
    
    return freqs, fft_mag

def spectrogram(f_name, max_duration=180, SR=44100, n_fft=4096):
    p, sr = open_audio_slice(f_name)
    
    # Resample if needed
    if sr != SR:
        p = soxr.resample(p, sr, SR)
    
    # Cut long audios
    p = p[:SR * max_duration]

    # Initialize transformer HERE to ensure n_fft matches the current request
    transformer = torchaudio.transforms.Spectrogram(n_fft=n_fft, power=2)
    
    # Compute STFT
    stft = transformer(torch.Tensor(p.T)).numpy()
    
    # Convert to dB
    stft_db = 10 * np.log10(np.clip(stft, 1e-10, 1e6))
    
    return stft_db

def visualize_fft(freqs, fft_mag, sr):
    # Convert to Decibels (dB) for better visualization of peaks
    # adding 1e-10 prevents log(0) errors
    fmin = 0
    fmax = int(sr/2)
    fft_db = 20 * np.log10(fft_mag + 1e-10)

    plt.figure(figsize=(10, 6))
    plt.plot(freqs, fft_db)
    
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.title(f'FFT Spectrum (SR={sr} Hz)')
    plt.grid(True, alpha=0.3)
    
    plt.xlim(fmin - 1000, fmax + 1000) 
    
    plt.tight_layout()
    plt.show()

def visualize_spectrogram(spectro, sr, n_fft, title="Spectrogram", viz_mean=False):
    """
    Visualizes a spectrogram with correct time (x) and frequency (y) axes.
    """
    # 1. Handle Channels: If shape is (Channels, Freq, Time), take the first channel
    if spectro.ndim == 3:
        spectro_shaped = spectro[0]
        
    # 2. Calculate Axis Limits
    # Torchaudio default hop_length is usually n_fft // 2
    hop_length = n_fft // 2 
    num_frames = spectro_shaped.shape[1]
    
    duration = (num_frames * hop_length) / sr
    nyquist = sr / 2

    plt.figure(figsize=(12, 6))
    plt.imshow(spectro_shaped, origin='lower', aspect='auto', 
               extent=[0, duration, 0, nyquist], cmap='inferno')

    plt.colorbar(format='%+2.0f dB')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency (Hz)')
    plt.title(title)
    plt.tight_layout()
    plt.show()

    if viz_mean:
        # 3. Mean STFT
        # Since we passed N_FFT above, this axis calculation is now guaranteed to match
        stft_freqs = torch.fft.rfftfreq(N_FFT, 1/SR).numpy()
        mean_stft = np.mean(spectro, axis=2)
        
        plt.figure(figsize=(10, 5))
        # Note: Access [0, :] because spectro shape is (Channels, Freq, Time)
        plt.plot(stft_freqs, mean_stft[0, :], label="Mean STFT")
        
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (dB)')
        plt.title(f'Averaged Spectrogram (N_FFT={N_FFT})')
        plt.grid(True, alpha=0.3)
        plt.show()

def visualize_fp(x_axis, fp, sr):
    """
    Modified to take x_axis explicitly so we see Hz, not indices.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(x_axis, fp)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Fakeprint Amplitude (Normalized)')
    plt.title(f'Fakeprint ({x_axis.min():.0f}-{x_axis.max():.0f} Hz)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    mp3_path = "data/signals/signal2_spikes_noise.wav"
    
    freqs, fft_mag = get_fft(mp3_path, SR)
    
    # visualize_fft(freqs, fft_mag, SR)

    spectro = spectrogram(mp3_path, SR=SR, n_fft=N_FFT)
    # visualize_spectrogram(spectro, SR, N_FFT, viz_mean = True)

    fp = fakeprint(spectro, f_range=[FMIN, FMAX], SR = SR)
    
    
    # 3. Reconstruct the Frequency Axis for Visualization
    # We must replicate the logic inside fakeprint to know which frequencies correspond to 'fp'
    num_bins = spectro.shape[1] # Freq dim
    full_freqs = np.linspace(0, SR / 2, num=num_bins)
    
    # Apply the same mask used in 'curve_profile'
    mask = (FMIN < full_freqs) & (full_freqs < FMAX)
    fp_freqs = full_freqs[mask]

    # 4. Visualize
    # Ensure dimensions match before plotting
    if len(fp_freqs) != len(fp):
        print(f"Warning: Size mismatch. Axis: {len(fp_freqs)}, Data: {len(fp)}")
        # Fallback: just plot generic axis if mismatch occurs due to floating point
        visualize_fp(np.linspace(FMIN, FMAX, len(fp)), fp, SR)
    else:
        visualize_fp(fp_freqs, fp, SR)



if __name__ == "__main__":
    main()