import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os

# Configuration
DURATION = 5  # seconds
SAMPLE_RATE = 44100
FIGURES_OUT = "outputs/figures/signals/"
AUDIO_OUT = "data/signals/"

os.makedirs(AUDIO_OUT, exist_ok=True)
t = np.linspace(0, DURATION, DURATION * SAMPLE_RATE, endpoint=False)

# Signal 1: Clean frequencies
frequencies = [6000, 10000, 15000]  # Hz
amplitudes = [0.5, 0.4, 0.3]
signal1 = np.zeros_like(t)
for freq, amp in zip(frequencies, amplitudes):
    signal1 += amp * np.sin(2 * np.pi * freq * t)
signal1 = signal1 / np.max(np.abs(signal1))  # normalize

# Signal 2: Narrow triangular peaks in FFT + visible noise floor
frequencies = [6000, 10000, 15000]  # Hz - center of peaks
amplitudes = [0.4, 0.35, 0.3]
peak_width = 15  # Hz - controls the width of the triangular peaks

signal2 = np.zeros_like(t)
for center_freq, amp in zip(frequencies, amplitudes):
    # Create a cluster of frequencies around the center to form a triangular peak
    num_components = 9
    for offset in np.linspace(-peak_width, peak_width, num_components):
        # Triangular weighting: highest at center, tapering to edges
        weight = 1 - abs(offset) / peak_width
        freq = center_freq + offset
        signal2 += amp * weight * np.sin(2 * np.pi * freq * t + np.random.uniform(0, 2*np.pi))

# Add broadband noise at ~20% of peak amplitude (in FFT terms)
noise_amplitude = 0.25  # tuned to get ~20% visible noise floor relative to peaks
noise = np.random.randn(len(t)) * noise_amplitude
signal2 += noise
signal2 = signal2 / np.max(np.abs(signal2))  # normalize

# Save as WAV files
signal1_int = np.int16(signal1 * 32767)
signal2_int = np.int16(signal2 * 32767)

wavfile.write(os.path.join(AUDIO_OUT, "signal1_harmonics.wav"), SAMPLE_RATE, signal1_int)
wavfile.write(os.path.join(AUDIO_OUT, "signal2_spikes_noise.wav"), SAMPLE_RATE, signal2_int)

# Compute FFTs
n = len(t)
freq_axis = np.fft.rfftfreq(n, 1/SAMPLE_RATE)

fft1 = np.abs(np.fft.rfft(signal1)) / n
fft2 = np.abs(np.fft.rfft(signal2)) / n

# Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Signal 1 waveform (first 50ms)
samples_to_show = int(0.05 * SAMPLE_RATE)
axes[0, 0].plot(t[:samples_to_show] * 1000, signal1[:samples_to_show], color='steelblue', linewidth=0.5)
axes[0, 0].set_title("Signal 1: Clean Frequencies")
axes[0, 0].set_xlabel("Time (ms)")
axes[0, 0].set_ylabel("Amplitude")
axes[0, 0].grid(True, alpha=0.3)

# Signal 1 FFT
axes[0, 1].plot(freq_axis, fft1, color='steelblue', linewidth=0.8)
axes[0, 1].set_title(f"Signal 1 FFT: peaks @ {frequencies} Hz")
axes[0, 1].set_xlabel("Frequency (Hz)")
axes[0, 1].set_ylabel("Magnitude")
axes[0, 1].set_xlim(0, 3000)
axes[0, 1].grid(True, alpha=0.3)

# Signal 2 waveform (first 50ms)
axes[1, 0].plot(t[:samples_to_show] * 1000, signal2[:samples_to_show], color='darkorange', linewidth=0.5)
axes[1, 0].set_title("Signal 2: Spikes + Noise")
axes[1, 0].set_xlabel("Time (ms)")
axes[1, 0].set_ylabel("Amplitude")
axes[1, 0].grid(True, alpha=0.3)

# Signal 2 FFT
axes[1, 1].plot(freq_axis, fft2, color='darkorange', linewidth=0.8)
axes[1, 1].set_title(f"Signal 2 FFT: peaks @ {frequencies} Hz + noise floor")
axes[1, 1].set_xlabel("Frequency (Hz)")
axes[1, 1].set_ylabel("Magnitude")
axes[1, 1].set_xlim(0, 3000)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_OUT, "fft for signals 1 and 2"), dpi=150)
plt.close()