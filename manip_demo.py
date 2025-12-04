import numpy as np
import matplotlib.pyplot as plt
import librosa
import scipy.signal

def plot_spectrum(y, sr, title, ax):
    D = np.abs(librosa.stft(y))
    # Average over time to see static peaks
    mag = np.mean(D, axis=1)
    freqs = librosa.fft_frequencies(sr=sr)
    ax.plot(freqs, mag)
    ax.set_title(title)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_xlim(0, 12000) # Zoom in on relevant area
    ax.grid(True, alpha=0.3)

# 1. Create a Synthetic "AI Signal"
# Base signal: Silence + Specific "Artifact" Peaks at 8kHz and 10kHz
sr = 22050
duration = 2.0
t = np.linspace(0, duration, int(sr * duration))

# Artifacts: Sine waves (which appear as Dirac peaks in spectrum)
# Simulating a model with artifacts at 4kHz and 8kHz
artifact_1 = 0.1 * np.sin(2 * np.pi * 4000 * t)
artifact_2 = 0.1 * np.sin(2 * np.pi * 8000 * t)
signal = artifact_1 + artifact_2 # Pure artifacts for clarity

fig, axes = plt.subplots(4, 1, figsize=(10, 15))
plt.subplots_adjust(hspace=0.5)

# Plot Original
plot_spectrum(signal, sr, "Original Signal (Artifacts at 4kHz & 8kHz)", axes[0])
axes[0].axvline(4000, color='r', linestyle='--', alpha=0.5)
axes[0].axvline(8000, color='r', linestyle='--', alpha=0.5)

# 2. Apply Pitch Shift (+2 Semitones)
# n_steps = 2. Frequency should increase by factor 2^(2/12) ~= 1.12
# 4000 -> 4490 Hz, 8000 -> 8980 Hz
y_pitch = librosa.effects.pitch_shift(signal, sr=sr, n_steps=2)
plot_spectrum(y_pitch, sr, "Pitch Shift (+2 semitones): Peaks Move Right", axes[1])
axes[1].axvline(4000, color='r', linestyle='--', label="Original Pos")
axes[1].legend()

# 3. Apply Time Stretch (Slow down by 1.5x)
# Ideally, frequency content should NOT change
y_stretch = librosa.effects.time_stretch(signal, rate=1/1.5)
plot_spectrum(y_stretch, sr, "Time Stretch (1.5x slower): Peaks Stay Fixed", axes[2])
axes[2].axvline(4000, color='r', linestyle='--', label="Original Pos")

# 4. Apply Resampling (Downsample to 8000Hz)
# Nyquist becomes 4000Hz. The 8kHz peak should disappear. 
# The 4kHz peak might be aliased or attenuated depending on the filter.
target_sr = 8000
y_resample = librosa.resample(signal, orig_sr=sr, target_sr=target_sr)
# Note: plotting with new sr
D_res = np.abs(librosa.stft(y_resample))
mag_res = np.mean(D_res, axis=1)
freqs_res = librosa.fft_frequencies(sr=target_sr)
axes[3].plot(freqs_res, mag_res)
axes[3].set_title(f"Resampling (to {target_sr}Hz): High Freq Peaks Erased")
axes[3].set_xlabel("Frequency (Hz)")
axes[3].set_xlim(0, 6000) # Different scale
axes[3].axvline(4000, color='r', linestyle='--', label="Original Pos")
axes[3].text(2000, np.max(mag_res)/2, "8kHz Peak is GONE due to Nyquist limit", color='red')

plt.savefig("manip_demo.png")
plt.show()