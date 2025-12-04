import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from scipy.fft import fft, fftfreq

def compute_spectrum(y, sr):
    """Compute magnitude spectrum using FFT"""
    n = len(y)
    Y = fft(y)
    freqs = fftfreq(n, 1/sr)[:n//2]
    mag = 2.0/n * np.abs(Y[:n//2])
    return freqs, mag

def plot_spectrum(freqs, mag, title, ax, xlim=None):
    """Plot frequency spectrum"""
    ax.plot(freqs, mag, linewidth=1.5)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    if xlim:
        ax.set_xlim(xlim)
    ax.grid(True, alpha=0.3)

# Original signal at 44.1 kHz
sr_orig = 44100
duration = 2.0
t_orig = np.linspace(0, duration, int(sr_orig * duration), endpoint=False)

# Realistic signal with 3 frequency components
f1, f2, f3 = 3000, 6000, 12000
signal = (0.5 * np.sin(2 * np.pi * f1 * t_orig) + 
          0.3 * np.sin(2 * np.pi * f2 * t_orig) + 
          0.2 * np.sin(2 * np.pi * f3 * t_orig))

# Target sample rate
sr_target = 16000
nyquist_target = sr_target / 2  # 8000 Hz

# Design anti-aliasing low-pass filter
# Cutoff at ~90% of new Nyquist to avoid aliasing
cutoff = 0.9 * nyquist_target
sos = scipy.signal.butter(8, cutoff, btype='low', fs=sr_orig, output='sos')

# Apply low-pass filter
signal_filtered = scipy.signal.sosfilt(sos, signal)

# Resample to 16 kHz
num_samples = int(len(signal) * sr_target / sr_orig)
signal_resampled = scipy.signal.resample(signal_filtered, num_samples)
t_resampled = np.linspace(0, duration, num_samples, endpoint=False)

# Compute spectra
freqs_orig, mag_orig = compute_spectrum(signal, sr_orig)
freqs_filt, mag_filt = compute_spectrum(signal_filtered, sr_orig)
freqs_res, mag_res = compute_spectrum(signal_resampled, sr_target)

# Create figure with 4 subplots
fig, axes = plt.subplots(4, 1, figsize=(12, 14))
plt.subplots_adjust(hspace=0.4)

# 1. Original signal spectrum
plot_spectrum(freqs_orig, mag_orig, 
              f"Original Signal @ {sr_orig} Hz (Peaks at {f1}, {f2}, {f3} Hz)",
              axes[0], xlim=(0, 15000))
for f, label in [(f1, f'{f1} Hz'), (f2, f'{f2} Hz'), (f3, f'{f3} Hz')]:
    axes[0].axvline(f, color='r', linestyle='--', alpha=0.7, linewidth=1.5)
axes[0].axvline(nyquist_target, color='green', linestyle=':', 
                linewidth=2, label=f'New Nyquist ({nyquist_target} Hz)')
axes[0].legend(loc='upper right')

# 2. Filter frequency response
w, h = scipy.signal.sosfreqz(sos, worN=8192, fs=sr_orig)
axes[1].plot(w, 20 * np.log10(np.abs(h)), 'b', linewidth=2)
axes[1].set_title(f"Anti-aliasing LPF Response (Cutoff ≈ {cutoff:.0f} Hz)", 
                  fontsize=12, fontweight='bold')
axes[1].set_xlabel("Frequency (Hz)")
axes[1].set_ylabel("Gain (dB)")
axes[1].axvline(cutoff, color='r', linestyle='--', label=f'Cutoff ({cutoff:.0f} Hz)')
axes[1].axvline(nyquist_target, color='green', linestyle=':', 
                linewidth=2, label=f'New Nyquist ({nyquist_target} Hz)')
axes[1].axhline(-3, color='gray', linestyle=':', alpha=0.5, label='-3 dB')
axes[1].set_xlim(0, 15000)
axes[1].set_ylim(-80, 5)
axes[1].grid(True, alpha=0.3)
axes[1].legend(loc='upper right')

# 3. Filtered signal spectrum
plot_spectrum(freqs_filt, mag_filt,
              f"After LPF @ {sr_orig} Hz (12 kHz peak attenuated)",
              axes[2], xlim=(0, 15000))
axes[2].axvline(f1, color='r', linestyle='--', alpha=0.7, linewidth=1.5)
axes[2].axvline(f2, color='r', linestyle='--', alpha=0.7, linewidth=1.5)
axes[2].axvline(nyquist_target, color='green', linestyle=':', 
                linewidth=2, label=f'New Nyquist ({nyquist_target} Hz)')
axes[2].legend(loc='upper right')

# 4. Resampled signal spectrum
plot_spectrum(freqs_res, mag_res,
              f"After Resampling to {sr_target} Hz (Only peaks below Nyquist remain)",
              axes[3], xlim=(0, 10000))
axes[3].axvline(f1, color='r', linestyle='--', alpha=0.7, linewidth=1.5, 
                label=f'{f1} Hz (preserved)')
axes[3].axvline(f2, color='r', linestyle='--', alpha=0.7, linewidth=1.5,
                label=f'{f2} Hz (preserved)')
axes[3].axvline(nyquist_target, color='green', linestyle=':', 
                linewidth=2, label=f'Nyquist ({nyquist_target} Hz)')
axes[3].text(9000, np.max(mag_res)*0.8, 
             f'{f3} Hz peak\nremoved by LPF', 
             color='red', fontsize=10, ha='center',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
axes[3].legend(loc='upper right')

plt.savefig("resampling_demo.png", dpi=150, bbox_inches='tight')
print(f"Figure saved: resampling_demo.png")
plt.show()

