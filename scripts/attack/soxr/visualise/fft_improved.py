"""
python scripts/attack/soxr/visualise/fft_improved.py -og data/ai/fake_00001_suno_0.mp3 -nw data/signals/resampled/fake_00001_suno_0_rs.mp3           
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal as scipy_signal
from scipy.interpolate import interp1d

from pydub import AudioSegment

def read_audio(file_path):
    audio = AudioSegment.from_file(file_path)
    sr = audio.frame_rate
    samples = np.array(audio.get_array_of_samples())
    if audio.channels == 2:
        samples = samples.reshape((-1, 2))
    return sr, samples


def smooth_spectrum(freqs, mag, window_size=50):
    """Apply moving average smoothing to spectrum"""
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(mag, kernel, mode='same')
    return smoothed


def plot_fft_comparison(original_path, new_path):
    # Read audio files
    sr1, audio1 = read_audio(original_path)
    sr2, audio2 = read_audio(new_path)
    
    print(f"Original: SR={sr1}, samples={len(audio1)}")
    print(f"Resampled: SR={sr2}, samples={len(audio2)}")
    
    # Convert to mono if stereo
    if audio1.ndim > 1:
        audio1 = audio1.mean(axis=1)
    if audio2.ndim > 1:
        audio2 = audio2.mean(axis=1)
    
    # Apply windowing
    window1 = scipy_signal.windows.hann(len(audio1))
    window2 = scipy_signal.windows.hann(len(audio2))
    
    # Compute FFTs with windows
    fft1 = np.abs(np.fft.rfft(audio1 * window1))
    fft2 = np.abs(np.fft.rfft(audio2 * window2))
    
    # Frequency axes
    freqs1 = np.fft.rfftfreq(len(audio1), 1/sr1)
    freqs2 = np.fft.rfftfreq(len(audio2), 1/sr2)
    
    # Convert to dB
    mag1 = 20 * np.log10(fft1 + 1e-10)
    mag2 = 20 * np.log10(fft2 + 1e-10)
    
    # Interpolate to common frequency grid for comparison
    max_freq = min(freqs1.max(), freqs2.max(), 8000)
    common_freqs = np.linspace(0, max_freq, 5000)
    
    interp1 = interp1d(freqs1, mag1, kind='linear', fill_value='extrapolate')
    interp2 = interp1d(freqs2, mag2, kind='linear', fill_value='extrapolate')
    
    mag1_interp = interp1(common_freqs)
    mag2_interp = interp2(common_freqs)
    
    # Compute difference
    diff = mag2_interp - mag1_interp
    
    # Smooth versions for easier viewing
    smooth_window = 100
    mag1_smooth = smooth_spectrum(common_freqs, mag1_interp, smooth_window)
    mag2_smooth = smooth_spectrum(common_freqs, mag2_interp, smooth_window)
    diff_smooth = smooth_spectrum(common_freqs, diff, smooth_window)
    
    # Create figure with multiple views
    fig = plt.figure(figsize=(14, 10))
    
    # 1. Overlay comparison (smoothed)
    ax1 = fig.add_subplot(3, 2, 1)
    ax1.plot(common_freqs, mag1_smooth, 'b-', alpha=0.7, linewidth=1, label='Original')
    ax1.plot(common_freqs, mag2_smooth, 'r-', alpha=0.7, linewidth=1, label='Resampled')
    ax1.set_ylabel('Magnitude (dB)')
    ax1.set_title('Smoothed Spectra Overlay')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 8000)
    
    # 2. Difference plot (smoothed)
    ax2 = fig.add_subplot(3, 2, 2)
    ax2.fill_between(common_freqs, diff_smooth, 0, 
                     where=(diff_smooth > 0), color='green', alpha=0.5, label='Resampled > Original')
    ax2.fill_between(common_freqs, diff_smooth, 0,
                     where=(diff_smooth < 0), color='red', alpha=0.5, label='Resampled < Original')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_ylabel('Difference (dB)')
    ax2.set_title('Smoothed Difference (Resampled - Original)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 8000)
    
    # 3. Spectrogram-style difference heatmap
    ax3 = fig.add_subplot(3, 2, 3)
    # Create a 2D representation of the difference
    diff_2d = diff.reshape(1, -1)
    im = ax3.imshow(np.tile(diff_2d, (50, 1)), aspect='auto', 
                    extent=[0, max_freq, 0, 1],
                    cmap='RdBu_r', vmin=-10, vmax=10)
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_title('Difference Heatmap (Blue=Original louder, Red=Resampled louder)')
    ax3.set_yticks([])
    plt.colorbar(im, ax=ax3, label='dB difference')
    
    # 4. Absolute difference histogram
    ax4 = fig.add_subplot(3, 2, 4)
    ax4.hist(np.abs(diff), bins=100, edgecolor='black', alpha=0.7)
    ax4.axvline(x=np.mean(np.abs(diff)), color='r', linestyle='--', 
                label=f'Mean: {np.mean(np.abs(diff)):.2f} dB')
    ax4.axvline(x=np.median(np.abs(diff)), color='g', linestyle='--',
                label=f'Median: {np.median(np.abs(diff)):.2f} dB')
    ax4.set_xlabel('Absolute Difference (dB)')
    ax4.set_ylabel('Count')
    ax4.set_title('Distribution of Absolute Differences')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Difference by frequency band
    ax5 = fig.add_subplot(3, 2, 5)
    bands = [(0, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000)]
    band_labels = ['0-500', '500-1k', '1k-2k', '2k-4k', '4k-8k']
    band_diffs = []
    for low, high in bands:
        mask = (common_freqs >= low) & (common_freqs < high)
        band_diffs.append(np.mean(np.abs(diff[mask])))
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(bands)))
    bars = ax5.bar(band_labels, band_diffs, color=colors, edgecolor='black')
    ax5.set_xlabel('Frequency Band (Hz)')
    ax5.set_ylabel('Mean Absolute Difference (dB)')
    ax5.set_title('Difference by Frequency Band')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for bar, val in zip(bars, band_diffs):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    # 6. Running RMS difference
    ax6 = fig.add_subplot(3, 2, 6)
    window = 200
    rms_diff = np.sqrt(np.convolve(diff**2, np.ones(window)/window, mode='same'))
    ax6.plot(common_freqs, rms_diff, 'purple', linewidth=1)
    ax6.fill_between(common_freqs, rms_diff, alpha=0.3, color='purple')
    ax6.set_xlabel('Frequency (Hz)')
    ax6.set_ylabel('RMS Difference (dB)')
    ax6.set_title('Running RMS Difference')
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim(0, 8000)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Mean absolute difference: {np.mean(np.abs(diff)):.3f} dB")
    print(f"Max absolute difference: {np.max(np.abs(diff)):.3f} dB")
    print(f"RMS difference: {np.sqrt(np.mean(diff**2)):.3f} dB")
    print(f"Correlation coefficient: {np.corrcoef(mag1_interp, mag2_interp)[0,1]:.6f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-og', '--original', required=True)
    parser.add_argument('-nw', '--new', required=True)
    args = parser.parse_args()
    
    plot_fft_comparison(args.original, args.new)