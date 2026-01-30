"""
python scripts/attack/soxr/visualise/fft.py -og data/signals/signal1.wav -nw data/signals/signal1_rs.wav
"""

"""
Fixed version of compute_fft.py
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal as scipy_signal

from pydub import AudioSegment

def read_audio(file_path):
    audio = AudioSegment.from_file(file_path)
    sr = audio.frame_rate
    samples = np.array(audio.get_array_of_samples())
    if audio.channels == 2:
        samples = samples.reshape((-1, 2))
    return sr, samples


def plot_fft_comparison(original_path, new_path):
    # Read audio files
    sr1, audio1 = read_audio(original_path)
    sr2, audio2 = read_audio(new_path)
    
    # Convert to mono if stereo
    if audio1.ndim > 1:
        audio1 = audio1.mean(axis=1)
    if audio2.ndim > 1:
        audio2 = audio2.mean(axis=1)
    
    # APPLY WINDOWING
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
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    
    ax1.plot(freqs1, mag1, linewidth=0.5)
    ax1.set_ylabel('Magnitude (dB)')
    ax1.set_title('Original (with Hann window)')
    ax1.set_ylim(bottom=np.max(mag1) - 100)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(freqs2, mag2, linewidth=0.5)
    ax2.set_ylabel('Magnitude (dB)')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_title('Resampled (with Hann window)')
    ax2.set_ylim(bottom=np.max(mag2) - 100)
    ax2.grid(True, alpha=0.3)
    
    ax2.set_xlim(0, 25000)
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-og', '--original', required=True)
    parser.add_argument('-nw', '--new', required=True)
    args = parser.parse_args()
    
    plot_fft_comparison(args.original, args.new)