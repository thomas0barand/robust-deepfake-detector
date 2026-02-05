import torchaudio

import numpy as np
import matplotlib.pyplot as plt

from utils import get_stft, get_cqt, get_fakeprint


N_FFT = 1 << 14  # 16384

new_sr = 16000
nyquist = new_sr / 2
F_MIN = 32.7  # C1 note frequency
BINS_PER_OCTAVE = 96
HOP_LENGTH = N_FFT // 2
MODE = "cqt"  # "stft" or "cqt"

n_octaves = np.log2(nyquist / F_MIN) - 0.1
N_BINS = int(n_octaves * BINS_PER_OCTAVE)

F_RANGE = [500, 5000]


def plot_corr(freqs1, fp1, freqs2, fp2, log_scale=True):
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


if __name__ == "__main__":
    audio_path = "data/ai/fake_00001_suno_0.mp3"
    su_audio_path = "data/signals/spedup/fake_suno_0_su.mp3"

    waveform, sr = torchaudio.load(audio_path)
    su_waveform, su_sr = torchaudio.load(su_audio_path)
    print(f"Loaded audio with shape {waveform.shape} and sample rate {sr}")
    print(f"Loaded sped-up audio with shape {su_waveform.shape} and sample rate {su_sr}")
    
    stft = get_stft(waveform, N_FFT, HOP_LENGTH)
    su_stft = get_stft(su_waveform, N_FFT, HOP_LENGTH)

    cqt = get_cqt(waveform, sr, N_BINS, BINS_PER_OCTAVE, HOP_LENGTH)
    su_cqt = get_cqt(su_waveform, su_sr, N_BINS, BINS_PER_OCTAVE, HOP_LENGTH)

    spectrum = cqt if MODE=="cqt" else stft
    su_spectrum = su_cqt if MODE=="cqt" else su_stft

    freqs, fp = get_fakeprint(spectrum, sr, f_range=F_RANGE, mode=MODE, log=False)
    su_freqs, su_fp = get_fakeprint(su_spectrum, su_sr, f_range=F_RANGE, mode=MODE, log=False)
    
    plot_corr(freqs, fp, su_freqs, su_fp, log_scale=True)
