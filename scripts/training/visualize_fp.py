import numpy as np
import matplotlib.pyplot as plt

from src.models.utils import get_freqs

N_FFT = 16384
SR = 48000
BINS_PER_OCTAVE = 96
F_RANGE = [200, 6000]
F_MIN = 32.7

TRANSFORM = "stft"
SHOW_AI_FP = True

N = 10


ai_dir = "src/checkpoints/fp/ai"
human_dir = "src/checkpoints/fp/human"

dir_path = ai_dir if SHOW_AI_FP else human_dir
print(f"Loading fakeprints from: {dir_path}")

def load_fp(file_path):
    file = np.load(file_path)
    fakeprints = file[TRANSFORM]
    return fakeprints

fakeprints = load_fp(f"{dir_path}/fakeprints_01.npz")

freqs, _ = get_freqs(
    n_fft=N_FFT,
    transform=TRANSFORM,
    sr=SR,
    bins_per_octave=BINS_PER_OCTAVE,
    freq_range=F_RANGE,
    f_min=F_MIN,
)

def plot_fp(freqs, fakeprints, log_scale=True):
    plt.figure(figsize=(12, 5))
    for i, fp in enumerate(fakeprints):
        plt.plot(freqs, fp, label=f"Fakeprint {i}", alpha=0.8)
    if log_scale:
        plt.xscale('log')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Normalized Residue')
    plt.title('Fakeprint Comparison')
    plt.legend()
    plt.grid()
    plt.show()

plot_fp(freqs, fakeprints[:N], log_scale=True)