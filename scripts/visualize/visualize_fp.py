import numpy as np
import matplotlib.pyplot as plt

from src.utils.fakeprints import get_freqs, get_freqs_mask


F_RANGE = [200, 10000]
TRANSFORM = "cqt"
SHOW_AI_FP = False

N = 10


ai_dir = "data/train/ai"
human_dir = "data/train/human"


dir_path = ai_dir if SHOW_AI_FP else human_dir
print(f"Loading fakeprints from: {dir_path}")

file = np.load(f"{dir_path}/fakeprints_01.npz")

fakeprints = file[TRANSFORM]
n_fft = file["n_fft"].item()
sampling_rate = file["sampling_rate"].item()
bins_per_octave = file["bins_per_octave"].item()
f_min = file["f_min"].item()

log = (TRANSFORM in ["cqt", "vqt"])
freqs = get_freqs(
    n_fft=n_fft,
    sr=sampling_rate,
    log=log,
    bins_per_octave=bins_per_octave,
    f_min=f_min,
)

mask = get_freqs_mask(freqs, sampling_rate, freq_range=F_RANGE)

freqs = freqs[mask]
fakeprints = fakeprints[:, mask]


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