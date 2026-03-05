import os
import sys
import torch
import numpy as np
import glob

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
os.chdir(project_root)

ai_dir = "src/checkpoints/fp/ai"
human_dir = "src/checkpoints/fp/human"

os.makedirs(os.path.join(ai_dir), exist_ok=True)
os.makedirs(os.path.join(human_dir), exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

N_FFT = 1 << 14  # 16384
SR = 48000
nyquist = SR / 2
F_MIN = 32.7  # C1 note frequency
BINS_PER_OCTAVE = 96
F_RANGE = [200, 6000]

from src.models.utils import preprocess_fakeprints

num_samples = 500

data_dir = "data/ai"

file_paths = glob.glob(f"{data_dir}/*.mp3")
file_paths = file_paths[:num_samples]
if not file_paths:
    raise FileNotFoundError(f"No .mp3 files found in {data_dir}")

ai_fp = preprocess_fakeprints(
    file_paths,
    n_fft=N_FFT,
    sampling_rate=SR,
    bins_per_octave=BINS_PER_OCTAVE,
    freq_range=F_RANGE,
    device=DEVICE,
)


np.savez(f"{ai_dir}/fakeprints_01.npz", **ai_fp)



import glob
from src.models.utils import preprocess_fakeprints

num_samples = 500

data_dir = "/path/to/datasets/fma_small"

file_paths = glob.glob(f"{data_dir}/**/*.mp3", recursive=True)
file_paths = file_paths[:num_samples]

human_fp = preprocess_fakeprints(
    file_paths,
    n_fft=N_FFT,
    sampling_rate=SR,
    bins_per_octave=BINS_PER_OCTAVE,
    freq_range=F_RANGE,
    device=DEVICE,
)

np.savez(f"{human_dir}/fakeprints_01.npz", **human_fp)


from src.models.utils import get_freqs

TRANSFORM = "cqt"
SHOW_AI_FP = True

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



import matplotlib.pyplot as plt

N = 10

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