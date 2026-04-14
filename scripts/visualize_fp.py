import glob
import numpy as np
import matplotlib.pyplot as plt

from src.utils import get_freqs, get_freqs_mask


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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/suno_v3_5/train/attack", help="Directory containing .npz files")
    parser.add_argument("--transform", type=str, choices=["stft", "cqt"], default="stft", help="Transform type to visualize")
    parser.add_argument("--freq_range", nargs=2, type=int, default=[1000, 7000], help="Frequency range to plot")
    parser.add_argument("--max_files", type=int, default=5, help="Maximum number of .npz files to show")
    args = parser.parse_args()

    file_paths = glob.glob(f"{args.data_dir}/fakeprints_*.npz")
    file_path = np.random.choice(file_paths, size=1)[0]

    print(f"Loading fakeprints from: {file_path}")
    file = np.load(file_path)

    fakeprints = file[args.transform]

    n_fft = file["n_fft"].item()
    sampling_rate = file["sampling_rate"].item()
    bins_per_octave = file["bins_per_octave"].item()
    log = (args.transform == "cqt")

    freqs = get_freqs(
        n_fft=n_fft,
        sr=sampling_rate,
        log=log,
        bins_per_octave=bins_per_octave,
    )

    mask = get_freqs_mask(freqs, sampling_rate, freq_range=args.freq_range)

    freqs = freqs[mask]
    fakeprints = fakeprints[:, mask]

    plot_fp(freqs, fakeprints[:args.max_files], log_scale=log)
