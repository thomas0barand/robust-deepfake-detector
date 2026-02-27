import os
import glob
import numpy as np
import torch

from torch.utils.data import Dataset
from src.utils import get_freqs, get_freqs_mask


class FakeprintDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        mode: str = "cqt",
        freq_range: list = [200, 16000],
        n_fft: int = 16384,
        sampling_rate: int = 44100,
        bins_per_octave: int = 192,
    ):
        self.samples = []  # list of (fakeprint, label)

        log = (mode == "cqt")
        freqs = get_freqs(n_fft=n_fft, sr=sampling_rate, log=log, bins_per_octave=bins_per_octave)
        mask = get_freqs_mask(freqs, sampling_rate, freq_range=freq_range)

        for label, subdir in [(0, "human"), (1, "ai")]:
            npz_paths = sorted(glob.glob(os.path.join(data_dir, subdir, "*.npz")))
            for path in npz_paths:
                data = np.load(path)
                assert n_fft == data["n_fft"].item()
                assert sampling_rate == data["sampling_rate"].item()
                assert bins_per_octave == data["bins_per_octave"].item()
                fakeprints = data[mode]  # (N, feature_dim)
                fakeprints = fakeprints[:, mask]  # (N, feature_dim_masked)
                for fp in fakeprints:
                    self.samples.append((fp, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fp, label = self.samples[idx]
        return torch.from_numpy(fp).float(), torch.tensor(label).float()
