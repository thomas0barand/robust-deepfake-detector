import os
import glob
import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset
from src.utils import get_freqs, get_freqs_mask


class FakeprintDataset(Dataset):
    def __init__(
        self,
        ai_dir,
        human_dir,
        mode="stft",
        freq_range: list = [5000, 16000],
        n_fft: int = 16384,
        sampling_rate: int = 44100,
        bins_per_octave: int = 192,
    ):
        self.ai_dir = ai_dir
        self.human_dir = human_dir
        self.mode = mode
        self.freq_range = freq_range
        self.n_fft = n_fft
        self.sampling_rate = sampling_rate
        self.bins_per_octave = bins_per_octave

        self.samples = []  # list of (fakeprint, label)

        log = (mode == "cqt")
        freqs = get_freqs(n_fft=n_fft, sr=sampling_rate, log=log, bins_per_octave=bins_per_octave)
        mask = get_freqs_mask(freqs, sampling_rate, freq_range=freq_range)

        for label, directory in [(0, self.human_dir), (1, self.ai_dir)]:
            npz_paths = sorted(glob.glob(os.path.join(directory, "*.npz")))
            for path in npz_paths:

                data = np.load(path)

                assert n_fft == data["n_fft"].item()
                assert sampling_rate == data["sampling_rate"].item()
                assert bins_per_octave == data["bins_per_octave"].item()

                fakeprints = data[mode]
                fakeprints = fakeprints[:, mask]  # (N, feature_dim)
                attack_factors = data.get("attack_factors", [1.0])  # Optional attack factors for augmentation

                for fp, attack_factor in zip(fakeprints, attack_factors):
                    
                    self.samples.append((fp, label, attack_factor))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fp, label, attack_factor = self.samples[idx]
        return torch.from_numpy(fp).float(), torch.tensor(label).float(), torch.tensor(attack_factor).float()
