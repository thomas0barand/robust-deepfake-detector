import os
import glob
import numpy as np
import torch

from torch.utils.data import Dataset


class FakeprintDataset(Dataset):
    def __init__(self, data_dir: str, mode: str = "cqt"):
        self.samples = []  # list of (fakeprint, label)

        for label, subdir in [(0, "human"), (1, "ai")]:
            npz_paths = sorted(glob.glob(os.path.join(data_dir, subdir, "*.npz")))
            for path in npz_paths:
                data = np.load(path)
                fakeprints = data[mode]  # (N, feature_dim)
                for fp in fakeprints:
                    self.samples.append((fp, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fp, label = self.samples[idx]
        return torch.from_numpy(fp).float(), torch.tensor(label).float()