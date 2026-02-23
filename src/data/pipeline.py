import argparse

import os
import glob
import librosa
import torch
import torchaudio
import soxr

import numpy as np

from nnAudio.features import CQT
from torchaudio.transforms import Spectrogram
from tqdm import tqdm, trange

from src.models.utils import get_spectrum, get_freqs, get_fakeprints


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess audio files into fakeprint representations")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing original audio files")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to save processed fakeprint shards")
    parser.add_argument("--speed_up", action="store_true", help="Whether to apply random speed changes for data augmentation")
    parser.add_argument("--shard_size", type=int, default=500, help="Number of files to process per shard")
    parser.add_argument("--shard_start", type=int, default=0, help="Shard index to start from (for resuming)")
    parser.add_argument("--num_shards", type=int, default=None, help="Total number of shards to process (for resuming)")
    parser.add_argument("--n_fft", type=int, default=16384, help="FFT size")
    parser.add_argument("--sampling_rate", type=int, default=48000, help="Target sampling rate for audio")
    parser.add_argument("--bins_per_octave", type=int, default=96, help="Number of CQT bins per octave")
    parser.add_argument("--freq_range", type=int, nargs=2, default=[200, 6000], metavar=("F_MIN", "F_MAX"), help="Frequency range to keep in the fakeprints")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for processing (e.g. 'cpu' or 'cuda')")
    return parser.parse_args()


def is_valid_mp3(file_path):
    try:
        # duration=1.0 only decodes the first second of audio
        y, sr = librosa.load(file_path, sr=None, duration=1.0)
        return True
    except Exception as e:
        print(f"Error opening {file_path}: {e}")
        return False


def speed_up(waveform, sr, speed_factor):
    if speed_factor == 1.0:
        return waveform
    
    device = waveform.device
    new_sr = int(sr * speed_factor)
    resampled_waveform = soxr.resample(waveform.cpu().T, sr, new_sr, quality="VHQ").T

    return torch.from_numpy(resampled_waveform).to(device)


def preprocess_fakeprints(
    file_paths,
    manip_func=None,
    n_fft=16384,
    sampling_rate=48000,
    bins_per_octave=96,
    freq_range=[1000, 22000],
    db_range=[-80, 5],
    f_min=32.7, # C1 note frequency
    device=torch.device("cpu"),
):
    assert device.type != "mps", "MPS device is not supported for this preprocessing pipeline. Please use CPU or CUDA."

    hop_length = n_fft // 2
    nyquist = sampling_rate / 2  # Maximum frequency that can be represented
    n_octaves = np.log2(nyquist / f_min) - 0.1  # Subtract a small margin to ensure we don't exceed Nyquist
    nbins = int(n_octaves * bins_per_octave)  # Total number of CQT bins to cover the desired frequency range

    cqt_transform = CQT(
        sr=sampling_rate,
        hop_length=hop_length,
        fmin=f_min,
        n_bins=nbins,
        bins_per_octave=bins_per_octave,
        output_format="Magnitude",
        verbose=False,
    ).to(device)

    stft_transform = Spectrogram(n_fft=n_fft, power=2, hop_length=hop_length).to(device)

    cqt_freqs, cqt_mask = get_freqs(
        n_fft=n_fft,
        sr=sampling_rate,
        transform="cqt",
        bins_per_octave=bins_per_octave,
        freq_range=freq_range,
        f_min=f_min
    )

    stft_freqs, stft_mask = get_freqs(
        n_fft=n_fft,
        sr=sampling_rate,
        transform="stft",
        bins_per_octave=bins_per_octave,
        freq_range=freq_range,
        f_min=f_min
    )

    cqt_fakeprints = []
    stft_fakeprints = []
    speed_factors = []
    for path in tqdm(file_paths, leave=False):
        try:
            waveform, sr = torchaudio.load(path, channels_first=True)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            continue
        
        if manip_func is not None:
            speed_factor = np.random.uniform(0.9, 1.1) # Random speed factor for augmentation
            waveform = manip_func(waveform, sr, speed_factor)
            speed_factors.append(speed_factor)
        else:
            speed_factors.append(1.0)

        if sr != sampling_rate:
            waveform = soxr.resample(waveform.T, sr, sampling_rate, quality="VHQ").T
            waveform = torch.from_numpy(waveform).to(device)

        waveform = waveform.mean(dim=0, keepdim=True).to(device)  # Convert to mono

        cqt = get_spectrum(cqt_transform, waveform) # (1, n_bins, T')
        cqt = cqt.mean(dim=-1).squeeze(0)  # (n_bins,)

        stft = get_spectrum(stft_transform, waveform) # (1, n_bins, T')
        stft = stft.mean(dim=-1).squeeze(0)  # (n_bins,)
        
        cqt_spec_crop = cqt[cqt_mask]
        cqt_fp = get_fakeprints(cqt_spec_crop, cqt_freqs, db_range=db_range)
        cqt_fakeprints.append(cqt_fp)

        stft_spec_crop = stft[stft_mask]
        stft_fp = get_fakeprints(stft_spec_crop, stft_freqs, db_range=db_range)
        stft_fakeprints.append(stft_fp)

    return {
        "cqt": torch.stack(cqt_fakeprints, dim=0).cpu().numpy(),
        "stft": torch.stack(stft_fakeprints, dim=0).cpu().numpy(),
        "speed_factors": np.array(speed_factors),
        "n_fft": n_fft,
        "sampling_rate": sampling_rate,
        "bins_per_octave": bins_per_octave,
        "freq_range": freq_range,
        "db_range": db_range,
        "f_min": f_min,
    }

def pipeline(
    data_dir,
    out_dir,
    manip_func=None,
    shard_size=500,
    shard_start=0,
    num_shards=None,
    n_fft=16384,
    sampling_rate=48000,
    bins_per_octave=96,
    freq_range=[200, 6000],
    device=torch.device("cpu"),
):
    os.makedirs(out_dir, exist_ok=True)
    file_paths = glob.glob(f"{data_dir}/**/*.mp3", recursive=True)
    file_paths = sorted([p for p in file_paths if is_valid_mp3(p)])
    shards = [file_paths[i:i+shard_size] for i in range(0, len(file_paths), shard_size)]
    print(f"Total files: {len(file_paths)}, Shards: {len(shards)}")

    end = len(shards) if not num_shards else min(shard_start + num_shards, len(shards))
    for i in trange(shard_start, end):
        shard_paths = shards[i]
        shard = preprocess_fakeprints(
            shard_paths,
            manip_func=manip_func,
            n_fft=n_fft,
            sampling_rate=sampling_rate,
            bins_per_octave=bins_per_octave,
            freq_range=freq_range,
            device=device,
        )
        np.savez(f"{out_dir}/fakeprints_{i+1:02d}.npz", **shard)


if __name__ == "__main__":
    args = parse_args()
    manip_func = speed_up if args.speed_up else None
    pipeline(
        args.data_dir,
        args.out_dir,
        manip_func=manip_func,
        shard_size=args.shard_size,
        shard_start=args.shard_start,
        num_shards=args.num_shards,
        n_fft=args.n_fft,
        sampling_rate=args.sampling_rate,
        bins_per_octave=args.bins_per_octave,
        freq_range=args.freq_range,
        device=torch.device(args.device),
    )