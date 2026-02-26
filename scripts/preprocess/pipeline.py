import argparse

import os
import glob
import torch
import librosa
import soxr

import numpy as np

from nnAudio.features import STFT, CQT
from tqdm import tqdm, trange

from src.utils import load_audio, get_spectrum, get_fakeprints


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess audio files into fakeprint representations")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing original audio files")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to save processed fakeprint shards")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for processing files")
    parser.add_argument("--max_duration", type=float, default=30.0, help="Maximum duration (in seconds) to load from each audio file")
    parser.add_argument("--speed_up", action="store_true", help="Whether to apply random speed changes for data augmentation")
    parser.add_argument("--shard_size", type=int, default=500, help="Number of files to process per shard")
    parser.add_argument("--shard_start", type=int, default=0, help="Shard index to start from (for resuming)")
    parser.add_argument("--num_shards", type=int, default=None, help="Total number of shards to process (for resuming)")
    parser.add_argument("--n_fft", type=int, default=16384, help="FFT size")
    parser.add_argument("--sampling_rate", type=int, default=44100, help="Target sampling rate for audio")
    parser.add_argument("--bins_per_octave", type=int, default=192, help="Number of CQT bins per octave")
    parser.add_argument("--hull_area", type=int, default=20, help="Area parameter for lower hull in fakeprint extraction")
    parser.add_argument("--fmin", type=float, default=32.7, help="Minimum frequency for transforms (default is C1 note)")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for processing (e.g. 'cpu', 'mps' or 'cuda')")
    return parser.parse_args()


def is_valid_mp3(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None, duration=1.0)  # Try loading a short segment to check validity
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
    stft_transform,
    cqt_transform,
    batch_size=16,
    max_duration=60.0,
    manip_func=None,
    n_fft=16384,
    sampling_rate=44100,
    bins_per_octave=192,
    hull_area=20,
    device=torch.device("cpu"),
):
    
    hop_length = n_fft // 2

    stft_transform = stft_transform.to(device)
    cqt_transform = cqt_transform.to(device)
    
    stft_fakeprints = []
    cqt_fakeprints = []
    speed_factors = []

    num_batches = (len(file_paths) + batch_size - 1) // batch_size
    for i in trange(num_batches, leave=False, desc="Extracting fakeprints"):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(file_paths))
        batch_files = file_paths[start:end]

        batch_waves = []
        for path in tqdm(batch_files, leave=False, desc=f"Loading audio files for batch {i+1}/{num_batches}"):
            waveform, sr = load_audio(path, max_duration=max_duration)
            if manip_func is not None:
                speed_factor = np.random.uniform(0.9, 1.1) # Random speed factor for augmentation
                waveform = manip_func(waveform, sr, speed_factor)
                speed_factors.append(speed_factor)
            else:
                speed_factors.append(1.0)

            if sr != sampling_rate:
                waveform = soxr.resample(waveform.T, sr, sampling_rate, quality="VHQ").T
                waveform = torch.from_numpy(waveform)

            batch_waves.append(waveform)

        lengths = torch.tensor([w.shape[-1] for w in batch_waves], device=device)
        L_max = max(w.shape[-1] for w in batch_waves)

        padded = torch.zeros(len(batch_waves), 1, L_max)
        for k, w in enumerate(batch_waves):
            padded[k, :, :w.shape[-1]] = w
        padded = padded.to(device) # (B, 1, Lmax)

        stft_batch = get_spectrum(stft_transform, padded) # (B, n_bins, T')
        cqt_batch = get_spectrum(cqt_transform, padded) # (B, n_bins, T')

        T_frames = stft_batch.shape[-1]
        frame_lengths = ((lengths - n_fft) // hop_length + 1).unsqueeze(1) # (B, 1)
        mask = torch.arange(T_frames, device=device).unsqueeze(0) < frame_lengths # (B, T')

        # Average over time dimension, accounting for varying lengths
        stft_batch = (stft_batch * mask.unsqueeze(1)).sum(-1) / frame_lengths.float() # (B, n_bins)
        cqt_batch = (cqt_batch * mask.unsqueeze(1)).sum(-1) / frame_lengths.float() # (B, n_bins)

        stft_fp = get_fakeprints(stft_batch, area=hull_area)
        cqt_fp = get_fakeprints(cqt_batch, area=hull_area)

        stft_fakeprints.append(stft_fp)
        cqt_fakeprints.append(cqt_fp)

    stft_fakeprints = torch.cat(stft_fakeprints, dim=0)
    cqt_fakeprints = torch.cat(cqt_fakeprints, dim=0)
    speed_factors = np.array(speed_factors)

    return {
        "stft": stft_fakeprints.cpu().numpy(),
        "cqt": cqt_fakeprints.cpu().numpy(),
        "speed_factors": speed_factors,
        "n_fft": n_fft,
        "sampling_rate": sampling_rate,
        "bins_per_octave": bins_per_octave,
        "hull_area": hull_area,
    }

def pipeline(
    data_dir,
    out_dir,
    batch_size=16,
    max_duration=60.0,
    manip_func=None,
    shard_size=500,
    shard_start=0,
    num_shards=None,
    n_fft=16384,
    sampling_rate=44100,
    bins_per_octave=192,
    hull_area=20,
    fmin=32.7,
    device=torch.device("cpu"),
):
    os.makedirs(out_dir, exist_ok=True)
    file_paths = glob.glob(f"{data_dir}/**/*.mp3", recursive=True)
    file_paths = sorted([p for p in file_paths if is_valid_mp3(p)])
    shards = [file_paths[i:i+shard_size] for i in range(0, len(file_paths), shard_size)]
    print(f"Total files: {len(file_paths)}, Shards: {len(shards)}")

    hop_length = n_fft // 2
    fmax = sampling_rate / 2  # Maximum frequency that can be represented

    stft_transform = STFT(
        n_fft=n_fft,
        sr=sampling_rate,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax,
        output_format="Magnitude",
        verbose=False,
    )

    cqt_transform = CQT(
        sr=sampling_rate,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax,
        bins_per_octave=bins_per_octave,
        output_format="Magnitude",
        verbose=False,
    )

    end = len(shards) if not num_shards else min(shard_start + num_shards, len(shards))
    for i in trange(shard_start, end):
        shard_paths = shards[i]
        shard = preprocess_fakeprints(
            shard_paths,
            stft_transform=stft_transform,
            cqt_transform=cqt_transform,
            batch_size=batch_size,
            max_duration=max_duration,
            manip_func=manip_func,
            n_fft=n_fft,
            sampling_rate=sampling_rate,
            bins_per_octave=bins_per_octave,
            hull_area=hull_area,
            device=device,
        )
        np.savez(f"{out_dir}/fakeprints_{i+1:02d}.npz", **shard)


if __name__ == "__main__":
    args = parse_args()
    manip_func = speed_up if args.speed_up else None
    pipeline(
        args.data_dir,
        args.out_dir,
        batch_size=args.batch_size,
        max_duration=args.max_duration,
        manip_func=manip_func,
        shard_size=args.shard_size,
        shard_start=args.shard_start,
        num_shards=args.num_shards,
        n_fft=args.n_fft,
        sampling_rate=args.sampling_rate,
        bins_per_octave=args.bins_per_octave,
        hull_area=args.hull_area,
        fmin=args.fmin,
        device=torch.device(args.device),
    )