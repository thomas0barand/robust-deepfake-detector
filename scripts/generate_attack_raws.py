import os
import csv
import argparse
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Literal
from src.utils import load_audio
from src.data.preprocessing import pitch_shift, time_stretch


def apply_attack(tracks_folder, attack_type: Literal["time_stretch", "pitch_shift"], lo_bound: float, hi_bound: float, custom_output_filename: str = None):
    """
    Apply either time stretch or pitch shift attack to raw audio.
    """
    if not custom_output_filename:
        attacked_tracks_path = tracks_folder + "_" + attack_type
    else:
        attacked_tracks_path = custom_output_filename

    os.makedirs(attacked_tracks_path, exist_ok=True)
    metadata = dict()

    all_files = list(Path(tracks_folder).iterdir())
    print(f"Found {len(all_files)} files in {tracks_folder}")

    attack_transform = None
    if attack_type == "time_stretch":
        attack_transform = time_stretch
    elif attack_type == "pitch_shift":
        attack_transform = pitch_shift
    else:
        raise ValueError(f"Unknown attack type: {attack_type!r}. Expected 'time_stretch' or 'pitch_shift'.")

    # Apply attack
    for i, file_path in enumerate(all_files):
        waveform, sr = load_audio(str(file_path))
        if waveform is None:
            print(f"  [{i+1}/{len(all_files)}] Skipping {file_path.name} (failed to load)")
            continue
        ratio = np.random.uniform(lo_bound, hi_bound) # Continuous speed factors for augmentation
        attacked_waveform = attack_transform(waveform, sr, ratio)
        metadata[file_path.name] = ratio
        out_path = os.path.join(attacked_tracks_path, file_path.name)
        sf.write(out_path, attacked_waveform.squeeze(0).numpy(), sr)
        print(f"  [{i+1}/{len(all_files)}] {file_path.name} — ratio: {ratio}")

    print(f"Done. {len(metadata)} files saved to {attacked_tracks_path}")

    # Save dictionary as CSV
    csv_path = os.path.join(attacked_tracks_path, "metadata.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", attack_type + "_ratio"])
        for name, ratio in metadata.items():
            writer.writerow([name, ratio])
    print(f"Metadata saved to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply time stretch or pitch shift attack to raw audio files.")
    parser.add_argument("tracks_folder", type=str, help="Path to the folder containing audio files.")
    parser.add_argument("--attack_type", type=str, choices=["time_stretch", "pitch_shift"], default="time_stretch", help="Attack type to apply (default: time_stretch).")
    parser.add_argument("--lo", type=float, default=0.8, help="Lower bound of the attack ratio (default: 0.8).")
    parser.add_argument("--hi", type=float, default=1.2, help="Upper bound of the attack ratio (default: 1.2).")
    parser.add_argument("--out_filename", type=str, default=None, help="Custom output filename.")

    args = parser.parse_args()

    apply_attack(args.tracks_folder, args.attack_type, args.lo, args.hi, args.out_filename)