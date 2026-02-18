import os
import argparse
from glob import glob
from tqdm import tqdm
import torchaudio

def process_pitch_shift(input_dir, output_dir, n_steps):
    """
    n_steps: relative change in semitones. 
    Can be positive (higher pitch) or negative (lower pitch).
    """
    output_dir = os.path.join(output_dir, f"ps_{n_steps}")
    os.makedirs(output_dir, exist_ok=True)
    search_path = os.path.join(input_dir, "*.mp3")
    files = glob(search_path)
    print(f"Found {len(files)} files in {input_dir}")

    for f_path in tqdm(files):
        filename = os.path.basename(f_path)
        save_path = os.path.join(output_dir, filename)

        waveform, sr = torchaudio.load(f_path)
        # n_steps is in semitones (12 semitones = 1 octave)
        pitch_shifter = torchaudio.transforms.PitchShift(sample_rate=sr, n_steps=n_steps)
        shifted_waveform = pitch_shifter(waveform)
        torchaudio.save(save_path, shifted_waveform, sr, format="mp3")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pitch shift all MP3s in a folder")
    parser.add_argument("--input", type=str, default="data/human", help="Input folder")
    parser.add_argument("--output", type=str, default="attack/pitch_shift_human", help="Output folder")
    parser.add_argument("--semitones", type=float, default=2.0, help="Steps to shift (e.g., 2.0 or -1.5)")
    
    args = parser.parse_args()

    process_pitch_shift(args.input, args.output, args.semitones)