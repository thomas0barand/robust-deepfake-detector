#!/usr/bin/env python3
"""
Resample audio to new sample rate

Run:
python scripts/attack/soxr/compute/resampling.py data/signals/originals/signal1.mp3 --sr 16000 -n signal1_rs.mp3
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import soxr
import soundfile as sf

parser = argparse.ArgumentParser(description="Resample audio and plot FFT comparison")
parser.add_argument("input", help="Input audio file")
parser.add_argument("--sr", type=int, default=None, help="Target sample rate (default: keep original)")
parser.add_argument("-n", type=str, required=True, help="name of the audio file to resample")
args = parser.parse_args()

# load and resample
audio, sr = sf.read(args.input) 
if audio.ndim > 1:
    audio = audio.mean(axis=1)

new_sr = args.sr if args.sr else sr
audio_rs = soxr.resample(audio, sr, new_sr) if sr != new_sr else audio

# save the file
name = args.n
sf.write(f"data/signals/resampled/{name}", audio_rs, new_sr)