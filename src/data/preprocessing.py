import torch
import soxr
import librosa
import numpy as np

import src.utils.pyrubberband.pyrb as pyrb

def pitch_shift(waveform, sr, pitch_factor):
    # pitch_factor is a multiplicative frequency ratio (1.0 = no change,
    # 2.0 = one octave up, 0.5 = one octave down). Must be strictly positive.
    if pitch_factor == 1.0:
        return waveform

    device = waveform.device
    audio_np = waveform.cpu().numpy()  # (channels, samples)

    shifted_channels = [
        pyrb.pitch_shift_2(channel, sr, ratio=pitch_factor)
        for channel in audio_np
    ]

    return torch.from_numpy(np.stack(shifted_channels, axis=0)).to(device)