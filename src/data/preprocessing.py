import torch
import soxr
import numpy as np

from src.utils.pitch_shift_ext import pitch_shift_2


def pitch_shift(waveform, sr, pitch_factor):
    # pitch_factor is a multiplicative frequency ratio (1.0 = no change,
    # 2.0 = one octave up, 0.5 = one octave down). Must be strictly positive.
    if pitch_factor == 1.0:
        return waveform

    device = waveform.device
    audio_np = waveform.cpu().numpy()  # (channels, samples)

    shifted_channels = [
        pitch_shift_2(channel, sr, ratio=pitch_factor)
        for channel in audio_np
    ]

    return torch.from_numpy(np.stack(shifted_channels, axis=0)).to(device)

def resample(waveform, sr, speed_factor):
    # waveform is a tensor of shape (channels, samples)
    if speed_factor == 1.0:
        return waveform
    
    device = waveform.device
    new_sr = int(sr / speed_factor)
    resampled_waveform = soxr.resample(waveform.cpu().numpy().T, sr, new_sr, quality="VHQ").T

    return torch.from_numpy(resampled_waveform).to(device)
