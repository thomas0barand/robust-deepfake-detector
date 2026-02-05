import torch
import torchaudio
import soxr
import numpy as np

from utils import get_stft, get_cqt, get_fakeprint, plot_fp, plot_correlation


N_FFT = 1 << 14  # 16384

SR = 48000
nyquist = SR / 2
F_MIN = 32.7  # C1 note frequency
BINS_PER_OCTAVE = 96
HOP_LENGTH = N_FFT // 2
MODE = "cqt"  # "stft" or "cqt"
AI = True

n_octaves = np.log2(nyquist / F_MIN) - 0.1
N_BINS = int(n_octaves * BINS_PER_OCTAVE)

F_RANGE = [200, 6000]  # Frequency range for fakeprint analysis


if __name__ == "__main__":
    if AI:
        audio_path = "/Users/emiledugelay/datasets/suno_v5_500/0a0ad7222cab7a839aa40ed228af2ad2a6654c97a3a7077a0358b215857e35c1.mp3"
        su_audio_path = "data/signals/spedup/sunov5_s102.mp3"
    else:
        audio_path = "data/human/human2.mp3"
        su_audio_path = "data/signals/spedup/human2_s102.mp3"

    waveform, sr = torchaudio.load(audio_path, channels_first=False)
    su_waveform, su_sr = torchaudio.load(su_audio_path, channels_first=False)
    print(f"Loaded audio with shape {waveform.shape} and sample rate {sr}")
    print(f"Loaded sped-up audio with shape {su_waveform.shape} and sample rate {su_sr}")

    if sr != SR:
        waveform = soxr.resample(waveform, sr, SR)
        sr = SR
    if su_sr != SR:
        su_waveform = soxr.resample(su_waveform, su_sr, SR)
        su_sr = SR
    
    stft = get_stft(waveform, N_FFT, HOP_LENGTH)
    su_stft = get_stft(su_waveform, N_FFT, HOP_LENGTH)

    cqt = get_cqt(waveform, sr, N_BINS, BINS_PER_OCTAVE, HOP_LENGTH)
    su_cqt = get_cqt(su_waveform, su_sr, N_BINS, BINS_PER_OCTAVE, HOP_LENGTH)

    spectrum = cqt if MODE=="cqt" else stft
    su_spectrum = su_cqt if MODE=="cqt" else su_stft

    freqs, fp = get_fakeprint(spectrum, sr, f_range=F_RANGE, mode=MODE)
    su_freqs, su_fp = get_fakeprint(su_spectrum, su_sr, f_range=F_RANGE, mode=MODE)

    print(cqt.shape)
    print(freqs.shape, fp.shape)
    
    plot_fp(freqs, fp, su_freqs, su_fp, log_scale=True)
    refined_lag, refined_val = plot_correlation(fp, su_fp)

    # load model weights and plot vs frequency
    ckpt_path = "src/checkpoints/models/robustdetector-use_conv=False.ckpt"
    ckpt = torch.load(ckpt_path)
    state_dict = ckpt["state_dict"]
    weights = state_dict["linear_proj.weights"].cpu().numpy().squeeze()
    
    plot_correlation(su_fp, weights)
