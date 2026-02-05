import os
import numpy as np
import argparse
import torchaudio
import matplotlib.pyplot as plt
from scipy import interpolate

# Constants
N_FFT = 1 << 14

def get_stft(waveform):
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    stft_transformer = torchaudio.transforms.Spectrogram(n_fft=N_FFT, power=2)
    stft = stft_transformer(waveform).numpy()
    stft = 10 * np.log10(np.clip(stft, 1e-10, 1e6))
    return stft

def lower_hull(x, area=10):
    idx, hull = [], []
    for i in range(len(x)-area+1):
        patch = x[i:i+area]
        rel_idx = np.argmin(patch)
        abs_idx = rel_idx + i
        if abs_idx not in idx:
            idx.append(abs_idx)
            hull.append(patch[rel_idx])
    
    if len(idx) < 2:
        return np.array([0, len(x)-1]), np.array([x[0], x[-1]])

    if idx[0] != 0:
        idx.insert(0, 0); hull.insert(0, x[0])
    if idx[-1] != len(x)-1:
        idx.append(len(x)-1); hull.append(x[-1])
    return np.array(idx), np.array(hull)

def get_fakeprint(stft, sr, f_range=[5000, 16000]):
    fp = np.mean(stft, axis=(0, 2))
    x_real = np.linspace(0, sr / 2, num=len(fp))
    actual_fmax = min(f_range[1], sr // 2)
    mask = (x_real >= f_range[0]) & (x_real <= actual_fmax)
    
    x_crop = x_real[mask]
    fp_crop = fp[mask]
    
    lower_x, lower_c = lower_hull(fp_crop, area=10)
    low_hull_curve = interpolate.interp1d(x_crop[lower_x], lower_c, kind="quadratic")(x_crop)
    
    residue = np.clip(fp_crop - low_hull_curve, 0, None)
    residue = residue / (1e-6 + np.max(residue))
    
    return x_crop, residue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Original Fakeprint with 8 Pitch-Shifted versions.")
    parser.add_argument("--orig", type=str, required=True, help="Path to original .mp3 file")
    parser.add_argument("--ps_dir", type=str, default="attack/pitch_shift_human",  help="Path to directory containing ps_ subdirs")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--fmin", type=int, default=0)
    parser.add_argument("--fmax", type=int, default=8000)
    parser.add_argument("--out", type=str, default="ps_grid.png")
    parser.add_argument("--log", action="store_true", help="Plot frequency on a log10 scale")
    args = parser.parse_args()

    ps_variants = ["ps_-2.0", "ps_-1.5", "ps_-1.0", "ps_-0.5", "ps_0.5", "ps_1.0", "ps_1.5", "ps_2.0"]
    song_name = os.path.basename(args.orig)
    
    wav_o, sr_o = torchaudio.load(args.orig)
    if sr_o != args.sr:
        wav_o = torchaudio.transforms.Resample(sr_o, args.sr)(wav_o)
    
    stft_o = get_stft(wav_o)
    x_orig, res_orig = get_fakeprint(stft_o, args.sr, f_range=[args.fmin, args.fmax])

    # Transform to Log if requested
    if args.log:
        # Avoid log(0) by clipping to a small positive value
        x_orig_plot = np.log10(np.clip(x_orig, 1, None))
        x_label = "Log10 Frequency (Hz)"
        x_lims = [np.log10(max(1, args.fmin)), np.log10(args.fmax)]
    else:
        x_orig_plot = x_orig
        x_label = "Frequency (Hz)"
        x_lims = [args.fmin, args.fmax]

    # 2. Setup Plotting
    fig, axes = plt.subplots(3, 3, figsize=(18, 14), sharex=False, sharey=True)
    axes = axes.flatten()

    # Plot Original
    axes[0].plot(x_orig_plot, res_orig, color='blue', lw=1.2)
    axes[0].set_title(f"ORIGINAL\n{song_name}", fontsize=10, fontweight='bold')
    axes[0].set_xlabel(x_label, fontsize=8)
    axes[0].set_xlim(x_lims)
    axes[0].grid(True, alpha=0.3)

    # 3. Load and plot each Pitch-Shifted version
    for i, ps_folder in enumerate(ps_variants):
        ax = axes[i + 1] 
        ps_path = os.path.join(args.ps_dir, ps_folder, song_name)
        
        if not os.path.exists(ps_path):
            ax.text(0.5, 0.5, f"File Missing:\n{ps_folder}", ha='center', color='gray')
            ax.set_title(f"Attack: {ps_folder}", fontsize=10)
            continue

        wav_a, sr_a = torchaudio.load(ps_path)
        if sr_a != args.sr:
            wav_a = torchaudio.transforms.Resample(sr_a, args.sr)(wav_a)
        
        stft_a = get_stft(wav_a)
        x_att, res_att = get_fakeprint(stft_a, args.sr, f_range=[args.fmin, args.fmax])

        # Apply log transform to attacked x-axis
        x_att_plot = np.log10(np.clip(x_att, 1, None)) if args.log else x_att

        # Plot comparison
        ax.plot(x_orig_plot, res_orig, color='blue', alpha=0.25, label="Orig")
        ax.plot(x_att_plot, res_att, color='red', lw=1.1, label=ps_folder)
        ax.set_title(f"Attack: {ps_folder}", fontsize=10)
        ax.set_xlabel(x_label, fontsize=8)
        ax.set_xlim(x_lims)
        ax.grid(True, alpha=0.3)

    # Final Layout adjustments
    plt.suptitle(f"Fakeprint Analysis (Pitch Shift Attacks)\nFile: {song_name} {'(Log Scale)' if args.log else ''}", fontsize=16, y=0.98)
    fig.text(0.02, 0.5, 'Normalized Residue Amplitude', va='center', rotation='vertical', fontsize=12)
    
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.93])
    plt.savefig(args.out, dpi=200)
    print(f"Grid visualization saved to {args.out}")