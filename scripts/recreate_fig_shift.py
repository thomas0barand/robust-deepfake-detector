"""
Recreate two figures:
  - Fakeprints of a generated track vs. the same track at alpha=0.84 (log-freq).
  - Cross-correlation between learned conv weights w and each fakeprint
    (x-axis mapped to speed modification factor).
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from src.models import RobustDetector


CKPT = "checkpoints/attack/suno_v5/robustdetector-sonics-log_stft-use_conv-lamb=0.1.ckpt"
FP_FILE = "data/sonics/noattack/train/suno_v5/fakeprints_02.npz"
FP_INDEX = 0
ALPHA = 0.84
OUT_FP = "outputs/fig_fakeprints_shift.png"
OUT_CC = "outputs/fig_crosscorr_shift.png"


def main():
    model = RobustDetector.load_from_checkpoint(CKPT, map_location="cpu").eval()

    weights = model.linear_proj.weights.detach().cpu().squeeze()  # (F,)
    F_dim = weights.numel()
    bpo = model.bins_per_octave_stft  # 1920

    # log-frequency axis matching the model's feature dim
    freqs = torch.logspace(
        math.log10(model.freq_range[0]),
        math.log10(model.freq_range[1]),
        steps=F_dim,
    ).numpy()

    # original fakeprint (alpha = 1) remapped to log-frequency
    data = np.load(FP_FILE)
    fp_raw = torch.from_numpy(data["stft"][FP_INDEX:FP_INDEX + 1]).float()  # (1, n_bins)
    fp_raw = fp_raw[:, model.mask]
    fp_log = model.stft_to_log(fp_raw).squeeze(0)  # (F,)

    # simulate alpha != 1 as a shift in log-freq (speed-up compresses freq axis)
    shift_bins = int(round(math.log2(ALPHA) * bpo))
    fp_shifted = torch.roll(fp_log, shifts=shift_bins)
    if shift_bins < 0:
        fp_shifted[shift_bins:] = 0
    elif shift_bins > 0:
        fp_shifted[:shift_bins] = 0

    # normalize for display
    def norm(x):
        return x / (x.max() + 1e-8)

    fp_a = norm(fp_log.numpy())
    fp_b = norm(fp_shifted.numpy())

    # -------- Plot 1: fakeprints with shift arrows --------
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(freqs, fp_a, color="tab:blue", linewidth=0.9, label="No speed modification")
    ax.plot(freqs, fp_b, color="tab:orange", linewidth=0.9, alpha=0.9,
            label=rf"Speed Modification of $\alpha = {ALPHA}$")
    ax.set_xscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylim(0, 1.05)

    # arrows between the top peaks of the original and their shifted counterparts
    top_idx = np.argsort(fp_a)[-20:]
    top_idx = top_idx[fp_a[top_idx] > 0.6]
    selected = []
    for i in sorted(top_idx, reverse=True):
        if all(abs(freqs[i] - freqs[j]) / freqs[j] > 0.05 for j in selected):
            selected.append(i)
        if len(selected) >= 4:
            break

    for i in selected:
        x_from = freqs[i]
        x_to = x_from * ALPHA
        y = max(fp_a[i], 0.75)
        ax.annotate("", xy=(x_to, y), xytext=(x_from, y),
                    arrowprops=dict(arrowstyle="->", color="dimgray", lw=0.9))

    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_FP, dpi=150, bbox_inches="tight")
    print(f"Saved {OUT_FP}")
    plt.close(fig)

    # -------- Plot 2: cross-correlation with learned weights --------
    def cross_corr(fp):
        fp_n = F.normalize(torch.clamp(fp, max=8).unsqueeze(0), p=2, dim=-1)
        x = fp_n.unsqueeze(1)                      # (1, 1, F)
        w = weights.unsqueeze(0).unsqueeze(0)      # (1, 1, F)
        cc = F.conv1d(x, w, padding="same").squeeze().numpy()
        return cc

    cc_a = cross_corr(fp_log)
    cc_b = cross_corr(fp_shifted)

    # map bin lag -> speed factor: alpha = 2^(lag / bpo)
    lags = np.arange(F_dim) - F_dim // 2
    sf_axis = 2 ** (lags / bpo)

    xlim = (0.65, 1.43)
    m = (sf_axis >= xlim[0]) & (sf_axis <= xlim[1])

    # convert to dB-ish scale for display (mirrors reference figure range)
    def to_db(x):
        return 20 * np.log10(np.abs(x) + 1e-8)

    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(sf_axis[m], to_db(cc_a[m]), color="tab:blue", linewidth=0.8,
            label="No speed modification")
    ax.plot(sf_axis[m], to_db(cc_b[m]), color="tab:orange", linewidth=0.8,
            label=rf"Speed Modification of $\alpha = {ALPHA}$")
    ax.set_xlabel("Speed Modification Factor")
    ax.set_xlim(xlim)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_CC, dpi=150, bbox_inches="tight")
    print(f"Saved {OUT_CC}")
    plt.close(fig)


if __name__ == "__main__":
    main()
