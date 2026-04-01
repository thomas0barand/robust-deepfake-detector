import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

CKPT_BCE  = "checkpoints/attack/robustdetector-log_stft-use_conv-lamb=0.0.ckpt"
CKPT_HYB  = "checkpoints/attack/robustdetector-log_stft-use_conv-lamb=0.1.ckpt"
OUT_PATH  = "paper/figures/weights_comparison.png"

FREQ_RANGE      = [5000, 16000]
BINS_PER_OCTAVE = 1920


def load_weights(path):
    ck = torch.load(path, map_location="cpu")
    return ck["state_dict"]["linear_proj.weights"].squeeze().numpy()


def log_freq_axis(fmin, fmax, n_bins):
    return np.logspace(math.log10(fmin), math.log10(fmax), n_bins)


def main():
    w_bce = load_weights(CKPT_BCE)
    w_hyb = load_weights(CKPT_HYB)

    assert w_bce.shape == w_hyb.shape, "Weight shapes differ between checkpoints"
    n_bins = w_bce.shape[0]
    freqs  = log_freq_axis(*FREQ_RANGE, n_bins)  # Hz, log-spaced

    fig, axes = plt.subplots(2, 1, figsize=(7, 4.5), sharex=True)
    fig.subplots_adjust(hspace=0.08)

    plot_cfg = [
        (w_bce, "#d62728", r"$\lambda = 0$  (BCE only)"),
        (w_hyb, "#1f77b4", r"$\lambda = 0.1$  (hybrid loss)"),
    ]

    for ax, (w, color, label) in zip(axes, plot_cfg):
        ax.plot(freqs , w, color=color, linewidth=0.6, label=label)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)
        ax.set_ylabel("Weight value", fontsize=8)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.7)
        ax.tick_params(labelsize=7)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ax.grid(axis="x", which="both", linestyle=":", linewidth=0.4, alpha=0.5)

    axes[1].set_xlabel("Frequency (Hz)", fontsize=8)

    # log x-axis with explicit ticks in kHz
    ticks = [k * 1000 for k in [5, 6, 7, 8, 9, 10, 12, 14, 16]]
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
        ax.xaxis.set_minor_locator(ticker.NullLocator())

    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    print(f"Saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
