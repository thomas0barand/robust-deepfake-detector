import numpy as np
import matplotlib.pyplot as plt

from src.models.detector import RobustDetector


def plot_weights(
    ckpt_path: str,
    output_path: str = None,
):
    model = RobustDetector.load_from_checkpoint(ckpt_path)
    model.eval()

    # Extract linear weights (1, feature_dim)
    weights = model.linear_proj.weights.detach().cpu().numpy().squeeze()  # (feature_dim,)
    freqs = model.freqs.detach().cpu().numpy()  # (feature_dim,)

    assert len(weights) == len(freqs), f"Shape mismatch: weights {weights.shape} vs freqs {freqs.shape}"

    plt.figure(figsize=(14, 6))
    plt.plot(freqs, weights, linewidth=0.8, alpha=0.8)
    plt.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Weight Value')
    plt.title('Logistic Regression Weights vs Frequency (positive = AI indicator, negative = Human indicator)')
    plt.grid(alpha=0.3)
    plt.xscale("log")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, help="Path to .ckpt file")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save the figure")
    args = parser.parse_args()

    plot_weights(args.ckpt_path, output_path=args.output)