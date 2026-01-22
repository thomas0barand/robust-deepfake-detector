import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--sr", type=int, default=44100, help="Sampling rate used for fakeprints")
parser.add_argument("--fmin", type=int, default=5000, help="Low freq cutoff")
parser.add_argument("--fmax", type=int, default=16000, help="High freq cutoff")
args = parser.parse_args()

# Load weights
weights_data = np.load("outputs/models/weights.npy", allow_pickle=True).item()
W = weights_data["W"].flatten()
B = weights_data["B"][0]

# Compute frequency axis (matching compute_fakeprints.py)
N_FFT = 1 << 14  # 16384
SR = args.sr
FMIN = args.fmin
FMAX = args.fmax

# Full frequency range
x_freqs = np.linspace(0, SR / 2, num=(N_FFT//2)+1)

# Filter to the range used in fakeprints
freq_mask = (x_freqs >= FMIN) & (x_freqs <= min(FMAX, SR/2))
frequencies = x_freqs[freq_mask]

# Ensure same length as weights
frequencies = frequencies[:len(W)]

print(f"Number of features: {len(W)}")
print(f"Sampling rate: {SR} Hz")
print(f"Frequency range: [{frequencies[0]:.0f}, {frequencies[-1]:.0f}] Hz")
print(f"Intercept: {B:.4f}")
print(f"Weight range: [{W.min():.4f}, {W.max():.4f}]")
print(f"Mean: {W.mean():.4f}, Std: {W.std():.4f}")

# Plot: weight values vs frequency
plt.figure(figsize=(14, 6))
plt.plot(frequencies, W, linewidth=0.8, alpha=0.8)
plt.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Weight Value')
plt.title('Logistic Regression Weights vs Frequency (positive = AI indicator, negative = Human indicator)')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/figures/weights.png', dpi=300, bbox_inches='tight')
print("\nSaved to outputs/figures/weights.png")
plt.show()
