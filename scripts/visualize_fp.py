import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def visualize_fakeprints(input_file, output_dir, sr, fmin, fmax, n_fft):
    # Load the data
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} not found.")
        return
    
    data = np.load(input_file, allow_pickle=True).item()
    print(f"Loaded {len(data)} fingerprints from {input_file}")

    # Re-calculate frequency axis for the x-axis labels
    # The fakeprint function in your script clips the frequencies between fmin and fmax
    # We need to find the specific indices to match the 1D arrays stored in the .npy
    full_freq_axis = np.linspace(0, sr / 2, num=(n_fft // 2) + 1)
    mask = (full_freq_axis > fmin) & (full_freq_axis < fmax)
    x_freqs = full_freq_axis[mask]

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Plotting loop
    for filename, residues in data.items():
        plt.figure(figsize=(10, 4))
        
        # Plot the residue curve
        # If residues length differs slightly due to rounding, we adjust the x_freqs
        current_x = x_freqs[:len(residues)]
        
        plt.plot(current_x, residues, color='#1f77b4', lw=1.5)
        
        plt.title(f"Fakeprint (Peak Residues): {filename}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Normalized Amplitude")
        plt.grid(True, alpha=0.3)
        plt.ylim(-0.05, 1.05)
        plt.tight_layout()

        # Save the plot
        filename = filename.replace(".mp3", "")
        save_path = os.path.join(output_dir, f"{filename}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved visualization to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize fakeprint residues as PNG plots")
    parser.add_argument("--load", help="path to the .npy fakeprint file", type=str, required=True)
    parser.add_argument("--out", help="directory to save pngs", type=str, default="outputs/figures/fakeprints/human")
    parser.add_argument("--sr", help="sampling rate (must match compute script)", type=int, default=16000)
    parser.add_argument("--fmin", type=int, default=0)
    parser.add_argument("--fmax", type=int, default=8000)
    parser.add_argument("--n_fft", type=int, default=1 << 14)
    
    args = parser.parse_args()

    visualize_fakeprints(args.load, args.out, args.sr, args.fmin, args.fmax, args.n_fft)