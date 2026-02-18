import numpy as np
import soundfile as sf
import subprocess

# Parameters
sample_rate = 44100
duration = 5  # seconds
frequencies = [200, 1000, 5000, 10000]  # Hz

output_path = "data/signals/originals/signal2.mp3"

# Generate time array
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

# Generate harmonics (equal amplitude for each)
signal = np.zeros_like(t)
for freq in frequencies:
    signal += np.sin(2 * np.pi * freq * t)

# Normalize harmonics
signal = signal / len(frequencies)

# Add very light noise (barely audible)
noise = np.random.normal(0.01, 0.1, len(t))
signal = signal + noise

# Normalize to prevent clipping
signal = signal / np.max(np.abs(signal)) * 0.9

# Convert to 16-bit PCM
signal_int = (signal * 32767).astype(np.int16)

# Pipe directly to ffmpeg (no temp file)
process = subprocess.Popen([
    "ffmpeg", "-y",
    "-f", "s16le",
    "-ar", str(sample_rate),
    "-ac", "1",
    "-i", "pipe:0",
    "-codec:a", "libmp3lame", "-qscale:a", "2",
    output_path
], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

process.stdin.write(signal_int.tobytes())
process.stdin.close()
process.wait()

print(f"Audio saved to: {output_path}")