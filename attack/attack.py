import argparse
import os

import numpy as np
from scipy.io import wavfile

from resampling.resampling import to_float32, reduce_ratio, parse_window, resample_audio, float32_to_int16


p = argparse.ArgumentParser(description="Deterministic, parameterized audio resampling script.")
p.add_argument("--in", dest="in_path", required=True, help="Input audio path (WAV recommended).")
p.add_argument("--out", dest="out_path", required=True, help="Output audio path (WAV).")

# Two ways to specify the resampling:
group = p.add_mutually_exclusive_group(required=True)
group.add_argument("--ratio", type=str, help="Resampling ratio as 'up/down' (e.g. 160/147).")
group.add_argument("--to_sr", type=int, help="Target sample rate in Hz (uses input SR to derive ratio).")

p.add_argument("--window", type=str, default="kaiser,12.0",
               help="FIR window: 'hann'|'hamming'|'blackman'| 'kaiser,<beta>' (default: kaiser,12.0)")
p.add_argument("--pad", type=str, default="reflect",
               choices=["constant", "edge", "linear_ramp", "maximum", "mean", "median",
                        "minimum", "reflect", "symmetric", "wrap"],
               help="Padding mode passed to resample_poly (boundary behavior). Default: reflect.")
p.add_argument("--dtype_out", type=str, default="int16", choices=["int16", "float32"],
               help="Output PCM dtype. Default: int16.")

args = p.parse_args()

# Load (WAV)
sr_in, x = wavfile.read(args.in_path)
x_f = to_float32(x)

# Determine up/down
if args.ratio:
    up_s, down_s = args.ratio.split("/")
    up, down = int(up_s), int(down_s)
    if up <= 0 or down <= 0:
        raise ValueError("up/down must be positive integers.")
    up, down = reduce_ratio(up, down)
    sr_out = int(round(sr_in * (up / down)))
else:
    sr_out = int(args.to_sr)
    if sr_out <= 0:
        raise ValueError("to_sr must be positive.")
    # Make a rational approximation with bounded denominator for interpretability
    # You can tighten/loosen max_den to trade exactness vs. simple ratios
    max_den = 10_000
    frac = sr_out / sr_in
    # rational approximation
    down = max_den
    up = int(round(frac * down))
    up, down = reduce_ratio(up, down)
    sr_out = int(round(sr_in * (up / down)))

window = parse_window(args.window)

y = resample_audio(
    x_f,
    up=up,
    down=down,
    window=window,
    pad_mode=args.pad,
)

# Write output
if args.dtype_out == "int16":
    y_out = float32_to_int16(y)
else:
    y_out = y.astype(np.float32)

# Ensure directory exists
os.makedirs(os.path.dirname(os.path.abspath(args.out_path)), exist_ok=True)
wavfile.write(args.out_path, sr_out, y_out)

print(f"Input SR: {sr_in} Hz")
print(f"Resample ratio: {up}/{down} = {up/down:.10f}")
print(f"Output SR (written): {sr_out} Hz")
print(f"Wrote: {args.out_path}")