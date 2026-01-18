import numpy as np
from scipy.signal import resample_poly


def gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return abs(a)


def reduce_ratio(up: int, down: int):
    g = gcd(up, down)
    return up // g, down // g


def parse_window(window_str: str):
    """
    Interpretable window options for the FIR prototype used by resample_poly.
    SciPy accepts:
      - ('kaiser', beta)
      - 'hann', 'hamming', 'blackman', etc.
    """
    w = window_str.strip().lower()
    if w.startswith("kaiser"):
        # formats accepted: "kaiser,12.0" or "kaiser:12.0"
        sep = "," if "," in w else (":" if ":" in w else None)
        if sep is None:
            raise ValueError("Kaiser window needs a beta, e.g. 'kaiser,12.0'")
        beta = float(w.split(sep, 1)[1])
        return ("kaiser", beta)
    return w  # e.g. 'hann', 'hamming', 'blackman'


def to_float32(x: np.ndarray):
    # Convert int PCM to float32 in [-1, 1]
    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        # handle signed/unsigned
        if info.min < 0:
            scale = max(abs(info.min), info.max)
            return (x.astype(np.float32) / scale).clip(-1.0, 1.0)
        else:
            return (x.astype(np.float32) / info.max).clip(0.0, 1.0) * 2 - 1
    return x.astype(np.float32)


def float32_to_int16(x: np.ndarray):
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).round().astype(np.int16)


def resample_audio(
    x: np.ndarray,
    up: int,
    down: int,
    window,
    pad_mode: str,
):
    """
    Core resampling. All parameters are explicit and modifiable.
    - up/down : resampling ratio
    - window  : filter/window shaping (interpretability: transition/stopband tradeoff)
    - pad_mode: controls boundary behavior (can affect phase/edge artifacts)
    """
    # resample_poly works per channel; handle mono/stereo/multi-channel
    if x.ndim == 1:
        return resample_poly(x, up, down, window=window, padtype=pad_mode)
    else:
        ys = []
        for ch in range(x.shape[1]):
            ys.append(resample_poly(x[:, ch], up, down, window=window, padtype=pad_mode))
        return np.stack(ys, axis=1)
