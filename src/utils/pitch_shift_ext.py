import numpy as np
import pyrubberband


def pitch_shift_2(y, sr, ratio, rbargs=None):
    """Apply pitch shift using a frequency ratio (1.0 = no change, 2.0 = octave up)."""
    if ratio == 1.0:
        return y
    n_steps = 12.0 * np.log2(ratio)
    return pyrubberband.pitch_shift(y, sr, n_steps, rbargs=rbargs)
