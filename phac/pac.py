
from collections import namedtuple
from typing import Tuple

import numpy as np

from .util import indices_of_binned_phase
from .metrics import _modulation_index
from .signal import Signal

PACResult: Tuple[float] = namedtuple("PACResult", "modulation_index mean_phase_coherence")

def phase_amplitude_coupling(samples, sr, slow_band, fast_band):
    signal = Signal(samples, sr)
    phase = signal.phase(slow_band)
    envelope = signal.envelope(fast_band)
    indices = indices_of_binned_phase(phase, num_bins=12)
    phi_avg = np.array([np.median(phase[idx]) for idx in indices])
    env_avg = np.array([np.median(envelope[idx]) for idx in indices])
    mi = _modulation_index(env_avg)
    mpc = (env_avg * np.exp(1.0j*phi_avg)).mean()
    return PACResult(modulation_index=mi, mean_phase_coherence=mpc)
