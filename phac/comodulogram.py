
import itertools

import numpy as np
import pandas as pd

from .signal import Signal 
from .frequency_band import FrequencyBand
from .metrics import _modulation_index
from .util import indices_of_binned_phase
from .filter_series import FilterSeries

def comodulogram(samples: np.ndarray, sampling_rate: float,
        slow_filters: FilterSeries, fast_filters: FilterSeries) -> pd.DataFrame:
    assert isinstance(slow_filters, FilterSeries)
    assert isinstance(fast_filters, FilterSeries)
    signal = Signal(samples, sampling_rate)
    # compute band-filtered phase of the slow component
    phases_by_freq = {
        band.center: signal.phase(band)
        for band in slow_filters
    }
    # Compute bin indices from the phase signals
    bin_idxs_by_freq = {
        center_freq: indices_of_binned_phase(phase)
        for center_freq, phase in phases_by_freq.items()
    }
    # compute band-filtered amplitudes of the fast component
    amps_by_freq = {
        band.center: signal.envelope(band)
        for band in fast_filters
    }
    # Average fast-band amplitudes within slow-band phase bins
    avg_amp_by_freqs = {
        (f_slow, f_fast): np.array([np.median(amp[idx]) for idx in idxs])
        for (f_slow, idxs), (f_fast, amp) in itertools.product(
            bin_idxs_by_freq.items(), amps_by_freq.items()
        )
    }
    # compute modulation indices from the average
    mi = {
        freqs: _modulation_index(amps)
        for freqs, amps in avg_amp_by_freqs.items()
    }
    s = pd.Series(data=mi)
    s.index.names = ['f_slow', 'f_fast']
    return s.unstack('f_slow')
