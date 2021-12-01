from typing import Tuple

import numpy as np

from cached_property import cached_property

from .hilbert import hilbert
from .util import filtfilt


class Signal:

    def __init__(self, signal: np.ndarray, sampling_rate: float):
        self.signal = signal
        self.sampling_rate = sampling_rate

    @cached_property
    def time(self) -> np.ndarray:
        return np.arange(self.signal.size)/self.sampling_rate

    def filtered(self, band: Tuple[float, float]) -> np.ndarray:
        try:
            assert isinstance(band, tuple), f"band must be tuple, got '{band}'"
            assert len(band) == 2, f"band '{band}'"
        except AssertionError as err:
            raise ValueError(str(err))
        return filtfilt(self.signal, self.sampling_rate,
                        fmin=band[0], fmax=band[1])

    def phase(self, band) -> np.ndarray:
        x = self.filtered(band)
        phi = np.angle(hilbert(x)) + np.pi/2
        return np.mod(phi, 2*np.pi)

    @staticmethod
    def _hilbert_envelope(filtered_signal: np.ndarray) -> np.ndarray:
        return np.abs(hilbert(filtered_signal))

    @staticmethod
    def _max_envelope(filtered_signal: np.ndarray) -> np.ndarray:
        x = filtered_signal - filtered_signal.mean()
        x = np.abs(x)
        maxidx = np.concatenate([
            [False], (x[2:] < x[1:-1]) & (x[:-2] < x[1:-1]), [False]])
        maxis = x[maxidx]
        idx = np.arange(maxidx.size)
        return np.interp(idx, idx[maxidx], maxis)

    def envelope(self, band, method: str = 'hilbert') -> np.ndarray:
        envelope_fn = {
            'hilbert': self._hilbert_envelope,
            'max': self._max_envelope
        }
        x = self.filtered(band)
        try:
            return envelope_fn[method](x)
        except KeyError:
            raise ValueError(f"method ({method}) must be one of "
                             f"{envelope_fn.keys()}")
