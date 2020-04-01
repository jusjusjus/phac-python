
from typing import List, Tuple

import numpy as np
from functools import lru_cache
from scipy.signal import butter as _butter
from scipy.signal import hilbert as _hilbert
from scipy.signal import filtfilt as _filtfilt

def indices_of_binned_phase(phase: np.ndarray, num_bins: int=18) -> List[np.ndarray]:
    """return list of indices each with values in bins
    
    Parameters
    ----------
    phase: np.ndarray
        phase variable with wrapped around the interval [0, 2*pi).

    num_bins: int, default=18
        number of equidistant bins in the range [0, 2*pi).
    """
    try:
        assert np.all(phase>=0), "All phase values must be greater or equal than 0"
        assert np.all(phase<2.0*np.pi), "All phase values must be smaller than 2*pi"
    except AssertionError as err:
        raise ValueError(str(err))
    sorting = np.argsort(phase)
    bin_limits = np.linspace(0, 2*np.pi, num=num_bins+1)
    bin_limit_idx = phase[sorting].searchsorted(bin_limits)
    return [sorting[slice(*ij)]
        for ij in zip(bin_limit_idx, bin_limit_idx[1:])]


def trapezoid(n: int, m: int, dtype=np.float64) -> np.ndarray:
    """return array with trapezoid values

    The returned array `t` rises in `m+1` steps linearly from near zero to
    `t[m+1]==1`, then to drop linearly from `t[-m-1]==1` to near zero.

    Example
    -------
    ```python
    t = trapzoid(9, 4)
    # t -> [0.2 0.4 0.6 0.8 1.  1.  0.8 0.6 0.4 0.2]
    ```
    
    Parameters
    ----------
    n: int
        length of the whole array
    m: int
        length of each ramp
    """
    try:
        assert n > 0, "segment length n < 0 (%s); must be larger"%n
        assert m > 0, "ramp length m < 0 (%s); must be larger"%m
        assert n > 2*m, "segment length %s < 2*%s"%(n, m)
    except AssertionError as err:
        raise ValueError(str(err))
    tr = np.linspace(0, 1, m+1, endpoint=False)[1:]
    middle = np.ones(n-2*m, dtype=np.float64)
    return np.concatenate([tr, middle, tr[::-1]]).astype(dtype)


def _highpass(sr: float, f: float) -> Tuple[np.ndarray, np.ndarray]:
    highpass_freq = 2*f/sr
    return _butter(4, highpass_freq, btype='high')

def _lowpass(sr: float, f: float) -> Tuple[np.ndarray, np.ndarray]:
    lowpass_freq = 2*f/sr
    return _butter(4, lowpass_freq, btype='low')

def _bandpass(sr: float, band: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    bandpass_freqs = 2*np.array(band)/sr
    return _butter(4, bandpass_freqs, btype='band')

@lru_cache(maxsize=128)
def _pass(sr: float, band: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    try:
        assert not all(f is None for f in band), "fmin and fmax is `None`."
        if band[1] is None:
            return _highpass(sr, band[0])
        elif band[0] is None:
            return _lowpass(sr, band[1])
        else:
            assert band[0] < band[1], "fmin >= fmax (fmin: %s, fmax: %s)"%band
            return _bandpass(sr, band)
    except AssertionError as err:
        raise ValueError(str(err))

def filtfilt(x: np.ndarray, sr: float, fmin: float=None, fmax: float=None, axis: int=-1) -> np.ndarray:
    b, a = _pass(sr, (fmin, fmax))
    return _filtfilt(b, a, x, axis=axis, padtype='constant', method='pad')


def downsample(x: np.ndarray, sr_old: float, sr_new: float, fmax: float) -> np.ndarray:
    """resample `x` linearly with sampling rate `sr_new`

    Applies 4-th order Butterworth low-pass filter with `fmax` to `x`.  Then
    resamples linearly to `sr_new`.  Resampled times start at `t[0]=0` and ends
    at `t[-1]=(int(np.round(x.size*sr_new/sr_old))-1)/sr_new`.  If
    `sr_new==sr_old`, resampling is not performed, but the filter is still
    applied.

    Parameters
    ----------
    x: np.ndarray
        samples to be down-sampled.
    sr_old: float
        sampling rate in Hz of `x`.
    sr_new: float
        sampling rate in Hz to which `x` shall be resampled.
    fmax:
        float low-pass edge of the filter to be applied to `x` before down-sampling.
    """
    # Filter before down-sampling
    try:
        assert sr_old > sr_new, "new sampling rate larger than old rate (sr_old: %s, sr_new: %s)"%(sr_old, sr_new)
        assert fmax <= 0.4 * sr_new, "fmax must be less than 0.8*f_nyquist of the new sampling rate (fmax: %s)"%fmax
    except AssertionError as err:
        raise ValueError(str(err))
    x = filtfilt(x, sr=sr_old, fmax=fmax)
    # Resample to new sampling rate
    if not sr_old == sr_new:
        t_old = np.arange(x.size)/sr_old
        num_samples_new = np.round(x.size/sr_old * sr_new)
        t_new = np.arange(int(num_samples_new))/sr_new
        x = np.interp(t_new, t_old, x)
    return x

def phase_difference(phi0: np.ndarray, phi1: np.ndarray) -> np.ndarray:
    """computes linear phase differences smaller than `pi`"""
    dphi = phi0-phi1
    dphi = np.mod(dphi+np.pi, 2*np.pi) - np.pi
    return dphi
