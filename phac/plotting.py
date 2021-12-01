import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt

from .signal import Signal
from .util import indices_of_binned_phase
from .metrics import _modulation_index


def plot_phase_amplitude_decomposition(x: np.ndarray, sr: float,
                                       slow_band: Tuple[float, float],
                                       fast_band: Tuple[float, float]):
    limits = x.min(), x.max()
    dl = 0.1*(limits[1]-limits[0])
    limits = (limits[0]-dl, limits[1]+dl)
    x = Signal(x, sr)
    slow_filtered = x.filtered(slow_band)
    fast_filtered = x.filtered(fast_band)
    envelope = x.envelope(fast_band)

    # plt.figure(figsize=(20, 7))

    ax = plt.subplot(211)
    bandstr = '-'.join(map(str, slow_band))
    plt.title("Raw and slow-filtered (%s) signal" % bandstr, fontsize=15)
    plt.plot(x.time, x.signal, 'k-')
    plt.plot(x.time, slow_filtered, 'r--')
    plt.grid()

    plt.subplot(212, sharex=ax, sharey=ax)
    bandstr = '-'.join(map(str, fast_band))
    plt.title("Fast-filtered (%s) signal and envelope" % bandstr, fontsize=15)
    plt.plot(x.time, fast_filtered, 'r--')
    plt.plot(x.time, envelope, 'g-')
    plt.xlim(x.time[0], x.time[-1])
    plt.ylim(*limits)
    plt.grid()

    plt.tight_layout()


def plot_phase_amplitude_coupling(x: np.ndarray, sr: float,
                                  slow_band: Tuple[float, float],
                                  fast_band: Tuple[float, float]):
    limits = x.min(), x.max()
    dl = 0.1*(limits[1]-limits[0])
    limits = (limits[0]-dl, limits[1]+dl)
    x = Signal(x, sr)
    phase = x.phase(slow_band)
    envelope = x.envelope(fast_band)

    indices = indices_of_binned_phase(phase, num_bins=12)
    phi_avg = np.array([np.mean(phase[idx]) for idx in indices])
    env_avg = np.array([np.mean(envelope[idx]) for idx in indices])
    mi = _modulation_index(env_avg)

    # plt.figure(figsize=(5, 3))
    plt.title("MI(%.2g-%.2g,%.2g-%.2g) = %.3g" % (
        slow_band[0] or 0.0, slow_band[1], fast_band[0], fast_band[1], mi))
    plt.plot(phase, envelope, 'ko', ms=3, alpha=0.4)
    plt.plot(phi_avg, env_avg, 'ro', ms=5, mec='k', label='binned median')
    plt.xlim(0, 2*np.pi)
    plt.xlabel("Phase (rad)", fontsize=15)
    plt.ylabel(r"Amplitude ($\mu$V)", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend()
    plt.grid()

    plt.tight_layout()


def plot_comodulogram(C, **kwargs):
    assert C.columns.name == 'f_slow' and C.index.name == 'f_fast'
    f_slow = list(C.columns)
    f_fast = list(C.index)
    extent = [min(f_slow), max(f_slow)] + [min(f_fast), max(f_fast)]
    plt.imshow(C, origin='lower', aspect='auto', extent=extent, **kwargs)
    plt.colorbar()
