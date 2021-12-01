import numpy as np
from typing import Tuple
from ..util import filtfilt


def random_sin(t: np.ndarray, frequency: float, dphi: float) -> np.ndarray:
    random_phase_walk = dphi * np.random.randn(t.size).cumsum()
    return np.sin(2.*np.pi*frequency*t + random_phase_walk)


def filtered_noise(n: int, sr: float, band: Tuple[float, float]) -> np.ndarray:
    """"""
    x = np.random.randn(n)
    x = filtfilt(x, sr, fmin=band[0], fmax=band[1])
    x = x/np.std(x)
    return x


def validate_parameters(frequency, dphi, band, amplitude, coupling,
                        sampling_rate):
    """raise ValueError if parameters outside the acceptable range"""
    try:
        assert dphi < frequency/3, "dphi (%s) should be smaller than \
                25 pcnt the frequency (%s Hz)" % (dphi, frequency)
        assert band[1] is None or band[0] < band[1], "bandpass parameters \
                must be ordered %s" % str(band)
        assert band[0] > frequency, "Lower band limit of '%s' should be > \
                frequency=%s" % (band, frequency)
        assert band[1] is None or band[1] < sampling_rate/2, "Upper band \
                limit of '%s' should be < sampling rate=%s" % (
                        band, sampling_rate)
        assert amplitude >= 0.0, "amplitude of noisy modulated signal \
                should be larger or equal than zero"
        assert 0 <= coupling <= 1, "coupling has to be between 0 (no \
                coupling) and 1 (full coupling)"
    except AssertionError as err:
        raise ValueError(str(err))


def sin_with_noise(t: np.ndarray, frequency: float = 20.0,
                   dphi: float = 3.0, band: Tuple[float, float] = (50.0, 90.0),
                   amplitude: float = 0.5,
                   coupling: float = 0.5) -> np.ndarray:
    """
    Sinusoidal rhythm with modulated high-frequency noise have properties

    Parameters
    ----------
    `t`: np.ndarray
        time points at which to sample the output, in seconds.
    `frequency`: float
        in Hz.  Average frequency of the sin wave
    `dphi`: float
        Wave-to-wave fluctuations in frequency
    `band`: 2-tuple
        frequency minimum and maximum for band-filtered noise
    `amplitude`: float
        Amplitude of the filtered noise that is modulated.
    `coupling`: float
        in (0, 1).  How much of the noise is coupling into the signal
    """
    sr = 1/(t[1]-t[0])
    validate_parameters(frequency, dphi, band, amplitude, coupling, sr)
    scaled_dphi = dphi*np.sqrt(1/sr)
    x = random_sin(t, frequency, scaled_dphi)
    dx = amplitude * filtered_noise(t.size, sr, band)
    # compute normalized amplitude modulation (0 - 1)
    modulation = x-x.min()
    modulation /= modulation.max()
    x += (1+(modulation-1)*coupling) * dx
    return x
