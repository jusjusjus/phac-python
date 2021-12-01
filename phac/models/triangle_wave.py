import numpy as np


def square_wave(t: np.ndarray, frequency: float,
                jitter: float, tilt: float) -> np.ndarray:
    period = 1/frequency
    y = np.zeros(t.shape[0])
    for ti in np.arange(0, t[-1], period):
        tti = ti + jitter*period*np.random.randn()
        n0, n1 = t.searchsorted((tti, tti+period*tilt))
        y[n0:n1] = 1.0

    return y-y.mean()


def triangle_wave(t: np.ndarray,
                  frequency: float = 20.0,
                  jitter: float = 0.2,
                  tilt: float = 0.8) -> np.ndarray:
    """return triangular wave patterns sampled at t

    Parameters
    ----------

    `frequency`: float, default is 20.0
        Average period of the triangular wave

    `jitter`: float, default is 0.2
        from 0 to 1.  Wave-to-wave fluctuations in period

    `tilt`: float, default is 0.8
        from 0 to 1.  Whether the wave is tilted backwards (at `tilt=0`) or
        forward (at `tilt=1`)
    """
    return np.cumsum(square_wave(t, frequency, jitter, tilt))
