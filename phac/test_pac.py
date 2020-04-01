
import pytest
import numpy as np
from .pac import phase_amplitude_coupling, PACResult
from .models import sin_with_noise

@pytest.mark.parametrize('coupling, expected', [
    (0.1, PACResult(0.000849, 0)),
    (0.3, PACResult(0.000956, 0)),
    (0.5, PACResult(0.001295, 0)),
    (0.7, PACResult(0.00215, 0)),
    (0.9, PACResult(0.003522, 0)),
])
def test_pac_on_sin_with_noise(coupling, expected):
    """Check PAC values for `sin_with_noise`"""
    np.random.seed(42)
    sr = 256.0 # Hz
    T = 10.0 # sec.
    t = np.arange(int(T*sr))/sr
    x = sin_with_noise(t,
        frequency = 20.0, # Hz
        dphi = 3.0,
        band = (40.0, 100), # Hz
        amplitude = 0.5,
        coupling = coupling
    )
    pac = phase_amplitude_coupling(x, sr,
        slow_band = (15.0, 25.0),
        fast_band = (70.0-15.0, 70.0+15.0)
    )
    assert pac.modulation_index == pytest.approx(expected.modulation_index, abs=1e-5)
