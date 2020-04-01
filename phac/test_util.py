
from .util import (
    indices_of_binned_phase,
    trapezoid,
    phase_difference,
    downsample,
    np
)
import pytest


@pytest.mark.parametrize("phi", [-0.1, 2.*np.pi])
def test_indices_of_binned_phase_raises_error(phi):
    with pytest.raises(ValueError):
        phase = np.array([1, phi, 5])
        indices_of_binned_phase(phase, num_bins=3)


def test_indices_of_binned_phase():
    phase = np.array([np.pi, 0, 2.*np.pi-0.1])
    indices = indices_of_binned_phase(phase, num_bins=3)
    for idx, i in zip(indices, [1, 0, 2]):
        assert idx.size == 1
        assert idx[0] == i 


@pytest.mark.parametrize("n, m", [
    (-1, 21), # negative
    (64, -1), # negative
    (64, 32), # 2*m > n
])
def test_trapezoid_raises_error(n, m):
    with pytest.raises(ValueError):
        trapezoid(n, m)


@pytest.mark.parametrize("n, m", [
    (9, 4),
    (10, 4),
    (127, 33),
])
def test_trapezoid(n, m):
    arr = trapezoid(n, m)
    assert arr.size == n
    assert np.all(arr[:m] < 1.0)
    assert np.all(arr[-m:] < 1.0)
    assert arr[m] == arr[-m-1] == pytest.approx(1.0)
    assert arr[m:-m] == pytest.approx(1.0)
    assert arr[:m]+arr[-m:] == pytest.approx(1.0)


@pytest.mark.parametrize("factor", [
    1/5, 1/4, 1/3, 1/2, 2/3,
])
def test_downsample(factor):

    def sin(sampling_rate, f=1.0, T=10.0):
        t = np.arange(int(sampling_rate*T))/sampling_rate
        return np.sin(2*np.pi*f*t)

    # original sampling rate
    sr = 50.0 # Hz
    # new sampling rate
    sr_new = sr*factor # Hz
    arr = sin(sr)
    expected = sin(sr_new)
    resampled = downsample(arr, sr, sr_new, fmax=0.4*sr_new)
    # because of edge effects the first and last 10% aren't as precise.
    sl = slice(int(0.1*expected.size), int(0.9*expected.size))
    assert resampled[sl] == pytest.approx(expected[sl], abs=1e-2)

@pytest.mark.parametrize("phase", np.linspace(0, 2*np.pi, num=20))
def test_phase_difference(phase):
    """test phase difference of `phase+0.1 - phase`
    
    it is assumed that differences < dphi are valid, whereas values > dphi are
    potentially biased, b/c we wrap around the circle.
    """
    dphi = np.pi-0.1
    phi0 = np.mod(phase+dphi, 2*np.pi)
    phi1 = np.mod(phase, 2*np.pi)
    assert phase_difference(phi0, phi1) == pytest.approx(dphi)
