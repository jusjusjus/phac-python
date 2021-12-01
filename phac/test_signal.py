import pytest
import numpy as np
from .signal import Signal
from .util import filtfilt, phase_difference


def time(sampling_rate, T=10.0):
    return np.arange(int(sampling_rate*T))/sampling_rate


def phase(t, f, phi0=0.0):
    return 2*np.pi*f*t + phi0


@pytest.fixture
def signal_and_time():
    sr = 128.0  # Hz
    T = 10.0  # s
    f = 5.13435  # Hz
    t = time(sr, T)
    s = np.sin(phase(t, f))
    signal = Signal(s, sr)
    return t, signal


@pytest.fixture
def signal(signal_and_time):
    return signal_and_time[1]


def test_time(signal_and_time):
    t, signal = signal_and_time
    assert signal.time == pytest.approx(t)


@pytest.mark.parametrize('band', [35.0, (10,)])
def test_filtered_fails(band, signal):
    with pytest.raises(ValueError):
        signal.filtered(band=band)


def test_filtered():
    sr = 128.0
    band = (20.0, 50.0)
    x = np.random.randn(1024)
    signal = Signal(x, sr)
    expected = filtfilt(x, sr, fmin=band[0], fmax=band[1])
    assert signal.filtered(band) == pytest.approx(expected)


@pytest.mark.parametrize('sr, f, band', [
    (32.0, 5.0, (3.0, 7.0)),
    (32.0, 10.0, (8.0, 12.0)),
    (128.0, 10.0, (5.0, 15.0)),
    (512.0, 20.0, (5.0, 35.0)),
    (512.0, 50.0, (30.0, 60.0)),
])
def test_phase(sr, f, band):
    phi0 = 2.0*np.pi*np.random.rand()
    t = time(sr)
    phi = np.mod(phase(t, f, phi0), 2*np.pi)
    x = np.sin(phi) + 0.05*np.random.randn(phi.size)
    signal = Signal(x, sr)
    sl = slice(int(0.1*phi.size), int(0.9*phi.size))
    dphi = phase_difference(signal.phase(band), phi)
    assert np.mean(dphi) == pytest.approx(0.0, abs=1e-2)
    assert np.std(dphi) == pytest.approx(0.0, abs=1e-1)
    assert dphi[sl] == pytest.approx(0.0, abs=1e-1)


@pytest.mark.parametrize('sr, f, band', [
    (32.0, 5.0, (3.0, 7.0)),
    (32.0, 10.0, (8.0, 12.0)),
    (128.0, 10.0, (5.0, 15.0)),
    (512.0, 20.0, (5.0, 35.0)),
    (512.0, 50.0, (30.0, 60.0)),
])
def test_envelope(sr, f, band):
    phi0 = 2.0*np.pi*np.random.rand()
    t = time(sr)
    phi = np.mod(phase(t, f, phi0), 2*np.pi)
    x = np.sin(phi) + 0.05*np.random.randn(phi.size)
    signal = Signal(x, sr)
    sl = slice(int(0.1*phi.size), int(0.9*phi.size))
    envelope = signal.envelope(band)
    assert envelope[sl].mean() == pytest.approx(1.0, abs=1e-1)
    assert envelope[sl] == pytest.approx(envelope[sl].mean(), abs=1e-1)
