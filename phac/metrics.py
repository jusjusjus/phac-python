import numpy as np

from .util import indices_of_binned_phase

_SMALL = 1e-9


def normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    try:
        assert np.all(x >= 0.0), "values must be positive semi-definite."
        s = x.sum()
        assert s > 0.0, "sum must be positive."
        return x / s
    except AssertionError as err:
        raise ValueError(str(err))


def shannon_entropy(P: np.ndarray) -> float:
    try:
        assert np.all(P >= 0.0), "probability must be positive semi-definite."
        assert np.all(P <= 1.0), "probability must be smaller than 1."
        assert np.abs(P.sum() - 1) < _SMALL, "sum must be exactly 1."
    except AssertionError as err:
        raise ValueError(str(err))
    P = P.astype(np.float64)
    return -np.sum(P*np.log(P+_SMALL))


def _modulation_index(average_amplitudes: np.ndarray) -> float:
    """return modulation index

    The modulation index is defined as the normalized KL distance between
    normalized average amplitudes and the uniform distribution (Tort et al,
    2010):  average amplitudes are normalized to be probability like.  The
    distance between the distribution an a uniform distribution is computed.
    The result is divided by the maximal theoretical value $\\log N$ for N
    bins.

    $\\text{mi} = \\frac{\\log(N)-H(P)}{\\log(N)} =
    1 + \\frac{1}{\\log(N)} \\sum p_i\\log(p_i)$

    Parameters
    ----------
    average_amplitudes: np.ndarray
        array of amplitude averages binned conditional on phase sections.
    """
    average_amplitudes = average_amplitudes.astype(np.float64)
    try:
        assert np.all(average_amplitudes > 0), \
                "Envelope-derived amplitudes must be positive."
    except AssertionError as err:
        raise ValueError(str(err))
    # normalize to something probability-like
    P = normalize(average_amplitudes)
    # computed KL distance: log(N)-H(P), and normalize with log(N)
    return 1.0 - shannon_entropy(P) / np.log(P.size)


def modulation_index(phase: np.ndarray, amplitude: np.ndarray) -> float:
    """return modulation index

    The modulation index is defined as the normalized KL distance between
    normalized average amplitudes and the uniform distribution (Tort et al,
    2010):  average amplitudes are normalized to be probability like.  The
    distance between the distribution an a uniform distribution is computed.
    The result is divided by the maximal theoretical value $\\log N$ for N
    bins.

    $\\text{mi}=\\frac{\\log(N)-H(P)}{\\log(N)} =
    1 + \\frac{1}{\\log(N)}\\sum p_i\\log(p_i)$

    Parameters
    ----------
    phase: np.ndarray
        array of phase values
    amplitude: np.ndarray
        array of amplitude values
    """
    indices = indices_of_binned_phase(phase, num_bins=12)
    avg_amps = np.array([np.median(amplitude[idx]) for idx in indices],
                        dtype=np.float64)
    return _modulation_index(avg_amps)


def mean_phase_coherence(phase: np.ndarray, amplitude: np.ndarray) -> float:
    """return mean phase coherence

    $C = \\sum_j a_j \\exp{i\\phi_j}$

    Parameters
    ----------
    phase: np.ndarray
        array of phase values
    amplitude: np.ndarray
        array of amplitude values
    """
    z = amplitude * np.exp(1.0j*phase)
    return np.abs(np.mean(z))
