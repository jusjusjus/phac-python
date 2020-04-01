
import pytest
import numpy as np
from .metrics import (
    normalize,
    shannon_entropy,
    _modulation_index
)


def test_normalize_fails():
    with pytest.raises(ValueError):
        normalize(np.array([-1., 2]))

@pytest.mark.parametrize("x", (np.random.rand(1024),))
def test_normalize(x):
    assert sum(normalize(x)) == pytest.approx(1.0)


@pytest.mark.parametrize("P", [
    [-0.3, 0.5],
    [1.3, 0.5],
    [0.3, 0.5],
])
def test_shannon_entropy_fails(P):
    P = np.asarray(P)
    with pytest.raises(ValueError):
        shannon_entropy(P)

@pytest.mark.parametrize("x, expected", [
    ([1, 0, 0], 0.0), # entropy is null
    ([0, 0, 0, 1, 0, 0], 0.0), # entropy is null
    ([1, 1, 1], np.log(3)),
    ([1, 1, 1, 1, 1], np.log(5)),
])
def test_shannon_entropy(x, expected):
    x = np.asarray(x, dtype=np.float64)
    x = normalize(x)
    assert shannon_entropy(x) == pytest.approx(expected, abs=1e-8)


def test_modulation_index_fails():
    with pytest.raises(ValueError):
        _modulation_index(np.array([-1, 1, 1]))
