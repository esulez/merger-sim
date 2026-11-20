import numpy as np

from mergersim.demand import shares, price_jacobian, invert_shares
from mergersim.merger import diversion_ratio


def test_shares_sum_below_one_with_outside():
    delta = np.array([1.0, 0.5, -0.5])
    s = shares(delta)
    assert s.shape == (3,)
    assert np.sum(s) < 1.0


def test_price_jacobian_symmetry_cross_effects():
    s = np.array([0.5, 0.3, 0.2])
    alpha = 1.0
    G = price_jacobian(s, alpha)
    own = alpha * (s * (1 - s))
    assert np.allclose(np.diag(G), -own)
    assert G[0, 1] == alpha * s[0] * s[1]


def test_invert_shares_roundtrip():
    delta = np.array([1.0, -0.5, 2.0])
    s = shares(delta)
    recovered = invert_shares(s)
    assert np.allclose(recovered, delta)


def test_diversion_diagonal_zero_and_finite():
    s = np.array([0.4, 0.35, 0.25])
    D = diversion_ratio(s)
    assert np.allclose(np.diag(D), 0.0)
    assert np.all(np.isfinite(D))
    assert np.all(D >= 0.0)
