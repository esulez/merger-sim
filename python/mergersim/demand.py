"""Multinomial logit demand for calibrated merger simulation.

Model: delta_j = intercept_j - a * p_j for products j = 1..J, delta_0 = 0 for the
outside good, so s_j = exp(delta_j) / (1 + sum_k exp(delta_k)). ``a`` is the
absolute value of the price coefficient (a = -alpha > 0).

Calibration (Hausman-Leonard-Zona 1994; Werden-Froeb 1994): given observed shares,
prices and margins (p_j - c_j), logit + Nash-Bertrand implies
(p_j - c_j) = 1 / (a (1 - s_j)), so ``a`` is recovered product by product and
averaged (it is constant in the model), and intercepts follow from inverting the
share equation, delta_j = ln s_j - ln s_0.
"""

from __future__ import annotations

import numpy as np


class LogitDemand:
    """Calibrated multinomial logit demand for a single market."""

    def __init__(self, a: float, intercepts, market_size: float = 1.0):
        self.a = float(a)
        self.intercepts = np.asarray(intercepts, dtype=float)
        self.market_size = float(market_size)

    @classmethod
    def calibrate(cls, shares, prices, margins=None, lerner=None, market_size: float = 1.0):
        """Calibrate from observed shares, prices and margins.

        ``margins`` is p_j - c_j per product; alternatively pass a constant Lerner
        margin ``lerner`` (margins = lerner * prices).
        """
        shares = np.asarray(shares, dtype=float)
        prices = np.asarray(prices, dtype=float)
        outside = 1.0 - shares.sum()
        if outside <= 0:
            raise ValueError("shares must leave a positive outside-good share")
        if margins is None:
            if lerner is None:
                raise ValueError("provide either margins or a constant lerner margin")
            margins = float(lerner) * prices
        margins = np.asarray(margins, dtype=float)
        if np.any(margins <= 0):
            raise ValueError("margins must be positive")
        delta = np.log(shares) - np.log(outside)
        a_per_product = 1.0 / (margins * (1.0 - shares))
        a = float(np.mean(a_per_product))
        intercepts = delta + a * prices
        return cls(a, intercepts, market_size=market_size)

    def deltas(self, prices):
        return self.intercepts - self.a * np.asarray(prices, dtype=float)

    def shares(self, prices):
        d = self.deltas(prices)
        e = np.exp(d - d.max())
        return e / (1.0 + e.sum())

    def outside_share(self, prices):
        return 1.0 - self.shares(prices).sum()

    def quantities(self, prices):
        return self.market_size * self.shares(prices)

    def jacobian(self, prices):
        """A[j, i] = ds_j / dp_i = a * s_j * (s_i - 1[j == i])."""
        s = self.shares(prices)
        return self.a * s[:, None] * (s[None, :] - np.eye(s.size))

    def elasticities(self, prices):
        """E[j, i] = (p_i / s_j) * ds_j / dp_i."""
        s = self.shares(prices)
        p = np.asarray(prices, dtype=float)
        return self.jacobian(prices) * p[None, :] / s[:, None]

    def diversion_ratios(self, prices):
        """D[i, j] = fraction of i's lost sales diverted to j (i != j); logit: s_j / (1 - s_i)."""
        s = self.shares(prices)
        d = s[None, :] / (1.0 - s[:, None])
        np.fill_diagonal(d, 0.0)
        return d

    def consumer_surplus(self, prices):
        """Per-capita CS = (1/a) * ln(1 + sum_j exp(delta_j)); multiply by market_size for totals."""
        d = self.deltas(prices)
        m = max(0.0, float(d.max()))
        lse = m + np.log(np.exp(-m) + np.exp(d - m).sum())
        return lse / self.a
