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


class LinearDemand:
    """Linear demand over inside goods only (no outside good), per capita: s(p) = a - B p.

    ``B[j, i] = -ds_j/dp_i``: positive diagonal, non-positive off-diagonal for substitutes.
    Quantities are market_size * s. Calibration: with single-product firms the pre-merger
    FOC gives B_jj = q_j / m_j; cross slopes are NOT identified from one market's
    (shares, prices, margins) and require diversion ratios, since D_ij = -B_ij / B_jj,
    i.e. B_ij = -D_ij * B_jj. By default D is proxied by the share-proportional (logit-style)
    ratio s_j / (1 - s_i); pass ``diversion`` to use estimated diversion ratios.

    Caveats vs logit: shares can go negative at high prices, shares of inside goods sum to
    one (no outside good), and diversion ratios are constant in prices.
    """

    def __init__(self, intercepts, B, market_size=1.0):
        self.intercepts = np.asarray(intercepts, dtype=float)
        self.B = np.asarray(B, dtype=float)
        self.market_size = float(market_size)

    @classmethod
    def calibrate(cls, shares, prices, margins, market_size=1.0, diversion=None):
        s = np.asarray(shares, dtype=float)
        p = np.asarray(prices, dtype=float)
        m = np.asarray(margins, dtype=float)
        if np.any(m <= 0):
            raise ValueError("margins must be positive")
        own = s / m  # per-capita FOC: s_j = m_j * B_jj
        if diversion is None:
            D = s[None, :] / (1.0 - s[:, None])
            np.fill_diagonal(D, 0.0)
        else:
            D = np.asarray(diversion, dtype=float)
        J = p.size
        B = np.zeros((J, J))
        for i in range(J):
            for j in range(J):
                B[i, j] = own[i] if i == j else -D[i, j] * own[j]
        a = s + B @ p
        return cls(a, B, market_size=float(market_size))

    def shares(self, prices):
        return self.intercepts - self.B @ np.asarray(prices, dtype=float)

    def outside_share(self, prices):
        return 0.0

    def quantities(self, prices):
        return self.market_size * self.shares(prices)

    def jacobian(self, prices):
        """A[j, i] = ds_j / dp_i = -B[j, i] (constant in prices)."""
        return -self.B

    def elasticities(self, prices):
        """E[j, i] = (p_i / s_j) * ds_j / dp_i."""
        s = self.shares(prices)
        p = np.asarray(prices, dtype=float)
        return (-self.B) * p[None, :] / s[:, None]

    def diversion_ratios(self, prices=None):
        """D[i, j] = -B[i, j] / B[j, j] for i != j (constant in prices for linear demand)."""
        D = -self.B / np.diag(self.B)[None, :]
        np.fill_diagonal(D, 0.0)
        return D

    def consumer_surplus(self, prices):
        """Per-capita CS of the quasi-linear utility behind s = a - B p (exact quadratic form)."""
        p = np.asarray(prices, dtype=float)
        s = self.shares(p)
        Binv = np.linalg.inv(self.B)
        alpha = Binv @ self.intercepts
        return float(alpha @ s - 0.5 * s @ Binv @ s - p @ s)

    def closed_form_prices(self, costs, ownership):
        """Exact Nash-Bertrand prices: p = (B + B^T O)^-1 (a + B^T O c)."""
        c = np.asarray(costs, dtype=float)
        O = np.asarray(ownership, dtype=float)
        return np.linalg.solve(self.B + self.B.T @ O, self.intercepts + self.B.T @ O @ c)
