"""Single-market differentiated-products model with Nash-Bertrand pricing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import root

from .demand import LogitDemand


@dataclass
class Market:
    """One market: products, owning firms, observed prices/shares, calibrated logit demand, costs."""

    product_ids: list
    firm_ids: np.ndarray
    prices: np.ndarray
    shares: np.ndarray
    demand: LogitDemand
    costs: np.ndarray

    @classmethod
    def calibrated(cls, product_ids, firm_ids, prices, shares, margins=None, lerner=None, market_size=1.0):
        """Build a market by calibrating logit demand from observed data; costs = prices - margins."""
        prices = np.asarray(prices, dtype=float)
        shares = np.asarray(shares, dtype=float)
        demand = LogitDemand.calibrate(shares, prices, margins=margins, lerner=lerner, market_size=market_size)
        if margins is None:
            margins = float(lerner) * prices
        costs = prices - np.asarray(margins, dtype=float)
        return cls(list(product_ids), np.asarray(firm_ids), prices, shares, demand, costs)

    def ownership(self, firm_ids=None):
        """O[i, j] = 1 if products i and j are owned by the same firm."""
        f = self.firm_ids if firm_ids is None else np.asarray(firm_ids)
        return (f[:, None] == f[None, :]).astype(float)

    def firm_shares(self, shares=None, firm_ids=None):
        s = self.shares if shares is None else np.asarray(shares, dtype=float)
        f = self.firm_ids if firm_ids is None else np.asarray(firm_ids)
        return np.array([s[f == k].sum() for k in np.unique(f)])

    def hhi(self, shares=None, firm_ids=None):
        fs = self.firm_shares(shares, firm_ids)
        return float(10000.0 * np.sum(fs ** 2))

    def foc(self, prices, firm_ids=None):
        """Pricing FOC residuals: r_i = s_i + sum_k O[i,k] (p_k - c_k) ds_k/dp_i = 0 at equilibrium."""
        s = self.demand.shares(prices)
        A = self.demand.jacobian(prices)
        O = self.ownership(firm_ids)
        margins = np.asarray(prices, dtype=float) - self.costs
        weighted = A * margins[None, :]
        return s + (O * weighted.T).sum(axis=1)

    def solve_prices(self, firm_ids=None, p0=None):
        """Solve the Nash-Bertrand equilibrium price vector for the given ownership.

        Primary solver is the logit markup fixed-point iteration
        (Morrow-Skerlos 2011 style): m_i = (1/a + sum_{j in firm(i), j != i} m_j s_j) / (1 - s_i);
        scipy root-finding is used as a fallback/refinement.
        """
        p0 = np.asarray(self.prices if p0 is None else p0, dtype=float)
        p = self._fixed_point(firm_ids, p0)
        best_res = float(np.max(np.abs(self.foc(p, firm_ids))))
        if best_res > 1e-9:
            for method in ("hybr", "lm"):
                sol = root(self.foc, p, args=(firm_ids,), method=method)
                res = float(np.max(np.abs(self.foc(sol.x, firm_ids))))
                if res < best_res:
                    p, best_res = sol.x, res
                if best_res < 1e-10:
                    break
        if best_res > 1e-7:
            raise RuntimeError(f"pricing equilibrium did not converge (max FOC residual {best_res:.2e})")
        return p

    def _fixed_point(self, firm_ids, p0, max_iter=5000, tol=1e-12):
        p = p0.astype(float).copy()
        O = self.ownership(firm_ids)
        a = self.demand.a
        for _ in range(max_iter):
            s = self.demand.shares(p)
            m = p - self.costs
            cross = (O * (m * s)[None, :]).sum(axis=1) - m * s
            p_new = self.costs + (1.0 / a + cross) / (1.0 - s)
            if np.max(np.abs(p_new - p)) < tol:
                return p_new
            p = p_new
        return p

    def profits(self, prices, firm_ids=None):
        """Per-firm operating profits: market_size * sum_{j in f} (p_j - c_j) s_j."""
        s = self.demand.shares(prices)
        margins = np.asarray(prices, dtype=float) - self.costs
        f = self.firm_ids if firm_ids is None else np.asarray(firm_ids)
        M = self.demand.market_size
        return np.array([M * float(np.sum(margins[f == k] * s[f == k])) for k in np.unique(f)])


def equilibrium(demand: LogitDemand, firm_ids, costs, p0=None) -> Market:
    """Solve the pre-merger pricing equilibrium for given demand, ownership and costs."""
    firm_ids = np.asarray(firm_ids)
    costs = np.asarray(costs, dtype=float)
    p0 = costs + 1.0 if p0 is None else np.asarray(p0, dtype=float)
    m = Market(list(range(firm_ids.size)), firm_ids, p0, demand.shares(p0), demand, costs)
    p = m.solve_prices(p0=p0)
    return Market(m.product_ids, firm_ids, p, demand.shares(p), demand, costs)
