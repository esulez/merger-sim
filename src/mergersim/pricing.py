"""Bertrand-Nash pricing with differentiated products and logit demand."""

import warnings

import numpy as np

from .demand import price_jacobian, shares


def bertrand_equilibrium(marginal_costs, alpha, delta_nonprice, ownership, tol=1e-10, max_iter=2000, damping=0.5):
    """Solve for Bertrand-Nash prices via damped fixed-point iteration.

    Iterates p = c - (G^T O)^{-1} s, where G is the price Jacobian and
    O the binary ownership matrix; the update is under-relaxed by
    ``damping`` to guarantee convergence for well-behaved demand systems.
    """
    mc = np.asarray(marginal_costs, dtype=float)
    own = np.asarray(ownership, dtype=float)
    delta_np = np.asarray(delta_nonprice, dtype=float)
    prices = mc.copy()
    for _ in range(int(max_iter)):
        s = shares(delta_np - alpha * prices)
        G = price_jacobian(s, alpha)
        A = G.T * own
        try:
            step = np.linalg.solve(A, s)
        except np.linalg.LinAlgError:
            step, *_ = np.linalg.lstsq(A, s, rcond=None)
        prices_new = mc - step
        change = np.max(np.abs(prices_new - prices))
        prices = prices + damping * (prices_new - prices)
        if change < tol:
            break
    else:
        warnings.warn(
            f"Bertrand fixed point did not converge (last change {change:.2e})",
            RuntimeWarning,
        )
    return prices, shares(delta_np - alpha * prices)
