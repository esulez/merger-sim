"""Merger simulation and antitrust diagnostics on logit demand."""

import numpy as np

from .demand import ownership_matrix
from .pricing import bertrand_equilibrium


def hhi(shares_vec):
    """HHI in points from inside shares (renormalised over inside goods)."""
    s = np.asarray(shares_vec, dtype=float)
    s = s / s.sum()
    return 10000.0 * float(np.sum(s**2))


def diversion_ratio(s):
    """(J, J) diversion ratios D_ij = s_j / (1 - s_i); D_ii = 0."""
    s = np.asarray(s, dtype=float)
    D = np.outer(s, 1.0 / (1.0 - s))
    np.fill_diagonal(D, 0.0)
    return D


def upward_pricing_pressure(prices, shares_vec, margins, owner_ids, partner_a, partner_b):
    """Werden (1996) UPP for the merging parties' products.

    UPP_i = sum_j in partner_b D_ij * margin_j - margin_i for products of
    partner_a, and symmetrically for partner_b's products.
    """
    D = diversion_ratio(shares_vec)
    m = np.asarray(margins, dtype=float)
    agg = D @ m
    upp = np.full(len(prices), np.nan)
    mask_a = owner_ids == partner_a
    mask_b = owner_ids == partner_b
    upp[mask_a] = agg[mask_a]
    upp[mask_b] = agg[mask_b]
    return upp


def simulate_merger(marginal_costs, alpha, delta_nonprice, owner_ids, partner_a, partner_b):
    """Prices and shares before and after ``partner_b`` merges into ``partner_a``."""
    own_pre = ownership_matrix(owner_ids)
    own_post = own_pre.copy()
    ia = np.flatnonzero(owner_ids == partner_a)
    ib = np.flatnonzero(owner_ids == partner_b)
    own_post[np.ix_(ib, ia)] = 1.0
    own_post[np.ix_(ia, ib)] = 1.0

    p_pre, s_pre = bertrand_equilibrium(marginal_costs, alpha, delta_nonprice, own_pre)
    p_post, s_post = bertrand_equilibrium(marginal_costs, alpha, delta_nonprice, own_post)

    margins_pre = (p_pre - marginal_costs) / p_pre
    margins_post = (p_post - marginal_costs) / p_post
    return {
        "prices_before": p_pre,
        "prices_after": p_post,
        "delta_prices": p_post - p_pre,
        "shares_before": s_pre,
        "shares_after": s_post,
        "margins_before": margins_pre,
        "margins_post": margins_post,
        "hhi_before": hhi(s_pre),
        "hhi_after": hhi(s_post),
        "upp": upward_pricing_pressure(p_pre, s_pre, margins_pre, owner_ids, partner_a, partner_b),
    }
