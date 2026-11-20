"""Merger simulation: post-merger Nash-Bertrand equilibrium, UPP and welfare.

UPP follows Farrell & Shapiro (2010): for each product i of the merging firms,
    UPP_i = sum_{j in M, j != i} D_ij (p_j - c_j) - E_i c_i,
where D_ij is the diversion ratio from i to j (logit: s_j / (1 - s_i)) and E_i is
the proportional marginal-cost saving ("efficiency credit") on product i.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .market import Market


@dataclass
class MergerResult:
    merging_firms: list
    pre: dict
    post: dict
    upp: np.ndarray
    diversion: np.ndarray
    product_ids: list
    firm_ids_pre: np.ndarray
    firm_ids_post: np.ndarray


def _snapshot(market: Market, prices, firm_ids) -> dict:
    s = market.demand.shares(prices)
    M = market.demand.market_size
    return {
        "prices": np.asarray(prices, dtype=float),
        "shares": s,
        "quantities": M * s,
        "margins": np.asarray(prices, dtype=float) - market.costs,
        "profits": market.profits(prices, firm_ids),
        "hhi": market.hhi(s, firm_ids),
        "cs": M * market.demand.consumer_surplus(prices),
    }


def upward_pricing_pressure(market: Market, merging_firms, efficiency: float = 0.0) -> np.ndarray:
    """Net UPP for each product owned by the merging firms (Farrell-Shapiro 2010)."""
    p = market.prices
    margins = p - market.costs
    D = market.demand.diversion_ratios(p)
    in_m = np.isin(market.firm_ids, merging_firms)
    upp = np.zeros(p.size)
    for i in np.flatnonzero(in_m):
        partners = in_m & (np.arange(p.size) != i)
        upp[i] = float(D[i, partners] @ margins[partners] - efficiency * market.costs[i])
    return upp


def simulate_merger(market: Market, merging_firms, efficiency: float = 0.0) -> MergerResult:
    """Simulate a merger between ``merging_firms`` with proportional cost savings ``efficiency``.

    The merged firm re-optimizes all its products jointly (new ownership matrix);
    rivals keep their own products. Returns a MergerResult with pre/post snapshots.
    """
    merging_firms = list(merging_firms)
    firms_post = market.firm_ids.copy()
    firms_post[np.isin(firms_post, merging_firms)] = int(min(merging_firms))
    costs_post = market.costs.copy()
    in_m = np.isin(market.firm_ids, merging_firms)
    costs_post[in_m] = costs_post[in_m] * (1.0 - float(efficiency))
    post = Market(market.product_ids, firms_post, market.prices, market.shares, market.demand, costs_post)
    p_post = post.solve_prices(firm_ids=firms_post, p0=market.prices)
    pre = _snapshot(market, market.prices, market.firm_ids)
    post_snap = _snapshot(post, p_post, firms_post)
    return MergerResult(
        merging_firms=merging_firms,
        pre=pre,
        post=post_snap,
        upp=upward_pricing_pressure(market, merging_firms, efficiency),
        diversion=market.demand.diversion_ratios(market.prices),
        product_ids=list(market.product_ids),
        firm_ids_pre=market.firm_ids.copy(),
        firm_ids_post=firms_post,
    )


def report(result: MergerResult):
    """Return (per-product table, summary table) as pandas DataFrames."""
    df = pd.DataFrame(
        {
            "product": result.product_ids,
            "firm_pre": result.firm_ids_pre,
            "firm_post": result.firm_ids_post,
            "price_pre": result.pre["prices"],
            "price_post": result.post["prices"],
            "dprice_pct": 100.0 * (result.post["prices"] / result.pre["prices"] - 1.0),
            "share_pre": result.pre["shares"],
            "share_post": result.post["shares"],
            "upp": result.upp,
        }
    )
    summary = pd.DataFrame(
        {
            "HHI": [result.pre["hhi"], result.post["hhi"]],
            "consumer_surplus": [result.pre["cs"], result.post["cs"]],
            "total_profit": [result.pre["profits"].sum(), result.post["profits"].sum()],
        },
        index=["pre-merger", "post-merger"],
    )
    return df, summary
