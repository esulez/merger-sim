"""Data loaders: Nevo fake cereal data (bundled in data/nevo_cereal) and synthetic markets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .demand import LinearDemand, LogitDemand
from .market import Market, equilibrium

REPO_ROOT = Path(__file__).resolve().parents[2]
NEVO_PRODUCTS = REPO_ROOT / "data" / "nevo_cereal" / "nevo_products.csv"


def load_nevo_products(path=None) -> pd.DataFrame:
    """Load the Nevo cereal product data; falls back to PyBLP's bundled copy if missing."""
    path = Path(path) if path is not None else NEVO_PRODUCTS
    if not path.exists():
        import pyblp

        path = Path(pyblp.data.NEVO_PRODUCTS_LOCATION)
    return pd.read_csv(path)


def nevo_market(market_id, df=None, lerner: float = 0.3, market_size: float = 1.0,
                demand_type: str = "logit", diversion=None) -> Market:
    """Build a calibrated Market for one cereal market.

    The cereal data has no cost column, so a constant Lerner margin ``lerner`` is
    assumed (default 0.3). ``market_size`` only scales quantities/welfare (1.0 =
    per-capita). ``demand_type`` is "logit" or "linear".
    """
    df = load_nevo_products() if df is None else df
    m = df[df["market_ids"] == market_id].sort_values("product_ids")
    if m.empty:
        raise ValueError(f"unknown market_id {market_id!r}")
    return Market.calibrated(
        product_ids=m["product_ids"].tolist(),
        firm_ids=m["firm_ids"].to_numpy(),
        prices=m["prices"].to_numpy(),
        shares=m["shares"].to_numpy(),
        lerner=lerner,
        market_size=market_size,
        demand_type=demand_type,
        diversion=diversion,
    )


def synthetic_market(n_products: int = 8, n_firms: int | None = None, a_true: float = 2.0,
                     seed: int = 0, market_size: float = 1000.0, min_share: float = 0.02) -> Market:
    """Generate a synthetic logit market whose observed prices are the exact pre-merger equilibrium.

    Useful for testing: calibration should recover ``a_true`` exactly when firms are
    single-product (default: n_firms = n_products). Draws are rejected until every
    product has share > ``min_share`` and no price is degenerate, so the market is
    well behaved for calibration/merger tests.
    """
    if n_firms is None:
        n_firms = n_products
    if n_firms > n_products or n_products % n_firms != 0:
        raise ValueError("need 1 <= n_firms <= n_products and n_products divisible by n_firms")
    firm_ids = np.repeat(np.arange(1, n_firms + 1), n_products // n_firms)
    s = seed
    while True:
        rng = np.random.default_rng(s)
        intercepts = rng.normal(1.0, 1.0, n_products)
        costs = rng.uniform(0.2, 0.8, n_products)
        demand = LogitDemand(a_true, intercepts, market_size=market_size)
        try:
            m = equilibrium(demand, firm_ids, costs)
        except RuntimeError:
            s += 1
            continue
        if m.shares.min() > min_share and m.prices.max() < 20.0:
            return m
        s += 1


def synthetic_linear_market(n_products: int = 8, n_firms: int | None = None, own_slope: float = 2.0,
                            cross_slope: float = 0.5, seed: int = 0, market_size: float = 1000.0,
                            min_share: float = 0.02) -> Market:
    """Synthetic linear-demand market (s = a - B p, B_jj = own_slope, B_ij = -cross_slope)
    whose observed prices are the exact pre-merger equilibrium. Rejection-samples draws
    until all shares exceed ``min_share``."""
    if n_firms is None:
        n_firms = n_products
    if n_firms > n_products or n_products % n_firms != 0:
        raise ValueError("need 1 <= n_firms <= n_products and n_products divisible by n_firms")
    firm_ids = np.repeat(np.arange(1, n_firms + 1), n_products // n_firms)
    J = n_products
    B_true = own_slope * np.eye(J) - cross_slope * (np.ones((J, J)) - np.eye(J))
    s = seed
    while True:
        rng = np.random.default_rng(s)
        intercepts = rng.uniform(0.5, 1.5, J)
        costs = rng.uniform(0.1, 0.5, J)
        demand = LinearDemand(intercepts, B_true, market_size=market_size)
        try:
            m = equilibrium(demand, firm_ids, costs)
        except RuntimeError:
            s += 1
            continue
        if m.shares.min() > min_share:
            return m
        s += 1
