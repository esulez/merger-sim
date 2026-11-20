"""Synthetic market generator for examples and tests."""

import numpy as np
import pandas as pd

from .pricing import bertrand_equilibrium


def generate_market(n_products=6, alpha=1.0, seed=None):
    """Draw a random differentiated-products market.

    Returns a DataFrame with firm ids, product ids, characteristics x,
    non-price mean utility delta_np, and Bertrand-Nash prices/shares.
    """
    rng = np.random.default_rng(seed)
    mc = np.abs(rng.normal(loc=1.0, scale=0.25, size=n_products))
    delta_np = 2.0 + rng.normal(loc=0.0, scale=0.5, size=n_products)
    firms = rng.integers(0, 3, size=n_products)

    own = (firms[:, None] == firms[None, :]).astype(float)
    prices, sh = bertrand_equilibrium(mc, alpha, delta_np, own)

    return pd.DataFrame(
        {
            "firm_id": firms,
            "product_id": np.arange(n_products),
            "x": rng.normal(size=n_products),
            "marginal_cost": mc,
            "delta_np": delta_np,
            "alpha": alpha,
            "price": prices,
            "share": sh,
        }
    )
