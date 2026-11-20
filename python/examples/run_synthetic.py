"""Calibrate a synthetic logit market, then simulate a merger.

The synthetic market is generated from a known logit DGP whose observed prices are
the exact pre-merger Nash-Bertrand equilibrium, so calibration should recover the
true demand parameter. Run with: PYTHONPATH=python python python/examples/run_synthetic.py
"""

from mergersim.data import synthetic_market
from mergersim.market import Market
from mergersim.simulation import report, simulate_merger


def main():
    m = synthetic_market(n_products=8, a_true=2.0, seed=42, market_size=1000.0)
    cal = Market.calibrated(
        m.product_ids,
        m.firm_ids,
        m.prices,
        m.shares,
        margins=m.prices - m.costs,
        market_size=m.demand.market_size,
    )
    print(f"true a = {m.demand.a:.6f} | calibrated a = {cal.demand.a:.6f}")
    print(f"max |FOC residual| at observed prices: {abs(cal.foc(cal.prices)).max():.2e}")

    res = simulate_merger(cal, merging_firms=[1, 2], efficiency=0.0)
    table, summary = report(res)
    print("\nper-product results (firms 1 and 2 merge):")
    print(table.round(4).to_string(index=False))
    print("\nsummary:")
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
