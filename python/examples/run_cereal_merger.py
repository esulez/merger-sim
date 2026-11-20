"""Calibrated merger simulation on one market of Nevo's fake cereal data (Nevo 2000).

Firms 1 and 2 (the two largest in the data) are merged; the cereal data has no cost
column, so a constant Lerner margin of 0.3 is assumed. Run with:
PYTHONPATH=python python python/examples/run_cereal_merger.py [market_id] [firmA firmB]
"""

import sys

from mergersim.data import load_nevo_products, nevo_market
from mergersim.simulation import report, simulate_merger


def main():
    df = load_nevo_products()
    market_id = sys.argv[1] if len(sys.argv) > 1 else df["market_ids"].iloc[0]
    merging = [int(x) for x in sys.argv[2:4]] if len(sys.argv) >= 4 else [1, 2]

    m = nevo_market(market_id, df, lerner=0.3, market_size=1.0)
    print(f"market {market_id}: {len(m.product_ids)} products, firms {sorted(set(m.firm_ids))}")
    print(f"calibrated price coefficient a = {m.demand.a:.4f}")
    print(f"merging firms {merging} ...")

    res = simulate_merger(m, merging_firms=merging, efficiency=0.0)
    table, summary = report(res)
    print("\nper-product results:")
    print(table.round(4).to_string(index=False))
    print("\nsummary:")
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
