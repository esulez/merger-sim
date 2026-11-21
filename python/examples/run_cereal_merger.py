"""Calibrated merger simulation on one market of Nevo's fake cereal data (Nevo 2000).

Run with:
PYTHONPATH=python python python/examples/run_cereal_merger.py [--market C01Q1] [--firms 1 2] [--demand logit|linear] [--lerner 0.3]
"""

import argparse

from mergersim.data import load_nevo_products, nevo_market
from mergersim.simulation import report, simulate_merger


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--market", default=None, help="market_ids value, e.g. C01Q1 (default: first)")
    ap.add_argument("--firms", type=int, nargs=2, default=[1, 2], help="merging firm ids")
    ap.add_argument("--demand", choices=["logit", "linear"], default="logit")
    ap.add_argument("--lerner", type=float, default=0.3, help="assumed constant Lerner margin (no cost column in data)")
    args = ap.parse_args()

    df = load_nevo_products()
    market_id = args.market or df["market_ids"].iloc[0]
    m = nevo_market(market_id, df, lerner=args.lerner, market_size=1.0, demand_type=args.demand)
    print(f"market {market_id}: {len(m.product_ids)} products, firms {sorted(set(m.firm_ids))}, demand={args.demand}")
    if args.demand == "logit":
        print(f"calibrated price coefficient a = {m.demand.a:.4f}")
    else:
        print(f"calibrated own slopes B_jj: mean {m.demand.B.diagonal().mean():.4f}")
    print(f"merging firms {args.firms} ...")

    res = simulate_merger(m, merging_firms=args.firms, efficiency=0.0)
    table, summary = report(res)
    print("\nper-product results:")
    print(table.round(4).to_string(index=False))
    print("\nsummary:")
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
