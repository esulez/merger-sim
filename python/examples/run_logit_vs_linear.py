"""Run the same merger under logit and linear demand on one cereal market and compare.

Run with: PYTHONPATH=python python python/examples/run_logit_vs_linear.py [market_id] [firmA firmB]
"""

import sys

import numpy as np

from mergersim.data import load_nevo_products, nevo_market
from mergersim.simulation import report, simulate_merger


def main():
    df = load_nevo_products()
    market_id = sys.argv[1] if len(sys.argv) > 1 else df["market_ids"].iloc[0]
    merging = [int(x) for x in sys.argv[2:4]] if len(sys.argv) >= 4 else [1, 2]

    results = {}
    for demand_type in ("logit", "linear"):
        m = nevo_market(market_id, df, lerner=0.3, demand_type=demand_type)
        res = simulate_merger(m, merging_firms=merging, efficiency=0.0)
        results[demand_type] = (m, res)
        in_m = np.isin(m.firm_ids, merging)
        dp = 100.0 * (res.post["prices"][in_m] / res.pre["prices"][in_m] - 1.0)
        print(f"[{demand_type:6s}] market {market_id}, merging firms {merging}")
        print(f"  mean price change of merging products: {dp.mean():+.2f}%  (min {dp.min():+.2f}%, max {dp.max():+.2f}%)")
        print(f"  HHI {res.pre['hhi']:.0f} -> {res.post['hhi']:.0f} | "
              f"CS {res.pre['cs']:.5f} -> {res.post['cs']:.5f} | "
              f"profit {res.pre['profits'].sum():.5f} -> {res.post['profits'].sum():.5f} | "
              f"max UPP {res.upp.max():.4f}")

    m_l, r_l = results["logit"]
    m_n, r_n = results["linear"]
    t_l, _ = report(r_l)
    t_n, _ = report(r_n)
    cmp = t_l[["product", "firm_pre", "dprice_pct"]].rename(columns={"dprice_pct": "dprice_logit"})
    cmp["dprice_linear"] = t_n["dprice_pct"].to_numpy()
    print("\nper-product price change (%):")
    print(cmp.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
