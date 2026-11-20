"""End-to-end example: calibrated logit-Bertrand merger simulation."""

import numpy as np

from mergersim.data import generate_market
from mergersim.merger import simulate_merger

data = generate_market(n_products=6, alpha=1.0, seed=0)
firms = data["firm_id"].to_numpy()

result = simulate_merger(
    marginal_costs=data["marginal_cost"].to_numpy(),
    alpha=float(data["alpha"].iloc[0]),
    delta_nonprice=data["delta_np"].to_numpy(),
    owner_ids=firms,
    partner_a=firms[0],
    partner_b=firms[-1],
)

merged = (firms == firms[0]) | (firms == firms[-1])
print("Product-level results (merging products flagged with *):")
print(f"{'firm':>4} {'merge':>5} {'p0':>7} {'p1':>7} {'dp %':>7}")
for i in range(len(data)):
    flag = "*" if merged[i] else ""
    dpct = 100.0 * result["delta_prices"][i] / result["prices_before"][i]
    print(f"{firms[i]:>4} {flag:>5} {result['prices_before'][i]:>7.3f} "
          f"{result['prices_after'][i]:>7.3f} {dpct:>6.2f}%")

print(f"\nHHI (inside goods): {result['hhi_before']:.0f} -> {result['hhi_after']:.0f}")
upp = result["upp"]
print("UPP on merging parties' products:", np.round(upp[merged], 3))
