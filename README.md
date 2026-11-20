# merger-sim

Calibrated merger simulation in Python (R and Stata versions to follow).

The workhorse implemented here is differentiated-products **Bertrand-Nash
pricing with logit demand** (Werden & Froeb 1994), plus the diagnostics
used in practice: HHI, diversion ratios and upward pricing pressure
(Werden 1996 / Farrell-Shapiro 2010).

## Layout

```
src/mergersim/
  demand.py      logit shares, exact price Jacobian, share inversion, ownership matrix
  pricing.py     Bertrand-Nash equilibrium (damped fixed point on the FOCs)
  merger.py      merger simulation, HHI, diversion ratios, UPP
  data.py        synthetic market generator
docs/literature_review.md   core papers and what each contributes
examples/example_logit_merger.py  end-to-end demo
tests/          pytest suite
```

## Installation

```bash
pip install -e .
python -m pytest tests/                # run the test suite
python examples/example_logit_merger.py
```

## API

```python
from mergersim import generate_market, simulate_merger

data = generate_market(n_products=6, alpha=1.0, seed=0)
result = simulate_merger(
    marginal_costs=data["marginal_cost"].to_numpy(),
    alpha=1.0,
    delta_nonprice=data["delta_np"].to_numpy(),
    owner_ids=data["firm_id"].to_numpy(),
    partner_a=firms[0], partner_b=firms[-1],
)
result["delta_prices"]   # per-product price changes
result["hhi_after"]      # HHI after the merger
```

## Data

The simulation is calibrated from three inputs: observed prices, observed
shares (with an outside good), and the price coefficient alpha. For real
applications, any dataset at market-product level with price, share/quantity,
ownership and characteristics works; the Nevo (2000) cereal data is the
canonical public example.
