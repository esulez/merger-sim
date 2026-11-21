# python/ — calibrated merger simulation in Python

Package `mergersim` implements the calibrated merger-simulation pipeline:

| module | contents |
|---|---|
| `mergersim/demand.py` | `LogitDemand` and `LinearDemand`: calibration from shares/prices/margins, shares, elasticities, diversion ratios, consumer surplus (linear also has a closed-form equilibrium price solver) |
| `mergersim/market.py` | `Market`: ownership matrix, HHI, Nash–Bertrand FOCs and equilibrium price solver |
| `mergersim/simulation.py` | `simulate_merger` (post-merger equilibrium), `upward_pricing_pressure` (Farrell–Shapiro 2010), `report` |
| `mergersim/data.py` | Nevo cereal loader (`data/nevo_cereal/`), synthetic-market generator |

## Install and run

```bash
pip install -r python/requirements.txt
PYTHONPATH=python python python/examples/run_synthetic.py
PYTHONPATH=python python python/examples/run_cereal_merger.py                          # market C01Q1, firms 1+2, logit
PYTHONPATH=python python python/examples/run_cereal_merger.py --demand linear         # same, linear demand
PYTHONPATH=python python python/examples/run_logit_vs_linear.py                       # compare both demand forms
PYTHONPATH=python python python/examples/make_figures.py                              # regenerate docs/figures/
PYTHONPATH=python python -m unittest discover -s python/tests -v
```

## Minimal usage

```python
from mergersim.data import nevo_market
from mergersim.simulation import simulate_merger, report

m = nevo_market("C01Q1", lerner=0.3, demand_type="logit")   # or "linear"
res = simulate_merger(m, merging_firms=[1, 2], efficiency=0.05)
table, summary = report(res)                   # per-product price/share changes + UPP; HHI/CS/profit
```

See `docs/calibrated_merger_simulation.md` for the full technical write-up (math, figures, code references).

Method: Hausman, Leonard & Zona (1994); Werden & Froeb (1994); Farrell & Shapiro (2010) — see `docs/literature.md`.
