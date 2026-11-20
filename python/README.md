# python/ — calibrated merger simulation in Python

Package `mergersim` implements the calibrated merger-simulation pipeline:

| module | contents |
|---|---|
| `mergersim/demand.py` | `LogitDemand`: calibration from shares/prices/margins, shares, elasticities, diversion ratios, consumer surplus |
| `mergersim/market.py` | `Market`: ownership matrix, HHI, Nash–Bertrand FOCs and equilibrium price solver |
| `mergersim/simulation.py` | `simulate_merger` (post-merger equilibrium), `upward_pricing_pressure` (Farrell–Shapiro 2010), `report` |
| `mergersim/data.py` | Nevo cereal loader (`data/nevo_cereal/`), synthetic-market generator |

## Install and run

```bash
pip install -r python/requirements.txt
PYTHONPATH=python python python/examples/run_synthetic.py
PYTHONPATH=python python python/examples/run_cereal_merger.py            # market C01Q1, firms 1+2
PYTHONPATH=python python python/examples/run_cereal_merger.py C05Q3 3 4   # custom market / firms
PYTHONPATH=python python -m unittest discover -s python/tests -v
```

## Minimal usage

```python
from mergersim.data import nevo_market
from mergersim.simulation import simulate_merger, report

m = nevo_market("C01Q1", lerner=0.3)          # calibrate logit from shares/prices/margins
res = simulate_merger(m, merging_firms=[1, 2], efficiency=0.05)
table, summary = report(res)                   # per-product price/share changes + UPP; HHI/CS/profit
```

Method: Hausman, Leonard & Zona (1994); Werden & Froeb (1994); Farrell & Shapiro (2010) — see `docs/literature.md`.
