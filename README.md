# merger-sim

Calibrated merger simulation code. Python implementation first; R and Stata ports planned (see `r/`, `stata/`).

## What it does

Implements the standard calibrated merger-simulation recipe (Hausman, Leonard & Zona 1994; Werden & Froeb 1994; Farrell & Shapiro 2010):

1. **Calibrate demand** for the products in a market (multinomial logit or linear) from pre-merger shares, prices and margins.
2. **Impute marginal costs** from the pre-merger Nash–Bertrand first-order conditions.
3. **Re-solve the pricing game** under the post-merger ownership structure and compare equilibria: prices, shares, HHI, diversion ratios, upward pricing pressure (UPP), profits and consumer surplus.

## Layout

- `docs/literature.md` — annotated review of the merger-simulation literature
- `docs/calibrated_merger_simulation.md` — technical paper: how the simulation works, logit vs linear demand, with math, figures and code references
- `papers/` — open-access PDFs of the key papers selected for this project
- `data/` — datasets (Nevo cereal data bundled in `data/nevo_cereal/`; BTS airline downloader `data/fetch_bts.py`)
- `python/` — the Python implementation: `mergersim` package + examples + tests
- `r/`, `stata/` — planned ports of the same pipeline

## Quickstart (Python)

```bash
pip install -r python/requirements.txt
PYTHONPATH=python python python/examples/run_synthetic.py      # synthetic market, recovers the DGP
PYTHONPATH=python python python/examples/run_cereal_merger.py  # Nevo cereal market merger
PYTHONPATH=python python -m unittest discover -s python/tests -v
```

See `docs/literature.md` for the papers behind the method.
