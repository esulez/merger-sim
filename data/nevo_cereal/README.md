# Nevo fake cereal data

Provenance: the semi-fabricated ready-to-eat cereal dataset from Aviv Nevo's Matlab code for Nevo (2000), as distributed with the [PyBLP](https://pyblp.readthedocs.io) package (`pyblp.data.NEVO_PRODUCTS_LOCATION` / `NEVO_AGENTS_LOCATION`), originally archived on Eric Rasmusen's website. Copied here so the examples run without PyBLP installed.

- `nevo_products.csv` — 2,256 rows = 94 markets (`market_ids`, city × quarter) × 24 products. Columns: `market_ids`, `city_ids`, `quarter`, `product_ids`, `firm_ids` (5 firms: 1, 2, 3, 4, 6), `brand_ids`, `shares`, `prices`, `sugar`, `mushy`, `demand_instruments0`–`demand_instruments19` (pre-computed BLP instruments).
- `nevo_agents.csv` — demographic draws (income, age, child) for random-coefficients estimation.

Note: there is no cost column, so the calibrated examples assume a constant Lerner margin (default 0.3).
