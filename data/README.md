# data/

Datasets for the merger simulations.

- `nevo_cereal/` — Nevo's "fake" ready-to-eat cereal data (Nevo 2000), as distributed with PyBLP. Bundled here so the Python examples run offline. See `nevo_cereal/README.md` for provenance and columns.
- `bts/` — (empty) landing directory for US airline data. `fetch_bts.py` downloads the BTS DB1B 10% ticket sample (and can be pointed at T-100 segment data) from TranStats; run it explicitly, the files are large.
- `fetch_bts.py` — downloader for BTS airline data (not run by default).

Planned additions: BLP car data (available via `pyblp.data`), Ivaldi–Verboven European car data.
