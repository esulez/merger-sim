"""Download US airline data from BTS TranStats for airline merger simulations.

Usage:
    python data/fetch_bts.py 2019 1        # downloads DB1B_2019_1Q.zip into data/bts/
    python data/fetch_bts.py --t100 2019 1  # downloads T-100 segment data instead

Direct zip pattern: https://transtats.bts.gov/PREDIR/DB1B_{year}_{q}Q.zip
DB1B is the 10% sample of domestic airline tickets (origin, destination, carrier,
fare, ...); T-100 has segment-level passengers. Both are public domain (US Gov).
Large files (DB1B quarter ~ 200-400 MB zip) - run explicitly, not at install time.
"""

import sys
import urllib.request
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / "bts"


def fetch(year: int, quarter: int, table: str = "DB1B") -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    name = f"{table}_{year}_{quarter}Q.zip"
    url = f"https://transtats.bts.gov/PREDIR/{name}"
    dest = OUT_DIR / name
    print(f"downloading {url} -> {dest}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as f:
        f.write(r.read())
    print(f"saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


if __name__ == "__main__":
    args = sys.argv[1:]
    table = "DB1B"
    if args and args[0] == "--t100":
        table = "T100SEG" if len(args) == 1 else args[0]
        args = args[1:]
    year, quarter = int(args[0]), int(args[1])
    fetch(year, quarter, table=table)
