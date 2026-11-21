"""Generate the figures for docs/calibrated_merger_simulation.md into docs/figures/.

Run with: PYTHONPATH=python python python/examples/make_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mergersim.data import load_nevo_products, nevo_market
from mergersim.simulation import simulate_merger

FIG_DIR = Path(__file__).resolve().parents[2] / "docs" / "figures"


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_nevo_products()
    mid = df["market_ids"].iloc[0]
    m_logit = nevo_market(mid, df, lerner=0.3, demand_type="logit")
    m_lin = nevo_market(mid, df, lerner=0.3, demand_type="linear")

    i = int(np.argmax(m_logit.shares))
    grid = np.linspace(0.0, 3.0 * m_logit.prices[i], 200)

    s_logit = np.array([m_logit.demand.shares(_p(m_logit.prices, i, x))[i] for x in grid])
    s_lin = np.array([m_lin.demand.shares(_p(m_lin.prices, i, x))[i] for x in grid])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(grid, s_logit, label="logit")
    ax.plot(grid, s_lin, label="linear")
    ax.axhline(0.0, color="k", lw=0.6)
    ax.axvline(m_logit.prices[i], color="gray", ls="--", lw=0.8, label="observed price")
    ax.set_xlabel(f"price of product {m_logit.product_ids[i]} (other prices fixed)")
    ax.set_ylabel("market share")
    ax.set_title(f"Own-price demand curve, cereal market {mid}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_demand_forms.png", dpi=150)
    plt.close(fig)

    j = next(k for k in range(len(m_logit.product_ids)) if k != i and m_logit.firm_ids[k] != m_logit.firm_ids[i])
    d_logit = np.array([m_logit.demand.diversion_ratios(_p(m_logit.prices, i, x))[i, j] for x in grid])
    d_lin = np.array([m_lin.demand.diversion_ratios(_p(m_lin.prices, i, x))[i, j] for x in grid])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(grid, d_logit, label="logit")
    ax.plot(grid, d_lin, label="linear (constant)")
    ax.axvline(m_logit.prices[i], color="gray", ls="--", lw=0.8, label="observed price")
    ax.set_xlabel(f"price of product {m_logit.product_ids[i]}")
    ax.set_ylabel(f"diversion ratio D_{{{i},{j}}}")
    ax.set_title(f"Diversion ratio vs price, cereal market {mid}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_diversion_forms.png", dpi=150)
    plt.close(fig)

    r_logit = simulate_merger(m_logit, merging_firms=[1, 2])
    r_lin = simulate_merger(m_lin, merging_firms=[1, 2])
    x = np.arange(len(r_logit.product_ids))
    dp_logit = 100.0 * (r_logit.post["prices"] / r_logit.pre["prices"] - 1.0)
    dp_lin = 100.0 * (r_lin.post["prices"] / r_lin.pre["prices"] - 1.0)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - 0.2, dp_logit, width=0.4, label="logit")
    ax.bar(x + 0.2, dp_lin, width=0.4, label="linear")
    ax.set_xticks(x)
    ax.set_xticklabels(r_logit.product_ids, rotation=90, fontsize=6)
    ax.set_ylabel("post-merger price change (%)")
    ax.set_title(f"Merger of firms 1 and 2, cereal market {mid}: logit vs linear demand")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_cereal_logit_vs_linear.png", dpi=150)
    plt.close(fig)

    dp = r_logit.post["prices"] - r_logit.pre["prices"]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(r_logit.upp, dp)
    ax.axhline(0.0, color="k", lw=0.6)
    ax.axvline(0.0, color="k", lw=0.6)
    ax.set_xlabel("UPP_i (Farrell-Shapiro 2010 screen)")
    ax.set_ylabel("actual price change (full simulation)")
    ax.set_title(f"UPP vs simulated price effect, cereal market {mid}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_upp_vs_dp.png", dpi=150)
    plt.close(fig)

    print("figures saved to", FIG_DIR)


def _p(prices, i, x):
    p = prices.copy()
    p[i] = x
    return p


if __name__ == "__main__":
    main()
