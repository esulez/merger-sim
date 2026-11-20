import numpy as np

from mergersim.pricing import bertrand_equilibrium
from mergersim.merger import hhi, simulate_merger
from mergersim.data import generate_market


def test_merger_raises_prices():
    data = generate_market(n_products=6, seed=0)
    firms = data["firm_id"].to_numpy()
    mc = data["marginal_cost"].to_numpy()
    delta_np = data["delta_np"].to_numpy()

    own_pre = (firms[:, None] == firms[None, :]).astype(float)
    own_post = own_pre.copy()
    ia, ib = 0, 1
    merged_firms = (firms == ia) | (firms == ib)
    own_post[np.ix_(merged_firms, merged_firms)] = 1.0

    p_pre, s_pre = bertrand_equilibrium(mc, 1.0, delta_np, own_pre)
    p_post, s_post = bertrand_equilibrium(mc, 1.0, delta_np, own_post)
    x = float(np.max(p_post[merged_firms] - p_pre[merged_firms]))
    assert x > 0
    assert np.all(p_post[merged_firms] > mc[merged_firms])


def test_hhi_renormalised():
    assert hhi(np.array([0.5, 0.25, 0.25])) == 10000.0 * (0.25 + 0.0625 + 0.0625)


def test_merger_simulation_output_shapes():
    data = generate_market(n_products=6, seed=0)
    firms = data["firm_id"].to_numpy()
    mc = data["marginal_cost"].to_numpy()
    delta_np = data["delta_np"].to_numpy()
    result = simulate_merger(
        marginal_costs=mc,
        alpha=1.0,
        delta_nonprice=delta_np,
        owner_ids=firms,
        partner_a=firms[0],
        partner_b=firms[-1],
    )
    for key in ("prices_before", "prices_after", "delta_prices", "hhi_before", "hhi_after", "shares_after", "margins_post", "upp"):
        assert key in result
    merged = (firms == firms[0]) | (firms == firms[-1])
    assert np.all(result["delta_prices"][merged] >= -1e-8)
    assert result["hhi_before"] > 0 and result["hhi_after"] > 0


def test_diversion_diagonal_zero_and_finite():
    s = np.array([0.4, 0.35, 0.25])
    from mergersim.merger import diversion_ratio
    D = diversion_ratio(s)
    assert np.allclose(np.diag(D), 0.0)
    assert np.all(np.isfinite(D))
    assert np.all(D >= 0.0)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
