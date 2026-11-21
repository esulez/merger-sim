"""Tests for linear-demand calibration and merger simulation. Run: PYTHONPATH=python python -m unittest discover -s python/tests -v"""

import unittest

import numpy as np

from mergersim.data import load_nevo_products, nevo_market, synthetic_linear_market
from mergersim.market import Market
from mergersim.simulation import simulate_merger


class TestLinearDemand(unittest.TestCase):
    def test_calibration_recovers_B(self):
        m = synthetic_linear_market(n_products=6, own_slope=2.0, cross_slope=0.5, seed=0)
        true_D = m.demand.diversion_ratios()
        cal = Market.calibrated(m.product_ids, m.firm_ids, m.prices, m.shares,
                                margins=m.prices - m.costs, market_size=m.demand.market_size,
                                demand_type="linear", diversion=true_D)
        np.testing.assert_allclose(cal.demand.B, m.demand.B, atol=1e-8)
        np.testing.assert_allclose(cal.demand.shares(cal.prices), m.shares, atol=1e-10)

    def test_premerger_foc_holds(self):
        m = synthetic_linear_market(seed=1)
        self.assertLess(np.max(np.abs(m.foc(m.prices))), 1e-8)

    def test_merger_raises_merged_prices_and_solves_foc(self):
        m = synthetic_linear_market(n_products=6, seed=2, market_size=1000.0)
        res = simulate_merger(m, merging_firms=[1, 2])
        in_m = np.isin(m.firm_ids, [1, 2])
        self.assertTrue(np.all(res.post["prices"][in_m] > res.pre["prices"][in_m]))
        self.assertLess(np.max(np.abs(m.foc(res.post["prices"], res.firm_ids_post))), 1e-7)
        self.assertGreater(res.post["hhi"], res.pre["hhi"])

    def test_diversion_constant_in_prices(self):
        m = synthetic_linear_market(seed=3)
        np.testing.assert_allclose(m.demand.diversion_ratios(m.prices),
                                   m.demand.diversion_ratios(m.prices * 1.3))

    def test_linear_share_unbounded_logit_bounded(self):
        df = load_nevo_products()
        mid = df["market_ids"].iloc[0]
        m_lin = nevo_market(mid, df, demand_type="linear")
        m_log = nevo_market(mid, df, demand_type="logit")
        p_high = m_lin.prices.copy()
        p_high[0] = 10.0 * m_lin.prices[0]
        self.assertLess(m_lin.demand.shares(p_high)[0], 0.0)
        self.assertGreater(m_log.demand.shares(p_high)[0], 0.0)
        self.assertLess(m_log.demand.shares(p_high).sum(), 1.0)

    def test_cereal_linear_merger_smoke(self):
        df = load_nevo_products()
        m = nevo_market(df["market_ids"].iloc[0], df, demand_type="linear")
        res = simulate_merger(m, merging_firms=[1, 3])
        self.assertTrue(np.all(np.isfinite(res.post["prices"])))
        in_m = np.isin(m.firm_ids, [1, 3])
        self.assertGreater(res.post["prices"][in_m].mean(), res.pre["prices"][in_m].mean())
        self.assertGreater(res.post["shares"].min(), 0.0)


if __name__ == "__main__":
    unittest.main()
