"""Tests for the calibrated logit merger simulation. Run: PYTHONPATH=python python -m unittest discover -s python/tests -v"""

import unittest

import numpy as np

from mergersim.data import nevo_market, load_nevo_products, synthetic_market
from mergersim.market import Market
from mergersim.simulation import report, simulate_merger, upward_pricing_pressure


class TestDemandCalibration(unittest.TestCase):
    def test_shares_roundtrip(self):
        m = synthetic_market(n_products=8, a_true=2.0, seed=1)
        cal = Market.calibrated(m.product_ids, m.firm_ids, m.prices, m.shares,
                                margins=m.prices - m.costs, market_size=m.demand.market_size)
        np.testing.assert_allclose(cal.demand.shares(cal.prices), m.shares, atol=1e-10)

    def test_calibration_recovers_a(self):
        m = synthetic_market(n_products=8, a_true=2.0, seed=2)
        cal = Market.calibrated(m.product_ids, m.firm_ids, m.prices, m.shares,
                                margins=m.prices - m.costs)
        self.assertAlmostEqual(cal.demand.a, m.demand.a, places=8)

    def test_premerger_foc_holds(self):
        m = synthetic_market(n_products=8, a_true=1.5, seed=3)
        self.assertLess(np.max(np.abs(m.foc(m.prices))), 1e-8)

    def test_diversion_logit_formula(self):
        m = synthetic_market(seed=4)
        D = m.demand.diversion_ratios(m.prices)
        s = m.shares
        for i in range(len(s)):
            for j in range(len(s)):
                if i != j:
                    self.assertAlmostEqual(D[i, j], s[j] / (1.0 - s[i]), places=12)


class TestMergerSimulation(unittest.TestCase):
    def setUp(self):
        self.m = synthetic_market(n_products=8, a_true=2.0, seed=5, market_size=1000.0)
        self.res = simulate_merger(self.m, merging_firms=[1, 2], efficiency=0.0)

    def test_merged_products_raise_prices(self):
        in_m = np.isin(self.m.firm_ids, [1, 2])
        dp = self.res.post["prices"] - self.res.pre["prices"]
        self.assertTrue(np.all(dp[in_m] > 0), "no-synergy merger should raise merging firms' prices")

    def test_rivals_weakly_raise_prices(self):
        out = ~np.isin(self.m.firm_ids, [1, 2])
        dp = self.res.post["prices"] - self.res.pre["prices"]
        self.assertTrue(np.all(dp[out] >= -1e-8))

    def test_post_merger_foc_holds(self):
        r = self.m.foc(self.res.post["prices"], self.res.firm_ids_post)
        self.assertLess(np.max(np.abs(r)), 1e-7)

    def test_hhi_rises_and_cs_falls(self):
        self.assertGreater(self.res.post["hhi"], self.res.pre["hhi"])
        self.assertLess(self.res.post["cs"], self.res.pre["cs"])

    def test_shares_valid(self):
        self.assertLess(self.res.post["shares"].sum(), 1.0)
        self.assertTrue(np.all(self.res.post["shares"] > 0))

    def test_upp_positive_without_efficiencies(self):
        upp = upward_pricing_pressure(self.m, [1, 2], efficiency=0.0)
        in_m = np.isin(self.m.firm_ids, [1, 2])
        self.assertTrue(np.all(upp[in_m] > 0))
        self.assertTrue(np.all(upp[~in_m] == 0))

    def test_upp_net_of_efficiencies(self):
        upp0 = upward_pricing_pressure(self.m, [1, 2], efficiency=0.0)
        upp1 = upward_pricing_pressure(self.m, [1, 2], efficiency=0.9)
        in_m = np.isin(self.m.firm_ids, [1, 2])
        np.testing.assert_allclose(upp1[in_m], upp0[in_m] - 0.9 * self.m.costs[in_m], atol=1e-12)
        self.assertTrue(np.all(upp1[in_m] < 0), "large efficiency credit should make net UPP negative")

    def test_report_runs(self):
        df, summary = report(self.res)
        self.assertEqual(len(df), len(self.m.product_ids))
        self.assertEqual(list(summary.index), ["pre-merger", "post-merger"])


class TestCerealData(unittest.TestCase):
    def test_cereal_market_merger_smoke(self):
        df = load_nevo_products()
        m = nevo_market(df["market_ids"].iloc[0], df, lerner=0.3)
        self.assertEqual(len(m.product_ids), 24)
        res = simulate_merger(m, merging_firms=[1, 2], efficiency=0.0)
        self.assertTrue(np.all(np.isfinite(res.post["prices"])))
        self.assertLess(res.post["shares"].sum(), 1.0)
        in_m = np.isin(m.firm_ids, [1, 2])
        self.assertGreater(res.post["prices"][in_m].mean(), res.pre["prices"][in_m].mean())


if __name__ == "__main__":
    unittest.main()
