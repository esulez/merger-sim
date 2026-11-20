# Literature review: calibrated merger simulation

Merger simulation predicts the effects of a hypothetical horizontal merger using only pre-merger data. The standard recipe (used by antitrust agencies since the 1990s) has three steps:

1. **Specify/calibrate demand** for the products in the market (logit, nested logit, random-coefficients logit, AIDS, linear).
2. **Impute marginal costs** from the pre-merger Nash–Bertrand (or Cournot) first-order conditions, given observed prices, shares and margins.
3. **Re-solve the pricing game** under the post-merger ownership structure and compare the equilibria (prices, shares, HHI, profits, consumer surplus).

A lighter-weight screen is **upward pricing pressure (UPP)**, which needs only margins and diversion ratios, not a full demand system.

## 1. Foundations

- Farrell & Shapiro (1990), "Horizontal Mergers: An Equilibrium Analysis", *American Economic Review* 80(1):107–126 — the founding equilibrium treatment of horizontal mergers (Cournot); price/welfare propositions, the "no synergies ⇒ price rises" result, and a critique of HHI-based screens.
- Nocke & Whinston (2010), "Dynamic Merger Review", *Journal of Political Economy* 118(6):1201–1251 — dynamic extension; mergers as an endogenous choice.

## 2. The calibrated Nash–Bertrand recipe (HLZ / Werden–Froeb)

- Hausman, Leonard & Zona (1994), "Competitive Analysis with Differentiated Products", *Annales d'Économie et de Statistique* 34:159–180 — the agency template: calibrate demand from shares/prices, infer margins, simulate the post-merger equilibrium.
- Werden & Froeb (1994), "The Effects of Mergers in Differentiated Products Industries: Logit Demand and Merger Policy", *Journal of Law, Economics, and Organization* 10(2):407–426 — logit demand plus merger; closed-form elasticities and diversion.
- Crooke, Froeb, Tschantz & Werden (1999), "Effects of Assumed Demand Form on Simulated Post-Merger Equilibria", *Review of Industrial Organization* 15:205–220 — sensitivity of simulated results to the demand form.
- Froeb, Tschantz & Werden (2005), "The Effects of Merger Efficiencies on Consumers of Differentiated Products", *European Competition Journal* 4(2):245–262 — adding marginal-cost efficiencies to the simulation.
- Epstein & Rubinfeld (2001), "Merger Simulation: A Simplified Approach with New Applications", UC Berkeley working paper — simplified calibration from shares and diversion ratios.
- Hausman & Leonard (2005), "Using Merger Simulation Models: Testing the Underlying Assumptions", *International Journal of Industrial Organization* 23:693–698.

## 3. Demand estimation (BLP and friends)

- Berry (1994), "Estimating Discrete-Choice Models of Product Differentiation", *RAND Journal of Economics* 25(2):242–262 — inversion of the share equations to recover mean utilities.
- Berry, Levinsohn & Pakes (1995), "Automobile Prices in Market Equilibrium", *Econometrica* 63(4):841–890 — BLP random-coefficients demand plus Nash–Bertrand supply; US car prices.
- Nevo (2000), "A Practitioner's Guide to Estimation of Random-Coefficients Logit Models of Demand", *Journal of Economic Management and Strategy* 9(4):513–548 — the workhorse estimation guide; its "fake cereal" data is bundled in this repo (`data/nevo_cereal/`).
- Nevo (2001), "Measuring Market Power in the Ready-to-Eat Cereal Industry", *Econometrica* 68(2):307–342.
- Nevo (2000), "Mergers with Differentiated Products: The Case of the Ready-to-Eat Cereal Industry", *RAND Journal of Economics* 31(3):395–421 — cereal merger simulations; the example in `python/examples/run_cereal_merger.py` mimics these.
- Ivaldi & Verboven (2005), "Quantifying the Effects from Horizontal Mergers in European Competition Policy", *International Journal of Industrial Organization* 23(4):345–369 — European car market; European Commission merger cases.
- Shapiro (1996), "Mergers with Differentiated Products", *Antitrust* 10:23–30.

## 4. Screening tools: UPP and diversion ratios

- [Farrell & Shapiro (2010), "Antitrust Evaluation of Horizontal Mergers: An Economic Alternative to Market Definition", *B.E. Journal of Theoretical Economics* 10(1), Art. 9](https://faculty.haas.berkeley.edu/shapiro/alternative.pdf) — **open access**. Defines net UPP for product 1 as `UPP_1 = D_12 (P2 − C2) − E_1 C1` (generalized to `Σ_j D_1j (P_j − C_j) − E_1 C_1` over the merging partner's products); under logit, `D_12 = s_2/(1 − s_1)`. Implemented in `python/mergersim/simulation.py`.
- Jaffe & Weyl (2013), "Empirical Properties of Diversion Ratios", NBER Working Paper — what diversion ratios look like in practice.

## 5. Applications and validation

- Peters (2006), "Evaluating the Performance of Merger Simulation: Evidence from the U.S. Airline Industry", *Journal of Law and Economics* 49(2):627–649 — back-tests merger simulations against consummated airline mergers.
- Björnerstedt & Verboven (2016), "Does Merger Simulation Work? A 'Natural Experiment' in the Swedish Analgesics Market", *Journal of Industrial Economics* 64(4):550–582.
- Miller & Weinberg (2017), "Understanding the Price Effects of the MillerCoors Joint Venture", *Econometrica* 85(6):1763–1791.
- Gowrisankaran, Nevo & Town (2015), "Mergers When Prices Are Negotiated: Evidence from the Hospital Industry", *American Economic Review* 105(1):172–203 — bargained (not posted) prices.
- Carlton, Israel, MacSwain & Orlov (2019), "The Price Effects of Mergers in U.S. Airline Markets", working paper — retrospective difference-in-differences estimates for US airline mergers.
- Nocke & Schutz (2018), "An Aggregative Games Approach to Merger Analysis in Multiproduct-Firm Oligopoly", CEPR DP 12905 (published in *RAND Journal of Economics*, 2025) — multiproduct-firm merger analysis.
- Doraszelski & Markovich (2008), "Identifying Dynamic Oligopoly Games", *Review of Economic Studies* 75(4):984–1004 — dynamic oligopoly games.

## 6. Surveys and policy documents

- Budzinski & Ruhmer (2010), "Merger Simulation in Competition Policy: A Survey", *Journal of Competition Law & Economics* 6(2):277–319, [doi:10.1093/joclec/nhp014](https://doi.org/10.1093/joclec/nhp014) — the survey of the field.
- Nevo & Whinston (2010), "Taking the Dogma out of Econometrics: Structural Modeling and Credible Inference", *Journal of Economic Perspectives* 24(2):69–82, [doi:10.1257/jep.24.2.69](https://doi.org/10.1257/jep.24.2.69) — open-access PDF via U. Chicago HCE.
- [DOJ/FTC 2010 Horizontal Merger Guidelines](https://www.justice.gov/atr/2010-merger-guidelines) (§6.4 on merger simulation) and the [2023 Merger Guidelines](https://www.justice.gov/atr/2023-merger-guidelines).
- [FTC staff paper, "Demand System Estimation and its Application to Horizontal Merger Analysis"](https://www.ftc.gov/sites/default/files/documents/reports/demand-system-estimation-and-its-application-horizontal-merger-analysis/wp246_0.pdf).

## 7. Data

- **Nevo fake cereal data** — bundled in `data/nevo_cereal/` (as shipped with [PyBLP](https://pyblp.readthedocs.io/en/stable/_api/pyblp.data.html)): 94 city-quarter markets × 24 products with shares, prices, attributes. Used by `python/examples/run_cereal_merger.py`.
- **US airlines (BTS DB1B 10% ticket sample + T-100 segment data)** — public at [BTS airline data downloads](https://www.bts.gov/airline-data-downloads); downloader script: `data/fetch_bts.py`. This is the data behind Peters (2006) and Carlton et al. (2019).
- **US cars (BLP 1995)** — bundled with PyBLP (`pyblp.data`), extracted by Andrews, Gentzkow & Shapiro (2017) from the original BLP code.

## Where papers are saved

Open-access PDFs of the papers selected for this project are stored in `papers/` (selection in progress). Every paper above is cited with a DOI or link regardless.
