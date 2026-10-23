# Pre-registered predictions: Experiment 3, class 2, position sweep (country at i, capital at j, every i < j ≤ 32)

Written 2026-10-15, before any data from this sweep. Its inputs are the results of R1-out
(`results/20261015-190957_exp3rel`) and R1-in (`results/20261015-203150_exp3rel_in`), which correspond to placements
(30, 31) and (31, 32). Config: `configs/exp3_positions.yaml`.

## Design

- **Placements:** country at model position i, capital at j, for every 1 ≤ i < j ≤ 32, which gives 496
  placements. When j = 32 the capital is the final query. Otherwise the query is the row's own random token.
- **Rows:** the first 50 base rows per class of R1-out / R1-in (rare 1000–39999, common 256–999). The same rows are
  used at every placement, so placements are compared within row.
- **Arms per placement and row** (11 cells each, 44 cells in total):
  - **correct:** each country with its own capital.
  - **wrong:** two fixed derangements, d1 and d2. Each capital appears once per derangement.
  - **placebo:** a third fixed derangement, d3.
  - d1, d2 and d3 are mutually disjoint and never pair a country with its own capital. They come from one random
    Latin square (seed 3).
- **T** = mean(correct) − mean(wrong), per row. Every country and every capital appears once in the correct arm and
  once on average in the wrong arm, so additive token effects cancel.
- **Placebo T** = mean(d3) − mean(wrong).
- **Late** = layers 6–11, the mean over their 72 heads. Metrics are masses at the final query: BOS, country (i),
  capital (j), previous (31, when it is neither i nor j), self (32, when it is not j), and filler.
- **Significant placement:** Wilcoxon over the 50 rows, BH over the 496 placements within a metric and base class,
  q ≤ 0.05 and |T| ≥ 0.005.

## Predictions

Each must hold in both base classes.

| id | prediction | why |
|---|---|---|
| S1 | **Completion at the query survives a gap.** For j = 32 (the capital is the query) and every i in 26–31, late BOS T < 0 with a 95% row CI excluding 0 | R1-in: −0.018 / −0.034 at i = 31 |
| S2 | **It weakens with distance.** For j = 32: T(31, 32) − mean over i ≤ 16 of T(i, 32) < 0, with a 95% row CI excluding 0 | the adjacent pair carried most of the effect in Test A, and distance weakened Exp 2's effects |
| S3 | **No BOS effect before a random query.** Among the 465 placements with j ≤ 31, ≤ 10% are significant on late BOS | R1-out: equivalent or −0.004 at (30, 31) |
| S4 | **Within-pair shift near the query.** For adjacent pairs (j = i + 1) with j in 28–31, late T(country mass) − T(capital mass) > 0, with a 95% row CI excluding 0 | R1-out: the correct pairing moves attention from the capital onto the country |
| S5 | **Placebo.** ≤ 5% of the 496 placements are significant on the placebo T for late BOS | the pipeline does not manufacture pairing effects |

## Checks

- Every cell holds the right ids at i and j and equals its base row elsewhere. No repeats, no BOS, and no stimulus
  or demonstration id appears in a base row.
- Attention rows sum to 1.
- Determinism: one placement, recomputed on 2 rows per class, matches exactly.
- TL vs HF: placements (1, 2), (30, 31) and (31, 32) on 5 rows per class, max |Δ| < 1e-4.
- Consistency: at (30, 31) and (31, 32), the sign of late BOS T agrees with R1-out / R1-in. This is reported only:
  the wrong arm differs (2 derangements here vs the full grid there) and only 50 of the 200 rows are used.

## Reported without a prediction

- Triangle maps of T for every metric.
- Profiles over position and gap.
- Mean attention by key position, correct vs wrong, at selected placements.
- Meaningful-head counts per placement (per-head Wilcoxon, BH over 144 heads, |T| ≥ 0.02).
