# TDA 1: persistent homology of attention-head behaviour profiles

Run `20261025-125719_exp4tda1`. Pre-registration: `PREDICTIONS.md` (copy of `exp4/PREDICTIONS_tda1.md`, written before any topology was computed). Saved results only: `results/20261024-000548_exp4ctrl/controlled.npz` and `results/20261024-150407_exp4cap/capital.npz`.

**Profile.** One point per head (144). Each point is 8 D(d) curves × 31 distances = 248 features, in attention units:
- country arm with a random final token, its own capital, and a wrong capital;
- placebo-word arm with a random final token;
- each in common and rare filler.

**Method.**
- Vietoris–Rips H0 and H1 with ripser.
- Stability: 500 row bootstraps (Fasy et al. bottleneck band c_k; Hennig Jaccard recurrence at a fixed cut).
- Nulls, 500 draws each: N1, row sign-flip (noise); N2, moment-matched Gaussian (shape).
- Cut: ε* = max(N1's 95th percentile of the largest H0 death, 2·c0).

## Checks

| check | result |
|---|---|
| rebuilt per-row means = saved means, Exp 4 | PASS (max |Δ| 0.0e+00) |
| rebuilt per-row means = saved means, Exp 4b (float16) | PASS (max |Δ| 2.7e-05) |
| ripser H0 deaths = single-linkage heights, raw | PASS (max |Δ| 5.8e-08) |
| ripser H0 deaths = single-linkage heights, corr | PASS (max |Δ| 3.0e-08) |
| ripser H0 deaths = single-linkage heights, std | PASS (max |Δ| 1.1e-06) |
| N1 mean profile ≈ 0 | max |mean over draws| 2.0e-03 vs max |profile| 0.516 |

## Clarification of the pre-registration (decided after a 20-draw smoke test, before the full run)

- **Ambiguity.** The pre-registration did not say whether a null or bootstrap cloud is z-scored by its own per-feature SDs or by the main profiles' SDs.
- **Choice.** Here every cloud uses the main profiles' SDs, so all clouds are measured in one fixed metric. That is the reading that matches N1's stated purpose: separations that estimation noise produces under the main metric.
- **Why not self-standardisation.** In a noise cloud, each feature's variance across heads sits in a few high-noise heads. Self-standardisation therefore isolates those heads far more than the real metric would.
- **The other reading, for transparency.** With self-standardised N1 clouds, std gives N1 q95 47.83, ε* 47.83, 1 component(s), of which 1 have ≥ 2 heads.

## Thresholds

| variant | c0 (H0 band half-width) | c1 | N1 q95 max H0 death | N2 q95 max H0 death | ε* | N1 q95 max H1 ratio | N2 q95 max H1 ratio | H0 bars > N1 / > band / > N2 |
|---|---|---|---|---|---|---|---|---|
| raw | 0.04952 | 0.01779 | 0.1894 | 0.6092 | 0.1894 | 1.179 | 1.383 | 31 / 75 / 4 |
| corr | 0.09827 | 0.1051 | 0.7993 | 0.5684 | 0.7993 | 1.676 | 2.722 | 0 / 34 / 0 |
| std | 3.686 | 1.216 | 8.094 | 21.01 | 8.094 | 1.138 | 1.323 | 34 / 39 / 7 |

## Pre-registered predictions (raw metric = main analysis)

| id | verdict | detail |
|---|---|---|
| T1-a | FAIL | ε* 0.1894; 32 components: 3 with ≥ 2 heads [110 heads (rec 0.98), 3 heads (rec 0.55), 2 heads (rec 0.92)], 29 isolated heads; stable (rec ≥ 0.75) clusters: 2 |
| T1-b | FAIL | L0H2 in a 1-head component, L4H10 in a 1-head component, L3H0 in a 1-head component, L1H11 in a 1-head component; together: False; previous-token heads in their component(s): none; configuration recurs in 0% of resamples |
| T1-c | FAIL | common: ARI +0.123 [+0.083, +0.138] (labels do not match persistent clusters); rare: ARI +0.209 [+0.193, +0.247] (partly) |
| T1-d | PASS | 25 H1 features; beyond band 0, beyond N2 ratio 0, beyond N1 ratio 2; band & N2: 0; top ratio feature: birth 0.1542, death 0.1874, ratio 1.215 (N2 q95 1.383, N1 q95 1.179), persistence 0.03314 (2·c1 0.03559) |

## Robustness: the same criteria under the other metrics

| id | raw | corr | std | robust |
|---|---|---|---|---|
| T1-a | FAIL | FAIL | FAIL | yes |
| T1-b | FAIL | FAIL | FAIL | yes |
| T1-c | FAIL | FAIL | FAIL | yes |
| T1-d | PASS | PASS | PASS | yes |

corr details:

- T1-a FAIL: ε* 0.7993; 1 components: 1 with ≥ 2 heads [144 heads (rec 1.00)], 0 isolated heads; stable (rec ≥ 0.75) clusters: 1
- T1-b FAIL: L0H2 in a 144-head component, L4H10 in a 144-head component, L3H0 in a 144-head component, L1H11 in a 144-head component; together: True; previous-token heads in their component(s): L1H0, L2H2, L2H3, L2H5, L2H8, L2H9, L3H2, L3H6, L3H7, L3H8, L4H11; configuration recurs in 0% of resamples
- T1-c FAIL: common: ARI +0.000 [+0.000, +0.000] (labels do not match persistent clusters); rare: ARI +0.000 [+0.000, +0.000] (labels do not match persistent clusters)
- T1-d PASS: 61 H1 features; beyond band 1, beyond N2 ratio 0, beyond N1 ratio 0; band & N2: 0; top ratio feature: birth 0.4709, death 0.7685, ratio 1.632 (N2 q95 2.722, N1 q95 1.676), persistence 0.2976 (2·c1 0.2102)

std details:

- T1-a FAIL: ε* 8.094; 35 components: 1 with ≥ 2 heads [110 heads (rec 0.96)], 34 isolated heads; stable (rec ≥ 0.75) clusters: 1
- T1-b FAIL: L0H2 in a 1-head component, L4H10 in a 1-head component, L3H0 in a 1-head component, L1H11 in a 1-head component; together: False; previous-token heads in their component(s): none; configuration recurs in 0% of resamples
- T1-c FAIL: common: ARI +0.055 [+0.015, +0.063] (labels do not match persistent clusters); rare: ARI +0.048 [+0.016, +0.094] (labels do not match persistent clusters)
- T1-d PASS: 30 H1 features; beyond band 1, beyond N2 ratio 0, beyond N1 ratio 3; band & N2: 0; top ratio feature: birth 8.143, death 10.61, ratio 1.303 (N2 q95 1.323, N1 q95 1.138), persistence 2.466 (2·c1 2.432)

## Persistent components at ε*: raw Euclidean (main)

Gap to rest = smallest distance from a member to a non-member (the H0 death at which the component merges). Internal = largest single-linkage merge inside it. Recurrence = mean best Jaccard over resamples.

| # | size | gap to rest | internal | recurrence | members | labels (common) | labels (rare) |
|---|---|---|---|---|---|---|---|
| 1 | 110 | 0.1924 | 0.1847 | 0.98 | L0H0, L0H1, L0H3, L0H5, L0H6, L0H8, L0H11, L1H1, L1H2, L1H3, … (+100) | no effect 61, local 17, gated prev 12, global 12, negative 6, plain prev 2 | no effect 82, local 12, negative 8, gated prev 5, plain prev 3 |
| 2 | 3 | 0.1924 | 0.1814 | 0.55 | L0H7, L2H0, L2H4 | negative 3 | no effect 2, negative 1 |
| 3 | 2 | 0.2203 | 0.1694 | 0.92 | L2H5, L3H8 | gated prev 1, local 1 | local 2 |

Isolated heads (29; gap to nearest head, recurrence): L0H2 (0.841, rec 1.00); L0H4 (0.205, rec 1.00); L0H9 (0.254, rec 1.00); L0H10 (0.274, rec 1.00); L1H0 (0.217, rec 0.98); L1H5 (1.42, rec 1.00); L1H6 (0.201, rec 1.00); L1H11 (0.41, rec 1.00); L2H2 (0.204, rec 0.96); L2H3 (0.241, rec 1.00); L2H8 (0.343, rec 1.00); L2H9 (0.241, rec 1.00); L3H0 (0.854, rec 1.00); L3H1 (0.213, rec 0.98); L3H2 (0.202, rec 0.95); L3H4 (0.59, rec 1.00); L3H6 (0.344, rec 1.00); L3H7 (0.204, rec 0.96); L4H4 (0.231, rec 1.00); L4H8 (0.287, rec 1.00); L4H10 (0.841, rec 1.00); L5H6 (0.252, rec 1.00); L7H0 (0.219, rec 1.00); L9H8 (0.574, rec 1.00); L10H0 (0.263, rec 1.00); L10H3 (0.364, rec 1.00); L11H2 (0.376, rec 1.00); L11H7 (0.364, rec 1.00); L11H8 (0.252, rec 1.00).

## Persistent components at ε*: correlation distance

Gap to rest = smallest distance from a member to a non-member (the H0 death at which the component merges). Internal = largest single-linkage merge inside it. Recurrence = mean best Jaccard over resamples.

| # | size | gap to rest | internal | recurrence | members | labels (common) | labels (rare) |
|---|---|---|---|---|---|---|---|
| 1 | 144 | inf | 0.5668 | 1.00 | L0H0, L0H1, L0H2, L0H3, L0H4, L0H5, L0H6, L0H7, L0H8, L0H9, … (+134) | no effect 70, local 26, global 17, gated prev 15, negative 12, plain prev 4 | no effect 97, local 21, negative 12, gated prev 5, plain prev 5, global 4 |

Isolated heads (0; gap to nearest head, recurrence): none.

## Persistent components at ε*: per-feature standardised

Gap to rest = smallest distance from a member to a non-member (the H0 death at which the component merges). Internal = largest single-linkage merge inside it. Recurrence = mean best Jaccard over resamples.

| # | size | gap to rest | internal | recurrence | members | labels (common) | labels (rare) |
|---|---|---|---|---|---|---|---|
| 1 | 110 | 8.12 | 8.088 | 0.96 | L0H0, L0H1, L0H3, L0H4, L0H5, L0H7, L0H8, L0H10, L0H11, L1H0, … (+100) | no effect 57, local 19, gated prev 14, negative 11, global 5, plain prev 4 | no effect 76, local 15, negative 9, gated prev 5, plain prev 5 |

Isolated heads (34; gap to nearest head, recurrence): L0H2 (22.8, rec 1.00); L0H6 (9.34, rec 0.98); L0H9 (8.56, rec 1.00); L1H5 (43.6, rec 1.00); L1H6 (8.61, rec 0.99); L1H7 (8.73, rec 0.99); L1H11 (9.48, rec 1.00); L2H1 (8.69, rec 1.00); L2H3 (8.12, rec 0.56); L2H8 (12, rec 1.00); L3H0 (18.7, rec 1.00); L3H1 (10.5, rec 1.00); L3H4 (24.1, rec 1.00); L3H6 (10.7, rec 1.00); L4H8 (15, rec 1.00); L4H10 (18.7, rec 1.00); L5H3 (8.94, rec 1.00); L8H6 (8.88, rec 1.00); L8H11 (8.68, rec 0.93); L9H8 (29.5, rec 1.00); L9H11 (11.4, rec 1.00); L10H0 (12.6, rec 1.00); L10H3 (25.4, rec 1.00); L10H4 (14.5, rec 1.00); L10H6 (8.27, rec 0.79); L10H11 (8.73, rec 0.99); L11H0 (8.88, rec 1.00); L11H1 (11.4, rec 1.00); L11H2 (19.9, rec 1.00); L11H4 (9.43, rec 1.00); L11H6 (12.6, rec 1.00); L11H7 (19.9, rec 1.00); L11H8 (10, rec 1.00); L11H9 (10.6, rec 1.00).

## Focus heads: component at ε* (size)

| head | group | raw | corr | std |
|---|---|---|---|---|
| L0H2 | same-country | #4 (1) | #1 (144) | #2 (1) |
| L4H10 | same-country | #24 (1) | #1 (144) | #17 (1) |
| L3H0 | same-country | #16 (1) | #1 (144) | #12 (1) |
| L1H11 | same-country | #11 (1) | #1 (144) | #8 (1) |
| L1H0 | Exp 1 previous-token | #8 (1) | #1 (144) | #1 (110) |
| L2H2 | Exp 1 previous-token | #12 (1) | #1 (144) | #1 (110) |
| L2H3 | Exp 1 previous-token | #13 (1) | #1 (144) | #10 (1) |
| L2H5 | Exp 1 previous-token | #3 (2) | #1 (144) | #1 (110) |
| L2H8 | Exp 1 previous-token | #14 (1) | #1 (144) | #11 (1) |
| L2H9 | Exp 1 previous-token | #15 (1) | #1 (144) | #1 (110) |
| L3H2 | Exp 1 previous-token | #18 (1) | #1 (144) | #1 (110) |
| L3H6 | Exp 1 previous-token | #20 (1) | #1 (144) | #15 (1) |
| L3H7 | Exp 1 previous-token | #21 (1) | #1 (144) | #1 (110) |
| L3H8 | Exp 1 previous-token | #3 (2) | #1 (144) | #1 (110) |
| L4H11 | Exp 1 previous-token | #1 (110) | #1 (144) | #1 (110) |
| L9H8 | late any-distance | #27 (1) | #1 (144) | #21 (1) |
| L10H3 | late any-distance | #29 (1) | #1 (144) | #24 (1) |
| L11H7 | late any-distance | #31 (1) | #1 (144) | #33 (1) |
| L11H2 | late any-distance | #30 (1) | #1 (144) | #30 (1) |
| L11H8 | local country-reading | #32 (1) | #1 (144) | #34 (1) |

## Agreement with the threshold labels (ARI)

| variant | filler | ARI at ε* [95% boot] | labels 5+6 merged | single linkage, 6 clusters | Ward, 6 clusters |
|---|---|---|---|---|---|
| raw | common | +0.123 [+0.083, +0.138] | +0.109 | -0.010 | +0.127 |
| raw | rare | +0.209 [+0.193, +0.247] | +0.198 | -0.049 | +0.085 |
| corr | common | +0.000 [+0.000, +0.000] | +0.000 | -0.006 | +0.051 |
| corr | rare | +0.000 [+0.000, +0.000] | +0.000 | +0.009 | +0.030 |
| std | common | +0.055 [+0.015, +0.063] | +0.079 | +0.005 | +0.120 |
| std | rare | +0.048 [+0.016, +0.094] | +0.087 | -0.001 | -0.002 |

Reference: ARI between the common and rare labels themselves = +0.246.

Labels × raw components (common filler; components with ≥ 2 heads, then isolated heads):

| label | #1 (110) | #2 (3) | #3 (2) | isolated |
|---|---|---|---|---|
| global | 12 | 0 | 0 | 5 |
| local | 17 | 0 | 1 | 8 |
| gated prev | 12 | 0 | 1 | 2 |
| negative | 6 | 3 | 0 | 3 |
| plain prev | 2 | 0 | 0 | 2 |
| no effect | 61 | 0 | 0 | 9 |

## H1 features (raw), largest death/birth first

| birth | death | ratio | persistence | > 2·c1 | > N2 ratio q95 | > N1 ratio q95 |
|---|---|---|---|---|---|---|
| 0.1542 | 0.1874 | 1.215 | 0.03314 | False | False | True |
| 0.09829 | 0.1166 | 1.186 | 0.0183 | False | False | True |
| 0.208 | 0.2402 | 1.155 | 0.03219 | False | False | False |
| 0.06258 | 0.07118 | 1.137 | 0.008601 | False | False | False |
| 0.08552 | 0.09721 | 1.137 | 0.01169 | False | False | False |
| 0.2211 | 0.2474 | 1.119 | 0.02628 | False | False | False |
| 0.07475 | 0.08291 | 1.109 | 0.008153 | False | False | False |
| 0.1291 | 0.142 | 1.100 | 0.01295 | False | False | False |

## Exploratory (not pre-registered; raw metric)

Isolated heads at ε*, by gap. Curve shares are each curve's fraction of the head's squared profile norm; the two largest are shown.

| head | gap | nearest head | recurrence | profile norm | largest curve shares | Exp 4 label common / rare |
|---|---|---|---|---|---|---|
| L1H5 | 1.42 | L3H4 | 1.00 | 2.24 | common: country, wrong capital 63%, rare: country, wrong capital 18% | no effect / no effect |
| L3H0 | 0.854 | L4H10 | 1.00 | 2.29 | rare: country, own capital 56%, common: country, own capital 43% | no effect / no effect |
| L0H2 | 0.841 | L4H10 | 1.00 | 2.85 | common: country, own capital 54%, rare: country, own capital 46% | local / no effect |
| L4H10 | 0.841 | L0H2 | 1.00 | 2.09 | common: country, own capital 57%, rare: country, own capital 41% | no effect / no effect |
| L3H4 | 0.59 | L11H2 | 1.00 | 1.06 | common: country, wrong capital 82%, rare: country, wrong capital 16% | gated prev / no effect |
| L9H8 | 0.574 | L11H7 | 1.00 | 1.09 | common: country, random T 63%, common: country, own capital 20% | global / global |
| L1H11 | 0.41 | L0H10 | 1.00 | 1.03 | common: country, own capital 63%, rare: country, own capital 36% | no effect / no effect |
| L11H2 | 0.376 | L7H6 | 1.00 | 0.658 | common: country, wrong capital 54%, rare: country, wrong capital 14% | global / global |
| L10H3 | 0.364 | L11H7 | 1.00 | 1.06 | rare: country, random T 29%, common: country, wrong capital 20% | global / global |
| L11H7 | 0.364 | L10H3 | 1.00 | 0.883 | common: country, wrong capital 25%, common: country, own capital 24% | global / global |
| L3H6 | 0.344 | L2H3 | 1.00 | 0.637 | common: word, random T 46%, common: country, random T 25% | local / local |
| L2H8 | 0.343 | L5H4 | 1.00 | 0.594 | rare: country, own capital 22%, common: country, own capital 19% | negative / negative |
| L4H8 | 0.287 | L7H6 | 1.00 | 0.504 | common: country, wrong capital 51%, rare: country, wrong capital 34% | no effect / no effect |
| L0H10 | 0.274 | L4H4 | 1.00 | 0.712 | common: country, own capital 49%, rare: country, own capital 40% | no effect / no effect |
| L10H0 | 0.263 | L8H11 | 1.00 | 0.563 | common: country, random T 42%, common: country, own capital 29% | global / no effect |
| L0H9 | 0.254 | L5H9 | 1.00 | 0.422 | common: country, wrong capital 64%, rare: country, wrong capital 21% | no effect / no effect |
| L11H8 | 0.252 | L11H3 | 1.00 | 0.54 | common: country, wrong capital 41%, common: country, own capital 38% | local / local |
| L5H6 | 0.252 | L6H8 | 1.00 | 0.334 | common: country, own capital 36%, rare: country, own capital 34% | plain prev / negative |
| L2H3 | 0.241 | L2H9 | 1.00 | 0.392 | common: word, random T 44%, common: country, random T 28% | local / local |
| L2H9 | 0.241 | L2H3 | 1.00 | 0.45 | common: country, random T 33%, common: word, random T 26% | local / local |
| L4H4 | 0.231 | L10H7 | 1.00 | 0.524 | common: country, own capital 57%, rare: country, own capital 18% | no effect / no effect |
| L7H0 | 0.219 | L3H2 | 1.00 | 0.354 | common: country, own capital 27%, common: country, random T 25% | local / local |
| L1H0 | 0.217 | L3H10 | 0.98 | 0.393 | common: country, wrong capital 37%, common: country, own capital 30% | negative / negative |
| L3H1 | 0.213 | L5H3 | 0.98 | 0.355 | rare: country, random T 53%, common: country, random T 23% | local / local |
| L0H4 | 0.205 | L2H6 | 1.00 | 0.282 | common: country, own capital 48%, rare: country, own capital 40% | negative / no effect |
| L2H2 | 0.204 | L3H7 | 0.96 | 0.352 | rare: country, own capital 43%, common: country, own capital 35% | plain prev / local |
| L3H7 | 0.204 | L2H2 | 0.96 | 0.481 | rare: country, own capital 51%, common: country, own capital 39% | local / plain prev |
| L3H2 | 0.202 | L4H3 | 0.95 | 0.357 | common: word, random T 37%, common: country, own capital 15% | gated prev / plain prev |
| L1H6 | 0.201 | L5H11 | 1.00 | 0.363 | common: country, own capital 61%, rare: country, own capital 31% | no effect / no effect |

Named groups across scales. Scale = the smallest single-linkage height at which the group shares one component. Size = that component's size; it is a cluster of exactly the group only when size = group size. Gap = that component's distance to the rest. Bootstrap = the share of resamples in which the component at the resample's own connecting scale is exactly the group.

| group | scale | component size | gap | gap / scale | bootstrap exact |
|---|---|---|---|---|---|
| same-country (4) | 1.17 | 143 | 1.42 | 1.21 | 0% |
| same-country trio L0H2, L4H10, L3H0 (post hoc) | 0.854 | 3 | 1.17 | 1.37 | 100% |
| late any-distance (4) | 0.574 | 139 | 0.59 | 1.03 | 0% |
| Exp 1 previous-token without L4H11 (10) | 0.344 | 134 | 0.376 | 1.09 | 0% |

## Caveats

- **144 points is a small cloud.**
  - Single linkage (H0) sees gaps but not density. Modes inside the large component, or a dense core with a thin bridge, are invisible to it.
  - H1 has low power. A loop needs several heads spread around it, so 'no persistent H1' does not rule out a family sampled by only a few heads.
  - Every quantity is conditional on these 144 heads. The bootstrap resamples rows (measurement noise), not heads.
- **The metric decides what 'close' means.**
  - raw is dominated by the few heads with large effects (L0H2's own-capital curve is the largest).
  - corr and std amplify noise in near-zero heads and features.
- **The fixed-ε* cut is the finest partition supported above noise.** Coarser groupings exist higher up the dendrogram. The six-cluster cuts in the ARI table are descriptive only.
- **The labels are per filler type and use prev_score**, which the profile lacks (hence the merged-label ARI). The labels themselves agree across filler types only at the reference ARI above.
- **The Fasy band assumes the row bootstrap approximates the sampling law of the profiles.** That is reasonable for 200 exchangeable rows. The capital arrays are float16 (rebuild error ≤ 3e-5, far below every threshold).

## Files

- `tda1.npz`: the main profiles [144, 248]; per variant, the distance matrix, diagrams, linkage, partition, MDS coordinates, null statistics and bootstrap bottleneck distances and ARIs.
- `tda1_summary.json`: thresholds, verdicts and library versions.
- `tda1_clusters.csv`: every component per variant, with its members and labels.
- `figures/persistence.png`: H0 deaths by rank and the H1 diagram per variant, with the bootstrap band and both nulls.
- `figures/dendrogram_{raw,corr,std}.png`: single linkage with the ε* cut.
- `figures/mds_{raw,corr,std}.png`: 2-D metric MDS, coloured by component and marked by label.
