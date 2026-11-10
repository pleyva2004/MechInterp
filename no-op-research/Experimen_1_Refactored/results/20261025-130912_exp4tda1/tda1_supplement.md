# TDA 1 supplement

Supplement to the final run `20261025-114921_exp4tda1` (reproduced bit-for-bit as `20261025-125719_exp4tda1`). Script: `tda1_supplement.py` in this folder. Sections B are pre-registered items the run reported only partly; sections C are sensitivity checks that were **not pre-registered**.

## A. Independent cross-check of the raw path

| check | value |
|---|---|
| max|P(saved means) - P(run)| | 2.68e-05 |
| max|d_raw - run d_raw| | 0 |
| ARI(independent ε* partition, run partition) | 1 |
| ARI vs labels, common (independent) | 0.123 |
| ARI vs labels, rare (independent) | 0.209 |

Run values for comparison: ARI common +0.123, rare +0.209. The profile difference is the float16 storage of Exp 4b's per-row arrays (pre-registered tolerance 1e-4).

## B1. ARI with labels 5 + 6 merged, with 95% bootstrap interval (pre-registered: an interval for each ARI)

| variant | filler | ARI (labels as is) [95%] | ARI (5+6 merged) [95%] |
|---|---|---|---|
| raw | common | +0.123 [+0.083, +0.138] | +0.109 [+0.066, +0.125] |
| raw | rare | +0.209 [+0.193, +0.247] | +0.198 [+0.187, +0.242] |
| corr | common | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| corr | rare | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| std | common | +0.055 [+0.015, +0.063] | +0.079 [+0.036, +0.091] |
| std | rare | +0.048 [+0.016, +0.094] | +0.087 [+0.053, +0.140] |

## B2. H0 bars beyond N2, H1 features beyond N1 (pre-registered 'under the other null')

**raw.** N2 q95 of the largest H0 death = 0.6092; 4 bar(s) beyond it:

- merge at 0.8412: L0H2 joins a 1-head group
- merge at 0.8543: L3H0 joins a 2-head group
- merge at 1.169: L3H0, L0H2, L4H10 joins a 140-head group
- merge at 1.42: L1H5 joins a 143-head group

H1 features with death/birth > N1 q95 (1.179): birth 0.1542, death 0.1874, ratio 1.215, persistence 0.03314 (inside the 2·c1 = 0.03559 band); birth 0.09829, death 0.1166, ratio 1.186, persistence 0.0183 (inside the 2·c1 = 0.03559 band).

**corr.** N2 q95 of the largest H0 death = 0.5684; 0 bar(s) beyond it:

- none

H1 features with death/birth > N1 q95 (1.676): none.

**std.** N2 q95 of the largest H0 death = 21.01; 7 bar(s) beyond it:

- merge at 22.84: L0H2 joins a 2-head group
- merge at 23.99: L11H2, L11H7 joins a 135-head group
- merge at 24.14: L3H4 joins a 137-head group
- merge at 24.97: L0H2, L3H0, L4H10 joins a 138-head group
- merge at 25.39: L10H3 joins a 141-head group
- merge at 29.49: L9H8 joins a 142-head group
- merge at 43.62: L1H5 joins a 143-head group

H1 features with death/birth > N1 q95 (1.138): birth 8.143, death 10.61, ratio 1.303, persistence 2.466 (beyond the 2·c1 = 2.432 band); birth 3.729, death 4.317, ratio 1.158, persistence 0.5885 (inside the 2·c1 = 2.432 band); birth 2.162, death 2.484, ratio 1.148, persistence 0.3211 (inside the 2·c1 = 2.432 band).

## C1. Monte Carlo error of the thresholds (not pre-registered)

95% intervals of each 95th percentile from resampling its 500 draws (or 500 bootstrap bottleneck distances).

| variant | ε* = N1 q95 max H0 death [MC 95%] | 2·c0 | 2·c1 [MC 95%] | N2 q95 max H1 ratio [MC 95%] | top H1 ratio | top-ratio feature persistence |
|---|---|---|---|---|---|---|
| raw | 0.1894 [0.1777, 0.1997] | 0.09905 | 0.03559 [0.03364, 0.03701] | 1.383 [1.369, 1.396] | 1.215 | 0.03314 |
| corr | 0.7993 [0.7899, 0.8062] | 0.1965 | 0.2102 [0.1981, 0.2292] | 2.722 [2.588, 2.904] | 1.632 | 0.2976 |
| std | 8.094 [7.731, 8.394] | 7.373 | 2.432 [2.213, 2.466] | 1.323 [1.311, 1.327] | 1.303 | 2.466 |

T1-a (≥ 3 components of ≥ 2 heads with recurrence ≥ 0.75) re-evaluated across the Monte Carlo interval of ε* (each point re-cuts the main and all 500 resample dendrograms):

| variant | ε | components | multi-head components: size (recurrence) | stable | T1-a | ARI common / rare |
|---|---|---|---|---|---|---|
| raw | 0.1777 | 34 | 108 (0.99), 2 (0.87), 2 (0.80), 2 (0.74) | 3 | PASS | +0.137 / +0.242 |
| raw | 0.1813 | 34 | 108 (0.99), 2 (0.77), 2 (0.85), 2 (0.60) | 3 | PASS | +0.137 / +0.242 |
| raw | 0.185 | 32 | 110 (0.98), 3 (0.60), 2 (0.89) | 2 | FAIL | +0.123 / +0.209 |
| raw | 0.1887 | 32 | 110 (0.98), 3 (0.56), 2 (0.92) | 2 | FAIL | +0.123 / +0.209 |
| raw | 0.1894 (ε*) | 32 | 110 (0.98), 3 (0.55), 2 (0.92) | 2 | FAIL | +0.123 / +0.209 |
| raw | 0.1924 | 32 | 110 (0.98), 3 (0.46), 2 (0.94) | 2 | FAIL | +0.123 / +0.209 |
| raw | 0.196 | 31 | 113 (0.99), 2 (0.95) | 2 | FAIL | +0.092 / +0.214 |
| raw | 0.1997 | 31 | 113 (0.99), 2 (0.95) | 2 | FAIL | +0.092 / +0.214 |
| std | 7.731 | 37 | 108 (0.95) | 1 | FAIL | +0.052 / +0.054 |
| std | 7.841 | 37 | 108 (0.96) | 1 | FAIL | +0.052 / +0.054 |
| std | 7.952 | 37 | 108 (0.96) | 1 | FAIL | +0.052 / +0.054 |
| std | 8.062 | 36 | 109 (0.96) | 1 | FAIL | +0.047 / +0.039 |
| std | 8.094 (ε*) | 35 | 110 (0.96) | 1 | FAIL | +0.055 / +0.048 |
| std | 8.173 | 34 | 111 (0.96) | 1 | FAIL | +0.050 / +0.033 |
| std | 8.283 | 33 | 112 (0.96) | 1 | FAIL | +0.039 / +0.042 |
| std | 8.394 | 33 | 112 (0.96) | 1 | FAIL | +0.039 / +0.042 |

### C1b. Raw ε* under other N1 seeds (not pre-registered)

500 sign-flip draws per seed; the run used seed 1. T1-a passes by the letter only for ε below 0.1814, the merge at which L0H7 joins L2H0 + L2H4.

| N1 seed | ε* | multi-head components: members (recurrence) | stable | T1-a |
|---|---|---|---|---|
| 1 | 0.1894 | 110 heads (0.98); L0H7, L2H0, L2H4 (0.55); L2H5, L3H8 (0.92) | 2 | FAIL |
| 11 | 0.1757 | 108 heads (0.99); L2H0, L2H4 (0.91); L2H5, L3H8 (0.77); L5H3, L9H3 (0.80) | 4 | PASS |
| 21 | 0.1975 | 113 heads (0.99); L2H5, L3H8 (0.95) | 2 | FAIL |
| 31 | 0.1825 | 108 heads (0.98); L0H7, L2H0, L2H4 (0.64); L2H5, L3H8 (0.86); L5H3, L9H3 (0.54) | 2 | FAIL |
| 41 | 0.1852 | 110 heads (0.98); L0H7, L2H0, L2H4 (0.60); L2H5, L3H8 (0.89) | 2 | FAIL |
| 51 | 0.1798 | 108 heads (0.99); L2H0, L2H4 (0.82); L2H5, L3H8 (0.82); L5H3, L9H3 (0.65) | 3 | PASS |

## C2. std T1-d under the other reading of the pre-registration (not pre-registered)

The run measured every std cloud with the main profiles' SDs (its disclosed clarification) and reported the self-standardised alternative only for N1's ε*. Here N2 and the bootstrap are self-standardised too.

- N2 q95 of the largest H1 ratio: 1.327 (MC 95% 1.319–1.335); main-SD reading 1.323.
- 2·c1: 2.169; main-SD reading 2.432.
- Top std H1 feature: ratio 1.303, persistence 2.466.
- T1-d under std with this reading: **PASS** (0 feature(s) beyond both).

## C3. Named groups under the robustness metrics (exploratory)

- **corr, same-country (4)** connect at 0.09593 in a 11-head component (gap to rest 0.1044): L0H1, L0H2, L0H5, L0H10, L1H6, L1H11, L3H0, L4H4, L4H10, L5H11, L6H6. Exp 1 previous-token heads in it: none.
- **corr, late any-distance (4)** connect at 0.204 in a 110-head component (gap to rest 0.2049). Exp 1 previous-token heads in it: L1H0, L2H3, L2H8, L2H9, L3H2, L3H6.
- **std, same-country (4)** connect at 24.97 in a 141-head component (gap to rest 25.39). Exp 1 previous-token heads in it: L1H0, L2H2, L2H3, L2H5, L2H8, L2H9, L3H2, L3H6, L3H7, L3H8, L4H11.
- **std, late any-distance (4)** connect at 29.49 in a 143-head component (gap to rest 43.62). Exp 1 previous-token heads in it: L1H0, L2H2, L2H3, L2H5, L2H8, L2H9, L3H2, L3H6, L3H7, L3H8, L4H11.
- **Largest component, raw vs std** (110 heads each): 95 shared. raw only: L0H6, L10H11, L10H4, L10H6, L11H0, L11H1, L11H4, L11H6, L11H9, L1H7, L2H1, L5H3, L8H11, L8H6, L9H11. std only: L0H10, L0H4, L0H7, L1H0, L2H0, L2H2, L2H4, L2H5, L2H9, L3H2, L3H7, L3H8, L4H4, L5H6, L7H0.
- **corr cut.** N1's smallest largest-H0-death over 500 draws is 0.6296, above the real data's largest H0 death 0.5668: under correlation distance, pure-noise profiles are less correlated with each other than any real pair of heads, so ε* is above every bar by construction and the corr T1-a/b/c verdicts are uninformative rather than independent confirmations.
