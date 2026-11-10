# TDA analysis 3: 1-D persistence of each head's distance curve

Run `20261025-102102_exp4tda3`. 144 heads × 4 conditions × 2 fillers = 1152 curves of 31 points (d = 1..31). Saved arrays only: observed curves are the saved means of `results/20261024-000548_exp4ctrl/controlled.npz` and `results/20261024-150407_exp4cap/capital.npz`; per-row curves feed the null. Predictions: `exp4/PREDICTIONS_tda3.md` (written before any persistence was computed). Null: 2000 row-bootstrap mean-zero curves per head × condition × filler, seed 0; bars significant at the 99th percentile of the null and ≥ 0.02.

## Checks

| check | result |
|---|---|
| bars vs gudhi SimplexTree, continuous curves | 0 of 1000 curves differ |
| bars vs gudhi SimplexTree, curves with ties | 0 of 1000 curves differ |
| essential bar = range | True |
| endpoint extrema, hand-built [3, 1, 2] | True |
| max |recomputed row mean − saved mean| | random/common 0.0e+00; own/common 1.4e-05; wrong/common 1.4e-05; fact/common 1.8e-05; random/rare 0.0e+00; own/rare 1.3e-05; wrong/rare 2.7e-05; fact/rare 2.8e-05 |
| held-out null: share of fresh null curves with range > θ_ess (expect 5% / 1%) | 0.051 / 0.011 |
| held-out null: share with a non-essential bar > θ_non (expect 5% / 1%) | 0.051 / 0.010 |

## Verdicts on the pre-registered checks

| id | verdict | numbers | under the sensitivity settings |
|---|---|---|---|
| C1 | PASS | common: class plateau, f(31) +0.1830, f(30) +0.1448; peak at 31 pers 0.0408 vs θ 0.0170 (non-essential), sig True | rare: class U-shape, f(31) +0.0557, f(30) +0.0491; peak at 31 pers 0.0241 vs θ 0.0090 (non-essential), sig True | q95: PASS, no margin: PASS, rel 0.3: PASS, rel 0.7: PASS (stable) |
| C2 | PASS | random/common: near-peak decay (sign +1, d* 1, R 0.065, 1 peaks) | own/common: near-peak decay (sign +1, d* 1, R 0.151, 1 peaks) | wrong/common: near-peak decay (sign +1, d* 1, R 0.152, 1 peaks) | random/rare: near-peak decay (sign +1, d* 1, R 0.041, 1 peaks) | own/rare: near-peak decay (sign +1, d* 1, R 0.066, 1 peaks) | wrong/rare: near-peak decay (sign +1, d* 1, R 0.066, 1 peaks) | q95: PASS, no margin: PASS, rel 0.3: PASS, rel 0.7: PASS (stable) |
| C3 | PASS | common: sign +1, gate True, d* 30, R 0.272 (θ 0.016), class far rise, f(30) +0.410, f(31) +0.309 | rare: sign +1, gate True, d* 30, R 0.379 (θ 0.018), class far rise, f(30) +0.516, f(31) +0.384 | q95: PASS, no margin: PASS, rel 0.3: PASS, rel 0.7: PASS (stable) |
| C4 | PASS | common: plateau (R 0.076, θ 0.002, ρ 0.18, mean +0.376, 0 significant bars at 2 ≤ d ≤ 30) | rare: plateau (R 0.071, θ 0.004, ρ 0.19, mean +0.345, 0 significant bars at 2 ≤ d ≤ 30) | q95: PASS, no margin: PASS, rel 0.3: PASS, rel 0.7: PASS (stable) |
| C5 | PASS | 0 bars at d ≥ 9 among 66 curves; closest non-essential bar to its null threshold: L3H7 wrong/common peak@12: 0.0147 vs θ_non 0.0154 | q95: PASS, no margin: PASS, rel 0.3: PASS, rel 0.7: PASS (stable) |

Descriptive expectations (not pass/fail):

| id | held | numbers |
|---|---|---|
| E1 | yes | null or plateau in the random condition: 90 / 110 of 144 (common / rare) |
| E2 | yes | heads with ≥ 1 significant feature, own vs random: common 87 vs 57, rare 75 vs 35 |
| E3 | yes | L9H8 / L10H3 / L11H7 / L11H2, random: common: plateau, plateau, far rise, plateau; rare: U-shape, plateau, far rise, plateau |

## Shape-class counts (primary settings; each row sums to 144)

In the five feature classes, the number in brackets is how many of them have sign −, i.e. their extremum is a dip ("far rise" with sign − is attention that is lower than the control at the far end).

| condition | filler | null | plateau | near-peak decay | mid-peak | far rise | U-shape | multi-modal |
|---|---|---|---|---|---|---|---|---|
| random | common | 83 | 7 | 50 (10) | 0 (0) | 3 (3) | 0 (0) | 1 (0) |
| random | rare | 108 | 2 | 32 (9) | 0 (0) | 1 (1) | 1 (0) | 0 (0) |
| own | common | 47 | 16 | 73 (25) | 0 (0) | 7 (5) | 0 (0) | 1 (0) |
| own | rare | 66 | 8 | 65 (30) | 3 (3) | 2 (0) | 0 (0) | 0 (0) |
| wrong | common | 59 | 16 | 58 (18) | 0 (0) | 7 (7) | 1 (1) | 3 (0) |
| wrong | rare | 75 | 7 | 57 (23) | 2 (2) | 3 (2) | 0 (0) | 0 (0) |
| fact | common | 95 | 16 | 31 (16) | 0 (0) | 2 (0) | 0 (0) | 0 (0) |
| fact | rare | 106 | 8 | 27 (16) | 0 (0) | 3 (0) | 0 (0) | 0 (0) |

### Heads with a mid-peak, far-rise, U-shape or multi-modal class in at least one filler

| condition | head | common | rare |
|---|---|---|---|
| random | L5H2 | multi-modal, d* 1 | near-peak decay (dip), d* 1 |
| random | L7H7 | far rise (dip), d* 31 | near-peak decay, d* 2 |
| random | L9H8 | plateau, d* 1 | U-shape, d* 1 |
| random | L11H1 | far rise (dip), d* 31 | null, d* 30 |
| random | L11H7 | far rise (dip), d* 31 | far rise (dip), d* 31 |
| own | L0H5 | far rise, d* 31 | far rise, d* 31 |
| own | L3H0 | far rise, d* 30 | far rise, d* 30 |
| own | L6H8 | multi-modal, d* 1 | near-peak decay, d* 1 |
| own | L6H10 | far rise (dip), d* 31 | null, d* 30 |
| own | L7H7 | far rise (dip), d* 30 | null, d* 10 |
| own | L8H0 | far rise (dip), d* 31 | null, d* 2 |
| own | L10H1 | far rise (dip), d* 31 | mid-peak (dip), d* 4 |
| own | L10H4 | near-peak decay, d* 1 | mid-peak (dip), d* 5 |
| own | L10H11 | near-peak decay, d* 1 | mid-peak (dip), d* 4 |
| own | L11H9 | far rise (dip), d* 31 | plateau, d* 27 |
| wrong | L2H2 | multi-modal, d* 1 | near-peak decay (dip), d* 2 |
| wrong | L3H0 | null, d* 28 | far rise, d* 30 |
| wrong | L4H8 | plateau, d* 30 | far rise (dip), d* 30 |
| wrong | L5H2 | multi-modal, d* 1 | near-peak decay (dip), d* 1 |
| wrong | L6H8 | multi-modal, d* 1 | near-peak decay, d* 1 |
| wrong | L6H10 | far rise (dip), d* 30 | far rise (dip), d* 30 |
| wrong | L7H6 | far rise (dip), d* 31 | near-peak decay, d* 1 |
| wrong | L7H7 | U-shape (dip), d* 30 | null, d* 29 |
| wrong | L10H1 | far rise (dip), d* 31 | null, d* 4 |
| wrong | L10H4 | near-peak decay, d* 1 | mid-peak (dip), d* 5 |
| wrong | L10H11 | near-peak decay, d* 1 | mid-peak (dip), d* 4 |
| wrong | L11H1 | far rise (dip), d* 31 | null, d* 1 |
| wrong | L11H5 | far rise (dip), d* 31 | null, d* 14 |
| wrong | L11H7 | far rise (dip), d* 31 | near-peak decay, d* 1 |
| wrong | L11H9 | far rise (dip), d* 31 | plateau, d* 27 |
| fact | L0H5 | far rise, d* 31 | far rise, d* 31 |
| fact | L3H0 | far rise, d* 30 | far rise, d* 30 |
| fact | L4H8 | plateau, d* 31 | far rise, d* 31 |

### Sensitivity of the counts (all conditions and fillers pooled, 1152 curves)

| setting | null | plateau | near-peak decay | mid-peak | far rise | U-shape | multi-modal |
|---|---|---|---|---|---|---|---|
| primary | 639 | 80 | 393 | 5 | 28 | 2 | 5 |
| q95 | 633 | 80 | 399 | 5 | 28 | 2 | 5 |
| no margin | 240 | 76 | 570 | 16 | 86 | 142 | 22 |
| rel 0.3 | 639 | 57 | 406 | 5 | 37 | 3 | 5 |
| rel 0.7 | 640 | 109 | 374 | 5 | 19 | 0 | 5 |

Thresholds (median over the 144 heads, and the share of heads whose θ_ess is above the 0.02 margin, i.e. where the null rather than the margin sets the cut):

| condition | filler | θ_ess q99 | θ_non q99 | θ_ess q95 | θ_non q95 | share θ_ess q99 > 0.02 |
|---|---|---|---|---|---|---|
| random | common | 0.0117 | 0.0089 | 0.0098 | 0.0073 | 0.28 |
| random | rare | 0.0107 | 0.0077 | 0.0088 | 0.0062 | 0.19 |
| own | common | 0.0088 | 0.0068 | 0.0075 | 0.0058 | 0.19 |
| own | rare | 0.0097 | 0.0067 | 0.0082 | 0.0054 | 0.18 |
| wrong | common | 0.0084 | 0.0065 | 0.0073 | 0.0055 | 0.20 |
| wrong | rare | 0.0092 | 0.0062 | 0.0077 | 0.0051 | 0.17 |
| fact | common | 0.0016 | 0.0011 | 0.0013 | 0.0009 | 0.00 |
| fact | rare | 0.0013 | 0.0008 | 0.0010 | 0.0006 | 0.01 |

## Ranking: curves with the largest significant range

R is the range of the curve (the persistence of its dominant extremum); d* is the location of that extremum. Both fillers must pass the gate to be listed.

**random** (heads whose curve passes the gate in both fillers: 26; ranked by the smaller of the two ranges)

| head | R common | R rare | class common | class rare | d* c / r | features c / r |
|---|---|---|---|---|---|---|
| L3H6 | 0.244 | 0.144 | near-peak decay | near-peak decay | 1 / 1 | 1 / 1 |
| L2H8 | 0.153 | 0.117 | near-peak decay | near-peak decay | 1 / 1 | 1 / 1 |
| L9H3 | 0.163 | 0.104 | near-peak decay | near-peak decay | 1 / 1 | 1 / 1 |
| L3H1 | 0.104 | 0.166 | near-peak decay | near-peak decay | 1 / 1 | 1 / 1 |
| L2H9 | 0.171 | 0.089 | near-peak decay | near-peak decay | 1 / 1 | 1 / 1 |
| L6H8 | 0.093 | 0.079 | near-peak decay | near-peak decay | 2 / 1 | 2 / 1 |
| L7H0 | 0.162 | 0.078 | near-peak decay | near-peak decay | 1 / 1 | 1 / 1 |
| L2H3 | 0.157 | 0.071 | near-peak decay | near-peak decay | 1 / 1 | 1 / 1 |
| L5H3 | 0.091 | 0.071 | near-peak decay | near-peak decay | 1 / 1 | 1 / 1 |
| L2H5 | 0.093 | 0.069 | near-peak decay | near-peak decay | 1 / 1 | 1 / 1 |

**own** (heads whose curve passes the gate in both fillers: 68; ranked by the smaller of the two ranges)

| head | R common | R rare | class common | class rare | d* c / r | features c / r |
|---|---|---|---|---|---|---|
| L3H0 | 0.272 | 0.379 | far rise | far rise | 30 / 30 | 2 / 2 |
| L3H7 | 0.262 | 0.316 | near-peak decay | near-peak decay | 1 / 1 | 1 / 1 |
| L5H6 | 0.208 | 0.199 | near-peak decay | near-peak decay | 1 / 1 | 1 / 1 |
| L2H8 | 0.183 | 0.197 | near-peak decay | near-peak decay | 1 / 1 | 1 / 1 |
| L2H2 | 0.161 | 0.198 | near-peak decay | near-peak decay | 1 / 1 | 1 / 1 |
| L4H7 | 0.258 | 0.158 | near-peak decay | near-peak decay | 1 / 1 | 1 / 1 |
| L2H5 | 0.154 | 0.178 | near-peak decay | near-peak decay | 1 / 1 | 1 / 1 |
| L7H0 | 0.164 | 0.122 | near-peak decay | near-peak decay | 1 / 1 | 1 / 1 |
| L5H4 | 0.136 | 0.122 | near-peak decay | near-peak decay | 1 / 1 | 1 / 1 |
| L3H8 | 0.170 | 0.119 | near-peak decay | near-peak decay | 1 / 1 | 1 / 1 |

**wrong** (heads whose curve passes the gate in both fillers: 58; ranked by the smaller of the two ranges)

| head | R common | R rare | class common | class rare | d* c / r | features c / r |
|---|---|---|---|---|---|---|
| L2H8 | 0.154 | 0.170 | near-peak decay | near-peak decay | 1 / 1 | 1 / 1 |
| L5H4 | 0.122 | 0.124 | near-peak decay | near-peak decay | 1 / 1 | 1 / 1 |
| L4H7 | 0.183 | 0.115 | near-peak decay | near-peak decay | 1 / 1 | 1 / 1 |
| L5H6 | 0.149 | 0.099 | near-peak decay | near-peak decay | 1 / 1 | 1 / 1 |
| L3H10 | 0.138 | 0.086 | near-peak decay | near-peak decay | 1 / 1 | 1 / 1 |
| L0H7 | 0.086 | 0.086 | near-peak decay | near-peak decay | 2 / 1 | 1 / 1 |
| L6H8 | 0.085 | 0.086 | multi-modal | near-peak decay | 1 / 1 | 2 / 1 |
| L1H0 | 0.136 | 0.076 | near-peak decay | near-peak decay | 1 / 1 | 1 / 1 |
| L4H0 | 0.141 | 0.075 | near-peak decay | near-peak decay | 1 / 1 | 1 / 1 |
| L3H2 | 0.075 | 0.086 | near-peak decay | near-peak decay | 1 / 1 | 1 / 1 |

**fact** (heads whose curve passes the gate in both fillers: 31; ranked by the smaller of the two ranges)

| head | R common | R rare | class common | class rare | d* c / r | features c / r |
|---|---|---|---|---|---|---|
| L3H7 | 0.265 | 0.265 | near-peak decay | near-peak decay | 1 / 1 | 1 / 1 |
| L3H0 | 0.258 | 0.345 | far rise | far rise | 30 / 30 | 2 / 2 |
| L2H2 | 0.187 | 0.180 | near-peak decay | near-peak decay | 1 / 1 | 1 / 1 |
| L2H9 | 0.165 | 0.152 | near-peak decay | near-peak decay | 1 / 1 | 1 / 1 |
| L2H5 | 0.125 | 0.116 | near-peak decay | near-peak decay | 1 / 1 | 1 / 1 |
| L1H11 | 0.166 | 0.096 | near-peak decay | near-peak decay | 1 / 1 | 1 / 1 |
| L0H4 | 0.082 | 0.079 | near-peak decay | near-peak decay | 1 / 1 | 1 / 1 |
| L3H8 | 0.076 | 0.083 | near-peak decay | near-peak decay | 1 / 1 | 1 / 1 |
| L0H2 | 0.073 | 0.071 | plateau | plateau | 1 / 1 | 1 / 1 |
| L10H7 | 0.092 | 0.071 | near-peak decay | near-peak decay | 1 / 1 | 1 / 1 |

## Caveats

- Each curve has 31 points, so features closer than a few points are barely resolvable and an interior peak needs support from both sides; "interior" means 2 ≤ d ≤ 30.
- The null is a bootstrap of the mean curve with its shape removed. It assumes exchangeable rows and gives each head the noise level it actually has, which is highest where the effect is, so it is conservative for strong heads. It does not model the 11 fixed country/capital pairs: they are averaged inside each row, so pair-to-pair variation enters the row noise and is not independent sampling.
- The essential bar has persistence equal to the range, so every curve with variation has an essential peak and an essential trough of equal size. The one on the side away from the median is treated as the baseline, not a feature. That choice is mine, stated before the run.
- For most heads the 0.02 margin, not the bootstrap null, sets the cut (see the share column above). Without the margin the null class shrinks and many small end-point wiggles become U-shapes (sensitivity table), so "noise-robust" here means robust to sampling noise of the mean curve plus a size floor chosen by the repo's earlier convention, not by this null.
- A class is about the mean curve's shape, not about attention magnitude: L0H2 with its own capital is a plateau because its slide from d = 1 to d = 31 is under half its height (ρ < 0.5), although that slide is many times its null threshold. The ρ rule was fixed before the run; the rel 0.3 / 0.7 rows show its influence.
- At the 99th percentile roughly 1% of the 1152 curves (about 11) would pass the gate on noise alone; the held-out calibration above checks the rate on the null, not on real data, so any single borderline curve is uncertain.
- The cut-offs 0.02, ρ = 0.5 and the zones 3 and 25 were fixed beforehand; the sensitivity rows show how much the counts move. The capital runs were stored as float16; the recomputed row means match the saved means to the precision shown in the checks, and only the null uses the per-row arrays.
