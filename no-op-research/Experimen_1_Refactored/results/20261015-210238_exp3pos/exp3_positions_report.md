# Experiment 3, class 2: position sweep of the country → capital pair

Run `20261015-210238_exp3pos`. Country at i, capital at j, every 1 ≤ i < j ≤ 32 (496 placements); 50 rows per base class; per placement and row 11 correct, 22 wrong (2 derangements) and 11 placebo cells. T = mean(correct) − mean(wrong); late = layers 6–11. Significant = BH q ≤ 0.05 over placements and |T| ≥ 0.005.

## Checks

| check | result |
|---|---|
| design (ids at i/j, base elsewhere, no repeats/BOS/stimulus ids) | PASS (asserted per placement) |
| attention rows sum to 1 | PASS (atol 0.0001, asserted per batch) |
| determinism, rare | PASS (max |Δ| 0.0e+00) |
| TL vs HF, rare | PASS (max |Δ| 4.3e-05) |
| determinism, common | PASS (max |Δ| 0.0e+00) |
| TL vs HF, common | PASS (max |Δ| 1.9e-05) |
| consistency rare (30, 31) (report only) | late BOS T -0.0007 |
| consistency rare (31, 32) (report only) | late BOS T -0.0154 |
| consistency common (30, 31) (report only) | late BOS T -0.0032 |
| consistency common (31, 32) (report only) | late BOS T -0.0331 |

## Pre-registered predictions

| id | verdict | detail |
|---|---|---|
| S1 | PASS | rare: i=26 -0.0211 (hi -0.0192), i=27 -0.0210 (hi -0.0188), i=28 -0.0178 (hi -0.0158), i=29 -0.0185 (hi -0.0165), i=30 -0.0167 (hi -0.0146), i=31 -0.0154 (hi -0.0131); common: i=26 -0.0243 (hi -0.0220), i=27 -0.0231 (hi -0.0201), i=28 -0.0240 (hi -0.0211), i=29 -0.0229 (hi -0.0204), i=30 -0.0223 (hi -0.0195), i=31 -0.0331 (hi -0.0299) |
| S2 | FAIL | rare: T(near) − mean T(far) +0.0134 [+0.0111, +0.0156]; common: T(near) − mean T(far) -0.0016 [-0.0059, +0.0023] |
| S3 | FAIL | rare: 72/465 = 15.5%; common: 4/465 = 0.9% |
| S4 | PASS | rare: (27,28) +0.0023 (lo +0.0019), (28,29) +0.0027 (lo +0.0024), (29,30) +0.0035 (lo +0.0031), (30,31) +0.0064 (lo +0.0057); common: (27,28) +0.0045 (lo +0.0039), (28,29) +0.0057 (lo +0.0050), (29,30) +0.0067 (lo +0.0060), (30,31) +0.0122 (lo +0.0111) |
| S5 | PASS | rare: 0/496 = 0.0%; common: 0/496 = 0.0% |

## Summary by region of the triangle (late T, mean over placements)

| base | region | placements | BOS T | country T | capital T | significant on BOS | significant placebo |
|---|---|---|---|---|---|---|---|
| rare | capital at the final token (j = 32) | 31 | -0.0250 | +0.0005 | -0.0021 | 31 | 0 |
| rare | adjacent, j ≥ 28, j < 32 | 4 | -0.0025 | +0.0013 | -0.0024 | 0 | 0 |
| rare | adjacent, j < 28 | 26 | -0.0018 | +0.0001 | -0.0010 | 0 | 0 |
| rare | non-adjacent, j < 32 | 435 | -0.0035 | +0.0001 | -0.0002 | 72 | 0 |
| common | capital at the final token (j = 32) | 31 | -0.0288 | +0.0006 | -0.0060 | 31 | 0 |
| common | adjacent, j ≥ 28, j < 32 | 4 | -0.0035 | +0.0021 | -0.0052 | 0 | 0 |
| common | adjacent, j < 28 | 26 | -0.0022 | +0.0001 | -0.0020 | 0 | 0 |
| common | non-adjacent, j < 32 | 435 | -0.0031 | +0.0001 | -0.0021 | 4 | 0 |

## Largest late BOS effects

| base | (i, j) | T | 95% CI | q | meaningful heads |
|---|---|---|---|---|---|
| rare | (1, 32) | -0.0340 | [-0.0371, -0.0309] | 2.8e-09 | 73 |
| rare | (2, 32) | -0.0322 | [-0.0350, -0.0295] | 2.8e-09 | 73 |
| rare | (4, 32) | -0.0311 | [-0.0346, -0.0278] | 2.8e-09 | 72 |
| rare | (3, 32) | -0.0310 | [-0.0342, -0.0279] | 2.8e-09 | 70 |
| rare | (6, 32) | -0.0306 | [-0.0345, -0.0268] | 2.8e-09 | 67 |
| rare | (10, 32) | -0.0290 | [-0.0325, -0.0259] | 2.8e-09 | 64 |
| rare | (9, 32) | -0.0282 | [-0.0309, -0.0255] | 2.8e-09 | 64 |
| rare | (5, 32) | -0.0281 | [-0.0307, -0.0257] | 2.8e-09 | 71 |
| common | (1, 32) | -0.0385 | [-0.0436, -0.0334] | 1.4e-08 | 71 |
| common | (3, 32) | -0.0375 | [-0.0411, -0.0339] | 1.4e-08 | 69 |
| common | (2, 32) | -0.0349 | [-0.0389, -0.0311] | 1.4e-08 | 69 |
| common | (31, 32) | -0.0331 | [-0.0367, -0.0299] | 1.4e-08 | 73 |
| common | (4, 32) | -0.0328 | [-0.0364, -0.0293] | 1.4e-08 | 66 |
| common | (5, 32) | -0.0323 | [-0.0361, -0.0288] | 1.4e-08 | 68 |
| common | (11, 32) | -0.0307 | [-0.0346, -0.0273] | 1.4e-08 | 70 |
| common | (13, 32) | -0.0307 | [-0.0347, -0.0271] | 1.4e-08 | 67 |

## Figures

- `figures/triangle_T.png`: T on BOS, country and capital mass for every placement (dots = significant).
- `figures/triangle_placebo_BOS.png`: the same for the placebo contrast.
- `figures/profiles_BOS.png`: late BOS T along the adjacent diagonal and with the capital at 32 or 31.
- `figures/attention_patterns.png`: change in late attention by key position at selected placements.
- `figures/meaningful_heads.png`: heads with a meaningful BOS change per placement.
