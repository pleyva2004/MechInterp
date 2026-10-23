# Experiment 3, class 2, R1-in: country → capital / city AT the final token

Run `20261015-203150_exp3rel_in`. TL attention (HF cross-check on 50 rows per base class). 200 rows per base class; per row 2 × 121 grid cells, none, 11 anchor and 11 capital-alone cells. Layout: country at 31, capital / city / anchor word at 32 (the query).

T = mean(correct cells) − mean(wrong cells): each capital (the query token) sits once in a correct cell and 10 times in wrong ones, so its own query effect cancels. Late = layers 6–11. Categories: 'country' = 31 (previous token), 'partner' = 32 (the query itself), 'self' is empty.

## Checks

| check | result |
|---|---|
| design (grid and extra cells; no repeats/BOS/stimulus ids in bases) | PASS (asserted per row) |
| attention rows sum to 1 | PASS (atol 0.0001, asserted per batch) |
| determinism, rare (5 rows) | PASS (max |Δ| 0.0e+00) |
| TL vs HF, rare (50 rows) | PASS (max |Δ| 5.0e-05, argmax-key agreement 99.9999%) |
| manipulation, rare | PASS (11/11 countries prefer their capital) |
| determinism, common (5 rows) | PASS (max |Δ| 0.0e+00) |
| TL vs HF, common (50 rows) | PASS (max |Δ| 2.9e-05, argmax-key agreement 99.9999%) |
| manipulation, common | PASS (11/11 countries prefer their capital) |

## Manipulation check (log p at the country for the query position, in context)

| base | partner | own − mean(other 10) | countries > 0 | own ranked 1st |
|---|---|---|---|---|
| rare | capital | +2.36 | 11/11 | 79% |
| rare | city | +2.15 | 11/11 | 72% |
| common | capital | +2.70 | 11/11 | 82% |
| common | city | +2.51 | 11/11 | 86% |

## Pre-registered predictions

| id | verdict | detail |
|---|---|---|
| Rin-1 | PASS | rare: negative -0.0180 [-0.0194, -0.0167], perm p< 0.0001; common: negative -0.0338 [-0.0354, -0.0321], perm p< 0.0001 |
| Rin-2 | FAIL | rare: country +0.0024 [+0.0020, +0.0028] vs |BOS| 0.0180; query +0.0003, filler +0.0153; common: country +0.0063 [+0.0058, +0.0068] vs |BOS| 0.0338; query -0.0015, filler +0.0290 |
| Rin-3 | FAIL | rare: 3 Sink→Country switches, placebo mean 0.45; common: 4 Sink→Country switches, placebo mean 0.70 |
| Rin-4 | FAIL | rare: equivalent -0.0026 [-0.0038, -0.0015], perm p< 0.1340; common: negative -0.0179 [-0.0192, -0.0164], perm p< 0.0001 |
| Rin-5 | PASS | rare: -0.00100 [-0.00129, -0.00072] per nat; common: -0.00249 [-0.00286, -0.00214] per nat |
| Rin-6 | PASS | rare: 11/11; common: 11/11 |
| Rin-7 | FAIL | rare: rho -0.282 [-0.362, -0.190]; common: rho -0.227 [-0.325, -0.123] |

## Pooled late-layer T

| base | grid | metric | T | 95% CI | 90% CI | perm p (>) | perm p (<) | verdict |
|---|---|---|---|---|---|---|---|---|
| rare | capital | BOS | -0.0180 | [-0.0194, -0.0167] | [-0.0191, -0.0169] | 1.0000 | 0.0001 | negative |
| rare | capital | country | +0.0024 | [+0.0020, +0.0028] | [+0.0021, +0.0027] | 0.0001 | 1.0000 | positive |
| rare | capital | partner | +0.0003 | [-0.0005, +0.0011] | [-0.0004, +0.0010] | 0.4179 | 0.5822 | equivalent |
| rare | capital | self | +0.0000 | [+0.0000, +0.0000] | [+0.0000, +0.0000] | 1.0000 | 1.0000 | equivalent |
| rare | capital | filler | +0.0153 | [+0.0138, +0.0169] | [+0.0140, +0.0166] | 0.0001 | 1.0000 | positive |
| rare | city | BOS | -0.0154 | [-0.0163, -0.0144] | [-0.0161, -0.0146] | 1.0000 | 0.0001 | negative |
| rare | city | country | +0.0029 | [+0.0026, +0.0031] | [+0.0026, +0.0031] | 0.0001 | 1.0000 | positive |
| rare | city | partner | +0.0046 | [+0.0042, +0.0051] | [+0.0042, +0.0050] | 0.0069 | 0.9932 | positive |
| rare | city | self | +0.0000 | [+0.0000, +0.0000] | [+0.0000, +0.0000] | 1.0000 | 1.0000 | equivalent |
| rare | city | filler | +0.0079 | [+0.0068, +0.0089] | [+0.0070, +0.0088] | 0.0021 | 0.9980 | positive |
| common | capital | BOS | -0.0338 | [-0.0354, -0.0321] | [-0.0351, -0.0324] | 1.0000 | 0.0001 | negative |
| common | capital | country | +0.0063 | [+0.0058, +0.0068] | [+0.0059, +0.0067] | 0.0001 | 1.0000 | positive |
| common | capital | partner | -0.0015 | [-0.0023, -0.0008] | [-0.0022, -0.0009] | 0.8111 | 0.1890 | equivalent |
| common | capital | self | +0.0000 | [+0.0000, +0.0000] | [+0.0000, +0.0000] | 1.0000 | 1.0000 | equivalent |
| common | capital | filler | +0.0290 | [+0.0276, +0.0303] | [+0.0279, +0.0301] | 0.0001 | 1.0000 | positive |
| common | city | BOS | -0.0159 | [-0.0171, -0.0148] | [-0.0169, -0.0149] | 1.0000 | 0.0001 | negative |
| common | city | country | +0.0035 | [+0.0031, +0.0039] | [+0.0032, +0.0038] | 0.0001 | 1.0000 | positive |
| common | city | partner | +0.0047 | [+0.0042, +0.0053] | [+0.0043, +0.0052] | 0.0104 | 0.9897 | positive |
| common | city | self | +0.0000 | [+0.0000, +0.0000] | [+0.0000, +0.0000] | 1.0000 | 1.0000 | equivalent |
| common | city | filler | +0.0077 | [+0.0067, +0.0087] | [+0.0068, +0.0086] | 0.0001 | 1.0000 | positive |

## Capital − city, dose, anchor (late BOS)

| base | quantity | value | verdict |
|---|---|---|---|
| rare | T capital − T city | -0.0026 [-0.0038, -0.0015], perm p< 0.1340 | equivalent |
| rare | Spearman(surprisal of own capital, late BOS) within correct cells | -0.282 [-0.362, -0.190] | |
| rare | correct − anchor | -0.1438 [-0.1486, -0.1390] | negative |
| rare | wrong − anchor | -0.1258 [-0.1304, -0.1210] | negative |
| rare | capital alone − anchor | -0.1280 [-0.1341, -0.1222] | negative |
| rare | placebo T (capital), 20 derangements | [-0.0021, +0.0035], 0 with p < 0.05; mean switches 0.45 | |
| rare | slope of late BOS on log p(capital given country), wrong cells | -0.00100 [-0.00129, -0.00072] per nat | |
| rare | placebo T (city), 20 derangements | [-0.0023, +0.0050], 1 with p < 0.05; mean switches 0.65 | |
| rare | slope of late BOS on log p(city given country), wrong cells | -0.00279 [-0.00311, -0.00248] per nat | |
| common | T capital − T city | -0.0179 [-0.0192, -0.0164], perm p< 0.0001 | negative |
| common | Spearman(surprisal of own capital, late BOS) within correct cells | -0.227 [-0.325, -0.123] | |
| common | correct − anchor | -0.1144 [-0.1185, -0.1102] | negative |
| common | wrong − anchor | -0.0806 [-0.0842, -0.0769] | negative |
| common | capital alone − anchor | -0.0927 [-0.0981, -0.0875] | negative |
| common | placebo T (capital), 20 derangements | [-0.0027, +0.0075], 2 with p < 0.05; mean switches 0.70 | |
| common | slope of late BOS on log p(capital given country), wrong cells | -0.00249 [-0.00286, -0.00214] per nat | |
| common | placebo T (city), 20 derangements | [-0.0037, +0.0075], 1 with p < 0.05; mean switches 0.75 | |
| common | slope of late BOS on log p(city given country), wrong cells | +0.00134 [+0.00099, +0.00168] per nat | |

## Per country (late BOS, diagonal residual, × 10⁻³)

| base | grid | France | Germany | Italy | Spain | Russia | Ireland | Ukraine | Japan | China | Egypt | Canada | < 0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rare | capital | -14.4 | -9.5 | -4.9 | -17.1 | -35.4 | -19.9 | -19.0 | -15.1 | -14.6 | -21.6 | -8.7 | 11/11 |
| rare | city | -20.4 | -8.8 | -18.2 | -5.4 | -26.4 | -50.8 | +4.7 | -1.6 | -13.3 | -9.6 | -4.0 | 10/11 |
| common | capital | -24.2 | -31.2 | -15.6 | -28.8 | -46.2 | -39.6 | -34.7 | -27.0 | -33.2 | -31.8 | -25.8 | 11/11 |
| common | city | -21.4 | -16.2 | -24.0 | -17.1 | -16.5 | -32.2 | +7.3 | -2.5 | -18.0 | -8.8 | -9.7 | 10/11 |

## Heads

| base | grid | metric | meaningful | top 5 by |T| |
|---|---|---|---|---|
| rare | capital | BOS | 66 | L4H10 -0.202, L5H8 -0.177, L5H6 -0.171, L5H5 -0.170, L5H4 -0.104 |
| rare | capital | country | 42 | L0H2 +0.373, L3H7 -0.258, L4H10 +0.249, L2H2 -0.183, L1H11 +0.162 |
| rare | capital | partner | 37 | L5H8 +0.214, L5H5 +0.196, L1H11 -0.148, L6H9 +0.101, L0H4 -0.095 |
| rare | capital | self | 0 | L0H0 +0.000, L7H8 +0.000, L7H9 +0.000, L7H10 +0.000, L7H11 +0.000 |
| rare | capital | filler | 77 | L0H2 -0.223, L3H7 +0.197, L2H9 +0.109, L5H4 +0.105, L2H2 +0.096 |
| rare | city | BOS | 47 | L4H10 -0.170, L5H8 -0.116, L5H6 -0.116, L10H7 -0.087, L5H4 -0.084 |
| rare | city | country | 37 | L0H2 +0.316, L3H7 -0.200, L4H10 +0.190, L2H9 -0.136, L2H2 -0.128 |
| rare | city | partner | 24 | L5H8 +0.135, L7H11 +0.099, L4H7 -0.075, L2H2 +0.055, L0H4 -0.050 |
| rare | city | self | 0 | L0H0 +0.000, L7H8 +0.000, L7H9 +0.000, L7H10 +0.000, L7H11 +0.000 |
| rare | city | filler | 60 | L0H2 -0.202, L3H7 +0.159, L2H9 +0.093, L3H8 +0.086, L2H2 +0.074 |
| common | capital | BOS | 73 | L5H5 -0.246, L5H8 -0.213, L4H10 -0.194, L5H6 -0.134, L10H10 -0.128 |
| common | capital | country | 47 | L0H2 +0.398, L4H10 +0.266, L3H7 -0.257, L1H11 +0.232, L1H5 -0.198 |
| common | capital | partner | 59 | L5H5 +0.285, L5H8 +0.263, L1H11 -0.227, L2H2 +0.110, L1H5 +0.104 |
| common | capital | self | 0 | L0H0 +0.000, L7H8 +0.000, L7H9 +0.000, L7H10 +0.000, L7H11 +0.000 |
| common | capital | filler | 83 | L0H2 -0.219, L3H7 +0.171, L5H10 -0.099, L2H9 +0.098, L5H4 +0.097 |
| common | city | BOS | 58 | L4H10 -0.154, L5H8 -0.135, L10H7 -0.132, L0H2 -0.099, L6H8 -0.087 |
| common | city | country | 38 | L0H2 +0.338, L4H10 +0.202, L3H7 -0.198, L2H9 -0.146, L2H2 -0.137 |
| common | city | partner | 36 | L5H8 +0.172, L4H7 -0.104, L7H11 +0.100, L5H5 +0.076, L2H2 +0.062 |
| common | city | self | 0 | L0H0 +0.000, L7H8 +0.000, L7H9 +0.000, L7H10 +0.000, L7H11 +0.000 |
| common | city | filler | 63 | L0H2 -0.199, L3H7 +0.146, L3H8 +0.093, L5H4 +0.087, L2H9 +0.080 |

## Role switches, correct vs wrong

- rare, capital, argmax labels: 17 heads; L1H5 Country→Diffuse (+0.68), L2H5 Country→Diffuse (+0.15), L2H1 Diffuse→Sink (+0.29), L0H2 Sink→Country (+1.00), L4H7 Partner→Country (+0.47), L5H6 Sink→Country (+0.52), L6H8 Sink→Country (+0.17), L2H2 Country→Partner (+0.75), L2H3 Country→Partner (+0.54), L5H8 Sink→Partner (+0.31), L8H1 Sink→Partner (+0.23), L2H2 Country→Filler (+0.23), L2H3 Country→Filler (+0.18), L2H5 Country→Filler (+0.50), L2H9 Country→Filler (+0.49), L3H7 Country→Filler (+0.80), L3H8 Country→Filler (+0.19)
- rare, capital, mass labels: 24 heads; L7H6 Filler→Sink (+0.17), L0H2 Filler→Country (+0.83), L4H7 Partner→Country (+0.17), L5H6 Sink→Country (+0.51), L6H8 Sink→Country (+0.17), L2H2 Country→Partner (+0.34), L5H8 Sink→Partner (+0.31), L7H11 Filler→Partner (+0.18), L8H1 Sink→Partner (+0.22), L2H2 Country→Filler (+0.56), L2H9 Country→Filler (+0.29), L3H4 Sink→Filler (+0.18), L3H7 Country→Filler (+0.85), L3H8 Country→Filler (+0.40), L3H10 Sink→Filler (+0.24), L4H6 Sink→Filler (+0.24), L5H3 Sink→Filler (+0.41), L5H4 Sink→Filler (+0.19), L6H7 Sink→Filler (+0.22), L8H11 Sink→Filler (+0.20), L9H1 Sink→Filler (+0.23), L10H6 Sink→Filler (+0.21), L10H10 Sink→Filler (+0.20), L11H6 Sink→Filler (+0.19)
- rare, city, argmax labels: 16 heads; L1H5 Country→Diffuse (+0.30), L1H9 Sink→Diffuse (+0.30), L2H1 Diffuse→Sink (+0.39), L3H11 Country→Sink (+0.17), L0H2 Sink→Country (+1.00), L0H7 Diffuse→Country (+0.20), L4H7 Partner→Country (+0.60), L5H6 Sink→Country (+0.45), L6H0 Sink→Country (+0.15), L6H8 Sink→Country (+0.32), L7H11 Sink→Partner (+0.35), L2H3 Country→Filler (+0.20), L2H5 Country→Filler (+0.46), L2H9 Country→Filler (+0.25), L3H7 Country→Filler (+0.52), L3H8 Country→Filler (+0.21)
- rare, city, mass labels: 14 heads; L7H6 Filler→Sink (+0.20), L4H7 Partner→Country (+0.15), L5H6 Sink→Country (+0.45), L6H8 Sink→Country (+0.32), L7H11 Filler→Partner (+0.27), L2H2 Country→Filler (+0.39), L2H9 Country→Filler (+0.58), L3H7 Country→Filler (+0.82), L3H8 Country→Filler (+0.47), L4H6 Sink→Filler (+0.16), L4H7 Partner→Filler (+0.26), L5H3 Sink→Filler (+0.33), L8H2 Sink→Filler (+0.28), L8H11 Sink→Filler (+0.24)
- common, capital, argmax labels: 21 heads; L1H5 Country→Diffuse (+0.39), L2H1 Diffuse→Sink (+0.37), L5H0 Partner→Sink (+0.16), L10H11 Partner→Sink (+0.16), L0H2 Sink→Country (+1.00), L4H7 Partner→Country (+0.65), L4H10 Sink→Country (+0.18), L5H6 Sink→Country (+0.17), L6H8 Sink→Country (+0.31), L1H5 Country→Partner (+0.41), L2H2 Country→Partner (+0.81), L2H3 Country→Partner (+0.56), L2H5 Country→Partner (+0.22), L3H7 Country→Partner (+0.23), L5H8 Sink→Partner (+0.81), L2H5 Country→Filler (+0.32), L2H9 Country→Filler (+0.25), L3H7 Country→Filler (+0.47), L3H8 Country→Filler (+0.20), L5H4 Sink→Filler (+0.21), L10H4 Sink→Filler (+0.15)
- common, capital, mass labels: 29 heads; L1H6 Filler→Sink (+0.21), L5H0 Partner→Sink (+0.15), L5H10 Filler→Sink (+0.21), L0H2 Filler→Country (+1.00), L4H7 Partner→Country (+0.59), L4H10 Sink→Country (+0.18), L5H6 Sink→Country (+0.17), L6H8 Sink→Country (+0.31), L2H2 Country→Partner (+0.45), L5H8 Sink→Partner (+0.81), L1H5 Country→Filler (+0.82), L2H2 Country→Filler (+0.34), L2H9 Country→Filler (+0.62), L3H7 Country→Filler (+0.81), L3H11 Sink→Filler (+0.16), L5H4 Sink→Filler (+0.26), L6H3 Sink→Filler (+0.26), L7H5 Sink→Filler (+0.26), L9H11 Sink→Filler (+0.16), L10H2 Sink→Filler (+0.21), L10H3 Sink→Filler (+0.18), L10H4 Sink→Filler (+0.18), L10H8 Sink→Filler (+0.22), L10H11 Sink→Filler (+0.17), L11H1 Sink→Filler (+0.34), L11H6 Sink→Filler (+0.37), L11H7 Sink→Filler (+0.22), L11H9 Sink→Filler (+0.38), L11H11 Sink→Filler (+0.37)
- common, city, argmax labels: 13 heads; L2H10 Partner→Diffuse (+0.27), L2H1 Diffuse→Sink (+0.43), L0H2 Sink→Country (+1.00), L4H7 Partner→Country (+0.52), L6H8 Sink→Country (+0.17), L5H8 Sink→Partner (+0.28), L7H11 Sink→Partner (+0.32), L10H7 Sink→Partner (+0.40), L2H2 Country→Filler (+0.18), L2H3 Country→Filler (+0.16), L2H5 Country→Filler (+0.30), L3H7 Country→Filler (+0.28), L3H8 Country→Filler (+0.25)
- common, city, mass labels: 16 heads; L5H10 Filler→Sink (+0.19), L0H2 Filler→Country (+0.74), L4H7 Partner→Country (+0.45), L6H8 Sink→Country (+0.17), L5H8 Sink→Partner (+0.28), L7H11 Sink→Partner (+0.31), L10H7 Sink→Partner (+0.40), L2H2 Country→Filler (+0.45), L2H9 Country→Filler (+0.75), L3H7 Country→Filler (+0.74), L3H8 Country→Filler (+0.18), L5H4 Sink→Filler (+0.35), L7H5 Sink→Filler (+0.18), L10H4 Sink→Filler (+0.16), L11H1 Sink→Filler (+0.22), L11H7 Sink→Filler (+0.21)

## Figures

- `figures/head_T_BOS.png`, `figures/pooled_T_BOS.png`, `figures/residual_grids_BOS.png`
