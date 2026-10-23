# Experiment 3, class 2, R1-out: country → capital / city before a random final token

Run `20261015-190957_exp3rel`. TL attention (HF cross-check on 50 rows per base class). 200 rows per base class; 11 countries, so 2 × 121 cells + none per row. Layout: country at 30, capital or city at 31, the row's own random token at 32 (the query).

T = mean(correct cells) − mean(wrong cells); every token sits once in a correct cell and 10 times in wrong ones, so token identity cancels. Late = layers 6–11, mean over their heads.

## Checks

| check | result |
|---|---|
| design (ids at 30/31, base elsewhere, no repeats/BOS/stimulus ids in bases) | PASS (asserted per row) |
| attention rows sum to 1 | PASS (atol 0.0001, asserted per batch) |
| determinism, rare (5 rows) | PASS (max |Δ| 0.0e+00) |
| TL vs HF, rare (50 rows) | PASS (max |Δ| 2.6e-05, argmax-key agreement 99.9999%) |
| manipulation, rare | PASS (11/11 countries prefer their capital) |
| determinism, common (5 rows) | PASS (max |Δ| 0.0e+00) |
| TL vs HF, common (50 rows) | PASS (max |Δ| 2.0e-05, argmax-key agreement 99.9999%) |
| manipulation, common | PASS (11/11 countries prefer their capital) |

## Manipulation check (next-token log p at the country, in context)

| base | partner | own − mean(other 10), mean over countries | countries > 0 | own ranked 1st (mean over countries) |
|---|---|---|---|---|
| rare | capital | +2.35 | 11/11 | 78% |
| rare | city | +2.15 | 11/11 | 72% |
| common | capital | +2.72 | 11/11 | 83% |
| common | city | +2.53 | 11/11 | 87% |

## Pre-registered predictions

| id | verdict | detail |
|---|---|---|
| Rout-1 | FAIL | rare: equivalent -0.0009 [-0.0017, -0.0000], perm p 0.9474; common: negative -0.0037 [-0.0048, -0.0027], perm p 0.9999 |
| Rout-2 | FAIL | rare: positive +0.0057 [+0.0049, +0.0064], perm p 0.0001; common: equivalent +0.0009 [-0.0001, +0.0020], perm p 0.1922 |
| Rout-3 | FAIL | rare: 3/11; common: 2/11 |
| Rout-4 | PASS | rare/capital: 2/20 p<0.05, mean switches 0.00; rare/city: 1/20 p<0.05, mean switches 0.00; common/capital: 0/20 p<0.05, mean switches 0.00; common/city: 1/20 p<0.05, mean switches 0.00 |

## Pooled late-layer T (95% row CI; permutation p over pairings; three-way verdict)

| base | grid | metric | T | 95% CI | 90% CI | perm p (>) | perm p (<) | Wilcoxon p | verdict |
|---|---|---|---|---|---|---|---|---|---|
| rare | capital | BOS | -0.0009 | [-0.0017, -0.0000] | [-0.0016, -0.0002] | 0.9474 | 0.0527 | 0.037 | equivalent |
| rare | capital | country | +0.0026 | [+0.0024, +0.0027] | [+0.0024, +0.0027] | 0.0001 | 1.0000 | 5.9e-34 | positive |
| rare | capital | partner | -0.0041 | [-0.0044, -0.0038] | [-0.0043, -0.0038] | 1.0000 | 0.0001 | 1.7e-34 | negative |
| rare | capital | self | +0.0005 | [+0.0003, +0.0006] | [+0.0004, +0.0006] | 0.0001 | 1.0000 | 5.2e-10 | positive |
| rare | capital | filler | +0.0019 | [+0.0012, +0.0026] | [+0.0013, +0.0025] | 0.0001 | 1.0000 | 1.9e-08 | positive |
| rare | city | BOS | +0.0057 | [+0.0049, +0.0064] | [+0.0050, +0.0063] | 0.0001 | 1.0000 | 4.3e-26 | positive |
| rare | city | country | +0.0011 | [+0.0009, +0.0012] | [+0.0010, +0.0012] | 0.0001 | 1.0000 | 1.4e-26 | positive |
| rare | city | partner | -0.0042 | [-0.0044, -0.0039] | [-0.0044, -0.0040] | 1.0000 | 0.0001 | 1.4e-34 | negative |
| rare | city | self | +0.0003 | [+0.0002, +0.0005] | [+0.0002, +0.0004] | 0.0020 | 0.9981 | 0.00073 | positive |
| rare | city | filler | -0.0029 | [-0.0035, -0.0022] | [-0.0034, -0.0023] | 1.0000 | 0.0001 | 1.4e-13 | negative |
| common | capital | BOS | -0.0037 | [-0.0048, -0.0027] | [-0.0046, -0.0028] | 0.9999 | 0.0002 | 1.1e-10 | negative |
| common | capital | country | +0.0038 | [+0.0035, +0.0041] | [+0.0036, +0.0040] | 0.0001 | 1.0000 | 1.8e-34 | positive |
| common | capital | partner | -0.0076 | [-0.0080, -0.0071] | [-0.0080, -0.0072] | 1.0000 | 0.0001 | 7.5e-34 | negative |
| common | capital | self | +0.0007 | [+0.0005, +0.0008] | [+0.0005, +0.0008] | 0.0001 | 1.0000 | 1.1e-11 | positive |
| common | capital | filler | +0.0068 | [+0.0059, +0.0079] | [+0.0060, +0.0077] | 0.0001 | 1.0000 | 2.5e-25 | positive |
| common | city | BOS | +0.0009 | [-0.0001, +0.0020] | [+0.0000, +0.0018] | 0.1922 | 0.8079 | 0.069 | equivalent |
| common | city | country | +0.0025 | [+0.0023, +0.0027] | [+0.0023, +0.0027] | 0.0001 | 1.0000 | 3.5e-33 | positive |
| common | city | partner | -0.0056 | [-0.0060, -0.0053] | [-0.0059, -0.0054] | 1.0000 | 0.0001 | 2.2e-34 | negative |
| common | city | self | +0.0003 | [+0.0002, +0.0005] | [+0.0002, +0.0005] | 0.0035 | 0.9966 | 0.00053 | positive |
| common | city | filler | +0.0019 | [+0.0011, +0.0028] | [+0.0012, +0.0026] | 0.0210 | 0.9791 | 7.9e-05 | positive |

## Per country (late BOS, diagonal residual after double-centring)

| base | grid | France | Germany | Italy | Spain | Russia | Ireland | Ukraine | Japan | China | Egypt | Canada | > 0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rare | capital | -2.4 | -1.8 | -0.2 | -1.2 | -4.3 | -1.4 | +2.5 | -2.7 | -1.3 | +0.2 | +4.2 | 3/11 |
| rare | city | +5.0 | +3.1 | +6.1 | +0.0 | +5.7 | +14.0 | +8.2 | +5.7 | +1.7 | +4.9 | +2.2 | 11/11 |
| common | capital | -5.3 | -8.2 | -2.3 | -6.1 | -6.0 | -1.3 | -2.8 | -3.6 | +0.4 | -6.5 | +4.3 | 2/11 |
| common | city | +3.5 | +0.8 | -6.6 | -3.6 | -0.9 | +8.1 | +4.8 | +8.8 | +0.3 | -3.9 | -2.2 | 6/11 |

(× 10⁻³)

## Placebo, capital − city, total effect, association

| base | quantity | value |
|---|---|---|
| rare | placebo T (capital), 20 derangements | range [-0.0008, +0.0014], 2 with p < 0.05; mean switching heads 0.00 (argmax), 0.00 (mass) |
| rare | placebo T (city), 20 derangements | range [-0.0017, +0.0005], 1 with p < 0.05; mean switching heads 0.00 (argmax), 0.00 (mass) |
| rare | T capital − T city (late BOS) | -0.0065 [-0.0072, -0.0059] |
| rare | capital correct − none (late BOS) | +0.0125 [+0.0061, +0.0193] |
| rare | capital wrong − none (late BOS) | +0.0134 [+0.0069, +0.0201] |
| rare | slope of late BOS on log p(capital given country), wrong cells | +0.00078 [+0.00053, +0.00105] per nat |
| rare | city correct − none (late BOS) | +0.0163 [+0.0098, +0.0229] |
| rare | city wrong − none (late BOS) | +0.0106 [+0.0041, +0.0174] |
| rare | slope of late BOS on log p(city given country), wrong cells | +0.00194 [+0.00156, +0.00236] per nat |
| common | placebo T (capital), 20 derangements | range [-0.0006, +0.0017], 0 with p < 0.05; mean switching heads 0.00 (argmax), 0.00 (mass) |
| common | placebo T (city), 20 derangements | range [-0.0020, +0.0012], 1 with p < 0.05; mean switching heads 0.00 (argmax), 0.00 (mass) |
| common | T capital − T city (late BOS) | -0.0047 [-0.0054, -0.0039] |
| common | capital correct − none (late BOS) | +0.0466 [+0.0379, +0.0559] |
| common | capital wrong − none (late BOS) | +0.0504 [+0.0415, +0.0597] |
| common | slope of late BOS on log p(capital given country), wrong cells | +0.00365 [+0.00313, +0.00421] per nat |
| common | city correct − none (late BOS) | +0.0556 [+0.0469, +0.0651] |
| common | city wrong − none (late BOS) | +0.0547 [+0.0461, +0.0640] |
| common | slope of late BOS on log p(city given country), wrong cells | +0.00237 [+0.00179, +0.00294] per nat |

## Heads

| base | grid | metric | meaningful heads (q ≤ 0.05, |T| ≥ 0.02) | top 5 by |T| |
|---|---|---|---|---|
| rare | capital | BOS | 5 | L6H8 -0.034, L5H6 +0.026, L8H11 -0.022, L5H3 -0.022, L7H0 +0.021 |
| rare | capital | country | 2 | L6H8 +0.032, L9H8 +0.021, L5H6 +0.018, L9H3 +0.017, L7H0 +0.015 |
| rare | capital | partner | 15 | L5H6 -0.045, L1H0 -0.033, L7H0 -0.032, L4H3 -0.030, L9H8 -0.030 |
| rare | capital | self | 0 | L4H7 +0.017, L1H0 +0.007, L4H3 +0.006, L2H2 +0.006, L3H11 +0.005 |
| rare | capital | filler | 1 | L1H0 +0.021, L6H10 -0.015, L11H2 +0.015, L8H8 +0.014, L1H1 +0.014 |
| rare | city | BOS | 1 | L7H0 +0.031, L5H10 +0.017, L5H3 -0.017, L5H6 +0.017, L11H1 +0.016 |
| rare | city | country | 0 | L6H8 +0.020, L9H8 +0.015, L3H6 +0.013, L10H3 -0.012, L7H0 +0.011 |
| rare | city | partner | 11 | L7H0 -0.038, L4H3 -0.033, L3H6 -0.030, L3H2 -0.029, L3H3 -0.029 |
| rare | city | self | 0 | L4H7 +0.012, L4H3 +0.007, L2H2 +0.006, L3H6 +0.006, L3H2 +0.006 |
| rare | city | filler | 0 | L8H1 -0.015, L11H9 -0.014, L10H8 -0.013, L11H5 -0.013, L10H11 -0.013 |
| common | capital | BOS | 18 | L7H7 +0.046, L7H0 +0.043, L6H8 -0.040, L10H10 -0.040, L9H9 -0.040 |
| common | capital | country | 5 | L9H8 +0.043, L6H8 +0.034, L9H3 +0.027, L5H6 +0.024, L5H2 +0.022 |
| common | capital | partner | 24 | L9H8 -0.074, L1H0 -0.050, L4H0 -0.048, L9H3 -0.047, L7H0 -0.043 |
| common | capital | self | 0 | L4H7 +0.019, L5H0 -0.011, L4H1 +0.010, L4H3 +0.008, L2H5 -0.008 |
| common | capital | filler | 10 | L10H10 +0.037, L1H0 +0.036, L7H7 -0.034, L9H9 +0.032, L11H6 +0.028 |
| common | city | BOS | 8 | L5H2 -0.049, L7H7 +0.034, L7H0 +0.031, L5H3 -0.026, L10H7 -0.025 |
| common | city | country | 4 | L9H8 +0.031, L5H2 +0.028, L6H8 +0.027, L10H7 +0.020, L11H2 -0.020 |
| common | city | partner | 20 | L4H0 -0.044, L9H8 -0.043, L3H2 -0.043, L3H3 -0.040, L7H0 -0.039 |
| common | city | self | 0 | L4H7 +0.016, L4H3 +0.009, L4H1 +0.008, L5H0 -0.007, L4H0 +0.006 |
| common | city | filler | 4 | L7H7 -0.026, L8H2 +0.023, L1H0 +0.023, L5H3 +0.022, L5H10 -0.019 |

## Role switches, correct vs wrong (McNemar, BH, ≥ 15-point shift)

- rare, capital, argmax labels: 0 heads; none
- rare, capital, mass labels: 0 heads; none
- rare, city, argmax labels: 0 heads; none
- rare, city, mass labels: 0 heads; none
- common, capital, argmax labels: 0 heads; none
- common, capital, mass labels: 0 heads; none
- common, city, argmax labels: 0 heads; none
- common, city, mass labels: 0 heads; none

## Figures

- `figures/head_T_BOS.png`: T per head, BOS mass.
- `figures/pooled_T_BOS.png`: late-layer T with CIs, placebo derangements and the equivalence band.
- `figures/residual_grids_BOS.png`: 11 × 11 residual grids of late BOS (correct pairs on the diagonal).
