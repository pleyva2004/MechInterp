# Experiment 4b: controlled difference with the capital as the final token

Run `20261024-150407_exp4cap`. 200 rows per filler type; per row, τ = 1..31 and 11 countries, with each country's own capital (own) or a deranged capital (wrong) at 32 in both versions of the pair. D = attention from the final token to τ, country − original token. Fact contrast = D_own − D_wrong. Meaningful = BH q ≤ 0.05 over the 31 × 144 cells and |D| ≥ 0.02. Random-final-token numbers are Exp 4's (`results/20261024-000548_exp4ctrl`), same rows and countries.

## Checks

| check | result |
|---|---|
| design (ids at τ and 32, control row elsewhere, no repeats/BOS/stimulus ids) | PASS (asserted per τ, arm, condition) |
| attention rows sum to 1 | PASS (atol 0.0001, asserted per batch) |
| wrong-capital derangement has no fixed point | PASS |
| determinism, common | PASS (max |Δ| 0.0e+00) |
| TL vs HF, common | PASS (max |Δ| 1.2e-05) |
| controls identical own vs wrong, common | PASS (max |Δ| 3.0e-07) |
| determinism, rare | PASS (max |Δ| 0.0e+00) |
| TL vs HF, rare | PASS (max |Δ| 4.3e-05) |
| controls identical own vs wrong, rare | PASS (max |Δ| 2.4e-07) |

## Pre-registered predictions

| id | verdict | detail |
|---|---|---|
| F1 | PASS | common: -0.0314 [-0.0330, -0.0297]; rare: -0.0172 [-0.0186, -0.0159] |
| F2 | PASS | common: -0.0333 [-0.0348, -0.0318]; rare: -0.0278 [-0.0290, -0.0267] |
| F3 | FAIL | common: -0.0007 [-0.0026, +0.0012], q 0.29; rare: +0.0030 [+0.0017, +0.0043], q 1.1e-05 |
| F4 | FAIL | common: L9H8 30/30, L10H3 0/30, L11H7 0/30, L11H2 0/30; rare: L9H8 30/30, L10H3 0/30, L11H7 0/30, L11H2 0/30 |
| F5 | PASS | common: meaningful at d ≥ 2 in: none; rare: meaningful at d ≥ 2 in: none |
| F6 | FAIL | common: beyond d = 8: L1H0 (d 9); rare: beyond d = 8: none |

## Focus heads

| head | filler | prev (own) | D_own(1) | D_wrong(1) | D_random(1) | fact(1) [95% CI] | fact d 2–4 | fact d 16–31 | positive fact d ≥ 2 | D_own d 16–31 | label (own) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| L1H0 | common | 0.128 | +0.1336 | +0.1394 | -0.1035 | -0.0057 [-0.0067, -0.0049] | -0.0091 | -0.0016 | 0/30 | +0.0035 | local country-tracking |
| L1H0 | rare | 0.172 | +0.0515 | +0.0768 | -0.0131 | -0.0253 [-0.0260, -0.0246] | -0.0195 | -0.0007 | 0/30 | +0.0004 | local country-tracking |
| L2H2 | common | 0.369 | -0.1611 | +0.0263 | +0.0081 | -0.1874 [-0.1897, -0.1850] | -0.0530 | -0.0000 | 0/30 | -0.0000 | negative |
| L2H2 | rare | 0.433 | -0.1984 | -0.0187 | +0.0841 | -0.1797 [-0.1816, -0.1777] | -0.0413 | -0.0000 | 0/30 | -0.0000 | negative |
| L2H3 | common | 0.236 | -0.0255 | +0.0117 | +0.1567 | -0.0372 [-0.0380, -0.0365] | -0.0208 | -0.0001 | 0/30 | -0.0000 | negative |
| L2H3 | rare | 0.281 | -0.0583 | -0.0214 | +0.0714 | -0.0369 [-0.0375, -0.0364] | -0.0180 | -0.0000 | 0/30 | -0.0000 | negative |
| L2H5 | common | 0.313 | -0.1538 | -0.0289 | +0.0927 | -0.1249 [-0.1270, -0.1226] | -0.0529 | -0.0000 | 0/30 | -0.0001 | negative |
| L2H5 | rare | 0.342 | -0.1780 | -0.0615 | +0.0695 | -0.1165 [-0.1180, -0.1149] | -0.0433 | -0.0000 | 0/30 | -0.0000 | negative |
| L2H8 | common | 0.263 | -0.1831 | -0.1539 | -0.1530 | -0.0292 [-0.0301, -0.0283] | -0.0132 | -0.0000 | 0/30 | -0.0003 | negative |
| L2H8 | rare | 0.273 | -0.1966 | -0.1700 | -0.1174 | -0.0266 [-0.0274, -0.0257] | -0.0106 | -0.0000 | 0/30 | -0.0001 | negative |
| L2H9 | common | 0.295 | -0.0373 | +0.1277 | +0.1707 | -0.1650 [-0.1665, -0.1634] | -0.0915 | -0.0001 | 0/30 | +0.0000 | negative |
| L2H9 | rare | 0.337 | -0.1162 | +0.0355 | +0.0889 | -0.1517 [-0.1533, -0.1503] | -0.0668 | -0.0000 | 0/30 | -0.0000 | negative |
| L3H2 | common | 0.293 | -0.1003 | -0.0747 | +0.0868 | -0.0256 [-0.0270, -0.0243] | -0.0094 | -0.0000 | 0/30 | -0.0000 | negative |
| L3H2 | rare | 0.319 | -0.1050 | -0.0863 | +0.0124 | -0.0187 [-0.0198, -0.0176] | -0.0061 | -0.0000 | 0/30 | -0.0000 | negative |
| L3H6 | common | 0.173 | +0.1030 | +0.1023 | +0.2436 | +0.0007 [-0.0008, +0.0022] | -0.0090 | -0.0001 | 0/30 | +0.0001 | local country-tracking |
| L3H6 | rare | 0.243 | +0.0398 | +0.0437 | +0.1440 | -0.0039 [-0.0050, -0.0029] | -0.0064 | -0.0000 | 0/30 | +0.0000 | content-gated previous-token |
| L3H7 | common | 0.505 | -0.2619 | +0.0034 | +0.0714 | -0.2653 [-0.2679, -0.2628] | -0.0652 | -0.0000 | 0/30 | -0.0000 | negative |
| L3H7 | rare | 0.539 | -0.3158 | -0.0508 | +0.0187 | -0.2650 [-0.2678, -0.2623] | -0.0523 | -0.0000 | 0/30 | -0.0000 | negative |
| L3H8 | common | 0.335 | -0.1705 | -0.0949 | +0.0409 | -0.0756 [-0.0769, -0.0742] | -0.0252 | -0.0000 | 0/30 | -0.0000 | negative |
| L3H8 | rare | 0.356 | -0.1190 | -0.0363 | +0.1064 | -0.0828 [-0.0842, -0.0813] | -0.0255 | -0.0000 | 0/30 | -0.0000 | negative |
| L4H11 | common | 0.989 | -0.0750 | -0.0060 | +0.0110 | -0.0690 [-0.0764, -0.0619] | -0.0021 | -0.0000 | 0/30 | -0.0000 | negative |
| L4H11 | rare | 0.992 | -0.1307 | -0.0036 | +0.0118 | -0.1271 [-0.1451, -0.1093] | -0.0019 | -0.0000 | 0/30 | -0.0000 | negative |
| L11H8 | common | 0.053 | +0.1599 | +0.1606 | +0.0648 | -0.0007 [-0.0026, +0.0012] | -0.0062 | -0.0013 | 0/30 | +0.0249 | global country-tracking |
| L11H8 | rare | 0.059 | +0.0713 | +0.0683 | +0.0391 | +0.0030 [+0.0017, +0.0043] | -0.0003 | +0.0015 | 0/30 | +0.0123 | local country-tracking |
| L9H8 | common | 0.012 | +0.0964 | +0.0359 | +0.1864 | +0.0605 [+0.0577, +0.0632] | +0.0511 | +0.0511 | 30/30 | +0.0880 | global country-tracking |
| L9H8 | rare | 0.022 | +0.0892 | +0.0312 | +0.0892 | +0.0580 [+0.0548, +0.0613] | +0.0366 | +0.0225 | 30/30 | +0.0390 | global country-tracking |
| L10H3 | common | 0.008 | +0.1326 | +0.1265 | +0.0974 | +0.0061 [+0.0039, +0.0083] | -0.0255 | -0.0257 | 0/30 | +0.0544 | global country-tracking |
| L10H3 | rare | 0.036 | +0.1262 | +0.1289 | +0.1270 | -0.0027 [-0.0049, -0.0005] | -0.0237 | -0.0137 | 0/30 | +0.0370 | global country-tracking |
| L11H7 | common | 0.012 | +0.1226 | +0.1117 | +0.0773 | +0.0109 [+0.0095, +0.0122] | -0.0042 | -0.0012 | 0/30 | +0.0653 | global country-tracking |
| L11H7 | rare | 0.021 | +0.0804 | +0.0725 | +0.0497 | +0.0080 [+0.0069, +0.0091] | -0.0007 | +0.0028 | 0/30 | +0.0424 | global country-tracking |
| L11H2 | common | 0.008 | +0.0239 | +0.0879 | +0.0398 | -0.0639 [-0.0659, -0.0619] | -0.0599 | -0.0636 | 0/30 | +0.0255 | global country-tracking |
| L11H2 | rare | 0.018 | +0.0072 | +0.0499 | +0.0242 | -0.0427 [-0.0450, -0.0404] | -0.0348 | -0.0375 | 0/30 | +0.0083 | no country effect |

## Label counts with its capital at T (all 144 heads)

| filler | global country-tracking | local country-tracking | content-gated previous-token | negative | plain previous-token | no country effect |
|---|---|---|---|---|---|---|
| common | 34 | 27 | 15 | 25 | 0 | 43 |
| rare | 13 | 25 | 11 | 35 | 0 | 60 |

## Strongest fact contrasts (mean over d 2–31)

| filler | head | mean fact d 2–31 | fact(1) | positive fact d ≥ 2 |
|---|---|---|---|---|
| common | L0H2 | +0.3621 | +0.4012 | 30/30 |
| common | L4H10 | +0.2533 | +0.2774 | 30/30 |
| common | L3H0 | +0.2518 | +0.1355 | 30/30 |
| common | L3H4 | -0.1713 | -0.1334 | 0/30 |
| common | L1H5 | -0.1576 | -0.1679 | 0/30 |
| common | L1H11 | +0.1229 | +0.2321 | 30/30 |
| common | L0H9 | -0.0724 | -0.1022 | 0/30 |
| common | L11H2 | -0.0622 | -0.0639 | 0/30 |
| common | L0H10 | +0.0567 | +0.0719 | 30/30 |
| common | L9H8 | +0.0514 | +0.0605 | 30/30 |
| rare | L0H2 | +0.3365 | +0.3750 | 30/30 |
| rare | L3H0 | +0.2783 | +0.1339 | 30/30 |
| rare | L4H10 | +0.2034 | +0.2563 | 30/30 |
| rare | L1H5 | -0.0989 | -0.1056 | 0/30 |
| rare | L1H11 | +0.0924 | +0.1589 | 30/30 |
| rare | L3H4 | -0.0847 | -0.0745 | 0/30 |
| rare | L0H10 | +0.0529 | +0.0674 | 30/30 |
| rare | L0H9 | -0.0528 | -0.0756 | 0/30 |
| rare | L1H6 | +0.0378 | +0.0364 | 30/30 |
| rare | L11H2 | -0.0362 | -0.0427 | 0/30 |

## Late BOS fact (layers 6–11): its capital − wrong capital, country at τ

| filler | d = 1 | d = 2 | d = 4 | d = 8 | d = 16 | d = 31 | mean d 16–31 |
|---|---|---|---|---|---|---|---|
| common | -0.0314 [-0.0330, -0.0297] | -0.0221 [-0.0238, -0.0206] | -0.0242 [-0.0259, -0.0224] | -0.0262 [-0.0278, -0.0247] | -0.0291 [-0.0310, -0.0273] | -0.0403 [-0.0427, -0.0378] | -0.0333 [-0.0348, -0.0318] |
| rare | -0.0172 [-0.0186, -0.0159] | -0.0162 [-0.0174, -0.0149] | -0.0199 [-0.0212, -0.0185] | -0.0217 [-0.0230, -0.0204] | -0.0257 [-0.0272, -0.0242] | -0.0332 [-0.0349, -0.0315] | -0.0278 [-0.0290, -0.0267] |

## Files

- `capital.npz`: per condition and filler type, the stats of each contrast [distance, layer, head], prev_score, labels, mean attention rows, per-row arm masses (float16) and control (float16, mean over the row's 11 controls); per filler type the fact and own − random stats and the per-row late BOS fact.
- `figures/focus_<filler>.png`: D(d) for the focus heads with its capital, a wrong capital and a random final token.
- `figures/fact.png`: fact contrast for the focus heads, and the late BOS fact by distance.
- `figures/layers_own.png`: mean D over each layer's heads with its capital at T.
- `viz_data.json`: input of `exp4.build_viz`, with all three final-token conditions.
