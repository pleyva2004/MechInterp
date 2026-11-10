# Experiment 4: controlled difference, country vs original token at τ

Run `20261024-000548_exp4ctrl`. 200 random rows per filler type; per row and τ = 1..31, the country variant (11 countries, averaged within row) and a matched placebo word vs the unchanged row. Measure: attention from the final token (32) to τ, per head. d = 32 − τ. Meaningful = BH q ≤ 0.05 over the 31 × 144 cells and |D| ≥ 0.02.

## Checks

| check | result |
|---|---|
| design (ids at τ, row elsewhere, no repeats/BOS/stimulus ids) | PASS (asserted per τ and arm) |
| attention rows sum to 1 | PASS (atol 0.0001, asserted per batch) |
| determinism, common | PASS (max |Δ| 0.0e+00) |
| TL vs HF, common | PASS (max |Δ| 1.4e-05) |
| L4H11 prev_score ≥ 0.9, common | PASS (0.986) |
| determinism, rare | PASS (max |Δ| 0.0e+00) |
| TL vs HF, rare | PASS (max |Δ| 1.3e-05) |
| L4H11 prev_score ≥ 0.9, rare | PASS (0.985) |

## Pre-registered predictions

| id | verdict | detail |
|---|---|---|
| C1 | FAIL | common: 11 heads with prev ≥ 0.2; following the country at d ≥ 2: L2H3, L2H5, L2H9, L3H6, L3H7; rare: 10 heads with prev ≥ 0.2; following the country at d ≥ 2: L2H2, L2H3, L2H5, L2H9, L3H6, L3H8 |
| C2 | FAIL | common: plain 2/11 (L2H2, L4H11); rare: plain 3/10 (L3H2, L3H7, L4H11) |
| C3 | PASS | common: prev 0.081, D(1) +0.0648 [+0.0539, +0.0754], q 2.9e-22; rare: prev 0.064, D(1) +0.0391 [+0.0317, +0.0466], q 1.7e-17 |
| C4 | PASS | common: D(1) +0.0648, mean D(16–31) +0.0064; rare: D(1) +0.0391, mean D(16–31) +0.0070 |
| C5 | PASS | common: D_country(1) − D_word(1) +0.0358 [+0.0295, +0.0420]; rare: D_country(1) − D_word(1) +0.0359 [+0.0305, +0.0414] |
| C6 | PASS | common: L0H6, L1H7, L2H1, L3H5, L6H4, L7H7, L8H8, L8H11, L9H8, L10H0, L10H3, L10H6, L10H11, L11H1, L11H2, L11H3, L11H7; rare: L9H8, L10H3, L11H2, L11H7 |
| H-loc | FAIL | common: LI(5-8) +0.0057, LI(9-11) +0.0090 [+0.0075, +0.0105], difference -0.0033 [-0.0046, -0.0020]; rare: LI(5-8) +0.0043, LI(9-11) +0.0047 [+0.0036, +0.0059], difference -0.0004 [-0.0015, +0.0006] |

## Focus heads

| head | source | filler | prev_score | D(1) [95% CI] | mean D d 2–4 | mean D d 16–31 | positive d ≥ 2 | country − word at d 1 [95% CI] | label |
|---|---|---|---|---|---|---|---|---|---|
| L1H0 | Exp 1 previous-token | common | 0.224 | -0.1035 [-0.1294, -0.0783] | -0.0567 | -0.0001 | 0/30 | -0.0612 [-0.0702, -0.0518] | negative |
| L1H0 | Exp 1 previous-token | rare | 0.187 | -0.0131 [-0.0293, +0.0009] | -0.0169 | -0.0003 | 0/30 | -0.0237 [-0.0328, -0.0151] | negative |
| L2H2 | Exp 1 previous-token | common | 0.505 | +0.0081 [-0.0246, +0.0415] | -0.0080 | +0.0000 | 0/30 | -0.0747 [-0.0911, -0.0591] | plain previous-token |
| L2H2 | Exp 1 previous-token | rare | 0.475 | +0.0841 [+0.0586, +0.1086] | +0.0226 | +0.0000 | 1/30 | +0.0116 [-0.0025, +0.0257] | local country-tracking |
| L2H3 | Exp 1 previous-token | common | 0.263 | +0.1567 [+0.1293, +0.1823] | +0.0709 | +0.0002 | 5/30 | -0.0349 [-0.0433, -0.0263] | local country-tracking |
| L2H3 | Exp 1 previous-token | rare | 0.303 | +0.0714 [+0.0528, +0.0909] | +0.0298 | +0.0000 | 3/30 | -0.0385 [-0.0466, -0.0312] | local country-tracking |
| L2H5 | Exp 1 previous-token | common | 0.305 | +0.0927 [+0.0750, +0.1104] | +0.0287 | +0.0000 | 2/30 | +0.0957 [+0.0873, +0.1039] | local country-tracking |
| L2H5 | Exp 1 previous-token | rare | 0.343 | +0.0695 [+0.0515, +0.0892] | +0.0219 | +0.0000 | 1/30 | +0.1125 [+0.1026, +0.1213] | local country-tracking |
| L2H8 | Exp 1 previous-token | common | 0.283 | -0.1530 [-0.1766, -0.1295] | -0.0711 | -0.0001 | 0/30 | -0.0610 [-0.0677, -0.0538] | negative |
| L2H8 | Exp 1 previous-token | rare | 0.246 | -0.1174 [-0.1404, -0.0957] | -0.0627 | -0.0001 | 0/30 | -0.0504 [-0.0580, -0.0427] | negative |
| L2H9 | Exp 1 previous-token | common | 0.365 | +0.1707 [+0.1497, +0.1929] | +0.1034 | +0.0001 | 5/30 | +0.0142 [+0.0052, +0.0239] | local country-tracking |
| L2H9 | Exp 1 previous-token | rare | 0.367 | +0.0889 [+0.0723, +0.1057] | +0.0400 | +0.0000 | 3/30 | +0.0332 [+0.0248, +0.0413] | local country-tracking |
| L3H2 | Exp 1 previous-token | common | 0.372 | +0.0868 [+0.0528, +0.1190] | +0.0059 | +0.0000 | 0/30 | -0.1007 [-0.1121, -0.0896] | content-gated previous-token |
| L3H2 | Exp 1 previous-token | rare | 0.416 | +0.0124 [-0.0163, +0.0401] | +0.0036 | +0.0000 | 0/30 | -0.0953 [-0.1062, -0.0839] | plain previous-token |
| L3H6 | Exp 1 previous-token | common | 0.285 | +0.2436 [+0.2112, +0.2754] | +0.1034 | +0.0004 | 7/30 | -0.0729 [-0.0881, -0.0582] | local country-tracking |
| L3H6 | Exp 1 previous-token | rare | 0.342 | +0.1440 [+0.1150, +0.1747] | +0.0634 | +0.0000 | 4/30 | -0.0304 [-0.0430, -0.0167] | local country-tracking |
| L3H7 | Exp 1 previous-token | common | 0.501 | +0.0714 [+0.0392, +0.1018] | +0.0098 | +0.0000 | 1/30 | -0.0266 [-0.0414, -0.0123] | local country-tracking |
| L3H7 | Exp 1 previous-token | rare | 0.555 | +0.0187 [-0.0033, +0.0423] | +0.0032 | +0.0000 | 0/30 | -0.0087 [-0.0210, +0.0033] | plain previous-token |
| L3H8 | Exp 1 previous-token | common | 0.264 | +0.0409 [+0.0099, +0.0715] | +0.0063 | +0.0000 | 0/30 | +0.0459 [+0.0314, +0.0600] | content-gated previous-token |
| L3H8 | Exp 1 previous-token | rare | 0.334 | +0.1064 [+0.0832, +0.1310] | +0.0349 | +0.0000 | 3/30 | +0.1035 [+0.0918, +0.1155] | local country-tracking |
| L4H11 | Exp 1 previous-token | common | 0.986 | +0.0110 [+0.0000, +0.0256] | -0.0014 | -0.0000 | 0/30 | +0.0019 [-0.0008, +0.0046] | plain previous-token |
| L4H11 | Exp 1 previous-token | rare | 0.985 | +0.0118 [+0.0046, +0.0209] | -0.0010 | +0.0000 | 0/30 | +0.0027 [+0.0009, +0.0045] | plain previous-token |
| L4H7 | adjacent-only country in Exp 10 | common | 0.183 | +0.0190 [+0.0038, +0.0341] | +0.0125 | +0.0009 | 0/30 | -0.0205 [-0.0289, -0.0123] | no country effect |
| L4H7 | adjacent-only country in Exp 10 | rare | 0.196 | +0.0052 [-0.0063, +0.0168] | +0.0098 | +0.0003 | 0/30 | -0.0168 [-0.0239, -0.0097] | no country effect |
| L5H6 | adjacent-only country in Exp 10 | common | 0.423 | +0.0111 [-0.0270, +0.0511] | -0.0043 | +0.0000 | 0/30 | -0.0040 [-0.0233, +0.0154] | plain previous-token |
| L5H6 | adjacent-only country in Exp 10 | rare | 0.352 | -0.0510 [-0.0785, -0.0238] | -0.0015 | -0.0000 | 0/30 | -0.0619 [-0.0753, -0.0475] | negative |
| L6H8 | adjacent-only country in Exp 10 | common | 0.386 | +0.0447 [+0.0151, +0.0714] | -0.0273 | -0.0000 | 0/30 | +0.0015 [-0.0120, +0.0146] | content-gated previous-token |
| L6H8 | adjacent-only country in Exp 10 | rare | 0.323 | +0.0662 [+0.0466, +0.0857] | -0.0050 | -0.0000 | 0/30 | +0.0199 [+0.0068, +0.0332] | content-gated previous-token |
| L11H8 | adjacent-only country in Exp 10 (advisor) | common | 0.081 | +0.0648 [+0.0539, +0.0754] | +0.0411 | +0.0064 | 9/30 | +0.0358 [+0.0295, +0.0420] | local country-tracking |
| L11H8 | adjacent-only country in Exp 10 (advisor) | rare | 0.064 | +0.0391 [+0.0317, +0.0466] | +0.0272 | +0.0070 | 7/30 | +0.0359 [+0.0305, +0.0414] | local country-tracking |
| L0H2 | country at every Exp 10 position | common | 0.029 | +0.0226 [+0.0178, +0.0275] | +0.0207 | +0.0159 | 4/30 | +0.0127 [+0.0091, +0.0166] | local country-tracking |
| L0H2 | country at every Exp 10 position | rare | 0.032 | +0.0150 [+0.0089, +0.0210] | +0.0172 | +0.0141 | 0/30 | +0.0148 [+0.0107, +0.0189] | no country effect |

## Label counts (all 144 heads)

| filler | global country-tracking | local country-tracking | content-gated previous-token | negative | plain previous-token | no country effect |
|---|---|---|---|---|---|---|
| common | 17 | 26 | 15 | 12 | 4 | 70 |
| rare | 4 | 21 | 5 | 12 | 5 | 97 |

## Strongest far-distance country effects (mean D over d 16–31)

| filler | head | mean D d 16–31 | D(1) | positive d ≥ 2 | label |
|---|---|---|---|---|---|
| common | L9H8 | +0.1520 | +0.1864 | 30/30 | global country-tracking |
| common | L10H3 | +0.0780 | +0.0974 | 30/30 | global country-tracking |
| common | L11H7 | +0.0593 | +0.0773 | 30/30 | global country-tracking |
| common | L10H0 | +0.0592 | +0.0910 | 30/30 | global country-tracking |
| common | L11H2 | +0.0428 | +0.0398 | 30/30 | global country-tracking |
| common | L8H11 | +0.0359 | +0.0788 | 30/30 | global country-tracking |
| common | L1H7 | +0.0287 | +0.0381 | 30/30 | global country-tracking |
| common | L7H7 | +0.0282 | +0.0417 | 27/30 | global country-tracking |
| rare | L10H3 | +0.0836 | +0.1270 | 30/30 | global country-tracking |
| rare | L11H7 | +0.0440 | +0.0497 | 30/30 | global country-tracking |
| rare | L9H8 | +0.0388 | +0.0892 | 30/30 | global country-tracking |
| rare | L11H2 | +0.0385 | +0.0242 | 30/30 | global country-tracking |
| rare | L0H2 | +0.0141 | +0.0150 | 0/30 | no country effect |
| rare | L6H4 | +0.0134 | +0.0187 | 0/30 | no country effect |
| rare | L11H10 | +0.0109 | +0.0278 | 3/30 | local country-tracking |
| rare | L11H4 | +0.0105 | +0.0140 | 0/30 | no country effect |

## Locality by layer (stretch): mean Δ(d 2–4) − mean Δ(d 16–31), mean over the layer's heads

| filler | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | L11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| common | -0.0044 | -0.0012 | +0.0056 | +0.0136 | -0.0022 | +0.0024 | +0.0025 | +0.0083 | +0.0097 | +0.0104 | +0.0060 | +0.0105 |
| rare | -0.0017 | -0.0008 | +0.0076 | +0.0203 | +0.0014 | +0.0033 | +0.0037 | +0.0047 | +0.0054 | +0.0067 | +0.0032 | +0.0043 |

common: LI(5-8) +0.0057 [+0.0045, +0.0069]; LI(9-11) +0.0090 [+0.0075, +0.0105]; LI(5-8 − 9-11) -0.0033 [-0.0046, -0.0020]

rare: LI(5-8) +0.0043 [+0.0035, +0.0052]; LI(9-11) +0.0047 [+0.0036, +0.0059]; LI(5-8 − 9-11) -0.0004 [-0.0015, +0.0006]

## Files

- `controlled.npz`: per filler type, the stats of each contrast [distance, layer, head], prev_score, labels, the mean attention rows (control [L, H, K]; country / word [distance, L, H, K]), the per-row control attention [R, L, H, K] and per-row arm masses on τ / BOS / previous [R, distance, L, H, 3] (`row_differences` rebuilds the per-row contrasts), and the placebo words.
- `figures/focus_heads.png`: D(d) for the focus heads, both filler types, country (solid) and word (dashed).
- `figures/heads_by_distance.png`: D(d) for all 144 heads.
- `figures/layers_by_distance.png`: mean D over each layer's heads, and locality by layer.
- `viz_data.json`: input of `exp4.build_viz` (the interactive viewer).
