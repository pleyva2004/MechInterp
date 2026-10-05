# exp2: a natural-language phrase in random-token context

- Run `20261004-155651_exp2`. Bases: rare [1000, 39999], common [256, 999], 500 rows each, BOS + 32 random ids, no repeats, no phrase ids. Metric at the final query (position 32).
- Phrases: `dogs` = ' big dogs bark' [1263, 6844, 21405] (shuffled [6844, 1263, 21405]), `birds` = ' small birds sing' [1402, 10087, 1702] (shuffled [10087, 1402, 1702]), `men` = ' old men walk' [1468, 1450, 2513] (shuffled [1450, 1468, 2513]), `kids` = ' young kids play' [1862, 3988, 711] (shuffled [3988, 1862, 711]), `water` = ' hot water flows' [3024, 1660, 15623] (shuffled [1660, 3024, 15623]), `wind` = ' cold wind blows' [4692, 2344, 20385] (shuffled [2344, 4692, 20385]), `stars` = ' bright stars shine' [6016, 5788, 18340] (shuffled [5788, 6016, 18340]), `trees` = ' tall trees grow' [7331, 7150, 1663] (shuffled [7150, 7331, 1663]), `cars` = ' fast cars race' [3049, 5006, 3234] (shuffled [5006, 3049, 3234]), `bells` = ' loud bells ring' [7812, 30987, 5858] (shuffled [30987, 7812, 5858]), `horses` = ' wild horses run' [4295, 14260, 1057] (shuffled [14260, 4295, 1057]), `babies` = ' tiny babies sleep' [7009, 11903, 3993] (shuffled [11903, 7009, 3993]). Placements (model positions): near [29, 30, 31].
- Controls: `shuffled` (same ids, scrambled), `matched` / `matched_b` (per row, random ids with the same leading_space, is_alpha, is_capitalized and id class as each phrase token; pool sizes {'dogs': [16017, 16017, 16017], 'birds': [16017, 16017, 16017], 'men': [16017, 16017, 16017], 'kids': [16018, 16018, 280], 'water': [16017, 16017, 16017], 'wind': [16017, 16017, 16017], 'stars': [16017, 16017, 16017], 'trees': [16017, 16017, 16017], 'cars': [16017, 16017, 16017], 'bells': [16017, 16017, 16017], 'horses': [16017, 16017, 16017], 'babies': [16017, 16017, 16017]}).
- Contrasts (paired per row, a − b): total = nl − none, order = nl − shuffled, identity = shuffled − matched, form = matched − none, placebo = matched − matched_b. total = order + identity + form.
- Meaningful = BH q ≤ 0.05 (Wilcoxon signed-rank, over every test of a metric) and |Δ| ≥ the metric's threshold {'sink_mass': 0.02, 'slot_mass': 0.02, 'slot_logratio': 0.1, 'prev_mass': 0.02, 'entropy': 0.05}. Bold = meaningful. Predictions were written before any data: [PREDICTIONS.md](PREDICTIONS.md).

## Checks

| check | backend | result | detail |
|---|---|---|---|
| Attention rows sum to 1 | TL | PASS | max abs(sum − 1) = 4.9e-07 |
| Attention rows sum to 1 | HF | PASS | max abs(sum − 1) = 4.7e-07 |
| Determinism (first 50 re-extracted) | TL | PASS | bit-identical |
| Determinism (first 50 re-extracted) | HF | PASS | bit-identical |
| Design invariants (no repeats, pairing, slot contents, pools) | - | PASS | checked before extraction |
| `none` rows replicate exp1 (rare) | TL | PASS | per-head mean sink_mass r = 0.9996 |
| `none` rows replicate exp1 (common) | TL | PASS | per-head mean sink_mass r = 0.9996 |
| Cross-library | TL vs HF | PASS | max |Δ| probability metrics 6.0e-05; labels agree 100.00%; meaningful decisions (sink_mass) agree 100.00%; max |Δ effect size| 2.4e-02 |

## 1. Pre-registered predictions

| id | kind | result | detail |
|---|---|---|---|
| C1 | pooled_term_sign | PASS | common: +0.0216 [+0.0184, +0.0246] over 12 phrases; rare: +0.0122 [+0.0105, +0.0140] over 12 phrases |
| C2 | pooled_term_sign | PASS | common: +0.0176 [+0.0143, +0.0209] over 12 phrases; rare: +0.0037 [+0.0002, +0.0074] over 12 phrases |
| C3 | phrase_count | FAIL | common: 12/12 phrases (need 8); opposite sign 0; rare: 6/12 phrases (need 8); opposite sign 2 |
| C4 | pooled_term_sign | FAIL | common: +0.0983 [+0.0789, +0.1168] over 12 phrases; rare: +0.0181 [-0.0046, +0.0413] over 12 phrases |
| C5a | controls | PASS | L4H11 prev_mass: min over cells 0.957 (floor 0.94); L5H1 sink_mass: min over cells 0.960 (floor 0.9); L0H1 mode Self in 240/240 cells |
| C5b | few_meaningful | PASS | common/babies/near: 0 (limit 1); common/bells/near: 0 (limit 1); common/birds/near: 0 (limit 1); common/cars/near: 0 (limit 1); common/dogs/near: 0 (limit 1); common/horses/near: 0 (limit 1); common/kids/near: 1 (limit 1): L11H11; common/men/near: 0 (limit 1); common/stars/near: 0 (limit 1); common/trees/near: 0 (limit 1); common/water/near: 0 (limit 1); common/wind/near: 1 (limit 1): L5H6; rare/babies/near: 1 (limit 1): L5H6; rare/bells/near: 0 (limit 1); rare/birds/near: 0 (limit 1); rare/cars/near: 0 (limit 1); rare/dogs/near: 0 (limit 1); rare/horses/near: 0 (limit 1); rare/kids/near: 1 (limit 1): L3H3; rare/men/near: 0 (limit 1); rare/stars/near: 0 (limit 1); rare/trees/near: 0 (limit 1); rare/water/near: 0 (limit 1); rare/wind/near: 0 (limit 1) |

Every phrase appears in every summary table. The long per-head tables (sections 3, 4, the head list in 7 and the per-variant means in 8) and the per-phrase figures cover only ['dogs']; the full per-head results for every phrase are in effects.parquet, additivity_heads.parquet and surprisal_*.parquet.

## 2. Meaningful heads per contrast: sink_mass

Out of 144 heads. The placebo row is the noise floor.

| base | phrase | placement | total | order | identity | form | placebo |
|---|---|---|---|---|---|---|---|
| common | babies | near | 77 | 51 | 62 | 62 | 0 |
| common | bells | near | 75 | 37 | 62 | 64 | 0 |
| common | birds | near | 75 | 43 | 58 | 61 | 0 |
| common | cars | near | 70 | 33 | 56 | 58 | 0 |
| common | dogs | near | 69 | 37 | 64 | 64 | 0 |
| common | horses | near | 67 | 43 | 62 | 64 | 0 |
| common | kids | near | 74 | 37 | 55 | 57 | 1 |
| common | men | near | 70 | 41 | 72 | 61 | 0 |
| common | stars | near | 76 | 43 | 61 | 63 | 0 |
| common | trees | near | 75 | 53 | 64 | 61 | 0 |
| common | water | near | 68 | 24 | 59 | 63 | 0 |
| common | wind | near | 66 | 20 | 65 | 62 | 1 |
| rare | babies | near | 54 | 31 | 54 | 18 | 1 |
| rare | bells | near | 61 | 21 | 46 | 22 | 0 |
| rare | birds | near | 53 | 43 | 64 | 25 | 0 |
| rare | cars | near | 64 | 29 | 46 | 25 | 0 |
| rare | dogs | near | 56 | 16 | 46 | 24 | 0 |
| rare | horses | near | 58 | 37 | 34 | 23 | 0 |
| rare | kids | near | 74 | 42 | 53 | 50 | 1 |
| rare | men | near | 63 | 38 | 57 | 22 | 0 |
| rare | stars | near | 61 | 15 | 43 | 25 | 0 |
| rare | trees | near | 62 | 38 | 37 | 22 | 0 |
| rare | water | near | 72 | 13 | 61 | 24 | 0 |
| rare | wind | near | 67 | 15 | 61 | 22 | 0 |

Effect sizes (sink_mass): 

| base | phrase | placement | total median |Δ| | order median |Δ| | identity median |Δ| | form median |Δ| | placebo median |Δ| | largest total |
|---|---|---|---|---|---|---|---|---|
| common | babies | near | 0.0242 | 0.0097 | 0.0143 | 0.0156 | 0.0018 | L7H0 (+0.250) |
| common | bells | near | 0.0230 | 0.0093 | 0.0172 | 0.0162 | 0.0016 | L5H6 (+0.193) |
| common | birds | near | 0.0224 | 0.0095 | 0.0138 | 0.0154 | 0.0020 | L7H0 (+0.243) |
| common | cars | near | 0.0191 | 0.0096 | 0.0165 | 0.0152 | 0.0017 | L6H8 (+0.232) |
| common | dogs | near | 0.0181 | 0.0093 | 0.0179 | 0.0154 | 0.0026 | L7H0 (+0.203) |
| common | horses | near | 0.0181 | 0.0082 | 0.0158 | 0.0162 | 0.0019 | L4H1 (+0.178) |
| common | kids | near | 0.0215 | 0.0088 | 0.0133 | 0.0131 | 0.0029 | L4H3 (+0.224) |
| common | men | near | 0.0211 | 0.0087 | 0.0201 | 0.0150 | 0.0023 | L6H8 (+0.227) |
| common | stars | near | 0.0227 | 0.0070 | 0.0146 | 0.0162 | 0.0017 | L7H0 (+0.226) |
| common | trees | near | 0.0240 | 0.0120 | 0.0140 | 0.0151 | 0.0024 | L9H8 (+0.214) |
| common | water | near | 0.0189 | 0.0057 | 0.0171 | 0.0156 | 0.0020 | L7H0 (+0.212) |
| common | wind | near | 0.0178 | 0.0047 | 0.0160 | 0.0153 | 0.0022 | L7H0 (+0.198) |
| rare | babies | near | 0.0147 | 0.0072 | 0.0129 | 0.0054 | 0.0013 | L7H0 (+0.173) |
| rare | bells | near | 0.0133 | 0.0057 | 0.0118 | 0.0065 | 0.0021 | L7H0 (+0.137) |
| rare | birds | near | 0.0144 | 0.0108 | 0.0172 | 0.0056 | 0.0014 | L7H0 (+0.171) |
| rare | cars | near | 0.0150 | 0.0071 | 0.0119 | 0.0064 | 0.0020 | L7H0 (+0.156) |
| rare | dogs | near | 0.0141 | 0.0046 | 0.0130 | 0.0066 | 0.0022 | L7H0 (+0.198) |
| rare | horses | near | 0.0157 | 0.0091 | 0.0087 | 0.0061 | 0.0026 | L6H8 (+0.109) |
| rare | kids | near | 0.0215 | 0.0096 | 0.0156 | 0.0150 | 0.0021 | L4H0 (-0.197) |
| rare | men | near | 0.0166 | 0.0097 | 0.0164 | 0.0060 | 0.0018 | L4H0 (-0.185) |
| rare | stars | near | 0.0145 | 0.0044 | 0.0140 | 0.0059 | 0.0012 | L7H0 (+0.188) |
| rare | trees | near | 0.0167 | 0.0094 | 0.0108 | 0.0057 | 0.0015 | L6H0 (+0.111) |
| rare | water | near | 0.0199 | 0.0048 | 0.0146 | 0.0066 | 0.0019 | L7H0 (+0.175) |
| rare | wind | near | 0.0178 | 0.0039 | 0.0137 | 0.0053 | 0.0012 | L7H0 (+0.171) |

## 2. Meaningful heads per contrast: slot_mass

Out of 144 heads. The placebo row is the noise floor.

| base | phrase | placement | total | order | identity | form | placebo |
|---|---|---|---|---|---|---|---|
| common | babies | near | 88 | 38 | 63 | 82 | 3 |
| common | bells | near | 72 | 28 | 59 | 80 | 0 |
| common | birds | near | 79 | 36 | 61 | 80 | 0 |
| common | cars | near | 81 | 27 | 62 | 82 | 0 |
| common | dogs | near | 83 | 34 | 67 | 85 | 0 |
| common | horses | near | 77 | 32 | 65 | 81 | 2 |
| common | kids | near | 77 | 24 | 55 | 58 | 0 |
| common | men | near | 69 | 30 | 63 | 81 | 0 |
| common | stars | near | 78 | 26 | 56 | 76 | 0 |
| common | trees | near | 81 | 30 | 53 | 78 | 0 |
| common | water | near | 70 | 23 | 65 | 79 | 0 |
| common | wind | near | 69 | 20 | 66 | 78 | 1 |
| rare | babies | near | 62 | 28 | 54 | 23 | 1 |
| rare | bells | near | 57 | 22 | 53 | 26 | 0 |
| rare | birds | near | 56 | 29 | 59 | 30 | 0 |
| rare | cars | near | 62 | 19 | 57 | 33 | 0 |
| rare | dogs | near | 65 | 22 | 52 | 27 | 0 |
| rare | horses | near | 55 | 25 | 54 | 33 | 1 |
| rare | kids | near | 69 | 28 | 57 | 36 | 2 |
| rare | men | near | 68 | 33 | 64 | 29 | 0 |
| rare | stars | near | 61 | 21 | 50 | 30 | 0 |
| rare | trees | near | 62 | 28 | 43 | 26 | 0 |
| rare | water | near | 70 | 21 | 70 | 29 | 0 |
| rare | wind | near | 76 | 16 | 74 | 28 | 0 |

Effect sizes (slot_mass): 

| base | phrase | placement | total median |Δ| | order median |Δ| | identity median |Δ| | form median |Δ| | placebo median |Δ| | largest total |
|---|---|---|---|---|---|---|---|---|
| common | babies | near | 0.0257 | 0.0096 | 0.0154 | 0.0232 | 0.0020 | L5H6 (-0.214) |
| common | bells | near | 0.0205 | 0.0066 | 0.0138 | 0.0224 | 0.0021 | L11H11 (+0.219) |
| common | birds | near | 0.0257 | 0.0067 | 0.0143 | 0.0218 | 0.0014 | L5H6 (-0.167) |
| common | cars | near | 0.0239 | 0.0051 | 0.0155 | 0.0226 | 0.0021 | L2H4 (-0.220) |
| common | dogs | near | 0.0256 | 0.0078 | 0.0184 | 0.0235 | 0.0016 | L2H4 (-0.232) |
| common | horses | near | 0.0212 | 0.0066 | 0.0173 | 0.0224 | 0.0015 | L2H4 (-0.217) |
| common | kids | near | 0.0219 | 0.0064 | 0.0138 | 0.0155 | 0.0019 | L2H4 (-0.182) |
| common | men | near | 0.0190 | 0.0056 | 0.0163 | 0.0225 | 0.0017 | L11H11 (+0.172) |
| common | stars | near | 0.0252 | 0.0067 | 0.0140 | 0.0217 | 0.0014 | L11H11 (+0.252) |
| common | trees | near | 0.0259 | 0.0064 | 0.0146 | 0.0217 | 0.0016 | L11H11 (+0.262) |
| common | water | near | 0.0188 | 0.0060 | 0.0176 | 0.0233 | 0.0020 | L11H11 (+0.276) |
| common | wind | near | 0.0193 | 0.0052 | 0.0185 | 0.0224 | 0.0013 | L11H11 (+0.269) |
| rare | babies | near | 0.0158 | 0.0062 | 0.0141 | 0.0058 | 0.0013 | L7H0 (-0.161) |
| rare | bells | near | 0.0148 | 0.0034 | 0.0143 | 0.0061 | 0.0014 | L3H1 (-0.193) |
| rare | birds | near | 0.0145 | 0.0060 | 0.0151 | 0.0068 | 0.0017 | L7H0 (-0.163) |
| rare | cars | near | 0.0148 | 0.0042 | 0.0167 | 0.0066 | 0.0014 | L7H0 (-0.166) |
| rare | dogs | near | 0.0150 | 0.0057 | 0.0135 | 0.0058 | 0.0014 | L7H0 (-0.183) |
| rare | horses | near | 0.0131 | 0.0059 | 0.0137 | 0.0060 | 0.0019 | L3H1 (-0.137) |
| rare | kids | near | 0.0186 | 0.0067 | 0.0155 | 0.0076 | 0.0016 | L3H6 (-0.200) |
| rare | men | near | 0.0184 | 0.0060 | 0.0182 | 0.0050 | 0.0016 | L4H0 (+0.189) |
| rare | stars | near | 0.0155 | 0.0047 | 0.0125 | 0.0059 | 0.0012 | L3H1 (-0.181) |
| rare | trees | near | 0.0148 | 0.0059 | 0.0119 | 0.0056 | 0.0015 | L3H1 (-0.207) |
| rare | water | near | 0.0199 | 0.0046 | 0.0195 | 0.0060 | 0.0015 | L3H6 (-0.240) |
| rare | wind | near | 0.0223 | 0.0035 | 0.0211 | 0.0063 | 0.0014 | L3H1 (-0.176) |

## 3. Tracked heads (exp1ext rule-selected + controls): Δ sink_mass

**rare bases, `dogs`, near**

| head | group | total | order | identity | form |
|---|---|---|---|---|---|
| L1H5 | sink rises | -0.003 | +0.001 | +0.001 | -0.005 |
| L1H7 | sink rises | +0.004 | +0.000 | +0.004 | -0.000 |
| L7H7 | sink rises | -0.006 | +0.010 | -0.014 | -0.002 |
| L9H11 | sink rises | +0.005 | -0.000 | +0.016 | -0.011 |
| L10H11 | sink rises | +0.014 | +0.003 | **+0.032** | **-0.020** |
| L11H1 | sink rises | -0.005 | +0.013 | -0.016 | -0.002 |
| L11H3 | sink rises | -0.008 | -0.008 | +0.004 | -0.004 |
| L2H1 | sink rises | **+0.037** | +0.004 | **+0.021** | +0.012 |
| L2H6 | sink rises | **+0.026** | -0.005 | -0.005 | **+0.036** |
| L3H6 | sink rises | +0.015 | +0.010 | -0.006 | +0.011 |
| L7H11 | sink rises | **-0.024** | -0.011 | +0.006 | -0.018 |
| L10H4 | sink rises | +0.016 | +0.004 | **+0.028** | -0.017 |
| L11H9 | sink rises | **+0.047** | +0.004 | **+0.045** | -0.002 |
| L1H8 | sink falls | -0.009 | +0.001 | -0.013 | +0.003 |
| L2H0 | sink falls | **+0.061** | +0.018 | **+0.028** | +0.014 |
| L2H4 | sink falls | **+0.061** | **+0.021** | +0.012 | **+0.028** |
| L3H3 | sink falls | +0.006 | -0.002 | +0.009 | -0.001 |
| L6H4 | sink falls | **+0.037** | +0.009 | **+0.030** | -0.001 |
| L6H8 | sink falls | -0.002 | -0.002 | **-0.049** | **+0.049** |
| L7H0 | sink falls | **+0.198** | **-0.035** | **+0.115** | **+0.119** |
| L10H9 | sink falls | +0.011 | -0.012 | +0.013 | +0.011 |
| L4H0 | sink falls | **-0.094** | **-0.080** | -0.006 | -0.008 |
| L5H2 | sink falls | **+0.033** | **+0.058** | **-0.022** | -0.004 |
| L5H6 | sink falls | **+0.152** | +0.002 | **+0.153** | -0.002 |
| L8H2 | sink falls | -0.017 | +0.008 | +0.001 | **-0.026** |
| L9H8 | sink falls | **+0.054** | -0.013 | **+0.052** | +0.015 |
| L0H6 | label mix only | +0.001 | -0.000 | +0.002 | -0.000 |
| L1H0 | label mix only | +0.001 | +0.001 | -0.000 | -0.000 |
| L3H2 | label mix only | -0.016 | +0.003 | **-0.030** | +0.011 |
| L10H0 | label mix only | **+0.042** | +0.000 | **+0.034** | +0.008 |
| L0H1 | control | +0.000 | -0.000 | +0.000 | -0.000 |
| L4H11 | control | +0.000 | +0.000 | +0.000 | -0.000 |
| L5H1 | control | -0.004 | -0.006 | +0.005 | -0.004 |

**common bases, `dogs`, near**

| head | group | total | order | identity | form |
|---|---|---|---|---|---|
| L1H5 | sink rises | -0.000 | +0.001 | +0.002 | -0.004 |
| L1H7 | sink rises | +0.001 | -0.003 | +0.018 | -0.013 |
| L7H7 | sink rises | **-0.047** | **+0.029** | **-0.038** | **-0.039** |
| L9H11 | sink rises | +0.000 | +0.010 | **+0.025** | **-0.035** |
| L10H11 | sink rises | **-0.049** | **+0.047** | +0.018 | **-0.114** |
| L11H1 | sink rises | **-0.050** | **+0.032** | **-0.058** | **-0.025** |
| L11H3 | sink rises | -0.007 | +0.020 | **+0.025** | **-0.052** |
| L2H1 | sink rises | **+0.060** | -0.001 | **+0.037** | **+0.024** |
| L2H6 | sink rises | **+0.035** | -0.017 | +0.006 | **+0.046** |
| L3H6 | sink rises | -0.003 | **+0.024** | -0.002 | **-0.024** |
| L7H11 | sink rises | +0.007 | -0.005 | **+0.026** | -0.013 |
| L10H4 | sink rises | +0.016 | +0.009 | **+0.042** | **-0.035** |
| L11H9 | sink rises | +0.002 | +0.002 | **+0.028** | **-0.029** |
| L1H8 | sink falls | -0.005 | +0.001 | -0.007 | +0.001 |
| L2H0 | sink falls | **+0.076** | +0.007 | **+0.029** | **+0.040** |
| L2H4 | sink falls | **+0.161** | +0.016 | **+0.026** | **+0.119** |
| L3H3 | sink falls | **-0.028** | -0.010 | -0.007 | -0.010 |
| L6H4 | sink falls | **+0.066** | +0.013 | +0.010 | **+0.042** |
| L6H8 | sink falls | **+0.069** | +0.012 | **-0.042** | **+0.099** |
| L7H0 | sink falls | **+0.203** | -0.011 | **+0.075** | **+0.139** |
| L10H9 | sink falls | +0.008 | -0.014 | -0.020 | **+0.042** |
| L4H0 | sink falls | **-0.040** | -0.016 | **-0.064** | **+0.040** |
| L5H2 | sink falls | **+0.112** | **+0.040** | **-0.044** | **+0.117** |
| L5H6 | sink falls | **+0.184** | **+0.158** | **+0.031** | -0.005 |
| L8H2 | sink falls | -0.022 | +0.003 | -0.009 | -0.015 |
| L9H8 | sink falls | **+0.178** | +0.002 | **+0.109** | **+0.068** |
| L0H6 | label mix only | -0.005 | -0.000 | +0.004 | -0.010 |
| L1H0 | label mix only | -0.002 | +0.002 | +0.002 | -0.006 |
| L3H2 | label mix only | **-0.030** | +0.006 | -0.018 | -0.018 |
| L10H0 | label mix only | **-0.036** | **+0.029** | **-0.031** | **-0.033** |
| L0H1 | control | +0.000 | -0.000 | -0.000 | +0.000 |
| L4H11 | control | -0.000 | +0.000 | -0.000 | -0.000 |
| L5H1 | control | +0.001 | -0.001 | +0.003 | -0.002 |

## 4. Does the final token attend to the phrase? Δ slot_mass, tracked heads

**rare bases, `dogs`, near**

| head | group | total | order | identity | form |
|---|---|---|---|---|---|
| L1H5 | sink rises | +0.019 | -0.002 | +0.015 | +0.006 |
| L1H7 | sink rises | +0.004 | +0.006 | -0.009 | +0.006 |
| L7H7 | sink rises | **-0.034** | **-0.020** | -0.017 | +0.003 |
| L9H11 | sink rises | **-0.038** | -0.009 | -0.018 | -0.010 |
| L10H11 | sink rises | **-0.042** | -0.019 | **-0.067** | **+0.043** |
| L11H1 | sink rises | +0.002 | **-0.023** | **+0.023** | +0.002 |
| L11H3 | sink rises | **-0.032** | +0.000 | **-0.031** | -0.001 |
| L2H1 | sink rises | **-0.037** | -0.001 | **-0.037** | +0.001 |
| L2H6 | sink rises | **-0.082** | -0.013 | **-0.043** | **-0.027** |
| L3H6 | sink rises | **-0.133** | **-0.082** | +0.002 | **-0.053** |
| L7H11 | sink rises | **-0.034** | -0.019 | -0.015 | +0.000 |
| L10H4 | sink rises | **-0.028** | -0.008 | **-0.057** | **+0.036** |
| L11H9 | sink rises | **-0.049** | +0.005 | **-0.074** | +0.020 |
| L1H8 | sink falls | +0.014 | -0.005 | **+0.022** | -0.003 |
| L2H0 | sink falls | **-0.116** | **-0.041** | **-0.030** | **-0.044** |
| L2H4 | sink falls | **-0.063** | **-0.022** | -0.020 | **-0.021** |
| L3H3 | sink falls | **-0.023** | -0.002 | -0.016 | -0.005 |
| L6H4 | sink falls | **-0.024** | -0.009 | -0.018 | +0.002 |
| L6H8 | sink falls | +0.008 | +0.001 | **+0.056** | **-0.049** |
| L7H0 | sink falls | **-0.183** | **+0.022** | **-0.100** | **-0.105** |
| L10H9 | sink falls | -0.013 | +0.004 | -0.012 | -0.005 |
| L4H0 | sink falls | **+0.091** | **+0.100** | -0.010 | +0.001 |
| L5H2 | sink falls | **-0.036** | **-0.065** | **+0.023** | +0.006 |
| L5H6 | sink falls | **-0.163** | -0.013 | **-0.141** | -0.010 |
| L8H2 | sink falls | +0.004 | +0.002 | -0.002 | +0.003 |
| L9H8 | sink falls | **-0.040** | -0.003 | -0.013 | **-0.024** |
| L0H6 | label mix only | **-0.020** | +0.000 | **-0.027** | +0.006 |
| L1H0 | label mix only | **-0.030** | **-0.037** | **-0.023** | **+0.030** |
| L3H2 | label mix only | **+0.037** | -0.016 | **+0.074** | -0.020 |
| L10H0 | label mix only | **-0.021** | -0.018 | +0.001 | -0.005 |
| L0H1 | control | -0.000 | +0.000 | -0.001 | +0.000 |
| L4H11 | control | -0.001 | -0.002 | +0.003 | -0.001 |
| L5H1 | control | -0.001 | +0.000 | -0.002 | -0.000 |

**common bases, `dogs`, near**

| head | group | total | order | identity | form |
|---|---|---|---|---|---|
| L1H5 | sink rises | -0.002 | -0.000 | +0.018 | -0.020 |
| L1H7 | sink rises | **+0.057** | +0.015 | **-0.037** | **+0.079** |
| L7H7 | sink rises | **+0.030** | -0.017 | +0.005 | **+0.042** |
| L9H11 | sink rises | -0.016 | -0.006 | -0.019 | +0.009 |
| L10H11 | sink rises | **+0.112** | **-0.038** | **-0.027** | **+0.177** |
| L11H1 | sink rises | **+0.062** | **-0.039** | **+0.045** | **+0.056** |
| L11H3 | sink rises | **+0.055** | -0.011 | **-0.033** | **+0.099** |
| L2H1 | sink rises | **+0.043** | -0.002 | -0.013 | **+0.058** |
| L2H6 | sink rises | **-0.078** | -0.003 | **-0.026** | **-0.049** |
| L3H6 | sink rises | **+0.031** | **-0.083** | -0.004 | **+0.119** |
| L7H11 | sink rises | **-0.021** | -0.011 | **-0.021** | +0.012 |
| L10H4 | sink rises | +0.008 | -0.007 | **-0.080** | **+0.095** |
| L11H9 | sink rises | **+0.023** | +0.009 | **-0.081** | **+0.096** |
| L1H8 | sink falls | +0.005 | -0.005 | +0.018 | -0.009 |
| L2H0 | sink falls | **-0.201** | **-0.035** | **-0.067** | **-0.098** |
| L2H4 | sink falls | **-0.232** | **-0.027** | **-0.042** | **-0.164** |
| L3H3 | sink falls | **+0.047** | +0.013 | +0.017 | +0.017 |
| L6H4 | sink falls | **+0.025** | -0.006 | **+0.036** | -0.005 |
| L6H8 | sink falls | -0.006 | -0.017 | **+0.050** | **-0.039** |
| L7H0 | sink falls | **-0.096** | +0.006 | **-0.071** | **-0.031** |
| L10H9 | sink falls | **+0.063** | **+0.034** | **-0.024** | **+0.052** |
| L4H0 | sink falls | **+0.092** | **+0.040** | **+0.048** | +0.003 |
| L5H2 | sink falls | **-0.038** | **-0.082** | **+0.041** | +0.003 |
| L5H6 | sink falls | **-0.194** | **-0.169** | **-0.028** | +0.003 |
| L8H2 | sink falls | **-0.020** | +0.010 | -0.008 | **-0.021** |
| L9H8 | sink falls | **-0.032** | -0.005 | -0.004 | **-0.023** |
| L0H6 | label mix only | **+0.069** | +0.001 | **-0.057** | **+0.125** |
| L1H0 | label mix only | **-0.129** | **-0.044** | **-0.028** | **-0.056** |
| L3H2 | label mix only | **+0.100** | -0.020 | **+0.046** | **+0.074** |
| L10H0 | label mix only | **+0.086** | **-0.038** | **+0.061** | **+0.064** |
| L0H1 | control | +0.003 | +0.000 | +0.003 | +0.001 |
| L4H11 | control | +0.000 | -0.000 | +0.000 | +0.001 |
| L5H1 | control | +0.000 | +0.000 | -0.001 | +0.001 |

## 5. Label-mix shift (TV distance)

Split-half TV floor within the `none` rows: rare 0.100, common 0.132.

Heads whose label-mix TV exceeds the floor, per contrast:

| base | phrase | placement | total | order | identity | form | placebo |
|---|---|---|---|---|---|---|---|
| common | babies | near | 31 | 11 | 22 | 17 | 0 |
| common | bells | near | 31 | 11 | 15 | 17 | 0 |
| common | birds | near | 25 | 8 | 18 | 18 | 0 |
| common | cars | near | 24 | 11 | 11 | 19 | 0 |
| common | dogs | near | 25 | 8 | 16 | 16 | 0 |
| common | horses | near | 24 | 11 | 15 | 20 | 0 |
| common | kids | near | 25 | 4 | 19 | 19 | 0 |
| common | men | near | 28 | 8 | 15 | 18 | 0 |
| common | stars | near | 27 | 5 | 12 | 17 | 0 |
| common | trees | near | 26 | 5 | 15 | 18 | 0 |
| common | water | near | 25 | 5 | 16 | 17 | 0 |
| common | wind | near | 23 | 3 | 18 | 18 | 0 |
| rare | babies | near | 25 | 11 | 18 | 7 | 0 |
| rare | bells | near | 20 | 6 | 7 | 7 | 0 |
| rare | birds | near | 22 | 13 | 16 | 7 | 0 |
| rare | cars | near | 24 | 11 | 15 | 7 | 0 |
| rare | dogs | near | 25 | 8 | 13 | 7 | 1 |
| rare | horses | near | 22 | 11 | 16 | 8 | 0 |
| rare | kids | near | 30 | 10 | 18 | 15 | 0 |
| rare | men | near | 25 | 13 | 15 | 6 | 0 |
| rare | stars | near | 20 | 4 | 10 | 8 | 0 |
| rare | trees | near | 22 | 9 | 12 | 8 | 0 |
| rare | water | near | 21 | 3 | 16 | 8 | 0 |
| rare | wind | near | 22 | 5 | 18 | 6 | 0 |

## 6. Pooled over phrases (sink_mass, total)

| base | placement | name | n_phrases | mean_of_means | min_mean | max_mean | n_meaningful_pos | n_meaningful_neg |
|---|---|---|---|---|---|---|---|---|
| common | near | L7H0 | 12 | 0.202 | 0.143 | 0.250 | 12 | 0 |
| common | near | L9H8 | 12 | 0.171 | 0.078 | 0.237 | 12 | 0 |
| common | near | L4H3 | 12 | 0.160 | 0.073 | 0.224 | 12 | 0 |
| rare | near | L7H0 | 12 | 0.139 | 0.038 | 0.198 | 12 | 0 |
| common | near | L4H1 | 12 | 0.136 | 0.069 | 0.178 | 12 | 0 |
| common | near | L2H4 | 12 | 0.131 | 0.075 | 0.173 | 12 | 0 |
| common | near | L6H0 | 12 | 0.126 | 0.095 | 0.212 | 12 | 0 |
| common | near | L5H6 | 12 | 0.126 | -0.014 | 0.207 | 11 | 0 |
| common | near | L6H8 | 12 | 0.122 | -0.009 | 0.232 | 11 | 0 |
| common | near | L6H10 | 12 | -0.114 | -0.163 | -0.053 | 0 | 12 |
| common | near | L6H4 | 12 | 0.107 | 0.045 | 0.185 | 12 | 0 |
| common | near | L5H2 | 12 | 0.102 | -0.030 | 0.166 | 10 | 1 |
| rare | near | L4H0 | 12 | -0.095 | -0.197 | -0.019 | 0 | 11 |
| common | near | L11H11 | 12 | -0.093 | -0.172 | 0.017 | 0 | 11 |
| common | near | L5H10 | 12 | 0.089 | 0.047 | 0.133 | 12 | 0 |

## 7. Additivity: does the combination add more than its single tokens?

Per row, interaction = (combo − none) − Σ(part − none); e.g. `dogs:nl` = dogs:big@29 + dogs:dogs@30 + dogs:bark@31; `dogs:shuffled` = dogs:dogs@29 + dogs:big@30 + dogs:bark@31. Depth values average the group's heads within each row; CI = bootstrap over rows. Metrics: sink_logit, sink_mass (sink_logit = log(p / (1 − p)), not squeezed by the probability bounds).

| metric | base | phrase | term | depth | mean | lo | hi | CI excludes 0 |
|---|---|---|---|---|---|---|---|---|
| sink_mass | common | dogs | nl | early | -0.0016 | -0.0028 | -0.0004 | True |
| sink_mass | common | dogs | nl | late | +0.0143 | +0.0102 | +0.0189 | True |
| sink_mass | common | dogs | shuffled | early | -0.0057 | -0.0068 | -0.0046 | True |
| sink_mass | common | dogs | shuffled | late | +0.0017 | -0.0022 | +0.0060 | False |
| sink_mass | rare | dogs | nl | early | +0.0008 | +0.0001 | +0.0014 | True |
| sink_mass | rare | dogs | nl | late | +0.0081 | +0.0054 | +0.0109 | True |
| sink_mass | rare | dogs | shuffled | early | +0.0016 | +0.0010 | +0.0022 | True |
| sink_mass | rare | dogs | shuffled | late | +0.0131 | +0.0106 | +0.0156 | True |
| sink_mass | common | dogs | nl−shuffled | early | +0.0041 | +0.0034 | +0.0048 | True |
| sink_mass | common | dogs | nl−shuffled | late | +0.0126 | +0.0098 | +0.0156 | True |
| sink_mass | rare | dogs | nl−shuffled | early | -0.0008 | -0.0013 | -0.0003 | True |
| sink_mass | rare | dogs | nl−shuffled | late | -0.0050 | -0.0071 | -0.0032 | True |
| sink_logit | common | dogs | nl | early | -0.0199 | -0.0274 | -0.0122 | True |
| sink_logit | common | dogs | nl | late | +0.0875 | +0.0642 | +0.1121 | True |
| sink_logit | common | dogs | shuffled | early | -0.0376 | -0.0452 | -0.0304 | True |
| sink_logit | common | dogs | shuffled | late | +0.0222 | -0.0001 | +0.0463 | False |
| sink_logit | rare | dogs | nl | early | +0.0113 | +0.0063 | +0.0157 | True |
| sink_logit | rare | dogs | nl | late | +0.0497 | +0.0339 | +0.0664 | True |
| sink_logit | rare | dogs | shuffled | early | +0.0171 | +0.0125 | +0.0215 | True |
| sink_logit | rare | dogs | shuffled | late | +0.0884 | +0.0737 | +0.1042 | True |
| sink_logit | common | dogs | nl−shuffled | early | +0.0177 | +0.0134 | +0.0219 | True |
| sink_logit | common | dogs | nl−shuffled | late | +0.0653 | +0.0501 | +0.0815 | True |
| sink_logit | rare | dogs | nl−shuffled | early | -0.0058 | -0.0092 | -0.0026 | True |
| sink_logit | rare | dogs | nl−shuffled | late | -0.0387 | -0.0511 | -0.0277 | True |

Every phrase's late-layer values are in section 9.

Order contrast against its additive prediction, across the 144 heads (every phrase):

| metric | base | phrase | pair | slope | r | r2 | mean_abs_observed | mean_abs_interaction |
|---|---|---|---|---|---|---|---|---|
| sink_mass | common | babies | nl−shuffled | 0.013 | 0.005 | 0.000 | 0.018 | 0.021 |
| sink_mass | common | bells | nl−shuffled | 0.112 | 0.042 | 0.002 | 0.014 | 0.016 |
| sink_mass | common | birds | nl−shuffled | 0.730 | 0.282 | 0.080 | 0.017 | 0.017 |
| sink_mass | common | cars | nl−shuffled | 0.165 | 0.052 | 0.003 | 0.013 | 0.014 |
| sink_mass | common | dogs | nl−shuffled | 0.496 | 0.233 | 0.054 | 0.014 | 0.015 |
| sink_mass | common | horses | nl−shuffled | -0.571 | -0.179 | 0.032 | 0.016 | 0.017 |
| sink_mass | common | kids | nl−shuffled | -0.751 | -0.317 | 0.100 | 0.014 | 0.017 |
| sink_mass | common | men | nl−shuffled | -0.237 | -0.103 | 0.011 | 0.015 | 0.017 |
| sink_mass | common | stars | nl−shuffled | -0.247 | -0.103 | 0.011 | 0.015 | 0.018 |
| sink_mass | common | trees | nl−shuffled | 0.254 | 0.093 | 0.009 | 0.019 | 0.019 |
| sink_mass | common | water | nl−shuffled | -0.048 | -0.016 | 0.000 | 0.011 | 0.011 |
| sink_mass | common | wind | nl−shuffled | -0.517 | -0.186 | 0.035 | 0.009 | 0.011 |
| sink_mass | rare | babies | nl−shuffled | 1.092 | 0.484 | 0.234 | 0.013 | 0.012 |
| sink_mass | rare | bells | nl−shuffled | 0.859 | 0.461 | 0.212 | 0.010 | 0.008 |
| sink_mass | rare | birds | nl−shuffled | 0.393 | 0.125 | 0.016 | 0.018 | 0.016 |
| sink_mass | rare | cars | nl−shuffled | 0.835 | 0.389 | 0.152 | 0.011 | 0.009 |
| sink_mass | rare | dogs | nl−shuffled | 0.653 | 0.424 | 0.180 | 0.009 | 0.009 |
| sink_mass | rare | horses | nl−shuffled | 1.526 | 0.609 | 0.370 | 0.015 | 0.013 |
| sink_mass | rare | kids | nl−shuffled | 0.211 | 0.063 | 0.004 | 0.015 | 0.016 |
| sink_mass | rare | men | nl−shuffled | 1.045 | 0.324 | 0.105 | 0.017 | 0.017 |
| sink_mass | rare | stars | nl−shuffled | 0.809 | 0.489 | 0.239 | 0.008 | 0.007 |
| sink_mass | rare | trees | nl−shuffled | 0.619 | 0.284 | 0.081 | 0.015 | 0.013 |
| sink_mass | rare | water | nl−shuffled | 0.850 | 0.341 | 0.116 | 0.008 | 0.008 |
| sink_mass | rare | wind | nl−shuffled | 0.518 | 0.145 | 0.021 | 0.008 | 0.008 |
| sink_logit | common | babies | nl−shuffled | 0.015 | 0.006 | 0.000 | 0.101 | 0.117 |
| sink_logit | common | bells | nl−shuffled | 0.097 | 0.043 | 0.002 | 0.082 | 0.092 |
| sink_logit | common | birds | nl−shuffled | 0.683 | 0.282 | 0.080 | 0.099 | 0.100 |
| sink_logit | common | cars | nl−shuffled | 0.327 | 0.116 | 0.013 | 0.077 | 0.075 |
| sink_logit | common | dogs | nl−shuffled | 0.419 | 0.221 | 0.049 | 0.084 | 0.090 |
| sink_logit | common | horses | nl−shuffled | -0.276 | -0.095 | 0.009 | 0.094 | 0.101 |
| sink_logit | common | kids | nl−shuffled | -0.391 | -0.166 | 0.028 | 0.083 | 0.100 |
| sink_logit | common | men | nl−shuffled | -0.123 | -0.059 | 0.003 | 0.086 | 0.100 |
| sink_logit | common | stars | nl−shuffled | -0.143 | -0.064 | 0.004 | 0.091 | 0.102 |
| sink_logit | common | trees | nl−shuffled | 0.208 | 0.083 | 0.007 | 0.114 | 0.112 |
| sink_logit | common | water | nl−shuffled | 0.027 | 0.010 | 0.000 | 0.063 | 0.065 |
| sink_logit | common | wind | nl−shuffled | -0.329 | -0.123 | 0.015 | 0.055 | 0.063 |
| sink_logit | rare | babies | nl−shuffled | 1.150 | 0.508 | 0.258 | 0.086 | 0.074 |
| sink_logit | rare | bells | nl−shuffled | 0.968 | 0.532 | 0.283 | 0.063 | 0.050 |
| sink_logit | rare | birds | nl−shuffled | 0.489 | 0.164 | 0.027 | 0.108 | 0.099 |
| sink_logit | rare | cars | nl−shuffled | 0.959 | 0.468 | 0.219 | 0.070 | 0.054 |
| sink_logit | rare | dogs | nl−shuffled | 0.805 | 0.503 | 0.253 | 0.061 | 0.058 |
| sink_logit | rare | horses | nl−shuffled | 1.698 | 0.621 | 0.385 | 0.104 | 0.088 |
| sink_logit | rare | kids | nl−shuffled | 0.386 | 0.116 | 0.013 | 0.096 | 0.096 |
| sink_logit | rare | men | nl−shuffled | 1.096 | 0.352 | 0.124 | 0.104 | 0.100 |
| sink_logit | rare | stars | nl−shuffled | 0.933 | 0.579 | 0.335 | 0.056 | 0.045 |
| sink_logit | rare | trees | nl−shuffled | 0.797 | 0.340 | 0.116 | 0.095 | 0.080 |
| sink_logit | rare | water | nl−shuffled | 0.850 | 0.292 | 0.086 | 0.056 | 0.056 |
| sink_logit | rare | wind | nl−shuffled | 0.022 | 0.006 | 0.000 | 0.054 | 0.056 |

Heads with a BH-significant interaction on sink_mass (q over every interaction test of the metric):

| base | phrase | combo | significant | late + | late − | largest |interaction| |
|---|---|---|---|---|---|---|
| common | dogs | nl | 110 | 42 | 11 | L8H1 (+0.091), L5H6 (+0.085), L7H7 (+0.082) |
| common | dogs | shuffled | 94 | 23 | 17 | L5H2 (-0.074), L7H0 (+0.063), L4H3 (-0.056) |
| common | birds | nl | 111 | 54 | 8 | L8H6 (+0.118), L8H1 (+0.110), L7H0 (+0.105) |
| common | birds | shuffled | 101 | 38 | 14 | L4H0 (+0.082), L4H3 (-0.070), L9H8 (+0.053) |
| common | men | nl | 110 | 49 | 6 | L8H6 (+0.124), L6H8 (+0.118), L7H7 (+0.102) |
| common | men | shuffled | 89 | 28 | 14 | L7H8 (-0.052), L4H3 (-0.049), L7H7 (+0.041) |
| common | kids | nl | 120 | 52 | 8 | L7H7 (+0.134), L8H6 (+0.111), L11H1 (+0.109) |
| common | kids | shuffled | 107 | 31 | 14 | L7H7 (+0.100), L6H8 (-0.070), L4H5 (+0.065) |
| common | water | nl | 117 | 49 | 13 | L5H4 (-0.117), L7H8 (-0.079), L8H7 (+0.078) |
| common | water | shuffled | 109 | 42 | 10 | L5H2 (-0.082), L7H0 (+0.081), L6H8 (-0.076) |
| common | wind | nl | 115 | 37 | 18 | L5H4 (-0.149), L7H11 (+0.111), L8H1 (+0.089) |
| common | wind | shuffled | 114 | 24 | 30 | L5H2 (-0.128), L6H8 (-0.110), L5H4 (-0.084) |
| common | stars | nl | 124 | 51 | 14 | L6H0 (+0.123), L7H7 (+0.115), L8H6 (+0.110) |
| common | stars | shuffled | 118 | 35 | 22 | L5H6 (+0.090), L3H11 (+0.080), L7H11 (+0.074) |
| common | trees | nl | 118 | 44 | 16 | L8H6 (+0.136), L8H1 (+0.109), L10H11 (+0.098) |
| common | trees | shuffled | 100 | 20 | 27 | L5H2 (-0.129), L6H8 (-0.099), L7H8 (-0.068) |
| common | cars | nl | 115 | 47 | 10 | L5H6 (+0.163), L4H0 (-0.065), L11H1 (+0.061) |
| common | cars | shuffled | 106 | 32 | 19 | L7H7 (+0.077), L3H1 (+0.059), L3H7 (-0.053) |
| common | bells | nl | 118 | 49 | 10 | L11H11 (+0.096), L5H6 (+0.095), L7H7 (+0.094) |
| common | bells | shuffled | 109 | 33 | 25 | L4H3 (-0.099), L11H11 (+0.082), L7H8 (-0.052) |
| common | horses | nl | 110 | 42 | 13 | L5H6 (+0.113), L7H8 (-0.101), L8H1 (+0.090) |
| common | horses | shuffled | 100 | 29 | 12 | L6H8 (-0.091), L5H2 (-0.083), L4H3 (-0.083) |
| common | babies | nl | 125 | 51 | 13 | L5H6 (+0.130), L8H6 (+0.124), L8H1 (+0.119) |
| common | babies | shuffled | 114 | 35 | 22 | L6H8 (-0.076), L7H7 (+0.068), L8H1 (+0.062) |
| rare | dogs | nl | 114 | 44 | 18 | L5H6 (+0.082), L5H4 (-0.063), L4H0 (-0.059) |
| rare | dogs | shuffled | 110 | 51 | 6 | L7H0 (+0.048), L6H11 (+0.047), L6H7 (+0.045) |
| rare | birds | nl | 118 | 57 | 12 | L7H0 (+0.112), L8H1 (+0.073), L5H6 (+0.066) |
| rare | birds | shuffled | 102 | 30 | 18 | L4H0 (+0.112), L5H6 (-0.068), L6H0 (+0.054) |
| rare | men | nl | 113 | 38 | 18 | L4H0 (-0.144), L6H8 (+0.121), L5H4 (-0.084) |
| rare | men | shuffled | 86 | 25 | 16 | L8H7 (+0.052), L5H4 (-0.048), L5H6 (-0.047) |
| rare | kids | nl | 119 | 37 | 22 | L4H0 (-0.148), L5H0 (+0.121), L7H7 (+0.102) |
| rare | kids | shuffled | 108 | 20 | 31 | L5H0 (+0.074), L3H3 (+0.070), L4H3 (+0.051) |
| rare | water | nl | 123 | 51 | 10 | L5H6 (+0.089), L3H3 (+0.086), L11H11 (+0.067) |
| rare | water | shuffled | 126 | 55 | 8 | L5H6 (+0.130), L7H0 (+0.110), L3H3 (+0.071) |
| rare | wind | nl | 115 | 44 | 12 | L11H11 (+0.085), L7H0 (+0.084), L6H8 (-0.062) |
| rare | wind | shuffled | 110 | 39 | 12 | L7H0 (+0.153), L8H7 (+0.103), L6H11 (+0.090) |
| rare | stars | nl | 116 | 47 | 8 | L6H0 (+0.063), L5H6 (+0.058), L8H1 (+0.054) |
| rare | stars | shuffled | 123 | 52 | 9 | L5H6 (+0.142), L8H7 (+0.063), L3H11 (+0.061) |
| rare | trees | nl | 121 | 46 | 16 | L5H6 (+0.123), L8H1 (+0.106), L7H0 (+0.098) |
| rare | trees | shuffled | 108 | 32 | 18 | L7H0 (+0.149), L3H3 (+0.071), L8H7 (+0.057) |
| rare | cars | nl | 104 | 41 | 11 | L5H6 (+0.143), L4H0 (-0.088), L7H0 (+0.057) |
| rare | cars | shuffled | 99 | 27 | 21 | L6H0 (+0.043), L3H1 (+0.042), L6H8 (-0.040) |
| rare | bells | nl | 122 | 51 | 12 | L5H4 (-0.093), L3H3 (+0.050), L11H11 (+0.050) |
| rare | bells | shuffled | 112 | 49 | 12 | L11H11 (+0.067), L5H6 (+0.054), L8H7 (+0.050) |
| rare | horses | nl | 111 | 49 | 7 | L3H3 (+0.060), L8H1 (+0.046), L4H5 (+0.038) |
| rare | horses | shuffled | 128 | 52 | 14 | L7H0 (+0.135), L6H11 (+0.087), L3H3 (+0.073) |
| rare | babies | nl | 114 | 39 | 19 | L4H0 (-0.109), L8H1 (+0.076), L5H6 (+0.067) |
| rare | babies | shuffled | 100 | 32 | 17 | L8H7 (+0.061), L3H11 (+0.045), L3H3 (+0.044) |

## 8. Surprisal: does sink_mass track how predictable the slot tokens are?

Slot surprisal = −log p of the tokens at positions [29, 30, 31] given everything before them (TL), summed. Slope = within-row regression across every variant (both sides demeaned per row), per nat; CI = bootstrap over rows.

| base | phrase | depth | slope | lo | hi |
|---|---|---|---|---|---|
| common | babies | early | +0.00040 | +0.00027 | +0.00053 |
| common | babies | late | +0.00029 | -0.00009 | +0.00069 |
| common | bells | early | +0.00044 | +0.00031 | +0.00057 |
| common | bells | late | +0.00045 | +0.00008 | +0.00081 |
| common | birds | early | -0.00005 | -0.00017 | +0.00006 |
| common | birds | late | +0.00002 | -0.00031 | +0.00035 |
| common | cars | early | +0.00041 | +0.00027 | +0.00055 |
| common | cars | late | +0.00036 | -0.00003 | +0.00074 |
| common | dogs | early | +0.00045 | +0.00033 | +0.00056 |
| common | dogs | late | +0.00042 | +0.00008 | +0.00078 |
| common | horses | early | -0.00018 | -0.00028 | -0.00007 |
| common | horses | late | +0.00021 | -0.00015 | +0.00056 |
| common | kids | early | -0.00002 | -0.00019 | +0.00015 |
| common | kids | late | -0.00046 | -0.00091 | -0.00000 |
| common | men | early | -0.00006 | -0.00017 | +0.00005 |
| common | men | late | +0.00017 | -0.00017 | +0.00049 |
| common | stars | early | +0.00019 | +0.00008 | +0.00031 |
| common | stars | late | +0.00036 | +0.00000 | +0.00069 |
| common | trees | early | +0.00005 | -0.00007 | +0.00016 |
| common | trees | late | +0.00013 | -0.00020 | +0.00044 |
| common | water | early | +0.00016 | +0.00004 | +0.00028 |
| common | water | late | +0.00022 | -0.00012 | +0.00060 |
| common | wind | early | +0.00044 | +0.00034 | +0.00055 |
| common | wind | late | +0.00029 | -0.00004 | +0.00061 |
| rare | babies | early | -0.00049 | -0.00060 | -0.00039 |
| rare | babies | late | -0.00083 | -0.00115 | -0.00051 |
| rare | bells | early | -0.00007 | -0.00015 | +0.00002 |
| rare | bells | late | -0.00039 | -0.00068 | -0.00008 |
| rare | birds | early | -0.00013 | -0.00022 | -0.00005 |
| rare | birds | late | +0.00030 | +0.00003 | +0.00059 |
| rare | cars | early | -0.00028 | -0.00035 | -0.00020 |
| rare | cars | late | -0.00130 | -0.00157 | -0.00104 |
| rare | dogs | early | -0.00012 | -0.00019 | -0.00004 |
| rare | dogs | late | -0.00124 | -0.00146 | -0.00102 |
| rare | horses | early | -0.00084 | -0.00093 | -0.00076 |
| rare | horses | late | -0.00094 | -0.00122 | -0.00067 |
| rare | kids | early | -0.00055 | -0.00064 | -0.00047 |
| rare | kids | late | -0.00099 | -0.00129 | -0.00071 |
| rare | men | early | -0.00005 | -0.00012 | +0.00002 |
| rare | men | late | -0.00033 | -0.00055 | -0.00011 |
| rare | stars | early | -0.00070 | -0.00078 | -0.00063 |
| rare | stars | late | -0.00075 | -0.00100 | -0.00050 |
| rare | trees | early | -0.00067 | -0.00075 | -0.00060 |
| rare | trees | late | -0.00070 | -0.00095 | -0.00046 |
| rare | water | early | -0.00072 | -0.00079 | -0.00064 |
| rare | water | late | -0.00151 | -0.00174 | -0.00129 |
| rare | wind | early | -0.00060 | -0.00067 | -0.00052 |
| rare | wind | late | -0.00136 | -0.00159 | -0.00113 |

Heads whose slope CI excludes 0, by sign (summed over phrases):

| base | depth | slope < 0 | slope > 0 |
|---|---|---|---|
| common | early | 375 | 321 |
| common | late | 302 | 346 |
| rare | early | 476 | 258 |
| rare | late | 482 | 231 |

Per-variant means (descriptive):

| base | phrase | variant | mean_slot_nll | mean_sink_mass_early | mean_sink_mass_late | mean_query_nll |
|---|---|---|---|---|---|---|
| common | dogs | none | 27.300 | 0.328 | 0.587 | 9.078 |
| common | dogs | nl | 27.664 | 0.335 | 0.596 | 9.314 |
| common | dogs | shuffled | 29.949 | 0.329 | 0.585 | 9.274 |
| common | dogs | matched | 38.842 | 0.337 | 0.594 | 9.257 |
| common | dogs | matched_b | 39.005 | 0.337 | 0.597 | 9.518 |
| common | dogs | dogs:big@29 | 27.639 | 0.328 | 0.581 | 9.067 |
| common | dogs | dogs:dogs@30 | 30.521 | 0.333 | 0.584 | 9.188 |
| common | dogs | dogs:bark@31 | 30.189 | 0.331 | 0.590 | 8.956 |
| common | dogs | dogs:dogs@29 | 30.709 | 0.331 | 0.586 | 9.089 |
| common | dogs | dogs:big@30 | 27.614 | 0.328 | 0.581 | 9.065 |
| rare | dogs | none | 37.977 | 0.333 | 0.599 | 12.582 |
| rare | dogs | nl | 27.193 | 0.335 | 0.615 | 12.394 |
| rare | dogs | shuffled | 29.859 | 0.334 | 0.615 | 12.315 |
| rare | dogs | matched | 36.976 | 0.335 | 0.603 | 12.424 |
| rare | dogs | matched_b | 37.044 | 0.336 | 0.606 | 12.466 |
| rare | dogs | dogs:big@29 | 34.834 | 0.332 | 0.599 | 12.570 |
| rare | dogs | dogs:dogs@30 | 36.115 | 0.337 | 0.609 | 12.637 |
| rare | dogs | dogs:bark@31 | 37.391 | 0.331 | 0.598 | 12.237 |
| rare | dogs | dogs:dogs@29 | 35.978 | 0.337 | 0.608 | 12.579 |
| rare | dogs | dogs:big@30 | 35.145 | 0.330 | 0.594 | 12.600 |

## 9. Across phrases

Phrases as the unit: pooled_mean = mean of the per-phrase depth-group interactions, CI from bootstrapping phrases; n_pos / n_neg = phrases whose own row-bootstrap CI excludes 0 on that side; spearman = across phrases, the term against the surprisal gap NLL(shuffled) − NLL(nl).

| metric | base | term | depth | n_phrases | pooled_mean | lo | hi | n_pos | n_neg | spearman_rho | spearman_p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sink_mass | common | nl | early | 12 | +0.0026 | +0.0008 | +0.0044 | 7 | 2 | -0.0350 | +0.9141 |
| sink_mass | common | nl | late | 12 | +0.0216 | +0.0184 | +0.0246 | 12 | 0 | -0.0979 | +0.7621 |
| sink_mass | common | shuffled | early | 12 | -0.0007 | -0.0027 | +0.0013 | 3 | 7 | -0.2378 | +0.4568 |
| sink_mass | common | shuffled | late | 12 | +0.0040 | +0.0016 | +0.0062 | 6 | 1 | +0.0070 | +0.9828 |
| sink_mass | rare | nl | early | 12 | +0.0034 | +0.0024 | +0.0046 | 11 | 0 | -0.0070 | +0.9828 |
| sink_mass | rare | nl | late | 12 | +0.0122 | +0.0105 | +0.0140 | 12 | 0 | +0.3077 | +0.3306 |
| sink_mass | rare | shuffled | early | 12 | +0.0031 | +0.0020 | +0.0044 | 12 | 0 | -0.3147 | +0.3191 |
| sink_mass | rare | shuffled | late | 12 | +0.0085 | +0.0053 | +0.0117 | 9 | 0 | -0.1189 | +0.7129 |
| sink_mass | common | nl−shuffled | early | 12 | +0.0033 | +0.0019 | +0.0047 | 9 | 1 | +0.1608 | +0.6175 |
| sink_mass | common | nl−shuffled | late | 12 | +0.0176 | +0.0143 | +0.0209 | 12 | 0 | -0.1818 | +0.5717 |
| sink_mass | rare | nl−shuffled | early | 12 | +0.0003 | -0.0006 | +0.0012 | 5 | 5 | +0.5175 | +0.0849 |
| sink_mass | rare | nl−shuffled | late | 12 | +0.0037 | +0.0002 | +0.0074 | 6 | 2 | +0.2238 | +0.4845 |
| sink_logit | common | nl | early | 12 | +0.0144 | +0.0027 | +0.0260 | 8 | 3 | +0.0559 | +0.8629 |
| sink_logit | common | nl | late | 12 | +0.1308 | +0.1124 | +0.1484 | 12 | 0 | -0.0979 | +0.7621 |
| sink_logit | common | shuffled | early | 12 | -0.0045 | -0.0163 | +0.0084 | 3 | 4 | -0.1399 | +0.6646 |
| sink_logit | common | shuffled | late | 12 | +0.0325 | +0.0191 | +0.0442 | 9 | 0 | +0.0280 | +0.9312 |
| sink_logit | rare | nl | early | 12 | +0.0264 | +0.0197 | +0.0336 | 12 | 0 | +0.0979 | +0.7621 |
| sink_logit | rare | nl | late | 12 | +0.0734 | +0.0632 | +0.0847 | 12 | 0 | +0.3566 | +0.2551 |
| sink_logit | rare | shuffled | early | 12 | +0.0239 | +0.0180 | +0.0308 | 12 | 0 | -0.2448 | +0.4433 |
| sink_logit | rare | shuffled | late | 12 | +0.0553 | +0.0350 | +0.0761 | 10 | 0 | -0.0839 | +0.7954 |
| sink_logit | common | nl−shuffled | early | 12 | +0.0189 | +0.0107 | +0.0270 | 10 | 0 | +0.2238 | +0.4845 |
| sink_logit | common | nl−shuffled | late | 12 | +0.0983 | +0.0789 | +0.1168 | 12 | 0 | -0.0699 | +0.8290 |
| sink_logit | rare | nl−shuffled | early | 12 | +0.0025 | -0.0029 | +0.0075 | 7 | 4 | +0.5385 | +0.0709 |
| sink_logit | rare | nl−shuffled | late | 12 | +0.0181 | -0.0046 | +0.0413 | 6 | 4 | +0.2238 | +0.4845 |

Late-layer interaction for every phrase (* = row-bootstrap CI excludes 0):

| metric | base | phrase | nl | shuffled | nl−shuffled | nll gap (shuffled − nl) |
|---|---|---|---|---|---|---|
| sink_mass | common | big dogs bark | +0.0143* | +0.0017 | +0.0126* | +2.28 |
| sink_mass | common | small birds sing | +0.0275* | +0.0081* | +0.0195* | +3.32 |
| sink_mass | common | old men walk | +0.0257* | +0.0042 | +0.0215* | +2.62 |
| sink_mass | common | young kids play | +0.0290* | +0.0069* | +0.0221* | +3.60 |
| sink_mass | common | hot water flows | +0.0173* | +0.0115* | +0.0058* | +3.02 |
| sink_mass | common | cold wind blows | +0.0112* | -0.0020 | +0.0132* | +4.88 |
| sink_mass | common | bright stars shine | +0.0277* | +0.0048* | +0.0229* | +2.10 |
| sink_mass | common | tall trees grow | +0.0222* | -0.0052* | +0.0274* | +2.43 |
| sink_mass | common | fast cars race | +0.0188* | +0.0061* | +0.0127* | +1.08 |
| sink_mass | common | loud bells ring | +0.0211* | +0.0034 | +0.0177* | +3.30 |
| sink_mass | common | wild horses run | +0.0158* | +0.0039 | +0.0118* | +3.45 |
| sink_mass | common | tiny babies sleep | +0.0285* | +0.0046* | +0.0240* | +2.57 |
| sink_mass | rare | big dogs bark | +0.0081* | +0.0131* | -0.0050* | +2.67 |
| sink_mass | rare | small birds sing | +0.0201* | +0.0050* | +0.0151* | +3.36 |
| sink_mass | rare | old men walk | +0.0110* | +0.0015 | +0.0095* | +2.73 |
| sink_mass | rare | young kids play | +0.0108* | -0.0014 | +0.0121* | +4.18 |
| sink_mass | rare | hot water flows | +0.0156* | +0.0171* | -0.0015 | +2.63 |
| sink_mass | rare | cold wind blows | +0.0124* | +0.0129* | -0.0005 | +3.49 |
| sink_mass | rare | bright stars shine | +0.0123* | +0.0121* | +0.0002 | +2.10 |
| sink_mass | rare | tall trees grow | +0.0152* | +0.0075* | +0.0077* | +2.74 |
| sink_mass | rare | fast cars race | +0.0096* | +0.0022 | +0.0074* | +2.32 |
| sink_mass | rare | loud bells ring | +0.0110* | +0.0099* | +0.0011 | +2.43 |
| sink_mass | rare | wild horses run | +0.0103* | +0.0163* | -0.0060* | +3.33 |
| sink_mass | rare | tiny babies sleep | +0.0098* | +0.0052* | +0.0046* | +1.88 |
| sink_logit | common | big dogs bark | +0.0875* | +0.0222 | +0.0653* | +2.28 |
| sink_logit | common | small birds sing | +0.1637* | +0.0552* | +0.1086* | +3.32 |
| sink_logit | common | old men walk | +0.1515* | +0.0302* | +0.1213* | +2.62 |
| sink_logit | common | young kids play | +0.1791* | +0.0538* | +0.1254* | +3.60 |
| sink_logit | common | hot water flows | +0.1008* | +0.0679* | +0.0328* | +3.02 |
| sink_logit | common | cold wind blows | +0.0781* | -0.0001 | +0.0782* | +4.88 |
| sink_logit | common | bright stars shine | +0.1645* | +0.0372* | +0.1273* | +2.10 |
| sink_logit | common | tall trees grow | +0.1407* | -0.0183 | +0.1590* | +2.43 |
| sink_logit | common | fast cars race | +0.1128* | +0.0472* | +0.0655* | +1.08 |
| sink_logit | common | loud bells ring | +0.1243* | +0.0237* | +0.1007* | +3.30 |
| sink_logit | common | wild horses run | +0.0930* | +0.0252* | +0.0678* | +3.45 |
| sink_logit | common | tiny babies sleep | +0.1733* | +0.0452* | +0.1281* | +2.57 |
| sink_logit | rare | big dogs bark | +0.0497* | +0.0884* | -0.0387* | +2.67 |
| sink_logit | rare | small birds sing | +0.1231* | +0.0335* | +0.0896* | +3.36 |
| sink_logit | rare | old men walk | +0.0635* | +0.0054 | +0.0581* | +2.73 |
| sink_logit | rare | young kids play | +0.0674* | -0.0073 | +0.0747* | +4.18 |
| sink_logit | rare | hot water flows | +0.0903* | +0.1105* | -0.0203* | +2.63 |
| sink_logit | rare | cold wind blows | +0.0784* | +0.0890* | -0.0106* | +3.49 |
| sink_logit | rare | bright stars shine | +0.0765* | +0.0771* | -0.0006 | +2.10 |
| sink_logit | rare | tall trees grow | +0.0934* | +0.0489* | +0.0445* | +2.74 |
| sink_logit | rare | fast cars race | +0.0603* | +0.0208* | +0.0395* | +2.32 |
| sink_logit | rare | loud bells ring | +0.0605* | +0.0573* | +0.0032 | +2.43 |
| sink_logit | rare | wild horses run | +0.0580* | +0.1035* | -0.0455* | +3.33 |
| sink_logit | rare | tiny babies sleep | +0.0596* | +0.0364* | +0.0232* | +1.88 |

## Figures

- [rare_dogs_near_sink_mass_contrasts](figures/rare_dogs_near_sink_mass_contrasts.png)
- [rare_dogs_sink_mass_tracked_forest](figures/rare_dogs_sink_mass_tracked_forest.png)
- [rare_dogs_near_slot_mass_contrasts](figures/rare_dogs_near_slot_mass_contrasts.png)
- [rare_dogs_slot_mass_tracked_forest](figures/rare_dogs_slot_mass_tracked_forest.png)
- [rare_dogs_sink_mass_layer_profile](figures/rare_dogs_sink_mass_layer_profile.png)
- [common_dogs_near_sink_mass_contrasts](figures/common_dogs_near_sink_mass_contrasts.png)
- [common_dogs_sink_mass_tracked_forest](figures/common_dogs_sink_mass_tracked_forest.png)
- [common_dogs_near_slot_mass_contrasts](figures/common_dogs_near_slot_mass_contrasts.png)
- [common_dogs_slot_mass_tracked_forest](figures/common_dogs_slot_mass_tracked_forest.png)
- [common_dogs_sink_mass_layer_profile](figures/common_dogs_sink_mass_layer_profile.png)
- [common_dogs_near_additivity](figures/common_dogs_near_additivity.png)
- [rare_dogs_near_additivity](figures/rare_dogs_near_additivity.png)
- [phrase_terms_sink_mass](figures/phrase_terms_sink_mass.png)
- [phrase_terms_sink_logit](figures/phrase_terms_sink_logit.png)
- [surprisal_variants](figures/surprisal_variants.png)
- [meaningful_counts_sink_mass](figures/meaningful_counts_sink_mass.png)
- [meaningful_counts_slot_mass](figures/meaningful_counts_slot_mass.png)

## Caveats

- 12 phrases from one template (pronoun verb object) are a small, hand-picked sample; CIs over phrases describe these phrases, not language in general.
- A slot at position p < 32 changes every later position's residual stream, so a contrast measures the slot tokens' total effect at the query (as keys and through earlier heads), not a single route.
- `matched` controls match surface form and id class, not frequency or meaning; `identity` therefore mixes everything else that distinguishes the phrase's tokens from same-form random words.
