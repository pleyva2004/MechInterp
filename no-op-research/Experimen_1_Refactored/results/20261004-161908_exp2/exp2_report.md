# exp2: a natural-language phrase in random-token context

- Run `20261004-161908_exp2`. Bases: rare [1000, 39999], common [256, 999], 500 rows each, BOS + 32 random ids, no repeats, no phrase ids. Metric at the final query (position 32).
- Phrases: `table` = ' on the table' [319, 262, 3084] (shuffled [262, 319, 3084]), `house` = ' in the house' [287, 262, 2156] (shuffled [262, 287, 2156]), `door` = ' at the door' [379, 262, 3420] (shuffled [262, 379, 3420]), `bed` = ' under the bed' [739, 262, 3996] (shuffled [262, 739, 3996]), `river` = ' near the river' [1474, 262, 7850] (shuffled [262, 1474, 7850]), `wall` = ' behind the wall' [2157, 262, 3355] (shuffled [262, 2157, 3355]), `room` = ' into the room' [656, 262, 2119] (shuffled [262, 656, 2119]), `hill` = ' over the hill' [625, 262, 12788] (shuffled [262, 625, 12788]), `city` = ' from the city' [422, 262, 1748] (shuffled [262, 422, 1748]), `dog` = ' with the dog' [351, 262, 3290] (shuffled [262, 351, 3290]), `street` = ' across the street' [1973, 262, 4675] (shuffled [262, 1973, 4675]), `window` = ' through the window' [832, 262, 4324] (shuffled [262, 832, 4324]). Placements (model positions): near [29, 30, 31].
- Controls: `shuffled` (same ids, scrambled), `matched` / `matched_b` (per row, random ids with the same leading_space, is_alpha, is_capitalized and id class as each phrase token; pool sizes {'table': [279, 279, 16019], 'house': [279, 279, 16019], 'door': [279, 279, 16019], 'bed': [279, 279, 16019], 'river': [16018, 280, 16018], 'wall': [16018, 280, 16018], 'room': [279, 279, 16019], 'hill': [279, 279, 16019], 'city': [279, 279, 16019], 'dog': [279, 279, 16019], 'street': [16018, 280, 16018], 'window': [279, 279, 16019]}).
- Contrasts (paired per row, a − b): total = nl − none, order = nl − shuffled, identity = shuffled − matched, form = matched − none, placebo = matched − matched_b. total = order + identity + form.
- Meaningful = BH q ≤ 0.05 (Wilcoxon signed-rank, over every test of a metric) and |Δ| ≥ the metric's threshold {'sink_mass': 0.02, 'slot_mass': 0.02, 'slot_logratio': 0.1, 'prev_mass': 0.02, 'entropy': 0.05}. Bold = meaningful. Predictions were written before any data: [PREDICTIONS.md](PREDICTIONS.md).

## Checks

| check | backend | result | detail |
|---|---|---|---|
| Attention rows sum to 1 | TL | PASS | max abs(sum − 1) = 4.7e-07 |
| Attention rows sum to 1 | HF | PASS | max abs(sum − 1) = 4.3e-07 |
| Determinism (first 50 re-extracted) | TL | PASS | bit-identical |
| Determinism (first 50 re-extracted) | HF | PASS | bit-identical |
| Design invariants (no repeats, pairing, slot contents, pools) | - | PASS | checked before extraction |
| `none` rows replicate exp1 (rare) | TL | PASS | per-head mean sink_mass r = 0.9993 |
| `none` rows replicate exp1 (common) | TL | PASS | per-head mean sink_mass r = 0.9995 |
| Cross-library | TL vs HF | PASS | max |Δ| probability metrics 7.0e-05; labels agree 100.00%; meaningful decisions (sink_mass) agree 100.00%; max |Δ effect size| 1.2e-01 |

## 1. Pre-registered predictions

| id | kind | result | detail |
|---|---|---|---|
| C1 | pooled_term_sign | PASS | common: +0.0093 [+0.0040, +0.0149] over 12 phrases; rare: +0.0327 [+0.0239, +0.0409] over 12 phrases |
| C2 | pooled_term_sign | PASS | common: +0.0131 [+0.0093, +0.0169] over 12 phrases; rare: +0.0168 [+0.0117, +0.0214] over 12 phrases |
| C3 | phrase_count | PASS | common: 11/12 phrases (need 8); opposite sign 0; rare: 11/12 phrases (need 8); opposite sign 0 |
| C4 | pooled_term_sign | PASS | common: +0.0756 [+0.0547, +0.0970] over 12 phrases; rare: +0.0985 [+0.0683, +0.1257] over 12 phrases |
| C5a | controls | PASS | L4H11 prev_mass: min over cells 0.974 (floor 0.94); L5H1 sink_mass: min over cells 0.973 (floor 0.9); L0H1 mode Self in 240/240 cells |
| C5b | few_meaningful | PASS | common/bed/near: 0 (limit 1); common/city/near: 0 (limit 1); common/dog/near: 0 (limit 1); common/door/near: 0 (limit 1); common/hill/near: 1 (limit 1): L7H8; common/house/near: 0 (limit 1); common/river/near: 0 (limit 1); common/room/near: 0 (limit 1); common/street/near: 1 (limit 1): L11H9; common/table/near: 0 (limit 1); common/wall/near: 1 (limit 1): L7H0; common/window/near: 0 (limit 1); rare/bed/near: 0 (limit 1); rare/city/near: 0 (limit 1); rare/dog/near: 1 (limit 1): L5H6; rare/door/near: 1 (limit 1): L5H6; rare/hill/near: 1 (limit 1): L5H6; rare/house/near: 1 (limit 1): L7H0; rare/river/near: 0 (limit 1); rare/room/near: 1 (limit 1): L5H6; rare/street/near: 0 (limit 1); rare/table/near: 1 (limit 1): L3H3; rare/wall/near: 0 (limit 1); rare/window/near: 0 (limit 1) |

Every phrase appears in every summary table. The long per-head tables (sections 3, 4, the head list in 7 and the per-variant means in 8) and the per-phrase figures cover only ['table']; the full per-head results for every phrase are in effects.parquet, additivity_heads.parquet and surprisal_*.parquet.

## 2. Meaningful heads per contrast: sink_mass

Out of 144 heads. The placebo row is the noise floor.

| base | phrase | placement | total | order | identity | form | placebo |
|---|---|---|---|---|---|---|---|
| common | bed | near | 82 | 42 | 78 | 42 | 0 |
| common | city | near | 77 | 47 | 76 | 50 | 0 |
| common | dog | near | 79 | 61 | 61 | 48 | 0 |
| common | door | near | 79 | 52 | 72 | 45 | 0 |
| common | hill | near | 80 | 37 | 75 | 47 | 1 |
| common | house | near | 73 | 35 | 72 | 43 | 0 |
| common | river | near | 86 | 49 | 83 | 49 | 0 |
| common | room | near | 74 | 47 | 66 | 48 | 0 |
| common | street | near | 75 | 36 | 74 | 49 | 1 |
| common | table | near | 78 | 33 | 64 | 41 | 0 |
| common | wall | near | 74 | 45 | 70 | 50 | 1 |
| common | window | near | 77 | 48 | 67 | 41 | 0 |
| rare | bed | near | 68 | 27 | 66 | 56 | 0 |
| rare | city | near | 84 | 47 | 76 | 51 | 0 |
| rare | dog | near | 84 | 47 | 69 | 52 | 1 |
| rare | door | near | 90 | 50 | 66 | 47 | 1 |
| rare | hill | near | 78 | 49 | 67 | 44 | 1 |
| rare | house | near | 75 | 35 | 62 | 51 | 1 |
| rare | river | near | 77 | 40 | 59 | 29 | 0 |
| rare | room | near | 80 | 52 | 57 | 50 | 1 |
| rare | street | near | 83 | 57 | 69 | 32 | 0 |
| rare | table | near | 82 | 42 | 65 | 53 | 1 |
| rare | wall | near | 88 | 60 | 63 | 28 | 0 |
| rare | window | near | 83 | 42 | 74 | 47 | 0 |

Effect sizes (sink_mass): 

| base | phrase | placement | total median |Δ| | order median |Δ| | identity median |Δ| | form median |Δ| | placebo median |Δ| | largest total |
|---|---|---|---|---|---|---|---|---|
| common | bed | near | 0.0258 | 0.0107 | 0.0227 | 0.0097 | 0.0030 | L5H6 (+0.209) |
| common | city | near | 0.0269 | 0.0087 | 0.0214 | 0.0118 | 0.0029 | L8H6 (+0.181) |
| common | dog | near | 0.0253 | 0.0138 | 0.0153 | 0.0112 | 0.0028 | L4H0 (-0.323) |
| common | door | near | 0.0245 | 0.0110 | 0.0200 | 0.0088 | 0.0023 | L4H3 (+0.133) |
| common | hill | near | 0.0248 | 0.0087 | 0.0217 | 0.0088 | 0.0022 | L4H0 (-0.192) |
| common | house | near | 0.0206 | 0.0081 | 0.0204 | 0.0093 | 0.0026 | L5H6 (+0.209) |
| common | river | near | 0.0274 | 0.0120 | 0.0235 | 0.0103 | 0.0032 | L8H6 (+0.221) |
| common | room | near | 0.0205 | 0.0106 | 0.0177 | 0.0118 | 0.0026 | L5H6 (+0.253) |
| common | street | near | 0.0219 | 0.0060 | 0.0217 | 0.0108 | 0.0032 | L5H6 (+0.267) |
| common | table | near | 0.0234 | 0.0053 | 0.0181 | 0.0094 | 0.0022 | L3H7 (+0.180) |
| common | wall | near | 0.0216 | 0.0098 | 0.0198 | 0.0112 | 0.0013 | L5H6 (+0.272) |
| common | window | near | 0.0223 | 0.0110 | 0.0173 | 0.0088 | 0.0023 | L6H0 (+0.178) |
| rare | bed | near | 0.0176 | 0.0080 | 0.0164 | 0.0117 | 0.0019 | L2H4 (-0.286) |
| rare | city | near | 0.0273 | 0.0088 | 0.0227 | 0.0122 | 0.0020 | L2H4 (-0.314) |
| rare | dog | near | 0.0305 | 0.0120 | 0.0164 | 0.0124 | 0.0025 | L4H0 (-0.367) |
| rare | door | near | 0.0257 | 0.0102 | 0.0147 | 0.0108 | 0.0018 | L2H4 (-0.316) |
| rare | hill | near | 0.0234 | 0.0099 | 0.0177 | 0.0101 | 0.0019 | L5H6 (+0.285) |
| rare | house | near | 0.0210 | 0.0060 | 0.0164 | 0.0109 | 0.0018 | L2H4 (-0.331) |
| rare | river | near | 0.0236 | 0.0090 | 0.0151 | 0.0078 | 0.0016 | L6H3 (-0.291) |
| rare | room | near | 0.0233 | 0.0124 | 0.0144 | 0.0105 | 0.0021 | L5H6 (+0.300) |
| rare | street | near | 0.0253 | 0.0127 | 0.0193 | 0.0077 | 0.0013 | L5H6 (+0.294) |
| rare | table | near | 0.0261 | 0.0126 | 0.0179 | 0.0123 | 0.0027 | L2H4 (-0.307) |
| rare | wall | near | 0.0300 | 0.0142 | 0.0172 | 0.0075 | 0.0018 | L5H6 (+0.309) |
| rare | window | near | 0.0272 | 0.0112 | 0.0228 | 0.0119 | 0.0015 | L2H4 (-0.247) |

## 2. Meaningful heads per contrast: slot_mass

Out of 144 heads. The placebo row is the noise floor.

| base | phrase | placement | total | order | identity | form | placebo |
|---|---|---|---|---|---|---|---|
| common | bed | near | 65 | 33 | 62 | 62 | 0 |
| common | city | near | 90 | 30 | 68 | 60 | 0 |
| common | dog | near | 84 | 45 | 61 | 55 | 0 |
| common | door | near | 76 | 32 | 59 | 58 | 0 |
| common | hill | near | 70 | 29 | 68 | 56 | 1 |
| common | house | near | 81 | 39 | 58 | 56 | 0 |
| common | river | near | 80 | 32 | 78 | 67 | 0 |
| common | room | near | 83 | 39 | 62 | 59 | 0 |
| common | street | near | 62 | 39 | 70 | 68 | 0 |
| common | table | near | 77 | 32 | 60 | 61 | 1 |
| common | wall | near | 67 | 41 | 74 | 68 | 1 |
| common | window | near | 71 | 31 | 57 | 56 | 0 |
| rare | bed | near | 95 | 41 | 55 | 53 | 0 |
| rare | city | near | 85 | 40 | 73 | 57 | 0 |
| rare | dog | near | 87 | 43 | 67 | 53 | 1 |
| rare | door | near | 95 | 46 | 67 | 55 | 1 |
| rare | hill | near | 89 | 40 | 66 | 53 | 1 |
| rare | house | near | 85 | 46 | 60 | 56 | 1 |
| rare | river | near | 91 | 36 | 70 | 35 | 0 |
| rare | room | near | 87 | 45 | 57 | 58 | 1 |
| rare | street | near | 94 | 44 | 77 | 34 | 0 |
| rare | table | near | 88 | 40 | 62 | 54 | 1 |
| rare | wall | near | 99 | 49 | 71 | 33 | 0 |
| rare | window | near | 91 | 44 | 63 | 60 | 0 |

Effect sizes (slot_mass): 

| base | phrase | placement | total median |Δ| | order median |Δ| | identity median |Δ| | form median |Δ| | placebo median |Δ| | largest total |
|---|---|---|---|---|---|---|---|---|
| common | bed | near | 0.0177 | 0.0070 | 0.0176 | 0.0164 | 0.0014 | L3H7 (-0.282) |
| common | city | near | 0.0276 | 0.0065 | 0.0181 | 0.0148 | 0.0019 | L3H3 (-0.220) |
| common | dog | near | 0.0269 | 0.0102 | 0.0158 | 0.0152 | 0.0021 | L4H0 (+0.312) |
| common | door | near | 0.0220 | 0.0083 | 0.0161 | 0.0157 | 0.0016 | L3H3 (-0.172) |
| common | hill | near | 0.0186 | 0.0041 | 0.0181 | 0.0152 | 0.0018 | L4H0 (+0.183) |
| common | house | near | 0.0249 | 0.0083 | 0.0154 | 0.0149 | 0.0015 | L5H6 (-0.213) |
| common | river | near | 0.0245 | 0.0071 | 0.0220 | 0.0173 | 0.0017 | L11H11 (+0.205) |
| common | room | near | 0.0245 | 0.0085 | 0.0155 | 0.0164 | 0.0015 | L5H6 (-0.264) |
| common | street | near | 0.0169 | 0.0074 | 0.0191 | 0.0188 | 0.0021 | L5H6 (-0.307) |
| common | table | near | 0.0218 | 0.0062 | 0.0159 | 0.0159 | 0.0015 | L3H7 (-0.285) |
| common | wall | near | 0.0185 | 0.0082 | 0.0210 | 0.0182 | 0.0017 | L3H7 (-0.282) |
| common | window | near | 0.0197 | 0.0061 | 0.0130 | 0.0148 | 0.0021 | L3H3 (-0.188) |
| rare | bed | near | 0.0325 | 0.0088 | 0.0154 | 0.0153 | 0.0015 | L2H4 (+0.303) |
| rare | city | near | 0.0320 | 0.0076 | 0.0204 | 0.0163 | 0.0017 | L2H4 (+0.332) |
| rare | dog | near | 0.0336 | 0.0079 | 0.0163 | 0.0152 | 0.0019 | L2H4 (+0.336) |
| rare | door | near | 0.0388 | 0.0078 | 0.0162 | 0.0161 | 0.0012 | L2H4 (+0.337) |
| rare | hill | near | 0.0341 | 0.0066 | 0.0171 | 0.0152 | 0.0013 | L5H6 (-0.291) |
| rare | house | near | 0.0345 | 0.0071 | 0.0149 | 0.0154 | 0.0015 | L2H4 (+0.349) |
| rare | river | near | 0.0331 | 0.0060 | 0.0193 | 0.0092 | 0.0019 | L2H4 (+0.298) |
| rare | room | near | 0.0336 | 0.0082 | 0.0112 | 0.0158 | 0.0014 | L2H4 (+0.323) |
| rare | street | near | 0.0349 | 0.0076 | 0.0222 | 0.0078 | 0.0012 | L3H6 (-0.352) |
| rare | table | near | 0.0339 | 0.0079 | 0.0163 | 0.0152 | 0.0017 | L3H7 (-0.389) |
| rare | wall | near | 0.0369 | 0.0114 | 0.0193 | 0.0078 | 0.0014 | L5H6 (-0.327) |
| rare | window | near | 0.0312 | 0.0093 | 0.0148 | 0.0151 | 0.0019 | L3H6 (-0.285) |

## 3. Tracked heads (exp1ext rule-selected + controls): Δ sink_mass

**rare bases, `table`, near**

| head | group | total | order | identity | form |
|---|---|---|---|---|---|
| L1H5 | sink rises | -0.017 | -0.000 | -0.013 | -0.004 |
| L1H7 | sink rises | **-0.021** | -0.000 | -0.016 | -0.005 |
| L7H7 | sink rises | -0.012 | -0.015 | -0.027 | **+0.030** |
| L9H11 | sink rises | **+0.046** | **+0.041** | **+0.022** | -0.016 |
| L10H11 | sink rises | **+0.040** | **+0.037** | +0.012 | -0.009 |
| L11H1 | sink rises | **+0.030** | +0.018 | **+0.029** | -0.017 |
| L11H3 | sink rises | +0.013 | +0.008 | **+0.029** | **-0.024** |
| L2H1 | sink rises | **-0.062** | -0.002 | **-0.053** | -0.007 |
| L2H6 | sink rises | **+0.028** | -0.003 | +0.001 | **+0.031** |
| L3H6 | sink rises | **+0.115** | +0.009 | **+0.081** | **+0.025** |
| L7H11 | sink rises | **+0.054** | **+0.020** | **+0.048** | -0.014 |
| L10H4 | sink rises | +0.015 | +0.012 | +0.018 | -0.015 |
| L11H9 | sink rises | **+0.068** | +0.017 | **+0.057** | -0.005 |
| L1H8 | sink falls | -0.008 | -0.001 | -0.011 | +0.004 |
| L2H0 | sink falls | **-0.088** | **+0.021** | **-0.074** | **-0.034** |
| L2H4 | sink falls | **-0.307** | -0.020 | **-0.249** | **-0.038** |
| L3H3 | sink falls | **+0.124** | +0.013 | **+0.078** | **+0.033** |
| L6H4 | sink falls | **-0.042** | **-0.032** | -0.010 | +0.000 |
| L6H8 | sink falls | **+0.063** | **+0.103** | **-0.033** | -0.007 |
| L7H0 | sink falls | **+0.194** | **+0.100** | **+0.051** | **+0.043** |
| L10H9 | sink falls | +0.005 | **+0.040** | -0.004 | **-0.031** |
| L4H0 | sink falls | **-0.142** | -0.012 | **-0.040** | **-0.089** |
| L5H2 | sink falls | **-0.253** | **+0.075** | **-0.208** | **-0.120** |
| L5H6 | sink falls | **+0.242** | **-0.037** | **+0.270** | +0.009 |
| L8H2 | sink falls | **-0.022** | +0.019 | -0.006 | **-0.034** |
| L9H8 | sink falls | **+0.020** | **-0.026** | **+0.065** | -0.020 |
| L0H6 | label mix only | +0.004 | -0.000 | +0.001 | +0.002 |
| L1H0 | label mix only | +0.003 | -0.000 | +0.002 | +0.001 |
| L3H2 | label mix only | **+0.073** | -0.017 | **+0.070** | +0.019 |
| L10H0 | label mix only | **+0.054** | -0.002 | **+0.059** | -0.003 |
| L0H1 | control | +0.000 | +0.000 | +0.000 | +0.000 |
| L4H11 | control | -0.000 | +0.000 | +0.000 | -0.001 |
| L5H1 | control | +0.012 | -0.001 | +0.009 | +0.004 |

**common bases, `table`, near**

| head | group | total | order | identity | form |
|---|---|---|---|---|---|
| L1H5 | sink rises | **-0.024** | -0.002 | -0.020 | -0.003 |
| L1H7 | sink rises | -0.010 | +0.001 | -0.009 | -0.002 |
| L7H7 | sink rises | -0.006 | -0.019 | +0.011 | +0.001 |
| L9H11 | sink rises | -0.018 | **+0.029** | +0.008 | **-0.054** |
| L10H11 | sink rises | **-0.027** | **+0.029** | -0.012 | **-0.044** |
| L11H1 | sink rises | **-0.046** | +0.001 | -0.016 | **-0.030** |
| L11H3 | sink rises | **-0.046** | +0.010 | -0.019 | **-0.037** |
| L2H1 | sink rises | **-0.031** | -0.000 | **-0.052** | **+0.022** |
| L2H6 | sink rises | **+0.023** | -0.003 | -0.015 | **+0.041** |
| L3H6 | sink rises | **+0.088** | **+0.022** | **+0.072** | -0.006 |
| L7H11 | sink rises | +0.011 | **+0.027** | **+0.031** | **-0.047** |
| L10H4 | sink rises | **-0.020** | +0.011 | **-0.021** | -0.010 |
| L11H9 | sink rises | **+0.023** | -0.006 | **+0.041** | -0.012 |
| L1H8 | sink falls | -0.007 | -0.001 | -0.007 | +0.001 |
| L2H0 | sink falls | **-0.032** | +0.014 | **-0.046** | +0.001 |
| L2H4 | sink falls | **-0.119** | -0.010 | **-0.146** | **+0.037** |
| L3H3 | sink falls | **+0.088** | **+0.020** | **+0.036** | **+0.032** |
| L6H4 | sink falls | **+0.049** | +0.009 | **+0.021** | +0.019 |
| L6H8 | sink falls | **+0.088** | **+0.064** | -0.003 | **+0.027** |
| L7H0 | sink falls | **+0.068** | **+0.026** | **-0.032** | **+0.074** |
| L10H9 | sink falls | **-0.048** | +0.005 | -0.011 | **-0.042** |
| L4H0 | sink falls | **-0.119** | **-0.067** | **-0.047** | -0.006 |
| L5H2 | sink falls | **-0.051** | **+0.064** | **-0.124** | +0.010 |
| L5H6 | sink falls | **+0.176** | **-0.050** | **+0.187** | **+0.039** |
| L8H2 | sink falls | **-0.122** | +0.001 | **-0.054** | **-0.069** |
| L9H8 | sink falls | **+0.086** | **-0.031** | **+0.066** | **+0.051** |
| L0H6 | label mix only | +0.002 | +0.000 | +0.007 | -0.005 |
| L1H0 | label mix only | -0.003 | -0.001 | +0.001 | -0.004 |
| L3H2 | label mix only | **+0.089** | -0.001 | **+0.079** | +0.011 |
| L10H0 | label mix only | +0.003 | -0.009 | +0.014 | -0.002 |
| L0H1 | control | -0.000 | -0.000 | +0.000 | -0.000 |
| L4H11 | control | +0.000 | +0.000 | -0.000 | +0.000 |
| L5H1 | control | +0.007 | +0.000 | +0.005 | +0.002 |

## 4. Does the final token attend to the phrase? Δ slot_mass, tracked heads

**rare bases, `table`, near**

| head | group | total | order | identity | form |
|---|---|---|---|---|---|
| L1H5 | sink rises | **+0.054** | -0.005 | **+0.059** | +0.000 |
| L1H7 | sink rises | +0.010 | -0.005 | **+0.026** | -0.010 |
| L7H7 | sink rises | **-0.066** | -0.009 | -0.006 | **-0.051** |
| L9H11 | sink rises | **-0.037** | +0.015 | **-0.026** | **-0.027** |
| L10H11 | sink rises | **-0.083** | **-0.031** | **-0.027** | **-0.025** |
| L11H1 | sink rises | **-0.050** | -0.003 | **-0.020** | **-0.027** |
| L11H3 | sink rises | **-0.049** | **-0.025** | +0.002 | **-0.026** |
| L2H1 | sink rises | -0.005 | +0.000 | +0.013 | -0.018 |
| L2H6 | sink rises | +0.008 | **-0.021** | **+0.027** | +0.002 |
| L3H6 | sink rises | **-0.376** | **-0.082** | **-0.216** | **-0.078** |
| L7H11 | sink rises | **-0.070** | -0.018 | -0.013 | **-0.039** |
| L10H4 | sink rises | -0.018 | **+0.033** | **-0.028** | **-0.023** |
| L11H9 | sink rises | **-0.060** | +0.002 | **-0.039** | **-0.023** |
| L1H8 | sink falls | **+0.059** | -0.002 | **+0.054** | +0.007 |
| L2H0 | sink falls | **+0.105** | **-0.069** | **+0.126** | **+0.048** |
| L2H4 | sink falls | **+0.326** | **+0.022** | **+0.258** | **+0.046** |
| L3H3 | sink falls | **-0.132** | **-0.026** | **-0.080** | **-0.026** |
| L6H4 | sink falls | **-0.034** | +0.012 | **-0.021** | **-0.025** |
| L6H8 | sink falls | **-0.126** | **-0.134** | +0.014 | -0.006 |
| L7H0 | sink falls | **-0.183** | **-0.107** | **-0.031** | **-0.045** |
| L10H9 | sink falls | **-0.024** | **-0.043** | +0.009 | +0.010 |
| L4H0 | sink falls | **+0.057** | +0.012 | -0.022 | **+0.067** |
| L5H2 | sink falls | **+0.139** | **-0.104** | **+0.146** | **+0.096** |
| L5H6 | sink falls | **-0.290** | +0.012 | **-0.283** | -0.018 |
| L8H2 | sink falls | **+0.031** | **-0.022** | **+0.058** | -0.005 |
| L9H8 | sink falls | **-0.046** | -0.003 | -0.012 | **-0.030** |
| L0H6 | label mix only | **-0.058** | +0.000 | **-0.024** | **-0.033** |
| L1H0 | label mix only | **+0.088** | -0.006 | **+0.038** | **+0.056** |
| L3H2 | label mix only | **-0.209** | **-0.052** | **-0.153** | -0.004 |
| L10H0 | label mix only | **-0.077** | -0.002 | **-0.029** | **-0.046** |
| L0H1 | control | -0.002 | -0.000 | -0.001 | -0.001 |
| L4H11 | control | -0.004 | -0.004 | -0.000 | +0.001 |
| L5H1 | control | -0.002 | +0.000 | -0.000 | -0.002 |

**common bases, `table`, near**

| head | group | total | order | identity | form |
|---|---|---|---|---|---|
| L1H5 | sink rises | **+0.026** | -0.003 | **+0.034** | -0.005 |
| L1H7 | sink rises | **+0.033** | -0.002 | +0.003 | **+0.032** |
| L7H7 | sink rises | +0.004 | -0.006 | +0.004 | +0.007 |
| L9H11 | sink rises | **-0.029** | -0.012 | -0.020 | +0.003 |
| L10H11 | sink rises | **+0.052** | +0.003 | -0.010 | **+0.060** |
| L11H1 | sink rises | -0.012 | -0.003 | **-0.029** | +0.020 |
| L11H3 | sink rises | **+0.034** | -0.009 | +0.001 | **+0.042** |
| L2H1 | sink rises | **+0.024** | +0.003 | +0.007 | +0.014 |
| L2H6 | sink rises | +0.011 | -0.013 | **+0.044** | -0.020 |
| L3H6 | sink rises | **-0.210** | **-0.075** | **-0.197** | **+0.062** |
| L7H11 | sink rises | **-0.027** | **-0.034** | -0.001 | +0.008 |
| L10H4 | sink rises | **+0.023** | +0.001 | -0.005 | **+0.027** |
| L11H9 | sink rises | **-0.030** | -0.002 | **-0.054** | **+0.025** |
| L1H8 | sink falls | **+0.027** | -0.001 | **+0.028** | -0.000 |
| L2H0 | sink falls | **+0.056** | **-0.069** | **+0.101** | **+0.024** |
| L2H4 | sink falls | **+0.089** | +0.011 | **+0.121** | **-0.042** |
| L3H3 | sink falls | **-0.182** | **-0.049** | **-0.090** | **-0.043** |
| L6H4 | sink falls | **-0.021** | +0.003 | +0.002 | **-0.026** |
| L6H8 | sink falls | **-0.092** | **-0.074** | **-0.037** | +0.019 |
| L7H0 | sink falls | +0.018 | -0.011 | **+0.037** | -0.007 |
| L10H9 | sink falls | +0.019 | **-0.044** | -0.004 | **+0.067** |
| L4H0 | sink falls | **+0.084** | **+0.094** | **-0.032** | +0.022 |
| L5H2 | sink falls | **+0.021** | **-0.112** | **+0.050** | **+0.083** |
| L5H6 | sink falls | **-0.202** | **+0.049** | **-0.213** | **-0.038** |
| L8H2 | sink falls | **+0.047** | +0.006 | **+0.064** | **-0.023** |
| L9H8 | sink falls | **-0.048** | -0.004 | -0.010 | **-0.034** |
| L0H6 | label mix only | **-0.037** | -0.000 | **-0.097** | **+0.060** |
| L1H0 | label mix only | **-0.045** | -0.010 | **-0.053** | +0.018 |
| L3H2 | label mix only | **-0.151** | **-0.046** | **-0.132** | **+0.027** |
| L10H0 | label mix only | **-0.027** | -0.004 | **-0.031** | +0.008 |
| L0H1 | control | -0.000 | +0.000 | -0.001 | +0.001 |
| L4H11 | control | -0.007 | -0.003 | -0.004 | +0.000 |
| L5H1 | control | -0.001 | -0.000 | -0.000 | -0.000 |

## 5. Label-mix shift (TV distance)

Split-half TV floor within the `none` rows: rare 0.112, common 0.112.

Heads whose label-mix TV exceeds the floor, per contrast:

| base | phrase | placement | total | order | identity | form | placebo |
|---|---|---|---|---|---|---|---|
| common | bed | near | 33 | 9 | 25 | 13 | 0 |
| common | city | near | 32 | 14 | 20 | 14 | 0 |
| common | dog | near | 33 | 13 | 24 | 12 | 0 |
| common | door | near | 29 | 10 | 21 | 14 | 0 |
| common | hill | near | 27 | 10 | 25 | 13 | 0 |
| common | house | near | 26 | 10 | 23 | 15 | 0 |
| common | river | near | 33 | 10 | 31 | 14 | 0 |
| common | room | near | 31 | 10 | 21 | 15 | 0 |
| common | street | near | 33 | 10 | 26 | 13 | 0 |
| common | table | near | 28 | 6 | 22 | 13 | 0 |
| common | wall | near | 32 | 10 | 22 | 10 | 0 |
| common | window | near | 30 | 9 | 21 | 12 | 0 |
| rare | bed | near | 25 | 10 | 24 | 10 | 0 |
| rare | city | near | 31 | 15 | 25 | 9 | 0 |
| rare | dog | near | 33 | 11 | 25 | 8 | 0 |
| rare | door | near | 37 | 12 | 21 | 9 | 0 |
| rare | hill | near | 29 | 11 | 23 | 11 | 0 |
| rare | house | near | 29 | 8 | 24 | 10 | 0 |
| rare | river | near | 23 | 8 | 19 | 8 | 0 |
| rare | room | near | 32 | 12 | 16 | 9 | 0 |
| rare | street | near | 40 | 15 | 24 | 9 | 0 |
| rare | table | near | 34 | 8 | 24 | 9 | 0 |
| rare | wall | near | 38 | 12 | 22 | 8 | 0 |
| rare | window | near | 30 | 12 | 19 | 10 | 0 |

## 6. Pooled over phrases (sink_mass, total)

| base | placement | name | n_phrases | mean_of_means | min_mean | max_mean | n_meaningful_pos | n_meaningful_neg |
|---|---|---|---|---|---|---|---|---|
| rare | near | L2H4 | 12 | -0.288 | -0.331 | -0.247 | 0 | 12 |
| rare | near | L5H2 | 12 | -0.250 | -0.312 | -0.218 | 0 | 12 |
| rare | near | L5H6 | 12 | 0.232 | 0.124 | 0.309 | 12 | 0 |
| rare | near | L6H3 | 12 | -0.213 | -0.291 | -0.177 | 0 | 12 |
| common | near | L5H6 | 12 | 0.178 | 0.090 | 0.272 | 12 | 0 |
| rare | near | L4H0 | 12 | -0.167 | -0.367 | -0.068 | 0 | 12 |
| common | near | L4H0 | 12 | -0.150 | -0.323 | -0.044 | 0 | 12 |
| rare | near | L7H0 | 12 | 0.148 | 0.018 | 0.229 | 11 | 0 |
| common | near | L4H3 | 12 | 0.147 | 0.047 | 0.201 | 12 | 0 |
| rare | near | L5H3 | 12 | -0.141 | -0.172 | -0.105 | 0 | 12 |
| common | near | L3H7 | 12 | 0.141 | 0.090 | 0.206 | 12 | 0 |
| common | near | L6H8 | 12 | 0.133 | 0.069 | 0.238 | 12 | 0 |
| common | near | L6H0 | 12 | 0.131 | 0.057 | 0.224 | 12 | 0 |
| common | near | L8H6 | 12 | 0.128 | 0.048 | 0.286 | 12 | 0 |
| rare | near | L5H4 | 12 | -0.115 | -0.165 | -0.063 | 0 | 12 |

## 7. Additivity: does the combination add more than its single tokens?

Per row, interaction = (combo − none) − Σ(part − none); e.g. `table:nl` = table:on@29 + table:the@30 + table:table@31; `table:shuffled` = table:the@29 + table:on@30 + table:table@31. Depth values average the group's heads within each row; CI = bootstrap over rows. Metrics: sink_logit, sink_mass (sink_logit = log(p / (1 − p)), not squeezed by the probability bounds).

| metric | base | phrase | term | depth | mean | lo | hi | CI excludes 0 |
|---|---|---|---|---|---|---|---|---|
| sink_mass | common | table | nl | early | +0.0078 | +0.0061 | +0.0094 | True |
| sink_mass | common | table | nl | late | -0.0003 | -0.0056 | +0.0053 | False |
| sink_mass | common | table | shuffled | early | +0.0073 | +0.0058 | +0.0087 | True |
| sink_mass | common | table | shuffled | late | -0.0009 | -0.0055 | +0.0038 | False |
| sink_mass | rare | table | nl | early | +0.0199 | +0.0186 | +0.0211 | True |
| sink_mass | rare | table | nl | late | +0.0329 | +0.0284 | +0.0375 | True |
| sink_mass | rare | table | shuffled | early | +0.0212 | +0.0201 | +0.0224 | True |
| sink_mass | rare | table | shuffled | late | +0.0206 | +0.0163 | +0.0248 | True |
| sink_mass | common | table | nl−shuffled | early | +0.0005 | -0.0002 | +0.0012 | False |
| sink_mass | common | table | nl−shuffled | late | +0.0006 | -0.0021 | +0.0035 | False |
| sink_mass | rare | table | nl−shuffled | early | -0.0014 | -0.0021 | -0.0008 | True |
| sink_mass | rare | table | nl−shuffled | late | +0.0123 | +0.0097 | +0.0151 | True |
| sink_logit | common | table | nl | early | +0.0793 | +0.0697 | +0.0894 | True |
| sink_logit | common | table | nl | late | -0.0000 | -0.0297 | +0.0318 | False |
| sink_logit | common | table | shuffled | early | +0.0692 | +0.0599 | +0.0779 | True |
| sink_logit | common | table | shuffled | late | -0.0021 | -0.0276 | +0.0234 | False |
| sink_logit | rare | table | nl | early | +0.1491 | +0.1409 | +0.1568 | True |
| sink_logit | rare | table | nl | late | +0.1940 | +0.1675 | +0.2217 | True |
| sink_logit | rare | table | shuffled | early | +0.1523 | +0.1451 | +0.1592 | True |
| sink_logit | rare | table | shuffled | late | +0.1232 | +0.0987 | +0.1476 | True |
| sink_logit | common | table | nl−shuffled | early | +0.0101 | +0.0061 | +0.0144 | True |
| sink_logit | common | table | nl−shuffled | late | +0.0021 | -0.0134 | +0.0185 | False |
| sink_logit | rare | table | nl−shuffled | early | -0.0032 | -0.0073 | +0.0003 | False |
| sink_logit | rare | table | nl−shuffled | late | +0.0708 | +0.0553 | +0.0873 | True |

Every phrase's late-layer values are in section 9.

Order contrast against its additive prediction, across the 144 heads (every phrase):

| metric | base | phrase | pair | slope | r | r2 | mean_abs_observed | mean_abs_interaction |
|---|---|---|---|---|---|---|---|---|
| sink_mass | common | bed | nl−shuffled | 1.367 | 0.668 | 0.446 | 0.016 | 0.011 |
| sink_mass | common | city | nl−shuffled | 1.194 | 0.354 | 0.126 | 0.019 | 0.018 |
| sink_mass | common | dog | nl−shuffled | 2.523 | 0.549 | 0.301 | 0.025 | 0.022 |
| sink_mass | common | door | nl−shuffled | 1.283 | 0.493 | 0.244 | 0.020 | 0.016 |
| sink_mass | common | hill | nl−shuffled | 0.932 | 0.560 | 0.313 | 0.015 | 0.011 |
| sink_mass | common | house | nl−shuffled | 1.275 | 0.494 | 0.244 | 0.016 | 0.014 |
| sink_mass | common | river | nl−shuffled | 1.257 | 0.537 | 0.289 | 0.019 | 0.016 |
| sink_mass | common | room | nl−shuffled | 1.154 | 0.407 | 0.165 | 0.019 | 0.016 |
| sink_mass | common | street | nl−shuffled | 0.284 | 0.114 | 0.013 | 0.016 | 0.017 |
| sink_mass | common | table | nl−shuffled | 0.764 | 0.390 | 0.152 | 0.012 | 0.012 |
| sink_mass | common | wall | nl−shuffled | 0.611 | 0.368 | 0.135 | 0.017 | 0.014 |
| sink_mass | common | window | nl−shuffled | 1.208 | 0.559 | 0.313 | 0.018 | 0.014 |
| sink_mass | rare | bed | nl−shuffled | 0.190 | 0.122 | 0.015 | 0.013 | 0.014 |
| sink_mass | rare | city | nl−shuffled | 0.367 | 0.135 | 0.018 | 0.019 | 0.018 |
| sink_mass | rare | dog | nl−shuffled | 0.234 | 0.075 | 0.006 | 0.020 | 0.019 |
| sink_mass | rare | door | nl−shuffled | 0.526 | 0.176 | 0.031 | 0.021 | 0.021 |
| sink_mass | rare | hill | nl−shuffled | -0.097 | -0.048 | 0.002 | 0.020 | 0.021 |
| sink_mass | rare | house | nl−shuffled | -0.096 | -0.045 | 0.002 | 0.015 | 0.016 |
| sink_mass | rare | river | nl−shuffled | 0.076 | 0.051 | 0.003 | 0.016 | 0.018 |
| sink_mass | rare | room | nl−shuffled | 0.134 | 0.064 | 0.004 | 0.020 | 0.020 |
| sink_mass | rare | street | nl−shuffled | -0.062 | -0.027 | 0.001 | 0.022 | 0.023 |
| sink_mass | rare | table | nl−shuffled | 0.104 | 0.048 | 0.002 | 0.016 | 0.016 |
| sink_mass | rare | wall | nl−shuffled | 0.413 | 0.192 | 0.037 | 0.023 | 0.022 |
| sink_mass | rare | window | nl−shuffled | 0.645 | 0.354 | 0.125 | 0.017 | 0.015 |
| sink_logit | common | bed | nl−shuffled | 1.262 | 0.680 | 0.463 | 0.099 | 0.068 |
| sink_logit | common | city | nl−shuffled | 0.981 | 0.358 | 0.128 | 0.110 | 0.100 |
| sink_logit | common | dog | nl−shuffled | 1.689 | 0.473 | 0.223 | 0.144 | 0.126 |
| sink_logit | common | door | nl−shuffled | 1.190 | 0.492 | 0.242 | 0.115 | 0.094 |
| sink_logit | common | hill | nl−shuffled | 0.801 | 0.534 | 0.285 | 0.090 | 0.068 |
| sink_logit | common | house | nl−shuffled | 1.114 | 0.481 | 0.231 | 0.092 | 0.083 |
| sink_logit | common | river | nl−shuffled | 0.960 | 0.451 | 0.203 | 0.110 | 0.095 |
| sink_logit | common | room | nl−shuffled | 0.980 | 0.398 | 0.158 | 0.109 | 0.092 |
| sink_logit | common | street | nl−shuffled | 0.273 | 0.127 | 0.016 | 0.093 | 0.097 |
| sink_logit | common | table | nl−shuffled | 0.670 | 0.384 | 0.148 | 0.071 | 0.070 |
| sink_logit | common | wall | nl−shuffled | 0.529 | 0.332 | 0.110 | 0.106 | 0.086 |
| sink_logit | common | window | nl−shuffled | 1.047 | 0.545 | 0.298 | 0.106 | 0.085 |
| sink_logit | rare | bed | nl−shuffled | 0.103 | 0.067 | 0.004 | 0.080 | 0.089 |
| sink_logit | rare | city | nl−shuffled | 0.534 | 0.241 | 0.058 | 0.110 | 0.106 |
| sink_logit | rare | dog | nl−shuffled | 0.374 | 0.132 | 0.017 | 0.115 | 0.114 |
| sink_logit | rare | door | nl−shuffled | 0.563 | 0.212 | 0.045 | 0.125 | 0.122 |
| sink_logit | rare | hill | nl−shuffled | -0.178 | -0.091 | 0.008 | 0.116 | 0.126 |
| sink_logit | rare | house | nl−shuffled | -0.046 | -0.024 | 0.001 | 0.086 | 0.094 |
| sink_logit | rare | river | nl−shuffled | 0.016 | 0.011 | 0.000 | 0.097 | 0.108 |
| sink_logit | rare | room | nl−shuffled | 0.270 | 0.129 | 0.017 | 0.122 | 0.118 |
| sink_logit | rare | street | nl−shuffled | 0.076 | 0.036 | 0.001 | 0.130 | 0.136 |
| sink_logit | rare | table | nl−shuffled | 0.199 | 0.095 | 0.009 | 0.099 | 0.097 |
| sink_logit | rare | wall | nl−shuffled | 0.447 | 0.204 | 0.042 | 0.148 | 0.139 |
| sink_logit | rare | window | nl−shuffled | 0.607 | 0.363 | 0.132 | 0.100 | 0.088 |

Heads with a BH-significant interaction on sink_mass (q over every interaction test of the metric):

| base | phrase | combo | significant | late + | late − | largest |interaction| |
|---|---|---|---|---|---|---|
| common | table | nl | 107 | 25 | 27 | L5H2 (-0.192), L3H7 (+0.155), L5H6 (+0.148) |
| common | table | shuffled | 103 | 21 | 24 | L5H2 (-0.216), L4H0 (+0.180), L5H6 (+0.158) |
| common | house | nl | 103 | 22 | 22 | L4H0 (+0.161), L5H2 (-0.146), L3H3 (+0.138) |
| common | house | shuffled | 107 | 12 | 35 | L4H0 (+0.204), L5H2 (-0.196), L5H6 (+0.160) |
| common | door | nl | 109 | 30 | 22 | L5H2 (-0.160), L4H0 (+0.155), L3H2 (+0.128) |
| common | door | shuffled | 107 | 11 | 43 | L5H2 (-0.249), L4H0 (+0.178), L4H5 (+0.157) |
| common | bed | nl | 103 | 47 | 5 | L5H6 (+0.175), L3H7 (+0.144), L4H0 (+0.131) |
| common | bed | shuffled | 117 | 41 | 16 | L5H6 (+0.187), L5H2 (-0.171), L4H0 (+0.171) |
| common | river | nl | 101 | 40 | 13 | L5H6 (+0.150), L7H11 (+0.144), L7H7 (+0.124) |
| common | river | shuffled | 110 | 30 | 26 | L5H6 (+0.145), L5H2 (-0.134), L3H7 (+0.097) |
| common | wall | nl | 100 | 36 | 14 | L3H3 (+0.173), L5H2 (-0.172), L3H7 (+0.167) |
| common | wall | shuffled | 93 | 26 | 14 | L5H2 (-0.191), L3H7 (+0.138), L3H2 (+0.113) |
| common | room | nl | 110 | 25 | 26 | L5H6 (+0.245), L5H2 (-0.191), L8H7 (+0.171) |
| common | room | shuffled | 111 | 17 | 38 | L5H2 (-0.218), L4H0 (+0.200), L4H5 (+0.175) |
| common | hill | nl | 97 | 29 | 12 | L5H2 (-0.141), L7H11 (+0.127), L5H6 (+0.120) |
| common | hill | shuffled | 106 | 30 | 19 | L5H2 (-0.147), L5H6 (+0.128), L4H0 (+0.114) |
| common | city | nl | 102 | 20 | 25 | L5H2 (-0.142), L3H3 (+0.136), L4H0 (+0.116) |
| common | city | shuffled | 116 | 7 | 49 | L5H2 (-0.229), L4H0 (+0.174), L6H8 (-0.152) |
| common | dog | nl | 102 | 42 | 10 | L8H6 (+0.162), L7H11 (+0.159), L7H7 (+0.157) |
| common | dog | shuffled | 92 | 8 | 30 | L5H2 (-0.199), L6H8 (-0.156), L4H0 (+0.144) |
| common | street | nl | 124 | 21 | 42 | L5H6 (+0.252), L5H2 (-0.231), L8H7 (+0.216) |
| common | street | shuffled | 119 | 13 | 49 | L5H2 (-0.252), L4H0 (+0.181), L3H3 (+0.165) |
| common | window | nl | 101 | 30 | 13 | L5H6 (+0.169), L5H2 (-0.159), L3H3 (+0.124) |
| common | window | shuffled | 114 | 17 | 37 | L5H2 (-0.224), L4H5 (+0.162), L4H0 (+0.132) |
| rare | table | nl | 127 | 56 | 9 | L5H6 (+0.261), L4H0 (+0.220), L6H0 (+0.176) |
| rare | table | shuffled | 119 | 54 | 7 | L5H6 (+0.280), L4H0 (+0.257), L3H3 (+0.134) |
| rare | house | nl | 120 | 53 | 8 | L4H0 (+0.205), L5H6 (+0.186), L6H0 (+0.162) |
| rare | house | shuffled | 122 | 54 | 9 | L4H0 (+0.295), L5H6 (+0.211), L4H5 (+0.135) |
| rare | door | nl | 126 | 59 | 8 | L4H0 (+0.204), L6H0 (+0.183), L4H5 (+0.176) |
| rare | door | shuffled | 117 | 37 | 19 | L4H0 (+0.225), L4H5 (+0.182), L3H8 (+0.114) |
| rare | bed | nl | 126 | 57 | 9 | L5H6 (+0.188), L4H0 (+0.183), L4H5 (+0.148) |
| rare | bed | shuffled | 126 | 59 | 6 | L4H0 (+0.210), L5H6 (+0.159), L4H1 (+0.142) |
| rare | river | nl | 122 | 54 | 11 | L5H6 (+0.231), L6H0 (+0.191), L8H7 (+0.115) |
| rare | river | shuffled | 113 | 51 | 8 | L5H6 (+0.157), L4H0 (+0.087), L4H1 (+0.071) |
| rare | wall | nl | 127 | 62 | 8 | L4H0 (+0.246), L6H0 (+0.220), L5H6 (+0.208) |
| rare | wall | shuffled | 121 | 57 | 5 | L4H0 (+0.227), L4H5 (+0.133), L3H3 (+0.121) |
| rare | room | nl | 126 | 51 | 10 | L5H6 (+0.261), L6H0 (+0.227), L4H0 (+0.185) |
| rare | room | shuffled | 117 | 34 | 21 | L4H0 (+0.258), L4H5 (+0.157), L5H4 (+0.133) |
| rare | hill | nl | 128 | 57 | 10 | L6H0 (+0.192), L5H6 (+0.174), L7H0 (+0.153) |
| rare | hill | shuffled | 127 | 58 | 7 | L5H6 (+0.144), L4H0 (+0.139), L4H5 (+0.130) |
| rare | city | nl | 112 | 37 | 15 | L3H3 (+0.142), L6H0 (+0.126), L4H0 (+0.104) |
| rare | city | shuffled | 116 | 17 | 39 | L4H0 (+0.187), L3H3 (+0.113), L5H4 (+0.104) |
| rare | dog | nl | 112 | 22 | 29 | L5H0 (+0.091), L7H7 (+0.080), L6H10 (+0.075) |
| rare | dog | shuffled | 104 | 20 | 32 | L4H0 (+0.172), L4H5 (+0.100), L4H1 (+0.095) |
| rare | street | nl | 127 | 60 | 8 | L5H6 (+0.300), L6H0 (+0.279), L3H3 (+0.229) |
| rare | street | shuffled | 126 | 57 | 9 | L5H6 (+0.232), L4H0 (+0.197), L3H3 (+0.187) |
| rare | window | nl | 126 | 59 | 7 | L6H0 (+0.193), L5H6 (+0.183), L7H0 (+0.162) |
| rare | window | shuffled | 127 | 58 | 9 | L4H0 (+0.186), L4H5 (+0.178), L5H4 (+0.106) |

## 8. Surprisal: does sink_mass track how predictable the slot tokens are?

Slot surprisal = −log p of the tokens at positions [29, 30, 31] given everything before them (TL), summed. Slope = within-row regression across every variant (both sides demeaned per row), per nat; CI = bootstrap over rows.

| base | phrase | depth | slope | lo | hi |
|---|---|---|---|---|---|
| common | bed | early | -0.00031 | -0.00041 | -0.00021 |
| common | bed | late | -0.00213 | -0.00253 | -0.00175 |
| common | city | early | +0.00040 | +0.00029 | +0.00051 |
| common | city | late | -0.00100 | -0.00131 | -0.00070 |
| common | dog | early | +0.00065 | +0.00051 | +0.00079 |
| common | dog | late | -0.00149 | -0.00186 | -0.00112 |
| common | door | early | +0.00042 | +0.00033 | +0.00052 |
| common | door | late | -0.00052 | -0.00078 | -0.00025 |
| common | hill | early | +0.00014 | +0.00003 | +0.00024 |
| common | hill | late | -0.00150 | -0.00187 | -0.00115 |
| common | house | early | +0.00040 | +0.00032 | +0.00048 |
| common | house | late | -0.00033 | -0.00058 | -0.00007 |
| common | river | early | +0.00045 | +0.00035 | +0.00054 |
| common | river | late | -0.00120 | -0.00148 | -0.00091 |
| common | room | early | +0.00014 | +0.00003 | +0.00025 |
| common | room | late | -0.00112 | -0.00143 | -0.00080 |
| common | street | early | +0.00030 | +0.00022 | +0.00039 |
| common | street | late | -0.00061 | -0.00086 | -0.00034 |
| common | table | early | +0.00043 | +0.00034 | +0.00052 |
| common | table | late | -0.00038 | -0.00066 | -0.00010 |
| common | wall | early | +0.00007 | -0.00002 | +0.00015 |
| common | wall | late | -0.00142 | -0.00171 | -0.00112 |
| common | window | early | +0.00026 | +0.00014 | +0.00037 |
| common | window | late | -0.00149 | -0.00184 | -0.00111 |
| rare | bed | early | +0.00018 | +0.00011 | +0.00024 |
| rare | bed | late | -0.00052 | -0.00073 | -0.00030 |
| rare | city | early | +0.00085 | +0.00078 | +0.00092 |
| rare | city | late | +0.00063 | +0.00042 | +0.00086 |
| rare | dog | early | +0.00127 | +0.00118 | +0.00136 |
| rare | dog | late | +0.00109 | +0.00083 | +0.00137 |
| rare | door | early | +0.00038 | +0.00033 | +0.00044 |
| rare | door | late | -0.00075 | -0.00094 | -0.00055 |
| rare | hill | early | +0.00048 | +0.00042 | +0.00055 |
| rare | hill | late | -0.00067 | -0.00088 | -0.00044 |
| rare | house | early | +0.00048 | +0.00042 | +0.00054 |
| rare | house | late | -0.00052 | -0.00070 | -0.00033 |
| rare | river | early | +0.00075 | +0.00068 | +0.00083 |
| rare | river | late | -0.00012 | -0.00035 | +0.00011 |
| rare | room | early | +0.00033 | +0.00025 | +0.00040 |
| rare | room | late | -0.00102 | -0.00123 | -0.00081 |
| rare | street | early | +0.00031 | +0.00025 | +0.00037 |
| rare | street | late | -0.00143 | -0.00162 | -0.00124 |
| rare | table | early | +0.00041 | +0.00035 | +0.00047 |
| rare | table | late | -0.00093 | -0.00112 | -0.00074 |
| rare | wall | early | +0.00024 | +0.00018 | +0.00030 |
| rare | wall | late | -0.00175 | -0.00193 | -0.00156 |
| rare | window | early | +0.00039 | +0.00032 | +0.00045 |
| rare | window | late | -0.00105 | -0.00126 | -0.00084 |

Heads whose slope CI excludes 0, by sign (summed over phrases):

| base | depth | slope < 0 | slope > 0 |
|---|---|---|---|
| common | early | 320 | 443 |
| common | late | 505 | 244 |
| rare | early | 357 | 430 |
| rare | late | 478 | 254 |

Per-variant means (descriptive):

| base | phrase | variant | mean_slot_nll | mean_sink_mass_early | mean_sink_mass_late | mean_query_nll |
|---|---|---|---|---|---|---|
| common | table | none | 26.951 | 0.330 | 0.600 | 9.124 |
| common | table | nl | 16.490 | 0.331 | 0.592 | 9.577 |
| common | table | shuffled | 23.017 | 0.330 | 0.583 | 9.458 |
| common | table | matched | 30.498 | 0.336 | 0.590 | 9.226 |
| common | table | matched_b | 30.734 | 0.336 | 0.590 | 9.202 |
| common | table | table:on@29 | 24.417 | 0.325 | 0.596 | 9.109 |
| common | table | table:the@30 | 23.792 | 0.323 | 0.607 | 9.161 |
| common | table | table:table@31 | 28.900 | 0.335 | 0.589 | 9.003 |
| common | table | table:the@29 | 23.842 | 0.319 | 0.595 | 9.117 |
| common | table | table:on@30 | 24.496 | 0.327 | 0.599 | 9.052 |
| rare | table | none | 37.781 | 0.330 | 0.591 | 12.590 |
| rare | table | nl | 16.895 | 0.318 | 0.607 | 12.272 |
| rare | table | shuffled | 25.087 | 0.317 | 0.590 | 12.204 |
| rare | table | matched | 32.155 | 0.325 | 0.576 | 12.470 |
| rare | table | matched_b | 32.314 | 0.327 | 0.580 | 12.418 |
| rare | table | table:on@29 | 32.800 | 0.319 | 0.590 | 12.673 |
| rare | table | table:the@30 | 31.522 | 0.307 | 0.583 | 12.989 |
| rare | table | table:table@31 | 35.563 | 0.333 | 0.584 | 12.117 |
| rare | table | table:the@29 | 32.155 | 0.307 | 0.585 | 12.715 |
| rare | table | table:on@30 | 32.328 | 0.316 | 0.583 | 12.836 |

## 9. Across phrases

Phrases as the unit: pooled_mean = mean of the per-phrase depth-group interactions, CI from bootstrapping phrases; n_pos / n_neg = phrases whose own row-bootstrap CI excludes 0 on that side; spearman = across phrases, the term against the surprisal gap NLL(shuffled) − NLL(nl).

| metric | base | term | depth | n_phrases | pooled_mean | lo | hi | n_pos | n_neg | spearman_rho | spearman_p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sink_mass | common | nl | early | 12 | +0.0091 | +0.0078 | +0.0105 | 12 | 0 | +0.4266 | +0.1667 |
| sink_mass | common | nl | late | 12 | +0.0093 | +0.0040 | +0.0149 | 7 | 1 | -0.7273 | +0.0074 |
| sink_mass | common | shuffled | early | 12 | +0.0069 | +0.0060 | +0.0081 | 12 | 0 | +0.0769 | +0.8122 |
| sink_mass | common | shuffled | late | 12 | -0.0037 | -0.0093 | +0.0019 | 3 | 6 | -0.6224 | +0.0307 |
| sink_mass | rare | nl | early | 12 | +0.0174 | +0.0156 | +0.0191 | 12 | 0 | +0.4965 | +0.1006 |
| sink_mass | rare | nl | late | 12 | +0.0327 | +0.0239 | +0.0409 | 11 | 0 | +0.5944 | +0.0415 |
| sink_mass | rare | shuffled | early | 12 | +0.0168 | +0.0150 | +0.0185 | 12 | 0 | +0.1888 | +0.5567 |
| sink_mass | rare | shuffled | late | 12 | +0.0159 | +0.0086 | +0.0228 | 10 | 2 | +0.3007 | +0.3423 |
| sink_mass | common | nl−shuffled | early | 12 | +0.0022 | +0.0012 | +0.0033 | 8 | 0 | +0.6014 | +0.0386 |
| sink_mass | common | nl−shuffled | late | 12 | +0.0131 | +0.0093 | +0.0169 | 11 | 0 | -0.1329 | +0.6806 |
| sink_mass | rare | nl−shuffled | early | 12 | +0.0007 | -0.0007 | +0.0021 | 6 | 4 | +0.5245 | +0.0800 |
| sink_mass | rare | nl−shuffled | late | 12 | +0.0168 | +0.0117 | +0.0214 | 11 | 0 | +0.4266 | +0.1667 |
| sink_logit | common | nl | early | 12 | +0.0837 | +0.0738 | +0.0932 | 12 | 0 | +0.6154 | +0.0332 |
| sink_logit | common | nl | late | 12 | +0.0560 | +0.0233 | +0.0902 | 8 | 1 | -0.7203 | +0.0082 |
| sink_logit | common | shuffled | early | 12 | +0.0638 | +0.0572 | +0.0705 | 12 | 0 | +0.1958 | +0.5419 |
| sink_logit | common | shuffled | late | 12 | -0.0196 | -0.0529 | +0.0128 | 3 | 6 | -0.6573 | +0.0202 |
| sink_logit | rare | nl | early | 12 | +0.1307 | +0.1167 | +0.1432 | 12 | 0 | +0.5455 | +0.0666 |
| sink_logit | rare | nl | late | 12 | +0.1963 | +0.1466 | +0.2428 | 11 | 0 | +0.5944 | +0.0415 |
| sink_logit | rare | shuffled | early | 12 | +0.1191 | +0.1069 | +0.1308 | 12 | 0 | +0.3706 | +0.2356 |
| sink_logit | rare | shuffled | late | 12 | +0.0977 | +0.0557 | +0.1370 | 10 | 2 | +0.2517 | +0.4299 |
| sink_logit | common | nl−shuffled | early | 12 | +0.0200 | +0.0132 | +0.0272 | 12 | 0 | +0.7692 | +0.0034 |
| sink_logit | common | nl−shuffled | late | 12 | +0.0756 | +0.0547 | +0.0970 | 11 | 0 | -0.2028 | +0.5273 |
| sink_logit | rare | nl−shuffled | early | 12 | +0.0116 | +0.0024 | +0.0211 | 8 | 2 | +0.5245 | +0.0800 |
| sink_logit | rare | nl−shuffled | late | 12 | +0.0985 | +0.0683 | +0.1257 | 11 | 0 | +0.4615 | +0.1309 |

Late-layer interaction for every phrase (* = row-bootstrap CI excludes 0):

| metric | base | phrase | nl | shuffled | nl−shuffled | nll gap (shuffled − nl) |
|---|---|---|---|---|---|---|
| sink_mass | common | on the table | -0.0003 | -0.0009 | +0.0006 | +6.53 |
| sink_mass | common | in the house | +0.0050 | -0.0086* | +0.0135* | +5.09 |
| sink_mass | common | at the door | +0.0051 | -0.0119* | +0.0170* | +6.82 |
| sink_mass | common | under the bed | +0.0227* | +0.0130* | +0.0097* | +4.75 |
| sink_mass | common | near the river | +0.0192* | +0.0034 | +0.0158* | +4.79 |
| sink_mass | common | behind the wall | +0.0165* | +0.0089* | +0.0076* | +5.93 |
| sink_mass | common | into the room | +0.0070* | -0.0091* | +0.0161* | +7.01 |
| sink_mass | common | over the hill | +0.0148* | +0.0071* | +0.0077* | +4.43 |
| sink_mass | common | from the city | -0.0002 | -0.0183* | +0.0180* | +6.07 |
| sink_mass | common | with the dog | +0.0208* | -0.0077* | +0.0285* | +4.45 |
| sink_mass | common | across the street | -0.0094* | -0.0167* | +0.0073* | +7.65 |
| sink_mass | common | through the window | +0.0106* | -0.0043 | +0.0149* | +6.13 |
| sink_mass | rare | on the table | +0.0329* | +0.0206* | +0.0123* | +8.19 |
| sink_mass | rare | in the house | +0.0303* | +0.0200* | +0.0102* | +5.37 |
| sink_mass | rare | at the door | +0.0365* | +0.0093* | +0.0272* | +8.95 |
| sink_mass | rare | under the bed | +0.0289* | +0.0283* | +0.0006 | +6.75 |
| sink_mass | rare | near the river | +0.0295* | +0.0146* | +0.0148* | +5.58 |
| sink_mass | rare | behind the wall | +0.0536* | +0.0247* | +0.0289* | +6.16 |
| sink_mass | rare | into the room | +0.0342* | +0.0111* | +0.0231* | +7.68 |
| sink_mass | rare | over the hill | +0.0354* | +0.0211* | +0.0142* | +6.21 |
| sink_mass | rare | from the city | +0.0122* | -0.0106* | +0.0227* | +6.95 |
| sink_mass | rare | with the dog | -0.0014 | -0.0078* | +0.0064* | +5.27 |
| sink_mass | rare | across the street | +0.0597* | +0.0326* | +0.0271* | +9.92 |
| sink_mass | rare | through the window | +0.0409* | +0.0268* | +0.0142* | +8.29 |
| sink_logit | common | on the table | -0.0000 | -0.0021 | +0.0021 | +6.53 |
| sink_logit | common | in the house | +0.0264 | -0.0500* | +0.0764* | +5.09 |
| sink_logit | common | at the door | +0.0392* | -0.0578* | +0.0970* | +6.82 |
| sink_logit | common | under the bed | +0.1400* | +0.0762* | +0.0637* | +4.75 |
| sink_logit | common | near the river | +0.1186* | +0.0244 | +0.0942* | +4.79 |
| sink_logit | common | behind the wall | +0.0951* | +0.0509* | +0.0442* | +5.93 |
| sink_logit | common | into the room | +0.0376* | -0.0531* | +0.0908* | +7.01 |
| sink_logit | common | over the hill | +0.0903* | +0.0432* | +0.0471* | +4.43 |
| sink_logit | common | from the city | -0.0005 | -0.0982* | +0.0977* | +6.07 |
| sink_logit | common | with the dog | +0.1216* | -0.0383* | +0.1599* | +4.45 |
| sink_logit | common | across the street | -0.0630* | -0.1052* | +0.0422* | +7.65 |
| sink_logit | common | through the window | +0.0666* | -0.0252 | +0.0918* | +6.13 |
| sink_logit | rare | on the table | +0.1940* | +0.1232* | +0.0708* | +8.19 |
| sink_logit | rare | in the house | +0.1807* | +0.1262* | +0.0546* | +5.37 |
| sink_logit | rare | at the door | +0.2239* | +0.0619* | +0.1620* | +8.95 |
| sink_logit | rare | under the bed | +0.1769* | +0.1722* | +0.0047 | +6.75 |
| sink_logit | rare | near the river | +0.1791* | +0.0899* | +0.0893* | +5.58 |
| sink_logit | rare | behind the wall | +0.3178* | +0.1455* | +0.1723* | +6.16 |
| sink_logit | rare | into the room | +0.1966* | +0.0638* | +0.1327* | +7.68 |
| sink_logit | rare | over the hill | +0.2114* | +0.1266* | +0.0848* | +6.21 |
| sink_logit | rare | from the city | +0.0782* | -0.0522* | +0.1303* | +6.95 |
| sink_logit | rare | with the dog | +0.0048 | -0.0327* | +0.0375* | +5.27 |
| sink_logit | rare | across the street | +0.3494* | +0.1910* | +0.1584* | +9.92 |
| sink_logit | rare | through the window | +0.2424* | +0.1574* | +0.0850* | +8.29 |

## Figures

- [rare_table_near_sink_mass_contrasts](figures/rare_table_near_sink_mass_contrasts.png)
- [rare_table_sink_mass_tracked_forest](figures/rare_table_sink_mass_tracked_forest.png)
- [rare_table_near_slot_mass_contrasts](figures/rare_table_near_slot_mass_contrasts.png)
- [rare_table_slot_mass_tracked_forest](figures/rare_table_slot_mass_tracked_forest.png)
- [rare_table_sink_mass_layer_profile](figures/rare_table_sink_mass_layer_profile.png)
- [common_table_near_sink_mass_contrasts](figures/common_table_near_sink_mass_contrasts.png)
- [common_table_sink_mass_tracked_forest](figures/common_table_sink_mass_tracked_forest.png)
- [common_table_near_slot_mass_contrasts](figures/common_table_near_slot_mass_contrasts.png)
- [common_table_slot_mass_tracked_forest](figures/common_table_slot_mass_tracked_forest.png)
- [common_table_sink_mass_layer_profile](figures/common_table_sink_mass_layer_profile.png)
- [common_table_near_additivity](figures/common_table_near_additivity.png)
- [rare_table_near_additivity](figures/rare_table_near_additivity.png)
- [phrase_terms_sink_mass](figures/phrase_terms_sink_mass.png)
- [phrase_terms_sink_logit](figures/phrase_terms_sink_logit.png)
- [surprisal_variants](figures/surprisal_variants.png)
- [meaningful_counts_sink_mass](figures/meaningful_counts_sink_mass.png)
- [meaningful_counts_slot_mass](figures/meaningful_counts_slot_mass.png)

## Caveats

- 12 phrases from one template (pronoun verb object) are a small, hand-picked sample; CIs over phrases describe these phrases, not language in general.
- A slot at position p < 32 changes every later position's residual stream, so a contrast measures the slot tokens' total effect at the query (as keys and through earlier heads), not a single route.
- `matched` controls match surface form and id class, not frequency or meaning; `identity` therefore mixes everything else that distinguishes the phrase's tokens from same-form random words.
