# exp2: a natural-language phrase in random-token context

- Run `20261004-152303_exp2`. Bases: rare [1000, 39999], common [256, 999], 500 rows each, BOS + 32 random ids, no repeats, no phrase ids. Metric at the final query (position 32).
- Phrases: `sushi` = ' I like sushi' [314, 588, 36324] (shuffled [588, 314, 36324]), `water` = ' we need water' [356, 761, 1660] (shuffled [761, 356, 1660]), `pizza` = ' she ate pizza' [673, 15063, 14256] (shuffled [15063, 673, 14256]), `music` = ' they love music' [484, 1842, 2647] (shuffled [1842, 484, 2647]), `coffee` = ' he drinks coffee' [339, 11758, 6891] (shuffled [11758, 339, 6891]), `chess` = ' you play chess' [345, 711, 19780] (shuffled [711, 345, 19780]), `rain` = ' I hate rain' [314, 5465, 6290] (shuffled [5465, 314, 6290]), `cars` = ' we sell cars' [356, 3677, 5006] (shuffled [3677, 356, 5006]), `songs` = ' she wrote songs' [673, 2630, 7259] (shuffled [2630, 673, 7259]), `rice` = ' they grow rice' [484, 1663, 11464] (shuffled [1663, 484, 11464]), `bikes` = ' he fixed bikes' [339, 5969, 16715] (shuffled [5969, 339, 16715]), `money` = ' you want money' [345, 765, 1637] (shuffled [765, 345, 1637]). Placements (model positions): near [29, 30, 31].
- Controls: `shuffled` (same ids, scrambled), `matched` / `matched_b` (per row, random ids with the same leading_space, is_alpha, is_capitalized and id class as each phrase token; pool sizes {'sushi': [43, 280, 16019], 'water': [279, 279, 16019], 'pizza': [280, 16018, 16018], 'music': [280, 16018, 16018], 'coffee': [280, 16018, 16018], 'chess': [279, 279, 16019], 'rain': [43, 16018, 16018], 'cars': [280, 16018, 16018], 'songs': [280, 16018, 16018], 'rice': [280, 16018, 16018], 'bikes': [280, 16018, 16018], 'money': [279, 279, 16019]}).
- Contrasts (paired per row, a − b): total = nl − none, order = nl − shuffled, identity = shuffled − matched, form = matched − none, placebo = matched − matched_b. total = order + identity + form.
- Meaningful = BH q ≤ 0.05 (Wilcoxon signed-rank, over every test of a metric) and |Δ| ≥ the metric's threshold {'sink_mass': 0.02, 'slot_mass': 0.02, 'slot_logratio': 0.1, 'prev_mass': 0.02, 'entropy': 0.05}. Bold = meaningful. Predictions were written before any data: [PREDICTIONS.md](PREDICTIONS.md).

## Checks

| check | backend | result | detail |
|---|---|---|---|
| Attention rows sum to 1 | TL | PASS | max abs(sum − 1) = 4.6e-07 |
| Attention rows sum to 1 | HF | PASS | max abs(sum − 1) = 4.7e-07 |
| Determinism (first 50 re-extracted) | TL | PASS | bit-identical |
| Determinism (first 50 re-extracted) | HF | PASS | bit-identical |
| Design invariants (no repeats, pairing, slot contents, pools) | - | PASS | checked before extraction |
| `none` rows replicate exp1 (rare) | TL | PASS | per-head mean sink_mass r = 0.9995 |
| `none` rows replicate exp1 (common) | TL | PASS | per-head mean sink_mass r = 0.9997 |
| Cross-library | TL vs HF | PASS | max |Δ| probability metrics 6.4e-05; labels agree 100.00%; meaningful decisions (sink_mass) agree 100.00%; max |Δ effect size| 2.6e-02 |

## 1. Pre-registered predictions

| id | kind | result | detail |
|---|---|---|---|
| A1 | pooled_term_sign | PASS | common: +0.0125 [+0.0101, +0.0152] over 12 phrases; rare: +0.0112 [+0.0084, +0.0141] over 12 phrases |
| A2a | pooled_term_sign | PASS | common: +0.0033 [+0.0003, +0.0065] over 12 phrases; rare: +0.0072 [+0.0045, +0.0102] over 12 phrases |
| A2b | pooled_term_sign | PASS | common: +0.0112 [+0.0090, +0.0134] over 12 phrases; rare: +0.0097 [+0.0072, +0.0123] over 12 phrases |
| A3 | pooled_ratio_at_least | FAIL | common: VO +0.0125 / nl +0.0270 = 0.46; rare: VO +0.0112 / nl +0.0183 = 0.61 |
| A4 | pooled_abs_less | PASS | common: |triple| 0.0040 vs VO +0.0125; rare: |triple| 0.0014 vs VO +0.0112 |
| A5 | familiarity_rho_sign | PASS | common: rho -0.27 (p 0.391, n 12); rare: rho -0.55 (p 0.067, n 12) |
| A6a | controls | PASS | L4H11 prev_mass: min over cells 0.949 (floor 0.94); L5H1 sink_mass: min over cells 0.971 (floor 0.9); L0H1 mode Self in 312/312 cells |
| A6b | few_meaningful | PASS | common/bikes/near: 0 (limit 1); common/cars/near: 0 (limit 1); common/chess/near: 0 (limit 1); common/coffee/near: 0 (limit 1); common/money/near: 0 (limit 1); common/music/near: 0 (limit 1); common/pizza/near: 0 (limit 1); common/rain/near: 0 (limit 1); common/rice/near: 0 (limit 1); common/songs/near: 0 (limit 1); common/sushi/near: 0 (limit 1); common/water/near: 0 (limit 1); rare/bikes/near: 1 (limit 1): L4H0; rare/cars/near: 0 (limit 1); rare/chess/near: 0 (limit 1); rare/coffee/near: 0 (limit 1); rare/money/near: 0 (limit 1); rare/music/near: 0 (limit 1); rare/pizza/near: 0 (limit 1); rare/rain/near: 0 (limit 1); rare/rice/near: 1 (limit 1): L7H0; rare/songs/near: 0 (limit 1); rare/sushi/near: 0 (limit 1); rare/water/near: 0 (limit 1) |

Every phrase appears in every summary table. The long per-head tables (sections 3, 4, the head list in 7 and the per-variant means in 8) and the per-phrase figures cover only ['sushi']; the full per-head results for every phrase are in effects.parquet, additivity_heads.parquet and surprisal_*.parquet.

## 2. Meaningful heads per contrast: sink_mass

Out of 144 heads. The placebo row is the noise floor.

| base | phrase | placement | total | order | identity | form | placebo |
|---|---|---|---|---|---|---|---|
| common | bikes | near | 66 | 52 | 67 | 50 | 0 |
| common | cars | near | 68 | 50 | 61 | 50 | 0 |
| common | chess | near | 79 | 47 | 69 | 45 | 0 |
| common | coffee | near | 65 | 70 | 67 | 49 | 0 |
| common | money | near | 70 | 28 | 53 | 44 | 0 |
| common | music | near | 77 | 51 | 59 | 49 | 0 |
| common | pizza | near | 70 | 58 | 59 | 58 | 0 |
| common | rain | near | 73 | 62 | 72 | 50 | 0 |
| common | rice | near | 74 | 56 | 69 | 47 | 0 |
| common | songs | near | 77 | 68 | 68 | 51 | 0 |
| common | sushi | near | 74 | 62 | 61 | 37 | 0 |
| common | water | near | 69 | 67 | 59 | 46 | 0 |
| rare | bikes | near | 63 | 38 | 37 | 26 | 1 |
| rare | cars | near | 70 | 35 | 59 | 29 | 0 |
| rare | chess | near | 75 | 26 | 75 | 42 | 0 |
| rare | coffee | near | 74 | 46 | 60 | 26 | 0 |
| rare | money | near | 66 | 17 | 60 | 43 | 0 |
| rare | music | near | 74 | 30 | 68 | 26 | 0 |
| rare | pizza | near | 78 | 30 | 78 | 26 | 0 |
| rare | rain | near | 67 | 58 | 77 | 26 | 0 |
| rare | rice | near | 62 | 32 | 56 | 25 | 1 |
| rare | songs | near | 78 | 43 | 80 | 25 | 0 |
| rare | sushi | near | 77 | 29 | 72 | 47 | 0 |
| rare | water | near | 68 | 56 | 54 | 43 | 0 |

Effect sizes (sink_mass): 

| base | phrase | placement | total median |Δ| | order median |Δ| | identity median |Δ| | form median |Δ| | placebo median |Δ| | largest total |
|---|---|---|---|---|---|---|---|---|
| common | bikes | near | 0.0174 | 0.0123 | 0.0188 | 0.0114 | 0.0020 | L4H3 (+0.238) |
| common | cars | near | 0.0187 | 0.0103 | 0.0148 | 0.0105 | 0.0019 | L4H3 (+0.311) |
| common | chess | near | 0.0232 | 0.0115 | 0.0182 | 0.0114 | 0.0018 | L4H3 (+0.295) |
| common | coffee | near | 0.0140 | 0.0194 | 0.0167 | 0.0116 | 0.0026 | L4H3 (+0.333) |
| common | money | near | 0.0189 | 0.0078 | 0.0139 | 0.0113 | 0.0030 | L4H3 (+0.302) |
| common | music | near | 0.0232 | 0.0129 | 0.0154 | 0.0117 | 0.0022 | L4H3 (+0.319) |
| common | pizza | near | 0.0185 | 0.0127 | 0.0160 | 0.0129 | 0.0024 | L4H3 (+0.307) |
| common | rain | near | 0.0204 | 0.0159 | 0.0190 | 0.0097 | 0.0018 | L5H6 (+0.283) |
| common | rice | near | 0.0225 | 0.0115 | 0.0187 | 0.0114 | 0.0018 | L7H0 (+0.257) |
| common | songs | near | 0.0234 | 0.0165 | 0.0157 | 0.0123 | 0.0021 | L4H3 (+0.302) |
| common | sushi | near | 0.0226 | 0.0122 | 0.0159 | 0.0068 | 0.0015 | L7H0 (+0.244) |
| common | water | near | 0.0188 | 0.0181 | 0.0159 | 0.0094 | 0.0018 | L5H6 (+0.311) |
| rare | bikes | near | 0.0152 | 0.0122 | 0.0113 | 0.0071 | 0.0026 | L6H0 (+0.189) |
| rare | cars | near | 0.0192 | 0.0088 | 0.0162 | 0.0065 | 0.0019 | L7H0 (+0.216) |
| rare | chess | near | 0.0211 | 0.0052 | 0.0230 | 0.0084 | 0.0018 | L7H0 (+0.279) |
| rare | coffee | near | 0.0202 | 0.0100 | 0.0148 | 0.0072 | 0.0017 | L5H6 (+0.289) |
| rare | money | near | 0.0143 | 0.0070 | 0.0142 | 0.0093 | 0.0026 | L7H0 (+0.246) |
| rare | music | near | 0.0204 | 0.0073 | 0.0178 | 0.0062 | 0.0022 | L7H0 (+0.276) |
| rare | pizza | near | 0.0260 | 0.0068 | 0.0242 | 0.0066 | 0.0020 | L7H0 (+0.291) |
| rare | rain | near | 0.0178 | 0.0140 | 0.0215 | 0.0070 | 0.0020 | L5H6 (+0.251) |
| rare | rice | near | 0.0161 | 0.0095 | 0.0117 | 0.0068 | 0.0018 | L7H0 (+0.248) |
| rare | songs | near | 0.0267 | 0.0106 | 0.0241 | 0.0070 | 0.0015 | L4H3 (+0.214) |
| rare | sushi | near | 0.0241 | 0.0068 | 0.0204 | 0.0096 | 0.0020 | L7H0 (+0.237) |
| rare | water | near | 0.0177 | 0.0160 | 0.0146 | 0.0083 | 0.0019 | L5H6 (+0.229) |

## 2. Meaningful heads per contrast: slot_mass

Out of 144 heads. The placebo row is the noise floor.

| base | phrase | placement | total | order | identity | form | placebo |
|---|---|---|---|---|---|---|---|
| common | bikes | near | 77 | 29 | 55 | 67 | 0 |
| common | cars | near | 79 | 37 | 58 | 64 | 0 |
| common | chess | near | 84 | 32 | 75 | 61 | 0 |
| common | coffee | near | 79 | 41 | 64 | 67 | 3 |
| common | money | near | 71 | 31 | 56 | 58 | 0 |
| common | music | near | 72 | 37 | 62 | 68 | 0 |
| common | pizza | near | 82 | 29 | 55 | 71 | 1 |
| common | rain | near | 78 | 52 | 55 | 74 | 0 |
| common | rice | near | 79 | 37 | 58 | 66 | 0 |
| common | songs | near | 79 | 48 | 78 | 69 | 0 |
| common | sushi | near | 87 | 44 | 57 | 70 | 0 |
| common | water | near | 68 | 49 | 62 | 56 | 0 |
| rare | bikes | near | 64 | 30 | 41 | 38 | 1 |
| rare | cars | near | 71 | 32 | 59 | 37 | 0 |
| rare | chess | near | 69 | 17 | 64 | 60 | 0 |
| rare | coffee | near | 72 | 37 | 53 | 39 | 0 |
| rare | money | near | 80 | 25 | 55 | 60 | 0 |
| rare | music | near | 67 | 28 | 60 | 32 | 0 |
| rare | pizza | near | 75 | 19 | 61 | 39 | 0 |
| rare | rain | near | 74 | 51 | 60 | 29 | 0 |
| rare | rice | near | 67 | 34 | 48 | 40 | 1 |
| rare | songs | near | 76 | 38 | 73 | 36 | 0 |
| rare | sushi | near | 75 | 39 | 48 | 58 | 0 |
| rare | water | near | 81 | 44 | 56 | 60 | 0 |

Effect sizes (slot_mass): 

| base | phrase | placement | total median |Δ| | order median |Δ| | identity median |Δ| | form median |Δ| | placebo median |Δ| | largest total |
|---|---|---|---|---|---|---|---|---|
| common | bikes | near | 0.0222 | 0.0060 | 0.0144 | 0.0186 | 0.0018 | L5H2 (+0.205) |
| common | cars | near | 0.0252 | 0.0094 | 0.0150 | 0.0171 | 0.0019 | L4H3 (-0.186) |
| common | chess | near | 0.0259 | 0.0073 | 0.0213 | 0.0147 | 0.0014 | L4H3 (-0.204) |
| common | coffee | near | 0.0235 | 0.0094 | 0.0160 | 0.0199 | 0.0015 | L4H3 (-0.224) |
| common | money | near | 0.0197 | 0.0064 | 0.0151 | 0.0150 | 0.0016 | L5H6 (-0.186) |
| common | music | near | 0.0200 | 0.0092 | 0.0176 | 0.0192 | 0.0016 | L4H3 (-0.183) |
| common | pizza | near | 0.0239 | 0.0073 | 0.0139 | 0.0202 | 0.0018 | L4H3 (-0.198) |
| common | rain | near | 0.0229 | 0.0104 | 0.0132 | 0.0204 | 0.0014 | L5H6 (-0.268) |
| common | rice | near | 0.0234 | 0.0079 | 0.0156 | 0.0183 | 0.0016 | L10H9 (+0.157) |
| common | songs | near | 0.0246 | 0.0101 | 0.0213 | 0.0192 | 0.0016 | L5H4 (+0.222) |
| common | sushi | near | 0.0281 | 0.0103 | 0.0163 | 0.0201 | 0.0017 | L5H3 (+0.187) |
| common | water | near | 0.0186 | 0.0089 | 0.0158 | 0.0153 | 0.0014 | L5H6 (-0.286) |
| rare | bikes | near | 0.0169 | 0.0050 | 0.0111 | 0.0087 | 0.0018 | L6H0 (-0.186) |
| rare | cars | near | 0.0195 | 0.0072 | 0.0141 | 0.0082 | 0.0015 | L4H3 (-0.202) |
| rare | chess | near | 0.0186 | 0.0039 | 0.0170 | 0.0165 | 0.0016 | L7H0 (-0.237) |
| rare | coffee | near | 0.0201 | 0.0086 | 0.0140 | 0.0076 | 0.0019 | L5H6 (-0.285) |
| rare | money | near | 0.0233 | 0.0056 | 0.0158 | 0.0159 | 0.0016 | L4H3 (-0.214) |
| rare | music | near | 0.0179 | 0.0056 | 0.0159 | 0.0076 | 0.0014 | L7H0 (-0.243) |
| rare | pizza | near | 0.0218 | 0.0050 | 0.0153 | 0.0073 | 0.0015 | L7H0 (-0.241) |
| rare | rain | near | 0.0212 | 0.0122 | 0.0142 | 0.0080 | 0.0016 | L5H6 (-0.231) |
| rare | rice | near | 0.0186 | 0.0061 | 0.0122 | 0.0089 | 0.0017 | L7H0 (-0.206) |
| rare | songs | near | 0.0224 | 0.0086 | 0.0202 | 0.0088 | 0.0018 | L4H3 (-0.220) |
| rare | sushi | near | 0.0214 | 0.0084 | 0.0132 | 0.0162 | 0.0015 | L7H0 (-0.202) |
| rare | water | near | 0.0234 | 0.0085 | 0.0148 | 0.0157 | 0.0014 | L5H6 (-0.203) |

## 3. Tracked heads (exp1ext rule-selected + controls): Δ sink_mass

**rare bases, `sushi`, near**

| head | group | total | order | identity | form |
|---|---|---|---|---|---|
| L1H5 | sink rises | -0.004 | -0.003 | -0.003 | +0.002 |
| L1H7 | sink rises | +0.004 | -0.001 | +0.012 | -0.007 |
| L7H7 | sink rises | -0.013 | +0.001 | **-0.063** | **+0.049** |
| L9H11 | sink rises | **+0.047** | **+0.031** | **+0.029** | -0.014 |
| L10H11 | sink rises | **+0.034** | -0.002 | **+0.026** | +0.009 |
| L11H1 | sink rises | **+0.041** | -0.001 | **+0.052** | -0.009 |
| L11H3 | sink rises | **+0.030** | +0.015 | **+0.024** | -0.009 |
| L2H1 | sink rises | **+0.069** | +0.002 | **+0.084** | -0.017 |
| L2H6 | sink rises | **+0.021** | -0.001 | **+0.025** | -0.003 |
| L3H6 | sink rises | -0.005 | -0.003 | **-0.031** | **+0.029** |
| L7H11 | sink rises | **+0.054** | **+0.033** | **+0.021** | +0.001 |
| L10H4 | sink rises | **+0.044** | +0.009 | **+0.034** | +0.001 |
| L11H9 | sink rises | **+0.049** | -0.003 | **+0.045** | +0.006 |
| L1H8 | sink falls | +0.000 | +0.001 | -0.004 | +0.003 |
| L2H0 | sink falls | **-0.044** | +0.010 | -0.011 | **-0.043** |
| L2H4 | sink falls | **-0.073** | **+0.030** | **-0.050** | **-0.054** |
| L3H3 | sink falls | +0.012 | **-0.020** | -0.020 | **+0.053** |
| L6H4 | sink falls | **+0.093** | +0.003 | **+0.088** | +0.002 |
| L6H8 | sink falls | **+0.112** | **+0.079** | **+0.042** | -0.009 |
| L7H0 | sink falls | **+0.237** | **+0.056** | **+0.180** | +0.002 |
| L10H9 | sink falls | -0.007 | +0.019 | **+0.031** | **-0.056** |
| L4H0 | sink falls | **-0.166** | **-0.097** | +0.015 | **-0.084** |
| L5H2 | sink falls | **-0.059** | **+0.096** | -0.015 | **-0.141** |
| L5H6 | sink falls | **+0.142** | **+0.052** | **+0.070** | +0.021 |
| L8H2 | sink falls | **+0.067** | **+0.025** | **+0.051** | -0.009 |
| L9H8 | sink falls | **+0.104** | **+0.023** | **+0.094** | -0.013 |
| L0H6 | label mix only | +0.003 | -0.000 | +0.001 | +0.003 |
| L1H0 | label mix only | +0.002 | +0.001 | +0.001 | +0.000 |
| L3H2 | label mix only | -0.001 | -0.008 | -0.014 | **+0.021** |
| L10H0 | label mix only | **+0.057** | -0.012 | **+0.058** | +0.011 |
| L0H1 | control | -0.000 | -0.000 | -0.000 | +0.000 |
| L4H11 | control | +0.000 | +0.000 | -0.000 | +0.000 |
| L5H1 | control | +0.007 | -0.000 | +0.001 | +0.007 |

**common bases, `sushi`, near**

| head | group | total | order | identity | form |
|---|---|---|---|---|---|
| L1H5 | sink rises | -0.004 | -0.004 | -0.005 | +0.005 |
| L1H7 | sink rises | +0.009 | -0.000 | +0.012 | -0.003 |
| L7H7 | sink rises | **+0.055** | +0.004 | **+0.071** | -0.020 |
| L9H11 | sink rises | **+0.031** | **+0.075** | +0.001 | **-0.046** |
| L10H11 | sink rises | +0.014 | **+0.023** | +0.015 | **-0.024** |
| L11H1 | sink rises | **+0.122** | **+0.068** | **+0.083** | **-0.029** |
| L11H3 | sink rises | **+0.103** | **+0.095** | **+0.026** | -0.017 |
| L2H1 | sink rises | **+0.106** | +0.004 | **+0.095** | +0.006 |
| L2H6 | sink rises | +0.010 | -0.005 | +0.016 | -0.001 |
| L3H6 | sink rises | **-0.021** | -0.008 | **-0.025** | +0.011 |
| L7H11 | sink rises | **+0.127** | **+0.093** | **+0.077** | **-0.043** |
| L10H4 | sink rises | **+0.114** | **+0.066** | **+0.055** | -0.007 |
| L11H9 | sink rises | **+0.127** | **+0.081** | **+0.045** | +0.001 |
| L1H8 | sink falls | +0.003 | -0.000 | +0.002 | +0.002 |
| L2H0 | sink falls | -0.004 | +0.005 | -0.007 | -0.001 |
| L2H4 | sink falls | -0.004 | +0.008 | **-0.040** | **+0.028** |
| L3H3 | sink falls | **+0.048** | -0.020 | **+0.026** | **+0.041** |
| L6H4 | sink falls | **+0.126** | **-0.024** | **+0.136** | +0.014 |
| L6H8 | sink falls | **+0.210** | **+0.164** | +0.017 | **+0.028** |
| L7H0 | sink falls | **+0.244** | **+0.141** | **+0.025** | **+0.078** |
| L10H9 | sink falls | **+0.085** | **+0.142** | **-0.040** | -0.016 |
| L4H0 | sink falls | **-0.109** | **-0.062** | **-0.050** | +0.003 |
| L5H2 | sink falls | **+0.095** | **+0.151** | **-0.076** | +0.020 |
| L5H6 | sink falls | **+0.046** | **+0.047** | -0.027 | +0.026 |
| L8H2 | sink falls | **+0.128** | **+0.061** | **+0.098** | **-0.032** |
| L9H8 | sink falls | **+0.120** | **+0.067** | +0.000 | **+0.053** |
| L0H6 | label mix only | -0.005 | -0.000 | -0.001 | -0.004 |
| L1H0 | label mix only | -0.008 | +0.002 | -0.006 | -0.004 |
| L3H2 | label mix only | +0.006 | -0.013 | +0.001 | +0.017 |
| L10H0 | label mix only | **-0.056** | -0.010 | **-0.052** | +0.006 |
| L0H1 | control | -0.000 | -0.000 | -0.000 | -0.000 |
| L4H11 | control | +0.000 | +0.000 | -0.000 | +0.000 |
| L5H1 | control | +0.003 | -0.000 | +0.004 | -0.000 |

## 4. Does the final token attend to the phrase? Δ slot_mass, tracked heads

**rare bases, `sushi`, near**

| head | group | total | order | identity | form |
|---|---|---|---|---|---|
| L1H5 | sink rises | **+0.027** | -0.005 | **+0.038** | -0.006 |
| L1H7 | sink rises | **+0.022** | -0.005 | **+0.044** | -0.017 |
| L7H7 | sink rises | **-0.061** | -0.009 | -0.004 | **-0.047** |
| L9H11 | sink rises | **-0.029** | -0.010 | -0.011 | -0.009 |
| L10H11 | sink rises | **-0.074** | -0.007 | **-0.026** | **-0.041** |
| L11H1 | sink rises | **-0.043** | -0.004 | -0.011 | **-0.028** |
| L11H3 | sink rises | -0.009 | +0.020 | -0.008 | **-0.022** |
| L2H1 | sink rises | **-0.041** | -0.013 | -0.002 | **-0.025** |
| L2H6 | sink rises | +0.014 | +0.008 | **-0.046** | **+0.053** |
| L3H6 | sink rises | +0.010 | +0.019 | **+0.029** | **-0.038** |
| L7H11 | sink rises | **-0.057** | +0.001 | **-0.023** | **-0.035** |
| L10H4 | sink rises | -0.012 | **+0.023** | -0.002 | **-0.032** |
| L11H9 | sink rises | **-0.050** | -0.004 | -0.012 | **-0.034** |
| L1H8 | sink falls | +0.015 | +0.000 | +0.011 | +0.004 |
| L2H0 | sink falls | **+0.103** | **-0.044** | **+0.051** | **+0.097** |
| L2H4 | sink falls | **+0.081** | **-0.039** | **+0.056** | **+0.065** |
| L3H3 | sink falls | -0.010 | **+0.021** | +0.005 | **-0.037** |
| L6H4 | sink falls | **-0.023** | +0.003 | -0.001 | **-0.025** |
| L6H8 | sink falls | **-0.116** | **-0.069** | **-0.054** | +0.006 |
| L7H0 | sink falls | **-0.202** | **-0.033** | **-0.162** | -0.007 |
| L10H9 | sink falls | +0.011 | +0.001 | **-0.021** | **+0.031** |
| L4H0 | sink falls | **+0.160** | **+0.117** | -0.019 | **+0.062** |
| L5H2 | sink falls | **+0.034** | **-0.090** | -0.008 | **+0.131** |
| L5H6 | sink falls | **-0.146** | **-0.042** | **-0.084** | -0.020 |
| L8H2 | sink falls | -0.012 | **-0.023** | +0.017 | -0.006 |
| L9H8 | sink falls | **-0.031** | +0.007 | -0.011 | **-0.027** |
| L0H6 | label mix only | **-0.044** | +0.000 | -0.008 | **-0.036** |
| L1H0 | label mix only | **+0.031** | **-0.029** | -0.015 | **+0.074** |
| L3H2 | label mix only | -0.004 | **+0.029** | **-0.046** | +0.013 |
| L10H0 | label mix only | **-0.027** | +0.003 | +0.018 | **-0.047** |
| L0H1 | control | -0.000 | +0.000 | +0.001 | -0.001 |
| L4H11 | control | -0.006 | -0.003 | -0.001 | -0.001 |
| L5H1 | control | -0.003 | +0.000 | -0.000 | -0.002 |

**common bases, `sushi`, near**

| head | group | total | order | identity | form |
|---|---|---|---|---|---|
| L1H5 | sink rises | +0.011 | -0.003 | **+0.024** | -0.010 |
| L1H7 | sink rises | **+0.061** | -0.003 | **+0.032** | **+0.033** |
| L7H7 | sink rises | +0.005 | -0.002 | -0.016 | **+0.023** |
| L9H11 | sink rises | +0.001 | -0.015 | **-0.030** | **+0.046** |
| L10H11 | sink rises | **+0.092** | **+0.026** | +0.015 | **+0.051** |
| L11H1 | sink rises | -0.006 | -0.009 | **-0.020** | **+0.023** |
| L11H3 | sink rises | **+0.045** | **-0.034** | **+0.035** | **+0.043** |
| L2H1 | sink rises | +0.015 | -0.011 | +0.012 | +0.013 |
| L2H6 | sink rises | +0.016 | +0.013 | **-0.033** | **+0.036** |
| L3H6 | sink rises | **+0.117** | +0.014 | **+0.036** | **+0.068** |
| L7H11 | sink rises | -0.019 | -0.019 | **-0.021** | **+0.021** |
| L10H4 | sink rises | +0.005 | +0.012 | **-0.028** | **+0.021** |
| L11H9 | sink rises | -0.006 | -0.017 | -0.010 | **+0.022** |
| L1H8 | sink falls | +0.009 | +0.001 | +0.006 | +0.001 |
| L2H0 | sink falls | **+0.061** | **-0.034** | **+0.034** | **+0.062** |
| L2H4 | sink falls | -0.020 | **-0.027** | +0.018 | -0.011 |
| L3H3 | sink falls | **-0.072** | **+0.024** | **-0.057** | **-0.038** |
| L6H4 | sink falls | **+0.063** | **+0.029** | **+0.049** | -0.014 |
| L6H8 | sink falls | **-0.155** | **-0.162** | -0.011 | +0.018 |
| L7H0 | sink falls | **-0.096** | **-0.102** | **+0.024** | -0.017 |
| L10H9 | sink falls | **+0.052** | **-0.058** | **+0.051** | **+0.059** |
| L4H0 | sink falls | **+0.158** | **+0.106** | **+0.038** | +0.013 |
| L5H2 | sink falls | +0.001 | **-0.147** | **+0.057** | **+0.091** |
| L5H6 | sink falls | **-0.045** | **-0.042** | +0.018 | -0.021 |
| L8H2 | sink falls | **-0.028** | -0.020 | +0.012 | **-0.021** |
| L9H8 | sink falls | **+0.047** | -0.006 | **+0.084** | **-0.031** |
| L0H6 | label mix only | **+0.051** | +0.000 | +0.002 | **+0.050** |
| L1H0 | label mix only | **-0.054** | **-0.030** | **-0.095** | **+0.072** |
| L3H2 | label mix only | **+0.041** | **+0.025** | -0.008 | +0.024 |
| L10H0 | label mix only | **+0.174** | **+0.035** | **+0.128** | +0.011 |
| L0H1 | control | +0.001 | +0.000 | +0.001 | +0.001 |
| L4H11 | control | -0.000 | -0.001 | +0.001 | -0.001 |
| L5H1 | control | -0.001 | +0.000 | -0.000 | -0.000 |

## 5. Label-mix shift (TV distance)

Split-half TV floor within the `none` rows: rare 0.124, common 0.096.

Heads whose label-mix TV exceeds the floor, per contrast:

| base | phrase | placement | total | order | identity | form | placebo |
|---|---|---|---|---|---|---|---|
| common | bikes | near | 34 | 21 | 29 | 19 | 0 |
| common | cars | near | 32 | 18 | 25 | 22 | 0 |
| common | chess | near | 36 | 15 | 21 | 18 | 0 |
| common | coffee | near | 30 | 23 | 25 | 22 | 0 |
| common | money | near | 32 | 7 | 22 | 18 | 0 |
| common | music | near | 34 | 19 | 23 | 21 | 0 |
| common | pizza | near | 36 | 16 | 28 | 17 | 0 |
| common | rain | near | 34 | 23 | 28 | 17 | 0 |
| common | rice | near | 36 | 20 | 23 | 19 | 0 |
| common | songs | near | 33 | 14 | 25 | 21 | 0 |
| common | sushi | near | 40 | 17 | 22 | 16 | 0 |
| common | water | near | 32 | 24 | 26 | 17 | 0 |
| rare | bikes | near | 23 | 11 | 16 | 5 | 0 |
| rare | cars | near | 23 | 9 | 18 | 3 | 0 |
| rare | chess | near | 25 | 8 | 19 | 8 | 0 |
| rare | coffee | near | 20 | 11 | 18 | 3 | 0 |
| rare | money | near | 20 | 7 | 15 | 7 | 0 |
| rare | music | near | 27 | 10 | 16 | 4 | 0 |
| rare | pizza | near | 29 | 8 | 20 | 3 | 0 |
| rare | rain | near | 17 | 16 | 24 | 3 | 0 |
| rare | rice | near | 21 | 11 | 13 | 3 | 0 |
| rare | songs | near | 27 | 10 | 18 | 4 | 0 |
| rare | sushi | near | 21 | 7 | 18 | 6 | 0 |
| rare | water | near | 16 | 10 | 16 | 8 | 0 |

## 6. Pooled over phrases (sink_mass, total)

| base | placement | name | n_phrases | mean_of_means | min_mean | max_mean | n_meaningful_pos | n_meaningful_neg |
|---|---|---|---|---|---|---|---|---|
| common | near | L4H3 | 12 | 0.273 | 0.136 | 0.333 | 12 | 0 |
| rare | near | L7H0 | 12 | 0.236 | 0.179 | 0.291 | 12 | 0 |
| common | near | L7H0 | 12 | 0.235 | 0.096 | 0.305 | 12 | 0 |
| common | near | L6H0 | 12 | 0.217 | 0.131 | 0.288 | 12 | 0 |
| rare | near | L5H6 | 12 | 0.196 | 0.124 | 0.289 | 12 | 0 |
| common | near | L6H8 | 12 | 0.189 | 0.041 | 0.271 | 12 | 0 |
| rare | near | L6H0 | 12 | 0.186 | 0.094 | 0.247 | 12 | 0 |
| rare | near | L4H3 | 12 | 0.166 | 0.048 | 0.216 | 12 | 0 |
| rare | near | L6H3 | 12 | -0.143 | -0.179 | -0.119 | 0 | 12 |
| common | near | L5H6 | 12 | 0.141 | 0.018 | 0.311 | 11 | 0 |
| common | near | L9H8 | 12 | 0.134 | 0.084 | 0.174 | 12 | 0 |
| common | near | L8H6 | 12 | 0.134 | 0.066 | 0.199 | 12 | 0 |
| rare | near | L6H8 | 12 | 0.124 | 0.035 | 0.192 | 12 | 0 |
| common | near | L6H4 | 12 | 0.123 | 0.069 | 0.172 | 12 | 0 |
| common | near | L5H10 | 12 | 0.120 | 0.069 | 0.177 | 12 | 0 |

## 7. Additivity: does the combination add more than its single tokens?

Per row, interaction = (combo − none) − Σ(part − none); e.g. `sushi:nl` = sushi:I@29 + sushi:like@30 + sushi:sushi@31; `sushi:shuffled` = sushi:like@29 + sushi:I@30 + sushi:sushi@31; `sushi:PV` = sushi:I@29 + sushi:like@30; `sushi:VO` = sushi:like@30 + sushi:sushi@31; `sushi:PO` = sushi:I@29 + sushi:sushi@31. Depth values average the group's heads within each row; CI = bootstrap over rows. Metrics: sink_logit, sink_mass (sink_logit = log(p / (1 − p)), not squeezed by the probability bounds).

| metric | base | phrase | term | depth | mean | lo | hi | CI excludes 0 |
|---|---|---|---|---|---|---|---|---|
| sink_mass | common | sushi | nl | early | +0.0003 | -0.0008 | +0.0015 | False |
| sink_mass | common | sushi | nl | late | +0.0282 | +0.0237 | +0.0329 | True |
| sink_mass | common | sushi | shuffled | early | +0.0019 | +0.0008 | +0.0031 | True |
| sink_mass | common | sushi | shuffled | late | -0.0156 | -0.0202 | -0.0110 | True |
| sink_mass | common | sushi | PV | early | -0.0006 | -0.0013 | +0.0000 | False |
| sink_mass | common | sushi | PV | late | +0.0009 | -0.0016 | +0.0036 | False |
| sink_mass | common | sushi | VO | early | +0.0012 | +0.0003 | +0.0021 | True |
| sink_mass | common | sushi | VO | late | +0.0118 | +0.0086 | +0.0154 | True |
| sink_mass | common | sushi | PO | early | -0.0008 | -0.0015 | -0.0002 | True |
| sink_mass | common | sushi | PO | late | +0.0009 | -0.0016 | +0.0032 | False |
| sink_mass | rare | sushi | nl | early | +0.0077 | +0.0069 | +0.0086 | True |
| sink_mass | rare | sushi | nl | late | +0.0115 | +0.0083 | +0.0147 | True |
| sink_mass | rare | sushi | shuffled | early | +0.0067 | +0.0058 | +0.0076 | True |
| sink_mass | rare | sushi | shuffled | late | -0.0002 | -0.0036 | +0.0027 | False |
| sink_mass | rare | sushi | PV | early | +0.0028 | +0.0023 | +0.0033 | True |
| sink_mass | rare | sushi | PV | late | +0.0026 | +0.0008 | +0.0044 | True |
| sink_mass | rare | sushi | VO | early | +0.0057 | +0.0051 | +0.0063 | True |
| sink_mass | rare | sushi | VO | late | +0.0027 | +0.0002 | +0.0052 | True |
| sink_mass | rare | sushi | PO | early | +0.0006 | +0.0002 | +0.0010 | True |
| sink_mass | rare | sushi | PO | late | +0.0003 | -0.0013 | +0.0020 | False |
| sink_mass | common | sushi | nl−shuffled | early | -0.0016 | -0.0024 | -0.0007 | True |
| sink_mass | common | sushi | nl−shuffled | late | +0.0438 | +0.0396 | +0.0481 | True |
| sink_mass | rare | sushi | nl−shuffled | early | +0.0011 | +0.0003 | +0.0019 | True |
| sink_mass | rare | sushi | nl−shuffled | late | +0.0117 | +0.0091 | +0.0144 | True |
| sink_mass | common | sushi | triple | early | +0.0006 | +0.0000 | +0.0012 | True |
| sink_mass | common | sushi | triple | late | +0.0146 | +0.0118 | +0.0177 | True |
| sink_mass | rare | sushi | triple | early | -0.0014 | -0.0017 | -0.0010 | True |
| sink_mass | rare | sushi | triple | late | +0.0060 | +0.0044 | +0.0077 | True |
| sink_logit | common | sushi | nl | early | +0.0121 | +0.0051 | +0.0190 | True |
| sink_logit | common | sushi | nl | late | +0.1532 | +0.1290 | +0.1795 | True |
| sink_logit | common | sushi | shuffled | early | +0.0265 | +0.0187 | +0.0336 | True |
| sink_logit | common | sushi | shuffled | late | -0.0988 | -0.1254 | -0.0730 | True |
| sink_logit | common | sushi | PV | early | +0.0005 | -0.0035 | +0.0047 | False |
| sink_logit | common | sushi | PV | late | +0.0053 | -0.0093 | +0.0196 | False |
| sink_logit | common | sushi | VO | early | +0.0094 | +0.0038 | +0.0147 | True |
| sink_logit | common | sushi | VO | late | +0.0618 | +0.0442 | +0.0821 | True |
| sink_logit | common | sushi | PO | early | +0.0025 | -0.0012 | +0.0066 | False |
| sink_logit | common | sushi | PO | late | +0.0048 | -0.0095 | +0.0181 | False |
| sink_logit | rare | sushi | nl | early | +0.0521 | +0.0466 | +0.0581 | True |
| sink_logit | rare | sushi | nl | late | +0.0573 | +0.0384 | +0.0756 | True |
| sink_logit | rare | sushi | shuffled | early | +0.0459 | +0.0402 | +0.0520 | True |
| sink_logit | rare | sushi | shuffled | late | -0.0020 | -0.0217 | +0.0154 | False |
| sink_logit | rare | sushi | PV | early | +0.0159 | +0.0128 | +0.0192 | True |
| sink_logit | rare | sushi | PV | late | +0.0132 | +0.0026 | +0.0239 | True |
| sink_logit | rare | sushi | VO | early | +0.0388 | +0.0347 | +0.0427 | True |
| sink_logit | rare | sushi | VO | late | +0.0140 | -0.0010 | +0.0283 | False |
| sink_logit | rare | sushi | PO | early | +0.0056 | +0.0031 | +0.0083 | True |
| sink_logit | rare | sushi | PO | late | -0.0022 | -0.0111 | +0.0074 | False |
| sink_logit | common | sushi | nl−shuffled | early | -0.0144 | -0.0195 | -0.0088 | True |
| sink_logit | common | sushi | nl−shuffled | late | +0.2520 | +0.2289 | +0.2778 | True |
| sink_logit | rare | sushi | nl−shuffled | early | +0.0062 | +0.0017 | +0.0108 | True |
| sink_logit | rare | sushi | nl−shuffled | late | +0.0593 | +0.0436 | +0.0750 | True |
| sink_logit | common | sushi | triple | early | -0.0002 | -0.0037 | +0.0034 | False |
| sink_logit | common | sushi | triple | late | +0.0814 | +0.0663 | +0.0984 | True |
| sink_logit | rare | sushi | triple | early | -0.0082 | -0.0104 | -0.0062 | True |
| sink_logit | rare | sushi | triple | late | +0.0323 | +0.0225 | +0.0424 | True |

Every phrase's late-layer values are in section 9.

Order contrast against its additive prediction, across the 144 heads (every phrase):

| metric | base | phrase | pair | slope | r | r2 | mean_abs_observed | mean_abs_interaction |
|---|---|---|---|---|---|---|---|---|
| sink_mass | common | bikes | nl−shuffled | 1.292 | 0.507 | 0.257 | 0.018 | 0.015 |
| sink_mass | common | cars | nl−shuffled | 0.558 | 0.250 | 0.062 | 0.018 | 0.016 |
| sink_mass | common | chess | nl−shuffled | 0.770 | 0.396 | 0.157 | 0.016 | 0.015 |
| sink_mass | common | coffee | nl−shuffled | 1.206 | 0.438 | 0.192 | 0.025 | 0.023 |
| sink_mass | common | money | nl−shuffled | 0.329 | 0.259 | 0.067 | 0.011 | 0.013 |
| sink_mass | common | music | nl−shuffled | 0.970 | 0.384 | 0.148 | 0.020 | 0.018 |
| sink_mass | common | pizza | nl−shuffled | 1.429 | 0.502 | 0.252 | 0.022 | 0.018 |
| sink_mass | common | rain | nl−shuffled | 0.666 | 0.288 | 0.083 | 0.024 | 0.026 |
| sink_mass | common | rice | nl−shuffled | 0.293 | 0.130 | 0.017 | 0.020 | 0.019 |
| sink_mass | common | songs | nl−shuffled | 1.405 | 0.519 | 0.270 | 0.026 | 0.021 |
| sink_mass | common | sushi | nl−shuffled | -0.319 | -0.080 | 0.006 | 0.028 | 0.030 |
| sink_mass | common | water | nl−shuffled | 0.501 | 0.122 | 0.015 | 0.029 | 0.029 |
| sink_mass | rare | bikes | nl−shuffled | 1.516 | 0.662 | 0.439 | 0.017 | 0.011 |
| sink_mass | rare | cars | nl−shuffled | 0.759 | 0.480 | 0.230 | 0.013 | 0.010 |
| sink_mass | rare | chess | nl−shuffled | 0.284 | 0.195 | 0.038 | 0.010 | 0.011 |
| sink_mass | rare | coffee | nl−shuffled | 0.805 | 0.432 | 0.186 | 0.017 | 0.013 |
| sink_mass | rare | money | nl−shuffled | 0.251 | 0.230 | 0.053 | 0.009 | 0.009 |
| sink_mass | rare | music | nl−shuffled | 0.752 | 0.500 | 0.250 | 0.012 | 0.011 |
| sink_mass | rare | pizza | nl−shuffled | 0.911 | 0.669 | 0.447 | 0.011 | 0.008 |
| sink_mass | rare | rain | nl−shuffled | 1.977 | 0.743 | 0.552 | 0.025 | 0.019 |
| sink_mass | rare | rice | nl−shuffled | 0.547 | 0.394 | 0.156 | 0.014 | 0.011 |
| sink_mass | rare | songs | nl−shuffled | 0.320 | 0.161 | 0.026 | 0.018 | 0.015 |
| sink_mass | rare | sushi | nl−shuffled | 0.164 | 0.124 | 0.015 | 0.013 | 0.014 |
| sink_mass | rare | water | nl−shuffled | 1.912 | 0.637 | 0.406 | 0.022 | 0.018 |
| sink_logit | common | bikes | nl−shuffled | 1.243 | 0.584 | 0.341 | 0.107 | 0.080 |
| sink_logit | common | cars | nl−shuffled | 0.513 | 0.231 | 0.053 | 0.111 | 0.100 |
| sink_logit | common | chess | nl−shuffled | 0.758 | 0.376 | 0.141 | 0.098 | 0.090 |
| sink_logit | common | coffee | nl−shuffled | 1.126 | 0.432 | 0.186 | 0.154 | 0.135 |
| sink_logit | common | money | nl−shuffled | 0.327 | 0.253 | 0.064 | 0.069 | 0.074 |
| sink_logit | common | music | nl−shuffled | 1.088 | 0.424 | 0.179 | 0.121 | 0.109 |
| sink_logit | common | pizza | nl−shuffled | 1.273 | 0.503 | 0.253 | 0.130 | 0.106 |
| sink_logit | common | rain | nl−shuffled | 0.772 | 0.311 | 0.097 | 0.147 | 0.157 |
| sink_logit | common | rice | nl−shuffled | 0.366 | 0.164 | 0.027 | 0.118 | 0.115 |
| sink_logit | common | songs | nl−shuffled | 1.286 | 0.514 | 0.264 | 0.150 | 0.118 |
| sink_logit | common | sushi | nl−shuffled | -0.233 | -0.066 | 0.004 | 0.161 | 0.172 |
| sink_logit | common | water | nl−shuffled | 0.391 | 0.102 | 0.010 | 0.170 | 0.168 |
| sink_logit | rare | bikes | nl−shuffled | 1.528 | 0.682 | 0.465 | 0.104 | 0.069 |
| sink_logit | rare | cars | nl−shuffled | 0.618 | 0.376 | 0.141 | 0.082 | 0.070 |
| sink_logit | rare | chess | nl−shuffled | 0.171 | 0.111 | 0.012 | 0.069 | 0.075 |
| sink_logit | rare | coffee | nl−shuffled | 0.711 | 0.366 | 0.134 | 0.111 | 0.087 |
| sink_logit | rare | money | nl−shuffled | 0.276 | 0.233 | 0.054 | 0.057 | 0.057 |
| sink_logit | rare | music | nl−shuffled | 0.600 | 0.408 | 0.167 | 0.078 | 0.070 |
| sink_logit | rare | pizza | nl−shuffled | 0.911 | 0.621 | 0.386 | 0.075 | 0.054 |
| sink_logit | rare | rain | nl−shuffled | 1.945 | 0.730 | 0.533 | 0.159 | 0.119 |
| sink_logit | rare | rice | nl−shuffled | 0.523 | 0.416 | 0.173 | 0.085 | 0.071 |
| sink_logit | rare | songs | nl−shuffled | 0.141 | 0.070 | 0.005 | 0.109 | 0.095 |
| sink_logit | rare | sushi | nl−shuffled | 0.287 | 0.212 | 0.045 | 0.080 | 0.086 |
| sink_logit | rare | water | nl−shuffled | 1.970 | 0.618 | 0.381 | 0.137 | 0.112 |

Heads with a BH-significant interaction on sink_mass (q over every interaction test of the metric):

| base | phrase | combo | significant | late + | late − | largest |interaction| |
|---|---|---|---|---|---|---|
| common | sushi | nl | 111 | 48 | 9 | L7H0 (+0.111), L10H8 (+0.094), L6H8 (+0.094) |
| common | sushi | shuffled | 104 | 8 | 40 | L5H2 (-0.122), L10H9 (-0.074), L6H8 (-0.071) |
| common | water | nl | 118 | 63 | 2 | L5H6 (+0.197), L5H2 (+0.166), L7H0 (+0.161) |
| common | water | shuffled | 100 | 25 | 21 | L5H6 (+0.116), L4H0 (-0.096), L3H2 (-0.071) |
| common | pizza | nl | 119 | 60 | 3 | L6H8 (+0.132), L7H0 (+0.108), L6H0 (+0.100) |
| common | pizza | shuffled | 91 | 29 | 12 | L7H0 (+0.100), L4H3 (+0.098), L6H0 (+0.075) |
| common | music | nl | 112 | 46 | 10 | L4H3 (+0.124), L6H0 (+0.123), L7H0 (+0.102) |
| common | music | shuffled | 92 | 22 | 19 | L7H0 (+0.070), L5H6 (-0.059), L3H2 (-0.049) |
| common | coffee | nl | 113 | 49 | 4 | L8H7 (+0.147), L7H0 (+0.140), L4H3 (+0.123) |
| common | coffee | shuffled | 112 | 16 | 37 | L7H0 (+0.102), L8H7 (+0.089), L5H6 (-0.080) |
| common | chess | nl | 108 | 43 | 10 | L6H0 (+0.121), L6H8 (+0.113), L8H7 (+0.108) |
| common | chess | shuffled | 104 | 21 | 30 | L6H0 (+0.088), L7H0 (+0.086), L4H3 (+0.084) |
| common | rain | nl | 117 | 53 | 9 | L7H0 (+0.148), L5H6 (+0.138), L8H6 (+0.122) |
| common | rain | shuffled | 102 | 33 | 14 | L4H0 (-0.095), L3H8 (-0.080), L3H9 (-0.076) |
| common | cars | nl | 110 | 52 | 10 | L6H0 (+0.176), L7H11 (+0.138), L7H0 (+0.099) |
| common | cars | shuffled | 99 | 37 | 10 | L6H0 (+0.147), L6H11 (+0.062), L7H8 (+0.062) |
| common | songs | nl | 105 | 32 | 16 | L6H8 (+0.106), L4H5 (+0.096), L8H4 (+0.077) |
| common | songs | shuffled | 103 | 7 | 48 | L5H4 (-0.080), L3H7 (-0.070), L3H2 (-0.068) |
| common | rice | nl | 110 | 57 | 1 | L6H8 (+0.136), L5H6 (-0.121), L10H2 (+0.109) |
| common | rice | shuffled | 94 | 31 | 10 | L5H6 (-0.092), L5H2 (-0.075), L4H5 (-0.070) |
| common | bikes | nl | 89 | 39 | 5 | L6H0 (+0.089), L4H3 (+0.071), L7H0 (+0.064) |
| common | bikes | shuffled | 92 | 13 | 32 | L4H3 (+0.091), L4H5 (+0.075), L3H8 (-0.069) |
| common | money | nl | 115 | 53 | 7 | L5H6 (+0.205), L6H0 (+0.140), L5H2 (+0.134) |
| common | money | shuffled | 107 | 40 | 8 | L5H6 (+0.168), L7H0 (+0.127), L6H0 (+0.102) |
| common | sushi | PV | 114 | 26 | 22 | L5H0 (-0.064), L10H9 (+0.058), L4H0 (-0.055) |
| common | sushi | VO | 103 | 43 | 5 | L5H6 (+0.079), L6H0 (+0.058), L8H7 (+0.043) |
| common | sushi | PO | 89 | 23 | 18 | L7H0 (+0.043), L8H1 (-0.032), L9H8 (-0.027) |
| common | water | PV | 117 | 52 | 9 | L5H2 (+0.097), L10H9 (+0.066), L6H8 (+0.061) |
| common | water | VO | 115 | 60 | 3 | L5H6 (+0.156), L7H0 (+0.126), L8H7 (+0.098) |
| common | water | PO | 81 | 24 | 12 | L3H2 (-0.038), L5H6 (-0.026), L6H8 (+0.026) |
| common | pizza | PV | 110 | 45 | 10 | L5H2 (+0.055), L8H6 (+0.054), L11H5 (+0.053) |
| common | pizza | VO | 117 | 54 | 5 | L7H0 (+0.094), L6H0 (+0.087), L6H8 (+0.082) |
| common | pizza | PO | 81 | 30 | 7 | L3H3 (-0.028), L3H2 (-0.024), L7H0 (+0.024) |
| common | music | PV | 106 | 39 | 15 | L5H2 (+0.098), L4H3 (+0.062), L6H0 (+0.058) |
| common | music | VO | 96 | 35 | 8 | L5H2 (-0.050), L6H8 (-0.043), L7H11 (+0.042) |
| common | music | PO | 92 | 25 | 19 | L3H2 (-0.030), L7H0 (+0.025), L3H3 (-0.022) |
| common | coffee | PV | 111 | 51 | 9 | L5H2 (+0.060), L6H8 (+0.056), L8H6 (+0.046) |
| common | coffee | VO | 109 | 34 | 16 | L7H0 (+0.095), L8H7 (+0.095), L6H0 (+0.070) |
| common | coffee | PO | 82 | 20 | 17 | L7H0 (+0.029), L3H2 (-0.024), L9H8 (-0.022) |
| common | chess | PV | 115 | 41 | 14 | L11H5 (+0.042), L7H11 (+0.037), L8H6 (+0.037) |
| common | chess | VO | 108 | 40 | 12 | L6H0 (+0.077), L8H7 (+0.076), L7H0 (+0.066) |
| common | chess | PO | 78 | 18 | 15 | L8H5 (+0.020), L9H8 (-0.020), L5H4 (-0.019) |
| common | rain | PV | 119 | 44 | 13 | L5H2 (+0.077), L6H8 (+0.074), L4H3 (+0.072) |
| common | rain | VO | 97 | 36 | 5 | L10H9 (+0.045), L8H7 (+0.032), L9H9 (+0.031) |
| common | rain | PO | 90 | 17 | 17 | L4H3 (+0.045), L5H0 (-0.042), L3H8 (-0.036) |
| common | cars | PV | 110 | 44 | 14 | L5H2 (+0.075), L7H11 (+0.073), L8H6 (+0.070) |
| common | cars | VO | 109 | 46 | 6 | L6H0 (+0.116), L7H0 (+0.070), L8H7 (+0.062) |
| common | cars | PO | 88 | 26 | 13 | L3H2 (-0.040), L6H0 (+0.033), L3H7 (-0.027) |
| common | songs | PV | 104 | 19 | 29 | L8H11 (-0.039), L8H8 (-0.038), L7H7 (+0.038) |
| common | songs | VO | 100 | 33 | 13 | L6H8 (+0.083), L5H4 (-0.077), L4H5 (+0.077) |
| common | songs | PO | 81 | 15 | 16 | L3H2 (-0.035), L4H1 (-0.029), L9H8 (-0.021) |
| common | rice | PV | 110 | 55 | 5 | L8H6 (+0.070), L5H2 (+0.063), L7H11 (+0.056) |
| common | rice | VO | 101 | 46 | 8 | L5H6 (-0.075), L8H7 (+0.039), L6H0 (+0.037) |
| common | rice | PO | 81 | 26 | 11 | L9H8 (-0.037), L5H6 (-0.030), L7H0 (+0.029) |
| common | bikes | PV | 98 | 39 | 11 | L5H2 (+0.047), L4H3 (+0.034), L6H0 (+0.032) |
| common | bikes | VO | 115 | 47 | 11 | L5H6 (+0.104), L6H0 (+0.052), L7H0 (+0.048) |
| common | bikes | PO | 97 | 9 | 34 | L3H8 (-0.055), L3H2 (-0.051), L4H5 (+0.047) |
| common | money | PV | 118 | 42 | 14 | L5H2 (+0.088), L4H3 (+0.065), L6H0 (+0.052) |
| common | money | VO | 101 | 45 | 6 | L5H6 (+0.153), L7H0 (+0.085), L6H0 (+0.072) |
| common | money | PO | 93 | 31 | 8 | L5H6 (+0.046), L6H0 (+0.041), L6H11 (+0.034) |
| rare | sushi | nl | 105 | 38 | 14 | L5H2 (+0.138), L6H0 (+0.117), L5H6 (+0.103) |
| rare | sushi | shuffled | 101 | 24 | 24 | L5H6 (+0.107), L6H0 (+0.058), L6H3 (-0.057) |
| rare | water | nl | 103 | 43 | 8 | L5H6 (+0.126), L7H0 (+0.100), L6H8 (+0.093) |
| rare | water | shuffled | 114 | 29 | 31 | L4H0 (-0.150), L6H11 (-0.079), L5H4 (-0.070) |
| rare | pizza | nl | 111 | 54 | 5 | L5H6 (+0.122), L6H8 (+0.118), L6H0 (+0.100) |
| rare | pizza | shuffled | 106 | 52 | 6 | L6H0 (+0.111), L5H6 (+0.110), L7H0 (+0.071) |
| rare | music | nl | 109 | 48 | 4 | L5H6 (+0.111), L6H0 (+0.090), L3H3 (+0.088) |
| rare | music | shuffled | 104 | 48 | 5 | L7H0 (+0.075), L8H5 (+0.058), L4H0 (+0.056) |
| rare | coffee | nl | 122 | 58 | 5 | L5H6 (+0.206), L6H8 (+0.136), L6H0 (+0.129) |
| rare | coffee | shuffled | 111 | 48 | 9 | L7H0 (+0.121), L6H0 (+0.105), L8H7 (+0.086) |
| rare | chess | nl | 112 | 49 | 8 | L5H6 (+0.155), L6H0 (+0.122), L6H8 (+0.117) |
| rare | chess | shuffled | 111 | 50 | 7 | L5H6 (+0.119), L6H0 (+0.088), L4H3 (+0.082) |
| rare | rain | nl | 111 | 42 | 15 | L5H6 (+0.142), L7H0 (+0.096), L6H8 (+0.094) |
| rare | rain | shuffled | 112 | 39 | 18 | L4H0 (-0.180), L8H7 (-0.106), L6H11 (-0.105) |
| rare | cars | nl | 115 | 50 | 5 | L6H0 (+0.147), L5H6 (+0.099), L7H0 (+0.092) |
| rare | cars | shuffled | 108 | 44 | 11 | L6H0 (+0.134), L5H6 (+0.125), L7H0 (+0.073) |
| rare | songs | nl | 106 | 30 | 20 | L8H7 (+0.088), L4H5 (+0.082), L5H6 (+0.077) |
| rare | songs | shuffled | 111 | 26 | 29 | L7H0 (-0.089), L5H6 (+0.060), L5H4 (-0.054) |
| rare | rice | nl | 113 | 53 | 9 | L6H8 (+0.121), L6H0 (+0.098), L8H11 (+0.072) |
| rare | rice | shuffled | 97 | 36 | 13 | L6H0 (+0.075), L7H0 (+0.071), L8H7 (+0.068) |
| rare | bikes | nl | 97 | 39 | 11 | L5H6 (+0.133), L6H0 (+0.115), L7H0 (+0.074) |
| rare | bikes | shuffled | 85 | 14 | 22 | L4H3 (+0.070), L6H11 (-0.068), L4H0 (-0.066) |
| rare | money | nl | 122 | 57 | 4 | L5H6 (+0.260), L7H0 (+0.118), L6H0 (+0.117) |
| rare | money | shuffled | 124 | 51 | 11 | L5H6 (+0.293), L7H0 (+0.126), L9H3 (+0.081) |
| rare | sushi | PV | 122 | 30 | 26 | L5H2 (+0.128), L10H8 (+0.057), L6H0 (+0.056) |
| rare | sushi | VO | 105 | 29 | 22 | L5H6 (+0.147), L6H0 (+0.072), L8H7 (+0.044) |
| rare | sushi | PO | 75 | 16 | 13 | L6H8 (+0.022), L2H4 (+0.014), L8H1 (-0.013) |
| rare | water | PV | 117 | 36 | 19 | L5H2 (+0.056), L5H0 (+0.048), L6H8 (+0.044) |
| rare | water | VO | 100 | 41 | 8 | L5H6 (+0.141), L7H0 (+0.105), L9H3 (+0.061) |
| rare | water | PO | 78 | 16 | 14 | L5H6 (-0.058), L4H0 (-0.031), L5H4 (-0.029) |
| rare | pizza | PV | 125 | 48 | 15 | L5H2 (+0.049), L6H8 (+0.045), L11H5 (+0.029) |
| rare | pizza | VO | 119 | 59 | 4 | L5H6 (+0.106), L6H0 (+0.076), L6H8 (+0.065) |
| rare | pizza | PO | 91 | 41 | 3 | L5H6 (+0.027), L6H8 (+0.022), L6H0 (+0.021) |
| rare | music | PV | 104 | 35 | 14 | L6H8 (+0.060), L5H2 (+0.056), L4H3 (+0.052) |
| rare | music | VO | 116 | 54 | 7 | L5H6 (+0.070), L3H3 (+0.045), L7H0 (+0.033) |
| rare | music | PO | 90 | 29 | 11 | L8H7 (+0.022), L7H0 (+0.020), L8H5 (+0.018) |
| rare | coffee | PV | 111 | 43 | 14 | L4H0 (-0.048), L6H8 (+0.046), L8H7 (-0.044) |
| rare | coffee | VO | 121 | 48 | 12 | L5H6 (+0.119), L7H0 (+0.082), L6H0 (+0.063) |
| rare | coffee | PO | 103 | 49 | 5 | L5H6 (+0.027), L7H0 (+0.021), L6H8 (+0.020) |
| rare | chess | PV | 107 | 24 | 30 | L4H0 (-0.065), L6H3 (-0.034), L4H3 (+0.032) |
| rare | chess | VO | 120 | 60 | 5 | L5H6 (+0.101), L6H0 (+0.072), L4H3 (+0.061) |
| rare | chess | PO | 76 | 11 | 16 | L5H6 (+0.030), L4H0 (-0.019), L5H4 (-0.014) |
| rare | rain | PV | 121 | 35 | 30 | L7H4 (-0.070), L4H1 (-0.068), L9H10 (-0.066) |
| rare | rain | VO | 92 | 43 | 5 | L5H6 (+0.052), L7H0 (+0.045), L8H7 (+0.024) |
| rare | rain | PO | 74 | 8 | 14 | L4H0 (-0.034), L3H9 (-0.023), L6H11 (-0.022) |
| rare | cars | PV | 124 | 37 | 30 | L8H7 (-0.061), L5H2 (+0.055), L8H6 (+0.055) |
| rare | cars | VO | 103 | 51 | 6 | L5H6 (+0.113), L6H0 (+0.109), L7H0 (+0.081) |
| rare | cars | PO | 86 | 25 | 10 | L5H6 (+0.043), L6H0 (+0.038), L3H3 (-0.026) |
| rare | songs | PV | 113 | 35 | 20 | L4H5 (+0.048), L11H5 (+0.048), L8H1 (+0.042) |
| rare | songs | VO | 103 | 30 | 20 | L6H11 (+0.077), L5H4 (-0.076), L6H0 (+0.064) |
| rare | songs | PO | 68 | 16 | 7 | L5H6 (+0.032), L4H1 (-0.023), L6H8 (+0.016) |
| rare | rice | PV | 108 | 31 | 25 | L4H0 (-0.049), L5H2 (+0.039), L6H8 (+0.039) |
| rare | rice | VO | 108 | 54 | 7 | L5H6 (+0.056), L7H0 (+0.047), L9H3 (+0.045) |
| rare | rice | PO | 98 | 45 | 10 | L6H8 (+0.027), L5H4 (-0.025), L4H3 (-0.021) |
| rare | bikes | PV | 112 | 39 | 17 | L4H0 (-0.065), L6H8 (+0.039), L5H4 (-0.033) |
| rare | bikes | VO | 111 | 44 | 11 | L5H6 (+0.159), L7H0 (+0.073), L6H0 (+0.065) |
| rare | bikes | PO | 101 | 14 | 29 | L5H4 (-0.043), L4H0 (-0.043), L6H11 (-0.038) |
| rare | money | PV | 117 | 46 | 12 | L5H2 (+0.058), L4H3 (+0.055), L9H1 (+0.048) |
| rare | money | VO | 112 | 55 | 5 | L5H6 (+0.238), L7H0 (+0.109), L8H5 (+0.071) |
| rare | money | PO | 89 | 35 | 7 | L5H6 (+0.056), L6H0 (+0.026), L7H0 (+0.024) |

## 8. Surprisal: does sink_mass track how predictable the slot tokens are?

Slot surprisal = −log p of the tokens at positions [29, 30, 31] given everything before them (TL), summed. Slope = within-row regression across every variant (both sides demeaned per row), per nat; CI = bootstrap over rows.

| base | phrase | depth | slope | lo | hi |
|---|---|---|---|---|---|
| common | bikes | early | +0.00116 | +0.00100 | +0.00132 |
| common | bikes | late | +0.00047 | +0.00006 | +0.00089 |
| common | cars | early | +0.00064 | +0.00049 | +0.00079 |
| common | cars | late | +0.00030 | -0.00016 | +0.00071 |
| common | chess | early | -0.00001 | -0.00019 | +0.00017 |
| common | chess | late | +0.00020 | -0.00033 | +0.00072 |
| common | coffee | early | +0.00063 | +0.00051 | +0.00075 |
| common | coffee | late | +0.00014 | -0.00019 | +0.00043 |
| common | money | early | -0.00010 | -0.00023 | +0.00002 |
| common | money | late | -0.00095 | -0.00134 | -0.00059 |
| common | music | early | +0.00007 | -0.00005 | +0.00021 |
| common | music | late | -0.00076 | -0.00115 | -0.00037 |
| common | pizza | early | +0.00049 | +0.00033 | +0.00065 |
| common | pizza | late | +0.00032 | -0.00014 | +0.00077 |
| common | rain | early | +0.00021 | +0.00009 | +0.00032 |
| common | rain | late | -0.00128 | -0.00162 | -0.00096 |
| common | rice | early | +0.00049 | +0.00034 | +0.00064 |
| common | rice | late | +0.00015 | -0.00027 | +0.00056 |
| common | songs | early | +0.00032 | +0.00018 | +0.00045 |
| common | songs | late | +0.00033 | -0.00008 | +0.00070 |
| common | sushi | early | +0.00027 | +0.00011 | +0.00043 |
| common | sushi | late | +0.00036 | -0.00005 | +0.00079 |
| common | water | early | -0.00006 | -0.00019 | +0.00007 |
| common | water | late | -0.00139 | -0.00178 | -0.00097 |
| rare | bikes | early | +0.00039 | +0.00027 | +0.00049 |
| rare | bikes | late | -0.00063 | -0.00098 | -0.00031 |
| rare | cars | early | -0.00004 | -0.00012 | +0.00003 |
| rare | cars | late | -0.00170 | -0.00198 | -0.00142 |
| rare | chess | early | +0.00018 | +0.00011 | +0.00025 |
| rare | chess | late | -0.00232 | -0.00259 | -0.00204 |
| rare | coffee | early | +0.00012 | +0.00004 | +0.00019 |
| rare | coffee | late | -0.00238 | -0.00264 | -0.00211 |
| rare | money | early | -0.00021 | -0.00028 | -0.00015 |
| rare | money | late | -0.00161 | -0.00185 | -0.00139 |
| rare | music | early | -0.00018 | -0.00025 | -0.00012 |
| rare | music | late | -0.00253 | -0.00277 | -0.00229 |
| rare | pizza | early | -0.00027 | -0.00034 | -0.00019 |
| rare | pizza | late | -0.00328 | -0.00356 | -0.00300 |
| rare | rain | early | +0.00031 | +0.00026 | +0.00037 |
| rare | rain | late | -0.00084 | -0.00105 | -0.00061 |
| rare | rice | early | +0.00031 | +0.00023 | +0.00040 |
| rare | rice | late | -0.00138 | -0.00167 | -0.00109 |
| rare | songs | early | +0.00050 | +0.00042 | +0.00057 |
| rare | songs | late | -0.00189 | -0.00216 | -0.00162 |
| rare | sushi | early | +0.00026 | +0.00020 | +0.00033 |
| rare | sushi | late | -0.00072 | -0.00093 | -0.00050 |
| rare | water | early | +0.00038 | +0.00031 | +0.00045 |
| rare | water | late | -0.00044 | -0.00069 | -0.00019 |

Heads whose slope CI excludes 0, by sign (summed over phrases):

| base | depth | slope < 0 | slope > 0 |
|---|---|---|---|
| common | early | 354 | 349 |
| common | late | 369 | 287 |
| rare | early | 389 | 329 |
| rare | late | 601 | 156 |

Per-variant means (descriptive):

| base | phrase | variant | mean_slot_nll | mean_sink_mass_early | mean_sink_mass_late | mean_query_nll |
|---|---|---|---|---|---|---|
| common | sushi | none | 26.791 | 0.327 | 0.589 | 9.125 |
| common | sushi | nl | 25.323 | 0.333 | 0.646 | 10.302 |
| common | sushi | shuffled | 28.721 | 0.333 | 0.606 | 9.382 |
| common | sushi | matched | 31.781 | 0.333 | 0.586 | 9.364 |
| common | sushi | matched_b | 31.520 | 0.333 | 0.585 | 9.234 |
| common | sushi | sushi:I@29 | 25.141 | 0.324 | 0.589 | 9.138 |
| common | sushi | sushi:like@30 | 26.761 | 0.331 | 0.585 | 9.104 |
| common | sushi | sushi:sushi@31 | 32.152 | 0.332 | 0.622 | 9.534 |
| common | sushi | sushi:like@29 | 26.713 | 0.327 | 0.583 | 9.103 |
| common | sushi | sushi:I@30 | 25.172 | 0.326 | 0.595 | 9.175 |
| common | sushi | sushi:I@29+like@30 | 23.574 | 0.327 | 0.586 | 9.171 |
| common | sushi | sushi:like@30+sushi@31 | 30.513 | 0.337 | 0.629 | 9.814 |
| common | sushi | sushi:I@29+sushi@31 | 30.447 | 0.328 | 0.623 | 9.559 |
| rare | sushi | none | 37.886 | 0.332 | 0.595 | 12.567 |
| rare | sushi | nl | 23.262 | 0.326 | 0.623 | 12.972 |
| rare | sushi | shuffled | 29.913 | 0.325 | 0.616 | 12.498 |
| rare | sushi | matched | 32.345 | 0.327 | 0.583 | 12.733 |
| rare | sushi | matched_b | 32.195 | 0.327 | 0.581 | 12.662 |
| rare | sushi | sushi:I@29 | 33.657 | 0.325 | 0.591 | 12.578 |
| rare | sushi | sushi:like@30 | 34.623 | 0.326 | 0.591 | 12.838 |
| rare | sushi | sushi:sushi@31 | 37.956 | 0.331 | 0.620 | 12.183 |
| rare | sushi | sushi:like@29 | 34.691 | 0.325 | 0.595 | 12.580 |
| rare | sushi | sushi:I@30 | 33.484 | 0.325 | 0.591 | 12.837 |
| rare | sushi | sushi:I@29+like@30 | 26.969 | 0.321 | 0.589 | 12.924 |
| rare | sushi | sushi:like@30+sushi@31 | 32.665 | 0.331 | 0.619 | 12.541 |
| rare | sushi | sushi:I@29+sushi@31 | 33.584 | 0.325 | 0.616 | 12.312 |

## 9. Across phrases

Phrases as the unit: pooled_mean = mean of the per-phrase depth-group interactions, CI from bootstrapping phrases; n_pos / n_neg = phrases whose own row-bootstrap CI excludes 0 on that side; spearman = across phrases, the term against the surprisal gap NLL(shuffled) − NLL(nl).

| metric | base | term | depth | n_phrases | pooled_mean | lo | hi | n_pos | n_neg | spearman_rho | spearman_p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sink_mass | common | nl | early | 12 | +0.0023 | +0.0004 | +0.0043 | 7 | 2 | -0.1608 | +0.6175 |
| sink_mass | common | nl | late | 12 | +0.0270 | +0.0220 | +0.0322 | 12 | 0 | +0.4336 | +0.1591 |
| sink_mass | common | shuffled | early | 12 | -0.0032 | -0.0055 | -0.0005 | 2 | 9 | -0.6923 | +0.0126 |
| sink_mass | common | shuffled | late | 12 | +0.0019 | -0.0041 | +0.0073 | 5 | 4 | +0.2517 | +0.4299 |
| sink_mass | common | PV | early | 12 | +0.0021 | +0.0011 | +0.0031 | 8 | 0 | +0.1469 | +0.6488 |
| sink_mass | common | PV | late | 12 | +0.0092 | +0.0061 | +0.0120 | 10 | 1 | +0.4406 | +0.1517 |
| sink_mass | common | VO | early | 12 | +0.0002 | -0.0013 | +0.0017 | 6 | 4 | -0.6503 | +0.0220 |
| sink_mass | common | VO | late | 12 | +0.0125 | +0.0101 | +0.0152 | 12 | 0 | +0.0769 | +0.8122 |
| sink_mass | common | PO | early | 12 | -0.0017 | -0.0024 | -0.0009 | 1 | 11 | -0.3147 | +0.3191 |
| sink_mass | common | PO | late | 12 | +0.0013 | -0.0002 | +0.0026 | 4 | 1 | -0.0350 | +0.9141 |
| sink_mass | rare | nl | early | 12 | +0.0048 | +0.0035 | +0.0062 | 11 | 0 | -0.1259 | +0.6967 |
| sink_mass | rare | nl | late | 12 | +0.0183 | +0.0144 | +0.0223 | 12 | 0 | -0.0280 | +0.9312 |
| sink_mass | rare | shuffled | early | 12 | +0.0024 | -0.0003 | +0.0049 | 8 | 2 | -0.1329 | +0.6806 |
| sink_mass | rare | shuffled | late | 12 | +0.0080 | +0.0030 | +0.0129 | 7 | 2 | -0.0559 | +0.8629 |
| sink_mass | rare | PV | early | 12 | +0.0012 | +0.0003 | +0.0022 | 7 | 3 | +0.0979 | +0.7621 |
| sink_mass | rare | PV | late | 12 | +0.0041 | +0.0028 | +0.0055 | 10 | 0 | -0.3077 | +0.3306 |
| sink_mass | rare | VO | early | 12 | +0.0032 | +0.0020 | +0.0045 | 11 | 1 | -0.2448 | +0.4433 |
| sink_mass | rare | VO | late | 12 | +0.0112 | +0.0084 | +0.0141 | 12 | 0 | -0.2937 | +0.3541 |
| sink_mass | rare | PO | early | 12 | -0.0009 | -0.0017 | -0.0002 | 4 | 7 | +0.0699 | +0.8290 |
| sink_mass | rare | PO | late | 12 | +0.0016 | +0.0004 | +0.0027 | 6 | 1 | +0.0629 | +0.8459 |
| sink_mass | common | nl−shuffled | early | 12 | +0.0055 | +0.0032 | +0.0076 | 11 | 1 | +0.5804 | +0.0479 |
| sink_mass | common | nl−shuffled | late | 12 | +0.0252 | +0.0201 | +0.0307 | 12 | 0 | +0.1888 | +0.5567 |
| sink_mass | rare | nl−shuffled | early | 12 | +0.0024 | +0.0009 | +0.0039 | 10 | 1 | +0.2937 | +0.3541 |
| sink_mass | rare | nl−shuffled | late | 12 | +0.0103 | +0.0084 | +0.0127 | 12 | 0 | +0.0699 | +0.8290 |
| sink_mass | common | triple | early | 12 | +0.0018 | +0.0008 | +0.0026 | 10 | 1 | +0.6643 | +0.0185 |
| sink_mass | common | triple | late | 12 | +0.0040 | +0.0013 | +0.0066 | 8 | 1 | +0.2308 | +0.4705 |
| sink_mass | rare | triple | early | 12 | +0.0013 | +0.0006 | +0.0020 | 11 | 1 | +0.0769 | +0.8122 |
| sink_mass | rare | triple | late | 12 | +0.0014 | -0.0005 | +0.0034 | 6 | 3 | +0.5524 | +0.0625 |
| sink_logit | common | nl | early | 12 | +0.0181 | +0.0053 | +0.0301 | 8 | 2 | -0.1888 | +0.5567 |
| sink_logit | common | nl | late | 12 | +0.1516 | +0.1240 | +0.1801 | 12 | 0 | +0.4825 | +0.1121 |
| sink_logit | common | shuffled | early | 12 | -0.0179 | -0.0347 | +0.0001 | 2 | 9 | -0.6224 | +0.0307 |
| sink_logit | common | shuffled | late | 12 | +0.0082 | -0.0256 | +0.0389 | 5 | 4 | +0.1888 | +0.5567 |
| sink_logit | common | PV | early | 12 | +0.0178 | +0.0110 | +0.0244 | 10 | 0 | +0.2657 | +0.4038 |
| sink_logit | common | PV | late | 12 | +0.0542 | +0.0372 | +0.0701 | 10 | 0 | +0.4406 | +0.1517 |
| sink_logit | common | VO | early | 12 | -0.0018 | -0.0117 | +0.0083 | 5 | 5 | -0.6573 | +0.0202 |
| sink_logit | common | VO | late | 12 | +0.0685 | +0.0552 | +0.0827 | 12 | 0 | +0.1608 | +0.6175 |
| sink_logit | common | PO | early | 12 | -0.0092 | -0.0137 | -0.0042 | 1 | 10 | -0.3217 | +0.3079 |
| sink_logit | common | PO | late | 12 | +0.0089 | +0.0003 | +0.0162 | 5 | 1 | -0.0210 | +0.9484 |
| sink_logit | rare | nl | early | 12 | +0.0357 | +0.0270 | +0.0442 | 12 | 0 | -0.1608 | +0.6175 |
| sink_logit | rare | nl | late | 12 | +0.1119 | +0.0866 | +0.1375 | 12 | 0 | -0.0140 | +0.9656 |
| sink_logit | rare | shuffled | early | 12 | +0.0158 | -0.0009 | +0.0309 | 8 | 2 | -0.1888 | +0.5567 |
| sink_logit | rare | shuffled | late | 12 | +0.0484 | +0.0162 | +0.0802 | 7 | 3 | -0.0490 | +0.8799 |
| sink_logit | rare | PV | early | 12 | +0.0125 | +0.0068 | +0.0182 | 10 | 1 | -0.1189 | +0.7129 |
| sink_logit | rare | PV | late | 12 | +0.0241 | +0.0159 | +0.0325 | 10 | 0 | -0.1678 | +0.6021 |
| sink_logit | rare | VO | early | 12 | +0.0198 | +0.0111 | +0.0282 | 11 | 1 | -0.2727 | +0.3911 |
| sink_logit | rare | VO | late | 12 | +0.0716 | +0.0537 | +0.0902 | 11 | 0 | -0.3357 | +0.2861 |
| sink_logit | rare | PO | early | 12 | -0.0055 | -0.0101 | -0.0011 | 3 | 7 | +0.0839 | +0.7954 |
| sink_logit | rare | PO | late | 12 | +0.0088 | +0.0016 | +0.0154 | 6 | 1 | -0.0350 | +0.9141 |
| sink_logit | common | nl−shuffled | early | 12 | +0.0360 | +0.0202 | +0.0508 | 11 | 1 | +0.5105 | +0.0899 |
| sink_logit | common | nl−shuffled | late | 12 | +0.1433 | +0.1136 | +0.1753 | 12 | 0 | +0.3147 | +0.3191 |
| sink_logit | rare | nl−shuffled | early | 12 | +0.0200 | +0.0100 | +0.0297 | 11 | 1 | +0.1888 | +0.5567 |
| sink_logit | rare | nl−shuffled | late | 12 | +0.0635 | +0.0506 | +0.0786 | 12 | 0 | +0.0699 | +0.8290 |
| sink_logit | common | triple | early | 12 | +0.0114 | +0.0056 | +0.0171 | 9 | 1 | +0.5874 | +0.0446 |
| sink_logit | common | triple | late | 12 | +0.0200 | +0.0051 | +0.0345 | 7 | 1 | +0.1538 | +0.6331 |
| sink_logit | rare | triple | early | 12 | +0.0090 | +0.0038 | +0.0141 | 8 | 1 | +0.1469 | +0.6488 |
| sink_logit | rare | triple | late | 12 | +0.0074 | -0.0033 | +0.0184 | 5 | 3 | +0.5315 | +0.0754 |
| sink_mass | common | VO−PV | early | 12 | -0.0019 | -0.0033 | -0.0002 | 2 | 10 | -0.7832 | +0.0026 |
| sink_mass | common | VO−PV | late | 12 | +0.0033 | +0.0003 | +0.0065 | 8 | 4 | -0.1678 | +0.6021 |
| sink_mass | rare | VO−PV | early | 12 | +0.0020 | +0.0003 | +0.0036 | 10 | 2 | +0.0140 | +0.9656 |
| sink_mass | rare | VO−PV | late | 12 | +0.0072 | +0.0045 | +0.0102 | 12 | 0 | -0.0629 | +0.8459 |
| sink_logit | common | VO−PV | early | 12 | -0.0196 | -0.0295 | -0.0079 | 2 | 10 | -0.7762 | +0.0030 |
| sink_logit | common | VO−PV | late | 12 | +0.0143 | -0.0032 | +0.0318 | 8 | 4 | -0.1678 | +0.6021 |
| sink_logit | rare | VO−PV | early | 12 | +0.0073 | -0.0043 | +0.0189 | 10 | 2 | +0.0000 | +1.0000 |
| sink_logit | rare | VO−PV | late | 12 | +0.0475 | +0.0305 | +0.0667 | 12 | 0 | -0.2098 | +0.5128 |
| sink_mass | common | VO−PO | early | 12 | +0.0019 | +0.0002 | +0.0037 | 9 | 3 | -0.3007 | +0.3423 |
| sink_mass | common | VO−PO | late | 12 | +0.0112 | +0.0090 | +0.0134 | 12 | 0 | +0.0420 | +0.8970 |
| sink_mass | rare | VO−PO | early | 12 | +0.0041 | +0.0029 | +0.0055 | 11 | 1 | -0.0350 | +0.9141 |
| sink_mass | rare | VO−PO | late | 12 | +0.0097 | +0.0072 | +0.0123 | 12 | 0 | -0.3497 | +0.2652 |
| sink_logit | common | VO−PO | early | 12 | +0.0074 | -0.0035 | +0.0189 | 8 | 4 | -0.2937 | +0.3541 |
| sink_logit | common | VO−PO | late | 12 | +0.0596 | +0.0484 | +0.0715 | 12 | 0 | +0.0769 | +0.8122 |
| sink_logit | rare | VO−PO | early | 12 | +0.0253 | +0.0158 | +0.0341 | 11 | 1 | -0.0839 | +0.7954 |
| sink_logit | rare | VO−PO | late | 12 | +0.0629 | +0.0476 | +0.0786 | 12 | 0 | -0.3497 | +0.2652 |

Late-layer interaction for every phrase (* = row-bootstrap CI excludes 0):

| metric | base | phrase | nl | shuffled | PV | VO | PO | nl−shuffled | triple | nll gap (shuffled − nl) |
|---|---|---|---|---|---|---|---|---|---|---|
| sink_mass | common | I like sushi | +0.0282* | -0.0156* | +0.0009 | +0.0118* | +0.0009 | +0.0438* | +0.0146* | +3.40 |
| sink_mass | common | we need water | +0.0430* | +0.0032 | +0.0149* | +0.0220* | +0.0022 | +0.0398* | +0.0039* | +4.21 |
| sink_mass | common | she ate pizza | +0.0336* | +0.0072* | +0.0132* | +0.0202* | +0.0036* | +0.0264* | -0.0034* | +3.42 |
| sink_mass | common | they love music | +0.0212* | +0.0031 | +0.0088* | +0.0060* | +0.0015 | +0.0181* | +0.0049* | +2.86 |
| sink_mass | common | he drinks coffee | +0.0254* | -0.0054* | +0.0110* | +0.0101* | +0.0000 | +0.0307* | +0.0043* | +4.25 |
| sink_mass | common | you play chess | +0.0207* | +0.0030 | +0.0074* | +0.0130* | +0.0009 | +0.0177* | -0.0007 | +3.51 |
| sink_mass | common | I hate rain | +0.0310* | +0.0073* | +0.0122* | +0.0084* | +0.0002 | +0.0237* | +0.0102* | +4.75 |
| sink_mass | common | we sell cars | +0.0333* | +0.0157* | +0.0125* | +0.0152* | +0.0030* | +0.0176* | +0.0026 | +4.82 |
| sink_mass | common | she wrote songs | +0.0122* | -0.0164* | -0.0028* | +0.0088* | -0.0004 | +0.0286* | +0.0066* | +4.19 |
| sink_mass | common | they grow rice | +0.0340* | +0.0067* | +0.0166* | +0.0095* | +0.0031* | +0.0273* | +0.0047* | +3.56 |
| sink_mass | common | he fixed bikes | +0.0120* | -0.0067* | +0.0046* | +0.0098* | -0.0048* | +0.0187* | +0.0024* | +0.20 |
| sink_mass | common | you want money | +0.0299* | +0.0201* | +0.0109* | +0.0155* | +0.0056* | +0.0098* | -0.0022 | +2.89 |
| sink_mass | rare | I like sushi | +0.0115* | -0.0002 | +0.0026* | +0.0027* | +0.0003 | +0.0117* | +0.0060* | +6.65 |
| sink_mass | rare | we need water | +0.0167* | -0.0036* | +0.0061* | +0.0096* | +0.0004 | +0.0203* | +0.0006 | +5.89 |
| sink_mass | rare | she ate pizza | +0.0243* | +0.0182* | +0.0065* | +0.0188* | +0.0037* | +0.0061* | -0.0047* | +6.12 |
| sink_mass | rare | they love music | +0.0173* | +0.0107* | +0.0060* | +0.0089* | +0.0021* | +0.0066* | +0.0004 | +4.19 |
| sink_mass | rare | he drinks coffee | +0.0298* | +0.0168* | +0.0059* | +0.0135* | +0.0045* | +0.0130* | +0.0059* | +6.20 |
| sink_mass | rare | you play chess | +0.0231* | +0.0161* | +0.0004 | +0.0191* | -0.0002 | +0.0070* | +0.0038* | +4.75 |
| sink_mass | rare | I hate rain | +0.0112* | +0.0022 | +0.0007 | +0.0059* | -0.0010 | +0.0090* | +0.0056* | +6.55 |
| sink_mass | rare | we sell cars | +0.0206* | +0.0132* | +0.0026* | +0.0134* | +0.0029* | +0.0075* | +0.0017* | +6.26 |
| sink_mass | rare | she wrote songs | +0.0089* | -0.0023 | +0.0037* | +0.0071* | +0.0011 | +0.0112* | -0.0030* | +5.16 |
| sink_mass | rare | they grow rice | +0.0183* | +0.0083* | +0.0018* | +0.0100* | +0.0033* | +0.0100* | +0.0032* | +4.79 |
| sink_mass | rare | he fixed bikes | +0.0091* | -0.0034* | +0.0040* | +0.0075* | -0.0019* | +0.0124* | -0.0004 | +0.79 |
| sink_mass | rare | you want money | +0.0290* | +0.0196* | +0.0090* | +0.0186* | +0.0034* | +0.0094* | -0.0021* | +2.74 |
| sink_logit | common | I like sushi | +0.1532* | -0.0988* | +0.0053 | +0.0618* | +0.0048 | +0.2520* | +0.0814* | +3.40 |
| sink_logit | common | we need water | +0.2410* | +0.0168 | +0.0822* | +0.1236* | +0.0142* | +0.2242* | +0.0209* | +4.21 |
| sink_logit | common | she ate pizza | +0.1820* | +0.0350* | +0.0764* | +0.1053* | +0.0212* | +0.1469* | -0.0209* | +3.42 |
| sink_logit | common | they love music | +0.1214* | +0.0150 | +0.0518* | +0.0344* | +0.0092 | +0.1064* | +0.0259* | +2.86 |
| sink_logit | common | he drinks coffee | +0.1394* | -0.0367* | +0.0670* | +0.0527* | -0.0013 | +0.1761* | +0.0210* | +4.25 |
| sink_logit | common | you play chess | +0.1132* | +0.0162 | +0.0429* | +0.0702* | +0.0098 | +0.0970* | -0.0097 | +3.51 |
| sink_logit | common | I hate rain | +0.1684* | +0.0322* | +0.0693* | +0.0451* | +0.0023 | +0.1362* | +0.0518* | +4.75 |
| sink_logit | common | we sell cars | +0.1890* | +0.0828* | +0.0754* | +0.0842* | +0.0183* | +0.1062* | +0.0111 | +4.82 |
| sink_logit | common | she wrote songs | +0.0793* | -0.0847* | -0.0062 | +0.0514* | -0.0005 | +0.1640* | +0.0346* | +4.19 |
| sink_logit | common | they grow rice | +0.2004* | +0.0404* | +0.1005* | +0.0578* | +0.0191* | +0.1600* | +0.0230* | +3.56 |
| sink_logit | common | he fixed bikes | +0.0653* | -0.0326* | +0.0248* | +0.0516* | -0.0241* | +0.0980* | +0.0130 | +0.20 |
| sink_logit | common | you want money | +0.1661* | +0.1130* | +0.0610* | +0.0836* | +0.0334* | +0.0531* | -0.0119 | +2.89 |
| sink_logit | rare | I like sushi | +0.0573* | -0.0020 | +0.0132* | +0.0140 | -0.0022 | +0.0593* | +0.0323* | +6.65 |
| sink_logit | rare | we need water | +0.1085* | -0.0199* | +0.0361* | +0.0631* | +0.0028 | +0.1284* | +0.0064 | +5.89 |
| sink_logit | rare | she ate pizza | +0.1493* | +0.1134* | +0.0372* | +0.1173* | +0.0213* | +0.0359* | -0.0265* | +6.12 |
| sink_logit | rare | they love music | +0.1049* | +0.0660* | +0.0330* | +0.0603* | +0.0116* | +0.0389* | +0.0000 | +4.19 |
| sink_logit | rare | he drinks coffee | +0.1832* | +0.1001* | +0.0351* | +0.0887* | +0.0265* | +0.0831* | +0.0330* | +6.20 |
| sink_logit | rare | you play chess | +0.1417* | +0.1032* | +0.0011 | +0.1207* | +0.0002 | +0.0385* | +0.0196* | +4.75 |
| sink_logit | rare | I hate rain | +0.0652* | +0.0059 | +0.0040 | +0.0393* | -0.0068 | +0.0592* | +0.0287* | +6.55 |
| sink_logit | rare | we sell cars | +0.1253* | +0.0782* | +0.0192* | +0.0837* | +0.0156* | +0.0471* | +0.0068 | +6.26 |
| sink_logit | rare | she wrote songs | +0.0582* | -0.0197* | +0.0273* | +0.0398* | +0.0071 | +0.0779* | -0.0161* | +5.16 |
| sink_logit | rare | they grow rice | +0.1156* | +0.0553* | +0.0116* | +0.0658* | +0.0198* | +0.0603* | +0.0184* | +4.79 |
| sink_logit | rare | he fixed bikes | +0.0518* | -0.0242* | +0.0181* | +0.0488* | -0.0139* | +0.0760* | -0.0012 | +0.79 |
| sink_logit | rare | you want money | +0.1815* | +0.1243* | +0.0533* | +0.1180* | +0.0231* | +0.0571* | -0.0130* | +2.74 |

Pair familiarity (−log p of the pair's second token given BOS + its first, TL; lower = more familiar) against each pair's interaction, across phrases:

| metric | base | label | depth | n_phrases | mean_nll | spearman_rho | spearman_p |
|---|---|---|---|---|---|---|---|
| sink_mass | common | PV | early | 12 | +6.547 | -0.371 | +0.236 |
| sink_mass | common | PV | late | 12 | +6.547 | +0.168 | +0.602 |
| sink_mass | rare | PV | early | 12 | +6.547 | -0.783 | +0.003 |
| sink_mass | rare | PV | late | 12 | +6.547 | -0.182 | +0.572 |
| sink_logit | common | PV | early | 12 | +6.547 | -0.427 | +0.167 |
| sink_logit | common | PV | late | 12 | +6.547 | +0.168 | +0.602 |
| sink_logit | rare | PV | early | 12 | +6.547 | -0.790 | +0.002 |
| sink_logit | rare | PV | late | 12 | +6.547 | -0.217 | +0.499 |
| sink_mass | common | VO | early | 12 | +9.744 | -0.175 | +0.587 |
| sink_mass | common | VO | late | 12 | +9.744 | -0.273 | +0.391 |
| sink_mass | rare | VO | early | 12 | +9.744 | -0.063 | +0.846 |
| sink_mass | rare | VO | late | 12 | +9.744 | -0.545 | +0.067 |
| sink_logit | common | VO | early | 12 | +9.744 | -0.056 | +0.863 |
| sink_logit | common | VO | late | 12 | +9.744 | -0.252 | +0.430 |
| sink_logit | rare | VO | early | 12 | +9.744 | -0.049 | +0.880 |
| sink_logit | rare | VO | late | 12 | +9.744 | -0.559 | +0.059 |

| phrase | PV | VO |
|---|---|---|
| I hate rain | 6.01 | 11.19 |
| I like sushi | 4.84 | 10.97 |
| he drinks coffee | 10.06 | 7.30 |
| he fixed bikes | 9.99 | 12.80 |
| she ate pizza | 7.78 | 7.82 |
| she wrote songs | 4.40 | 15.18 |
| they grow rice | 7.86 | 9.68 |
| they love music | 6.24 | 6.74 |
| we need water | 3.31 | 8.34 |
| we sell cars | 7.28 | 9.76 |
| you play chess | 7.07 | 10.08 |
| you want money | 3.72 | 7.07 |

## Figures

- [rare_sushi_near_sink_mass_contrasts](figures/rare_sushi_near_sink_mass_contrasts.png)
- [rare_sushi_sink_mass_tracked_forest](figures/rare_sushi_sink_mass_tracked_forest.png)
- [rare_sushi_near_slot_mass_contrasts](figures/rare_sushi_near_slot_mass_contrasts.png)
- [rare_sushi_slot_mass_tracked_forest](figures/rare_sushi_slot_mass_tracked_forest.png)
- [rare_sushi_sink_mass_layer_profile](figures/rare_sushi_sink_mass_layer_profile.png)
- [common_sushi_near_sink_mass_contrasts](figures/common_sushi_near_sink_mass_contrasts.png)
- [common_sushi_sink_mass_tracked_forest](figures/common_sushi_sink_mass_tracked_forest.png)
- [common_sushi_near_slot_mass_contrasts](figures/common_sushi_near_slot_mass_contrasts.png)
- [common_sushi_slot_mass_tracked_forest](figures/common_sushi_slot_mass_tracked_forest.png)
- [common_sushi_sink_mass_layer_profile](figures/common_sushi_sink_mass_layer_profile.png)
- [common_sushi_near_additivity](figures/common_sushi_near_additivity.png)
- [rare_sushi_near_additivity](figures/rare_sushi_near_additivity.png)
- [phrase_terms_sink_mass](figures/phrase_terms_sink_mass.png)
- [phrase_terms_sink_logit](figures/phrase_terms_sink_logit.png)
- [surprisal_variants](figures/surprisal_variants.png)
- [meaningful_counts_sink_mass](figures/meaningful_counts_sink_mass.png)
- [meaningful_counts_slot_mass](figures/meaningful_counts_slot_mass.png)

## Caveats

- 12 phrases from one template (pronoun verb object) are a small, hand-picked sample; CIs over phrases describe these phrases, not language in general.
- A slot at position p < 32 changes every later position's residual stream, so a contrast measures the slot tokens' total effect at the query (as keys and through earlier heads), not a single route.
- `matched` controls match surface form and id class, not frequency or meaning; `identity` therefore mixes everything else that distinguishes the phrase's tokens from same-form random words.
