# exp2: a natural-language phrase in random-token context

- Run `20261003-230841_exp2`. Bases: rare [1000, 39999], common [256, 999], 1000 rows each, BOS + 32 random ids, no repeats, no phrase ids. Metric at the final query (position 32).
- Phrases: `sushi` = ' I like sushi' [314, 588, 36324] (shuffled [588, 314, 36324]). Placements (model positions): near [29, 30, 31].
- Controls: `shuffled` (same ids, scrambled), `matched` / `matched_b` (per row, random ids with the same leading_space, is_alpha, is_capitalized and id class as each phrase token; pool sizes {'sushi': [43, 280, 16019]}).
- Contrasts (paired per row, a − b): total = nl − none, order = nl − shuffled, identity = shuffled − matched, form = matched − none, placebo = matched − matched_b, I@29 = I@29 − none, like@30 = like@30 − none, sushi@31 = sushi@31 − none, like@29 = like@29 − none, I@30 = I@30 − none. total = order + identity + form.
- Meaningful = BH q ≤ 0.05 (Wilcoxon signed-rank, over every test of a metric) and |Δ| ≥ the metric's threshold {'sink_mass': 0.02, 'slot_mass': 0.02, 'slot_logratio': 0.1, 'prev_mass': 0.02, 'entropy': 0.05}. Bold = meaningful. Predictions were written before any data: [PREDICTIONS.md](PREDICTIONS.md).

## Checks

| check | backend | result | detail |
|---|---|---|---|
| Attention rows sum to 1 | TL | PASS | max abs(sum − 1) = 4.5e-07 |
| Attention rows sum to 1 | HF | PASS | max abs(sum − 1) = 4.5e-07 |
| Determinism (first 50 re-extracted) | TL | PASS | bit-identical |
| Determinism (first 50 re-extracted) | HF | PASS | bit-identical |
| Design invariants (no repeats, pairing, slot contents, pools) | - | PASS | checked before extraction |
| `none` rows replicate exp1 (rare) | TL | PASS | per-head mean sink_mass r = 0.9998 |
| `none` rows replicate exp1 (common) | TL | PASS | per-head mean sink_mass r = 0.9999 |
| Cross-library | TL vs HF | PASS | max |Δ| probability metrics 5.3e-05; labels agree 100.00%; meaningful decisions (sink_mass) agree 100.00%; max |Δ effect size| 5.4e-02 |

## 1. Pre-registered predictions

| id | kind | result | detail |
|---|---|---|---|
| Q1 | depth_term_sign | PASS | common: +0.0279 [+0.0244, +0.0313]; rare: +0.0132 [+0.0110, +0.0154] |
| Q2 | depth_term_sign | PASS | common: +0.0425 [+0.0396, +0.0456]; rare: +0.0091 [+0.0071, +0.0111] |
| Q3 | order_fit_r2_below | PASS | common: r² 0.001 (slope -0.15); rare: r² 0.007 (slope 0.11) |
| Q4 | surprisal_sign | PASS | common: slope -0.00139 [-0.00170, -0.00108] per nat; rare: slope -0.00107 [-0.00125, -0.00089] per nat |
| Q5 | surprisal_depth_ratio | PASS | common: |late| 0.00139 vs |early| 0.00022; rare: |late| 0.00107 vs |early| 0.00025 |
| Q6a | controls | PASS | L4H11 prev_mass: min over cells 0.980 (floor 0.95); L5H1 sink_mass: min over cells 0.971 (floor 0.9); L0H1 mode Self in 20/20 cells |
| Q6b | few_meaningful | PASS | common/sushi/near: 0 (limit 0); rare/sushi/near: 0 (limit 0) |

## 2. Meaningful heads per contrast: sink_mass

Out of 144 heads. The placebo row is the noise floor.

| contrast | common / sushi / near | rare / sushi / near |
|---|---|---|
| total | 73 | 80 |
| order | 61 | 27 |
| identity | 68 | 76 |
| form | 47 | 54 |
| placebo | 0 | 0 |
| I@29 | 28 | 26 |
| like@30 | 27 | 49 |
| sushi@31 | 73 | 48 |
| like@29 | 22 | 27 |
| I@30 | 39 | 42 |

Effect sizes (sink_mass): 

| contrast | base | phrase | placement | median |Δ| | 90th pct |Δ| | largest |
|---|---|---|---|---|---|---|
| total | common | sushi | near | 0.0212 | 0.1164 | L7H0 (+0.246) |
| order | common | sushi | near | 0.0128 | 0.0725 | L6H8 (+0.163) |
| identity | common | sushi | near | 0.0180 | 0.0637 | L6H0 (+0.153) |
| form | common | sushi | near | 0.0093 | 0.0403 | L4H3 (+0.104) |
| placebo | common | sushi | near | 0.0019 | 0.0064 | L4H8 (-0.014) |
| I@29 | common | sushi | near | 0.0075 | 0.0258 | L2H4 (-0.051) |
| like@30 | common | sushi | near | 0.0095 | 0.0282 | L4H8 (+0.060) |
| sushi@31 | common | sushi | near | 0.0205 | 0.0652 | L7H0 (+0.145) |
| like@29 | common | sushi | near | 0.0072 | 0.0233 | L9H1 (-0.042) |
| I@30 | common | sushi | near | 0.0115 | 0.0417 | L8H6 (+0.068) |
| total | rare | sushi | near | 0.0234 | 0.0783 | L7H0 (+0.233) |
| order | rare | sushi | near | 0.0069 | 0.0304 | L5H2 (+0.093) |
| identity | rare | sushi | near | 0.0215 | 0.0759 | L6H0 (+0.232) |
| form | rare | sushi | near | 0.0129 | 0.0449 | L5H2 (-0.128) |
| placebo | rare | sushi | near | 0.0018 | 0.0054 | L8H7 (-0.015) |
| I@29 | rare | sushi | near | 0.0071 | 0.0296 | L2H4 (-0.092) |
| like@30 | rare | sushi | near | 0.0117 | 0.0474 | L5H2 (-0.134) |
| sushi@31 | rare | sushi | near | 0.0127 | 0.0507 | L7H0 (+0.185) |
| like@29 | rare | sushi | near | 0.0079 | 0.0381 | L5H2 (-0.090) |
| I@30 | rare | sushi | near | 0.0103 | 0.0452 | L2H4 (-0.131) |

## 2. Meaningful heads per contrast: slot_mass

Out of 144 heads. The placebo row is the noise floor.

| contrast | common / sushi / near | rare / sushi / near |
|---|---|---|
| total | 88 | 73 |
| order | 44 | 40 |
| identity | 59 | 46 |
| form | 72 | 58 |
| placebo | 0 | 0 |
| I@29 | 39 | 39 |
| like@30 | 38 | 40 |
| sushi@31 | 67 | 39 |
| like@29 | 36 | 49 |
| I@30 | 42 | 52 |

Effect sizes (slot_mass): 

| contrast | base | phrase | placement | median |Δ| | 90th pct |Δ| | largest |
|---|---|---|---|---|---|---|
| total | common | sushi | near | 0.0272 | 0.0796 | L5H3 (+0.186) |
| order | common | sushi | near | 0.0104 | 0.0379 | L6H8 (-0.161) |
| identity | common | sushi | near | 0.0158 | 0.0527 | L6H0 (-0.134) |
| form | common | sushi | near | 0.0199 | 0.0494 | L2H3 (+0.139) |
| placebo | common | sushi | near | 0.0013 | 0.0061 | L4H0 (-0.015) |
| I@29 | common | sushi | near | 0.0090 | 0.0422 | L11H4 (+0.090) |
| like@30 | common | sushi | near | 0.0090 | 0.0482 | L2H8 (+0.105) |
| sushi@31 | common | sushi | near | 0.0179 | 0.0643 | L10H0 (+0.172) |
| like@29 | common | sushi | near | 0.0082 | 0.0387 | L4H5 (+0.092) |
| I@30 | common | sushi | near | 0.0089 | 0.0482 | L3H6 (+0.141) |
| total | rare | sushi | near | 0.0202 | 0.0669 | L7H0 (-0.201) |
| order | rare | sushi | near | 0.0082 | 0.0378 | L4H0 (+0.107) |
| identity | rare | sushi | near | 0.0133 | 0.0559 | L6H0 (-0.224) |
| form | rare | sushi | near | 0.0151 | 0.0483 | L5H2 (+0.117) |
| placebo | rare | sushi | near | 0.0014 | 0.0048 | L8H7 (+0.013) |
| I@29 | rare | sushi | near | 0.0117 | 0.0348 | L11H4 (+0.146) |
| like@30 | rare | sushi | near | 0.0112 | 0.0456 | L2H0 (+0.138) |
| sushi@31 | rare | sushi | near | 0.0104 | 0.0398 | L7H0 (-0.164) |
| like@29 | rare | sushi | near | 0.0118 | 0.0373 | L10H8 (+0.122) |
| I@30 | rare | sushi | near | 0.0138 | 0.0397 | L2H4 (+0.146) |

## 2. Meaningful heads per contrast: slot_logratio

Out of 144 heads. The placebo row is the noise floor.

| contrast | common / sushi / near | rare / sushi / near |
|---|---|---|
| total | 121 | 118 |
| order | 79 | 80 |
| identity | 110 | 105 |
| form | 114 | 108 |
| placebo | 0 | 0 |
| I@29 | 94 | 98 |
| like@30 | 96 | 105 |
| sushi@31 | 108 | 101 |
| like@29 | 88 | 100 |
| I@30 | 93 | 107 |

Effect sizes (slot_logratio): 

| contrast | base | phrase | placement | median |Δ| | 90th pct |Δ| | largest |
|---|---|---|---|---|---|---|
| total | common | sushi | near | 0.3718 | 0.9339 | L4H11 (-6.372) |
| order | common | sushi | near | 0.1195 | 0.5238 | L4H11 (-2.390) |
| identity | common | sushi | near | 0.2285 | 0.6301 | L4H11 (-1.643) |
| form | common | sushi | near | 0.2208 | 0.5890 | L4H11 (-2.339) |
| placebo | common | sushi | near | 0.0179 | 0.0566 | L4H11 (+0.165) |
| I@29 | common | sushi | near | 0.1573 | 0.4089 | L11H4 (+1.123) |
| like@30 | common | sushi | near | 0.1507 | 0.3768 | L9H1 (+1.163) |
| sushi@31 | common | sushi | near | 0.2402 | 0.7635 | L3H0 (+2.207) |
| like@29 | common | sushi | near | 0.1489 | 0.3601 | L9H1 (+1.186) |
| I@30 | common | sushi | near | 0.1706 | 0.4601 | L11H4 (+1.057) |
| total | rare | sushi | near | 0.3339 | 0.8927 | L4H11 (-4.800) |
| order | rare | sushi | near | 0.1373 | 0.4336 | L4H11 (-2.317) |
| identity | rare | sushi | near | 0.2235 | 0.7811 | L6H0 (-1.657) |
| form | rare | sushi | near | 0.2407 | 0.6760 | L5H1 (-1.414) |
| placebo | rare | sushi | near | 0.0184 | 0.0486 | L4H11 (+0.190) |
| I@29 | rare | sushi | near | 0.1744 | 0.4918 | L11H4 (+1.570) |
| like@30 | rare | sushi | near | 0.1917 | 0.4913 | L10H8 (+1.327) |
| sushi@31 | rare | sushi | near | 0.1919 | 0.4633 | L7H0 (-1.133) |
| like@29 | rare | sushi | near | 0.1859 | 0.4014 | L10H8 (+1.319) |
| I@30 | rare | sushi | near | 0.2229 | 0.5346 | L4H11 (-2.289) |

## 3. Tracked heads (exp1ext rule-selected + controls): Δ sink_mass

**rare bases, `sushi`, near**

| head | group | total | order | identity | form |
|---|---|---|---|---|---|
| L1H5 | sink rises | -0.002 | -0.003 | -0.001 | +0.003 |
| L1H7 | sink rises | +0.004 | -0.001 | +0.011 | -0.006 |
| L7H7 | sink rises | **-0.023** | -0.002 | **-0.052** | **+0.031** |
| L9H11 | sink rises | **+0.038** | **+0.030** | **+0.037** | **-0.029** |
| L10H11 | sink rises | **+0.028** | -0.005 | **+0.040** | -0.006 |
| L11H1 | sink rises | **+0.041** | -0.002 | **+0.058** | -0.015 |
| L11H3 | sink rises | **+0.027** | +0.013 | **+0.030** | -0.016 |
| L2H1 | sink rises | **+0.064** | +0.001 | **+0.079** | -0.016 |
| L2H6 | sink rises | **+0.021** | -0.002 | **+0.025** | -0.002 |
| L3H6 | sink rises | -0.007 | -0.003 | **-0.028** | **+0.024** |
| L7H11 | sink rises | **+0.043** | **+0.031** | **+0.020** | -0.008 |
| L10H4 | sink rises | **+0.041** | +0.006 | **+0.043** | -0.008 |
| L11H9 | sink rises | **+0.042** | -0.005 | **+0.053** | -0.006 |
| L1H8 | sink falls | +0.001 | +0.001 | -0.003 | +0.002 |
| L2H0 | sink falls | **-0.046** | +0.009 | -0.014 | **-0.041** |
| L2H4 | sink falls | **-0.077** | **+0.028** | **-0.054** | **-0.051** |
| L3H3 | sink falls | +0.016 | **-0.023** | -0.003 | **+0.042** |
| L6H4 | sink falls | **+0.087** | +0.001 | **+0.089** | -0.003 |
| L6H8 | sink falls | **+0.097** | **+0.075** | **+0.044** | **-0.022** |
| L7H0 | sink falls | **+0.233** | **+0.049** | **+0.183** | +0.001 |
| L10H9 | sink falls | -0.006 | +0.017 | **+0.023** | **-0.046** |
| L4H0 | sink falls | **-0.148** | **-0.088** | +0.008 | **-0.069** |
| L5H2 | sink falls | **-0.055** | **+0.093** | **-0.021** | **-0.128** |
| L5H6 | sink falls | **+0.136** | **+0.054** | **+0.076** | +0.006 |
| L8H2 | sink falls | **+0.064** | **+0.023** | **+0.056** | -0.015 |
| L9H8 | sink falls | **+0.095** | +0.020 | **+0.099** | **-0.023** |
| L0H6 | label mix only | +0.003 | -0.000 | +0.001 | +0.003 |
| L1H0 | label mix only | +0.002 | +0.001 | +0.000 | +0.000 |
| L3H2 | label mix only | +0.000 | -0.007 | -0.007 | +0.015 |
| L10H0 | label mix only | **+0.055** | -0.014 | **+0.066** | +0.003 |
| L0H1 | control | +0.000 | +0.000 | +0.000 | -0.000 |
| L4H11 | control | -0.000 | +0.000 | +0.000 | -0.000 |
| L5H1 | control | +0.005 | +0.000 | -0.002 | +0.006 |

**common bases, `sushi`, near**

| head | group | total | order | identity | form |
|---|---|---|---|---|---|
| L1H5 | sink rises | -0.005 | -0.004 | -0.005 | +0.004 |
| L1H7 | sink rises | +0.010 | +0.000 | +0.012 | -0.002 |
| L7H7 | sink rises | **+0.050** | -0.006 | **+0.087** | **-0.032** |
| L9H11 | sink rises | **+0.041** | **+0.076** | +0.015 | **-0.049** |
| L10H11 | sink rises | +0.016 | **+0.021** | **+0.024** | **-0.028** |
| L11H1 | sink rises | **+0.119** | **+0.068** | **+0.084** | **-0.033** |
| L11H3 | sink rises | **+0.100** | **+0.097** | **+0.035** | **-0.031** |
| L2H1 | sink rises | **+0.105** | +0.003 | **+0.099** | +0.003 |
| L2H6 | sink rises | +0.005 | -0.006 | +0.014 | -0.003 |
| L3H6 | sink rises | **-0.025** | -0.009 | **-0.021** | +0.005 |
| L7H11 | sink rises | **+0.136** | **+0.089** | **+0.095** | **-0.048** |
| L10H4 | sink rises | **+0.113** | **+0.065** | **+0.063** | -0.015 |
| L11H9 | sink rises | **+0.122** | **+0.079** | **+0.046** | -0.002 |
| L1H8 | sink falls | +0.002 | +0.000 | +0.001 | +0.001 |
| L2H0 | sink falls | -0.002 | +0.005 | -0.007 | +0.001 |
| L2H4 | sink falls | -0.007 | +0.007 | **-0.040** | **+0.026** |
| L3H3 | sink falls | **+0.051** | -0.020 | **+0.041** | **+0.029** |
| L6H4 | sink falls | **+0.124** | **-0.025** | **+0.137** | +0.012 |
| L6H8 | sink falls | **+0.216** | **+0.163** | **+0.031** | **+0.022** |
| L7H0 | sink falls | **+0.246** | **+0.141** | **+0.033** | **+0.072** |
| L10H9 | sink falls | **+0.088** | **+0.147** | **-0.034** | **-0.024** |
| L4H0 | sink falls | **-0.099** | **-0.057** | **-0.066** | **+0.024** |
| L5H2 | sink falls | **+0.105** | **+0.152** | **-0.058** | +0.011 |
| L5H6 | sink falls | **+0.028** | **+0.052** | **-0.056** | **+0.032** |
| L8H2 | sink falls | **+0.130** | **+0.065** | **+0.106** | **-0.041** |
| L9H8 | sink falls | **+0.123** | **+0.067** | +0.006 | **+0.050** |
| L0H6 | label mix only | -0.005 | -0.000 | -0.000 | -0.005 |
| L1H0 | label mix only | -0.008 | +0.002 | -0.005 | -0.005 |
| L3H2 | label mix only | +0.003 | -0.014 | +0.006 | +0.010 |
| L10H0 | label mix only | **-0.066** | -0.012 | **-0.056** | +0.003 |
| L0H1 | control | -0.000 | -0.000 | -0.000 | -0.000 |
| L4H11 | control | +0.000 | +0.000 | -0.000 | +0.000 |
| L5H1 | control | +0.003 | -0.000 | +0.003 | +0.000 |

## 4. Does the final token attend to the phrase? Δ slot_mass, tracked heads

**rare bases, `sushi`, near**

| head | group | total | order | identity | form |
|---|---|---|---|---|---|
| L1H5 | sink rises | **+0.020** | -0.005 | **+0.037** | -0.011 |
| L1H7 | sink rises | **+0.022** | -0.005 | **+0.042** | -0.015 |
| L7H7 | sink rises | **-0.064** | -0.007 | -0.011 | **-0.045** |
| L9H11 | sink rises | **-0.029** | -0.008 | -0.013 | -0.008 |
| L10H11 | sink rises | **-0.071** | -0.007 | **-0.033** | **-0.031** |
| L11H1 | sink rises | **-0.044** | -0.003 | -0.011 | **-0.030** |
| L11H3 | sink rises | -0.011 | +0.020 | -0.006 | **-0.026** |
| L2H1 | sink rises | **-0.043** | -0.013 | -0.002 | **-0.029** |
| L2H6 | sink rises | +0.008 | +0.009 | **-0.046** | **+0.044** |
| L3H6 | sink rises | +0.012 | **+0.021** | **+0.028** | **-0.037** |
| L7H11 | sink rises | **-0.053** | +0.004 | **-0.027** | **-0.030** |
| L10H4 | sink rises | -0.016 | **+0.025** | -0.004 | **-0.037** |
| L11H9 | sink rises | **-0.050** | -0.003 | -0.012 | **-0.035** |
| L1H8 | sink falls | +0.014 | +0.000 | +0.011 | +0.003 |
| L2H0 | sink falls | **+0.107** | **-0.042** | **+0.059** | **+0.091** |
| L2H4 | sink falls | **+0.082** | **-0.036** | **+0.058** | **+0.061** |
| L3H3 | sink falls | -0.013 | **+0.023** | -0.009 | **-0.028** |
| L6H4 | sink falls | **-0.021** | +0.005 | -0.000 | **-0.025** |
| L6H8 | sink falls | **-0.100** | **-0.064** | **-0.057** | **+0.021** |
| L7H0 | sink falls | **-0.201** | **-0.028** | **-0.164** | -0.009 |
| L10H9 | sink falls | +0.009 | +0.003 | -0.015 | **+0.022** |
| L4H0 | sink falls | **+0.142** | **+0.107** | -0.014 | **+0.048** |
| L5H2 | sink falls | **+0.029** | **-0.087** | -0.001 | **+0.117** |
| L5H6 | sink falls | **-0.139** | **-0.043** | **-0.090** | -0.006 |
| L8H2 | sink falls | -0.014 | **-0.022** | +0.015 | -0.006 |
| L9H8 | sink falls | **-0.027** | +0.006 | -0.011 | **-0.022** |
| L0H6 | label mix only | **-0.046** | +0.000 | -0.008 | **-0.037** |
| L1H0 | label mix only | **+0.029** | **-0.029** | -0.001 | **+0.059** |
| L3H2 | label mix only | -0.013 | **+0.025** | **-0.052** | +0.014 |
| L10H0 | label mix only | **-0.027** | +0.001 | +0.015 | **-0.043** |
| L0H1 | control | -0.001 | +0.000 | -0.000 | -0.001 |
| L4H11 | control | -0.004 | -0.003 | -0.001 | -0.001 |
| L5H1 | control | -0.001 | +0.000 | -0.000 | -0.001 |

**common bases, `sushi`, near**

| head | group | total | order | identity | form |
|---|---|---|---|---|---|
| L1H5 | sink rises | +0.010 | -0.003 | **+0.027** | -0.014 |
| L1H7 | sink rises | **+0.060** | -0.004 | **+0.034** | **+0.030** |
| L7H7 | sink rises | +0.004 | +0.001 | **-0.024** | **+0.026** |
| L9H11 | sink rises | -0.003 | -0.015 | **-0.032** | **+0.044** |
| L10H11 | sink rises | **+0.095** | **+0.028** | +0.016 | **+0.051** |
| L11H1 | sink rises | -0.005 | -0.009 | -0.015 | +0.019 |
| L11H3 | sink rises | **+0.050** | **-0.033** | **+0.040** | **+0.042** |
| L2H1 | sink rises | +0.017 | -0.010 | +0.008 | +0.018 |
| L2H6 | sink rises | **+0.023** | +0.014 | **-0.028** | **+0.037** |
| L3H6 | sink rises | **+0.121** | +0.015 | **+0.025** | **+0.080** |
| L7H11 | sink rises | **-0.026** | -0.018 | **-0.029** | **+0.022** |
| L10H4 | sink rises | +0.012 | +0.011 | **-0.026** | **+0.027** |
| L11H9 | sink rises | -0.005 | -0.017 | -0.009 | **+0.021** |
| L1H8 | sink falls | +0.009 | +0.001 | +0.006 | +0.002 |
| L2H0 | sink falls | **+0.056** | **-0.034** | **+0.032** | **+0.058** |
| L2H4 | sink falls | **-0.022** | **-0.025** | +0.019 | -0.016 |
| L3H3 | sink falls | **-0.068** | **+0.022** | **-0.067** | **-0.023** |
| L6H4 | sink falls | **+0.055** | **+0.027** | **+0.051** | **-0.022** |
| L6H8 | sink falls | **-0.156** | **-0.161** | **-0.023** | **+0.027** |
| L7H0 | sink falls | **-0.097** | **-0.103** | +0.016 | -0.011 |
| L10H9 | sink falls | **+0.048** | **-0.064** | **+0.049** | **+0.062** |
| L4H0 | sink falls | **+0.147** | **+0.101** | **+0.051** | -0.005 |
| L5H2 | sink falls | -0.001 | **-0.145** | **+0.041** | **+0.103** |
| L5H6 | sink falls | **-0.028** | **-0.046** | **+0.048** | **-0.030** |
| L8H2 | sink falls | **-0.030** | -0.018 | +0.013 | **-0.024** |
| L9H8 | sink falls | **+0.047** | -0.008 | **+0.088** | **-0.033** |
| L0H6 | label mix only | **+0.056** | +0.000 | -0.004 | **+0.059** |
| L1H0 | label mix only | **-0.064** | **-0.032** | **-0.095** | **+0.062** |
| L3H2 | label mix only | **+0.045** | **+0.024** | -0.016 | **+0.037** |
| L10H0 | label mix only | **+0.174** | **+0.036** | **+0.127** | +0.011 |
| L0H1 | control | +0.002 | +0.000 | +0.001 | +0.000 |
| L4H11 | control | -0.001 | -0.001 | -0.000 | +0.000 |
| L5H1 | control | -0.001 | +0.000 | -0.000 | -0.000 |

## 5. Label-mix shift (TV distance)

Split-half TV floor within the `none` rows: rare 0.086, common 0.088.

| contrast | base | phrase | placement | max TV | heads above floor | top |
|---|---|---|---|---|---|---|
| total | common | sushi | near | 0.514 | 42 | L2H8 (0.51), L3H6 (0.45), L8H6 (0.44), L6H8 (0.42), L6H4 (0.37), L3H2 (0.34) |
| order | common | sushi | near | 0.415 | 18 | L2H8 (0.42), L6H8 (0.36), L5H2 (0.28), L4H7 (0.25), L10H9 (0.22), L4H0 (0.22) |
| identity | common | sushi | near | 0.456 | 28 | L11H8 (0.46), L8H6 (0.37), L6H4 (0.33), L3H2 (0.26), L2H8 (0.26), L3H6 (0.25) |
| form | common | sushi | near | 0.222 | 17 | L2H9 (0.22), L2H0 (0.21), L1H0 (0.19), L0H7 (0.19), L2H4 (0.17), L4H3 (0.17) |
| placebo | common | sushi | near | 0.044 | 0 |  |
| I@29 | common | sushi | near | 0.159 | 5 | L2H0 (0.16), L3H6 (0.16), L8H6 (0.10), L2H4 (0.09), L2H8 (0.09) |
| like@30 | common | sushi | near | 0.375 | 12 | L2H0 (0.38), L2H8 (0.35), L2H5 (0.34), L1H1 (0.30), L3H8 (0.22), L2H9 (0.17) |
| sushi@31 | common | sushi | near | 0.459 | 33 | L3H6 (0.46), L11H8 (0.43), L6H4 (0.33), L0H6 (0.28), L2H8 (0.28), L2H9 (0.25) |
| like@29 | common | sushi | near | 0.246 | 4 | L2H0 (0.25), L2H8 (0.20), L2H5 (0.10), L3H8 (0.09) |
| I@30 | common | sushi | near | 0.369 | 17 | L3H6 (0.37), L2H0 (0.30), L2H8 (0.27), L8H6 (0.19), L1H1 (0.18), L0H7 (0.16) |
| total | rare | sushi | near | 0.556 | 28 | L2H8 (0.56), L2H0 (0.41), L0H7 (0.36), L1H1 (0.29), L5H3 (0.25), L2H5 (0.25) |
| order | rare | sushi | near | 0.332 | 13 | L2H8 (0.33), L3H1 (0.21), L2H0 (0.18), L2H5 (0.16), L10H8 (0.15), L6H8 (0.15) |
| identity | rare | sushi | near | 0.414 | 24 | L2H8 (0.41), L3H6 (0.34), L2H1 (0.24), L2H0 (0.24), L3H2 (0.23), L6H0 (0.22) |
| form | rare | sushi | near | 0.321 | 11 | L2H0 (0.32), L0H7 (0.26), L2H8 (0.20), L2H9 (0.16), L5H3 (0.12), L1H0 (0.11) |
| placebo | rare | sushi | near | 0.036 | 0 |  |
| I@29 | rare | sushi | near | 0.281 | 6 | L2H0 (0.28), L0H7 (0.26), L2H6 (0.15), L3H1 (0.13), L11H4 (0.12), L2H8 (0.11) |
| like@30 | rare | sushi | near | 0.655 | 14 | L2H0 (0.66), L1H1 (0.43), L2H8 (0.42), L2H5 (0.33), L0H7 (0.31), L3H8 (0.24) |
| sushi@31 | rare | sushi | near | 0.274 | 17 | L3H6 (0.27), L2H8 (0.26), L7H0 (0.20), L2H9 (0.20), L2H0 (0.19), L2H1 (0.16) |
| like@29 | rare | sushi | near | 0.411 | 8 | L2H0 (0.41), L2H8 (0.24), L2H6 (0.22), L5H3 (0.17), L0H7 (0.17), L1H1 (0.16) |
| I@30 | rare | sushi | near | 0.465 | 16 | L2H0 (0.47), L0H7 (0.41), L3H6 (0.29), L3H1 (0.28), L2H8 (0.27), L2H6 (0.16) |

## 7. Additivity: does the combination add more than its single tokens? (sink_mass)

Per row, interaction = (combo − none) − Σ(part − none); parts: `nl` = I@29 + like@30 + sushi@31; `shuffled` = like@29 + I@30 + sushi@31. Depth values average the group's heads within each row; CI = bootstrap over rows.

| base | term | depth | mean | lo | hi | CI excludes 0 |
|---|---|---|---|---|---|---|
| common | nl | early | +0.0007 | -0.0002 | +0.0015 | False |
| common | nl | late | +0.0279 | +0.0244 | +0.0313 | True |
| common | shuffled | early | +0.0017 | +0.0008 | +0.0025 | True |
| common | shuffled | late | -0.0146 | -0.0182 | -0.0112 | True |
| rare | nl | early | +0.0082 | +0.0075 | +0.0087 | True |
| rare | nl | late | +0.0132 | +0.0110 | +0.0154 | True |
| rare | shuffled | early | +0.0065 | +0.0059 | +0.0071 | True |
| rare | shuffled | late | +0.0040 | +0.0019 | +0.0062 | True |
| common | nl−shuffled | early | -0.0010 | -0.0016 | -0.0003 | True |
| common | nl−shuffled | late | +0.0425 | +0.0396 | +0.0456 | True |
| rare | nl−shuffled | early | +0.0017 | +0.0012 | +0.0022 | True |
| rare | nl−shuffled | late | +0.0091 | +0.0071 | +0.0111 | True |

Order contrast against its additive prediction, across the 144 heads:

| base | pair | slope | r | r2 | mean_abs_observed | mean_abs_interaction |
|---|---|---|---|---|---|---|
| common | nl−shuffled | -0.153 | -0.036 | 0.001 | 0.028 | 0.029 |
| rare | nl−shuffled | 0.106 | 0.081 | 0.007 | 0.012 | 0.014 |

Heads with a BH-significant interaction (q over every interaction test):

| base | combo | significant | late + | late − | largest |interaction| |
|---|---|---|---|---|---|
| common | nl | 122 | 52 | 9 | L7H0 (+0.114), L6H8 (+0.099), L10H8 (+0.090), L4H5 (+0.089), L9H3 (+0.085) |
| common | shuffled | 116 | 12 | 44 | L5H2 (-0.130), L10H9 (-0.075), L4H3 (+0.069), L6H8 (-0.066), L9H3 (+0.059) |
| rare | nl | 115 | 46 | 13 | L5H2 (+0.135), L6H0 (+0.111), L5H6 (+0.101), L4H3 (+0.087), L4H5 (+0.087) |
| rare | shuffled | 112 | 36 | 17 | L5H6 (+0.102), L6H3 (-0.054), L6H0 (+0.053), L8H5 (+0.050), L4H3 (+0.050) |

## 8. Surprisal: does sink_mass track how predictable the slot tokens are?

Slot surprisal = −log p of the tokens at positions [29, 30, 31] given everything before them (TL), summed. Slope = within-row regression across every variant (both sides demeaned per row), per nat; CI = bootstrap over rows.

| base | depth | slope | lo | hi |
|---|---|---|---|---|
| common | early | +0.00022 | +0.00012 | +0.00033 |
| common | late | -0.00139 | -0.00170 | -0.00108 |
| rare | early | +0.00025 | +0.00019 | +0.00030 |
| rare | late | -0.00107 | -0.00125 | -0.00089 |

Heads whose slope CI excludes 0, by sign:

| base | depth | slope < 0 | slope > 0 |
|---|---|---|---|
| common | early | 35 | 31 |
| common | late | 49 | 11 |
| rare | early | 33 | 30 |
| rare | late | 46 | 18 |

Per-variant means (descriptive):

| base | variant | mean_slot_nll | mean_sink_mass_early | mean_sink_mass_late | mean_query_nll |
|---|---|---|---|---|---|
| common | none | 27.031 | 0.329 | 0.594 | 8.952 |
| common | nl | 25.051 | 0.334 | 0.651 | 10.208 |
| common | shuffled | 28.497 | 0.334 | 0.611 | 9.302 |
| common | matched | 31.560 | 0.333 | 0.586 | 9.122 |
| common | matched_b | 31.501 | 0.334 | 0.585 | 9.096 |
| common | I@29 | 25.161 | 0.325 | 0.593 | 8.993 |
| common | like@30 | 26.684 | 0.332 | 0.594 | 9.016 |
| common | sushi@31 | 32.329 | 0.333 | 0.623 | 9.425 |
| common | like@29 | 26.660 | 0.329 | 0.587 | 8.945 |
| common | I@30 | 25.039 | 0.328 | 0.602 | 9.122 |
| rare | none | 37.845 | 0.333 | 0.604 | 12.443 |
| rare | nl | 23.201 | 0.326 | 0.628 | 12.907 |
| rare | shuffled | 29.792 | 0.326 | 0.623 | 12.457 |
| rare | matched | 32.306 | 0.328 | 0.585 | 12.600 |
| rare | matched_b | 32.303 | 0.328 | 0.588 | 12.588 |
| rare | I@29 | 33.580 | 0.325 | 0.597 | 12.459 |
| rare | like@30 | 34.534 | 0.327 | 0.600 | 12.644 |
| rare | sushi@31 | 37.958 | 0.332 | 0.626 | 12.185 |
| rare | like@29 | 34.633 | 0.327 | 0.602 | 12.467 |
| rare | I@30 | 33.466 | 0.326 | 0.598 | 12.656 |

## Figures

- [rare_sushi_near_sink_mass_contrasts](figures/rare_sushi_near_sink_mass_contrasts.png)
- [rare_sushi_near_sink_mass_singles](figures/rare_sushi_near_sink_mass_singles.png)
- [rare_sushi_sink_mass_tracked_forest](figures/rare_sushi_sink_mass_tracked_forest.png)
- [rare_sushi_near_slot_mass_contrasts](figures/rare_sushi_near_slot_mass_contrasts.png)
- [rare_sushi_near_slot_mass_singles](figures/rare_sushi_near_slot_mass_singles.png)
- [rare_sushi_slot_mass_tracked_forest](figures/rare_sushi_slot_mass_tracked_forest.png)
- [rare_sushi_sink_mass_layer_profile](figures/rare_sushi_sink_mass_layer_profile.png)
- [common_sushi_near_sink_mass_contrasts](figures/common_sushi_near_sink_mass_contrasts.png)
- [common_sushi_near_sink_mass_singles](figures/common_sushi_near_sink_mass_singles.png)
- [common_sushi_sink_mass_tracked_forest](figures/common_sushi_sink_mass_tracked_forest.png)
- [common_sushi_near_slot_mass_contrasts](figures/common_sushi_near_slot_mass_contrasts.png)
- [common_sushi_near_slot_mass_singles](figures/common_sushi_near_slot_mass_singles.png)
- [common_sushi_slot_mass_tracked_forest](figures/common_sushi_slot_mass_tracked_forest.png)
- [common_sushi_sink_mass_layer_profile](figures/common_sushi_sink_mass_layer_profile.png)
- [common_sushi_near_additivity](figures/common_sushi_near_additivity.png)
- [rare_sushi_near_additivity](figures/rare_sushi_near_additivity.png)
- [surprisal_variants](figures/surprisal_variants.png)
- [meaningful_counts_sink_mass](figures/meaningful_counts_sink_mass.png)
- [meaningful_counts_slot_mass](figures/meaningful_counts_slot_mass.png)

## Caveats

- One phrase is one item: effects of `nl` are effects of these three tokens at these positions, not of natural language in general, until several phrases agree.
- A slot at position p < 32 changes every later position's residual stream, so a contrast measures the slot tokens' total effect at the query (as keys and through earlier heads), not a single route.
- `matched` controls match surface form and id class, not frequency or meaning; `identity` therefore mixes everything else that distinguishes the phrase's tokens from same-form random words.
