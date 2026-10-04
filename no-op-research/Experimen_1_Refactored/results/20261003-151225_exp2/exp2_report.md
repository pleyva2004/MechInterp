# exp2: a natural-language phrase in random-token context

- Run `20261003-151225_exp2`. Bases: rare [1000, 39999], common [256, 999], 1000 rows each, BOS + 32 random ids, no repeats, no phrase ids. Metric at the final query (position 32).
- Phrases: `sushi` = ' I like sushi' [314, 588, 36324] (shuffled [588, 314, 36324]). Placements (model positions): near [29, 30, 31], far [14, 15, 16].
- Controls: `shuffled` (same ids, scrambled), `matched` / `matched_b` (per row, random ids with the same leading_space, is_alpha, is_capitalized and id class as each phrase token; pool sizes {'sushi': [43, 280, 16019]}).
- Contrasts (paired per row, a − b): total = nl − none, order = nl − shuffled, identity = shuffled − matched, form = matched − none, placebo = matched − matched_b. total = order + identity + form.
- Meaningful = BH q ≤ 0.05 (Wilcoxon signed-rank, over every test of a metric) and |Δ| ≥ the metric's threshold {'sink_mass': 0.02, 'slot_mass': 0.02, 'slot_logratio': 0.1, 'prev_mass': 0.02, 'entropy': 0.05}. Bold = meaningful. Predictions were written before any data: [PREDICTIONS.md](PREDICTIONS.md).

## Checks

| check | backend | result | detail |
|---|---|---|---|
| Attention rows sum to 1 | TL | PASS | max abs(sum − 1) = 4.6e-07 |
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
| P1 | far_is_small | FAIL | common/sushi: max |Δ| 0.123 (L8H2), median 0.0081; rare/sushi: max |Δ| 0.049 (L11H4), median 0.0046 |
| P2 | median_ratio | FAIL | common/sushi: median |Δ| 0.0212 vs 0.0081, ratio 2.6; rare/sushi: median |Δ| 0.0234 vs 0.0046, ratio 5.1 |
| P3 | heads_move | PASS | sushi: 10/20 heads with |Δ| >= 0.05 (L7H0, L5H6, L6H8, L4H0, L4H1, L5H2, L6H10, L2H4, L6H11, L8H5) |
| P4 | few_meaningful | FAIL | common/sushi/far: 14 (limit 2): L7H7, L7H11, L8H2, L9H8, L9H9, L10H1, L10H2, L10H8, L10H9, L10H10, L11H6, L11H9; common/sushi/near: 61 (limit 14): L3H8, L3H11, L4H0, L4H1, L4H5, L5H0, L5H2, L5H3, L5H4, L5H6, L5H9, L5H10; rare/sushi/far: 2 (limit 2): L8H2, L10H8; rare/sushi/near: 27 (limit 14): L2H4, L3H3, L3H9, L4H0, L4H3, L4H5, L5H2, L5H4, L5H6, L5H11, L6H5, L6H8 |
| P5 | controls | PASS | L4H11 prev_mass: min over cells 0.980 (floor 0.95); L5H1 sink_mass: min over cells 0.973 (floor 0.9); L0H1 mode Self in 20/20 cells |
| P6 | few_meaningful | PASS | common/sushi/far: 0 (limit 0); common/sushi/near: 0 (limit 0); rare/sushi/far: 0 (limit 0); rare/sushi/near: 0 (limit 0) |

## 2. Meaningful heads per contrast: sink_mass

Out of 144 heads. The placebo row is the noise floor.

| contrast | common / sushi / far | common / sushi / near | rare / sushi / far | rare / sushi / near |
|---|---|---|---|---|
| total | 37 | 73 | 25 | 80 |
| order | 14 | 61 | 2 | 27 |
| identity | 43 | 68 | 13 | 76 |
| form | 11 | 47 | 1 | 54 |
| placebo | 0 | 0 | 0 | 0 |

Effect sizes (sink_mass): 

| contrast | base | phrase | placement | median |Δ| | 90th pct |Δ| | largest |
|---|---|---|---|---|---|---|
| total | common | sushi | far | 0.0081 | 0.0491 | L8H2 (+0.123) |
| order | common | sushi | far | 0.0027 | 0.0197 | L8H2 (+0.039) |
| identity | common | sushi | far | 0.0096 | 0.0414 | L8H2 (+0.089) |
| form | common | sushi | far | 0.0041 | 0.0151 | L9H11 (-0.039) |
| placebo | common | sushi | far | 0.0006 | 0.0029 | L10H11 (+0.006) |
| total | common | sushi | near | 0.0212 | 0.1164 | L7H0 (+0.246) |
| order | common | sushi | near | 0.0128 | 0.0725 | L6H8 (+0.163) |
| identity | common | sushi | near | 0.0180 | 0.0637 | L6H0 (+0.153) |
| form | common | sushi | near | 0.0093 | 0.0403 | L4H3 (+0.104) |
| placebo | common | sushi | near | 0.0019 | 0.0064 | L4H8 (-0.014) |
| total | rare | sushi | far | 0.0046 | 0.0262 | L11H4 (-0.049) |
| order | rare | sushi | far | 0.0022 | 0.0100 | L10H8 (+0.042) |
| identity | rare | sushi | far | 0.0041 | 0.0191 | L9H8 (+0.035) |
| form | rare | sushi | far | 0.0028 | 0.0105 | L11H4 (-0.045) |
| placebo | rare | sushi | far | 0.0003 | 0.0021 | L11H7 (-0.004) |
| total | rare | sushi | near | 0.0234 | 0.0783 | L7H0 (+0.233) |
| order | rare | sushi | near | 0.0069 | 0.0304 | L5H2 (+0.093) |
| identity | rare | sushi | near | 0.0215 | 0.0759 | L6H0 (+0.232) |
| form | rare | sushi | near | 0.0129 | 0.0449 | L5H2 (-0.128) |
| placebo | rare | sushi | near | 0.0018 | 0.0054 | L8H7 (-0.015) |

## 2. Meaningful heads per contrast: slot_mass

Out of 144 heads. The placebo row is the noise floor.

| contrast | common / sushi / far | common / sushi / near | rare / sushi / far | rare / sushi / near |
|---|---|---|---|---|
| total | 32 | 88 | 20 | 73 |
| order | 1 | 44 | 3 | 40 |
| identity | 19 | 59 | 7 | 46 |
| form | 7 | 72 | 15 | 58 |
| placebo | 0 | 0 | 0 | 0 |

Effect sizes (slot_mass): 

| contrast | base | phrase | placement | median |Δ| | 90th pct |Δ| | largest |
|---|---|---|---|---|---|---|
| total | common | sushi | far | 0.0062 | 0.0287 | L10H0 (+0.097) |
| order | common | sushi | far | 0.0019 | 0.0126 | L10H8 (-0.035) |
| identity | common | sushi | far | 0.0039 | 0.0223 | L10H0 (+0.073) |
| form | common | sushi | far | 0.0037 | 0.0166 | L0H6 (+0.032) |
| placebo | common | sushi | far | 0.0003 | 0.0019 | L5H5 (-0.005) |
| total | common | sushi | near | 0.0272 | 0.0796 | L5H3 (+0.186) |
| order | common | sushi | near | 0.0104 | 0.0379 | L6H8 (-0.161) |
| identity | common | sushi | near | 0.0158 | 0.0527 | L6H0 (-0.134) |
| form | common | sushi | near | 0.0199 | 0.0494 | L2H3 (+0.139) |
| placebo | common | sushi | near | 0.0013 | 0.0061 | L4H0 (-0.015) |
| total | rare | sushi | far | 0.0030 | 0.0274 | L11H4 (+0.114) |
| order | rare | sushi | far | 0.0009 | 0.0084 | L10H8 (-0.067) |
| identity | rare | sushi | far | 0.0024 | 0.0134 | L10H8 (+0.079) |
| form | rare | sushi | far | 0.0026 | 0.0198 | L11H4 (+0.070) |
| placebo | rare | sushi | far | 0.0001 | 0.0010 | L11H7 (+0.004) |
| total | rare | sushi | near | 0.0202 | 0.0669 | L7H0 (-0.201) |
| order | rare | sushi | near | 0.0082 | 0.0378 | L4H0 (+0.107) |
| identity | rare | sushi | near | 0.0133 | 0.0559 | L6H0 (-0.224) |
| form | rare | sushi | near | 0.0151 | 0.0483 | L5H2 (+0.117) |
| placebo | rare | sushi | near | 0.0014 | 0.0048 | L8H7 (+0.013) |

## 2. Meaningful heads per contrast: slot_logratio

Out of 144 heads. The placebo row is the noise floor.

| contrast | common / sushi / far | common / sushi / near | rare / sushi / far | rare / sushi / near |
|---|---|---|---|---|
| total | 133 | 121 | 118 | 118 |
| order | 91 | 79 | 88 | 80 |
| identity | 110 | 110 | 119 | 105 |
| form | 122 | 114 | 111 | 108 |
| placebo | 0 | 0 | 0 | 0 |

Effect sizes (slot_logratio): 

| contrast | base | phrase | placement | median |Δ| | 90th pct |Δ| | largest |
|---|---|---|---|---|---|---|
| total | common | sushi | far | 0.3831 | 0.8623 | L3H0 (+1.929) |
| order | common | sushi | far | 0.1339 | 0.4016 | L5H7 (+0.893) |
| identity | common | sushi | far | 0.2344 | 0.5434 | L3H0 (+1.417) |
| form | common | sushi | far | 0.2865 | 0.5680 | L5H2 (+0.964) |
| placebo | common | sushi | far | 0.0161 | 0.0410 | L0H3 (-0.095) |
| total | common | sushi | near | 0.3718 | 0.9339 | L4H11 (-6.372) |
| order | common | sushi | near | 0.1195 | 0.5238 | L4H11 (-2.390) |
| identity | common | sushi | near | 0.2285 | 0.6301 | L4H11 (-1.643) |
| form | common | sushi | near | 0.2208 | 0.5890 | L4H11 (-2.339) |
| placebo | common | sushi | near | 0.0179 | 0.0566 | L4H11 (+0.165) |
| total | rare | sushi | far | 0.2495 | 0.7620 | L10H1 (-1.603) |
| order | rare | sushi | far | 0.1504 | 0.5312 | L5H7 (+1.220) |
| identity | rare | sushi | far | 0.2579 | 0.5787 | L3H0 (+1.131) |
| form | rare | sushi | far | 0.2367 | 0.7292 | L5H1 (-1.402) |
| placebo | rare | sushi | far | 0.0121 | 0.0405 | L0H5 (+0.115) |
| total | rare | sushi | near | 0.3339 | 0.8927 | L4H11 (-4.800) |
| order | rare | sushi | near | 0.1373 | 0.4336 | L4H11 (-2.317) |
| identity | rare | sushi | near | 0.2235 | 0.7811 | L6H0 (-1.657) |
| form | rare | sushi | near | 0.2407 | 0.6760 | L5H1 (-1.414) |
| placebo | rare | sushi | near | 0.0184 | 0.0486 | L4H11 (+0.190) |

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

**rare bases, `sushi`, far**

| head | group | total | order | identity | form |
|---|---|---|---|---|---|
| L1H5 | sink rises | +0.001 | +0.002 | -0.002 | +0.002 |
| L1H7 | sink rises | +0.001 | +0.001 | +0.001 | -0.001 |
| L7H7 | sink rises | +0.006 | +0.001 | -0.002 | +0.006 |
| L9H11 | sink rises | **+0.020** | +0.007 | +0.018 | -0.005 |
| L10H11 | sink rises | **+0.024** | +0.002 | **+0.021** | +0.001 |
| L11H1 | sink rises | **+0.025** | +0.001 | **+0.031** | -0.007 |
| L11H3 | sink rises | +0.009 | +0.002 | +0.014 | -0.007 |
| L2H1 | sink rises | **+0.027** | +0.004 | **+0.028** | -0.006 |
| L2H6 | sink rises | +0.009 | +0.000 | +0.004 | +0.005 |
| L3H6 | sink rises | +0.001 | +0.000 | +0.002 | -0.001 |
| L7H11 | sink rises | **+0.027** | +0.011 | +0.014 | +0.002 |
| L10H4 | sink rises | **+0.021** | +0.004 | +0.018 | -0.001 |
| L11H9 | sink rises | **+0.030** | +0.005 | **+0.024** | +0.001 |
| L1H8 | sink falls | -0.006 | +0.000 | -0.005 | -0.001 |
| L2H0 | sink falls | +0.001 | +0.000 | +0.000 | -0.000 |
| L2H4 | sink falls | -0.007 | +0.000 | -0.005 | -0.003 |
| L3H3 | sink falls | +0.005 | -0.000 | +0.008 | -0.002 |
| L6H4 | sink falls | +0.009 | +0.002 | +0.011 | -0.004 |
| L6H8 | sink falls | +0.008 | -0.002 | +0.019 | -0.010 |
| L7H0 | sink falls | -0.009 | -0.016 | +0.011 | -0.005 |
| L10H9 | sink falls | +0.005 | +0.009 | +0.008 | -0.013 |
| L4H0 | sink falls | -0.007 | +0.003 | -0.005 | -0.004 |
| L5H2 | sink falls | -0.004 | +0.005 | +0.002 | -0.011 |
| L5H6 | sink falls | -0.010 | +0.003 | -0.007 | -0.006 |
| L8H2 | sink falls | **+0.028** | **+0.027** | +0.009 | -0.009 |
| L9H8 | sink falls | **+0.035** | +0.006 | **+0.035** | -0.006 |
| L0H6 | label mix only | +0.002 | -0.000 | +0.000 | +0.002 |
| L1H0 | label mix only | -0.000 | +0.000 | -0.000 | +0.000 |
| L3H2 | label mix only | -0.001 | +0.000 | +0.002 | -0.003 |
| L10H0 | label mix only | **+0.028** | +0.003 | **+0.022** | +0.002 |
| L0H1 | control | +0.000 | +0.000 | -0.000 | +0.000 |
| L4H11 | control | -0.000 | -0.000 | -0.000 | -0.000 |
| L5H1 | control | -0.002 | -0.000 | -0.002 | +0.000 |

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

**common bases, `sushi`, far**

| head | group | total | order | identity | form |
|---|---|---|---|---|---|
| L1H5 | sink rises | -0.008 | +0.001 | -0.010 | +0.001 |
| L1H7 | sink rises | -0.007 | +0.001 | -0.003 | -0.005 |
| L7H7 | sink rises | **-0.029** | **-0.033** | **+0.032** | **-0.028** |
| L9H11 | sink rises | -0.004 | -0.003 | **+0.038** | **-0.039** |
| L10H11 | sink rises | **-0.050** | -0.007 | **-0.023** | **-0.020** |
| L11H1 | sink rises | **+0.053** | +0.017 | **+0.059** | **-0.022** |
| L11H3 | sink rises | **-0.027** | +0.007 | -0.019 | -0.015 |
| L2H1 | sink rises | **+0.049** | +0.004 | **+0.039** | +0.006 |
| L2H6 | sink rises | +0.002 | -0.000 | +0.001 | +0.001 |
| L3H6 | sink rises | -0.003 | -0.000 | -0.000 | -0.003 |
| L7H11 | sink rises | **+0.064** | **+0.024** | **+0.063** | **-0.022** |
| L10H4 | sink rises | +0.010 | -0.002 | **+0.034** | **-0.021** |
| L11H9 | sink rises | **+0.038** | **+0.021** | **+0.035** | -0.018 |
| L1H8 | sink falls | -0.006 | +0.000 | -0.005 | -0.001 |
| L2H0 | sink falls | -0.001 | +0.000 | -0.000 | -0.001 |
| L2H4 | sink falls | -0.007 | +0.001 | -0.003 | -0.005 |
| L3H3 | sink falls | +0.007 | +0.000 | +0.007 | -0.000 |
| L6H4 | sink falls | **+0.059** | +0.004 | **+0.053** | +0.002 |
| L6H8 | sink falls | **+0.028** | -0.001 | **+0.037** | -0.008 |
| L7H0 | sink falls | +0.005 | -0.012 | +0.008 | +0.009 |
| L10H9 | sink falls | **+0.075** | **+0.033** | **+0.047** | -0.004 |
| L4H0 | sink falls | -0.017 | +0.001 | -0.011 | -0.007 |
| L5H2 | sink falls | **+0.026** | +0.006 | **+0.025** | -0.006 |
| L5H6 | sink falls | -0.004 | +0.002 | -0.006 | +0.001 |
| L8H2 | sink falls | **+0.123** | **+0.039** | **+0.089** | -0.004 |
| L9H8 | sink falls | -0.013 | **+0.022** | **-0.045** | +0.010 |
| L0H6 | label mix only | -0.003 | -0.000 | -0.001 | -0.002 |
| L1H0 | label mix only | -0.002 | +0.000 | -0.001 | -0.002 |
| L3H2 | label mix only | +0.009 | -0.000 | +0.010 | -0.000 |
| L10H0 | label mix only | **-0.056** | -0.005 | **-0.047** | -0.005 |
| L0H1 | control | -0.000 | +0.000 | -0.000 | -0.000 |
| L4H11 | control | +0.000 | -0.000 | +0.000 | -0.000 |
| L5H1 | control | -0.000 | -0.000 | +0.001 | -0.001 |

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
| total | common | sushi | far | 0.189 | 6 | L6H4 (0.19), L8H2 (0.18), L11H6 (0.10), L10H9 (0.10), L8H6 (0.09), L2H1 (0.09) |
| order | common | sushi | far | 0.050 | 0 |  |
| identity | common | sushi | far | 0.168 | 3 | L6H4 (0.17), L8H2 (0.15), L8H6 (0.11) |
| form | common | sushi | far | 0.064 | 0 |  |
| placebo | common | sushi | far | 0.023 | 0 |  |
| total | common | sushi | near | 0.514 | 42 | L2H8 (0.51), L3H6 (0.45), L8H6 (0.44), L6H8 (0.42), L6H4 (0.37), L3H2 (0.34) |
| order | common | sushi | near | 0.415 | 18 | L2H8 (0.42), L6H8 (0.36), L5H2 (0.28), L4H7 (0.25), L10H9 (0.22), L4H0 (0.22) |
| identity | common | sushi | near | 0.456 | 28 | L11H8 (0.46), L8H6 (0.37), L6H4 (0.33), L3H2 (0.26), L2H8 (0.26), L3H6 (0.25) |
| form | common | sushi | near | 0.222 | 17 | L2H9 (0.22), L2H0 (0.21), L1H0 (0.19), L0H7 (0.19), L2H4 (0.17), L4H3 (0.17) |
| placebo | common | sushi | near | 0.044 | 0 |  |
| total | rare | sushi | far | 0.088 | 1 | L2H1 (0.09) |
| order | rare | sushi | far | 0.050 | 0 |  |
| identity | rare | sushi | far | 0.092 | 1 | L2H1 (0.09) |
| form | rare | sushi | far | 0.036 | 0 |  |
| placebo | rare | sushi | far | 0.018 | 0 |  |
| total | rare | sushi | near | 0.556 | 28 | L2H8 (0.56), L2H0 (0.41), L0H7 (0.36), L1H1 (0.29), L5H3 (0.25), L2H5 (0.25) |
| order | rare | sushi | near | 0.332 | 13 | L2H8 (0.33), L3H1 (0.21), L2H0 (0.18), L2H5 (0.16), L10H8 (0.15), L6H8 (0.15) |
| identity | rare | sushi | near | 0.414 | 24 | L2H8 (0.41), L3H6 (0.34), L2H1 (0.24), L2H0 (0.24), L3H2 (0.23), L6H0 (0.22) |
| form | rare | sushi | near | 0.321 | 11 | L2H0 (0.32), L0H7 (0.26), L2H8 (0.20), L2H9 (0.16), L5H3 (0.12), L1H0 (0.11) |
| placebo | rare | sushi | near | 0.036 | 0 |  |

## Figures

- [rare_sushi_near_sink_mass_contrasts](figures/rare_sushi_near_sink_mass_contrasts.png)
- [rare_sushi_far_sink_mass_contrasts](figures/rare_sushi_far_sink_mass_contrasts.png)
- [rare_sushi_sink_mass_tracked_forest](figures/rare_sushi_sink_mass_tracked_forest.png)
- [rare_sushi_near_slot_mass_contrasts](figures/rare_sushi_near_slot_mass_contrasts.png)
- [rare_sushi_far_slot_mass_contrasts](figures/rare_sushi_far_slot_mass_contrasts.png)
- [rare_sushi_slot_mass_tracked_forest](figures/rare_sushi_slot_mass_tracked_forest.png)
- [rare_sushi_sink_mass_layer_profile](figures/rare_sushi_sink_mass_layer_profile.png)
- [common_sushi_near_sink_mass_contrasts](figures/common_sushi_near_sink_mass_contrasts.png)
- [common_sushi_far_sink_mass_contrasts](figures/common_sushi_far_sink_mass_contrasts.png)
- [common_sushi_sink_mass_tracked_forest](figures/common_sushi_sink_mass_tracked_forest.png)
- [common_sushi_near_slot_mass_contrasts](figures/common_sushi_near_slot_mass_contrasts.png)
- [common_sushi_far_slot_mass_contrasts](figures/common_sushi_far_slot_mass_contrasts.png)
- [common_sushi_slot_mass_tracked_forest](figures/common_sushi_slot_mass_tracked_forest.png)
- [common_sushi_sink_mass_layer_profile](figures/common_sushi_sink_mass_layer_profile.png)
- [meaningful_counts_sink_mass](figures/meaningful_counts_sink_mass.png)
- [meaningful_counts_slot_mass](figures/meaningful_counts_slot_mass.png)

## Caveats

- One phrase is one item: effects of `nl` are effects of these three tokens at these positions, not of natural language in general, until several phrases agree.
- A slot at position p < 32 changes every later position's residual stream, so a contrast measures the slot tokens' total effect at the query (as keys and through earlier heads), not a single route.
- `matched` controls match surface form and id class, not frequency or meaning; `identity` therefore mixes everything else that distinguishes the phrase's tokens from same-form random words.
