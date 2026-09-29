# exp1ext: which tokens cause the rare-vs-common head changes

- Run `20260928-135243_exp1ext`. Rare = ids [1000, 39999], common = ids [256, 999] (byte tokens enter only as swept tokens). Input = BOS + 32 random ids, no repeats within a sequence; metric at the final query (position 32).
- Predictions were written before any data: [PREDICTIONS.md](PREDICTIONS.md). Bold = 95% bootstrap CI excludes 0.
- Stage 1: 1000 sequences per 2×2 cell, dose-response 500 per k. Stage 2: 100 base sequences per class; final-token sweep over ids 0–999 + 2000 rare ids; substitutions at positions [31, 16].

## Checks

| check | backend | result | detail |
|---|---|---|---|
| Attention rows sum to 1 | TL | PASS | max abs(sum − 1) = 5.0e-07 |
| Attention rows sum to 1 | HF | PASS | max abs(sum − 1) = 4.4e-07 |
| Determinism (first 50 re-extracted) | TL | PASS | bit-identical |
| Determinism (first 50 re-extracted) | HF | PASS | bit-identical |
| Cross-library, stage 1 (all cells) | TL vs HF | PASS | max |Δ| probability metrics 4.0e-05; labels agree 100.00% |
| Cross-library, stage 2 subset | TL vs HF | PASS | 2 bases per class × all tokens × all positions; max |Δ| 5.2e-05 |
| Replicates exp1 (AA vs A_rare) | TL | PASS | per-head mean sink_mass, r = 0.9996 across all heads (threshold 0.99) |
| Replicates exp1 (CC vs C_common_clean) | TL | PASS | per-head mean sink_mass, r = 0.9998 across all heads (threshold 0.99) |

## 1. Tracked heads

Selected from exp1 by label-mix TV ≥ 0.2 (A_rare vs each common condition); group = direction of the sink_mass change from rare to common. Controls were fixed in advance.

| name | group | both | tv_B_common | tv_C_common_clean | sink_change |
|---|---|---|---|---|---|
| L1H5 | sink rises | True | 0.273 | 0.372 | 0.051 |
| L1H7 | sink rises | True | 0.244 | 0.471 | 0.072 |
| L7H7 | sink rises | True | 0.513 | 0.505 | 0.307 |
| L9H11 | sink rises | True | 0.316 | 0.302 | 0.132 |
| L10H11 | sink rises | True | 0.439 | 0.523 | 0.252 |
| L11H1 | sink rises | True | 0.265 | 0.257 | 0.144 |
| L11H3 | sink rises | True | 0.285 | 0.250 | 0.124 |
| L2H1 | sink rises | False | 0.243 | 0.194 | 0.095 |
| L2H6 | sink rises | False | 0.099 | 0.209 | 0.050 |
| L3H6 | sink rises | False | 0.149 | 0.243 | 0.043 |
| L7H11 | sink rises | False | 0.214 | 0.190 | 0.081 |
| L10H4 | sink rises | False | 0.186 | 0.334 | 0.095 |
| L11H9 | sink rises | False | 0.176 | 0.360 | 0.069 |
| L1H8 | sink falls | True | 0.335 | 0.208 | -0.033 |
| L2H0 | sink falls | True | 0.473 | 0.485 | -0.107 |
| L2H4 | sink falls | True | 0.418 | 0.450 | -0.347 |
| L3H3 | sink falls | True | 0.365 | 0.282 | -0.232 |
| L6H4 | sink falls | True | 0.219 | 0.264 | -0.070 |
| L6H8 | sink falls | True | 0.397 | 0.289 | -0.206 |
| L7H0 | sink falls | True | 0.324 | 0.350 | -0.253 |
| L10H9 | sink falls | True | 0.219 | 0.207 | -0.346 |
| L4H0 | sink falls | False | 0.124 | 0.209 | -0.227 |
| L5H2 | sink falls | False | 0.196 | 0.237 | -0.322 |
| L5H6 | sink falls | False | 0.211 | 0.181 | -0.101 |
| L8H2 | sink falls | False | 0.054 | 0.252 | -0.105 |
| L9H8 | sink falls | False | 0.231 | 0.090 | -0.183 |
| L0H6 | label mix only | False | 0.341 | 0.054 | 0.000 |
| L1H0 | label mix only | False | 0.110 | 0.201 | 0.011 |
| L3H2 | label mix only | False | 0.157 | 0.231 | 0.014 |
| L10H0 | label mix only | False | 0.226 | 0.022 | -0.003 |
| L0H1 | control | False | 0.028 | 0.000 | 0.000 |
| L4H11 | control | False | 0.009 | 0.005 | 0.000 |
| L5H1 | control | False | 0.014 | 0.000 | 0.001 |

## 2. 2×2: is the change caused by the final token or by the context?

sink_mass per cell (first letter = context, second = final token; A = rare, C = common). query = change only the final token; context = change only the 31 tokens before it. driver = the component whose CI excludes 0 (both if they are within 2× of each other).

| head | group | AA | AC | CA | CC | total CC−AA | query AC−AA | context CA−AA | interaction | driver | TV AA→CC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| L1H5 | sink rises | 0.166 | 0.291 | 0.214 | 0.226 | **+0.060** [+0.055, +0.065] | **+0.124** [+0.118, +0.131] | **+0.047** [+0.044, +0.051] | **-0.111** [-0.117, -0.106] | query | 0.37 |
| L1H7 | sink rises | 0.117 | 0.187 | 0.162 | 0.199 | **+0.083** [+0.078, +0.087] | **+0.071** [+0.065, +0.075] | **+0.045** [+0.043, +0.047] | **-0.033** [-0.035, -0.031] | both | 0.41 |
| L7H7 | sink rises | 0.228 | 0.334 | 0.543 | 0.535 | **+0.307** [+0.290, +0.325] | **+0.106** [+0.091, +0.121] | **+0.315** [+0.300, +0.329] | **-0.113** [-0.131, -0.096] | context | 0.50 |
| L9H11 | sink rises | 0.272 | 0.376 | 0.420 | 0.404 | **+0.132** [+0.115, +0.150] | **+0.105** [+0.084, +0.124] | **+0.148** [+0.134, +0.162] | **-0.120** [-0.137, -0.102] | both | 0.36 |
| L10H11 | sink rises | 0.235 | 0.381 | 0.466 | 0.543 | **+0.309** [+0.288, +0.329] | **+0.146** [+0.125, +0.167] | **+0.231** [+0.218, +0.244] | **-0.069** [-0.086, -0.052] | both | 0.57 |
| L11H1 | sink rises | 0.383 | 0.565 | 0.469 | 0.540 | **+0.157** [+0.135, +0.180] | **+0.182** [+0.158, +0.206] | **+0.086** [+0.073, +0.098] | **-0.111** [-0.127, -0.094] | query | 0.31 |
| L11H3 | sink rises | 0.338 | 0.435 | 0.389 | 0.462 | **+0.124** [+0.104, +0.145] | **+0.098** [+0.076, +0.118] | **+0.051** [+0.041, +0.061] | **-0.025** [-0.038, -0.010] | both | 0.27 |
| L2H1 | sink rises | 0.225 | 0.344 | 0.170 | 0.298 | **+0.073** [+0.057, +0.087] | **+0.119** [+0.102, +0.134] | **-0.055** [-0.062, -0.049] | **+0.009** [+0.001, +0.018] | query | 0.17 |
| L2H6 | sink rises | 0.211 | 0.246 | 0.270 | 0.293 | **+0.082** [+0.074, +0.090] | **+0.035** [+0.030, +0.039] | **+0.059** [+0.053, +0.066] | **-0.012** [-0.015, -0.009] | both | 0.24 |
| L3H6 | sink rises | 0.070 | 0.069 | 0.125 | 0.136 | **+0.066** [+0.060, +0.072] | -0.001 [-0.005, +0.003] | **+0.055** [+0.050, +0.061] | **+0.011** [+0.005, +0.018] | context | 0.20 |
| L7H11 | sink rises | 0.348 | 0.528 | 0.469 | 0.441 | **+0.093** [+0.074, +0.112] | **+0.180** [+0.159, +0.201] | **+0.121** [+0.105, +0.135] | **-0.207** [-0.225, -0.189] | both | 0.26 |
| L10H4 | sink rises | 0.304 | 0.440 | 0.432 | 0.466 | **+0.161** [+0.140, +0.183] | **+0.136** [+0.113, +0.158] | **+0.128** [+0.117, +0.141] | **-0.103** [-0.120, -0.085] | both | 0.36 |
| L11H9 | sink rises | 0.322 | 0.487 | 0.453 | 0.486 | **+0.164** [+0.145, +0.183] | **+0.166** [+0.144, +0.188] | **+0.131** [+0.118, +0.144] | **-0.133** [-0.151, -0.115] | both | 0.38 |
| L1H8 | sink falls | 0.240 | 0.217 | 0.233 | 0.215 | **-0.025** [-0.028, -0.023] | **-0.024** [-0.026, -0.021] | **-0.007** [-0.009, -0.006] | **+0.005** [+0.004, +0.007] | query | 0.16 |
| L2H0 | sink falls | 0.249 | 0.212 | 0.196 | 0.149 | **-0.100** [-0.105, -0.095] | **-0.037** [-0.041, -0.032] | **-0.053** [-0.057, -0.048] | **-0.011** [-0.015, -0.007] | both | 0.43 |
| L2H4 | sink falls | 0.704 | 0.540 | 0.602 | 0.378 | **-0.326** [-0.344, -0.307] | **-0.165** [-0.181, -0.147] | **-0.102** [-0.113, -0.091] | **-0.059** [-0.073, -0.046] | both | 0.42 |
| L3H3 | sink falls | 0.572 | 0.382 | 0.598 | 0.406 | **-0.166** [-0.190, -0.140] | **-0.191** [-0.212, -0.166] | **+0.025** [+0.013, +0.038] | -0.001 [-0.016, +0.014] | query | 0.26 |
| L6H4 | sink falls | 0.333 | 0.296 | 0.292 | 0.240 | **-0.093** [-0.105, -0.081] | **-0.037** [-0.050, -0.025] | **-0.040** [-0.049, -0.032] | **-0.015** [-0.026, -0.004] | both | 0.30 |
| L6H8 | sink falls | 0.510 | 0.500 | 0.406 | 0.349 | **-0.161** [-0.179, -0.142] | -0.010 [-0.026, +0.005] | **-0.105** [-0.117, -0.090] | **-0.047** [-0.062, -0.031] | context | 0.24 |
| L7H0 | sink falls | 0.578 | 0.533 | 0.315 | 0.320 | **-0.258** [-0.275, -0.240] | **-0.045** [-0.061, -0.029] | **-0.263** [-0.280, -0.247] | **+0.050** [+0.032, +0.068] | context | 0.35 |
| L10H9 | sink falls | 0.816 | 0.783 | 0.506 | 0.470 | **-0.346** [-0.362, -0.329] | **-0.033** [-0.043, -0.023] | **-0.311** [-0.322, -0.299] | -0.002 [-0.018, +0.012] | context | 0.18 |
| L4H0 | sink falls | 0.790 | 0.640 | 0.626 | 0.552 | **-0.238** [-0.255, -0.218] | **-0.150** [-0.166, -0.133] | **-0.164** [-0.178, -0.149] | **+0.076** [+0.060, +0.093] | both | 0.19 |
| L5H2 | sink falls | 0.800 | 0.814 | 0.432 | 0.475 | **-0.325** [-0.343, -0.306] | **+0.014** [+0.005, +0.023] | **-0.367** [-0.382, -0.352] | **+0.028** [+0.014, +0.042] | context | 0.23 |
| L5H6 | sink falls | 0.569 | 0.546 | 0.452 | 0.471 | **-0.098** [-0.117, -0.080] | **-0.023** [-0.040, -0.007] | **-0.117** [-0.134, -0.101] | **+0.042** [+0.023, +0.063] | context | 0.19 |
| L8H2 | sink falls | 0.494 | 0.546 | 0.374 | 0.330 | **-0.164** [-0.181, -0.146] | **+0.052** [+0.035, +0.068] | **-0.120** [-0.133, -0.108] | **-0.096** [-0.112, -0.080] | context | 0.25 |
| L9H8 | sink falls | 0.654 | 0.633 | 0.562 | 0.524 | **-0.130** [-0.147, -0.113] | **-0.021** [-0.034, -0.008] | **-0.092** [-0.105, -0.078] | **-0.017** [-0.031, -0.004] | context | 0.08 |
| L0H6 | label mix only | 0.064 | 0.028 | 0.129 | 0.063 | -0.001 [-0.004, +0.002] | **-0.036** [-0.038, -0.034] | **+0.065** [+0.063, +0.067] | **-0.030** [-0.032, -0.028] | both | 0.04 |
| L1H0 | label mix only | 0.016 | 0.014 | 0.038 | 0.031 | **+0.015** [+0.012, +0.017] | **-0.002** [-0.003, -0.000] | **+0.022** [+0.020, +0.023] | **-0.005** [-0.007, -0.003] | context | 0.20 |
| L3H2 | label mix only | 0.088 | 0.091 | 0.105 | 0.133 | **+0.045** [+0.037, +0.053] | +0.003 [-0.002, +0.009] | **+0.017** [+0.011, +0.024] | **+0.024** [+0.017, +0.032] | context | 0.20 |
| L10H0 | label mix only | 0.537 | 0.601 | 0.665 | 0.672 | **+0.134** [+0.121, +0.149] | **+0.063** [+0.048, +0.078] | **+0.127** [+0.117, +0.138] | **-0.056** [-0.071, -0.043] | context | 0.02 |
| L0H1 | control | 0.000 | 0.000 | 0.000 | 0.000 | **+0.000** [+0.000, +0.000] | **+0.000** [+0.000, +0.000] | **+0.000** [+0.000, +0.000] | **-0.000** [-0.000, -0.000] | query | 0.00 |
| L4H11 | control | 0.000 | 0.000 | 0.000 | 0.000 | **-0.000** [-0.000, -0.000] | +0.000 [-0.000, +0.000] | **-0.000** [-0.000, -0.000] | -0.000 [-0.000, +0.000] | context | 0.00 |
| L5H1 | control | 0.973 | 0.987 | 0.987 | 0.990 | **+0.017** [+0.014, +0.020] | **+0.014** [+0.011, +0.016] | **+0.013** [+0.011, +0.016] | **-0.010** [-0.013, -0.007] | both | 0.00 |

Driver counts by group (tracked heads):

| group | both | context | query |
|---|---|---|---|
| label mix only | 1 | 3 | 0 |
| sink falls | 4 | 7 | 2 |
| sink rises | 8 | 2 | 3 |

Replication on fresh sequences: 13/15 of the heads selected in both exp1 comparisons again have TV(AA, CC) ≥ 0.2 with the same sink direction. Split-half TV noise floor within a pure cell: max 0.080 over all heads.

## 3. Dose-response: how many common context tokens does it take?

sink_mass as k of the 31 context tokens of a rare sequence become common (final token stays rare; k=31 equals the CA cell). 'k reaching half' = smallest k with at least half of the full change; a small value means a few tokens suffice, a value near the end means the aggregate context matters.

| head | group | k=0 | k=31 | change | k reaching half | k=1 share |
|---|---|---|---|---|---|---|
| L1H5 | sink rises | 0.165 | 0.211 | +0.046 | 31 | -0.00 |
| L1H7 | sink rises | 0.117 | 0.163 | +0.046 | 31 | -0.02 |
| L7H7 | sink rises | 0.227 | 0.539 | +0.313 | 31 | 0.00 |
| L9H11 | sink rises | 0.259 | 0.403 | +0.144 | 31 | 0.01 |
| L10H11 | sink rises | 0.233 | 0.449 | +0.215 | 31 | 0.01 |
| L11H1 | sink rises | 0.371 | 0.454 | +0.084 | 31 | 0.03 |
| L11H3 | sink rises | 0.330 | 0.371 | +0.041 | 31 | -0.03 |
| L2H1 | sink rises | 0.220 | 0.166 | -0.054 | 8 | 0.08 |
| L2H6 | sink rises | 0.217 | 0.270 | +0.053 | 16 | 0.04 |
| L3H6 | sink rises | 0.069 | 0.124 | +0.055 | 31 | 0.02 |
| L7H11 | sink rises | 0.342 | 0.463 | +0.121 | 31 | 0.00 |
| L10H4 | sink rises | 0.298 | 0.422 | +0.124 | 31 | 0.02 |
| L11H9 | sink rises | 0.320 | 0.452 | +0.132 | 31 | 0.02 |
| L1H8 | sink falls | 0.242 | 0.233 | -0.009 | 31 | 0.04 |
| L2H0 | sink falls | 0.251 | 0.196 | -0.055 | 16 | 0.04 |
| L2H4 | sink falls | 0.709 | 0.605 | -0.104 | 16 | 0.02 |
| L3H3 | sink falls | 0.566 | 0.596 | +0.030 | 8 | 0.11 |
| L6H4 | sink falls | 0.323 | 0.277 | -0.047 | 31 | 0.06 |
| L6H8 | sink falls | 0.509 | 0.404 | -0.105 | 16 | 0.01 |
| L7H0 | sink falls | 0.569 | 0.304 | -0.265 | 31 | 0.01 |
| L10H9 | sink falls | 0.819 | 0.517 | -0.302 | 31 | 0.02 |
| L4H0 | sink falls | 0.792 | 0.633 | -0.159 | 16 | 0.03 |
| L5H2 | sink falls | 0.795 | 0.436 | -0.359 | 16 | 0.02 |
| L5H6 | sink falls | 0.574 | 0.466 | -0.108 | 16 | 0.01 |
| L8H2 | sink falls | 0.489 | 0.360 | -0.129 | 31 | -0.00 |
| L9H8 | sink falls | 0.641 | 0.548 | -0.093 | 31 | 0.03 |
| L0H6 | label mix only | 0.064 | 0.129 | +0.065 | 31 | 0.01 |
| L1H0 | label mix only | 0.017 | 0.040 | +0.023 | 31 | 0.01 |
| L3H2 | label mix only | 0.085 | 0.103 | +0.018 | 8 | 0.05 |
| L10H0 | label mix only | 0.528 | 0.667 | +0.139 | 31 | 0.02 |
| L0H1 | control | 0.000 | 0.000 | +0.000 | 31 | 0.01 |
| L4H11 | control | 0.000 | 0.000 | -0.000 | 1 | 0.51 |
| L5H1 | control | 0.974 | 0.988 | +0.014 | 16 | 0.00 |

## 4. Final-token sweep: which final tokens move each head?

Per token: mean sink_mass with that token in the final slot, averaged over bases. reliability = correlation of per-token means between two random halves of the bases (only read rankings where it is high); held-out = the top-minus-bottom gap of the top/bottom 15 tokens chosen on one half, measured on the other half (shrinkage = winner's curse).

| head | group | bases | reliability | held-out gap | sd over tokens | highest 5 | lowest 5 |
|---|---|---|---|---|---|---|---|
| L1H5 | sink rises | rare | 1.00 | 0.413 / 0.407 | 0.088 | ` '`(705), `h`(71), `n`(77), `ud`(463), `ill`(359) | ` safeguards`(32673), ` embry`(20748), `cffff`(31727), `\x03`(191), ` hostility`(23594) |
| L1H5 | sink rises | common | 0.99 | 0.305 / 0.313 | 0.049 | ` TED`(38436), `&`(5), `Trivia`(23854), ` '`(705), ` Olympic`(11514) | `�`(11976), `cffff`(31727), ` embry`(20748), `と`(30201), ` angle`(9848) |
| L1H7 | sink rises | rare | 1.00 | 0.285 / 0.288 | 0.060 | `ill`(359), `ied`(798), `ight`(432), `ying`(1112), `se`(325) | ` Activ`(13144), ` VS`(22269), ` oxidation`(37767), ` Angular`(28147), ` Veh`(15118) |
| L1H7 | sink rises | common | 1.00 | 0.225 / 0.223 | 0.043 | `ied`(798), `ying`(1112), `ill`(359), `ank`(962), `ike`(522) | ` Activ`(13144), ` Veh`(15118), ` Angular`(28147), ` melee`(16837), ` ()`(7499) |
| L7H7 | sink rises | rare | 0.99 | 0.754 / 0.762 | 0.147 | ` <[`(29342), `【`(31854), `.[`(3693), `:(`(37498), `�`(145) | ` lobbyist`(35196), ` paranoia`(34370), ` insanity`(30949), ` electronics`(17075), ` disbelief`(29894) |
| L7H7 | sink rises | common | 0.97 | 0.615 / 0.621 | 0.121 | ` <[`(29342), `�`(26292), `.[`(3693), `�`(140), `�`(141) | ` hesitant`(32848), ` surprised`(6655), `awei`(38247), ` also`(635), ` paranoia`(34370) |
| L9H11 | sink rises | rare | 1.00 | 0.863 / 0.865 | 0.207 | `�`(148), `�`(172), `�`(146), `�`(134), `�`(140) | ` prostitutes`(32159), ` slavery`(13503), `ushima`(30474), `manuel`(18713), `clerosis`(31399) |
| L9H11 | sink rises | common | 0.99 | 0.711 / 0.744 | 0.139 | `�`(172), `�`(146), `�`(148), ` Posts`(12043), `�`(138) | ` hoping`(7725), ` constitute`(15613), ` somebody`(8276), `lishing`(20020), ` brutality`(24557) |
| L10H11 | sink rises | rare | 1.00 | 0.896 / 0.906 | 0.230 | `�`(131), ` Kir`(7385), `�`(134), `�`(157), `�`(129) | `\x04`(192), `\x0b`(199), `\x10`(204), `\x08`(196), `\x17`(211) |
| L10H11 | sink rises | common | 0.99 | 0.789 / 0.792 | 0.163 | `�`(131), `�`(132), `�`(128), `�`(134), `�`(157) | ` outstanding`(11660), ` Impossible`(38791), ` neighboring`(19651), ` leftover`(39191), `./`(19571) |
| L11H1 | sink rises | rare | 1.00 | 0.868 / 0.866 | 0.257 | ` unexpl`(31286), ` unle`(15809), `�`(160), `י�`(33951), ` Hamb`(26175) | ` rankings`(16905), ` convictions`(19131), `lasses`(28958), ` evaluations`(34109), ` scandals`(28449) |
| L11H1 | sink rises | common | 1.00 | 0.791 / 0.780 | 0.195 | `�`(132), `\n\n`(628), `�`(131), `�`(157), `�`(160) | ` alright`(23036), ` rankings`(16905), ` nominee`(10429), ` emerges`(25457), ` supervision`(20865) |
| L11H3 | sink rises | rare | 1.00 | 0.868 / 0.871 | 0.226 | `�`(131), `�`(26292), `�`(160), `�`(150), `�`(154) | `itability`(34147), ` prioritize`(32980), `/)`(34729), `ichever`(22617), `grades`(31177) |
| L11H3 | sink rises | common | 1.00 | 0.835 / 0.830 | 0.192 | `�`(131), `�`(26292), `�`(157), `�`(132), `�`(128) | `oples`(12614), ` preferable`(33887), ` customized`(27658), ` cheaper`(11721), ` neighboring`(19651) |
| L2H1 | sink rises | rare | 1.00 | 0.782 / 0.770 | 0.193 | `\x17`(211), `�`(182), `\x0b`(199), `\x02`(190), `�`(184) | ` mod`(953), ` Energy`(6682), ` data`(1366), ` conversion`(11315), ` pol`(755) |
| L2H1 | sink rises | common | 1.00 | 0.646 / 0.633 | 0.139 | `ced`(771), `ent`(298), `ever`(964), `hing`(722), `ces`(728) | ` Energy`(6682), ` Commander`(13353), ` Rox`(34821), ` Kyl`(39859), ` Reach`(25146) |
| L2H6 | sink rises | rare | 1.00 | 0.290 / 0.299 | 0.049 | `av`(615), ` (`(357), `ox`(1140), `her`(372), `ark`(668) | `)]`(15437), `\n`(198), `\n\n`(628), `--------`(982), `}`(92) |
| L2H6 | sink rises | common | 1.00 | 0.345 / 0.302 | 0.051 | `aring`(1723), `av`(615), `her`(372), `ement`(972), `ion`(295) | `\n`(198), `\n\n`(628), `)]`(15437), `):`(2599), `].`(4083) |
| L3H6 | sink rises | rare | 0.99 | 0.217 / 0.224 | 0.037 | ` ;;`(36792), ` VOL`(38570), ` \u200e`(24398), ` You`(921), ` \\\\`(26867) | `ortex`(26158), `ategories`(26129), `glers`(33641), `sonian`(35202), `antage`(36403) |
| L3H6 | sink rises | common | 0.99 | 0.287 / 0.264 | 0.048 | ` \u200e`(24398), ` ;;`(36792), ` THIS`(12680), ` You`(921), ` Since`(4619) | `meta`(28961), `focus`(37635), `icidal`(21488), `inav`(26802), `oreal`(39396) |
| L7H11 | sink rises | rare | 1.00 | 0.850 / 0.850 | 0.227 | `�`(164), `�`(157), ` <[`(29342), ` Ch`(609), `�`(165) | ` dealings`(29043), ` supervision`(20865), ` slavery`(13503), ` brutality`(24557), ` disbelief`(29894) |
| L7H11 | sink rises | common | 0.99 | 0.684 / 0.690 | 0.169 | ` <[`(29342), `【`(31854), `](`(16151), `�`(168), `�`(131) | `seeking`(38515), ` brutality`(24557), ` legalize`(35605), `majority`(35839), ` begging`(26732) |
| L10H4 | sink rises | rare | 1.00 | 0.870 / 0.866 | 0.242 | ` Hamb`(26175), ` usur`(39954), `�`(160), `�`(166), `�`(143) | `opus`(25790), `thritis`(34043), ` turbines`(35658), `rike`(8760), `ocytes`(30309) |
| L10H4 | sink rises | common | 1.00 | 0.739 / 0.763 | 0.171 | `�`(132), `�`(26292), `�`(131), `�`(166), ` produ`(990) | ` staggering`(23508), ` misunderstood`(33046), ` consciously`(30410), ` outstanding`(11660), ` tremendous`(12465) |
| L11H9 | sink rises | rare | 1.00 | 0.910 / 0.912 | 0.233 | `�`(166), ` bl`(698), `�`(169), ` Hamb`(26175), ` Hor`(6075) | `\x15`(209), `\x1a`(214), `\x03`(191), `龍�`(39820), `\x14`(208) |
| L11H9 | sink rises | common | 0.99 | 0.677 / 0.679 | 0.139 | ` IPv`(25961), `�`(26292), `�`(157), `�`(132), `�`(142) | ` Compare`(27814), `\x14`(208), `�`(179), `\x1d`(217), `\x03`(191) |
| L1H8 | sink falls | rare | 1.00 | 0.180 / 0.182 | 0.030 | ` Trend`(22836), `Loading`(19031), `inventoryQuantity`(39756), `OTOS`(33291), ` Template`(37350) | `\x1f`(219), `\x10`(204), `\x02`(190), `�`(182), `\x13`(207) |
| L1H8 | sink falls | common | 1.00 | 0.132 / 0.132 | 0.023 | `Loading`(19031), ` Yates`(34916), ` Trend`(22836), ` horizontally`(36774), `�`(144) | `her`(372), `."`(526), `ult`(586), `ition`(653), `ige`(10045) |
| L2H0 | sink falls | rare | 0.99 | 0.294 / 0.300 | 0.054 | ` popped`(22928), ` snapped`(20821), `!"`(2474), `?),`(33924), ` transforms`(31408) | `ly`(306), `�`(146), ` of`(286), `�`(11976), `ily`(813) |
| L2H0 | sink falls | common | 1.00 | 0.252 / 0.231 | 0.049 | `,"`(553), ` agrees`(14386), `!"`(2474), `?),`(33924), ` extracted`(21242) | `ly`(306), `ily`(813), `�`(146), `ity`(414), `ant`(415) |
| L2H4 | sink falls | rare | 1.00 | 0.793 / 0.803 | 0.179 | `Welcome`(14618), `Though`(10915), ` Fortunately`(20525), ` VOL`(38570), ` continuing`(8282) | `ly`(306), `ance`(590), `ed`(276), `ally`(453), `ing`(278) |
| L2H4 | sink falls | common | 1.00 | 0.827 / 0.817 | 0.224 | ` IPv`(25961), ` VOL`(38570), ` Comed`(37024), ` Unix`(33501), `Welcome`(14618) | `ance`(590), `ated`(515), `ally`(453), `ed`(276), `ily`(813) |
| L3H3 | sink falls | rare | 1.00 | 0.860 / 0.857 | 0.263 | ` usur`(39954), ` Remember`(11436), ` enjoyed`(8359), ` Since`(4619), ` bec`(639) | `en`(268), `al`(282), `em`(368), `il`(346), `ol`(349) |
| L3H3 | sink falls | common | 1.00 | 0.866 / 0.879 | 0.281 | ` surprised`(6655), ` usur`(39954), ` Since`(4619), ` dismay`(29793), ` recru`(8921) | `al`(282), `ys`(893), `en`(268), `il`(346), `od`(375) |
| L6H4 | sink falls | rare | 0.99 | 0.728 / 0.715 | 0.127 | `�`(155), `龍�`(39820), `\x06`(194), `\x07`(195), `\x15`(209) | ` The`(383), ` U`(471), ` Right`(6498), ` Economic`(11279), ` He`(679) |
| L6H4 | sink falls | common | 0.99 | 0.613 / 0.625 | 0.105 | `Redditor`(34832), `�`(155), ` Arduino`(27634), `�`(172), `�`(150) | ` Min`(1855), ` Social`(5483), ` Ord`(14230), ` Ch`(609), ` Ban`(10274) |
| L6H8 | sink falls | rare | 0.99 | 0.612 / 0.613 | 0.131 | ` Georg`(6850), ` Pap`(14185), ` Witt`(38005), ` goalt`(30896), ` Hort`(31693) | `ally`(453), `ually`(935), `ions`(507), `ation`(341), `ition`(653) |
| L6H8 | sink falls | common | 0.98 | 0.533 / 0.560 | 0.113 | ` Pap`(14185), ` Arduino`(27634), ` hairc`(37363), ` Mand`(13314), ` TRI`(37679) | `urally`(20221), `iously`(6819), `ually`(935), `itely`(3973), `ally`(453) |
| L7H0 | sink falls | rare | 0.97 | 0.511 / 0.520 | 0.098 | ` unimagin`(37656), `...`(986), ` Having`(11136), ` Sebast`(22787), ` Former`(14466) | ` so`(523), `uscript`(15817), ` year`(614), `sonian`(35202), `Â`(5523) |
| L7H0 | sink falls | common | 0.98 | 0.676 / 0.665 | 0.132 | `!"`(2474), `."`(526), `](`(16151), `"(`(18109), `,"`(553) | `accessible`(33780), `uscript`(15817), `ritional`(21297), `ructure`(5620), `uggets`(26550) |
| L10H9 | sink falls | rare | 0.99 | 0.478 / 0.473 | 0.081 | `�`(173), `�`(154), `\n\n`(628), ` Kir`(7385), `�`(159) | `Â`(5523), `′`(17478), `\xa0\xa0\xa0`(33477), ```(63), `ng`(782) |
| L10H9 | sink falls | common | 1.00 | 0.775 / 0.778 | 0.175 | `�`(173), `�`(159), `�`(157), `�`(166), ` �`(14519) | `\xa0\xa0\xa0`(33477), `Â`(5523), `ng`(782), `^`(61), `\xa0\xa0`(4603) |
| L4H0 | sink falls | rare | 0.99 | 0.719 / 0.706 | 0.141 | `.[`(3693), ` Produ`(21522), `))`(4008), ` absent`(13717), ` Getting`(18067) | ` them`(606), ` everything`(2279), `ed`(276), ` him`(683), ` their`(511) |
| L4H0 | sink falls | common | 0.99 | 0.726 / 0.724 | 0.136 | `.[`(3693), `…)`(38418), `))`(4008), ` allows`(3578), ` Produ`(21522) | ` them`(606), ` its`(663), ` him`(683), ` their`(511), ` everything`(2279) |
| L5H2 | sink falls | rare | 0.98 | 0.408 / 0.398 | 0.066 | `arg`(853), `�`(129), `�`(140), `qu`(421), `ew`(413) | `),`(828), `?),`(33924), `iously`(6819), `].`(4083), `).`(737) |
| L5H2 | sink falls | common | 0.99 | 0.690 / 0.702 | 0.142 | `�`(129), `�`(128), `�`(26292), `�`(157), `�`(134) | `enance`(36368), `urally`(20221), `clerosis`(31399), `iously`(6819), `izontal`(38342) |
| L5H6 | sink falls | rare | 0.96 | 0.546 / 0.564 | 0.099 | ` their`(511), ` 2050`(32215), ` them`(606), ` 401`(22219), ` 1985`(12863) | `ç`(16175), `ond`(623), `ew`(413), `ph`(746), `gh`(456) |
| L5H6 | sink falls | common | 0.95 | 0.465 / 0.499 | 0.094 | ` Learns`(30667), `iously`(6819), ` Where`(6350), ` comprising`(27918), `==`(855) | `ç`(16175), `ut`(315), `padding`(39231), `and`(392), `ph`(746) |
| L8H2 | sink falls | rare | 0.99 | 0.717 / 0.707 | 0.163 | `�`(152), ` Hamb`(26175), `�`(146), `י�`(33951), `�`(148) | ` safeguards`(32673), `?`(30), ` Fortunately`(20525), ` reconstruction`(25056), `itability`(34147) |
| L8H2 | sink falls | common | 0.99 | 0.788 / 0.788 | 0.154 | `�`(152), `�`(146), `�`(131), `Redditor`(34832), `�`(145) | ` constitute`(15613), ` every`(790), ` throughout`(3690), ` recommending`(34639), ` while`(981) |
| L9H8 | sink falls | rare | 0.98 | 0.573 / 0.586 | 0.108 | `�`(150), ` clen`(38566), `�`(146), `�`(23626), ` overhe`(34789) | ` throughout`(3690), ` from`(422), `azar`(29413), ` East`(3687), ` on`(319) |
| L9H8 | sink falls | common | 0.99 | 0.628 / 0.603 | 0.115 | `Redditor`(34832), `�`(175), `WH`(12418), `�`(150), `�`(176) | `azar`(29413), ` defender`(13191), `ariat`(21621), ` defenders`(16355), ` inland`(37874) |
| L0H6 | label mix only | rare | 1.00 | 0.161 / 0.164 | 0.030 | ` Yanuk`(37068), `Minecraft`(39194), `awei`(38247), ` refriger`(19866), `�`(144) | ` in`(287), ` the`(262), ` is`(318), ` was`(373), ` not`(407) |
| L0H6 | label mix only | common | 1.00 | 0.250 / 0.248 | 0.053 | `�`(144), ` Yanuk`(37068), `�`(153), ` Byron`(36719), ` tariff`(36427) | `ve`(303), `is`(271), `r`(81), `out`(448), `all`(439) |
| L1H0 | label mix only | rare | 1.00 | 0.112 / 0.110 | 0.014 | `!"`(2474), `."`(526), `,"`(553), `"(`(18109), `Though`(10915) | `ress`(601), `et`(316), `ide`(485), `ock`(735), `ut`(315) |
| L1H0 | label mix only | common | 1.00 | 0.237 / 0.222 | 0.034 | `!"`(2474), `."`(526), ` <[`(29342), `,"`(553), `�`(173) | `ight`(432), `ound`(633), `ack`(441), `ide`(485), `ock`(735) |
| L3H2 | label mix only | rare | 0.99 | 0.297 / 0.298 | 0.049 | `."`(526), `!"`(2474), ` Bond`(12812), ` Tory`(19037), `,"`(553) | `antage`(36403), `FFFF`(29312), `ن`(23338), `hement`(35347), `�`(11976) |
| L3H2 | label mix only | common | 0.98 | 0.277 / 0.258 | 0.048 | `,"`(553), ` \u200e`(24398), ` -----`(37404), ` Bond`(12812), `)]`(15437) | `ل`(13862), `quick`(24209), `�`(23626), `FFFF`(29312), `ن`(23338) |
| L10H0 | label mix only | rare | 1.00 | 0.880 / 0.870 | 0.167 | ` Patt`(24682), ` usur`(39954), ` Hamb`(26175), ` sou`(24049), `�`(160) | `\x15`(209), `\x1a`(214), `\x03`(191), `龍�`(39820), `\x12`(206) |
| L10H0 | label mix only | common | 0.99 | 0.552 / 0.504 | 0.100 | `�`(157), `\n\n`(628), `�`(160), `�`(132), `�`(131) | `�`(174), ``````(33153), `\x03`(191), `\x1d`(217), `�`(179) |
| L0H1 | control | rare | 1.00 | 0.011 / 0.011 | 0.002 | `\n`(198), `\n\n`(628), `.`(13), ` \|`(930), `:`(25) | ` Mand`(13314), ` Jacob`(12806), ` brute`(33908), `Lu`(25596), ` brid`(38265) |
| L0H1 | control | common | 1.00 | 0.010 / 0.010 | 0.002 | `\n`(198), `\n\n`(628), `.`(13), ` \|`(930), `:`(25) | ` Mand`(13314), ` Jacob`(12806), ` brute`(33908), `Lu`(25596), ` brid`(38265) |
| L4H11 | control | rare | 1.00 | 0.079 / 0.070 | 0.008 | `\x18`(212), `�`(182), `\x03`(191), `�`(124), `\x7f`(221) | `gew`(39909), `about`(10755), `gh`(456), `xt`(742), `aff`(2001) |
| L4H11 | control | common | 0.54 | 0.001 / 0.000 | 0.000 | `\x7f`(221), `�`(124), `\x1d`(217), `�`(182), `\x03`(191) | `against`(32826), ` of`(286), ` margins`(20241), `watch`(8340), ` orbit`(13066) |
| L5H1 | control | rare | 0.99 | 0.433 / 0.386 | 0.050 | ` Ar`(943), ` Ak`(9084), ` Az`(7578), ` Ch`(609), ` Ba`(8999) | `\x15`(209), `\x7f`(221), `\x03`(191), `\x12`(206), `�`(185) |
| L5H1 | control | common | 0.99 | 0.217 / 0.196 | 0.025 | ` Az`(7578), ` Social`(5483), ` Rail`(12950), ` Mem`(4942), ` Ba`(8999) | `\x03`(191), `\x0c`(200), `\x17`(211), `\x1c`(216), `�`(177) |

Mean sink_mass by final-token class (rare bases):

| name | byte | common | rare |
|---|---|---|---|
| L1H5 | 0.229 | 0.298 | 0.170 |
| L1H7 | 0.168 | 0.191 | 0.116 |
| L7H7 | 0.378 | 0.331 | 0.236 |
| L9H11 | 0.454 | 0.381 | 0.287 |
| L10H11 | 0.323 | 0.366 | 0.244 |
| L11H1 | 0.470 | 0.551 | 0.394 |
| L11H3 | 0.431 | 0.415 | 0.339 |
| L2H1 | 0.521 | 0.349 | 0.209 |
| L2H6 | 0.198 | 0.239 | 0.204 |
| L3H6 | 0.073 | 0.075 | 0.080 |
| L7H11 | 0.595 | 0.527 | 0.360 |
| L10H4 | 0.386 | 0.426 | 0.310 |
| L11H9 | 0.387 | 0.473 | 0.339 |
| L1H8 | 0.209 | 0.216 | 0.241 |
| L2H0 | 0.229 | 0.207 | 0.249 |
| L2H4 | 0.528 | 0.524 | 0.706 |
| L3H3 | 0.263 | 0.372 | 0.590 |
| L6H4 | 0.474 | 0.303 | 0.327 |
| L6H8 | 0.412 | 0.482 | 0.523 |
| L7H0 | 0.512 | 0.506 | 0.564 |
| L10H9 | 0.775 | 0.790 | 0.823 |
| L4H0 | 0.691 | 0.649 | 0.794 |
| L5H2 | 0.836 | 0.812 | 0.808 |
| L5H6 | 0.522 | 0.554 | 0.578 |
| L8H2 | 0.672 | 0.541 | 0.503 |
| L9H8 | 0.613 | 0.622 | 0.649 |
| L0H6 | 0.067 | 0.029 | 0.065 |
| L1H0 | 0.018 | 0.014 | 0.017 |
| L3H2 | 0.057 | 0.095 | 0.099 |
| L10H0 | 0.458 | 0.591 | 0.546 |
| L0H1 | 0.001 | 0.000 | 0.000 |
| L4H11 | 0.012 | 0.000 | 0.000 |
| L5H1 | 0.917 | 0.986 | 0.974 |

## 5. Substitution at position 31: which context tokens move each head?

Per token: Δ sink_mass vs the unsubstituted base (common tokens into rare bases; rare tokens into common bases), averaged over bases. reliability = correlation of per-token means between two random halves of the bases (only read rankings where it is high); held-out = the top-minus-bottom gap of the top/bottom 15 tokens chosen on one half, measured on the other half (shrinkage = winner's curse).

| head | group | bases | reliability | held-out gap | sd over tokens | highest 5 | lowest 5 |
|---|---|---|---|---|---|---|---|
| L1H5 | sink rises | rare | 0.96 | 0.032 / 0.035 | 0.005 | `I`(40), `In`(818), `F`(37), ` In`(554), ` M`(337) | ` and`(290), `,`(11), ` or`(393), ` the`(262), `-`(12) |
| L1H5 | sink rises | common | 0.92 | 0.030 / 0.033 | 0.005 | `):`(2599), ` Having`(11136), ` On`(1550), ` -----`(37404), ` So`(1406) | `isSpecialOrderable`(39755), `龍�`(39820), ` ng`(23370), `cffff`(31727), `′`(17478) |
| L1H7 | sink rises | rare | 0.98 | 0.024 / 0.026 | 0.004 | ` people`(661), `I`(40), ` I`(314), ` what`(644), ` world`(995) | `).`(737), `."`(526), `\n`(198), ` The`(383), `),`(828) |
| L1H7 | sink rises | common | 0.98 | 0.038 / 0.036 | 0.006 | `How`(2437), `Thank`(10449), ` Where`(6350), `Sadly`(36725), ` somebody`(8276) | `isSpecialOrderable`(39755), ` embryos`(39966), `))`(4008), ` eSports`(32717), ` GHz`(26499) |
| L7H7 | sink rises | rare | 0.95 | 0.463 / 0.488 | 0.088 | ` (`(357), ` [`(685), ` "`(366), `(`(7), ` '`(705) | `ublic`(841), `cept`(984), `rit`(799), `od`(375), `vern`(933) |
| L7H7 | sink rises | common | 0.95 | 0.326 / 0.332 | 0.050 | `_{`(23330), ` <[`(29342), `.[`(3693), `"(`(18109), `【`(31854) | `instead`(38070), ` frown`(22002), ` nodd`(13177), `��`(35069), `append`(33295) |
| L9H11 | sink rises | rare | 0.92 | 0.279 / 0.329 | 0.040 | ` [`(685), ` "`(366), ` (`(357), `[`(58), ` '`(705) | `&`(5), `�`(144), ` qu`(627), ` whe`(483), ` he`(339) |
| L9H11 | sink rises | common | 0.96 | 0.300 / 0.315 | 0.041 | ` <[`(29342), `.[`(3693), `](`(16151), `Redditor`(34832), `【`(31854) | `·`(9129), ` helped`(4193), ` withstand`(25073), ` coolest`(38889), ` assisting`(26508) |
| L10H11 | sink rises | rare | 0.90 | 0.194 / 0.276 | 0.043 | ` "`(366), ` (`(357), ` [`(685), `,"`(553), ` '`(705) | ` differe`(980), ` produ`(990), `qu`(421), ` qu`(627), ` supp`(802) |
| L10H11 | sink rises | common | 0.93 | 0.264 / 0.288 | 0.042 | `Redditor`(34832), ` <[`(29342), ` La`(4689), ` Posts`(12043), `](`(16151) | ` generate`(7716), ` negligible`(36480), ` unacceptable`(18010), ` inadequate`(20577), ` unused`(21958) |
| L11H1 | sink rises | rare | 0.87 | 0.215 / 0.283 | 0.038 | ` [`(685), ` "`(366), ` (`(357), `[`(58), `(`(7) | ` qu`(627), ` mon`(937), ` stud`(941), `qu`(421), ` kn`(638) |
| L11H1 | sink rises | common | 0.96 | 0.316 / 0.326 | 0.050 | ` <[`(29342), `.[`(3693), `Redditor`(34832), `【`(31854), `](`(16151) | ` commander`(11561), ` nominee`(10429), ` member`(2888), ` president`(1893), ` defender`(13191) |
| L11H3 | sink rises | rare | 0.86 | 0.139 / 0.199 | 0.028 | `{`(90), `�`(172), ` [`(685), `�`(173), `[`(58) | ` produ`(990), ` Com`(955), ` differe`(980), ` whe`(483), ` spe`(693) |
| L11H3 | sink rises | common | 0.93 | 0.260 / 0.270 | 0.040 | ` <[`(29342), `Redditor`(34832), `�`(26292), `_{`(23330), `.[`(3693) | `Example`(16281), ` generate`(7716), `Pay`(19197), ` Function`(15553), ` cheaper`(11721) |
| L2H1 | sink rises | rare | 0.97 | 0.076 / 0.097 | 0.014 | ` bec`(639), `@`(31), `�`(150), ` num`(997), `�`(133) | `\n`(198), `\n\n`(628), `).`(737), `."`(526), `.`(13) |
| L2H1 | sink rises | common | 0.96 | 0.122 / 0.118 | 0.019 | ` Corbyn`(15673), ` Miliband`(38199), ` postseason`(22905), `Redditor`(34832), ` scarcely`(32335) | ` 2048`(36117), `))`(4008), ` ()`(7499), ` additionally`(36527), ` entity`(9312) |
| L2H6 | sink rises | rare | 1.00 | 0.221 / 0.238 | 0.033 | ` (`(357), `(`(7), ` [`(685), `[`(58), ` '`(705) | `).`(737), `."`(526), `\n`(198), `\n\n`(628), `}`(92) |
| L2H6 | sink rises | common | 0.99 | 0.282 / 0.235 | 0.030 | ` <[`(29342), ` abandoning`(31309), ` opposing`(12330), ` wielding`(33949), ` delivering`(13630) | `].`(4083), `!"`(2474), `):`(2599), `)]`(15437), `))`(4008) |
| L3H6 | sink rises | rare | 0.99 | 0.191 / 0.208 | 0.037 | `;`(26), `,"`(553), ` when`(618), `."`(526), ` where`(810) | `J`(41), ` he`(339), `�`(160), ` fin`(957), `M`(44) |
| L3H6 | sink rises | common | 0.98 | 0.223 / 0.202 | 0.036 | `!"`(2474), `.[`(3693), `?),`(33924), `].`(4083), `…)`(38418) | ` Stef`(22350), ` STE`(24483), ` Mitch`(20472), ` Eli`(25204), ` Ian`(12930) |
| L7H11 | sink rises | rare | 0.93 | 0.300 / 0.353 | 0.050 | ` [`(685), ` (`(357), ` "`(366), `[`(58), `(`(7) | ` spe`(693), ` differe`(980), ` ev`(819), ` dis`(595), ` stud`(941) |
| L7H11 | sink rises | common | 0.98 | 0.386 / 0.381 | 0.071 | ` <[`(29342), `Redditor`(34832), `.[`(3693), `](`(16151), `_{`(23330) | ` doubted`(37104), ` econom`(1707), ` underest`(20164), ` traff`(11878), ` solic`(25806) |
| L10H4 | sink rises | rare | 0.85 | 0.170 / 0.207 | 0.029 | ` (`(357), ` [`(685), `@`(31), `\n\n`(628), ` This`(770) | `{`(90), `�`(142), `�`(133), `�`(157), `�`(134) |
| L10H4 | sink rises | common | 0.93 | 0.276 / 0.286 | 0.042 | ` <[`(29342), `utsche`(30433), ` La`(4689), `.[`(3693), `Redditor`(34832) | `\xa0\xa0\xa0`(33477), ` \\"`(19990), `\xa0\xa0`(4603), `_{`(23330), `·`(9129) |
| L11H9 | sink rises | rare | 0.91 | 0.210 / 0.249 | 0.040 | ` [`(685), ` "`(366), `[`(58), ` '`(705), `,"`(553) | `�`(154), `�`(133), `�`(157), ` ev`(819), ` differe`(980) |
| L11H9 | sink rises | common | 0.96 | 0.274 / 0.272 | 0.046 | ` <[`(29342), `Redditor`(34832), `.[`(3693), `utsche`(30433), `](`(16151) | ` tense`(20170), `define`(13086), `Footnote`(33795), ` rewarding`(23404), ` lever`(17124) |
| L1H8 | sink falls | rare | 0.99 | 0.025 / 0.025 | 0.004 | ` or`(393), ` because`(780), ` and`(290), ` if`(611), ` before`(878) | `."`(526), `\n`(198), `The`(464), `).`(737), `.`(13) |
| L1H8 | sink falls | common | 0.96 | 0.017 / 0.018 | 0.003 | ` favour`(7075), ` happ`(1147), ` organis`(13867), ` <=`(19841), ` prioritize`(32980) | `!"`(2474), `Though`(10915), `.[`(3693), ` \u200e`(24398), ` ()`(7499) |
| L2H0 | sink falls | rare | 0.99 | 0.148 / 0.148 | 0.024 | `,"`(553), `),`(828), `."`(526), `!`(0), `...`(986) | ` in`(287), ` of`(286), ` on`(319), ` at`(379), ` to`(284) |
| L2H0 | sink falls | common | 0.98 | 0.088 / 0.079 | 0.014 | `!"`(2474), `?),`(33924), `…)`(38418), ` mice`(10693), `/)`(34729) | ` On`(1550), ` throughout`(3690), ` Behind`(20787), ` 09`(7769), ` Bene`(36585) |
| L2H4 | sink falls | rare | 0.96 | 0.322 / 0.312 | 0.059 | ` use`(779), ` used`(973), ` look`(804), ` report`(989), `ays`(592) | ` and`(290), `,`(11), `.`(13), ` the`(262), ` The`(383) |
| L2H4 | sink falls | common | 0.91 | 0.157 / 0.169 | 0.028 | `](`(16151), ` helmets`(28359), ` protects`(17289), ` tools`(4899), ` foes`(20822) | `isSpecialOrderable`(39755), ` enthusi`(11273), ` On`(1550), ` Or`(1471), `your`(14108) |
| L3H3 | sink falls | rare | 0.93 | 0.293 / 0.305 | 0.052 | ` 19`(678), ` 200`(939), ` 8`(807), ` 9`(860), ` 10`(838) | `�`(175), ` an`(281), `]`(60), `)`(8), ` to`(284) |
| L3H3 | sink falls | common | 0.91 | 0.278 / 0.295 | 0.042 | ` 405`(36966), ` 257`(36100), ` 266`(37737), ` 224`(26063), ` 322`(38831) | `isSpecialOrderable`(39755), ` On`(1550), ` predec`(14060), ` STE`(24483), ` Neigh`(22505) |
| L6H4 | sink falls | rare | 0.92 | 0.141 / 0.139 | 0.028 | ` you`(345), ` who`(508), ` like`(588), ` they`(484), ` she`(673) | `--------`(982), `}`(92), `�`(174), `�`(124), `\x7f`(221) |
| L6H4 | sink falls | common | 0.94 | 0.204 / 0.206 | 0.034 | `Redditor`(34832), ` Arduino`(27634), ` Tumblr`(24434), ` bacterial`(23462), `Minecraft`(39194) | ` ng`(23370), `ONG`(18494), ` Learns`(30667), ` ACTIONS`(23054), `orf`(24263) |
| L6H8 | sink falls | rare | 0.97 | 0.367 / 0.388 | 0.086 | ` what`(644), ` [`(685), ` because`(780), ` when`(618), ` $`(720) | `cc`(535), `row`(808), `y`(88), `iew`(769), `ight`(432) |
| L6H8 | sink falls | common | 0.98 | 0.473 / 0.445 | 0.092 | ` <[`(29342), ` Getting`(18067), `_{`(23330), ` Where`(6350), ` ACTIONS`(23054) | `263`(29558), `cin`(17879), `lp`(34431), `§`(16273), `iffe`(22391) |
| L7H0 | sink falls | rare | 0.97 | 0.427 / 0.412 | 0.092 | ` here`(994), `ation`(341), `ility`(879), `ition`(653), `ement`(972) | `{`(90), `<`(27), ` des`(748), `'re`(821), `�`(172) |
| L7H0 | sink falls | common | 0.94 | 0.345 / 0.394 | 0.065 | ` explosion`(11278), ` collaboration`(12438), ` reconstruction`(25056), ` exploration`(13936), ` disbelief`(29894) | `′`(17478), ` <[`(29342), `Redditor`(34832), `·`(9129), `_{`(23330) |
| L10H9 | sink falls | rare | 0.95 | 0.172 / 0.182 | 0.037 | `�`(174), `ments`(902), `ause`(682), `ations`(602), `\x08`(196) | ` being`(852), ` and`(290), ` or`(393), `/`(14), ` been`(587) |
| L10H9 | sink falls | common | 0.97 | 0.390 / 0.407 | 0.063 | ` <[`(29342), `](`(16151), `_{`(23330), `.[`(3693), `【`(31854) | ` closely`(7173), ` discrim`(6534), ` tremend`(11039), ` depended`(33785), ` accompl`(6424) |
| L4H0 | sink falls | rare | 0.98 | 0.413 / 0.422 | 0.078 | `--------`(982), `ction`(596), `ility`(879), `olog`(928), `ement`(972) | `�`(175), ` of`(286), ` to`(284), ` with`(351), ` its`(663) |
| L4H0 | sink falls | common | 0.94 | 0.371 / 0.369 | 0.068 | `---------------`(24305), `?),`(33924), `):`(2599), `].`(4083), ` Fortunately`(20525) | ` La`(4689), ` pers`(2774), ` Wi`(11759), ` Seg`(31220), ` Yanuk`(37068) |
| L5H2 | sink falls | rare | 0.96 | 0.376 / 0.403 | 0.067 | ` This`(770), ` Un`(791), ` New`(968), ` Wh`(854), ` Sh`(911) | ` or`(393), ` and`(290), `\\`(59), `&`(5), `.`(13) |
| L5H2 | sink falls | common | 0.98 | 0.583 / 0.551 | 0.116 | `.[`(3693), ` Previous`(21801), ` <[`(29342), ` Several`(12168), ` THIS`(12680) | `\xa0\xa0\xa0`(33477), `Â`(5523), `214`(22291), `263`(29558), `106`(15801) |
| L5H6 | sink falls | rare | 0.89 | 0.354 / 0.445 | 0.087 | `...`(986), ` 7`(767), ` over`(625), ` 5`(642), ` v`(410) | `�`(172), `�`(175), `�`(145), `�`(154), `�`(156) |
| L5H6 | sink falls | common | 0.96 | 0.471 / 0.462 | 0.084 | ` ;;`(36792), ` 257`(36100), ` Royal`(8111), ` knocks`(36539), ` 322`(38831) | ` Neigh`(22505), ` unimagin`(37656), ` Mississ`(12732), `Thank`(10449), ` celeb`(10624) |
| L8H2 | sink falls | rare | 0.90 | 0.142 / 0.220 | 0.033 | ` "`(366), ` (`(357), `@`(31), ` [`(685), `."`(526) | ` even`(772), ` their`(511), ` over`(625), ` such`(884), ` differe`(980) |
| L8H2 | sink falls | common | 0.97 | 0.349 / 0.338 | 0.061 | `Redditor`(34832), ` <[`(29342), `【`(31854), `utsche`(30433), `.[`(3693) | ` withstand`(25073), ` shortly`(8972), ` plun`(19324), `Â`(5523), ` tightly`(17707) |
| L9H8 | sink falls | rare | 0.95 | 0.235 / 0.261 | 0.040 | ` The`(383), `The`(464), `."`(526), ` [`(685), `[`(58) | `�`(145), `�`(157), `�`(150), `�`(161), `�`(128) |
| L9H8 | sink falls | common | 0.97 | 0.423 / 0.403 | 0.063 | `Redditor`(34832), ` bacterial`(23462), ` neuronal`(36347), ` urinary`(38628), ` <[`(29342) | `י�`(33951), `ラ`(9263), `ス`(8943), `ç`(16175), ` Aviv`(28890) |
| L0H6 | label mix only | rare | 0.98 | 0.003 / 0.004 | 0.001 | `'t`(470), ` not`(407), ` do`(466), ` know`(760), ` have`(423) | `�`(143), `�`(154), ` differe`(980), ` sim`(985), `�`(132) |
| L0H6 | label mix only | common | 0.95 | 0.010 / 0.011 | 0.002 | `16`(1433), `ying`(1112), ` put`(1234), `18`(1507), `ox`(1140) | `龍�`(39820), `Applic`(33583), ` enthusi`(11273), `sonian`(35202), `Redditor`(34832) |
| L1H0 | label mix only | rare | 0.94 | 0.009 / 0.011 | 0.002 | ` in`(287), ` of`(286), ` to`(284), `C`(34), ` would`(561) | ` (`(357), `(`(7), `�`(173), ` But`(887), `\x0f`(203) |
| L1H0 | label mix only | common | 0.86 | 0.018 / 0.019 | 0.003 | `106`(15801), `/)`(34729), `559`(38605), `248`(23045), `173`(25399) | ` Where`(6350), `龍�`(39820), ` Remember`(11436), ` Getting`(18067), ` Did`(7731) |
| L3H2 | label mix only | rare | 0.98 | 0.206 / 0.225 | 0.037 | ` where`(810), ` because`(780), ` while`(981), ` but`(475), ` when`(618) | ` ag`(556), ` differe`(980), `�`(175), `�`(156), `�`(171) |
| L3H2 | label mix only | common | 0.98 | 0.208 / 0.204 | 0.044 | `':`(10354), ` aside`(7263), `?),`(33924), ` establishment`(9323), `):`(2599) | ` sus`(2341), ` Fant`(8751), ` La`(4689), ` Bene`(36585), `�`(26292) |
| L10H0 | label mix only | rare | 0.90 | 0.184 / 0.218 | 0.033 | ` The`(383), ` A`(317), ` '`(705), ` "`(366), ` [`(685) | `�`(173), `�`(176), `�`(153), `�`(156), `�`(154) |
| L10H0 | label mix only | common | 0.92 | 0.229 / 0.234 | 0.030 | `utsche`(30433), `Redditor`(34832), `isSpecialOrderable`(39755), ` Gender`(20247), `Prosecutors`(39401) | `Â`(5523), ``````(33153), `י`(25529), `ノ`(25053), `\xa0\xa0\xa0`(33477) |
| L0H1 | control | rare | 0.40 | 0.000 / 0.000 | 0.000 | `-`(12), `q`(80), `g`(70), `ch`(354), `z`(89) | ` how`(703), `ful`(913), ` use`(779), ` whe`(483), `�`(132) |
| L0H1 | control | common | 0.22 | 0.000 / 0.000 | 0.000 | `173`(25399), `263`(29558), ` criticized`(12318), `559`(38605), `ari`(2743) | ` Stew`(12194), ` gets`(3011), ` Getting`(18067), ` Set`(5345), `AGE`(11879) |
| L4H11 | control | rare | 0.96 | 0.001 / 0.001 | 0.000 | `\n`(198), `�`(138), `'t`(470), `�`(151), `\x1d`(217) | `pp`(381), ` comm`(725), ` D`(360), `amp`(696), ` K`(509) |
| L4H11 | control | common | 0.19 | 0.000 / 0.000 | 0.000 | ` Understanding`(28491), ` abandoning`(31309), ` surrounding`(7346), `…)`(38418), `enance`(36368) | `′`(17478), `probably`(26949), `cffff`(31727), `uo`(20895), `almost`(28177) |
| L5H1 | control | rare | 0.73 | 0.025 / 0.034 | 0.007 | ` The`(383), ` your`(534), ` A`(317), ` You`(921), ` or`(393) | ` prov`(899), `ility`(879), ` mod`(953), `ied`(798), `ics`(873) |
| L5H1 | control | common | 0.65 | 0.010 / 0.016 | 0.002 | ` Alicia`(39607), ` Cam`(7298), `_{`(23330), ` Martha`(27243), ` Dead`(5542) | `instead`(38070), `also`(14508), ` enh`(5881), `龍�`(39820), ` foremost`(20976) |

## 6. Substitution at position 16: which context tokens move each head?

Per token: Δ sink_mass vs the unsubstituted base (common tokens into rare bases; rare tokens into common bases), averaged over bases. reliability = correlation of per-token means between two random halves of the bases (only read rankings where it is high); held-out = the top-minus-bottom gap of the top/bottom 15 tokens chosen on one half, measured on the other half (shrinkage = winner's curse).

| head | group | bases | reliability | held-out gap | sd over tokens | highest 5 | lowest 5 |
|---|---|---|---|---|---|---|---|
| L1H5 | sink rises | rare | 0.86 | 0.009 / 0.011 | 0.002 | ` 8`(807), ` W`(370), `8`(23), ` br`(865), ` E`(412) | `,`(11), ` and`(290), `.`(13), ` the`(262), `:`(25) |
| L1H5 | sink rises | common | 0.93 | 0.015 / 0.017 | 0.003 | ` Loc`(15181), ` Mic`(7631), ` overhe`(34789), ` EN`(12964), ` happ`(1147) | `isSpecialOrderable`(39755), `龍�`(39820), `cffff`(31727), `tackle`(36346), `Redditor`(34832) |
| L1H7 | sink rises | rare | 0.98 | 0.006 / 0.007 | 0.001 | ` bec`(639), `�`(133), `�`(143), ` bel`(894), `�`(144) | `The`(464), `.`(13), `."`(526), ` the`(262), ` of`(286) |
| L1H7 | sink rises | common | 0.99 | 0.027 / 0.026 | 0.005 | `Bel`(12193), ` unimagin`(37656), ` Stan`(7299), ` Sr`(21714), ` Bene`(36585) | ` embryos`(39966), `isSpecialOrderable`(39755), ` eSports`(32717), ` Arduino`(27634), ` filesystem`(29905) |
| L7H7 | sink rises | rare | 0.88 | 0.031 / 0.039 | 0.006 | `=`(28), `:`(25), ` =`(796), ` (`(357), ` [`(685) | `�`(172), `�`(160), `."`(526), `�`(175), ` its`(663) |
| L7H7 | sink rises | common | 0.89 | 0.122 / 0.120 | 0.018 | ` ng`(23370), `](`(16151), `"(`(18109), `:(`(37498), ` der`(4587) | `tackle`(36346), ` potions`(28074), ` adject`(31129), `ilight`(15512), ` zombies`(19005) |
| L9H11 | sink rises | rare | 0.93 | 0.046 / 0.057 | 0.008 | ` for`(329), ` with`(351), ` and`(290), ` from`(422), ` after`(706) | `�`(175), `�`(171), `{`(90), `�`(170), `..`(492) |
| L9H11 | sink rises | common | 0.97 | 0.163 / 0.155 | 0.020 | `utsche`(30433), ` <[`(29342), ` der`(4587), `ña`(30644), ` ng`(23370) | ` \u200e`(24398), ` >>>`(13163), `\xa0\xa0\xa0`(33477), `…)`(38418), ` ;;`(36792) |
| L10H11 | sink rises | rare | 0.82 | 0.029 / 0.043 | 0.007 | ` on`(319), ` over`(625), ` with`(351), ` in`(287), `@`(31) | `�`(154), `{`(90), `�`(172), `�`(173), `�`(153) |
| L10H11 | sink rises | common | 0.94 | 0.113 / 0.128 | 0.020 | `utsche`(30433), ` der`(4587), `erto`(13806), `ige`(10045), `dyl`(30360) | ` Meaning`(30563), ` Function`(15553), ` values`(3815), ` Learns`(30667), ` routines`(31878) |
| L11H1 | sink rises | rare | 0.91 | 0.033 / 0.038 | 0.007 | `...`(986), ` from`(422), ` at`(379), ` —`(851), `—`(960) | `�`(154), `{`(90), ` stud`(941), ` man`(582), `�`(144) |
| L11H1 | sink rises | common | 0.95 | 0.118 / 0.140 | 0.020 | `izarre`(12474), `](`(16151), `Redditor`(34832), ` \u200e`(24398), ` eSports`(32717) | ` Presidency`(35696), ` Jaime`(38028), ` Gender`(20247), ` Chair`(9369), ` Enemy`(21785) |
| L11H3 | sink rises | rare | 0.90 | 0.040 / 0.033 | 0.007 | `ily`(813), `ious`(699), ` most`(749), `—`(960), `ually`(935) | `(`(7), `�`(170), ` [`(685), ` (`(357), `{`(90) |
| L11H3 | sink rises | common | 0.94 | 0.103 / 0.104 | 0.018 | `isSpecialOrderable`(39755), `ña`(30644), `�`(26292), `utsche`(30433), `erto`(13806) | ` electronics`(17075), ` Alcohol`(21051), `University`(21009), ` Arduino`(27634), ` Films`(30198) |
| L2H1 | sink rises | rare | 0.97 | 0.017 / 0.024 | 0.004 | ` '`(705), ` bec`(639), ` app`(598), `�`(133), `clud`(758) | `_`(62), ` the`(262), `)`(8), `-`(12), ` and`(290) |
| L2H1 | sink rises | common | 0.96 | 0.077 / 0.076 | 0.011 | ` Corbyn`(15673), ` Miliband`(38199), ` postseason`(22905), ` goalt`(30896), ` UKIP`(37809) | `isSpecialOrderable`(39755), ` 2048`(36117), `padding`(39231), `グ`(26095), `ignment`(16747) |
| L2H6 | sink rises | rare | 1.00 | 0.049 / 0.049 | 0.006 | ` (`(357), `(`(7), ` [`(685), `[`(58), ` '`(705) | `."`(526), `).`(737), `}`(92), `)`(8), `\n`(198) |
| L2H6 | sink rises | common | 0.99 | 0.077 / 0.063 | 0.007 | `【`(31854), ` debuted`(26376), ` <[`(29342), `_{`(23330), ` comprising`(27918) | `!"`(2474), `].`(4083), `)]`(15437), `?),`(33924), `))`(4008) |
| L3H6 | sink rises | rare | 0.91 | 0.007 / 0.007 | 0.001 | `cess`(919), `arg`(853), `ople`(643), ` ass`(840), `age`(496) | `\n\n`(628), `."`(526), ` [`(685), ` (`(357), `(`(7) |
| L3H6 | sink rises | common | 0.94 | 0.020 / 0.023 | 0.003 | ` scarcely`(32335), `Footnote`(33795), `′`(17478), ` \u200e`(24398), ` Martha`(27243) | ` <[`(29342), `isSpecialOrderable`(39755), ` filesystem`(29905), ` ;;`(36792), `Code`(10669) |
| L7H11 | sink rises | rare | 0.93 | 0.039 / 0.041 | 0.007 | ` for`(329), `@`(31), ` and`(290), `:`(25), ` on`(319) | `{`(90), `(`(7), `�`(176), `�`(175), `�`(172) |
| L7H11 | sink rises | common | 0.92 | 0.104 / 0.116 | 0.016 | `Redditor`(34832), ` Posts`(12043), `utsche`(30433), ` UKIP`(37809), ` ng`(23370) | `Though`(10915), `Shortly`(30513), ` adject`(31129), `OY`(21414), `Task`(25714) |
| L10H4 | sink rises | rare | 0.94 | 0.047 / 0.057 | 0.008 | ` –`(784), `:`(25), ` '`(705), ` =`(796), ` over`(625) | `}`(92), `{`(90), `."`(526), `,"`(553), `"`(1) |
| L10H4 | sink rises | common | 0.95 | 0.139 / 0.150 | 0.017 | `utsche`(30433), ` eSports`(32717), ` Aviv`(28890), `Minecraft`(39194), ` der`(4587) | ` \\"`(19990), `\xa0\xa0\xa0`(33477), `_{`(23330), `---------------`(24305), `\xa0\xa0`(4603) |
| L11H9 | sink rises | rare | 0.94 | 0.068 / 0.080 | 0.013 | `...`(986), ` over`(625), ` in`(287), ` into`(656), ` on`(319) | `�`(173), `�`(154), `�`(172), `�`(153), `�`(170) |
| L11H9 | sink rises | common | 0.95 | 0.132 / 0.144 | 0.020 | `utsche`(30433), ` eSports`(32717), `Minecraft`(39194), ` gamer`(26713), ` hacker`(23385) | ` Meaning`(30563), `_{`(23330), ` Compare`(27814), ` tense`(20170), `\xa0\xa0\xa0`(33477) |
| L1H8 | sink falls | rare | 0.99 | 0.013 / 0.013 | 0.002 | ` includ`(846), `ople`(643), `ween`(975), ` count`(954), `cept`(984) | `."`(526), `.`(13), `).`(737), `)`(8), `\n`(198) |
| L1H8 | sink falls | common | 0.98 | 0.010 / 0.011 | 0.002 | ` sle`(3133), ` conveniently`(29801), `ilib`(22282), ` Mult`(7854), `:(`(37498) | ` Cavs`(38943), ` Arduino`(27634), ` motherboard`(32768), `龍�`(39820), `!"`(2474) |
| L2H0 | sink falls | rare | 0.98 | 0.013 / 0.013 | 0.002 | ` some`(617), ` much`(881), ` most`(749), ` their`(511), `The`(464) | `."`(526), `,"`(553), `?`(30), ` ,`(837), ` .`(764) |
| L2H0 | sink falls | common | 0.96 | 0.013 / 0.011 | 0.002 | ` nylon`(36104), `ococ`(34403), ` aperture`(32729), ` vivo`(34714), `ocytes`(30309) | `!"`(2474), `isSpecialOrderable`(39755), ` lobbyist`(35196), `?),`(33924), `](`(16151) |
| L2H4 | sink falls | rare | 0.97 | 0.017 / 0.018 | 0.003 | ` much`(881), `The`(464), ` still`(991), ` some`(617), `A`(32) | `."`(526), ` -`(532), ` ,`(837), ` to`(284), `,"`(553) |
| L2H4 | sink falls | common | 0.88 | 0.017 / 0.024 | 0.003 | ` glanced`(27846), ` gobl`(29646), ` scarcely`(32335), ` essentially`(6986), ` trig`(5192) | `isSpecialOrderable`(39755), ` 2048`(36117), ` Billboard`(36229), ` psychedel`(27477), ` Authorization`(35263) |
| L3H3 | sink falls | rare | 0.96 | 0.021 / 0.025 | 0.005 | `erson`(882), `man`(805), `iff`(733), `ill`(359), `il`(346) | `\n\n`(628), `�`(175), `�`(173), `."`(526), `�`(174) |
| L3H3 | sink falls | common | 0.83 | 0.023 / 0.028 | 0.003 | ` potions`(28074), ` Dungeon`(11995), ` Eternal`(21475), ` Tumblr`(24434), ` perk`(34407) | `isSpecialOrderable`(39755), ` <[`(29342), ` comprising`(27918), ` 2048`(36117), `_{`(23330) |
| L6H4 | sink falls | rare | 0.89 | 0.021 / 0.027 | 0.004 | ` mod`(953), `ually`(935), `ally`(453), `and`(392), `ly`(306) | `{`(90), ` \|`(930), `,"`(553), ` $`(720), ` add`(751) |
| L6H4 | sink falls | common | 0.91 | 0.080 / 0.067 | 0.010 | `Redditor`(34832), `ocytes`(30309), ` bacterial`(23462), ` <[`(29342), ` Arduino`(27634) | `isSpecialOrderable`(39755), ` ng`(23370), ` ACTIONS`(23054), `tackle`(36346), ` Viktor`(31096) |
| L6H8 | sink falls | rare | 0.91 | 0.028 / 0.029 | 0.005 | `In`(818), `The`(464), `ating`(803), `uring`(870), ` still`(991) | ` [`(685), ` (`(357), ` 0`(657), `_`(62), ` This`(770) |
| L6H8 | sink falls | common | 0.86 | 0.042 / 0.040 | 0.007 | `CHAT`(31542), `Minecraft`(39194), ` Posts`(12043), `isSpecialOrderable`(39755), `ooo`(34160) | ` comprising`(27918), `_{`(23330), ` oxidation`(37767), ` neuronal`(36347), ` bacterial`(23462) |
| L7H0 | sink falls | rare | 0.81 | 0.029 / 0.037 | 0.006 | `(`(7), `[`(58), ` you`(345), ` our`(674), ` your`(534) | ` 0`(657), `@`(31), `:`(25), ` 19`(678), `\n\n`(628) |
| L7H0 | sink falls | common | 0.84 | 0.051 / 0.073 | 0.009 | ` Mahjong`(37380), `Minecraft`(39194), `龍�`(39820), `abulary`(22528), ` Goals`(28510) | `"(`(18109), `](`(16151), ` <[`(29342), `!"`(2474), `?),`(33924) |
| L10H9 | sink falls | rare | 0.91 | 0.027 / 0.028 | 0.005 | `)`(8), ` still`(991), `arch`(998), `com`(785), `port`(634) | `\n\n`(628), `;`(26), `�`(159), `).`(737), ` "`(366) |
| L10H9 | sink falls | common | 0.93 | 0.103 / 0.117 | 0.017 | `linux`(23289), `isSpecialOrderable`(39755), `utsche`(30433), `Michael`(13256), `ategories`(26129) | ` \\"`(19990), `′`(17478), ` therein`(27258), ` comprising`(27918), `【`(31854) |
| L4H0 | sink falls | rare | 0.98 | 0.038 / 0.040 | 0.006 | `\x1b`(215), `�`(184), `\x7f`(221), `�`(186), `\x1c`(216) | `.`(13), `,"`(553), `."`(526), `).`(737), `—`(960) |
| L4H0 | sink falls | common | 0.92 | 0.053 / 0.058 | 0.009 | ` adject`(31129), ` syll`(27226), ` Meaning`(30563), ` eagle`(31176), ` gobl`(29646) | `isSpecialOrderable`(39755), `ocytes`(30309), ` postseason`(22905), ` Cavs`(38943), `Prosecutors`(39401) |
| L5H2 | sink falls | rare | 0.97 | 0.055 / 0.061 | 0.008 | `ative`(876), `erson`(882), `\x1b`(215), `\t`(197), `�`(183) | ` (`(357), ` [`(685), `(`(7), ` "`(366), `[`(58) |
| L5H2 | sink falls | common | 0.89 | 0.066 / 0.073 | 0.010 | `linux`(23289), `Code`(10669), `abulary`(22528), `issors`(32555), `focus`(37635) | ` <[`(29342), `"(`(18109), `【`(31854), `.[`(3693), ` \\"`(19990) |
| L5H6 | sink falls | rare | 0.88 | 0.030 / 0.024 | 0.004 | ` mod`(953), ` $`(720), ` game`(983), ` ass`(840), ` app`(598) | `—`(960), `."`(526), `.`(13), `,"`(553), `(`(7) |
| L5H6 | sink falls | common | 0.77 | 0.040 / 0.034 | 0.006 | ` ;;`(36792), ` \u200e`(24398), ` filesystem`(29905), ` Unix`(33501), ` Haskell`(25271) | ` embryos`(39966), ` patients`(3871), `ocytes`(30309), `Prosecutors`(39401), `probably`(26949) |
| L8H2 | sink falls | rare | 0.96 | 0.067 / 0.072 | 0.011 | `\x12`(206), `\x14`(208), `\x03`(191), `\x16`(210), `\x11`(205) | `(`(7), `[`(58), ` [`(685), ` (`(357), `{`(90) |
| L8H2 | sink falls | common | 0.95 | 0.135 / 0.148 | 0.020 | `Redditor`(34832), `Minecraft`(39194), ` Haskell`(25271), `linux`(23289), ` eSports`(32717) | ` \\"`(19990), `\xa0\xa0\xa0`(33477), ` \\\\`(26867), `Â`(5523), `§`(16273) |
| L9H8 | sink falls | rare | 0.96 | 0.069 / 0.070 | 0.011 | `ious`(699), `ween`(975), ` ass`(840), ` you`(345), `ick`(624) | `�`(159), `�`(145), `�`(167), `�`(150), `�`(161) |
| L9H8 | sink falls | common | 0.97 | 0.249 / 0.228 | 0.033 | ` Arkansas`(14538), `Redditor`(34832), ` Massachusetts`(10140), `ichever`(22617), ` cock`(7540) | ` Kurdistan`(29439), ` Catalonia`(33859), ` Aviv`(28890), `ña`(30644), `י�`(33951) |
| L0H6 | label mix only | rare | 0.98 | 0.002 / 0.002 | 0.000 | `'t`(470), ` not`(407), ` do`(466), ` have`(423), ` would`(561) | `�`(143), `�`(154), ` differe`(980), `�`(132), `�`(172) |
| L0H6 | label mix only | common | 0.95 | 0.007 / 0.007 | 0.001 | `16`(1433), ` put`(1234), `18`(1507), `ying`(1112), `ari`(2743) | `龍�`(39820), `sonian`(35202), ` ACTIONS`(23054), `Redditor`(34832), ` enthusi`(11273) |
| L1H0 | label mix only | rare | 0.98 | 0.001 / 0.002 | 0.000 | `ig`(328), `�`(223), ` prov`(899), `ro`(305), `ch`(354) | `."`(526), `�`(144), `,"`(553), `!`(0), `?`(30) |
| L1H0 | label mix only | common | 0.95 | 0.006 / 0.006 | 0.001 | `prof`(5577), ` gn`(19967), `р`(21169), `perm`(16321), `rod`(14892) | `isSpecialOrderable`(39755), `Trivia`(23854), `Returns`(35561), ` ACTIONS`(23054), `inventoryQuantity`(39756) |
| L3H2 | label mix only | rare | 0.95 | 0.012 / 0.010 | 0.002 | ` game`(983), `cess`(919), `ife`(901), `vern`(933), `mer`(647) | `."`(526), `\n\n`(628), ` "`(366), `—`(960), `,"`(553) |
| L3H2 | label mix only | common | 0.92 | 0.024 / 0.021 | 0.003 | ` nylon`(36104), ` Coffee`(19443), ` edible`(35988), ` tuna`(38883), ` Corbyn`(15673) | `isSpecialOrderable`(39755), ` 2048`(36117), ` therein`(27258), ` ;;`(36792), `_{`(23330) |
| L10H0 | label mix only | rare | 0.96 | 0.076 / 0.093 | 0.014 | ` his`(465), `The`(464), `ious`(699), ` game`(983), ` those`(883) | `�`(173), `�`(170), `�`(153), `�`(176), `�`(171) |
| L10H0 | label mix only | common | 0.97 | 0.151 / 0.161 | 0.021 | `utsche`(30433), `razy`(5918), `Political`(35443), `izarre`(12474), ` somebody`(8276) | ` ;;`(36792), `\xa0\xa0\xa0`(33477), `Â`(5523), `י`(25529), ` >>>`(13163) |
| L0H1 | control | rare | 0.41 | 0.000 / 0.000 | 0.000 | `-`(12), `q`(80), `g`(70), `z`(89), `.`(13) | `ful`(913), ` use`(779), ` how`(703), `�`(132), ` whe`(483) |
| L0H1 | control | common | 0.20 | 0.000 / 0.000 | 0.000 | `173`(25399), ` criticized`(12318), `263`(29558), `559`(38605), `ari`(2743) | ` Stew`(12194), ` gets`(3011), ` Getting`(18067), ` Set`(5345), `AGE`(11879) |
| L4H11 | control | rare | 0.70 | 0.000 / 0.000 | 0.000 | `\x08`(196), `�`(184), `\x1f`(219), `\x1b`(215), `\x10`(204) | `,"`(553), `."`(526), `.`(13), ` "`(366), `;`(26) |
| L4H11 | control | common | 0.12 | 0.000 / 0.000 | 0.000 | ` Arduino`(27634), ` happ`(1147), ` troll`(13278), `Python`(37906), ` python`(21015) | `isSpecialOrderable`(39755), `.[`(3693), ` portrayed`(19152), ` comprising`(27918), ` enacted`(17814) |
| L5H1 | control | rare | 0.69 | 0.004 / 0.005 | 0.001 | `}`(92), `M`(44), ` mod`(953), `N`(45), `A`(32) | ` (`(357), `(`(7), `uring`(870), ` "`(366), ` so`(523) |
| L5H1 | control | common | 0.47 | 0.001 / 0.005 | 0.000 | `isSpecialOrderable`(39755), ` <[`(29342), `!"`(2474), `Redditor`(34832), `?),`(33924) | ` Apple`(4196), ` adject`(31129), ` substance`(9136), ` 09`(7769), ` imperson`(28671) |

## 7. Which token properties explain the final-token effect?

OLS of each head's per-token mean sink_mass (final-token sweep, rare bases) on token features; coefficient [95% CI]. log_prior = the model's mean log p(token | base) for the final slot, a frequency/familiarity proxy. Features are correlated, so read signs and CIs, not magnitudes, and do not over-read single heads.

| head | R² | is_alpha | is_byte | is_capitalized | is_digit | is_punct | leading_space | log_id | log_prior | n_chars |
|---|---|---|---|---|---|---|---|---|---|---|
| L1H5 | 0.64 | **+0.050** [+0.024, +0.076] | **-0.041** [-0.054, -0.028] | **+0.034** [+0.029, +0.039] | **+0.038** [+0.010, +0.065] | +0.017 [-0.009, +0.042] | **-0.062** [-0.067, -0.058] | **-0.014** [-0.016, -0.012] | **+0.019** [+0.017, +0.020] | **-0.001** [-0.003, -0.000] |
| L1H7 | 0.60 | **+0.068** [+0.050, +0.087] | **-0.039** [-0.048, -0.030] | **-0.009** [-0.013, -0.006] | **+0.068** [+0.048, +0.088] | **+0.042** [+0.023, +0.060] | **-0.068** [-0.071, -0.065] | **-0.013** [-0.014, -0.011] | **+0.004** [+0.003, +0.006] | **+0.001** [+0.000, +0.002] |
| L7H7 | 0.36 | **-0.158** [-0.216, -0.100] | **-0.109** [-0.137, -0.080] | **+0.064** [+0.053, +0.075] | **-0.163** [-0.225, -0.102] | **+0.076** [+0.019, +0.133] | **+0.016** [+0.006, +0.026] | -0.003 [-0.008, +0.002] | **+0.014** [+0.010, +0.018] | **-0.018** [-0.020, -0.015] |
| L9H11 | 0.43 | **-0.077** [-0.153, -0.001] | **-0.043** [-0.080, -0.005] | **+0.160** [+0.145, +0.175] | **-0.141** [-0.222, -0.060] | **+0.081** [+0.006, +0.156] | **+0.106** [+0.092, +0.119] | -0.004 [-0.011, +0.002] | **+0.010** [+0.005, +0.015] | **-0.037** [-0.040, -0.034] |
| L10H11 | 0.33 | +0.076 [-0.016, +0.168] | -0.030 [-0.075, +0.016] | **+0.078** [+0.061, +0.096] | **-0.134** [-0.231, -0.036] | +0.084 [-0.007, +0.174] | **+0.158** [+0.142, +0.174] | -0.006 [-0.014, +0.002] | **+0.010** [+0.004, +0.016] | **-0.041** [-0.045, -0.037] |
| L11H1 | 0.40 | **+0.169** [+0.072, +0.267] | -0.043 [-0.091, +0.005] | **+0.149** [+0.130, +0.168] | -0.075 [-0.178, +0.029] | +0.077 [-0.019, +0.172] | **+0.139** [+0.122, +0.156] | **-0.016** [-0.024, -0.007] | **+0.009** [+0.002, +0.015] | **-0.049** [-0.053, -0.045] |
| L11H3 | 0.33 | **+0.102** [+0.011, +0.192] | -0.017 [-0.062, +0.028] | **+0.113** [+0.096, +0.131] | -0.076 [-0.172, +0.020] | +0.029 [-0.060, +0.119] | **+0.149** [+0.133, +0.165] | **-0.015** [-0.023, -0.007] | **-0.012** [-0.018, -0.006] | **-0.046** [-0.050, -0.043] |
| L2H1 | 0.64 | +0.040 [-0.017, +0.097] | **-0.062** [-0.090, -0.034] | **-0.107** [-0.118, -0.096] | +0.002 [-0.059, +0.062] | **-0.063** [-0.119, -0.007] | **-0.192** [-0.202, -0.182] | **-0.047** [-0.052, -0.042] | **-0.041** [-0.045, -0.037] | **-0.016** [-0.018, -0.013] |
| L2H6 | 0.42 | **+0.114** [+0.096, +0.133] | +0.003 [-0.006, +0.013] | **-0.036** [-0.039, -0.032] | +0.009 [-0.010, +0.029] | +0.017 [-0.001, +0.035] | **-0.016** [-0.019, -0.013] | **-0.007** [-0.009, -0.005] | **-0.002** [-0.003, -0.000] | **-0.002** [-0.003, -0.001] |
| L3H6 | 0.44 | **-0.061** [-0.075, -0.047] | **-0.029** [-0.036, -0.022] | **+0.042** [+0.040, +0.045] | **-0.025** [-0.039, -0.010] | -0.011 [-0.025, +0.002] | **+0.032** [+0.030, +0.035] | **-0.002** [-0.003, -0.001] | +0.001 [-0.000, +0.002] | **-0.001** [-0.001, -0.000] |
| L7H11 | 0.57 | **-0.165** [-0.238, -0.092] | **-0.092** [-0.128, -0.056] | **+0.170** [+0.156, +0.184] | **-0.114** [-0.191, -0.037] | **+0.080** [+0.009, +0.152] | **+0.119** [+0.106, +0.132] | **-0.010** [-0.016, -0.004] | **+0.020** [+0.015, +0.025] | **-0.043** [-0.046, -0.040] |
| L10H4 | 0.32 | **+0.143** [+0.046, +0.240] | -0.021 [-0.069, +0.028] | **+0.098** [+0.079, +0.117] | -0.067 [-0.170, +0.036] | +0.094 [-0.001, +0.190] | **+0.129** [+0.112, +0.147] | -0.004 [-0.012, +0.005] | **+0.007** [+0.001, +0.014] | **-0.047** [-0.051, -0.043] |
| L11H9 | 0.41 | **+0.105** [+0.017, +0.193] | -0.004 [-0.047, +0.040] | **+0.131** [+0.114, +0.148] | **-0.125** [-0.218, -0.032] | +0.084 [-0.002, +0.170] | **+0.161** [+0.145, +0.176] | -0.004 [-0.012, +0.004] | **+0.022** [+0.016, +0.028] | **-0.037** [-0.040, -0.033] |
| L1H8 | 0.24 | **+0.037** [+0.024, +0.050] | **+0.016** [+0.010, +0.022] | **+0.008** [+0.006, +0.011] | **+0.031** [+0.017, +0.044] | **+0.036** [+0.023, +0.048] | **+0.008** [+0.006, +0.010] | **+0.009** [+0.008, +0.010] | **+0.002** [+0.001, +0.002] | **-0.001** [-0.002, -0.001] |
| L2H0 | 0.35 | **-0.081** [-0.102, -0.060] | **-0.017** [-0.028, -0.007] | **-0.014** [-0.018, -0.010] | **-0.095** [-0.117, -0.072] | **-0.053** [-0.074, -0.032] | **+0.046** [+0.042, +0.049] | **-0.002** [-0.004, -0.001] | **-0.011** [-0.012, -0.009] | +0.001 [-0.000, +0.002] |
| L2H4 | 0.59 | **-0.139** [-0.195, -0.083] | **+0.038** [+0.010, +0.066] | **+0.066** [+0.056, +0.077] | **-0.087** [-0.147, -0.028] | **-0.088** [-0.143, -0.033] | **+0.230** [+0.220, +0.240] | **+0.018** [+0.013, +0.023] | **-0.015** [-0.019, -0.011] | **-0.003** [-0.005, -0.000] |
| L3H3 | 0.82 | **-0.265** [-0.320, -0.211] | -0.015 [-0.042, +0.012] | **+0.105** [+0.095, +0.116] | **-0.099** [-0.157, -0.041] | **-0.068** [-0.122, -0.015] | **+0.422** [+0.412, +0.432] | **+0.021** [+0.016, +0.025] | -0.000 [-0.004, +0.003] | **+0.010** [+0.008, +0.013] |
| L6H4 | 0.50 | -0.039 [-0.084, +0.005] | +0.006 [-0.016, +0.028] | **-0.064** [-0.073, -0.056] | -0.017 [-0.064, +0.030] | **-0.101** [-0.145, -0.058] | **-0.069** [-0.077, -0.062] | **-0.016** [-0.020, -0.012] | **-0.042** [-0.045, -0.039] | **-0.013** [-0.015, -0.012] |
| L6H8 | 0.42 | **+0.128** [+0.079, +0.176] | **+0.081** [+0.056, +0.105] | **+0.039** [+0.029, +0.048] | -0.015 [-0.066, +0.037] | +0.014 [-0.034, +0.062] | **+0.148** [+0.139, +0.157] | **+0.009** [+0.005, +0.013] | +0.001 [-0.002, +0.004] | **-0.010** [-0.012, -0.008] |
| L7H0 | 0.31 | -0.027 [-0.067, +0.013] | -0.020 [-0.039, +0.000] | **+0.093** [+0.085, +0.101] | -0.017 [-0.059, +0.026] | **+0.043** [+0.004, +0.083] | **+0.053** [+0.046, +0.060] | **+0.006** [+0.003, +0.010] | -0.002 [-0.004, +0.001] | -0.001 [-0.003, +0.000] |
| L10H9 | 0.42 | +0.015 [-0.015, +0.045] | +0.007 [-0.008, +0.022] | **+0.048** [+0.042, +0.053] | +0.017 [-0.015, +0.049] | +0.024 [-0.006, +0.054] | **+0.098** [+0.093, +0.104] | +0.000 [-0.003, +0.003] | **-0.003** [-0.005, -0.001] | **-0.004** [-0.005, -0.002] |
| L4H0 | 0.41 | **-0.125** [-0.178, -0.072] | -0.020 [-0.047, +0.006] | **+0.060** [+0.050, +0.071] | +0.001 [-0.055, +0.058] | -0.044 [-0.097, +0.008] | **+0.116** [+0.107, +0.126] | **+0.011** [+0.006, +0.015] | **-0.018** [-0.022, -0.014] | **-0.003** [-0.005, -0.001] |
| L5H2 | 0.20 | **+0.048** [+0.019, +0.077] | **+0.043** [+0.028, +0.057] | **+0.035** [+0.029, +0.040] | +0.001 [-0.030, +0.032] | -0.019 [-0.047, +0.010] | **-0.014** [-0.019, -0.009] | +0.001 [-0.001, +0.004] | **-0.003** [-0.005, -0.001] | **-0.009** [-0.010, -0.007] |
| L5H6 | 0.28 | **-0.100** [-0.141, -0.059] | **-0.097** [-0.117, -0.076] | **+0.082** [+0.074, +0.090] | **+0.091** [+0.047, +0.134] | -0.006 [-0.046, +0.035] | **+0.045** [+0.038, +0.052] | **-0.005** [-0.008, -0.001] | -0.000 [-0.003, +0.002] | +0.001 [-0.001, +0.003] |
| L8H2 | 0.40 | +0.043 [-0.019, +0.105] | -0.005 [-0.036, +0.026] | **+0.065** [+0.053, +0.077] | -0.056 [-0.121, +0.010] | -0.013 [-0.074, +0.048] | **+0.016** [+0.005, +0.027] | **-0.012** [-0.017, -0.007] | **-0.027** [-0.031, -0.023] | **-0.041** [-0.043, -0.038] |
| L9H8 | 0.07 | **+0.096** [+0.045, +0.147] | +0.021 [-0.004, +0.046] | **-0.012** [-0.022, -0.002] | **+0.062** [+0.008, +0.116] | +0.019 [-0.031, +0.068] | **+0.014** [+0.005, +0.023] | **+0.009** [+0.005, +0.014] | **-0.009** [-0.012, -0.005] | **-0.013** [-0.015, -0.011] |
| L0H6 | 0.56 | -0.004 [-0.014, +0.006] | **+0.011** [+0.006, +0.016] | **+0.007** [+0.005, +0.009] | **-0.016** [-0.027, -0.006] | -0.001 [-0.011, +0.008] | **+0.008** [+0.006, +0.010] | **+0.002** [+0.001, +0.003] | **-0.010** [-0.010, -0.009] | **-0.002** [-0.003, -0.002] |
| L1H0 | 0.25 | **-0.021** [-0.027, -0.015] | **-0.012** [-0.015, -0.009] | **+0.012** [+0.011, +0.013] | **-0.013** [-0.019, -0.006] | +0.005 [-0.001, +0.011] | **+0.006** [+0.005, +0.007] | -0.000 [-0.001, +0.000] | +0.000 [-0.000, +0.000] | +0.000 [-0.000, +0.000] |
| L3H2 | 0.36 | **-0.044** [-0.064, -0.025] | **-0.058** [-0.068, -0.049] | **+0.056** [+0.052, +0.060] | **-0.040** [-0.061, -0.020] | +0.006 [-0.013, +0.025] | **+0.023** [+0.019, +0.026] | -0.001 [-0.003, +0.001] | **+0.004** [+0.003, +0.005] | -0.000 [-0.001, +0.001] |
| L10H0 | 0.37 | **+0.214** [+0.149, +0.279] | -0.004 [-0.036, +0.029] | **+0.063** [+0.051, +0.076] | -0.024 [-0.093, +0.044] | **+0.146** [+0.083, +0.210] | **+0.112** [+0.101, +0.124] | **+0.015** [+0.009, +0.020] | **+0.016** [+0.012, +0.021] | **-0.027** [-0.029, -0.024] |
| L0H1 | 0.11 | **-0.008** [-0.008, -0.007] | **+0.001** [+0.001, +0.001] | **+0.000** [+0.000, +0.000] | **-0.007** [-0.008, -0.007] | **-0.007** [-0.008, -0.006] | -0.000 [-0.000, +0.000] | **+0.000** [+0.000, +0.000] | **+0.000** [+0.000, +0.000] | **+0.000** [+0.000, +0.000] |
| L4H11 | 0.47 | **-0.018** [-0.021, -0.015] | **-0.003** [-0.004, -0.002] | **-0.002** [-0.003, -0.001] | **-0.019** [-0.022, -0.015] | **-0.022** [-0.024, -0.019] | **+0.003** [+0.003, +0.004] | **-0.003** [-0.003, -0.003] | **-0.004** [-0.004, -0.003] | **-0.001** [-0.001, -0.001] |
| L5H1 | 0.49 | **+0.098** [+0.081, +0.115] | **+0.012** [+0.003, +0.021] | **+0.024** [+0.020, +0.027] | **+0.108** [+0.089, +0.126] | **+0.132** [+0.114, +0.149] | **-0.014** [-0.017, -0.011] | **+0.018** [+0.016, +0.019] | **+0.023** [+0.022, +0.024] | **+0.003** [+0.003, +0.004] |

## 8. Observational (exp1 data): which key tokens absorb attention?

Mean log(p_k / p_sink) over each token's occurrences as a key (positions 1–31) in exp1's B and C sequences. The log-ratio cancels the softmax normalizer, so unlike raw p_k it is not pushed around by the other keys.

| head | condition | top by log(p_k/p_sink) |
|---|---|---|
| L1H5 | B_common | `,`(11), ` and`(290), `'s`(338), `."`(526), `:`(25) |
| L1H5 | C_common_clean | ` and`(290), `'s`(338), ` the`(262), `and`(392), `."`(526) |
| L1H7 | B_common | `�`(174), `�`(124), `\x0f`(203), `\x1c`(216), `�`(179) |
| L1H7 | C_common_clean | ` game`(983), ` report`(989), ` app`(598), `,"`(553), `==`(855) |
| L7H7 | B_common | `�`(154), `�`(175), `�`(172), `�`(182), `."`(526) |
| L7H7 | C_common_clean | ` 19`(678), ` $`(720), ` =`(796), ` want`(765), ` 5`(642) |
| L9H11 | B_common | ` –`(784), `—`(960), ` —`(851), `...`(986), `*`(9) |
| L9H11 | C_common_clean | ` =`(796), `�`(447), `..`(492), ` —`(851), ` –`(784) |
| L10H11 | B_common | `�`(154), ` report`(989), ` people`(661), `ments`(902), `�`(153) |
| L10H11 | C_common_clean | ` report`(989), ` people`(661), ` mod`(953), ` work`(670), ` act`(719) |
| L11H1 | B_common | ` own`(898), `self`(944), ` He`(679), ` her`(607), ` co`(763) |
| L11H1 | C_common_clean | ` him`(683), ` co`(763), ` man`(582), ` people`(661), ` own`(898) |
| L11H3 | B_common | ` game`(983), `--`(438), `(`(7), ` (`(357), `),`(828) |
| L11H3 | C_common_clean | ` work`(670), ` report`(989), ` game`(983), ` mod`(953), ` need`(761) |
| L2H1 | B_common | `�`(146), `_`(62), `�`(152), `$`(3), `�`(174) |
| L2H1 | C_common_clean | `—`(960), ` –`(784), ` $`(720), ` —`(851), `..`(492) |
| L2H6 | B_common | `."`(526), `).`(737), `\n`(198), `,"`(553), `.`(13) |
| L2H6 | C_common_clean | `."`(526), `).`(737), `),`(828), `,"`(553), ` .`(764) |
| L3H6 | B_common | ` 1`(352), ` N`(399), ` J`(449), ` fl`(781), ` We`(775) |
| L3H6 | C_common_clean | ` E`(412), ` Re`(797), ` J`(449), ` Tr`(833), ` Sh`(911) |
| L7H11 | B_common | `�`(175), `�`(181), `�`(176), `\x1c`(216), ` Sh`(911) |
| L7H11 | C_common_clean | ` Sh`(911), ` We`(775), ` Com`(955), ` And`(843), ` own`(898) |
| L10H4 | B_common | `{`(90), `*`(9), `"`(1), `."`(526), ` "`(366) |
| L10H4 | C_common_clean | `."`(526), ` "`(366), `==`(855), `--------`(982), ` '`(705) |
| L11H9 | B_common | `�`(154), `�`(155), `\x19`(213), `�`(153), `�`(181) |
| L11H9 | C_common_clean | `==`(855), `--------`(982), ` want`(765), ` use`(779), ` inv`(800) |
| L1H8 | B_common | `."`(526), `).`(737), `\n`(198), `,"`(553), `.`(13) |
| L1H8 | C_common_clean | `."`(526), `).`(737), ` the`(262), ` ,`(837), ` an`(281) |
| L2H0 | B_common | `@`(31), ` on`(319), ` 1`(352), ` 4`(604), ` from`(422) |
| L2H0 | C_common_clean | ` at`(379), ` if`(611), ` about`(546), ` by`(416), ` In`(554) |
| L2H4 | B_common | `.`(13), `."`(526), ` the`(262), ` and`(290), ` am`(716) |
| L2H4 | C_common_clean | `).`(737), ` the`(262), `."`(526), ` and`(290), `),`(828) |
| L3H3 | B_common | ` fl`(781), `meric`(946), ` point`(966), `\x12`(206), ` inv`(800) |
| L3H3 | C_common_clean | ` app`(598), ` if`(611), ` dec`(875), ` Re`(797), ` me`(502) |
| L6H4 | B_common | `�`(144), `us`(385), `�`(131), `�`(145), `�`(147) |
| L6H4 | C_common_clean | `um`(388), `us`(385), `ich`(488), `ah`(993), `ss`(824) |
| L6H8 | B_common | ` sub`(850), ` fl`(781), `�`(182), `�`(170), `[`(58) |
| L6H8 | C_common_clean | ` sh`(427), ` me`(502), ` sc`(629), ` O`(440), ` app`(598) |
| L7H0 | B_common | `."`(526), ` Un`(791), ` because`(780), ` de`(390), ` good`(922) |
| L7H0 | C_common_clean | `."`(526), ` des`(748), ` un`(555), ` very`(845), `).`(737) |
| L10H9 | B_common | ` been`(587), ` show`(905), ` those`(883), ` de`(390), ` because`(780) |
| L10H9 | C_common_clean | ` they`(484), ` about`(546), ` been`(587), ` their`(511), ` we`(356) |
| L4H0 | B_common | ` has`(468), `�`(176), `�`(170), ` int`(493), ` from`(422) |
| L4H0 | C_common_clean | ` about`(546), ` does`(857), ` if`(611), ` des`(748), ` It`(632) |
| L5H2 | B_common | `�`(170), `�`(143), `[`(58), `{`(90), `�`(136) |
| L5H2 | C_common_clean | ` me`(502), ` if`(611), ` im`(545), ` int`(493), ` sa`(473) |
| L5H6 | B_common | `\x06`(194), `\x19`(213), `�`(176), `\x0b`(199), `\x16`(210) |
| L5H6 | C_common_clean | ` It`(632), ` many`(867), ` own`(898), ` des`(748), ` pres`(906) |
| L8H2 | B_common | `\|`(91), `*`(9), `�`(170), `$`(3), `{`(90) |
| L8H2 | C_common_clean | `'t`(470), `In`(818), `es`(274), ` de`(390), `ue`(518) |
| L9H8 | B_common | `�`(145), `�`(144), `�`(131), `�`(157), `�`(150) |
| L9H8 | C_common_clean | `ang`(648), `ich`(488), `uch`(794), `um`(388), `ah`(993) |
| L0H6 | B_common | `�`(184), `\t`(197), `\x12`(206), `\x7f`(221), `\x02`(190) |
| L0H6 | C_common_clean | ` includ`(846), ` differe`(980), ` bec`(639), ` wor`(476), ` produ`(990) |
| L1H0 | B_common | ` fl`(781), `ial`(498), `\x06`(194), ` am`(716), ` y`(331) |
| L1H0 | C_common_clean | ` br`(865), ` Sh`(911), ` Re`(797), ` ad`(512), ` inter`(987) |
| L3H2 | B_common | `�`(143), ` int`(493), ` Un`(791), ` ag`(556), ` a`(257) |
| L3H2 | C_common_clean | ` his`(465), ` the`(262), ` app`(598), ` Re`(797), ` des`(748) |
| L10H0 | B_common | `�`(153), `�`(145), `�`(175), `�`(173), `�`(154) |
| L10H0 | C_common_clean | `�`(447), `==`(855), `--------`(982), ` �`(564), `----`(650) |
| L0H1 | B_common | `�`(187), ` stud`(941), `\x19`(213), ` bel`(894), `�`(170) |
| L0H1 | C_common_clean | ` tw`(665), ` fe`(730), ` n`(299), ` su`(424), ` wh`(348) |
| L4H11 | B_common | `�`(182), `ular`(934), `\x16`(210), `meric`(946), `\x12`(206) |
| L4H11 | C_common_clean | ` app`(598), ` dec`(875), ` also`(635), ` well`(880), `pp`(381) |
| L5H1 | B_common | `\x04`(192), `�`(150), `�`(176), `\x19`(213), `\x15`(209) |
| L5H1 | C_common_clean | ` This`(770), ` Un`(791), ` cons`(762), ` Ar`(943), `."`(526) |

## Figures

- `figures/stage1_interaction_*.png`, `figures/stage1_decomposition_*.png`, `figures/stage1_dose_*.png`: the 2×2 and dose-response
- `figures/tokens/<head>_p<position>_<metric>.png`: per-token graphs (the final-token sweep is position 32; substitutions at [31, 16])
- `figures/stage2_token_head_heatmap_p*.png`: tokens with the largest effects across tracked heads
- `figures/overlay/<head>.png`: exp1 sequences with each token shaded by the attention it receives
- `figures/received/<head>_<condition>.png`: attention received per key token (exp1 data)

## Caveats

- A substitution at position p < 32 changes positions p..32, so it measures the token's total effect: as a key and through earlier heads' writes into later positions. Only the final-token sweep isolates a single causal route (the query).
- Per-token means average over bases of one class; a token's effect can differ between rare and common bases, which is why both are shown.
- Tracked heads were chosen post hoc on exp1 data; the replication line above measures that on fresh sequences.
