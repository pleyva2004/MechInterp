# Experiment 3, E3.4: repeated phrase, induction as the positive control

Run `20261015-211838_exp3e34`. TL attention at the query (position 32); HF cross-check on 25 rows per base class. 500 rows per base class; 8 variants × 12 unit(s) per row (sushi, water, pizza, music, coffee, chess, rain, cars, songs, rice, bikes, money). Layout: {'phrase': [10, 11, 12], 'previous': 31, 'query': 32, 'dup': 11, 'copy': 12}. Late = layers 6–11, the mean over their heads. Attention is averaged over a row's units before masses and labels; masks come from each contrast's treated arm (placebo: the masks of repeat-rand).

Extraction: 96000 sequences in 596 s (161 seq/s, TL only).

## Checks

| check | result |
|---|---|
| design (base outside written positions; filler holds no written id; no BOS; repeats only where declared and present in every row) | PASS (asserted per variant and base class) |
| E3.4 layout (controls differ only at 10–12, repeat-nl-1 only at 31; ids as in the table; other-nl shares no id) | PASS (asserted per base class) |
| attention rows sum to 1 | PASS (atol 0.0001, asserted per batch; max |sum − 1| 4.6e-07) |
| determinism, rare (20 rows) | PASS (max |Δ| 0.0e+00) |
| TL vs HF, rare (25 rows, 2400 sequences) | PASS (max |Δ| 3.6e-05, argmax-key agreement 100.0000%) |
| placebo switching heads ≤ 1, rare | PASS (0 argmax, 0 mass) |
| L5H5 and L6H9 among induction heads, rare | PASS (10 induction heads; L5H5 in (0.88), L6H9 in (0.81)) |
| determinism, common (20 rows) | PASS (max |Δ| 0.0e+00) |
| TL vs HF, common (25 rows, 2400 sequences) | PASS (max |Δ| 2.7e-05, argmax-key agreement 100.0000%) |
| placebo switching heads ≤ 1, common | PASS (0 argmax, 0 mass) |
| L5H5 and L6H9 among induction heads, common | PASS (4 induction heads; L5H5 in (0.79), L6H9 in (0.70)) |

## Pre-registered predictions (exp3/PREDICTIONS_e34.md)

| id | verdict | detail |
|---|---|---|
| E4a | PASS | rare: 7 heads in L5-7 switch Sink→Copy [L5H0 (+0.87), L5H1 (+0.98), L5H5 (+1.00), L6H9 (+1.00), L7H1 (+0.49), L7H2 (+0.42), L7H10 (+1.00)]; placebo 0; beyond placebo 7 (need ≥ 3); common: 5 heads in L5-7 switch Sink→Copy [L5H0 (+0.46), L5H1 (+0.65), L5H5 (+1.00), L6H9 (+1.00), L7H10 (+1.00)]; placebo 0; beyond placebo 5 (need ≥ 3) |
| E4b | PASS | rare: excess +0.0853 [+0.0831, +0.0875] over 10 induction heads; mean BOS loss +0.5056, Copy gain +0.5278; common: excess +0.0401 [+0.0383, +0.0417] over 4 induction heads; mean BOS loss +0.6279, Copy gain +0.6631 |
| E4c | FAIL | rare: -0.1062, one-sided 95% bound -0.1099 (need > -0.050); per phrase -0.1062 [-0.1316, -0.0811], 0/12 > 0; alongside nl_vs_rand: -0.0955 [-0.1002, -0.0906]; common: -0.1791, one-sided 95% bound -0.1828 (need > -0.050); per phrase -0.1791 [-0.2063, -0.1526], 0/12 > 0; alongside nl_vs_rand: -0.0857 [-0.0917, -0.0796] |
| E4d | FAIL | rare: 3/10 listed heads lose BOS (meaningful, Δ<0) [L7H0 +0.030, L6H0 -0.090*, L8H7 +0.023, L7H11 +0.043, L8H5 +0.027, L6H11 -0.144*, L9H3 -0.003, L6H8 -0.358*, L10H2 +0.078, L8H11 +0.036]; placebo 0; null 95th pct 9.0 over 1000 sets of 10 from 62 other late heads (null mean 7.01); common: 4/10 listed heads lose BOS (meaningful, Δ<0) [L7H0 +0.054, L6H0 -0.062*, L8H7 +0.060, L7H11 +0.085, L8H5 -0.104*, L6H11 -0.128*, L9H3 +0.013, L6H8 -0.307*, L10H2 +0.051, L8H11 +0.088]; placebo 0; null 95th pct 9.0 over 1000 sets of 10 from 62 other late heads (null mean 7.01) |
| E4e | PASS | rare: +0.2289 [+0.2246, +0.2336]; per phrase +0.2289 [+0.2002, +0.2622], 12/12 > 0; common: +0.2134 [+0.2085, +0.2184]; per phrase +0.2134 [+0.1768, +0.2513], 12/12 > 0 |

## Induction heads (mean copy-target mass ≥ 0.3 at key 12 in repeat-rand)

| base | count | heads (mean copy-target mass) | L5H5 | L6H9 |
|---|---|---|---|---|
| rare | 10 | L5H5 0.88, L6H9 0.81, L5H1 0.74, L7H10 0.63, L10H7 0.43, L7H2 0.39, L10H1 0.38, L5H0 0.38, L9H6 0.36, L9H9 0.35 | yes (0.88) | yes (0.81) |
| common | 4 | L5H5 0.79, L6H9 0.70, L7H10 0.64, L5H1 0.53 | yes (0.79) | yes (0.70) |

## Late pooled per-row Δ (a − b), every contrast and measure

95% / 90% CIs bootstrap rows; verdict: positive / negative (95% CI excludes 0), equivalent (TOST, 90% CI inside ±0.005), else inconclusive. Per phrase: bootstrap over the per-unit means, with a sign count.

| base | contrast | measure | Δ | 95% CI | 90% CI | Wilcoxon p | verdict | per unit [95% CI] | units > 0 |
|---|---|---|---|---|---|---|---|---|---|
| rare | rand | BOS | -0.1551 | [-0.1586, -0.1519] | [-0.1580, -0.1524] | 1.3e-83 | negative | -0.1551 [-0.1570, -0.1533] | 0/12 |
| rare | rand | Previous | +0.0130 | [+0.0126, +0.0135] | [+0.0126, +0.0134] | 2.1e-83 | positive | +0.0130 [+0.0127, +0.0132] | 12/12 |
| rare | rand | Self | +0.0005 | [-0.0000, +0.0009] | [+0.0001, +0.0008] | 0.01 | equivalent | +0.0005 [+0.0002, +0.0007] | 10/12 |
| rare | rand | Copy | +0.0938 | [+0.0928, +0.0948] | [+0.0930, +0.0947] | 1.3e-83 | positive | +0.0938 [+0.0930, +0.0946] | 12/12 |
| rare | rand | Dup | +0.0024 | [+0.0021, +0.0027] | [+0.0021, +0.0026] | 4.5e-37 | positive | +0.0024 [+0.0022, +0.0025] | 12/12 |
| rare | rand | Delimiter | +0.0000 | [+0.0000, +0.0000] | [+0.0000, +0.0000] | 1 | equivalent | +0.0000 [+0.0000, +0.0000] | 0/12 |
| rare | rand | Span | +0.0015 | [+0.0012, +0.0018] | [+0.0012, +0.0017] | 3e-22 | positive | +0.0015 [+0.0014, +0.0016] | 12/12 |
| rare | rand | Filler | +0.0440 | [+0.0413, +0.0469] | [+0.0417, +0.0463] | 2.6e-75 | positive | +0.0440 [+0.0429, +0.0451] | 12/12 |
| rare | rand | Phrase | +0.1107 | [+0.1095, +0.1119] | [+0.1097, +0.1117] | 1.3e-83 | positive | +0.1107 [+0.1096, +0.1116] | 12/12 |
| rare | content | Copy | -0.0093 | [-0.0105, -0.0080] | [-0.0103, -0.0082] | 2.3e-35 | negative | -0.0093 [-0.0177, -0.0008] | 4/12 |
| rare | nl_vs_rand | Copy | +0.0009 | [-0.0004, +0.0022] | [-0.0002, +0.0020] | 0.13 | equivalent | +0.0009 [-0.0073, +0.0094] | 6/12 |
| rare | park | BOS | -0.1037 | [-0.1055, -0.1020] | [-0.1052, -0.1023] | 1.3e-83 | negative | -0.1037 [-0.1209, -0.0857] | 0/12 |
| rare | park | Previous | +0.0087 | [+0.0083, +0.0090] | [+0.0084, +0.0089] | 2.7e-83 | positive | +0.0087 [+0.0055, +0.0116] | 12/12 |
| rare | park | Self | -0.0024 | [-0.0027, -0.0022] | [-0.0026, -0.0022] | 7.4e-49 | negative | -0.0024 [-0.0053, +0.0003] | 7/12 |
| rare | park | Copy | +0.0873 | [+0.0861, +0.0884] | [+0.0863, +0.0882] | 1.3e-83 | positive | +0.0873 [+0.0786, +0.0964] | 12/12 |
| rare | park | Dup | +0.0040 | [+0.0039, +0.0041] | [+0.0040, +0.0041] | 1.3e-83 | positive | +0.0040 [+0.0022, +0.0060] | 10/12 |
| rare | park | Delimiter | +0.0000 | [+0.0000, +0.0000] | [+0.0000, +0.0000] | 1 | equivalent | +0.0000 [+0.0000, +0.0000] | 0/12 |
| rare | park | Span | +0.0004 | [+0.0003, +0.0005] | [+0.0003, +0.0004] | 1.5e-18 | positive | +0.0004 [-0.0007, +0.0014] | 8/12 |
| rare | park | Filler | +0.0058 | [+0.0042, +0.0074] | [+0.0045, +0.0072] | 5.6e-11 | positive | +0.0058 [-0.0121, +0.0228] | 6/12 |
| rare | park | Phrase | +0.1003 | [+0.0991, +0.1015] | [+0.0993, +0.1014] | 1.3e-83 | positive | +0.1003 [+0.0908, +0.1094] | 12/12 |
| rare | two_tok | BOS | -0.0651 | [-0.0671, -0.0629] | [-0.0668, -0.0634] | 1.4e-83 | negative | -0.0651 [-0.0810, -0.0476] | 1/12 |
| rare | two_tok | Previous | +0.0163 | [+0.0153, +0.0173] | [+0.0154, +0.0171] | 2.6e-74 | positive | +0.0163 [+0.0120, +0.0206] | 12/12 |
| rare | two_tok | Self | -0.0045 | [-0.0050, -0.0039] | [-0.0049, -0.0040] | 4.4e-40 | negative | -0.0045 [-0.0074, -0.0016] | 2/12 |
| rare | two_tok | Copy | +0.0475 | [+0.0464, +0.0487] | [+0.0466, +0.0485] | 1.3e-83 | positive | +0.0475 [+0.0414, +0.0530] | 12/12 |
| rare | two_tok | Dup | +0.0015 | [+0.0014, +0.0016] | [+0.0014, +0.0016] | 1.7e-79 | positive | +0.0015 [+0.0008, +0.0021] | 11/12 |
| rare | two_tok | Delimiter | +0.0000 | [+0.0000, +0.0000] | [+0.0000, +0.0000] | 1 | equivalent | +0.0000 [+0.0000, +0.0000] | 0/12 |
| rare | two_tok | Span | -0.0002 | [-0.0003, -0.0001] | [-0.0003, -0.0001] | 7.7e-05 | negative | -0.0002 [-0.0005, +0.0001] | 6/12 |
| rare | two_tok | Filler | +0.0044 | [+0.0021, +0.0065] | [+0.0026, +0.0061] | 0.00015 | positive | +0.0044 [-0.0095, +0.0186] | 6/12 |
| rare | two_tok | Phrase | +0.0651 | [+0.0636, +0.0665] | [+0.0639, +0.0663] | 1.3e-83 | positive | +0.0651 [+0.0586, +0.0714] | 12/12 |
| rare | nl_repeat | BOS | -0.0949 | [-0.0970, -0.0929] | [-0.0967, -0.0932] | 1.3e-83 | negative | -0.0949 [-0.1114, -0.0780] | 0/12 |
| rare | nl_repeat | Previous | +0.0084 | [+0.0080, +0.0087] | [+0.0081, +0.0087] | 9.9e-83 | positive | +0.0084 [+0.0048, +0.0116] | 11/12 |
| rare | nl_repeat | Self | -0.0039 | [-0.0041, -0.0036] | [-0.0041, -0.0036] | 3e-68 | negative | -0.0039 [-0.0069, -0.0012] | 3/12 |
| rare | nl_repeat | Copy | +0.0941 | [+0.0928, +0.0954] | [+0.0931, +0.0953] | 1.3e-83 | positive | +0.0941 [+0.0864, +0.1022] | 12/12 |
| rare | nl_repeat | Dup | +0.0010 | [+0.0007, +0.0014] | [+0.0007, +0.0014] | 3.5e-12 | positive | +0.0010 [-0.0004, +0.0028] | 6/12 |
| rare | nl_repeat | Delimiter | +0.0000 | [+0.0000, +0.0000] | [+0.0000, +0.0000] | 1 | equivalent | +0.0000 [+0.0000, +0.0000] | 0/12 |
| rare | nl_repeat | Span | -0.0014 | [-0.0018, -0.0010] | [-0.0017, -0.0011] | 3.7e-08 | negative | -0.0014 [-0.0020, -0.0007] | 2/12 |
| rare | nl_repeat | Filler | -0.0034 | [-0.0052, -0.0015] | [-0.0049, -0.0018] | 0.0014 | negative | -0.0034 [-0.0207, +0.0134] | 6/12 |
| rare | nl_repeat | Phrase | +0.1022 | [+0.1007, +0.1036] | [+0.1010, +0.1034] | 1.3e-83 | positive | +0.1022 [+0.0930, +0.1111] | 12/12 |
| rare | placebo | BOS | +0.0004 | [-0.0004, +0.0012] | [-0.0003, +0.0011] | 0.41 | equivalent | +0.0004 [-0.0001, +0.0009] | 9/12 |
| rare | placebo | Previous | -0.0000 | [-0.0002, +0.0001] | [-0.0001, +0.0001] | 0.76 | equivalent | -0.0000 [-0.0001, +0.0001] | 5/12 |
| rare | placebo | Self | -0.0000 | [-0.0002, +0.0001] | [-0.0001, +0.0000] | 0.81 | equivalent | -0.0000 [-0.0001, +0.0000] | 5/12 |
| rare | placebo | Copy | -0.0000 | [-0.0003, +0.0003] | [-0.0003, +0.0002] | 0.15 | equivalent | -0.0000 [-0.0002, +0.0001] | 5/12 |
| rare | placebo | Dup | -0.0001 | [-0.0004, +0.0002] | [-0.0004, +0.0001] | 0.63 | equivalent | -0.0001 [-0.0003, +0.0000] | 4/12 |
| rare | placebo | Delimiter | +0.0000 | [+0.0000, +0.0000] | [+0.0000, +0.0000] | 1 | equivalent | +0.0000 [+0.0000, +0.0000] | 0/12 |
| rare | placebo | Span | -0.0000 | [-0.0003, +0.0002] | [-0.0003, +0.0002] | 0.26 | equivalent | -0.0000 [-0.0001, +0.0001] | 7/12 |
| rare | placebo | Filler | -0.0001 | [-0.0008, +0.0006] | [-0.0007, +0.0005] | 0.61 | equivalent | -0.0001 [-0.0005, +0.0003] | 5/12 |
| rare | placebo | Phrase | -0.0002 | [-0.0007, +0.0002] | [-0.0006, +0.0002] | 0.83 | equivalent | -0.0002 [-0.0004, -0.0001] | 4/12 |
| common | rand | BOS | -0.1107 | [-0.1138, -0.1076] | [-0.1132, -0.1083] | 1.3e-83 | negative | -0.1107 [-0.1134, -0.1078] | 0/12 |
| common | rand | Previous | +0.0061 | [+0.0057, +0.0064] | [+0.0057, +0.0064] | 6e-77 | positive | +0.0061 [+0.0056, +0.0064] | 12/12 |
| common | rand | Self | +0.0029 | [+0.0025, +0.0032] | [+0.0026, +0.0031] | 3.2e-43 | positive | +0.0029 [+0.0024, +0.0032] | 12/12 |
| common | rand | Copy | +0.0614 | [+0.0605, +0.0623] | [+0.0607, +0.0621] | 1.3e-83 | positive | +0.0614 [+0.0608, +0.0619] | 12/12 |
| common | rand | Dup | +0.0010 | [+0.0006, +0.0013] | [+0.0007, +0.0012] | 5.1e-15 | positive | +0.0010 [+0.0009, +0.0010] | 12/12 |
| common | rand | Delimiter | +0.0000 | [+0.0000, +0.0000] | [+0.0000, +0.0000] | 1 | equivalent | +0.0000 [+0.0000, +0.0000] | 0/12 |
| common | rand | Span | +0.0012 | [+0.0009, +0.0016] | [+0.0010, +0.0015] | 3.6e-20 | positive | +0.0012 [+0.0011, +0.0014] | 12/12 |
| common | rand | Filler | +0.0382 | [+0.0355, +0.0408] | [+0.0360, +0.0404] | 5.3e-73 | positive | +0.0382 [+0.0360, +0.0404] | 12/12 |
| common | rand | Phrase | +0.0697 | [+0.0686, +0.0707] | [+0.0688, +0.0705] | 1.3e-83 | positive | +0.0697 [+0.0691, +0.0702] | 12/12 |
| common | content | Copy | +0.0004 | [-0.0009, +0.0017] | [-0.0007, +0.0016] | 0.52 | equivalent | +0.0004 [-0.0087, +0.0099] | 7/12 |
| common | nl_vs_rand | Copy | +0.0657 | [+0.0645, +0.0669] | [+0.0647, +0.0667] | 1.3e-83 | positive | +0.0657 [+0.0577, +0.0735] | 12/12 |
| common | park | BOS | -0.1199 | [-0.1214, -0.1183] | [-0.1211, -0.1186] | 1.3e-83 | negative | -0.1199 [-0.1394, -0.0994] | 0/12 |
| common | park | Previous | +0.0059 | [+0.0055, +0.0062] | [+0.0056, +0.0062] | 8.4e-79 | positive | +0.0059 [+0.0032, +0.0085] | 11/12 |
| common | park | Self | -0.0084 | [-0.0087, -0.0081] | [-0.0087, -0.0082] | 6.3e-83 | negative | -0.0084 [-0.0132, -0.0043] | 2/12 |
| common | park | Copy | +0.1045 | [+0.1035, +0.1054] | [+0.1036, +0.1053] | 1.3e-83 | positive | +0.1045 [+0.0944, +0.1142] | 12/12 |
| common | park | Dup | +0.0030 | [+0.0029, +0.0031] | [+0.0030, +0.0031] | 1.3e-83 | positive | +0.0030 [+0.0000, +0.0058] | 9/12 |
| common | park | Delimiter | +0.0000 | [+0.0000, +0.0000] | [+0.0000, +0.0000] | 1 | equivalent | +0.0000 [+0.0000, +0.0000] | 0/12 |
| common | park | Span | -0.0004 | [-0.0005, -0.0003] | [-0.0005, -0.0003] | 1.4e-14 | negative | -0.0004 [-0.0018, +0.0009] | 6/12 |
| common | park | Filler | +0.0154 | [+0.0137, +0.0170] | [+0.0140, +0.0167] | 1.4e-50 | positive | +0.0154 [-0.0065, +0.0346] | 7/12 |
| common | park | Phrase | +0.1129 | [+0.1119, +0.1139] | [+0.1121, +0.1138] | 1.3e-83 | positive | +0.1129 [+0.1032, +0.1223] | 12/12 |
| common | two_tok | BOS | -0.0377 | [-0.0398, -0.0355] | [-0.0395, -0.0359] | 5.4e-81 | negative | -0.0377 [-0.0504, -0.0245] | 1/12 |
| common | two_tok | Previous | +0.0194 | [+0.0183, +0.0205] | [+0.0185, +0.0203] | 3.3e-76 | positive | +0.0194 [+0.0161, +0.0228] | 12/12 |
| common | two_tok | Self | -0.0076 | [-0.0081, -0.0071] | [-0.0080, -0.0072] | 1.2e-74 | negative | -0.0076 [-0.0111, -0.0047] | 1/12 |
| common | two_tok | Copy | +0.0507 | [+0.0492, +0.0522] | [+0.0494, +0.0520] | 1.5e-83 | positive | +0.0507 [+0.0446, +0.0576] | 12/12 |
| common | two_tok | Dup | +0.0012 | [+0.0011, +0.0013] | [+0.0011, +0.0013] | 2.3e-57 | positive | +0.0012 [+0.0005, +0.0018] | 11/12 |
| common | two_tok | Delimiter | +0.0000 | [+0.0000, +0.0000] | [+0.0000, +0.0000] | 1 | equivalent | +0.0000 [+0.0000, +0.0000] | 0/12 |
| common | two_tok | Span | -0.0005 | [-0.0006, -0.0004] | [-0.0006, -0.0004] | 5.1e-20 | negative | -0.0005 [-0.0009, -0.0001] | 3/12 |
| common | two_tok | Filler | -0.0255 | [-0.0275, -0.0235] | [-0.0272, -0.0238] | 7.4e-62 | negative | -0.0255 [-0.0372, -0.0148] | 2/12 |
| common | two_tok | Phrase | +0.0708 | [+0.0689, +0.0727] | [+0.0692, +0.0724] | 1.3e-83 | positive | +0.0708 [+0.0651, +0.0776] | 12/12 |
| common | nl_repeat | BOS | -0.1116 | [-0.1137, -0.1096] | [-0.1134, -0.1098] | 1.3e-83 | negative | -0.1116 [-0.1291, -0.0934] | 0/12 |
| common | nl_repeat | Previous | +0.0044 | [+0.0040, +0.0048] | [+0.0041, +0.0048] | 5.8e-53 | positive | +0.0044 [+0.0009, +0.0075] | 9/12 |
| common | nl_repeat | Self | -0.0117 | [-0.0121, -0.0113] | [-0.0120, -0.0114] | 1.8e-83 | negative | -0.0117 [-0.0167, -0.0074] | 1/12 |
| common | nl_repeat | Copy | +0.1258 | [+0.1245, +0.1270] | [+0.1247, +0.1269] | 1.3e-83 | positive | +0.1258 [+0.1177, +0.1334] | 12/12 |
| common | nl_repeat | Dup | +0.0029 | [+0.0025, +0.0033] | [+0.0025, +0.0032] | 6.9e-40 | positive | +0.0029 [+0.0012, +0.0048] | 9/12 |
| common | nl_repeat | Delimiter | +0.0000 | [+0.0000, +0.0000] | [+0.0000, +0.0000] | 1 | equivalent | +0.0000 [+0.0000, +0.0000] | 0/12 |
| common | nl_repeat | Span | +0.0002 | [-0.0002, +0.0005] | [-0.0001, +0.0005] | 0.00064 | equivalent | +0.0002 [-0.0005, +0.0009] | 5/12 |
| common | nl_repeat | Filler | -0.0099 | [-0.0120, -0.0078] | [-0.0116, -0.0082] | 3.3e-19 | negative | -0.0099 [-0.0297, +0.0087] | 6/12 |
| common | nl_repeat | Phrase | +0.1333 | [+0.1318, +0.1347] | [+0.1320, +0.1344] | 1.3e-83 | positive | +0.1333 [+0.1258, +0.1400] | 12/12 |
| common | placebo | BOS | +0.0013 | [+0.0004, +0.0023] | [+0.0005, +0.0022] | 0.086 | positive | +0.0013 [+0.0009, +0.0018] | 12/12 |
| common | placebo | Previous | +0.0000 | [-0.0001, +0.0001] | [-0.0001, +0.0001] | 0.98 | equivalent | +0.0000 [-0.0001, +0.0001] | 5/12 |
| common | placebo | Self | +0.0000 | [-0.0001, +0.0001] | [-0.0000, +0.0001] | 0.55 | equivalent | +0.0000 [-0.0000, +0.0001] | 7/12 |
| common | placebo | Copy | -0.0001 | [-0.0005, +0.0002] | [-0.0004, +0.0002] | 0.16 | equivalent | -0.0001 [-0.0003, +0.0000] | 4/12 |
| common | placebo | Dup | -0.0004 | [-0.0007, -0.0000] | [-0.0007, -0.0001] | 0.91 | negative | -0.0004 [-0.0005, -0.0002] | 1/12 |
| common | placebo | Delimiter | +0.0000 | [+0.0000, +0.0000] | [+0.0000, +0.0000] | 1 | equivalent | +0.0000 [+0.0000, +0.0000] | 0/12 |
| common | placebo | Span | -0.0000 | [-0.0003, +0.0003] | [-0.0003, +0.0002] | 0.083 | equivalent | -0.0000 [-0.0002, +0.0001] | 7/12 |
| common | placebo | Filler | -0.0008 | [-0.0017, +0.0000] | [-0.0016, -0.0001] | 0.11 | equivalent | -0.0008 [-0.0013, -0.0004] | 2/12 |
| common | placebo | Phrase | -0.0005 | [-0.0011, +0.0000] | [-0.0010, -0.0001] | 0.43 | equivalent | -0.0005 [-0.0007, -0.0003] | 1/12 |

## Meaningful heads per contrast (BH q ≤ 0.05 over 144 heads, |Δ| ≥ 0.02; + / −)

| base | contrast | BOS | Previous | Self | Copy | Dup | Delimiter | Span | Filler | Phrase |
|---|---|---|---|---|---|---|---|---|---|---|
| rare | rand | 13 / 92 | 20 / 8 | 11 / 13 | 44 / 0 | 10 / 0 | 0 / 0 | 0 / 0 | 76 / 20 | 77 / 8 |
| rare | content | — | — | — | 20 / 20 | — | — | — | — | — |
| rare | nl_vs_rand | — | — | — | 27 / 18 | — | — | — | — | — |
| rare | park | 22 / 78 | 15 / 8 | 6 / 16 | 49 / 0 | 13 / 0 | 0 / 0 | 1 / 1 | 40 / 40 | 72 / 7 |
| rare | two_tok | 21 / 76 | 43 / 28 | 18 / 14 | 38 / 0 | 3 / 0 | 0 / 0 | 0 / 2 | 48 / 51 | 78 / 18 |
| rare | nl_repeat | 23 / 76 | 15 / 6 | 6 / 19 | 54 / 0 | 15 / 0 | 0 / 0 | 6 / 3 | 39 / 43 | 73 / 4 |
| rare | placebo | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| common | rand | 13 / 86 | 16 / 4 | 8 / 5 | 33 / 0 | 8 / 0 | 0 / 0 | 1 / 0 | 66 / 19 | 60 / 3 |
| common | content | — | — | — | 29 / 16 | — | — | — | — | — |
| common | nl_vs_rand | — | — | — | 50 / 7 | — | — | — | — | — |
| common | park | 14 / 83 | 15 / 4 | 9 / 26 | 52 / 3 | 10 / 1 | 0 / 0 | 1 / 1 | 46 / 35 | 79 / 6 |
| common | two_tok | 25 / 64 | 41 / 12 | 11 / 20 | 41 / 0 | 3 / 0 | 0 / 0 | 1 / 1 | 27 / 67 | 84 / 8 |
| common | nl_repeat | 21 / 86 | 19 / 5 | 6 / 32 | 64 / 0 | 15 / 0 | 0 / 0 | 3 / 0 | 37 / 52 | 96 / 3 |
| common | placebo | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

## Role switches (McNemar, BH, ≥ 15% shift)

- rare, placebo switching heads (either direction): argmax 0 [], mass 0
- rare, rand, argmax labels: 38 heads; L0H10 Self→Diffuse (+0.15), L1H9 Sink→Diffuse (+0.52), L2H6 Sink→Diffuse (+0.23), L7H1 Sink→Diffuse (+0.35), L7H7 Sink→Diffuse (+0.22), L10H4 Sink→Diffuse (+0.17), L11H1 Sink→Diffuse (+0.29), L11H5 Sink→Diffuse (+0.19), L11H11 Sink→Diffuse (+0.19), L8H6 Diffuse→Sink (+0.28), L3H3 Sink→Previous (+0.23), L3H11 Sink→Previous (+0.15), L4H3 Sink→Previous (+0.20), L5H6 Sink→Previous (+0.64), L6H0 Sink→Previous (+0.68), L6H8 Sink→Previous (+0.92), L7H0 Sink→Previous (+0.34), L5H0 Sink→Copy (+0.87), L5H1 Sink→Copy (+0.98), L5H5 Sink→Copy (+1.00), L6H9 Sink→Copy (+1.00), L7H1 Sink→Copy (+0.49), L7H2 Sink→Copy (+0.42), L7H10 Sink→Copy (+1.00), L8H1 Sink→Copy (+0.38), L8H6 Diffuse→Copy (+0.17), L9H6 Sink→Copy (+0.89), L9H9 Sink→Copy (+0.56), L10H0 Sink→Copy (+0.77), L10H1 Sink→Copy (+0.99), L10H6 Sink→Copy (+0.52), L10H7 Sink→Copy (+0.39), L10H11 Sink→Copy (+0.62), L11H9 Sink→Copy (+0.79), L0H5 Self→Dup (+1.00), L3H0 Sink→Dup (+1.00), L2H8 Previous→Filler (+0.19), L3H11 Sink→Filler (+0.21)
- rare, rand, mass labels: 52 heads; L6H4 Filler→Sink (+0.54), L7H11 Filler→Sink (+0.75), L8H2 Filler→Sink (+0.27), L3H3 Sink→Previous (+0.23), L5H6 Sink→Previous (+0.64), L6H0 Sink→Previous (+0.19), L6H8 Sink→Previous (+0.92), L7H0 Sink→Previous (+0.32), L5H0 Filler→Copy (+0.65), L5H1 Sink→Copy (+0.98), L5H5 Sink→Copy (+1.00), L6H9 Sink→Copy (+1.00), L7H2 Sink→Copy (+0.42), L7H10 Sink→Copy (+1.00), L9H6 Sink→Copy (+0.48), L9H9 Sink→Copy (+0.48), L10H1 Sink→Copy (+0.38), L10H7 Sink→Copy (+0.39), L0H5 Self→Dup (+1.00), L3H0 Sink→Dup (+1.00), L3H4 Sink→Filler (+0.96), L3H5 Sink→Filler (+0.32), L3H10 Sink→Filler (+0.44), L3H11 Sink→Filler (+0.28), L4H3 Sink→Filler (+0.27), L4H6 Sink→Filler (+0.40), L4H7 Self→Filler (+0.32), L5H3 Sink→Filler (+0.25), L5H9 Sink→Filler (+1.00), L6H0 Sink→Filler (+0.56), L6H2 Sink→Filler (+0.83), L6H5 Sink→Filler (+0.24), L6H6 Sink→Filler (+0.50), L6H7 Sink→Filler (+0.70), L7H1 Sink→Filler (+0.97), L7H6 Sink→Filler (+0.76), L8H1 Sink→Filler (+0.44), L9H0 Sink→Filler (+0.94), L9H2 Sink→Filler (+0.78), L9H6 Sink→Filler (+0.48), L10H0 Sink→Filler (+0.83), L10H1 Sink→Filler (+0.19), L10H3 Sink→Filler (+0.37), L10H6 Sink→Filler (+0.70), L10H10 Sink→Filler (+0.59), L11H1 Sink→Filler (+0.21), L11H2 Sink→Filler (+0.62), L11H4 Sink→Filler (+0.21), L11H5 Sink→Filler (+0.34), L11H6 Sink→Filler (+0.66), L11H7 Sink→Filler (+0.32), L11H11 Sink→Filler (+0.84)
- rare, park, argmax labels: 30 heads; L1H6 Sink→Diffuse (+0.72), L1H9 Sink→Diffuse (+0.57), L2H6 Sink→Diffuse (+0.22), L5H10 Sink→Diffuse (+0.18), L6H10 Sink→Diffuse (+0.25), L7H7 Sink→Diffuse (+0.23), L10H4 Sink→Diffuse (+0.51), L11H3 Sink→Diffuse (+0.17), L9H11 Diffuse→Sink (+0.44), L2H0 Sink→Previous (+0.20), L3H8 Sink→Previous (+0.25), L5H6 Sink→Previous (+0.30), L6H8 Sink→Previous (+0.97), L6H11 Sink→Previous (+0.59), L1H6 Sink→Copy (+0.21), L5H0 Sink→Copy (+0.93), L5H1 Sink→Copy (+0.86), L5H5 Sink→Copy (+1.00), L6H9 Sink→Copy (+0.92), L7H1 Sink→Copy (+0.93), L7H10 Sink→Copy (+0.21), L8H1 Sink→Copy (+0.52), L8H6 Sink→Copy (+0.92), L9H6 Sink→Copy (+0.22), L10H7 Sink→Copy (+0.91), L10H11 Sink→Copy (+1.00), L0H5 Self→Dup (+1.00), L3H0 Sink→Dup (+1.00), L2H9 Previous→Filler (+0.17), L7H7 Sink→Filler (+0.57)
- rare, park, mass labels: 38 heads; L4H2 Filler→Sink (+0.19), L7H11 Filler→Sink (+0.37), L10H2 Filler→Sink (+0.31), L11H2 Filler→Sink (+0.37), L11H6 Filler→Sink (+0.17), L5H6 Sink→Previous (+0.30), L6H8 Sink→Previous (+0.97), L6H11 Sink→Previous (+0.34), L5H0 Sink→Copy (+0.93), L5H1 Sink→Copy (+0.86), L5H5 Sink→Copy (+1.00), L6H9 Sink→Copy (+0.92), L7H1 Sink→Copy (+0.77), L7H10 Sink→Copy (+0.21), L8H1 Sink→Copy (+0.37), L8H6 Sink→Copy (+0.90), L9H6 Sink→Copy (+0.21), L10H7 Sink→Copy (+0.91), L10H11 Filler→Copy (+0.69), L0H5 Self→Dup (+1.00), L3H0 Sink→Dup (+1.00), L3H5 Sink→Filler (+0.17), L4H6 Sink→Filler (+0.80), L4H8 Sink→Filler (+0.48), L5H9 Sink→Filler (+0.99), L6H2 Sink→Filler (+0.99), L6H5 Sink→Filler (+0.45), L6H10 Sink→Filler (+0.54), L6H11 Sink→Filler (+0.26), L7H1 Sink→Filler (+0.23), L7H6 Sink→Filler (+0.69), L7H7 Sink→Filler (+0.44), L8H1 Sink→Filler (+0.21), L9H0 Sink→Filler (+0.35), L9H2 Sink→Filler (+0.25), L9H8 Sink→Filler (+0.22), L11H5 Sink→Filler (+0.22), L11H11 Sink→Filler (+0.49)
- rare, two_tok, argmax labels: 36 heads; L2H6 Sink→Diffuse (+0.27), L10H4 Sink→Diffuse (+0.42), L11H3 Sink→Diffuse (+0.33), L0H7 Self→Previous (+0.61), L1H0 Self→Previous (+0.65), L1H1 Diffuse→Previous (+0.71), L2H0 Sink→Previous (+0.72), L2H5 Filler→Previous (+0.16), L3H1 Sink→Previous (+0.89), L3H2 Self→Previous (+0.39), L3H6 Filler→Previous (+0.36), L4H1 Sink→Previous (+0.76), L4H3 Sink→Previous (+0.27), L5H3 Sink→Previous (+0.79), L5H6 Sink→Previous (+0.40), L6H0 Sink→Previous (+0.65), L6H8 Sink→Previous (+0.81), L6H11 Sink→Previous (+0.75), L2H8 Sink→Self (+0.76), L1H6 Diffuse→Copy (+0.16), L5H0 Sink→Copy (+0.76), L5H1 Sink→Copy (+0.86), L5H5 Sink→Copy (+0.28), L6H9 Sink→Copy (+0.40), L7H1 Sink→Copy (+0.74), L7H10 Sink→Copy (+0.15), L8H1 Sink→Copy (+0.49), L8H6 Sink→Copy (+0.89), L9H6 Sink→Copy (+0.21), L10H7 Sink→Copy (+0.88), L10H11 Diffuse→Copy (+0.37), L2H9 Previous→Filler (+0.68), L3H7 Previous→Filler (+0.63), L3H8 Sink→Filler (+0.22), L3H11 Previous→Filler (+0.27), L7H7 Diffuse→Filler (+0.51)
- rare, two_tok, mass labels: 56 heads; L3H4 Filler→Sink (+0.67), L4H8 Filler→Sink (+0.33), L7H11 Filler→Sink (+0.31), L10H1 Filler→Sink (+0.29), L10H2 Filler→Sink (+0.21), L10H6 Filler→Sink (+0.17), L10H10 Filler→Sink (+0.23), L11H4 Filler→Sink (+0.19), L11H5 Filler→Sink (+0.17), L11H6 Filler→Sink (+0.15), L0H7 Filler→Previous (+0.73), L1H0 Filler→Previous (+0.97), L2H2 Filler→Previous (+0.18), L2H5 Filler→Previous (+0.55), L3H1 Sink→Previous (+0.76), L3H2 Filler→Previous (+0.47), L3H6 Filler→Previous (+0.59), L4H1 Sink→Previous (+0.62), L4H3 Filler→Previous (+0.21), L5H6 Sink→Previous (+0.40), L6H0 Filler→Previous (+0.82), L6H8 Sink→Previous (+0.81), L6H11 Sink→Previous (+0.46), L4H7 Filler→Self (+0.47), L5H0 Filler→Copy (+0.86), L5H1 Sink→Copy (+0.86), L5H5 Sink→Copy (+0.28), L6H9 Sink→Copy (+0.40), L7H1 Sink→Copy (+0.70), L7H10 Sink→Copy (+0.16), L8H1 Filler→Copy (+0.37), L8H6 Filler→Copy (+0.89), L9H6 Sink→Copy (+0.20), L10H7 Sink→Copy (+0.88), L10H11 Filler→Copy (+0.66), L2H3 Previous→Filler (+0.30), L2H8 Previous→Filler (+0.26), L2H9 Previous→Filler (+0.30), L3H7 Previous→Filler (+0.82), L3H8 Previous→Filler (+0.54), L3H9 Sink→Filler (+0.16), L3H11 Sink→Filler (+0.42), L4H1 Sink→Filler (+0.20), L4H2 Sink→Filler (+0.38), L4H6 Sink→Filler (+0.15), L6H2 Sink→Filler (+0.89), L6H5 Sink→Filler (+0.41), L6H7 Sink→Filler (+0.85), L6H11 Sink→Filler (+0.36), L7H6 Sink→Filler (+0.31), L8H0 Sink→Filler (+0.18), L8H2 Sink→Filler (+0.21), L9H0 Sink→Filler (+0.39), L9H2 Sink→Filler (+0.45), L9H8 Sink→Filler (+0.34), L11H11 Sink→Filler (+0.18)
- rare, nl_repeat, argmax labels: 33 heads; L1H6 Sink→Diffuse (+0.68), L1H9 Sink→Diffuse (+0.77), L2H6 Sink→Diffuse (+0.23), L5H10 Sink→Diffuse (+0.18), L6H10 Sink→Diffuse (+0.25), L7H7 Sink→Diffuse (+0.23), L10H4 Sink→Diffuse (+0.46), L11H3 Sink→Diffuse (+0.16), L7H0 Previous→Sink (+0.16), L9H11 Diffuse→Sink (+0.49), L10H2 Filler→Sink (+0.17), L2H0 Sink→Previous (+0.22), L3H8 Sink→Previous (+0.28), L5H6 Sink→Previous (+0.24), L6H8 Sink→Previous (+0.96), L6H11 Sink→Previous (+0.58), L2H8 Sink→Self (+0.25), L1H6 Sink→Copy (+0.21), L5H0 Sink→Copy (+0.93), L5H1 Sink→Copy (+0.86), L5H5 Sink→Copy (+1.00), L6H9 Sink→Copy (+0.92), L7H1 Sink→Copy (+0.93), L7H10 Sink→Copy (+0.21), L8H1 Sink→Copy (+0.52), L8H6 Sink→Copy (+0.92), L9H6 Sink→Copy (+0.22), L10H7 Sink→Copy (+0.91), L10H11 Sink→Copy (+1.00), L0H5 Self→Dup (+1.00), L3H0 Sink→Dup (+1.00), L2H9 Previous→Filler (+0.18), L7H7 Sink→Filler (+0.57)
- rare, nl_repeat, mass labels: 39 heads; L4H2 Filler→Sink (+0.19), L7H0 Previous→Sink (+0.16), L7H11 Filler→Sink (+0.47), L10H2 Filler→Sink (+0.32), L10H10 Filler→Sink (+0.20), L11H2 Filler→Sink (+0.42), L11H6 Filler→Sink (+0.25), L5H6 Sink→Previous (+0.24), L6H8 Sink→Previous (+0.96), L6H11 Sink→Previous (+0.33), L5H0 Sink→Copy (+0.93), L5H1 Sink→Copy (+0.86), L5H5 Sink→Copy (+1.00), L6H9 Sink→Copy (+0.92), L7H1 Sink→Copy (+0.77), L7H10 Sink→Copy (+0.21), L8H1 Sink→Copy (+0.37), L8H6 Sink→Copy (+0.90), L9H6 Sink→Copy (+0.21), L10H7 Sink→Copy (+0.91), L10H11 Filler→Copy (+0.69), L0H5 Self→Dup (+1.00), L3H0 Sink→Dup (+1.00), L3H5 Sink→Filler (+0.17), L4H6 Sink→Filler (+0.71), L4H8 Sink→Filler (+0.44), L5H9 Sink→Filler (+0.99), L6H2 Sink→Filler (+0.99), L6H5 Sink→Filler (+0.45), L6H7 Sink→Filler (+0.15), L6H10 Sink→Filler (+0.55), L6H11 Sink→Filler (+0.26), L7H1 Sink→Filler (+0.23), L7H6 Sink→Filler (+0.69), L7H7 Sink→Filler (+0.47), L8H1 Sink→Filler (+0.21), L9H0 Sink→Filler (+0.25), L9H8 Sink→Filler (+0.20), L11H11 Sink→Filler (+0.41)
- rare, placebo, argmax labels: 0 heads; none
- rare, placebo, mass labels: 0 heads; none
- common, placebo switching heads (either direction): argmax 0 [], mass 0
- common, rand, argmax labels: 26 heads; L0H10 Self→Diffuse (+0.54), L1H5 Sink→Diffuse (+0.87), L1H6 Sink→Diffuse (+0.19), L1H7 Sink→Diffuse (+0.50), L1H9 Sink→Diffuse (+0.45), L5H3 Sink→Diffuse (+0.16), L7H1 Sink→Diffuse (+0.31), L11H9 Sink→Diffuse (+0.41), L6H4 Diffuse→Sink (+0.23), L8H6 Diffuse→Sink (+0.24), L3H3 Sink→Previous (+0.24), L3H8 Sink→Previous (+0.32), L4H3 Sink→Previous (+0.18), L5H6 Sink→Previous (+0.46), L6H8 Sink→Previous (+0.36), L7H0 Sink→Previous (+0.27), L5H0 Sink→Copy (+0.46), L5H1 Sink→Copy (+0.65), L5H5 Sink→Copy (+1.00), L6H9 Sink→Copy (+1.00), L7H10 Sink→Copy (+1.00), L8H1 Sink→Copy (+0.21), L10H1 Sink→Copy (+0.65), L11H9 Sink→Copy (+0.17), L0H5 Self→Dup (+1.00), L3H0 Sink→Dup (+0.87)
- common, rand, mass labels: 47 heads; L7H11 Filler→Sink (+0.52), L8H2 Filler→Sink (+0.70), L9H8 Filler→Sink (+0.28), L9H11 Filler→Sink (+0.41), L10H2 Filler→Sink (+0.28), L10H9 Filler→Sink (+0.19), L3H3 Sink→Previous (+0.24), L5H6 Sink→Previous (+0.46), L6H8 Sink→Previous (+0.43), L5H0 Sink→Copy (+0.16), L5H1 Sink→Copy (+0.65), L5H5 Sink→Copy (+1.00), L6H9 Sink→Copy (+1.00), L7H10 Sink→Copy (+0.99), L0H5 Self→Dup (+1.00), L3H0 Sink→Dup (+0.87), L3H5 Sink→Filler (+0.35), L3H11 Sink→Filler (+0.23), L4H6 Sink→Filler (+0.53), L4H8 Sink→Filler (+0.22), L5H0 Sink→Filler (+0.42), L5H9 Sink→Filler (+0.95), L5H10 Sink→Filler (+0.48), L6H0 Sink→Filler (+0.23), L6H3 Sink→Filler (+0.19), L6H5 Sink→Filler (+0.68), L6H6 Sink→Filler (+0.59), L6H7 Sink→Filler (+0.73), L6H10 Sink→Filler (+0.41), L7H1 Sink→Filler (+0.99), L7H6 Sink→Filler (+0.22), L7H7 Sink→Filler (+0.66), L8H1 Sink→Filler (+0.44), L9H9 Sink→Filler (+0.18), L10H0 Sink→Filler (+0.58), L10H1 Sink→Filler (+0.68), L10H3 Sink→Filler (+0.17), L10H8 Sink→Filler (+0.47), L10H10 Sink→Filler (+0.55), L10H11 Sink→Filler (+0.66), L11H1 Sink→Filler (+0.39), L11H5 Sink→Filler (+0.34), L11H6 Sink→Filler (+0.23), L11H7 Sink→Filler (+0.28), L11H9 Sink→Filler (+0.58), L11H10 Sink→Filler (+0.16), L11H11 Sink→Filler (+0.89)
- common, park, argmax labels: 35 heads; L1H5 Sink→Diffuse (+0.57), L5H10 Sink→Diffuse (+0.23), L6H7 Sink→Diffuse (+0.30), L7H7 Sink→Diffuse (+0.30), L10H4 Sink→Diffuse (+0.15), L2H1 Diffuse→Sink (+0.38), L7H0 Previous→Sink (+0.23), L8H2 Diffuse→Sink (+0.25), L9H11 Diffuse→Sink (+0.24), L2H0 Sink→Previous (+0.28), L5H6 Sink→Previous (+0.40), L6H8 Sink→Previous (+0.92), L6H11 Sink→Previous (+0.49), L8H5 Sink→Previous (+0.21), L2H10 Diffuse→Self (+0.35), L1H6 Sink→Copy (+0.98), L5H0 Sink→Copy (+0.78), L5H1 Sink→Copy (+0.74), L5H5 Sink→Copy (+1.00), L6H9 Sink→Copy (+0.79), L7H1 Sink→Copy (+0.88), L7H10 Sink→Copy (+0.63), L8H1 Sink→Copy (+0.73), L8H6 Sink→Copy (+0.84), L9H6 Sink→Copy (+0.53), L9H9 Sink→Copy (+0.45), L10H7 Sink→Copy (+1.00), L10H11 Sink→Copy (+1.00), L11H7 Sink→Copy (+0.27), L11H10 Sink→Copy (+0.27), L0H5 Self→Dup (+1.00), L3H0 Sink→Dup (+0.99), L2H9 Previous→Filler (+0.18), L3H7 Previous→Filler (+0.15), L7H7 Sink→Filler (+0.36)
- common, park, mass labels: 42 heads; L7H0 Previous→Sink (+0.21), L7H11 Filler→Sink (+0.42), L10H2 Filler→Sink (+0.29), L4H3 Filler→Previous (+0.18), L5H6 Sink→Previous (+0.40), L6H8 Sink→Previous (+0.89), L5H0 Sink→Copy (+0.77), L5H1 Sink→Copy (+0.74), L5H5 Sink→Copy (+1.00), L6H9 Sink→Copy (+0.79), L7H1 Sink→Copy (+0.52), L7H10 Sink→Copy (+0.63), L8H1 Sink→Copy (+0.51), L8H6 Sink→Copy (+0.72), L9H6 Sink→Copy (+0.53), L9H9 Sink→Copy (+0.45), L10H7 Sink→Copy (+1.00), L10H11 Sink→Copy (+1.00), L11H10 Sink→Copy (+0.21), L0H5 Self→Dup (+1.00), L3H0 Sink→Dup (+0.99), L4H6 Sink→Filler (+0.75), L4H8 Sink→Filler (+0.25), L5H9 Sink→Filler (+0.89), L6H2 Sink→Filler (+0.99), L6H3 Sink→Filler (+0.38), L6H5 Sink→Filler (+0.89), L6H10 Sink→Filler (+0.57), L6H11 Sink→Filler (+0.20), L7H1 Sink→Filler (+0.47), L7H6 Sink→Filler (+0.52), L7H7 Sink→Filler (+0.82), L8H1 Sink→Filler (+0.29), L8H5 Sink→Filler (+0.29), L10H1 Sink→Filler (+0.25), L10H3 Sink→Filler (+0.30), L10H8 Sink→Filler (+0.28), L11H1 Sink→Filler (+0.18), L11H5 Sink→Filler (+0.31), L11H7 Sink→Filler (+0.21), L11H9 Sink→Filler (+0.39), L11H11 Sink→Filler (+0.58)
- common, two_tok, argmax labels: 43 heads; L5H10 Sink→Diffuse (+0.22), L6H7 Sink→Diffuse (+0.31), L11H5 Diffuse→Sink (+0.20), L11H6 Diffuse→Sink (+0.15), L11H7 Diffuse→Sink (+0.17), L0H7 Filler→Previous (+0.62), L1H0 Filler→Previous (+0.77), L1H1 Diffuse→Previous (+0.74), L2H0 Sink→Previous (+0.39), L2H2 Filler→Previous (+0.21), L2H3 Self→Previous (+0.16), L2H5 Filler→Previous (+0.22), L3H1 Sink→Previous (+0.62), L3H2 Filler→Previous (+0.49), L3H6 Filler→Previous (+0.60), L4H1 Sink→Previous (+0.69), L4H3 Sink→Previous (+0.68), L5H2 Filler→Previous (+0.24), L5H3 Diffuse→Previous (+0.79), L5H6 Sink→Previous (+0.20), L6H0 Diffuse→Previous (+0.66), L6H8 Sink→Previous (+0.59), L6H11 Sink→Previous (+0.63), L8H5 Sink→Previous (+0.23), L2H8 Previous→Self (+0.76), L2H10 Diffuse→Self (+0.22), L1H6 Sink→Copy (+0.16), L5H0 Sink→Copy (+0.60), L5H1 Sink→Copy (+0.74), L6H9 Sink→Copy (+0.23), L7H1 Sink→Copy (+0.75), L7H10 Sink→Copy (+0.29), L8H1 Sink→Copy (+0.50), L8H6 Sink→Copy (+0.80), L9H6 Sink→Copy (+0.51), L9H9 Sink→Copy (+0.34), L10H7 Sink→Copy (+0.68), L11H10 Sink→Copy (+0.27), L3H0 Sink→Dup (+0.19), L2H9 Previous→Filler (+0.32), L3H7 Previous→Filler (+0.67), L3H8 Previous→Filler (+0.30), L7H7 Sink→Filler (+0.31)
- common, two_tok, mass labels: 53 heads; L4H5 Filler→Sink (+0.28), L4H8 Filler→Sink (+0.54), L5H2 Filler→Sink (+0.26), L7H0 Filler→Sink (+0.27), L7H11 Filler→Sink (+0.30), L10H1 Filler→Sink (+0.49), L10H2 Filler→Sink (+0.24), L10H8 Filler→Sink (+0.24), L10H10 Filler→Sink (+0.37), L11H9 Filler→Sink (+0.23), L0H7 Filler→Previous (+0.52), L1H0 Filler→Previous (+0.92), L2H2 Filler→Previous (+0.33), L2H5 Filler→Previous (+0.57), L3H1 Sink→Previous (+0.25), L3H2 Filler→Previous (+0.53), L3H6 Filler→Previous (+0.74), L4H1 Sink→Previous (+0.40), L4H3 Filler→Previous (+0.71), L5H2 Filler→Previous (+0.24), L5H6 Sink→Previous (+0.20), L6H0 Filler→Previous (+0.86), L6H8 Sink→Previous (+0.65), L4H7 Filler→Self (+0.16), L5H0 Filler→Copy (+0.68), L5H1 Sink→Copy (+0.74), L6H9 Sink→Copy (+0.23), L7H1 Sink→Copy (+0.48), L7H10 Sink→Copy (+0.35), L8H1 Filler→Copy (+0.46), L8H6 Filler→Copy (+0.72), L9H6 Sink→Copy (+0.51), L9H9 Filler→Copy (+0.39), L10H7 Sink→Copy (+0.68), L10H11 Filler→Copy (+0.35), L11H10 Sink→Copy (+0.20), L3H0 Sink→Dup (+0.19), L2H3 Previous→Filler (+0.19), L2H9 Previous→Filler (+0.22), L3H7 Previous→Filler (+0.75), L3H8 Previous→Filler (+0.38), L3H9 Sink→Filler (+0.28), L3H11 Sink→Filler (+0.25), L4H2 Sink→Filler (+0.33), L4H9 Sink→Filler (+0.15), L5H10 Sink→Filler (+0.22), L6H2 Sink→Filler (+0.66), L6H5 Sink→Filler (+0.28), L6H7 Sink→Filler (+0.25), L6H11 Sink→Filler (+0.52), L7H6 Sink→Filler (+0.35), L8H0 Sink→Filler (+0.37), L8H5 Sink→Filler (+0.22)
- common, nl_repeat, argmax labels: 39 heads; L1H5 Sink→Diffuse (+0.81), L1H7 Sink→Diffuse (+0.23), L1H9 Sink→Diffuse (+0.26), L5H10 Sink→Diffuse (+0.23), L6H7 Sink→Diffuse (+0.31), L7H7 Sink→Diffuse (+0.30), L10H4 Sink→Diffuse (+0.15), L2H1 Diffuse→Sink (+0.52), L6H4 Diffuse→Sink (+0.18), L7H0 Previous→Sink (+0.34), L8H2 Diffuse→Sink (+0.54), L9H11 Diffuse→Sink (+0.25), L10H2 Diffuse→Sink (+0.21), L11H6 Diffuse→Sink (+0.24), L2H0 Sink→Previous (+0.29), L5H6 Sink→Previous (+0.34), L6H8 Sink→Previous (+0.86), L6H11 Sink→Previous (+0.47), L8H5 Sink→Previous (+0.22), L2H10 Diffuse→Self (+0.18), L1H6 Sink→Copy (+0.98), L5H0 Sink→Copy (+0.78), L5H1 Sink→Copy (+0.74), L5H5 Sink→Copy (+1.00), L6H9 Sink→Copy (+0.79), L7H1 Sink→Copy (+0.88), L7H10 Sink→Copy (+0.63), L8H1 Sink→Copy (+0.73), L8H6 Sink→Copy (+0.84), L9H6 Sink→Copy (+0.53), L9H9 Sink→Copy (+0.45), L10H7 Sink→Copy (+1.00), L10H11 Sink→Copy (+1.00), L11H7 Sink→Copy (+0.27), L11H10 Sink→Copy (+0.27), L0H5 Self→Dup (+1.00), L3H0 Sink→Dup (+0.99), L2H9 Previous→Filler (+0.16), L7H7 Sink→Filler (+0.36)
- common, nl_repeat, mass labels: 45 heads; L7H0 Previous→Sink (+0.31), L7H11 Filler→Sink (+0.51), L8H0 Filler→Sink (+0.22), L10H2 Filler→Sink (+0.35), L10H10 Filler→Sink (+0.23), L11H2 Filler→Sink (+0.23), L11H6 Filler→Sink (+0.18), L4H1 Filler→Previous (+0.18), L4H3 Filler→Previous (+0.25), L5H6 Sink→Previous (+0.34), L6H8 Sink→Previous (+0.83), L5H0 Sink→Copy (+0.77), L5H1 Sink→Copy (+0.74), L5H5 Sink→Copy (+1.00), L6H9 Sink→Copy (+0.79), L7H1 Sink→Copy (+0.52), L7H10 Sink→Copy (+0.63), L8H1 Sink→Copy (+0.51), L8H6 Sink→Copy (+0.72), L9H6 Sink→Copy (+0.53), L9H9 Sink→Copy (+0.45), L10H7 Sink→Copy (+1.00), L10H11 Sink→Copy (+1.00), L11H10 Sink→Copy (+0.21), L0H5 Self→Dup (+1.00), L3H0 Sink→Dup (+0.99), L4H6 Sink→Filler (+0.60), L4H8 Sink→Filler (+0.23), L5H9 Sink→Filler (+0.89), L6H2 Sink→Filler (+0.98), L6H3 Sink→Filler (+0.32), L6H5 Sink→Filler (+0.88), L6H10 Sink→Filler (+0.55), L7H1 Sink→Filler (+0.47), L7H6 Sink→Filler (+0.52), L7H7 Sink→Filler (+0.85), L8H1 Sink→Filler (+0.29), L8H5 Sink→Filler (+0.29), L10H1 Sink→Filler (+0.17), L10H3 Sink→Filler (+0.29), L10H4 Sink→Filler (+0.23), L10H8 Sink→Filler (+0.27), L11H5 Sink→Filler (+0.19), L11H9 Sink→Filler (+0.26), L11H11 Sink→Filler (+0.47)
- common, placebo, argmax labels: 0 heads; none
- common, placebo, mass labels: 0 heads; none

## Transition matrices (argmax labels; nonzero source → destination counts)

- rare, rand: Diffuse→Sink 1, Diffuse→Copy 1, Sink→Diffuse 8, Sink→Previous 7, Sink→Copy 16, Sink→Dup 1, Sink→Filler 1, Previous→Filler 1, Self→Diffuse 1, Self→Dup 1
- rare, park: Diffuse→Sink 1, Sink→Diffuse 8, Sink→Previous 5, Sink→Copy 12, Sink→Dup 1, Sink→Filler 1, Previous→Filler 1, Self→Dup 1
- rare, two_tok: Diffuse→Previous 1, Diffuse→Copy 2, Diffuse→Filler 1, Sink→Diffuse 3, Sink→Previous 9, Sink→Self 1, Sink→Copy 10, Sink→Filler 1, Previous→Filler 3, Self→Previous 3, Filler→Previous 2
- rare, nl_repeat: Diffuse→Sink 1, Sink→Diffuse 8, Sink→Previous 5, Sink→Self 1, Sink→Copy 12, Sink→Dup 1, Sink→Filler 1, Previous→Sink 1, Previous→Filler 1, Self→Dup 1, Filler→Sink 1
- rare, placebo: none
- common, rand: Diffuse→Sink 2, Sink→Diffuse 7, Sink→Previous 6, Sink→Copy 8, Sink→Dup 1, Self→Diffuse 1, Self→Dup 1
- common, park: Diffuse→Sink 3, Diffuse→Self 1, Sink→Diffuse 5, Sink→Previous 5, Sink→Copy 15, Sink→Dup 1, Sink→Filler 1, Previous→Sink 1, Previous→Filler 2, Self→Dup 1
- common, two_tok: Diffuse→Sink 3, Diffuse→Previous 3, Diffuse→Self 1, Sink→Diffuse 2, Sink→Previous 8, Sink→Copy 12, Sink→Dup 1, Sink→Filler 1, Previous→Self 1, Previous→Filler 3, Self→Previous 1, Filler→Previous 7
- common, nl_repeat: Diffuse→Sink 6, Diffuse→Self 1, Sink→Diffuse 7, Sink→Previous 5, Sink→Copy 15, Sink→Dup 1, Sink→Filler 1, Previous→Sink 1, Previous→Filler 1, Self→Dup 1
- common, placebo: none

## TV distance between category profiles (per head; late mean, max)

| base | contrast | late mean | max (head) |
|---|---|---|---|
| rare | rand | 0.1927 | 0.8759 (L5H5) |
| rare | park | 0.1492 | 0.6794 (L5H5) |
| rare | two_tok | 0.1216 | 0.5462 (L5H1) |
| rare | nl_repeat | 0.1571 | 0.6869 (L5H5) |
| rare | placebo | 0.0010 | 0.0040 (L11H4) |
| common | rand | 0.1488 | 0.7819 (L5H5) |
| common | park | 0.1729 | 0.7320 (L5H5) |
| common | two_tok | 0.1127 | 0.4790 (L5H1) |
| common | nl_repeat | 0.1935 | 0.7466 (L5H5) |
| common | placebo | 0.0017 | 0.0041 (L10H4) |

## Layer profile (mean Δ over rows and heads, per layer)

| base | contrast | measure | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | L11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rare | rand | BOS | -0.001 | -0.013 | -0.015 | -0.110 | -0.021 | -0.187 | -0.212 | -0.181 | -0.013 | -0.186 | -0.180 | -0.158 |
| rare | rand | Copy | -0.000 | +0.000 | +0.000 | +0.001 | +0.001 | +0.171 | +0.075 | +0.119 | +0.040 | +0.098 | +0.159 | +0.072 |
| rare | content | Copy | +0.004 | +0.016 | -0.001 | +0.004 | +0.000 | -0.010 | -0.015 | -0.004 | +0.026 | -0.032 | -0.015 | -0.015 |
| rare | nl_vs_rand | Copy | +0.006 | +0.017 | -0.000 | +0.003 | +0.000 | -0.010 | -0.013 | +0.004 | +0.046 | -0.014 | -0.004 | -0.013 |
| rare | park | BOS | -0.001 | -0.013 | -0.021 | -0.073 | -0.020 | -0.148 | -0.196 | -0.157 | -0.057 | -0.075 | -0.091 | -0.046 |
| rare | park | Copy | +0.005 | +0.014 | +0.000 | +0.002 | +0.001 | +0.154 | +0.057 | +0.117 | +0.080 | +0.077 | +0.145 | +0.046 |
| rare | two_tok | BOS | -0.000 | +0.003 | -0.041 | -0.031 | -0.062 | -0.110 | -0.145 | -0.093 | -0.069 | -0.065 | -0.021 | +0.003 |
| rare | two_tok | Copy | +0.000 | +0.001 | +0.001 | +0.001 | +0.001 | +0.098 | +0.019 | +0.062 | +0.056 | +0.043 | +0.082 | +0.023 |
| rare | nl_repeat | BOS | -0.001 | -0.014 | -0.023 | -0.073 | -0.016 | -0.146 | -0.198 | -0.156 | -0.051 | -0.063 | -0.072 | -0.031 |
| rare | nl_repeat | Copy | +0.005 | +0.018 | -0.000 | +0.004 | +0.001 | +0.162 | +0.062 | +0.124 | +0.086 | +0.083 | +0.153 | +0.057 |
| rare | placebo | BOS | -0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.001 | +0.001 | +0.000 | -0.000 | +0.001 |
| rare | placebo | Copy | +0.000 | +0.000 | -0.000 | -0.000 | +0.000 | -0.000 | +0.000 | +0.000 | +0.000 | -0.000 | +0.000 | -0.000 |
| common | rand | BOS | -0.001 | -0.017 | -0.011 | -0.095 | -0.031 | -0.193 | -0.168 | -0.163 | -0.009 | -0.078 | -0.136 | -0.110 |
| common | rand | Copy | +0.000 | +0.000 | +0.000 | +0.001 | +0.002 | +0.143 | +0.064 | +0.096 | +0.028 | +0.048 | +0.091 | +0.041 |
| common | content | Copy | +0.008 | +0.021 | +0.000 | +0.005 | +0.001 | -0.009 | -0.015 | -0.002 | +0.049 | -0.019 | -0.010 | +0.000 |
| common | nl_vs_rand | Copy | +0.018 | +0.029 | +0.005 | +0.007 | +0.002 | +0.032 | +0.000 | +0.041 | +0.085 | +0.082 | +0.135 | +0.051 |
| common | park | BOS | -0.002 | -0.021 | -0.018 | -0.081 | -0.038 | -0.185 | -0.183 | -0.171 | -0.084 | -0.088 | -0.141 | -0.053 |
| common | park | Copy | +0.009 | +0.017 | +0.001 | +0.001 | +0.002 | +0.164 | +0.055 | +0.124 | +0.098 | +0.105 | +0.189 | +0.055 |
| common | two_tok | BOS | -0.001 | +0.000 | -0.020 | -0.035 | -0.016 | -0.074 | -0.090 | -0.066 | -0.040 | -0.035 | -0.014 | +0.019 |
| common | two_tok | Copy | -0.000 | +0.001 | +0.000 | +0.001 | +0.001 | +0.087 | +0.014 | +0.054 | +0.059 | +0.057 | +0.094 | +0.026 |
| common | nl_repeat | BOS | -0.004 | -0.027 | -0.018 | -0.082 | -0.030 | -0.180 | -0.173 | -0.160 | -0.069 | -0.080 | -0.141 | -0.047 |
| common | nl_repeat | Copy | +0.019 | +0.030 | +0.004 | +0.008 | +0.003 | +0.175 | +0.064 | +0.137 | +0.112 | +0.128 | +0.225 | +0.090 |
| common | placebo | BOS | -0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.001 | +0.001 | +0.001 | +0.001 | +0.002 | +0.002 |
| common | placebo | Copy | +0.000 | -0.000 | -0.000 | -0.000 | -0.000 | -0.000 | -0.000 | -0.000 | -0.000 | +0.000 | -0.000 | -0.000 |

## Figures

- `figures/attention_maps_rare.png`
- `figures/attention_maps_common.png`
- `figures/contrast_maps.png`
- `figures/transitions.png`
