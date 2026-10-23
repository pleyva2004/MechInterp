# Experiment 3, E3.1: query inside vs outside the phrase

Run `20261015-221059_exp3e31`. TL attention at the query (position 32); HF cross-check on 25 rows per base class. 500 rows per base class; 13 variants × 12 phrases per row (sushi, water, pizza, music, coffee, chess, rain, cars, songs, rice, bikes, money). Layout: {'context': [30, 31], 'query': 32, 'sanity': [29, 30, 31]}. Late = layers 6–11, the mean over their heads. Attention is averaged over a row's phrases before masses and labels; masks come from each contrast's treated arm (differences and placebo: PV+O's masks, Span 30, Previous 31, Self 32, phrase-position = keys 30–31).

Extraction: 156000 sequences in 967 s (161 seq/s, TL only).

## Checks

| check | result |
|---|---|
| design (base outside written positions, query included; filler holds no written id; no BOS; no repeated id in any row) | PASS (asserted per variant and base class) |
| E3.1 layout (rr+x vs PV+x and VP+x vs PV+x only at 30–31; PV+x / rr+x families only at 32; PVO+r vs rr+r only at 29–31; placebo vs rr+r only at 30–32; ids as in the table; O′ another phrase's object in O's matched pool; M and placebo ids in their pools) | PASS (asserted per base class) |
| attention rows sum to 1 | PASS (atol 0.0001, asserted per batch; max |sum − 1| 4.8e-07) |
| determinism, rare (20 rows) | PASS (max |Δ| 0.0e+00) |
| TL vs HF, rare (25 rows, 3900 sequences) | PASS (max |Δ| 2.6e-05, argmax-key agreement 100.0000%) |
| placebo switching heads ≤ 1, rare | PASS (0 argmax, 0 mass) |
| determinism, common (20 rows) | PASS (max |Δ| 0.0e+00) |
| TL vs HF, common (25 rows, 3900 sequences) | PASS (max |Δ| 2.1e-05, argmax-key agreement 99.9998%) |
| placebo switching heads ≤ 1, common | PASS (0 argmax, 0 mass) |

## Pre-registered predictions (exp3/PREDICTIONS_e31.md)

| id | verdict | detail |
|---|---|---|
| E1a | PASS | rare: per row +0.0288 [+0.0249, +0.0329] (need 95% CI > 0); per phrase +0.0288 [+0.0225, +0.0348], 12/12 > 0; common: per row +0.0328 [+0.0275, +0.0378] (need 95% CI > 0); per phrase +0.0328 [+0.0242, +0.0420], 11/12 > 0 |
| E1b | FAIL | rare: per row -0.0554 [-0.0592, -0.0517] (need 95% CI < 0); per phrase -0.0554 [-0.0684, -0.0397], 11/12 < 0; common: per row -0.0004 [-0.0052, +0.0045] (need 95% CI < 0); per phrase -0.0004 [-0.0143, +0.0144], 6/12 < 0 |
| E1c | FAIL | rare: 1 heads in L6-11 switch Sink→Span or Previous [L6H0 →Previous (+0.37)]; placebo 0; beyond placebo 1 (need ≥ 5); common: 0 heads in L6-11 switch Sink→Span or Previous [none]; placebo 0; beyond placebo 0 (need ≥ 5) |
| E1d | FAIL | rare: per phrase +0.0020 [-0.0033, +0.0075], 6/12 > 0 (need 95% CI over phrases > 0); per row +0.0020 [+0.0011, +0.0029]; common: per phrase -0.0071 [-0.0121, -0.0019], 3/12 > 0 (need 95% CI over phrases > 0); per row -0.0071 [-0.0083, -0.0058] |
| E1e | PASS | rare: per row -0.0538 [-0.0556, -0.0520] (need 95% CI < 0); per phrase -0.0538 [-0.0666, -0.0383], 11/12 < 0; common: per row -0.0257 [-0.0273, -0.0241] (need 95% CI < 0); per phrase -0.0257 [-0.0396, -0.0122], 10/12 < 0 |
| E1e-bet (bet, low confidence) | PASS | rare: per row -0.0368 [-0.0374, -0.0363] (need 95% CI < 0); per phrase -0.0368 [-0.0593, -0.0149], 9/12 < 0; common: per row -0.0268 [-0.0274, -0.0261] (need 95% CI < 0); per phrase -0.0268 [-0.0474, -0.0075], 9/12 < 0 |

## O′ per phrase (fixed in the config before any run)

Derangement (every object once as O′): True.

| phrase | O′ (from) | PV+O′ text |
|---|---|---|
| sushi | 7259 (songs) | ' I like songs' |
| water | 19780 (chess) | ' we need chess' |
| pizza | 2647 (music) | ' she ate music' |
| music | 5006 (cars) | ' they love cars' |
| coffee | 1637 (money) | ' he drinks money' |
| chess | 6891 (coffee) | ' you play coffee' |
| rain | 11464 (rice) | ' I hate rice' |
| cars | 6290 (rain) | ' we sell rain' |
| songs | 1660 (water) | ' she wrote water' |
| rice | 14256 (pizza) | ' they grow pizza' |
| bikes | 36324 (sushi) | ' he fixed sushi' |
| money | 16715 (bikes) | ' you want bikes' |

## Query log P at 32 (first 25 rows; reported, not gating)

Mean log P of each query id given the context's prefix (logits at 31). O beats x = share of row × phrase instances with log P(O) > log P(x) under PV. Lift = PV − rr, per query.

| base | context | O | O′ | M | r |
|---|---|---|---|---|---|
| rare | PV | -8.32 | -10.16 | -12.17 | -12.69 |
| rare | VP | -11.33 | -12.46 | -12.94 | -13.33 |
| rare | rr | -11.48 | -11.48 | -12.70 | -12.48 |
| common | PV | -9.76 | -11.56 | -13.02 | -9.89 |
| common | VP | -11.67 | -12.64 | -13.05 | -9.00 |
| common | rr | -12.13 | -12.13 | -13.18 | -8.54 |

| base | O beats O′ (PV) | O beats M (PV) | O beats r (PV) | lift O | lift O′ | lift M | lift r |
|---|---|---|---|---|---|---|---|
| rare | 87% | 95% | 97% | +3.15 | +1.31 | +0.53 | -0.21 |
| common | 80% | 91% | 54% | +2.38 | +0.57 | +0.16 | -1.35 |

Per phrase, PV context: mean log P(O) − log P(O′).

| base | sushi | water | pizza | music | coffee | chess | rain | cars | songs | rice | bikes | money |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rare | +0.08 | +3.52 | +2.01 | +0.69 | +2.16 | +3.41 | +1.49 | +1.74 | +0.13 | +1.07 | +0.72 | +5.04 |
| common | -1.72 | +4.83 | +0.35 | +1.95 | +1.44 | +2.79 | +1.73 | +0.38 | -0.23 | +3.24 | +1.23 | +5.70 |

## Late pooled per-row Δ, every contrast and measure

Two-arm: a − b. Difference: (plus) − (minus), per row. 95% / 90% CIs bootstrap rows; verdict: positive / negative (95% CI excludes 0), equivalent (TOST, 90% CI inside ±0.005), else inconclusive. Per phrase: bootstrap over the per-phrase means, with a sign count.

| base | contrast | arms | measure | Δ | 95% CI | 90% CI | Wilcoxon p | verdict | per phrase [95% CI] | phrases > 0 |
|---|---|---|---|---|---|---|---|---|---|---|
| rare | C_O | PV+O − rr+O | BOS | -0.0436 | [-0.0465, -0.0407] | [-0.0461, -0.0412] | 7.5e-75 | negative | -0.0436 [-0.0562, -0.0273] | 1/12 |
| rare | C_O | PV+O − rr+O | Previous | +0.0088 | [+0.0074, +0.0101] | [+0.0076, +0.0099] | 9.5e-33 | positive | +0.0088 [+0.0051, +0.0126] | 10/12 |
| rare | C_O | PV+O − rr+O | Self | +0.0050 | [+0.0044, +0.0056] | [+0.0045, +0.0055] | 2.7e-44 | positive | +0.0050 [+0.0025, +0.0076] | 10/12 |
| rare | C_O | PV+O − rr+O | Span | +0.0020 | [+0.0011, +0.0029] | [+0.0013, +0.0028] | 8.4e-08 | positive | +0.0020 [-0.0008, +0.0048] | 8/12 |
| rare | C_O | PV+O − rr+O | Filler | +0.0278 | [+0.0252, +0.0306] | [+0.0256, +0.0302] | 2.1e-54 | positive | +0.0278 [+0.0136, +0.0390] | 10/12 |
| rare | C_O | PV+O − rr+O | Phrase | +0.0108 | [+0.0093, +0.0123] | [+0.0095, +0.0120] | 2.2e-35 | positive | +0.0108 [+0.0063, +0.0153] | 11/12 |
| rare | C_Op | PV+Op − rr+Op | BOS | -0.0068 | [-0.0096, -0.0040] | [-0.0092, -0.0044] | 2.7e-05 | negative | -0.0068 [-0.0271, +0.0130] | 5/12 |
| rare | C_Op | PV+Op − rr+Op | Previous | +0.0085 | [+0.0071, +0.0098] | [+0.0074, +0.0096] | 4.2e-31 | positive | +0.0085 [+0.0054, +0.0118] | 12/12 |
| rare | C_Op | PV+Op − rr+Op | Self | +0.0017 | [+0.0011, +0.0022] | [+0.0012, +0.0021] | 5.8e-11 | positive | +0.0017 [-0.0008, +0.0040] | 8/12 |
| rare | C_Op | PV+Op − rr+Op | Span | -0.0030 | [-0.0039, -0.0021] | [-0.0037, -0.0022] | 2.1e-08 | negative | -0.0030 [-0.0055, -0.0003] | 3/12 |
| rare | C_Op | PV+Op − rr+Op | Filler | -0.0004 | [-0.0029, +0.0024] | [-0.0025, +0.0020] | 0.4 | equivalent | -0.0004 [-0.0166, +0.0156] | 6/12 |
| rare | C_Op | PV+Op − rr+Op | Phrase | +0.0055 | [+0.0040, +0.0070] | [+0.0042, +0.0068] | 1.3e-13 | positive | +0.0055 [+0.0019, +0.0091] | 9/12 |
| rare | C_M | PV+M − rr+M | BOS | +0.0102 | [+0.0078, +0.0124] | [+0.0082, +0.0121] | 8e-16 | positive | +0.0102 [+0.0046, +0.0153] | 10/12 |
| rare | C_M | PV+M − rr+M | Previous | +0.0035 | [+0.0023, +0.0047] | [+0.0025, +0.0045] | 1.4e-09 | positive | +0.0035 [+0.0001, +0.0068] | 9/12 |
| rare | C_M | PV+M − rr+M | Self | +0.0002 | [-0.0002, +0.0007] | [-0.0001, +0.0006] | 0.15 | equivalent | +0.0002 [-0.0007, +0.0013] | 4/12 |
| rare | C_M | PV+M − rr+M | Span | -0.0050 | [-0.0058, -0.0042] | [-0.0057, -0.0044] | 2.7e-29 | negative | -0.0050 [-0.0065, -0.0036] | 0/12 |
| rare | C_M | PV+M − rr+M | Filler | -0.0089 | [-0.0111, -0.0066] | [-0.0108, -0.0069] | 4.5e-13 | negative | -0.0089 [-0.0155, -0.0020] | 3/12 |
| rare | C_M | PV+M − rr+M | Phrase | -0.0016 | [-0.0028, -0.0002] | [-0.0027, -0.0004] | 0.053 | negative | -0.0016 [-0.0046, +0.0015] | 4/12 |
| rare | C_r | PV+r − rr+r | BOS | +0.0118 | [+0.0078, +0.0159] | [+0.0084, +0.0153] | 1.6e-08 | positive | +0.0118 [+0.0076, +0.0154] | 11/12 |
| rare | C_r | PV+r − rr+r | Previous | -0.0004 | [-0.0019, +0.0011] | [-0.0017, +0.0008] | 0.95 | equivalent | -0.0004 [-0.0037, +0.0030] | 6/12 |
| rare | C_r | PV+r − rr+r | Self | -0.0002 | [-0.0011, +0.0006] | [-0.0009, +0.0004] | 0.27 | equivalent | -0.0002 [-0.0008, +0.0004] | 4/12 |
| rare | C_r | PV+r − rr+r | Span | -0.0059 | [-0.0069, -0.0049] | [-0.0068, -0.0051] | 4e-27 | negative | -0.0059 [-0.0073, -0.0046] | 0/12 |
| rare | C_r | PV+r − rr+r | Filler | -0.0053 | [-0.0090, -0.0016] | [-0.0083, -0.0022] | 0.006 | negative | -0.0053 [-0.0102, +0.0001] | 3/12 |
| rare | C_r | PV+r − rr+r | Phrase | -0.0063 | [-0.0081, -0.0046] | [-0.0078, -0.0049] | 3e-12 | negative | -0.0063 [-0.0096, -0.0029] | 3/12 |
| rare | order_O | PV+O − VP+O | BOS | -0.0000 | [-0.0012, +0.0011] | [-0.0010, +0.0009] | 0.75 | equivalent | -0.0000 [-0.0200, +0.0169] | 7/12 |
| rare | order_O | PV+O − VP+O | Previous | -0.0022 | [-0.0026, -0.0017] | [-0.0025, -0.0018] | 3.3e-17 | negative | -0.0022 [-0.0114, +0.0069] | 5/12 |
| rare | order_O | PV+O − VP+O | Self | -0.0043 | [-0.0045, -0.0041] | [-0.0045, -0.0041] | 7.2e-82 | negative | -0.0043 [-0.0089, +0.0007] | 3/12 |
| rare | order_O | PV+O − VP+O | Span | -0.0056 | [-0.0060, -0.0052] | [-0.0059, -0.0052] | 2.1e-69 | negative | -0.0056 [-0.0099, -0.0013] | 2/12 |
| rare | order_O | PV+O − VP+O | Filler | +0.0120 | [+0.0109, +0.0132] | [+0.0110, +0.0130] | 5.8e-56 | positive | +0.0120 [-0.0047, +0.0299] | 7/12 |
| rare | order_O | PV+O − VP+O | Phrase | -0.0077 | [-0.0081, -0.0073] | [-0.0081, -0.0073] | 6.7e-78 | negative | -0.0077 [-0.0143, -0.0006] | 2/12 |
| rare | order_r | PV+r − VP+r | BOS | +0.0195 | [+0.0158, +0.0231] | [+0.0164, +0.0227] | 1.4e-27 | positive | +0.0195 [+0.0101, +0.0280] | 11/12 |
| rare | order_r | PV+r − VP+r | Previous | +0.0011 | [+0.0002, +0.0020] | [+0.0003, +0.0019] | 0.24 | positive | +0.0011 [-0.0037, +0.0059] | 7/12 |
| rare | order_r | PV+r − VP+r | Self | -0.0003 | [-0.0009, +0.0003] | [-0.0008, +0.0002] | 0.51 | equivalent | -0.0003 [-0.0008, +0.0004] | 4/12 |
| rare | order_r | PV+r − VP+r | Span | -0.0108 | [-0.0115, -0.0100] | [-0.0114, -0.0102] | 1e-80 | negative | -0.0108 [-0.0134, -0.0078] | 0/12 |
| rare | order_r | PV+r − VP+r | Filler | -0.0096 | [-0.0126, -0.0065] | [-0.0121, -0.0070] | 9.3e-12 | negative | -0.0096 [-0.0164, -0.0025] | 2/12 |
| rare | order_r | PV+r − VP+r | Phrase | -0.0097 | [-0.0107, -0.0087] | [-0.0105, -0.0089] | 5.9e-54 | negative | -0.0097 [-0.0136, -0.0056] | 2/12 |
| rare | sanity | PVO+r − rr+r | BOS | +0.0288 | [+0.0249, +0.0329] | [+0.0255, +0.0321] | 1.4e-38 | positive | +0.0288 [+0.0225, +0.0348] | 12/12 |
| rare | sanity | PVO+r − rr+r | Previous | -0.0142 | [-0.0157, -0.0128] | [-0.0155, -0.0130] | 4.8e-57 | negative | -0.0142 [-0.0165, -0.0122] | 0/12 |
| rare | sanity | PVO+r − rr+r | Self | +0.0002 | [-0.0007, +0.0010] | [-0.0006, +0.0009] | 0.16 | equivalent | +0.0002 [-0.0004, +0.0008] | 6/12 |
| rare | sanity | PVO+r − rr+r | Span | -0.0114 | [-0.0129, -0.0100] | [-0.0127, -0.0102] | 3.2e-44 | negative | -0.0114 [-0.0133, -0.0091] | 0/12 |
| rare | sanity | PVO+r − rr+r | Filler | -0.0033 | [-0.0065, -0.0001] | [-0.0060, -0.0006] | 0.18 | negative | -0.0033 [-0.0076, +0.0012] | 4/12 |
| rare | sanity | PVO+r − rr+r | Phrase | -0.0256 | [-0.0277, -0.0238] | [-0.0273, -0.0240] | 7.5e-71 | negative | -0.0256 [-0.0292, -0.0224] | 0/12 |
| rare | placebo | matched_b − matched | BOS | -0.0015 | [-0.0054, +0.0021] | [-0.0046, +0.0016] | 0.41 | equivalent | -0.0015 [-0.0057, +0.0023] | 7/12 |
| rare | placebo | matched_b − matched | Previous | +0.0002 | [-0.0003, +0.0007] | [-0.0002, +0.0006] | 0.37 | equivalent | +0.0002 [-0.0002, +0.0007] | 7/12 |
| rare | placebo | matched_b − matched | Self | +0.0003 | [-0.0003, +0.0008] | [-0.0002, +0.0007] | 0.42 | equivalent | +0.0003 [-0.0003, +0.0009] | 8/12 |
| rare | placebo | matched_b − matched | Span | +0.0001 | [-0.0004, +0.0006] | [-0.0003, +0.0005] | 0.84 | equivalent | +0.0001 [-0.0004, +0.0006] | 6/12 |
| rare | placebo | matched_b − matched | Filler | +0.0010 | [-0.0019, +0.0041] | [-0.0015, +0.0035] | 0.54 | equivalent | +0.0010 [-0.0025, +0.0045] | 7/12 |
| rare | placebo | matched_b − matched | Phrase | +0.0003 | [-0.0004, +0.0010] | [-0.0003, +0.0009] | 0.33 | equivalent | +0.0003 [-0.0004, +0.0009] | 9/12 |
| rare | interaction | (PV+O − rr+O) − (PV+r − rr+r) | BOS | -0.0554 | [-0.0592, -0.0517] | [-0.0586, -0.0522] | 1.7e-74 | negative | -0.0554 [-0.0684, -0.0397] | 1/12 |
| rare | interaction | (PV+O − rr+O) − (PV+r − rr+r) | Previous | +0.0091 | [+0.0079, +0.0104] | [+0.0081, +0.0102] | 6.7e-36 | positive | +0.0091 [+0.0055, +0.0126] | 11/12 |
| rare | interaction | (PV+O − rr+O) − (PV+r − rr+r) | Self | +0.0052 | [+0.0043, +0.0061] | [+0.0045, +0.0060] | 1.7e-27 | positive | +0.0052 [+0.0028, +0.0075] | 10/12 |
| rare | interaction | (PV+O − rr+O) − (PV+r − rr+r) | Span | +0.0079 | [+0.0072, +0.0087] | [+0.0073, +0.0086] | 2.6e-60 | positive | +0.0079 [+0.0057, +0.0101] | 12/12 |
| rare | interaction | (PV+O − rr+O) − (PV+r − rr+r) | Filler | +0.0331 | [+0.0300, +0.0361] | [+0.0305, +0.0356] | 6.2e-62 | positive | +0.0331 [+0.0205, +0.0441] | 11/12 |
| rare | interaction | (PV+O − rr+O) − (PV+r − rr+r) | Phrase | +0.0171 | [+0.0156, +0.0186] | [+0.0158, +0.0184] | 2.9e-61 | positive | +0.0171 [+0.0119, +0.0217] | 11/12 |
| rare | completion | (PV+O − rr+O) − (PV+M − rr+M) | BOS | -0.0538 | [-0.0556, -0.0520] | [-0.0552, -0.0523] | 1.3e-83 | negative | -0.0538 [-0.0666, -0.0383] | 1/12 |
| rare | completion | (PV+O − rr+O) − (PV+M − rr+M) | Previous | +0.0053 | [+0.0047, +0.0059] | [+0.0048, +0.0058] | 1.1e-46 | positive | +0.0053 [+0.0016, +0.0089] | 10/12 |
| rare | completion | (PV+O − rr+O) − (PV+M − rr+M) | Self | +0.0048 | [+0.0042, +0.0053] | [+0.0043, +0.0052] | 4.7e-46 | positive | +0.0048 [+0.0020, +0.0072] | 10/12 |
| rare | completion | (PV+O − rr+O) − (PV+M − rr+M) | Span | +0.0070 | [+0.0066, +0.0074] | [+0.0067, +0.0073] | 2e-77 | positive | +0.0070 [+0.0048, +0.0092] | 12/12 |
| rare | completion | (PV+O − rr+O) − (PV+M − rr+M) | Filler | +0.0367 | [+0.0351, +0.0382] | [+0.0354, +0.0380] | 3e-83 | positive | +0.0367 [+0.0244, +0.0481] | 11/12 |
| rare | completion | (PV+O − rr+O) − (PV+M − rr+M) | Phrase | +0.0123 | [+0.0116, +0.0130] | [+0.0117, +0.0129] | 9e-79 | positive | +0.0123 [+0.0074, +0.0170] | 11/12 |
| rare | plausibility | (PV+O − rr+O) − (PV+Op − rr+Op) | BOS | -0.0368 | [-0.0374, -0.0363] | [-0.0373, -0.0363] | 1.3e-83 | negative | -0.0368 [-0.0593, -0.0149] | 3/12 |
| rare | plausibility | (PV+O − rr+O) − (PV+Op − rr+Op) | Previous | +0.0003 | [+0.0001, +0.0005] | [+0.0001, +0.0004] | 0.034 | positive | +0.0003 [-0.0036, +0.0043] | 5/12 |
| rare | plausibility | (PV+O − rr+O) − (PV+Op − rr+Op) | Self | +0.0033 | [+0.0032, +0.0035] | [+0.0032, +0.0034] | 1.4e-83 | positive | +0.0033 [+0.0003, +0.0065] | 6/12 |
| rare | plausibility | (PV+O − rr+O) − (PV+Op − rr+Op) | Span | +0.0050 | [+0.0049, +0.0051] | [+0.0049, +0.0051] | 1.3e-83 | positive | +0.0050 [+0.0025, +0.0074] | 11/12 |
| rare | plausibility | (PV+O − rr+O) − (PV+Op − rr+Op) | Filler | +0.0282 | [+0.0277, +0.0287] | [+0.0277, +0.0286] | 1.3e-83 | positive | +0.0282 [+0.0116, +0.0449] | 10/12 |
| rare | plausibility | (PV+O − rr+O) − (PV+Op − rr+Op) | Phrase | +0.0053 | [+0.0050, +0.0055] | [+0.0051, +0.0055] | 1.3e-83 | positive | +0.0053 [-0.0003, +0.0110] | 7/12 |
| rare | order_x_query | (PV+O − VP+O) − (PV+r − VP+r) | BOS | -0.0196 | [-0.0233, -0.0159] | [-0.0228, -0.0164] | 1.2e-27 | negative | -0.0196 [-0.0352, -0.0067] | 3/12 |
| rare | order_x_query | (PV+O − VP+O) − (PV+r − VP+r) | Previous | -0.0032 | [-0.0041, -0.0024] | [-0.0040, -0.0025] | 8.1e-13 | negative | -0.0032 [-0.0100, +0.0038] | 5/12 |
| rare | order_x_query | (PV+O − VP+O) − (PV+r − VP+r) | Self | -0.0040 | [-0.0046, -0.0034] | [-0.0045, -0.0035] | 6.5e-34 | negative | -0.0040 [-0.0085, +0.0009] | 2/12 |
| rare | order_x_query | (PV+O − VP+O) − (PV+r − VP+r) | Span | +0.0052 | [+0.0045, +0.0059] | [+0.0046, +0.0058] | 1.5e-40 | positive | +0.0052 [+0.0021, +0.0083] | 10/12 |
| rare | order_x_query | (PV+O − VP+O) − (PV+r − VP+r) | Filler | +0.0216 | [+0.0185, +0.0246] | [+0.0190, +0.0242] | 1.9e-40 | positive | +0.0216 [+0.0073, +0.0359] | 10/12 |
| rare | order_x_query | (PV+O − VP+O) − (PV+r − VP+r) | Phrase | +0.0020 | [+0.0011, +0.0029] | [+0.0012, +0.0028] | 0.00032 | positive | +0.0020 [-0.0033, +0.0075] | 6/12 |
| common | C_O | PV+O − rr+O | BOS | +0.0083 | [+0.0057, +0.0110] | [+0.0060, +0.0105] | 9.2e-14 | positive | +0.0083 [-0.0054, +0.0242] | 5/12 |
| common | C_O | PV+O − rr+O | Previous | +0.0236 | [+0.0224, +0.0249] | [+0.0226, +0.0247] | 6.9e-79 | positive | +0.0236 [+0.0193, +0.0281] | 12/12 |
| common | C_O | PV+O − rr+O | Self | +0.0114 | [+0.0108, +0.0120] | [+0.0109, +0.0119] | 8.8e-78 | positive | +0.0114 [+0.0076, +0.0154] | 12/12 |
| common | C_O | PV+O − rr+O | Span | -0.0006 | [-0.0015, +0.0003] | [-0.0014, +0.0001] | 0.85 | equivalent | -0.0006 [-0.0028, +0.0017] | 4/12 |
| common | C_O | PV+O − rr+O | Filler | -0.0427 | [-0.0458, -0.0396] | [-0.0454, -0.0401] | 6.9e-64 | negative | -0.0427 [-0.0552, -0.0323] | 0/12 |
| common | C_O | PV+O − rr+O | Phrase | +0.0230 | [+0.0216, +0.0244] | [+0.0218, +0.0242] | 4.8e-74 | positive | +0.0230 [+0.0175, +0.0283] | 12/12 |
| common | C_Op | PV+Op − rr+Op | BOS | +0.0351 | [+0.0324, +0.0378] | [+0.0328, +0.0374] | 1.1e-63 | positive | +0.0351 [+0.0167, +0.0515] | 10/12 |
| common | C_Op | PV+Op − rr+Op | Previous | +0.0258 | [+0.0245, +0.0270] | [+0.0247, +0.0269] | 1e-80 | positive | +0.0258 [+0.0215, +0.0309] | 12/12 |
| common | C_Op | PV+Op − rr+Op | Self | +0.0065 | [+0.0059, +0.0071] | [+0.0060, +0.0070] | 1.2e-54 | positive | +0.0065 [+0.0027, +0.0105] | 10/12 |
| common | C_Op | PV+Op − rr+Op | Span | -0.0054 | [-0.0063, -0.0045] | [-0.0062, -0.0046] | 2.6e-23 | negative | -0.0054 [-0.0077, -0.0030] | 1/12 |
| common | C_Op | PV+Op − rr+Op | Filler | -0.0620 | [-0.0651, -0.0589] | [-0.0647, -0.0594] | 4.7e-77 | negative | -0.0620 [-0.0743, -0.0489] | 0/12 |
| common | C_Op | PV+Op − rr+Op | Phrase | +0.0204 | [+0.0190, +0.0218] | [+0.0192, +0.0217] | 4.6e-69 | positive | +0.0204 [+0.0155, +0.0259] | 12/12 |
| common | C_M | PV+M − rr+M | BOS | +0.0340 | [+0.0315, +0.0366] | [+0.0319, +0.0362] | 1.3e-65 | positive | +0.0340 [+0.0271, +0.0408] | 12/12 |
| common | C_M | PV+M − rr+M | Previous | +0.0217 | [+0.0204, +0.0230] | [+0.0206, +0.0228] | 5.3e-78 | positive | +0.0217 [+0.0170, +0.0263] | 12/12 |
| common | C_M | PV+M − rr+M | Self | +0.0044 | [+0.0038, +0.0050] | [+0.0039, +0.0049] | 1.8e-37 | positive | +0.0044 [+0.0030, +0.0059] | 12/12 |
| common | C_M | PV+M − rr+M | Span | -0.0057 | [-0.0065, -0.0048] | [-0.0064, -0.0050] | 4.6e-29 | negative | -0.0057 [-0.0073, -0.0040] | 0/12 |
| common | C_M | PV+M − rr+M | Filler | -0.0545 | [-0.0575, -0.0515] | [-0.0571, -0.0520] | 7.1e-76 | negative | -0.0545 [-0.0621, -0.0464] | 0/12 |
| common | C_M | PV+M − rr+M | Phrase | +0.0161 | [+0.0146, +0.0174] | [+0.0148, +0.0173] | 6.1e-58 | positive | +0.0161 [+0.0119, +0.0199] | 12/12 |
| common | C_r | PV+r − rr+r | BOS | +0.0088 | [+0.0036, +0.0139] | [+0.0045, +0.0131] | 2.3e-07 | positive | +0.0088 [+0.0019, +0.0152] | 10/12 |
| common | C_r | PV+r − rr+r | Previous | +0.0182 | [+0.0167, +0.0197] | [+0.0170, +0.0195] | 3e-67 | positive | +0.0182 [+0.0146, +0.0220] | 12/12 |
| common | C_r | PV+r − rr+r | Self | +0.0026 | [+0.0018, +0.0034] | [+0.0019, +0.0032] | 2.7e-08 | positive | +0.0026 [+0.0019, +0.0032] | 12/12 |
| common | C_r | PV+r − rr+r | Span | -0.0029 | [-0.0041, -0.0018] | [-0.0039, -0.0021] | 1.6e-06 | negative | -0.0029 [-0.0050, -0.0009] | 4/12 |
| common | C_r | PV+r − rr+r | Filler | -0.0266 | [-0.0309, -0.0221] | [-0.0302, -0.0228] | 3.6e-29 | negative | -0.0266 [-0.0347, -0.0176] | 1/12 |
| common | C_r | PV+r − rr+r | Phrase | +0.0153 | [+0.0134, +0.0171] | [+0.0137, +0.0168] | 8.9e-40 | positive | +0.0153 [+0.0126, +0.0178] | 12/12 |
| common | order_O | PV+O − VP+O | BOS | +0.0208 | [+0.0197, +0.0220] | [+0.0199, +0.0218] | 5.2e-80 | positive | +0.0208 [+0.0069, +0.0339] | 9/12 |
| common | order_O | PV+O − VP+O | Previous | +0.0044 | [+0.0039, +0.0048] | [+0.0040, +0.0047] | 2.1e-49 | positive | +0.0044 [-0.0022, +0.0114] | 6/12 |
| common | order_O | PV+O − VP+O | Self | +0.0002 | [-0.0002, +0.0005] | [-0.0001, +0.0005] | 0.32 | equivalent | +0.0002 [-0.0046, +0.0060] | 5/12 |
| common | order_O | PV+O − VP+O | Span | -0.0089 | [-0.0093, -0.0085] | [-0.0093, -0.0086] | 1.9e-83 | negative | -0.0089 [-0.0129, -0.0051] | 1/12 |
| common | order_O | PV+O − VP+O | Filler | -0.0165 | [-0.0175, -0.0154] | [-0.0174, -0.0156] | 4e-72 | negative | -0.0165 [-0.0273, -0.0045] | 2/12 |
| common | order_O | PV+O − VP+O | Phrase | -0.0046 | [-0.0050, -0.0041] | [-0.0049, -0.0042] | 6.9e-53 | negative | -0.0046 [-0.0099, +0.0016] | 3/12 |
| common | order_r | PV+r − VP+r | BOS | -0.0084 | [-0.0119, -0.0049] | [-0.0114, -0.0053] | 0.00013 | negative | -0.0084 [-0.0229, +0.0049] | 5/12 |
| common | order_r | PV+r − VP+r | Previous | +0.0142 | [+0.0131, +0.0153] | [+0.0132, +0.0152] | 4.7e-71 | positive | +0.0142 [+0.0079, +0.0203] | 11/12 |
| common | order_r | PV+r − VP+r | Self | +0.0019 | [+0.0012, +0.0024] | [+0.0013, +0.0024] | 9.1e-09 | positive | +0.0019 [+0.0007, +0.0031] | 9/12 |
| common | order_r | PV+r − VP+r | Span | -0.0117 | [-0.0126, -0.0108] | [-0.0125, -0.0110] | 1.3e-75 | negative | -0.0117 [-0.0159, -0.0078] | 0/12 |
| common | order_r | PV+r − VP+r | Filler | +0.0040 | [+0.0015, +0.0065] | [+0.0019, +0.0061] | 0.019 | positive | +0.0040 [-0.0067, +0.0153] | 7/12 |
| common | order_r | PV+r − VP+r | Phrase | +0.0025 | [+0.0012, +0.0037] | [+0.0015, +0.0035] | 0.00022 | positive | +0.0025 [-0.0015, +0.0065] | 6/12 |
| common | sanity | PVO+r − rr+r | BOS | +0.0328 | [+0.0275, +0.0378] | [+0.0284, +0.0371] | 1.6e-32 | positive | +0.0328 [+0.0242, +0.0420] | 11/12 |
| common | sanity | PVO+r − rr+r | Previous | +0.0148 | [+0.0133, +0.0163] | [+0.0135, +0.0161] | 4.1e-55 | positive | +0.0148 [+0.0112, +0.0182] | 12/12 |
| common | sanity | PVO+r − rr+r | Self | +0.0002 | [-0.0006, +0.0010] | [-0.0005, +0.0008] | 0.58 | equivalent | +0.0002 [-0.0005, +0.0009] | 6/12 |
| common | sanity | PVO+r − rr+r | Span | -0.0007 | [-0.0025, +0.0012] | [-0.0022, +0.0009] | 0.65 | equivalent | -0.0007 [-0.0034, +0.0023] | 4/12 |
| common | sanity | PVO+r − rr+r | Filler | -0.0471 | [-0.0514, -0.0427] | [-0.0508, -0.0435] | 5.4e-58 | negative | -0.0471 [-0.0560, -0.0388] | 0/12 |
| common | sanity | PVO+r − rr+r | Phrase | +0.0141 | [+0.0116, +0.0167] | [+0.0120, +0.0163] | 7.2e-23 | positive | +0.0141 [+0.0114, +0.0168] | 12/12 |
| common | placebo | matched_b − matched | BOS | +0.0023 | [-0.0004, +0.0052] | [+0.0000, +0.0048] | 0.072 | equivalent | +0.0023 [-0.0004, +0.0053] | 7/12 |
| common | placebo | matched_b − matched | Previous | -0.0001 | [-0.0006, +0.0004] | [-0.0005, +0.0004] | 0.93 | equivalent | -0.0001 [-0.0006, +0.0004] | 4/12 |
| common | placebo | matched_b − matched | Self | -0.0007 | [-0.0014, -0.0000] | [-0.0013, -0.0001] | 0.084 | negative | -0.0007 [-0.0013, -0.0000] | 4/12 |
| common | placebo | matched_b − matched | Span | -0.0001 | [-0.0006, +0.0003] | [-0.0005, +0.0002] | 0.83 | equivalent | -0.0001 [-0.0005, +0.0002] | 5/12 |
| common | placebo | matched_b − matched | Filler | -0.0014 | [-0.0036, +0.0008] | [-0.0033, +0.0004] | 0.26 | equivalent | -0.0014 [-0.0038, +0.0006] | 5/12 |
| common | placebo | matched_b − matched | Phrase | -0.0002 | [-0.0009, +0.0005] | [-0.0008, +0.0004] | 0.71 | equivalent | -0.0002 [-0.0009, +0.0004] | 4/12 |
| common | interaction | (PV+O − rr+O) − (PV+r − rr+r) | BOS | -0.0004 | [-0.0052, +0.0045] | [-0.0043, +0.0036] | 0.093 | equivalent | -0.0004 [-0.0143, +0.0144] | 6/12 |
| common | interaction | (PV+O − rr+O) − (PV+r − rr+r) | Previous | +0.0054 | [+0.0039, +0.0068] | [+0.0042, +0.0066] | 8.5e-14 | positive | +0.0054 [+0.0012, +0.0095] | 10/12 |
| common | interaction | (PV+O − rr+O) − (PV+r − rr+r) | Self | +0.0089 | [+0.0080, +0.0097] | [+0.0081, +0.0096] | 3.7e-49 | positive | +0.0089 [+0.0053, +0.0125] | 12/12 |
| common | interaction | (PV+O − rr+O) − (PV+r − rr+r) | Span | +0.0023 | [+0.0016, +0.0031] | [+0.0017, +0.0030] | 1.4e-09 | positive | +0.0023 [+0.0007, +0.0040] | 8/12 |
| common | interaction | (PV+O − rr+O) − (PV+r − rr+r) | Filler | -0.0161 | [-0.0203, -0.0119] | [-0.0196, -0.0127] | 2e-13 | negative | -0.0161 [-0.0274, -0.0050] | 2/12 |
| common | interaction | (PV+O − rr+O) − (PV+r − rr+r) | Phrase | +0.0077 | [+0.0060, +0.0094] | [+0.0063, +0.0092] | 1.2e-16 | positive | +0.0077 [+0.0029, +0.0123] | 11/12 |
| common | completion | (PV+O − rr+O) − (PV+M − rr+M) | BOS | -0.0257 | [-0.0273, -0.0241] | [-0.0270, -0.0244] | 7.5e-78 | negative | -0.0257 [-0.0396, -0.0122] | 2/12 |
| common | completion | (PV+O − rr+O) − (PV+M − rr+M) | Previous | +0.0019 | [+0.0014, +0.0024] | [+0.0015, +0.0023] | 9.5e-13 | positive | +0.0019 [-0.0030, +0.0068] | 8/12 |
| common | completion | (PV+O − rr+O) − (PV+M − rr+M) | Self | +0.0070 | [+0.0065, +0.0075] | [+0.0066, +0.0074] | 5e-70 | positive | +0.0070 [+0.0035, +0.0102] | 11/12 |
| common | completion | (PV+O − rr+O) − (PV+M − rr+M) | Span | +0.0050 | [+0.0048, +0.0053] | [+0.0048, +0.0053] | 1.7e-79 | positive | +0.0050 [+0.0034, +0.0068] | 12/12 |
| common | completion | (PV+O − rr+O) − (PV+M − rr+M) | Filler | +0.0118 | [+0.0104, +0.0132] | [+0.0106, +0.0130] | 5e-42 | positive | +0.0118 [+0.0019, +0.0227] | 8/12 |
| common | completion | (PV+O − rr+O) − (PV+M − rr+M) | Phrase | +0.0069 | [+0.0063, +0.0075] | [+0.0065, +0.0074] | 6.7e-63 | positive | +0.0069 [+0.0018, +0.0120] | 9/12 |
| common | plausibility | (PV+O − rr+O) − (PV+Op − rr+Op) | BOS | -0.0268 | [-0.0274, -0.0261] | [-0.0273, -0.0262] | 1.3e-83 | negative | -0.0268 [-0.0474, -0.0075] | 3/12 |
| common | plausibility | (PV+O − rr+O) − (PV+Op − rr+Op) | Previous | -0.0022 | [-0.0023, -0.0020] | [-0.0023, -0.0020] | 2.4e-69 | negative | -0.0022 [-0.0071, +0.0030] | 5/12 |
| common | plausibility | (PV+O − rr+O) − (PV+Op − rr+Op) | Self | +0.0049 | [+0.0048, +0.0050] | [+0.0048, +0.0050] | 1.3e-83 | positive | +0.0049 [-0.0000, +0.0096] | 9/12 |
| common | plausibility | (PV+O − rr+O) − (PV+Op − rr+Op) | Span | +0.0047 | [+0.0046, +0.0049] | [+0.0046, +0.0048] | 1.3e-83 | positive | +0.0047 [+0.0029, +0.0068] | 12/12 |
| common | plausibility | (PV+O − rr+O) − (PV+Op − rr+Op) | Filler | +0.0193 | [+0.0187, +0.0199] | [+0.0188, +0.0198] | 1.3e-83 | positive | +0.0193 [+0.0054, +0.0340] | 8/12 |
| common | plausibility | (PV+O − rr+O) − (PV+Op − rr+Op) | Phrase | +0.0026 | [+0.0024, +0.0028] | [+0.0024, +0.0027] | 2.2e-70 | positive | +0.0026 [-0.0027, +0.0084] | 7/12 |
| common | order_x_query | (PV+O − VP+O) − (PV+r − VP+r) | BOS | +0.0292 | [+0.0256, +0.0328] | [+0.0261, +0.0323] | 1.5e-43 | positive | +0.0292 [+0.0123, +0.0481] | 10/12 |
| common | order_x_query | (PV+O − VP+O) − (PV+r − VP+r) | Previous | -0.0099 | [-0.0109, -0.0087] | [-0.0107, -0.0089] | 3.3e-47 | negative | -0.0099 [-0.0166, -0.0029] | 3/12 |
| common | order_x_query | (PV+O − VP+O) − (PV+r − VP+r) | Self | -0.0017 | [-0.0024, -0.0010] | [-0.0023, -0.0011] | 1.1e-05 | negative | -0.0017 [-0.0069, +0.0043] | 4/12 |
| common | order_x_query | (PV+O − VP+O) − (PV+r − VP+r) | Span | +0.0028 | [+0.0020, +0.0035] | [+0.0022, +0.0034] | 2.3e-10 | positive | +0.0028 [-0.0008, +0.0065] | 7/12 |
| common | order_x_query | (PV+O − VP+O) − (PV+r − VP+r) | Filler | -0.0205 | [-0.0232, -0.0178] | [-0.0227, -0.0182] | 6.6e-39 | negative | -0.0205 [-0.0377, -0.0067] | 3/12 |
| common | order_x_query | (PV+O − VP+O) − (PV+r − VP+r) | Phrase | -0.0071 | [-0.0083, -0.0058] | [-0.0081, -0.0061] | 4.4e-28 | negative | -0.0071 [-0.0121, -0.0019] | 3/12 |

## logit(BOS) check on the differences (late mean per row; mass clipped to [1e-06, 1 − 1e-06])

| base | contrast | Δ logit | 95% CI | verdict | same sign as the mass Δ |
|---|---|---|---|---|---|
| rare | interaction | -0.2794 | [-0.3029, -0.2571] | negative | yes |
| rare | completion | -0.2708 | [-0.2805, -0.2613] | negative | yes |
| rare | plausibility | -0.2014 | [-0.2045, -0.1984] | negative | yes |
| rare | order_x_query | -0.0921 | [-0.1138, -0.0707] | negative | yes |
| common | interaction | +0.0190 | [-0.0073, +0.0462] | inconclusive | no |
| common | completion | -0.1170 | [-0.1250, -0.1091] | negative | yes |
| common | plausibility | -0.1315 | [-0.1345, -0.1283] | negative | yes |
| common | order_x_query | +0.1720 | [+0.1520, +0.1920] | positive | yes |

## Meaningful heads per contrast (BH q ≤ 0.05 over 144 heads, |Δ| ≥ 0.02; + / −; — = empty mask)

| base | contrast | BOS | Previous | Self | Copy | Dup | Delimiter | Span | Filler | Phrase |
|---|---|---|---|---|---|---|---|---|---|---|
| rare | C_O | 35 / 61 | 36 / 26 | 21 / 14 | — | — | — | 27 / 28 | 57 / 33 | 47 / 52 |
| rare | C_Op | 38 / 48 | 33 / 25 | 9 / 8 | — | — | — | 22 / 34 | 39 / 36 | 39 / 47 |
| rare | C_M | 51 / 35 | 31 / 24 | 6 / 8 | — | — | — | 22 / 28 | 23 / 39 | 38 / 37 |
| rare | C_r | 45 / 30 | 23 / 19 | 7 / 6 | — | — | — | 19 / 26 | 22 / 29 | 32 / 40 |
| rare | order_O | 35 / 41 | 30 / 37 | 11 / 13 | — | — | — | 24 / 33 | 50 / 12 | 14 / 37 |
| rare | order_r | 50 / 23 | 23 / 32 | 6 / 2 | — | — | — | 16 / 30 | 26 / 27 | 6 / 32 |
| rare | sanity | 45 / 20 | 4 / 33 | 9 / 1 | — | — | — | 21 / 40 | 21 / 23 | 20 / 50 |
| rare | placebo | 0 / 0 | 0 / 0 | 0 / 0 | — | — | — | 0 / 0 | 0 / 0 | 0 / 0 |
| rare | interaction | 22 / 59 | 29 / 17 | 16 / 9 | — | — | — | 18 / 8 | 50 / 22 | 38 / 15 |
| rare | completion | 21 / 59 | 21 / 21 | 15 / 9 | — | — | — | 15 / 9 | 55 / 16 | 28 / 20 |
| rare | plausibility | 8 / 62 | 21 / 17 | 4 / 2 | — | — | — | 13 / 1 | 45 / 12 | 27 / 12 |
| rare | order_x_query | 24 / 40 | 23 / 24 | 6 / 9 | — | — | — | 21 / 16 | 45 / 8 | 22 / 17 |
| common | C_O | 44 / 39 | 47 / 11 | 28 / 7 | — | — | — | 18 / 16 | 13 / 83 | 55 / 17 |
| common | C_Op | 58 / 23 | 47 / 11 | 19 / 5 | — | — | — | 16 / 23 | 2 / 92 | 48 / 16 |
| common | C_M | 59 / 20 | 46 / 8 | 10 / 5 | — | — | — | 18 / 18 | 3 / 102 | 55 / 16 |
| common | C_r | 35 / 20 | 34 / 10 | 2 / 3 | — | — | — | 18 / 15 | 7 / 63 | 42 / 15 |
| common | order_O | 55 / 23 | 33 / 36 | 15 / 6 | — | — | — | 22 / 37 | 22 / 32 | 17 / 33 |
| common | order_r | 18 / 29 | 35 / 24 | 3 / 4 | — | — | — | 16 / 34 | 26 / 8 | 14 / 23 |
| common | sanity | 54 / 11 | 35 / 12 | 5 / 1 | — | — | — | 26 / 17 | 8 / 87 | 54 / 21 |
| common | placebo | 0 / 0 | 0 / 0 | 0 / 0 | — | — | — | 0 / 0 | 0 / 0 | 0 / 0 |
| common | interaction | 36 / 39 | 32 / 20 | 25 / 6 | — | — | — | 15 / 8 | 21 / 57 | 36 / 25 |
| common | completion | 23 / 50 | 24 / 21 | 21 / 4 | — | — | — | 12 / 7 | 38 / 24 | 25 / 18 |
| common | plausibility | 11 / 50 | 20 / 18 | 13 / 4 | — | — | — | 12 / 0 | 34 / 11 | 25 / 14 |
| common | order_x_query | 59 / 16 | 23 / 36 | 9 / 5 | — | — | — | 21 / 19 | 11 / 43 | 18 / 26 |

## Role switches (McNemar, BH, ≥ 15% shift)

- rare, placebo switching heads (either direction): argmax 0 [], mass 0
- rare, C_O, argmax labels: 32 switches; L10H2 Sink→Diffuse (+0.36), L10H4 Sink→Diffuse (+0.17), L11H1 Sink→Diffuse (+0.32), L11H3 Sink→Diffuse (+0.17), L11H7 Sink→Diffuse (+0.64), L11H9 Sink→Diffuse (+0.30), L1H9 Diffuse→Sink (+0.17), L3H11 Previous→Sink (+0.36), L4H3 Previous→Sink (+0.17), L5H0 Filler→Sink (+0.20), L5H6 Previous→Sink (+0.37), L6H10 Diffuse→Sink (+0.23), L7H0 Previous→Sink (+0.21), L7H7 Diffuse→Sink (+0.66), L8H6 Diffuse→Sink (+0.46), L10H11 Diffuse→Sink (+0.35), L1H0 Self→Previous (+0.67), L1H1 Diffuse→Previous (+0.55), L2H3 Span→Previous (+0.32), L2H8 Span→Previous (+0.55), L3H2 Self→Previous (+0.35), L3H8 Span→Previous (+0.26), L4H0 Sink→Previous (+0.97), L6H0 Sink→Previous (+0.37), L11H8 Diffuse→Previous (+0.17), L0H7 Previous→Span (+0.88), L2H5 Previous→Span (+0.31), L3H1 Sink→Span (+0.60), L3H6 Previous→Span (+0.52), L5H3 Sink→Span (+0.88), L10H2 Sink→Filler (+0.26), L10H3 Sink→Filler (+0.23)
- rare, C_O, mass labels: 41 switches; L3H4 Filler→Sink (+0.39), L4H3 Previous→Sink (+0.19), L4H8 Filler→Sink (+0.75), L5H0 Filler→Sink (+0.82), L5H6 Previous→Sink (+0.37), L5H10 Filler→Sink (+0.41), L6H4 Filler→Sink (+0.39), L6H10 Filler→Sink (+0.65), L7H0 Previous→Sink (+0.21), L7H6 Filler→Sink (+0.20), L7H7 Filler→Sink (+0.18), L8H1 Filler→Sink (+0.38), L10H8 Filler→Sink (+0.20), L1H0 Filler→Previous (+0.16), L2H3 Span→Previous (+0.15), L2H8 Filler→Previous (+0.64), L3H2 Self→Previous (+0.35), L3H8 Span→Previous (+0.20), L4H0 Sink→Previous (+0.97), L6H0 Sink→Previous (+0.37), L4H7 Filler→Self (+0.22), L2H5 Previous→Span (+0.19), L3H1 Sink→Span (+0.43), L3H6 Previous→Span (+0.21), L2H0 Sink→Filler (+0.18), L2H9 Previous→Filler (+0.80), L3H6 Previous→Filler (+0.38), L3H11 Previous→Filler (+0.26), L4H2 Sink→Filler (+0.39), L5H3 Sink→Filler (+0.31), L8H0 Sink→Filler (+0.48), L8H2 Sink→Filler (+0.20), L8H5 Sink→Filler (+0.23), L10H0 Sink→Filler (+0.24), L10H1 Sink→Filler (+0.32), L10H6 Sink→Filler (+0.31), L10H10 Sink→Filler (+0.64), L11H2 Sink→Filler (+0.43), L11H4 Sink→Filler (+0.41), L11H6 Sink→Filler (+0.43), L11H11 Sink→Filler (+0.16)
- rare, C_Op, argmax labels: 25 switches; L1H9 Diffuse→Sink (+0.18), L4H3 Previous→Sink (+0.17), L5H0 Filler→Sink (+0.20), L5H6 Previous→Sink (+0.37), L6H10 Diffuse→Sink (+0.23), L7H7 Diffuse→Sink (+0.73), L7H11 Diffuse→Sink (+0.26), L8H6 Diffuse→Sink (+0.61), L10H11 Diffuse→Sink (+0.43), L1H0 Self→Previous (+0.66), L1H1 Diffuse→Previous (+0.53), L2H3 Span→Previous (+0.33), L2H5 Span→Previous (+0.23), L2H8 Span→Previous (+0.55), L3H2 Self→Previous (+0.35), L3H8 Span→Previous (+0.27), L3H11 Span→Previous (+0.19), L4H0 Sink→Previous (+0.96), L5H4 Sink→Previous (+0.33), L7H0 Sink→Previous (+0.19), L8H7 Sink→Previous (+0.24), L3H6 Previous→Self (+0.65), L0H7 Previous→Span (+0.80), L3H1 Sink→Span (+0.55), L5H3 Sink→Span (+0.78)
- rare, C_Op, mass labels: 38 switches; L3H4 Filler→Sink (+0.37), L4H3 Previous→Sink (+0.20), L4H6 Filler→Sink (+0.16), L4H8 Filler→Sink (+0.70), L5H0 Filler→Sink (+0.77), L5H6 Previous→Sink (+0.37), L5H10 Filler→Sink (+0.51), L6H0 Filler→Sink (+0.16), L6H4 Filler→Sink (+0.27), L6H10 Filler→Sink (+0.43), L7H6 Filler→Sink (+0.19), L7H7 Filler→Sink (+0.22), L8H1 Filler→Sink (+0.39), L8H2 Filler→Sink (+0.16), L9H1 Filler→Sink (+0.16), L10H8 Filler→Sink (+0.35), L2H3 Filler→Previous (+0.36), L2H5 Filler→Previous (+0.37), L2H8 Filler→Previous (+0.64), L3H2 Self→Previous (+0.35), L3H8 Span→Previous (+0.30), L4H0 Sink→Previous (+0.96), L5H4 Sink→Previous (+0.33), L7H0 Sink→Previous (+0.19), L8H7 Sink→Previous (+0.24), L3H6 Previous→Self (+0.31), L4H7 Filler→Self (+0.20), L3H1 Sink→Span (+0.42), L2H0 Sink→Filler (+0.17), L2H9 Previous→Filler (+0.52), L3H6 Previous→Filler (+0.35), L4H2 Sink→Filler (+0.48), L5H3 Sink→Filler (+0.37), L8H0 Sink→Filler (+0.32), L10H10 Sink→Filler (+0.29), L11H2 Sink→Filler (+0.32), L11H4 Sink→Filler (+0.24), L11H6 Sink→Filler (+0.30)
- rare, C_M, argmax labels: 22 switches; L5H3 Sink→Diffuse (+0.21), L4H3 Previous→Sink (+0.34), L5H0 Diffuse→Sink (+0.24), L5H6 Previous→Sink (+0.33), L7H7 Diffuse→Sink (+0.57), L8H6 Diffuse→Sink (+0.58), L10H4 Diffuse→Sink (+0.17), L10H11 Diffuse→Sink (+0.30), L1H0 Self→Previous (+0.64), L1H1 Diffuse→Previous (+0.41), L2H3 Span→Previous (+0.34), L2H5 Span→Previous (+0.21), L2H8 Sink→Previous (+0.66), L3H2 Span→Previous (+0.27), L3H8 Sink→Previous (+0.32), L4H0 Sink→Previous (+0.16), L7H0 Sink→Previous (+0.15), L3H6 Previous→Self (+0.49), L0H7 Diffuse→Span (+0.74), L3H1 Sink→Span (+0.48), L3H6 Previous→Span (+0.17), L5H3 Sink→Span (+0.23)
- rare, C_M, mass labels: 25 switches; L4H3 Previous→Sink (+0.39), L4H6 Filler→Sink (+0.23), L4H8 Filler→Sink (+0.63), L5H0 Filler→Sink (+0.35), L5H6 Previous→Sink (+0.33), L5H10 Filler→Sink (+0.38), L6H0 Filler→Sink (+0.23), L6H10 Filler→Sink (+0.16), L7H7 Filler→Sink (+0.23), L8H1 Filler→Sink (+0.27), L10H8 Filler→Sink (+0.27), L2H3 Filler→Previous (+0.40), L2H5 Filler→Previous (+0.35), L2H8 Filler→Previous (+0.70), L3H2 Span→Previous (+0.29), L3H8 Sink→Previous (+0.34), L4H0 Sink→Previous (+0.16), L7H0 Sink→Previous (+0.15), L3H6 Previous→Self (+0.30), L4H7 Filler→Self (+0.34), L3H1 Sink→Span (+0.41), L2H0 Sink→Filler (+0.16), L2H9 Previous→Filler (+0.26), L3H6 Previous→Filler (+0.17), L5H3 Sink→Filler (+0.28)
- rare, C_r, argmax labels: 22 switches; L3H11 Span→Sink (+0.20), L4H3 Previous→Sink (+0.26), L5H0 Filler→Sink (+0.15), L5H6 Previous→Sink (+0.30), L6H8 Previous→Sink (+0.20), L7H7 Diffuse→Sink (+0.26), L8H6 Diffuse→Sink (+0.27), L1H1 Diffuse→Previous (+0.26), L2H3 Span→Previous (+0.34), L2H5 Span→Previous (+0.25), L2H8 Span→Previous (+0.57), L2H9 Span→Previous (+0.16), L3H2 Self→Previous (+0.17), L3H8 Sink→Previous (+0.28), L4H0 Sink→Previous (+0.16), L5H4 Sink→Previous (+0.19), L3H6 Previous→Self (+0.25), L0H7 Filler→Span (+0.46), L2H0 Sink→Span (+0.20), L3H1 Previous→Span (+0.22), L3H6 Previous→Span (+0.17), L5H3 Sink→Span (+0.16)
- rare, C_r, mass labels: 19 switches; L3H4 Filler→Sink (+0.16), L4H3 Previous→Sink (+0.21), L4H8 Filler→Sink (+0.23), L5H6 Previous→Sink (+0.30), L6H8 Previous→Sink (+0.20), L8H1 Filler→Sink (+0.17), L10H8 Filler→Sink (+0.17), L2H3 Filler→Previous (+0.31), L2H5 Filler→Previous (+0.33), L2H8 Filler→Previous (+0.64), L3H2 Self→Previous (+0.19), L3H8 Sink→Previous (+0.31), L4H0 Sink→Previous (+0.16), L5H4 Sink→Previous (+0.19), L3H6 Previous→Self (+0.20), L3H1 Previous→Span (+0.16), L2H9 Previous→Filler (+0.29), L4H2 Sink→Filler (+0.15), L5H3 Sink→Filler (+0.23)
- rare, order_O, argmax labels: 20 switches; L11H7 Sink→Diffuse (+0.37), L11H9 Sink→Diffuse (+0.16), L2H0 Previous→Sink (+0.83), L3H1 Previous→Sink (+0.29), L5H4 Previous→Sink (+0.66), L5H6 Previous→Sink (+0.28), L6H11 Previous→Sink (+0.43), L7H0 Previous→Sink (+0.98), L7H11 Diffuse→Sink (+0.17), L11H3 Diffuse→Sink (+0.17), L1H0 Span→Previous (+1.00), L2H8 Span→Previous (+1.00), L4H0 Sink→Previous (+0.99), L11H8 Self→Previous (+0.18), L3H6 Previous→Self (+0.23), L0H7 Previous→Span (+1.00), L2H5 Previous→Span (+0.51), L3H1 Previous→Span (+0.66), L3H6 Previous→Span (+0.69), L5H3 Previous→Span (+0.94)
- rare, order_O, mass labels: 24 switches; L3H1 Previous→Sink (+0.18), L5H4 Previous→Sink (+0.66), L5H6 Previous→Sink (+0.28), L6H11 Previous→Sink (+0.43), L7H0 Previous→Sink (+0.98), L9H1 Filler→Sink (+0.24), L1H0 Filler→Previous (+0.19), L2H8 Span→Previous (+1.00), L4H0 Sink→Previous (+0.99), L2H5 Previous→Span (+0.33), L3H1 Previous→Span (+0.48), L3H6 Previous→Span (+0.35), L0H7 Previous→Filler (+0.71), L2H5 Previous→Filler (+0.37), L2H9 Previous→Filler (+0.44), L3H6 Previous→Filler (+0.55), L4H2 Sink→Filler (+0.34), L4H6 Sink→Filler (+0.25), L5H10 Sink→Filler (+0.18), L6H4 Sink→Filler (+0.52), L8H0 Sink→Filler (+0.36), L8H2 Sink→Filler (+0.15), L8H5 Sink→Filler (+0.18), L11H4 Sink→Filler (+0.21)
- rare, order_r, argmax labels: 17 switches; L3H1 Previous→Sink (+0.27), L5H4 Previous→Sink (+0.20), L5H6 Previous→Sink (+0.40), L6H0 Previous→Sink (+0.15), L7H0 Previous→Sink (+0.19), L2H8 Span→Previous (+0.93), L2H9 Span→Previous (+0.28), L3H8 Span→Previous (+0.42), L3H11 Sink→Previous (+0.18), L4H0 Sink→Previous (+0.18), L3H6 Previous→Self (+0.42), L0H7 Previous→Span (+0.65), L2H0 Previous→Span (+0.30), L2H2 Previous→Span (+0.25), L3H1 Previous→Span (+0.28), L3H6 Previous→Span (+0.39), L5H3 Previous→Span (+0.19)
- rare, order_r, mass labels: 20 switches; L3H1 Previous→Sink (+0.16), L5H4 Previous→Sink (+0.20), L5H6 Previous→Sink (+0.40), L7H0 Previous→Sink (+0.19), L2H3 Filler→Previous (+0.19), L2H8 Span→Previous (+0.92), L2H9 Filler→Previous (+0.18), L3H8 Span→Previous (+0.43), L4H0 Sink→Previous (+0.18), L3H6 Previous→Self (+0.35), L0H7 Previous→Span (+0.16), L2H2 Previous→Span (+0.25), L3H1 Previous→Span (+0.22), L3H6 Previous→Span (+0.26), L0H7 Previous→Filler (+0.27), L3H6 Previous→Filler (+0.28), L5H0 Sink→Filler (+0.18), L5H3 Sink→Filler (+0.16), L6H4 Sink→Filler (+0.17), L6H10 Sink→Filler (+0.18)
- rare, sanity, argmax labels: 13 switches; L1H1 Previous→Diffuse (+0.34), L2H6 Previous→Sink (+0.16), L3H11 Previous→Sink (+0.31), L4H3 Previous→Sink (+0.23), L5H6 Previous→Sink (+0.31), L6H0 Previous→Sink (+0.17), L6H8 Previous→Sink (+0.20), L7H0 Previous→Sink (+0.18), L2H9 Span→Previous (+0.21), L3H8 Sink→Previous (+0.18), L3H2 Span→Self (+0.17), L0H7 Previous→Span (+0.21), L2H8 Previous→Span (+0.61)
- rare, sanity, mass labels: 24 switches; L3H11 Previous→Sink (+0.27), L4H3 Previous→Sink (+0.23), L5H6 Previous→Sink (+0.31), L5H10 Filler→Sink (+0.31), L6H0 Previous→Sink (+0.20), L6H4 Filler→Sink (+0.16), L6H8 Previous→Sink (+0.20), L7H0 Previous→Sink (+0.20), L2H2 Span→Previous (+0.17), L2H9 Span→Previous (+0.43), L3H7 Span→Previous (+0.16), L3H2 Previous→Self (+0.17), L3H6 Span→Self (+0.15), L4H7 Filler→Self (+0.21), L0H7 Filler→Span (+0.42), L1H0 Previous→Span (+0.15), L2H0 Sink→Span (+0.28), L2H3 Previous→Span (+0.41), L2H5 Previous→Span (+0.36), L2H8 Previous→Span (+0.57), L3H1 Sink→Span (+0.24), L4H2 Sink→Filler (+0.21), L5H3 Sink→Filler (+0.35), L6H10 Sink→Filler (+0.18)
- rare, placebo, argmax labels: 0 switches; none
- rare, placebo, mass labels: 0 switches; none
- common, placebo switching heads (either direction): argmax 0 [], mass 0
- common, C_O, argmax labels: 29 switches; L1H5 Sink→Diffuse (+0.17), L2H0 Previous→Diffuse (+0.26), L2H10 Self→Diffuse (+0.29), L11H7 Sink→Diffuse (+0.15), L2H1 Diffuse→Sink (+0.17), L3H11 Previous→Sink (+0.27), L4H1 Previous→Sink (+0.15), L4H3 Previous→Sink (+0.27), L4H5 Previous→Sink (+0.25), L5H2 Filler→Sink (+0.45), L5H4 Previous→Sink (+0.57), L5H6 Previous→Sink (+0.55), L7H0 Previous→Sink (+0.35), L8H6 Diffuse→Sink (+0.63), L1H0 Self→Previous (+0.80), L1H1 Diffuse→Previous (+0.50), L2H2 Span→Previous (+0.16), L2H3 Span→Previous (+0.50), L2H8 Sink→Previous (+0.55), L2H9 Span→Previous (+0.24), L3H2 Span→Previous (+0.48), L3H8 Sink→Previous (+0.46), L4H0 Sink→Previous (+0.90), L6H0 Diffuse→Previous (+0.64), L11H8 Diffuse→Previous (+0.35), L0H7 Previous→Span (+0.76), L2H5 Previous→Span (+0.31), L3H6 Previous→Span (+0.40), L5H3 Sink→Span (+0.69)
- common, C_O, mass labels: 44 switches; L3H1 Filler→Sink (+0.17), L3H11 Filler→Sink (+0.35), L4H1 Filler→Sink (+0.21), L4H3 Filler→Sink (+0.48), L4H5 Filler→Sink (+0.43), L4H6 Filler→Sink (+0.55), L4H8 Filler→Sink (+0.56), L5H0 Filler→Sink (+0.18), L5H2 Filler→Sink (+0.61), L5H4 Filler→Sink (+0.70), L5H6 Previous→Sink (+0.55), L5H10 Filler→Sink (+0.34), L6H3 Filler→Sink (+0.17), L6H4 Filler→Sink (+0.44), L6H7 Filler→Sink (+0.50), L6H8 Previous→Sink (+0.17), L6H10 Filler→Sink (+0.41), L6H11 Filler→Sink (+0.27), L7H0 Filler→Sink (+0.44), L7H7 Filler→Sink (+0.32), L7H11 Filler→Sink (+0.28), L8H6 Filler→Sink (+0.51), L8H7 Filler→Sink (+0.22), L9H3 Filler→Sink (+0.23), L9H8 Filler→Sink (+0.25), L10H8 Filler→Sink (+0.23), L10H11 Filler→Sink (+0.52), L1H0 Filler→Previous (+0.56), L2H2 Span→Previous (+0.17), L2H3 Filler→Previous (+0.31), L2H8 Filler→Previous (+0.71), L3H2 Filler→Previous (+0.52), L3H8 Filler→Previous (+0.54), L4H0 Sink→Previous (+0.90), L6H0 Filler→Previous (+0.69), L0H7 Previous→Span (+0.25), L2H5 Previous→Span (+0.26), L3H6 Previous→Span (+0.30), L2H9 Previous→Filler (+0.31), L3H7 Previous→Filler (+0.22), L8H0 Sink→Filler (+0.40), L11H1 Sink→Filler (+0.28), L11H7 Sink→Filler (+0.24), L11H10 Sink→Filler (+0.17)
- common, C_Op, argmax labels: 27 switches; L2H0 Previous→Diffuse (+0.38), L2H10 Self→Diffuse (+0.34), L4H3 Previous→Sink (+0.27), L4H5 Previous→Sink (+0.25), L5H2 Filler→Sink (+0.45), L5H4 Span→Sink (+0.17), L5H6 Previous→Sink (+0.55), L6H0 Diffuse→Sink (+0.19), L8H6 Diffuse→Sink (+0.63), L11H6 Diffuse→Sink (+0.15), L1H0 Self→Previous (+0.80), L1H1 Diffuse→Previous (+0.48), L2H2 Span→Previous (+0.16), L2H3 Span→Previous (+0.52), L2H5 Span→Previous (+0.30), L2H8 Sink→Previous (+0.55), L2H9 Span→Previous (+0.25), L3H2 Span→Previous (+0.48), L3H8 Sink→Previous (+0.46), L4H0 Sink→Previous (+0.90), L6H0 Diffuse→Previous (+0.31), L7H0 Sink→Previous (+0.31), L8H7 Sink→Previous (+0.51), L11H8 Diffuse→Previous (+0.30), L3H6 Previous→Self (+0.71), L0H7 Previous→Span (+0.67), L5H3 Sink→Span (+0.47)
- common, C_Op, mass labels: 45 switches; L3H1 Filler→Sink (+0.23), L3H11 Filler→Sink (+0.24), L4H1 Filler→Sink (+0.21), L4H3 Filler→Sink (+0.48), L4H5 Filler→Sink (+0.43), L4H6 Filler→Sink (+0.57), L4H8 Filler→Sink (+0.53), L5H0 Filler→Sink (+0.18), L5H2 Filler→Sink (+0.61), L5H4 Filler→Sink (+0.32), L5H6 Previous→Sink (+0.55), L5H10 Filler→Sink (+0.40), L6H0 Filler→Sink (+0.24), L6H3 Filler→Sink (+0.19), L6H4 Filler→Sink (+0.22), L6H7 Filler→Sink (+0.66), L6H8 Previous→Sink (+0.17), L6H10 Filler→Sink (+0.30), L6H11 Filler→Sink (+0.26), L7H7 Filler→Sink (+0.32), L7H11 Filler→Sink (+0.33), L8H6 Filler→Sink (+0.50), L9H3 Filler→Sink (+0.24), L9H8 Filler→Sink (+0.25), L10H1 Filler→Sink (+0.26), L10H4 Filler→Sink (+0.15), L10H8 Filler→Sink (+0.28), L10H11 Filler→Sink (+0.47), L11H9 Filler→Sink (+0.26), L11H11 Filler→Sink (+0.18), L1H0 Filler→Previous (+0.24), L2H2 Span→Previous (+0.17), L2H3 Filler→Previous (+0.49), L2H5 Filler→Previous (+0.54), L2H8 Filler→Previous (+0.71), L2H9 Filler→Previous (+0.20), L3H2 Filler→Previous (+0.52), L3H8 Filler→Previous (+0.54), L4H0 Sink→Previous (+0.90), L5H4 Filler→Previous (+0.17), L6H0 Filler→Previous (+0.34), L7H0 Filler→Previous (+0.42), L8H7 Sink→Previous (+0.52), L3H6 Previous→Self (+0.61), L8H0 Sink→Filler (+0.29)
- common, C_M, argmax labels: 24 switches; L2H0 Sink→Diffuse (+0.26), L3H11 Previous→Sink (+0.17), L4H3 Previous→Sink (+0.36), L5H2 Filler→Sink (+0.29), L5H4 Previous→Sink (+0.35), L5H6 Previous→Sink (+0.50), L6H0 Diffuse→Sink (+0.32), L6H8 Previous→Sink (+0.20), L8H6 Diffuse→Sink (+0.61), L11H6 Diffuse→Sink (+0.15), L1H0 Self→Previous (+0.75), L1H1 Diffuse→Previous (+0.28), L2H2 Self→Previous (+0.21), L2H3 Span→Previous (+0.55), L2H5 Span→Previous (+0.29), L2H8 Sink→Previous (+0.60), L2H9 Span→Previous (+0.23), L3H2 Span→Previous (+0.42), L3H8 Sink→Previous (+0.44), L4H0 Sink→Previous (+0.36), L7H0 Sink→Previous (+0.30), L11H8 Diffuse→Previous (+0.18), L3H6 Previous→Self (+0.62), L0H7 Previous→Span (+0.61)
- common, C_M, mass labels: 42 switches; L3H1 Filler→Sink (+0.20), L3H11 Filler→Sink (+0.38), L4H1 Filler→Sink (+0.21), L4H3 Filler→Sink (+0.56), L4H5 Filler→Sink (+0.26), L4H6 Filler→Sink (+0.60), L4H8 Filler→Sink (+0.39), L5H0 Filler→Sink (+0.19), L5H2 Filler→Sink (+0.50), L5H4 Filler→Sink (+0.57), L5H6 Previous→Sink (+0.50), L5H10 Filler→Sink (+0.42), L6H0 Filler→Sink (+0.42), L6H3 Filler→Sink (+0.18), L6H7 Filler→Sink (+0.60), L6H8 Previous→Sink (+0.25), L6H11 Filler→Sink (+0.16), L7H7 Filler→Sink (+0.31), L8H6 Filler→Sink (+0.26), L9H8 Filler→Sink (+0.23), L10H1 Filler→Sink (+0.23), L10H4 Filler→Sink (+0.22), L10H8 Filler→Sink (+0.27), L10H10 Filler→Sink (+0.19), L10H11 Filler→Sink (+0.33), L11H1 Filler→Sink (+0.18), L11H5 Filler→Sink (+0.17), L11H9 Filler→Sink (+0.33), L11H11 Filler→Sink (+0.26), L1H0 Filler→Previous (+0.15), L2H2 Self→Previous (+0.22), L2H3 Filler→Previous (+0.49), L2H5 Filler→Previous (+0.47), L2H8 Filler→Previous (+0.73), L2H9 Filler→Previous (+0.36), L3H2 Filler→Previous (+0.47), L3H8 Filler→Previous (+0.55), L4H0 Sink→Previous (+0.36), L6H0 Filler→Previous (+0.22), L7H0 Filler→Previous (+0.39), L3H6 Previous→Self (+0.59), L4H7 Filler→Self (+0.17)
- common, C_r, argmax labels: 19 switches; L4H3 Previous→Sink (+0.40), L6H0 Diffuse→Sink (+0.16), L6H4 Diffuse→Sink (+0.21), L6H8 Previous→Sink (+0.20), L8H6 Diffuse→Sink (+0.37), L1H1 Diffuse→Previous (+0.19), L2H3 Span→Previous (+0.53), L2H5 Span→Previous (+0.25), L2H8 Span→Previous (+0.49), L2H9 Span→Previous (+0.28), L3H2 Span→Previous (+0.41), L3H7 Span→Previous (+0.19), L3H8 Sink→Previous (+0.38), L7H0 Span→Previous (+0.18), L8H7 Sink→Previous (+0.17), L10H9 Sink→Previous (+0.23), L2H2 Previous→Span (+0.16), L2H4 Previous→Span (+0.24), L3H6 Previous→Span (+0.24)
- common, C_r, mass labels: 20 switches; L3H1 Filler→Sink (+0.20), L3H11 Filler→Sink (+0.20), L4H3 Filler→Sink (+0.46), L4H6 Filler→Sink (+0.21), L5H2 Filler→Sink (+0.23), L6H0 Filler→Sink (+0.33), L6H8 Previous→Sink (+0.21), L9H8 Filler→Sink (+0.18), L2H3 Filler→Previous (+0.46), L2H5 Filler→Previous (+0.32), L2H8 Filler→Previous (+0.63), L2H9 Filler→Previous (+0.28), L3H2 Span→Previous (+0.44), L3H7 Span→Previous (+0.19), L3H8 Filler→Previous (+0.40), L7H0 Filler→Previous (+0.26), L8H7 Filler→Previous (+0.18), L2H4 Previous→Span (+0.20), L3H6 Previous→Span (+0.20), L6H10 Sink→Filler (+0.16)
- common, order_O, argmax labels: 19 switches; L2H0 Previous→Diffuse (+0.26), L2H0 Previous→Sink (+0.45), L3H1 Previous→Sink (+0.57), L4H1 Previous→Sink (+0.20), L4H5 Previous→Sink (+0.31), L5H4 Previous→Sink (+0.85), L5H6 Previous→Sink (+0.27), L6H11 Previous→Sink (+0.35), L7H0 Previous→Sink (+0.54), L9H11 Filler→Sink (+0.18), L1H0 Span→Previous (+1.00), L2H8 Span→Previous (+1.00), L4H0 Span→Previous (+0.99), L11H8 Diffuse→Previous (+0.38), L3H6 Previous→Self (+0.36), L0H7 Previous→Span (+0.98), L2H5 Previous→Span (+0.48), L3H6 Previous→Span (+0.62), L5H3 Previous→Span (+0.75)
- common, order_O, mass labels: 31 switches; L3H1 Previous→Sink (+0.47), L4H1 Previous→Sink (+0.19), L4H5 Previous→Sink (+0.31), L5H4 Previous→Sink (+0.82), L5H6 Previous→Sink (+0.27), L6H11 Previous→Sink (+0.33), L7H0 Previous→Sink (+0.54), L7H11 Filler→Sink (+0.42), L8H6 Filler→Sink (+0.35), L10H1 Filler→Sink (+0.17), L10H4 Filler→Sink (+0.15), L10H8 Filler→Sink (+0.26), L10H10 Filler→Sink (+0.15), L10H11 Filler→Sink (+0.31), L11H3 Filler→Sink (+0.24), L1H0 Filler→Previous (+0.63), L2H8 Span→Previous (+1.00), L4H0 Span→Previous (+0.99), L3H6 Previous→Self (+0.22), L0H7 Previous→Span (+0.29), L2H5 Previous→Span (+0.37), L3H6 Previous→Span (+0.47), L0H7 Previous→Filler (+0.47), L2H5 Previous→Filler (+0.32), L2H9 Previous→Filler (+0.44), L3H6 Previous→Filler (+0.30), L3H7 Previous→Filler (+0.29), L4H2 Sink→Filler (+0.31), L6H4 Sink→Filler (+0.28), L8H0 Sink→Filler (+0.34), L8H5 Sink→Filler (+0.16)
- common, order_r, argmax labels: 17 switches; L1H1 Previous→Diffuse (+0.15), L2H0 Previous→Diffuse (+0.16), L2H8 Span→Previous (+0.79), L2H9 Span→Previous (+0.21), L3H7 Sink→Previous (+0.27), L3H8 Sink→Previous (+0.51), L3H11 Sink→Previous (+0.17), L7H0 Sink→Previous (+0.21), L8H7 Sink→Previous (+0.20), L10H9 Sink→Previous (+0.29), L3H6 Previous→Self (+0.21), L0H7 Previous→Span (+0.35), L2H0 Previous→Span (+0.27), L2H2 Previous→Span (+0.30), L2H4 Previous→Span (+0.38), L3H3 Previous→Span (+0.19), L3H6 Previous→Span (+0.46)
- common, order_r, mass labels: 16 switches; L3H1 Previous→Sink (+0.16), L2H8 Span→Previous (+0.82), L2H9 Span→Previous (+0.23), L3H7 Sink→Previous (+0.25), L3H8 Sink→Previous (+0.52), L3H11 Filler→Previous (+0.15), L7H0 Sink→Previous (+0.21), L8H7 Sink→Previous (+0.21), L10H9 Sink→Previous (+0.18), L3H6 Previous→Self (+0.19), L2H2 Previous→Span (+0.28), L2H4 Previous→Span (+0.29), L3H3 Previous→Span (+0.17), L3H6 Previous→Span (+0.37), L3H6 Previous→Filler (+0.23), L5H0 Sink→Filler (+0.29)
- common, sanity, argmax labels: 21 switches; L0H7 Previous→Diffuse (+0.21), L1H1 Previous→Diffuse (+0.39), L2H0 Previous→Diffuse (+0.22), L3H11 Previous→Sink (+0.22), L4H3 Previous→Sink (+0.35), L5H2 Filler→Sink (+0.16), L5H6 Previous→Sink (+0.19), L6H0 Diffuse→Sink (+0.24), L6H4 Diffuse→Sink (+0.37), L6H8 Previous→Sink (+0.34), L7H0 Previous→Sink (+0.33), L8H6 Diffuse→Sink (+0.40), L2H2 Span→Previous (+0.21), L2H3 Span→Previous (+0.29), L2H5 Span→Previous (+0.31), L2H9 Span→Previous (+0.29), L3H2 Span→Previous (+0.37), L3H6 Span→Previous (+0.45), L3H7 Span→Previous (+0.24), L2H8 Previous→Span (+0.66), L1H0 Previous→Filler (+0.18)
- common, sanity, mass labels: 26 switches; L3H1 Filler→Sink (+0.16), L3H11 Filler→Sink (+0.31), L4H3 Span→Sink (+0.42), L5H2 Filler→Sink (+0.19), L5H6 Previous→Sink (+0.19), L5H10 Filler→Sink (+0.28), L6H0 Filler→Sink (+0.40), L6H4 Filler→Sink (+0.21), L6H8 Previous→Sink (+0.36), L7H0 Previous→Sink (+0.39), L8H6 Filler→Sink (+0.16), L9H8 Filler→Sink (+0.24), L10H9 Filler→Sink (+0.18), L11H5 Filler→Sink (+0.17), L11H9 Filler→Sink (+0.22), L2H2 Span→Previous (+0.21), L2H9 Span→Previous (+0.46), L3H2 Span→Previous (+0.38), L3H6 Span→Previous (+0.49), L3H7 Span→Previous (+0.28), L2H3 Filler→Span (+0.29), L2H5 Previous→Span (+0.19), L2H8 Previous→Span (+0.59), L0H7 Previous→Filler (+0.19), L2H0 Previous→Filler (+0.16), L6H10 Sink→Filler (+0.19)
- common, placebo, argmax labels: 0 switches; none
- common, placebo, mass labels: 0 switches; none

## Transition matrices (argmax labels; nonzero source → destination counts)

- rare, C_O: Diffuse→Sink 5, Diffuse→Previous 2, Sink→Diffuse 6, Sink→Previous 2, Sink→Span 2, Sink→Filler 2, Previous→Sink 4, Previous→Span 3, Self→Previous 2, Span→Previous 3, Filler→Sink 1
- rare, C_Op: Diffuse→Sink 6, Diffuse→Previous 1, Sink→Previous 4, Sink→Span 2, Previous→Sink 2, Previous→Self 1, Previous→Span 1, Self→Previous 2, Span→Previous 5, Filler→Sink 1
- rare, C_M: Diffuse→Sink 5, Diffuse→Previous 1, Diffuse→Span 1, Sink→Diffuse 1, Sink→Previous 4, Sink→Span 2, Previous→Sink 2, Previous→Self 1, Previous→Span 1, Self→Previous 1, Span→Previous 3
- rare, C_r: Diffuse→Sink 2, Diffuse→Previous 1, Sink→Previous 3, Sink→Span 2, Previous→Sink 3, Previous→Self 1, Previous→Span 2, Self→Previous 1, Span→Sink 1, Span→Previous 4, Filler→Sink 1, Filler→Span 1
- rare, order_O: Diffuse→Sink 2, Sink→Diffuse 2, Sink→Previous 1, Previous→Sink 6, Previous→Self 1, Previous→Span 5, Self→Previous 1, Span→Previous 2
- rare, order_r: Sink→Previous 2, Previous→Sink 5, Previous→Self 1, Previous→Span 6, Span→Previous 3
- rare, sanity: Sink→Previous 1, Previous→Diffuse 1, Previous→Sink 7, Previous→Span 2, Span→Previous 1, Span→Self 1
- rare, placebo: none
- common, C_O: Diffuse→Sink 2, Diffuse→Previous 3, Sink→Diffuse 2, Sink→Previous 3, Sink→Span 1, Previous→Diffuse 1, Previous→Sink 7, Previous→Span 3, Self→Diffuse 1, Self→Previous 1, Span→Previous 4, Filler→Sink 1
- common, C_Op: Diffuse→Sink 3, Diffuse→Previous 3, Sink→Previous 5, Sink→Span 1, Previous→Diffuse 1, Previous→Sink 3, Previous→Self 1, Previous→Span 1, Self→Diffuse 1, Self→Previous 1, Span→Sink 1, Span→Previous 5, Filler→Sink 1
- common, C_M: Diffuse→Sink 3, Diffuse→Previous 2, Sink→Diffuse 1, Sink→Previous 4, Previous→Sink 5, Previous→Self 1, Previous→Span 1, Self→Previous 2, Span→Previous 4, Filler→Sink 1
- common, C_r: Diffuse→Sink 3, Diffuse→Previous 1, Sink→Previous 3, Previous→Sink 2, Previous→Span 3, Span→Previous 7
- common, order_O: Diffuse→Previous 1, Previous→Diffuse 1, Previous→Sink 8, Previous→Self 1, Previous→Span 4, Span→Previous 3, Filler→Sink 1
- common, order_r: Sink→Previous 6, Previous→Diffuse 2, Previous→Self 1, Previous→Span 6, Span→Previous 2
- common, sanity: Diffuse→Sink 3, Previous→Diffuse 3, Previous→Sink 5, Previous→Span 1, Previous→Filler 1, Span→Previous 7, Filler→Sink 1
- common, placebo: none

## TV distance between category profiles (per head; late mean, max)

| base | contrast | late mean | max (head) |
|---|---|---|---|
| rare | C_O | 0.1109 | 0.5760 (L4H0) |
| rare | C_Op | 0.0800 | 0.5025 (L4H0) |
| rare | C_M | 0.0574 | 0.2962 (L2H8) |
| rare | C_r | 0.0422 | 0.2453 (L2H8) |
| rare | order_O | 0.0629 | 0.5245 (L4H0) |
| rare | order_r | 0.0451 | 0.3781 (L3H6) |
| rare | sanity | 0.0492 | 0.2477 (L7H0) |
| rare | placebo | 0.0028 | 0.0072 (L11H11) |
| common | C_O | 0.0900 | 0.5858 (L4H0) |
| common | C_Op | 0.0868 | 0.5193 (L4H0) |
| common | C_M | 0.0743 | 0.3134 (L2H8) |
| common | C_r | 0.0517 | 0.2586 (L2H8) |
| common | order_O | 0.0627 | 0.5943 (L4H0) |
| common | order_r | 0.0355 | 0.4010 (L3H6) |
| common | sanity | 0.0675 | 0.2894 (L4H3) |
| common | placebo | 0.0032 | 0.0081 (L8H2) |

## Layer profile (mean Δ over rows and heads, per layer)

| base | contrast | measure | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | L11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rare | C_O | BOS | -0.000 | +0.000 | -0.018 | +0.023 | -0.060 | -0.006 | +0.002 | -0.001 | -0.047 | -0.037 | -0.083 | -0.096 |
| rare | C_O | Phrase | +0.004 | +0.018 | +0.026 | -0.033 | +0.041 | +0.018 | +0.009 | +0.025 | +0.017 | +0.011 | -0.006 | +0.008 |
| rare | C_Op | BOS | +0.000 | +0.001 | -0.018 | +0.027 | -0.044 | +0.011 | +0.018 | +0.022 | -0.024 | -0.004 | -0.023 | -0.030 |
| rare | C_Op | Phrase | -0.002 | +0.009 | +0.031 | -0.033 | +0.027 | +0.010 | -0.003 | +0.017 | +0.022 | +0.004 | -0.011 | +0.004 |
| rare | C_M | BOS | -0.000 | +0.000 | -0.023 | +0.008 | -0.022 | +0.009 | +0.020 | +0.014 | -0.006 | +0.010 | +0.014 | +0.010 |
| rare | C_M | Phrase | +0.005 | +0.015 | +0.046 | -0.008 | +0.015 | +0.007 | -0.010 | +0.007 | +0.004 | -0.003 | -0.011 | +0.004 |
| rare | C_r | BOS | -0.000 | +0.001 | -0.016 | +0.024 | -0.015 | -0.002 | +0.008 | +0.016 | +0.002 | +0.014 | +0.017 | +0.014 |
| rare | C_r | Phrase | -0.000 | +0.008 | +0.032 | -0.027 | +0.001 | +0.005 | -0.013 | -0.002 | -0.004 | -0.008 | -0.013 | +0.001 |
| rare | order_O | BOS | +0.000 | +0.000 | +0.004 | +0.011 | -0.019 | +0.008 | -0.011 | +0.026 | -0.009 | +0.034 | -0.011 | -0.029 |
| rare | order_O | Phrase | -0.001 | -0.001 | -0.014 | -0.059 | -0.006 | -0.039 | -0.017 | -0.020 | -0.000 | -0.004 | -0.006 | +0.000 |
| rare | order_r | BOS | +0.000 | -0.000 | +0.006 | +0.008 | -0.006 | -0.001 | -0.006 | +0.019 | +0.015 | +0.032 | +0.035 | +0.021 |
| rare | order_r | Phrase | -0.002 | -0.003 | -0.011 | -0.054 | -0.018 | -0.022 | -0.015 | -0.019 | -0.008 | -0.006 | -0.007 | -0.004 |
| rare | sanity | BOS | -0.000 | +0.000 | -0.004 | -0.001 | -0.013 | +0.001 | +0.022 | +0.031 | +0.029 | +0.039 | +0.031 | +0.021 |
| rare | sanity | Phrase | -0.003 | +0.008 | +0.008 | -0.029 | -0.019 | -0.011 | -0.035 | -0.034 | -0.033 | -0.024 | -0.025 | -0.003 |
| rare | placebo | BOS | +0.000 | -0.000 | -0.000 | -0.001 | -0.000 | +0.000 | -0.000 | -0.001 | -0.001 | -0.002 | -0.002 | -0.004 |
| rare | placebo | Phrase | -0.000 | +0.000 | +0.001 | -0.000 | +0.000 | +0.000 | +0.001 | +0.000 | -0.000 | +0.000 | +0.000 | +0.000 |
| rare | interaction | BOS | -0.000 | -0.001 | -0.002 | -0.000 | -0.045 | -0.004 | -0.006 | -0.017 | -0.048 | -0.050 | -0.101 | -0.110 |
| rare | interaction | Phrase | +0.005 | +0.009 | -0.006 | -0.006 | +0.040 | +0.012 | +0.022 | +0.026 | +0.021 | +0.019 | +0.007 | +0.007 |
| rare | completion | BOS | -0.000 | +0.000 | +0.005 | +0.015 | -0.038 | -0.014 | -0.018 | -0.015 | -0.040 | -0.046 | -0.097 | -0.106 |
| rare | completion | Phrase | -0.001 | +0.002 | -0.020 | -0.025 | +0.027 | +0.010 | +0.019 | +0.017 | +0.014 | +0.014 | +0.006 | +0.004 |
| rare | plausibility | BOS | -0.001 | -0.001 | -0.000 | -0.004 | -0.017 | -0.016 | -0.016 | -0.023 | -0.023 | -0.033 | -0.061 | -0.066 |
| rare | plausibility | Phrase | +0.006 | +0.008 | -0.005 | -0.000 | +0.014 | +0.007 | +0.012 | +0.007 | -0.004 | +0.008 | +0.005 | +0.004 |
| rare | order_x_query | BOS | -0.000 | +0.000 | -0.001 | +0.003 | -0.013 | +0.009 | -0.004 | +0.007 | -0.025 | +0.002 | -0.046 | -0.051 |
| rare | order_x_query | Phrase | +0.001 | +0.002 | -0.003 | -0.005 | +0.012 | -0.017 | -0.002 | -0.001 | +0.008 | +0.002 | +0.001 | +0.004 |
| common | C_O | BOS | -0.002 | -0.001 | -0.008 | +0.007 | +0.001 | +0.048 | +0.057 | +0.027 | +0.015 | -0.003 | -0.005 | -0.042 |
| common | C_O | Phrase | +0.023 | +0.039 | +0.065 | +0.015 | +0.050 | -0.000 | +0.017 | +0.026 | +0.023 | +0.026 | +0.015 | +0.030 |
| common | C_Op | BOS | -0.001 | -0.000 | -0.010 | +0.012 | +0.020 | +0.056 | +0.069 | +0.043 | +0.024 | +0.025 | +0.035 | +0.014 |
| common | C_Op | Phrase | +0.017 | +0.029 | +0.070 | +0.015 | +0.035 | -0.003 | +0.006 | +0.025 | +0.032 | +0.021 | +0.009 | +0.028 |
| common | C_M | BOS | -0.002 | -0.001 | -0.013 | +0.009 | +0.036 | +0.050 | +0.058 | +0.036 | +0.022 | +0.018 | +0.039 | +0.031 |
| common | C_M | Phrase | +0.021 | +0.032 | +0.075 | +0.020 | +0.021 | -0.002 | +0.007 | +0.017 | +0.020 | +0.019 | +0.008 | +0.025 |
| common | C_r | BOS | -0.000 | +0.001 | -0.004 | +0.025 | +0.042 | +0.021 | +0.032 | +0.025 | +0.004 | -0.011 | +0.008 | -0.005 |
| common | C_r | Phrase | -0.002 | +0.008 | +0.050 | +0.010 | -0.002 | +0.012 | +0.006 | +0.012 | +0.018 | +0.020 | +0.010 | +0.024 |
| common | order_O | BOS | +0.000 | +0.000 | +0.003 | +0.032 | -0.003 | +0.021 | +0.007 | +0.033 | +0.012 | +0.030 | +0.035 | +0.010 |
| common | order_O | Phrase | -0.001 | -0.003 | -0.008 | -0.056 | -0.011 | -0.048 | -0.011 | -0.017 | +0.003 | -0.000 | -0.003 | +0.001 |
| common | order_r | BOS | +0.000 | +0.000 | +0.003 | +0.021 | +0.004 | -0.015 | -0.015 | -0.005 | -0.012 | -0.006 | -0.002 | -0.010 |
| common | order_r | Phrase | -0.002 | -0.003 | -0.009 | -0.043 | -0.021 | -0.004 | +0.001 | -0.000 | +0.004 | +0.003 | +0.003 | +0.004 |
| common | sanity | BOS | -0.001 | -0.001 | +0.013 | +0.006 | +0.034 | +0.019 | +0.056 | +0.056 | +0.029 | +0.005 | +0.025 | +0.025 |
| common | sanity | Phrase | -0.001 | +0.008 | +0.020 | +0.024 | -0.006 | +0.014 | -0.005 | -0.003 | +0.012 | +0.028 | +0.019 | +0.034 |
| common | placebo | BOS | +0.000 | +0.000 | -0.001 | -0.000 | -0.001 | +0.001 | +0.002 | +0.001 | +0.003 | +0.002 | +0.003 | +0.003 |
| common | placebo | Phrase | -0.000 | +0.000 | +0.001 | +0.001 | +0.001 | -0.001 | -0.000 | +0.000 | -0.000 | -0.000 | -0.000 | -0.001 |
| common | interaction | BOS | -0.002 | -0.002 | -0.004 | -0.018 | -0.041 | +0.027 | +0.025 | +0.002 | +0.011 | +0.007 | -0.012 | -0.036 |
| common | interaction | Phrase | +0.025 | +0.031 | +0.015 | +0.005 | +0.052 | -0.012 | +0.010 | +0.014 | +0.005 | +0.006 | +0.005 | +0.006 |
| common | completion | BOS | +0.000 | -0.000 | +0.004 | -0.002 | -0.035 | -0.002 | -0.002 | -0.008 | -0.007 | -0.021 | -0.044 | -0.073 |
| common | completion | Phrase | +0.002 | +0.007 | -0.010 | -0.005 | +0.029 | +0.001 | +0.010 | +0.009 | +0.003 | +0.007 | +0.007 | +0.006 |
| common | plausibility | BOS | -0.001 | -0.001 | +0.002 | -0.005 | -0.019 | -0.008 | -0.012 | -0.016 | -0.009 | -0.028 | -0.040 | -0.055 |
| common | plausibility | Phrase | +0.005 | +0.009 | -0.005 | +0.000 | +0.014 | +0.003 | +0.010 | +0.001 | -0.009 | +0.005 | +0.006 | +0.002 |
| common | order_x_query | BOS | +0.000 | +0.000 | +0.001 | +0.011 | -0.007 | +0.037 | +0.022 | +0.037 | +0.024 | +0.035 | +0.037 | +0.019 |
| common | order_x_query | Phrase | +0.000 | +0.000 | +0.001 | -0.014 | +0.010 | -0.043 | -0.012 | -0.017 | -0.001 | -0.004 | -0.006 | -0.003 |

## Figures

- `figures/attention_maps_rare.png`
- `figures/attention_maps_common.png`
- `figures/contrast_maps.png`
- `figures/transitions.png`
