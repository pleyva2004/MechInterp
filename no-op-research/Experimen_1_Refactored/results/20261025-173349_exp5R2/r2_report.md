# Experiment 5, R2: the no-op test

Run `20261025-173349_exp5R2` on `20261025-170021_exp5res` (100 rows per filler type), 2000 row-bootstrap resamples, 160 s. Hypotheses: `exp5/PREDICTIONS.md`; operational choices: `exp5/PREDICTIONS_R2.md` (written before this script read the full data). Code: `exp5/r2_noop.py`.

Write of head (l, i) at T: w = z W_O[l, i] (processed weights). Spread = RMS distance of the writes from their mean / RMS norm of x_l(T), over a set of sequences. BOS share = mean attention from T to BOS.

## Verdicts

| id | verdict | short |
|---|---|---|
| R2-a | PASS | common ρ = -0.673 [-0.694, -0.660]; rare ρ = -0.696 [-0.713, -0.681] |
| R2-b | PASS | common ρ = -0.785 [-0.809, -0.761]; rare ρ = -0.815 [-0.843, -0.762] |
| R2-c | descriptive | common: rest 0.0211 (n 14) vs active 0.0585 (n 42); rare: rest 0.0183 (n 24) vs active 0.0485 (n 45) |

Pass rule: the row-bootstrap 95% CI of Spearman ρ is below 0 in both filler types.

## R2-a: BOS share vs write spread (144 heads, random final token)

| quantity | common | rare |
|---|---|---|
| **ρ (primary)** | -0.673 [-0.694, -0.660] | -0.696 [-0.713, -0.681] |
| ρ, S plus the random controls (sensitivity) | -0.673 [-0.694, -0.660] | -0.697 [-0.713, -0.681] |
| ρ, spread minus expected fp16 rounding variance (check) | -0.673 [-0.694, -0.660] | -0.696 [-0.713, -0.681] |
| ρ, unnormalised spread (exploratory) | -0.195 [-0.222, -0.175] | -0.291 [-0.322, -0.272] |
| ρ, within-row spread: only the country / τ varies (exploratory) | -0.502 [-0.519, -0.478] | -0.491 [-0.512, -0.458] |
| mean within-layer ρ, 12 heads each (exploratory) | -0.649 [-0.679, -0.630] | -0.693 [-0.716, -0.670] |
| partial ρ controlling for layer (exploratory) | -0.704 [-0.726, -0.687] | -0.764 [-0.779, -0.752] |

**Post hoc (added after the first full run, `20261025-172046_exp5R2`; not in the addendum).** How much of R2-a is the mechanical end (BOS share near 1 writes a near-constant vector by construction) or layer 0, and is the fall in spread more than dilution? The BOS term of a write is the same vector in every sequence, so if the rest of a head's attention wrote as variably as any other head's, spread would scale with 1 − share; ρ of spread / (1 − share) tests what is left. Head subsets fixed from the point estimates.

| post hoc | common ρ [CI] (n heads) | rare ρ [CI] (n heads) |
|---|---|---|
| share ≤ 0.9 | -0.662 [-0.683, -0.649] (142) | -0.661 [-0.679, -0.645] (138) |
| share ≤ 0.8 | -0.599 [-0.624, -0.586] (130) | -0.553 [-0.576, -0.535] (120) |
| layers 1–11 | -0.605 [-0.631, -0.589] (132) | -0.637 [-0.658, -0.618] (132) |
| dilution: spread / (1 − share), share ≤ 0.95 | +0.290 [+0.267, +0.304] (143) | +0.364 [+0.342, +0.375] (142) |
| dilution, partial ρ controlling for layer (1 − share floored at 0.05) | +0.283 [+0.250, +0.315] (144) | +0.490 [+0.444, +0.508] (144) |

Heads with BOS share > 0.9: common 2 (L5H1 0.99, L6H9 0.94), rare 6 (L5H1 0.97, L5H7 0.94, L7H4 0.93, L7H8 0.91, L7H9 0.96, L8H4 0.92).

Within-layer ρ (12 heads each, point estimates):

| layer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| common ρ | -0.90 | -0.08 | -0.34 | -0.68 | -0.94 | -0.77 | -0.77 | -0.76 | -0.90 | -0.64 | -0.43 | -0.58 |
| common mean BOS share | 0.07 | 0.16 | 0.18 | 0.41 | 0.49 | 0.66 | 0.57 | 0.67 | 0.61 | 0.68 | 0.60 | 0.44 |
| rare ρ | -0.76 | -0.17 | -0.34 | -0.20 | -0.84 | -0.92 | -0.76 | -0.90 | -0.90 | -0.87 | -0.93 | -0.75 |
| rare mean BOS share | 0.07 | 0.15 | 0.21 | 0.37 | 0.54 | 0.67 | 0.62 | 0.71 | 0.66 | 0.69 | 0.57 | 0.43 |

## R2-b: Δ BOS share vs Δ spread, own − wrong capital (layers 6–11, 72 heads)

| quantity | common | rare |
|---|---|---|
| **ρ (primary; one stream denominator per layer)** | -0.785 [-0.809, -0.761] | -0.815 [-0.843, -0.762] |
| ρ, each condition's own denominator (sensitivity) | -0.785 [-0.809, -0.751] | -0.834 [-0.855, -0.779] |
| mean within-layer ρ (exploratory) | -0.739 [-0.762, -0.695] | -0.700 [-0.748, -0.660] |
| partial ρ controlling for layer (exploratory) | -0.792 [-0.814, -0.768] | -0.808 [-0.833, -0.767] |
| mean Δ BOS share over the 72 heads | -0.033 [-0.035, -0.032] | -0.025 [-0.027, -0.024] |
| mean Δ spread over the 72 heads | +0.005 [+0.005, +0.006] | +0.004 [+0.003, +0.004] |

Heads with Δ BOS share < 0: common 62/72, rare 63/72.

## R2-c: median spread, resting (BOS share ≥ 0.8) vs active (≤ 0.3) heads

| set | filler | resting: median (n) | active: median (n) | active / resting [95% CI] |
|---|---|---|---|---|
| random (primary) | common | 0.0211 (14) | 0.0585 (42) | 2.78 [2.53, 3.16] |
| random (primary) | rare | 0.0183 (24) | 0.0485 (45) | 2.65 [2.54, 3.20] |
| own | common | 0.0188 (9) | 0.0444 (51) | 2.36 [2.23, 2.60] |
| own | rare | 0.0131 (11) | 0.0477 (53) | 3.65 [3.20, 3.92] |
| wrong | common | 0.0183 (13) | 0.0429 (47) | 2.35 [2.25, 2.64] |
| wrong | rare | 0.0178 (16) | 0.0472 (52) | 2.65 [2.37, 3.13] |

## Notable heads

Share = BOS share; spread on the random set; Δ = own − wrong (layers 6–11 only); TwoNN on the random set; RMS ‖w‖ and ‖mean w‖ divided by RMS ‖x_l‖ (both exclude the folded constant).

| head | filler | share random | share own | share wrong | spread | Δ share | Δ spread | TwoNN | RMS ‖w‖ | ‖mean w‖ |
|---|---|---|---|---|---|---|---|---|---|---|
| L0H2 | common | 0.203 | 0.187 | 0.300 | 0.2372 | — | — | 4.3 | 0.664 | 0.620 |
| L0H2 | rare | 0.190 | 0.172 | 0.266 | 0.2641 | — | — | 3.8 | 0.918 | 0.879 |
| L4H11 | common | 0.000 | 0.000 | 0.000 | 0.0984 | — | — | 6.7 | 0.116 | 0.062 |
| L4H11 | rare | 0.000 | 0.000 | 0.000 | 0.0819 | — | — | 7.8 | 0.102 | 0.060 |
| L9H8 | common | 0.452 | 0.671 | 0.652 | 0.0762 | +0.019 | +0.0030 | 5.7 | 0.090 | 0.048 |
| L9H8 | rare | 0.655 | 0.633 | 0.635 | 0.0405 | -0.002 | +0.0034 | 4.2 | 0.053 | 0.034 |
| L10H3 | common | 0.623 | 0.353 | 0.354 | 0.0464 | -0.001 | +0.0046 | 6.2 | 0.077 | 0.061 |
| L10H3 | rare | 0.404 | 0.197 | 0.206 | 0.0607 | -0.008 | +0.0033 | 5.4 | 0.111 | 0.093 |
| L11H2 | common | 0.634 | 0.581 | 0.579 | 0.0433 | +0.001 | -0.0016 | 3.8 | 0.060 | 0.042 |
| L11H2 | rare | 0.565 | 0.462 | 0.490 | 0.0451 | -0.029 | +0.0029 | 4.4 | 0.055 | 0.032 |
| L11H7 | common | 0.481 | 0.344 | 0.380 | 0.0407 | -0.036 | +0.0038 | 6.6 | 0.083 | 0.072 |
| L11H7 | rare | 0.454 | 0.311 | 0.339 | 0.0417 | -0.028 | +0.0027 | 7.6 | 0.089 | 0.078 |
| L11H8 | common | 0.000 | 0.000 | 0.000 | 0.0760 | +0.000 | +0.0010 | 5.0 | 1.172 | 1.170 |
| L11H8 | rare | 0.000 | 0.000 | 0.000 | 0.0683 | +0.000 | +0.0003 | 5.1 | 1.171 | 1.169 |

Extremes of the R2-a scatter (highest-spread resting heads, lowest-spread active heads):

- common: L3H0 (share 0.81, spread 0.0334), L5H5 (share 0.85, spread 0.0298), L1H2 (share 0.01, spread 0.0164), L2H7 (share 0.12, spread 0.0168)
- rare: L8H10 (share 0.81, spread 0.0306), L6H9 (share 0.89, spread 0.0300), L1H2 (share 0.01, spread 0.0141), L1H10 (share 0.00, spread 0.0146)

Largest |Δ BOS share| in layers 6–11:

- common: L7H10 (Δ share -0.158, Δ spread +0.0498), L9H9 (Δ share -0.121, Δ spread +0.0454), L11H10 (Δ share -0.104, Δ spread +0.0077), L10H10 (Δ share -0.100, Δ spread +0.0165), L10H7 (Δ share -0.087, Δ spread +0.0101), L10H6 (Δ share -0.084, Δ spread +0.0085)
- rare: L7H10 (Δ share -0.096, Δ spread +0.0266), L6H9 (Δ share -0.086, Δ spread +0.0190), L10H10 (Δ share -0.086, Δ spread +0.0129), L9H9 (Δ share -0.080, Δ spread +0.0216), L10H7 (Δ share -0.071, Δ spread +0.0115), L11H10 (Δ share -0.068, Δ spread +0.0049)

## Exploratory: topology and norms of each head's write cloud (random set; not pre-registered tests)

Writes in the isometric frame / the layer's stream RMS norm, exact duplicates removed, PCA to 20 dimensions for TwoNN and H0. Spearman ρ vs BOS share across the 144 heads (point estimates), and the partial ρ controlling for layer.

| metric | common ρ | common partial ρ | rare ρ | rare partial ρ |
|---|---|---|---|---|
| TwoNN intrinsic dimension | -0.229 | -0.050 | -0.286 | -0.183 |
| PCA participation ratio (64-d) | -0.312 | -0.251 | -0.324 | -0.210 |
| Rips H0 total persistence | -0.597 | -0.736 | -0.706 | -0.784 |
| RMS ‖w‖ / RMS ‖x_l‖ | -0.749 | -0.586 | -0.785 | -0.677 |
| ‖mean w‖ / RMS ‖x_l‖ | -0.657 | -0.445 | -0.723 | -0.569 |

- common: PCA-20 keeps a median 0.836 of variance (min 0.749); unique points per head median 3300 of 3300 (min 1644). TwoNN range 1.7–9.1.

- rare: PCA-20 keeps a median 0.817 of variance (min 0.717); unique points per head median 3300 of 3300 (min 1512). TwoNN range 1.6–10.1.

## Checks

| check | common | rare |
|---|---|---|
| spread from 768-d writes vs isometric sums (4 heads) | PASS (max rel |Δ| 4.5e-16) | PASS (max rel |Δ| 4.4e-16) |
| BOS share in [0, 1] | PASS | PASS |
| stream RMS at l = 0 identical, own vs wrong (built-in null) | PASS (rel |Δ| 0.0e+00) | PASS (rel |Δ| 0.0e+00) |
| H0 total persistence: Prim vs ripser vs scipy MST (L9H8, 400 points) | PASS (rel |Δ| 7.0e-10) | PASS (rel |Δ| 2.0e-09) |
| heads excluded for near-zero writes (RMS ‖w‖ < 1e-6 RMS ‖x_l‖) | 0 | 0 |
| rows | 100 | 100 |

- fp16 floor, common: expected rounding noise / measured spread has median 0.000, max 0.004; heads above 0.5: none.

- fp16 floor, rare: expected rounding noise / measured spread has median 0.000, max 0.002; heads above 0.5: none.

## Caveats

- The same 11 country–capital pairs throughout: CIs are over rows, so beyond 0 means consistent, not general.
- Writes are TransformerLens's processed writes: mean-centred, and without each head's folded constant (β_ln1 W_V + b_V) W_O (moved into b_O). A constant offset does not change spread, so R2-a–c are unaffected, but the norms above are of the non-constant part only.
- BOS's stream is identical in every sequence, so a head with BOS share exactly 1 writes an exactly constant vector; the extreme of R2-a is partly mechanical.
- Depth confound: BOS share and the stream norm both change with depth. The within-layer and partial ρ rows show how much of R2-a survives within a layer.
- The topology metrics see within-row neighbours first (the 33 sequences of a row differ only at τ), and for nearly constant heads the cloud's fine structure is at fp16 resolution (see the floor check).
- Three τ values only (31, 16, 1) for the per-sequence clouds.

## Files

- `r2_stats.npz`: per-head statistics per filler (keys `<filler>_<name>`).
- `r2_summary.json`: verdicts and every ρ with its CI.
- `fig_r2a_scatter.png`, `fig_r2b_delta.png`, `fig_r2_explore.png`.
