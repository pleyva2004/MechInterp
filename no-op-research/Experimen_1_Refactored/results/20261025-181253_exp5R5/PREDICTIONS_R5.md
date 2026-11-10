# Operational addendum R5: routes, not weights (value-weighted attention graphs)

Written 2026-10-25, before the full Exp 5 data (`results/20261025-170021_exp5res`) were complete or read. What I had
seen: `exp5/PREDICTIONS.md`, `exp5/extract.py`, `exp4/tda4_graph.py`, `configs/exp4_tda4.yaml`, `exp4/LOG.md`, TDA 4's
report (`results/20261025-113614_exp4tda4/tda4_report.md`), and the 2-row smoke extraction (used only for array shapes,
value-norm scales and timing: vw row sums are about 40–930 per query depending on layer; value weighting shrinks the
BOS share of the capital's row, e.g. from ~0.5 to ~0.15 in middle layers). Code: `exp5/r5_routes.py`.

This addendum makes operational choices only. Hypotheses R5-a / R5-b and the R5-a pass rule are those of
`exp5/PREDICTIONS.md` and are not changed.

## Data

`graphs_{common,rare}.npz`, 100 rows per filler: `raw_{own,wrong}` and `vw_{own,wrong}` [R, 7 τ, 12 layers, 33, 33]
(fp16; per-row means over the row's 11 cells); `raw_/vw_{random,control}` only for full-sample context. Only the 12
layer graphs are analysed (the per-head arrays are means over rows, so they cannot be bootstrapped; they are used only
for a descriptive per-head breakdown of the capital→country edge in layer 0).

**Pairing.** Exp 5 pairs cells by capital, TDA 4 by country. For the per-row mean graphs this is immaterial: own is the
same 11 sequences (country k, capital k), and wrong is the same 11 (country c, capital wrong[c]) = (country inv[k],
capital k). The per-row graphs are therefore the same objects as TDA 4's for the same rows. This is checked numerically
against Exp 4b's saved query-32 weights (`results/20261024-150407_exp4cap/capital.npz`, first 100 rows).

## Graph kinds

All computed in float64 from the fp16 arrays. Node 0 is BOS; positions 1..32; T = 32.

| kind | per-row graph G_r[q, k] | role |
|---|---|---|
| raw | as stored: mean over heads of attention (rows sum to 1) | sanity check; R5-b comparison |
| **vw** | vw_r[q, k] / Σ_{k' ≤ q} vw_r[q, k'], per row r and query q; the sum includes BOS and the diagonal | **R5-a (primary)** |
| vwM | the un-normalised vw per-row graphs are averaged first (full sample or bootstrap resample), then each query row is normalised (ratio of means) | sensitivity |
| vwU | vw_r[q, k] / c_{b,l}, un-normalised; c_{b,l} = mean over rows and query positions 1..32 of the row sums of `vw_control` (filler b, layer l), shared by every condition and τ; weights clipped to ≤ 1 (number clipped reported) | exploratory |

- vw normalises before averaging so that each data row contributes a graph with unit row sums, exactly as raw does; the
  full-sample / bootstrap graph is then the mean of normalised per-row graphs. Each per-row graph is already a mean over
  11 cells, so this is not a per-sequence normalisation (not available from the saved arrays).
- BOS is removed afterwards by `tda4_graph.dist(·, bos=False)` without renormalising, as TDA 4 does for raw ("noBOS
  keeps raw weights"). So in vw a weight is the share of query q's total value-weighted write that comes from key k.
- The self-edge (diagonal) is not an edge in the filtration but stays in the normaliser, as in raw.

## Topology (reused, not copied)

From `exp4/tda4_graph.py`: `dist(G, bos=False)` (w(i, j) = G[max, min], distance 1 − w), `summarize(d, τ, 32, False,
tda)` and its `persistent` / `rep_cycle` rules, with `tda` = the `tda` block of `configs/exp4_tda4.yaml`
(persist_min 0.002, persist_ratio 0.75, boot_stable 0.8, boot_absent 0.2). A matching-pair cycle (MPC) is present iff
`summarize(...)["mpc"] is not None`: a persistent H1 class whose shortest-path representative at birth uses the edge
{32, τ}. "Own" = with its capital (country k at τ, capital k at 32); "wrong" = with a wrong capital.

**Ties (added during development on the smoke data, before the full data were read).** The per-row graphs are stored
in fp16, so exact weight ties occur (and vwU's clipping makes ties at w = 1). `rep_cycle` looks for a path strictly
below the birth value and raises when the connecting edges tie with the birth edge (seen on the 2-row smoke data in
vw / vwM layer 9 and vwU layers 0 / 9). `r5_routes.py` wraps `tda4_graph.rep_cycle` (no copy, no edit of
`tda4_graph.py`): only when the original raises, it retries allowing edges with filtration ≤ b except the birth edge
itself (gudhi's order among tied edges is arbitrary too). The number of fallbacks is reported.

## Bootstrap

- n_boot = 1000 resamples of the 100 rows with replacement, per filler (`numpy.random.default_rng([0, filler_index])`,
  seed 0 = the TDA 4 seed; common = 0, rare = 1).
- The same resampled row indices are used for every τ, layer, kind and condition (paired comparisons). Each bootstrap
  graph is the mean of the resampled per-row graphs (for vwM: mean, then normalise).
- Bootstrap frequency f = share of the 1000 resamples with an MPC.

## R5-a verdict rule (as pre-registered)

Layer 0 (layer graph), kind vw, BOS removed, for each filler ∈ {common, rare} and τ ∈ {16, 8, 1}: f_own ≥ 0.8 and
f_wrong ≤ 0.2 (inclusive, as TDA 4's boot_stable / boot_absent). PASS iff all 6 cells meet it; otherwise FAIL, listing
the failing cells. Reported alongside, not part of the verdict: TDA 4's stricter rule that also requires an MPC in the
full-sample own graph and none in the full-sample wrong graph, and the vwM / vwU results.

## Sanity check (raw graphs)

1. **Data identity:** Exp 5's per-row raw query-32 weights on (τ, BOS, 31) for own and wrong vs Exp 4b's
   `{own,wrong}/{b}/country_masses` (mean over heads, first 100 rows), all 7 τ and 12 layers; tolerance 2e-3 (both fp16).
2. **Topology:** with raw graphs, classify every (layer, τ) cell with TDA 4's rule (`mpc_stable` / `mpc_absent`
   logic: full sample + bootstrap ≥ 0.8 / ≤ 0.2) into O / B / W / o / w / . exactly as TDA 4's report does, and compare
   the 12 × 7 × 2 grid with TDA 4's final run (`results/20261025-113614_exp4tda4`, `tda4.json` + `tda_arrays.npz`).
   Expected: layer 0 at τ 16, 8, 1 is O in both fillers. Agreement is likely but not guaranteed: 100 vs 200 rows, fp16
   storage of per-row graphs, and different bootstrap draws (1000 vs 100). Also reported: the full-sample layer-0 MPC
   cycles next to TDA 4's, and max |Δ| between the 100-row and 200-row mean graphs. If the layer-0 reproduction fails,
   R5-a is still evaluated but flagged.

## R5-b and descriptives (exploratory)

- f_own and f_wrong for all 12 layers × 7 τ, both fillers, for raw, vw, vwM and vwU (heatmaps; tables for layers 0–4).
- Bootstrap-only category per cell: O (own ≥ 0.8, wrong ≤ 0.2), B (both ≥ 0.8), W (wrong ≥ 0.8, own ≤ 0.2),
  · (both ≤ 0.2), ~ (anything else). A raw loop "survives value weighting" if its raw category (O or B) is unchanged
  under vw; it "is lost" if the vw category drops to · (or W/~ for the affected condition). Focus: layers 1, 3, 4 at
  τ 16 / 8 / 1 (B in TDA 4), layer 4 at τ 24 and layer 0 at τ 24 (O in TDA 4).
- Capital→country edge E1 = w(32, τ) per layer, τ and condition, raw vs vw: mean over rows with 95% percentile
  bootstrap CIs (same 1000 resamples), the ratio vw / raw (ratio of means, with CI), and own − wrong under each kind.
  Also: the BOS share of query 32 (raw vs vw), E1's rank among the 496 non-BOS edges of the mean graph, and in layer 0
  each head's share of E1 under raw vs vw (from `head_{raw,vw}_own`, means over rows; descriptive).

## Outputs

New folder `results/<timestamp>_exp5R5/`: `r5_report.md`, `r5.npz` (bootstrap frequencies, full-sample MPC flags,
E1 tables), `r5.json` (verdict, sanity check, full-sample MPC cycles), figures (`freq_<filler>.png`: layer × τ
frequency heatmaps for own / wrong, raw vs vw (+ vwM, vwU); `arcs_L0_t16_<filler>.png`: arc diagrams of layer 0 at
τ = 16, raw vs vw, own vs wrong, via `tda4_graph._arcs`; `e1_ratio.png`), `PREDICTIONS.md` copies and `versions.json`.

## Caveats to report

The same 11 fixed pairs throughout (rows are resampled, pairs are not: "consistent", not "general"); topology of the
mean graph, not of each sequence; value norms exclude each head's constant folded value-bias term (β_ln1 W_V + b_V) W_O
(moved into b_O by TransformerLens), so vw measures the key-dependent part of each write; norms ignore direction
(writes from different keys can cancel); the persistence thresholds were calibrated on raw attention.
