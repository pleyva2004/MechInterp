# Operational addendum R4: own and wrong across layers (Experiment 5)

Written 2026-10-25, before R4 touched the full data (`results/20261025-170021_exp5res`, still being written at the
time). What I had seen: `exp5/PREDICTIONS.md`, the docstring of `exp5/extract.py`, `onepager/residual_topology_notes.tex`,
`exp4/LOG.md`, `exp4/tda2_mapper.py`, and the 2-row smoke run (`exp5_smoke/20261025-165925_exp5res`, whose rows are
the first 2 of the 100). Code: `exp5/r4_flow.py`. Development ran only on the smoke run, and the smoke numbers were
used only for the operational choices marked below. This file does not change R4-a, R4-b or their pass rules in
`exp5/PREDICTIONS.md`.

## Points

- **Data:** `resid` from `seq_<b>.npz`. Conditions own (0) and wrong (1). τ ∈ {16, 1} are scored; τ = 31 is
  descriptive. All 100 rows and 11 cells. Stream states l = 1..12. x_0 is identical within a pair, and this is checked.
- **Layer:** "layer l" means the state index. x_l is the stream before block l (after blocks 0..l − 1); x_12 is after
  block 11, before `ln_final`.
- **Normalisation:** per state, x ← (x − mean) / sqrt(var + 1e-5) over the 768 dimensions. This is LayerNorm without
  its gain or bias, with GPT-2's ε, computed in float64.
- **Capital-centring:** per (row, cell, τ, layer), subtract the mean of the own/wrong pair. Own becomes +δ/2 and wrong
  −δ/2, with δ = x̂_own − x̂_wrong.
  - 2200 points per layer, 26,400 per (filler, τ) cloud.
  - The cloud is exactly symmetric, so its mean is 0 (checked).
- **PCA:** k = 50, fitted on each (filler, τ) cloud pooled over the 12 layers.
  - Uncentred, which equals centred because the mean is 0. Refitted in every bootstrap resample.
  - Sensitivity: k = 20, 100 and 768 (no reduction) at the main cover, every m.
  - The share of each layer's squared norm that the PCA keeps is reported.
- **Metric:** Euclidean distance on the PCA coordinates.

## Mapper

- **Lens:** the layer l ∈ 1..12. A cover is windows of w consecutive layers with step s: layers [1 + js, js + w].
  - These equal kmapper's uniform `Cover` on the lens l − 1, with (n, p, limits) from `exp4.tda2_mapper.layer_cover`
    (checked).
- **Clustering:** average linkage (scipy, Euclidean) on each window's points.
  - The tree is cut at ε = m × D_med, where D_med is the median pairwise distance among that window's points.
  - D_med is recomputed for every window and every resample, so the rule is scale-free per window.
  - Every point lands in exactly one node per window. There is no noise class.
- **Graph:** one node per cluster per window. Two nodes are joined when they share a point, that is, a state in a
  layer common to both windows.
- **Main setting:** w = 2, s = 1, m = 1. This gives 11 windows: layers 1–2, 2–3, …, 11–12.
- **Grid (20 settings):** covers (w, s) ∈ {(2, 1), (3, 1), (4, 1), (4, 2)} × m ∈ {0.8, 0.9, 1.0, 1.1, 1.2}.
  - Every cover tiles layers 1..12 exactly.
  - The m range was set on the smoke data. There, m ≥ 1.4 put each window into one node, and m ≤ 0.6 left most nodes
    under 5 points (with 2 rows). The grid spans the range in between.

## Operational definitions

- **Checked node:** a node holding ≥ 5 points. Bootstrap duplicates count.
- **Pure node:** a checked node whose own share is ≥ 0.8 or ≤ 0.2.
- **Clean window:** every checked node in it is pure.
- **l\*:** the first layer of the earliest window j such that windows j, j + 1, …, last are all clean.
  - It is undefined, and R4-a fails for that graph, if the last window (which contains layer 12) is not clean.
  - "From l* through l = 12" is read on window starts. With w = 2, a clean window starting at l covers l and l + 1.
- **R4-a holds** for a setting or resample when l* exists for all four scored graphs: (common, rare) × (τ = 16, 1).
- **R4-b holds** when, in addition, the largest of the four l* is ≤ 4.

## Verdict and stability

- **Verdict:** the main setting on the full data, as the hypothesis is stated.
- **Stability numbers reported:**
  - s_grid: the share of the 20 settings in which the hypothesis holds on the full data;
  - s_boot: the share of the 100 resamples × 20 settings;
  - main boot: the share of the 100 resamples at the main setting;
  - l* per setting, and its bootstrap distribution per graph at the main setting.
- **Label:** "stable" if the main outcome (holds or fails) recurs in ≥ 70% of grid settings and in ≥ 70% of
  main-setting resamples; otherwise "unstable". This qualifies the verdict and does not change it.
- **Bootstrap:** 100 resamples of the 100 rows with replacement, seed 0, each filler type independently.
  - The same row indices are used for every τ and both conditions, because they are the same rows.
  - Each resample rebuilds the cloud, the PCA, every tree and every graph.

## Robustness view (no Mapper)

- For each layer and each (filler, τ ∈ {31, 16, 1}): average linkage on the layer's 2200 points (the PCA-50
  coordinates of the pooled cloud), cut into 2 clusters (`maxclust`).
- Purity = Σ over the two clusters of the majority-condition count / N. The minority cluster's share is reported
  alongside, because a 0.99 / 0.01 split has purity near 0.5 whatever the labels.
- 95% percentile interval from the same bootstrap resamples.

## Sensitivity and diagnostics (descriptive, no verdicts)

- **PCA dimension:** as above.
- **Global-scale rule:** the main cover cut at m × one scale for the whole cloud. The scale is the median pairwise
  distance of 4000 random points (seed 0).
- **Per-layer rescaled variant:** each layer's points are divided by the layer's RMS point norm before the PCA.
  - Run on the full grid with the full data, and at the main cover (every m) in each bootstrap resample.
  - Reason (smoke data): the normalised δ at layer 12 is about half that of layer 11. A window that mixes the two
    therefore merges the layer-12 points into one central node at the window's scale, even where the per-layer
    2-cluster split separates them. The variant shows whether a failure in the last windows is this scale mismatch
    rather than own and wrong merging within a layer.
- **Label nulls on the main graphs:** the geometry and the nodes are fixed, and only the labels are permuted.
  - Pair flips: own ↔ wrong swapped within random (row, cell) pairs, the same in every layer; 1000 draws, seed 1.
  - Cell flips: own ↔ wrong swapped for all rows of a set of cells; all 1023 non-identity patterns, with cell 0 fixed.
  - Reported: the share of patterns that have an l*, and the mean share of points in pure nodes.
  - Why: δ changes both the match and the country's identity (country k vs the country whose deranged capital is k).
    The derangement is a single 11-cycle, so no two cells mirror each other. If own and wrong are separated only as
    (cell, condition) groups, with no direction shared across capitals, purity survives cell flips.
- **Cells per pure node:** the median, and the share of pure nodes that hold a single cell.
- **Flow description per window:** the share of points in pure-own, pure-wrong, mixed and small nodes; the edges that
  join a pure-own and a pure-wrong node; and the connected components after l* that hold both.
- **τ = 31:** the main setting, the grid on the full data and the robustness view. No verdict. Not bootstrapped for
  Mapper; it is bootstrapped for the robustness view.

## Checks (not predictions)

- The normalised own and wrong x_0(T) are identical (max |Δ| < 1e-5).
- The capital-centred cloud is exactly symmetric.
- The cover windows equal kmapper's intervals.
- The hand-rolled Mapper equals kmapper 2.1.0 at the main setting (same nodes and edges), for each scored graph.

## Caveats known in advance

- The same 11 pairs and one fixed derangement throughout. The bootstrap resamples rows only.
- Mapper is sensitive to the cover and the scale. Smaller m gives finer nodes, which are more often pure.
- R4-a is strict: one mixed node of ≥ 5 points in any later window breaks it.
- The smoke data informed only the m range and the rescaled variant.
