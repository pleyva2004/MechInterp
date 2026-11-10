# Operational addendum R1: when does the stream at T know the match?

Written 2026-10-25, before `exp5/r1_match.py` was run on the full extraction (`results/20261025-170021_exp5res`). Only
the 2-row smoke extraction was used, to develop and test the code. Synthetic data was used to time the H0 null. The
hypotheses and pass rules are the ones in `exp5/PREDICTIONS.md` (R1-a, R1-b, R1-c). This file fixes only how they
are computed. Where a pass rule needs an interpretation, the interpretation is stated here, before the data.

## Data

- `seq_<b>.npz`, `resid` [R = 100, cond, τ, cell, 13, 768] fp16, b ∈ {common, rare}.
- own = cond 0, wrong = cond 1, τ ∈ {31, 16, 1}, cells k = 0..10.
- Cell k has capital k at T in both conditions. own has country k at τ; wrong has country inv[k] (Exp 4b
  derangement, seed 5), so the two states differ only at τ.
- Stream state l = 0..11 is `hook_resid_pre` of layer l; l = 12 is `blocks.11.hook_resid_post`.
- Arrays are cast to float64 after loading.

## Normalisation and δ

- Each state vector x ∈ R^768 becomes x̂ = (x − mean(x)) / sqrt(mean((x − mean(x))²) + 1e-5): LayerNorm without gain
  or bias, with GPT-2's ε.
- δ_l = x̂_l(own) − x̂_l(wrong) per (row, cell, τ), giving N = 100 × 11 = 1,100 δ's per (filler type, τ, l).
- All statistics pool cells and rows within τ. Pooling is what makes the test about the match: within a row, the 11
  cells contain each country once as own and once as wrong (inv is a permutation), and each capital once on both sides.
  So country and capital main effects cancel in the mean δ, and only a capital × country interaction survives.

## R1-a

- max over rows, cells, τ and both filler types of |δ_0| (every coordinate), after normalisation.
- PASS if < 1e-5. The raw (unnormalised) maximum is reported too.

## R1-b

- **Statistic.** S_l(τ) = ‖mean_i δ_i‖ / RMS(δ), with RMS(δ) = sqrt(mean_i ‖δ_i‖²). It is computed in the full 768
  normalised dimensions; there is no PCA, because S is not a distance. S ∈ [0, 1]: it is 1 only if all δ's are equal.
- **Sign-flip null.**
  - B = 2,000 draws.
  - In each draw, every δ (one per row, cell and τ) gets an independent sign s = ±1. The same signs apply at every
    layer.
  - Seeds: `default_rng([0, filler index])`.
  - RMS is invariant to flips, so only the numerator changes.
- **p-value and correction.** p = (1 + #{S_null ≥ S_obs}) / (1 + B). BH (`exp1.compare.bh_fdr`) runs over the 36
  cells of a filler type (layers 1–12 × 3 τ).
- **A cell passes** if S_obs is above the null's 95th percentile and q < 0.05.
- **R1-b PASSES** if all 36 cells pass in both filler types.
- **"The first layer where R1-b holds"** for a (filler type, τ) is the smallest l ∈ 1..12 whose cell passes.
- **Reported alongside (not criteria):**
  - a row-bootstrap 95% percentile CI of S (1,000 resamples of the 100 rows, the same resamples at every τ and layer);
  - a row-level sign-flip null: one sign per (row, τ) shared by the row's 11 δ's. The per-δ null is conservative,
    because the main effects cancel within a row; the row-level null is the calibrated one if rows are exchangeable.
  - S with the 3 largest-magnitude dimensions zeroed (largest mean |x̂| over the own and wrong states of that τ and
    layer), against its own sign-flip null. This is the massive-activation check of the notes.

## R1-c

- **Cloud.** For each (filler type, τ, l ≥ 1):
  - V_k = the top k right singular vectors of the uncentred δ matrix [1,100 × 768]. This is PCA of the symmetrised
    cloud, whose mean is exactly 0.
  - Y = δ V_k.
  - The cloud is Z = [Y; −Y], 2,200 points.
  - k = 50 decides the verdict. k = 20 and k = 100 are checks.
- **H0.** Euclidean distances in float64. The block structure [[A, B], [B, A]] makes a pair and its mirror image
  bitwise equal.
  - Rips H0 death times come from ripser (`maxdim=0`, `distance_matrix=True`) on every observed cloud.
  - Null clouds use scipy single linkage. Its merge heights are the Rips H0 death times by definition, and it is about
    20 times faster. The script reports the largest ripser-vs-scipy difference on the observed clouds.
- **Statistic: the 2-component gap = d(1) − d(2)**, where d(1) ≥ d(2) are the two largest finite H0 death times.
  - Count the essential (infinite) bar as the largest death. Then d(1) and d(2) are the pre-registration's "second and
    third largest death times".
  - The gap is the range of scales over which the cloud has exactly two components.
  - The difference between the 2nd and 3rd largest *finite* deaths is not used. It measures a 3-component state.
  - On a mirror-symmetric cloud, every merge that does not join a set to its own mirror image happens twice at the
    same scale. So a positive gap requires the last merge to join two mirror halves.
  - d(1) / d(2) is reported as a scale-free description.
- **Null (an interpretation of "the sign-flip null").**
  - Flipping the sign of any δ maps {δ} ∪ {−δ} onto itself. A literal sign-flip null on the symmetrised cloud is
    therefore degenerate: every draw reproduces the observed diagram, and the observed gap can never beat it.
  - Instead, draw b uses the cloud Y_i − Ȳ + Ȳ^(b), where Ȳ^(b) = mean_i s_i^(b) Y_i is built from the R1-b signs of
    draw b. The cloud is then symmetrised as above.
  - So the null keeps the observed scatter around its mean and replaces the observed consistent shift with one of the
    size sign-flip noise produces. "No consistent direction" predicts exactly that cloud under a location shift.
  - The basis V_k is the observed one, fitted once and not refitted per draw.
  - k = 50 uses the first 2,000 draws; k = 20 and k = 100 use the first 500 (for compute).
- **Beating the null.** p = (1 + #{gap_null ≥ gap_obs}) / (1 + B). A cell beats the null if gap_obs is above the
  null's 95th percentile and BH q < 0.05. BH runs over the tested cells of a filler type, the same convention as R1-b.
  If the observed gap and every null gap are 0 (exact mirror ties), p = 1.
- **Tested cells.** For each (filler type, τ), layers from the first layer where R1-b holds through 12. If R1-b holds
  at no layer for that (filler type, τ), layers 1–12 are tested, which is the stricter reading.
- **R1-c PASSES** if every tested cell beats the null at k = 50 in both filler types. The verdict counts as robust if
  k = 20 and k = 100 give the same verdict under the same rule.
- **Robustness view (not a criterion).** Cut the PCA-50 cloud Z into 2 clusters with scipy average linkage and with
  single linkage (`fcluster`, `maxclust` = 2). Report each cut's agreement with the sign labels (+ for δ, − for −δ),
  max(acc, 1 − acc), and the size of the smaller cluster.

## Probe (non-topological baseline)

- **Data.** Per (filler type, τ, l): the 2,200 normalised states x̂ (own = 1, wrong = 0), 768 features.
- **Model.** `StandardScaler` + `LogisticRegression` (L2, C = 1, lbfgs, `max_iter` 5,000).
- **Cross-validation.** `GroupKFold(5)` with rows as groups, so both members of a pair, and all of a row's cells, are
  always in the same fold. Accuracy is pooled over the 2,200 out-of-fold predictions.
- **Null band.** In each of 10 draws per layer, the two labels of each (row, cell) pair are swapped at random (the
  sign-flip unit), and the probe is refitted with the same CV. The band is the 2.5–97.5 percentile of these
  accuracies, pooled over layers 1–12 per (filler type, τ).
- **Leave-one-cell-out (`GroupKFold(11)`, cells as groups).** Also reported, as a stricter test of one shared
  direction. In training, the held-out pair's capital never appears, and its own country appears only in a wrong
  pair.
- **Sanity check.** Layer 0 must give exactly 0.5, because own and wrong states are identical there.

## Exploratory (labelled as such in the report)

- ‖mean δ_l‖ and RMS(δ_l) per layer, τ and filler type, in normalised units (‖x̂‖ = √768).
- Cosine similarities of the mean δ directions:
  - across τ (31 vs 16, 31 vs 1, 16 vs 1);
  - between filler types at the same τ;
  - between rows 1–50 and 51–100 at the same τ. This split-half value is the reliability ceiling for the other two.

## Outputs

`results/<timestamp>_exp5R1/` contains:
- `r1_report.md`;
- `r1_verdicts.json`;
- `r1_stats.npz`;
- figures:
  - `r1_layers.png`: S with its CI and null 95th percentile, and probe accuracy with the null band, by layer, for each
    τ and filler type;
  - `r1_h0.png`: the gap vs its null, and the linkage agreement;
  - `r1_pca_<b>.png`: the first two principal axes of ±δ at l = 1, 4, 8, 12;
- both prediction files;
- `versions.json`.
