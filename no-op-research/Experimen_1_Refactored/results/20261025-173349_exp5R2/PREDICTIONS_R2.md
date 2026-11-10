# Operational addendum R2: the no-op test (Experiment 5)

Written 2026-10-25 before R2 touched the full data (`results/20261025-170021_exp5res`, still being written at the
time). What I had seen: `exp5/PREDICTIONS.md`, the docstring of `exp5/extract.py`, `onepager/residual_topology_notes.tex`,
`exp4/LOG.md`, and the array **shapes** of the 2-row smoke run. Code: `exp5/r2_noop.py`; development ran only on the
smoke run. This file fixes operational choices. It does not change R2-a, R2-b, R2-c or their pass rules in
`exp5/PREDICTIONS.md`.

## Quantities

- **Write** of head (l, i) on sequence s: w = z_{l,i}(T) W_O[l, i], with the saved processed W_O (`weights.npz`).
  Computed exactly in a 64-d isometric frame: with the SVD W_O[l, i] = U Σ Vᵀ, y = z U Σ has ‖y₁ − y₂‖ = ‖w₁ − w₂‖ and
  ‖y‖ = ‖w‖. Checked against the 768-d writes on a few heads (relative difference < 1e-5).
- **Spread** over a set S: sqrt(mean_{s∈S} ‖w_s − w̄_S‖²) / sqrt(mean_{s∈S} ‖x_l(s)‖²), with x_l(T) = `resid[..., l, :]`
  (the stream before layer l, the one the head reads), as stored (TransformerLens's processed stream is already
  mean-centred). The shared design's per-state LayerNorm normalisation is for distances between stream states; here
  the stream enters only as one scale per layer, as the hypothesis states.
- **BOS share** over S: the mean of `att[..., l, i, 0]` over S.
- fp16 arrays are cast to float32; sums are accumulated in float64. Writes are centred on the head's overall mean before
  the per-row sums are formed, so the sum-of-squares identity does not lose precision.

## Sets S

- **R2-a and R2-c:** S = every random-final-token sequence: cond 2, all three τ ∈ {31, 16, 1}, all 11 cells, all rows
  (33 sequences per row). The random controls (`rnd_ctrl_*`, the base rows, one per row) are **not** in the primary S,
  because they have no country at τ and the hypothesis is about the cond-2 set. Sensitivity: S plus the random controls
  (34 per row), reported but not used for the verdict.
- **R2-b:** S_own = cond 0, S_wrong = cond 1, each all τ, cells, rows; layers 6–11 (72 heads). Δ share = share(S_own) −
  share(S_wrong); Δ spread = spread(S_own) − spread(S_wrong), each spread taken around its own condition's mean write.
  **One denominator per layer for both terms:** the RMS norm of x_l(T) over S_own ∪ S_wrong, so Δ spread reflects only
  the writes (separate denominators would add a term proportional to each head's spread level, which itself tracks BOS
  share). Sensitivity: each condition's own denominator, reported but not used for the verdict.

## Statistics

- Point estimates use all rows. Spearman ρ across heads (144 for R2-a, 72 for R2-b), `scipy.stats.spearmanr`.
- **Row bootstrap:** 2000 resamples of the 100 rows with replacement, seed 0, separately per filler type. Each
  resample recomputes both per-head quantities (BOS share and spread, or both Δs) from per-row sufficient statistics
  (per row and head: the sum of y, the sum of ‖y‖², the sum of BOS attention; per row and layer: the sum of ‖x_l‖²),
  which is exact for resampled sets. 95% CI = 2.5th and 97.5th percentiles of ρ.
- **Pass rules (from PREDICTIONS.md):** R2-a passes if the CI's upper end is < 0 in both filler types; R2-b likewise.
- **R2-c (descriptive):** on the R2-a set, the median spread of heads with BOS share ≥ 0.8 and of heads with ≤ 0.3,
  with counts, plus the ratio of the two medians with a bootstrap CI (group membership recomputed in each resample).
  The same table is given for the own and wrong sets.
- **Exclusions:** none planned. A head would be excluded only if its RMS write norm were below 1e-6 of the stream
  RMS norm; the report states whether any was.

## Exploratory (labelled as such; no verdicts)

- **Depth confound.** BOS share tends to rise with depth and so does the stream norm, so the normalised spread could
  fall with depth for reasons unrelated to resting. Reported: ρ with the unnormalised spread; the mean of the 12
  within-layer ρ (12 heads each); the partial Spearman controlling for layer (Pearson correlation of the rank residuals
  after regressing both ranks on layer dummies). Each with the same row bootstrap.
- **Within-row spread** (added after the smoke run, still before the full data): the same ratio with the writes'
  variance taken about each row's own mean, i.e. only the country and τ vary (context and final token fixed). Same
  bootstrap. A check on how much of the spread is the final token and context rather than the country.
- **fp16 floor:** z is stored in fp16, so a nearly constant head's spread could be rounding noise. The expected
  rounding variance (spacing² / 12 per element, mapped through W_O) is compared with each head's measured variance, and
  ρ is recomputed with the measured variance minus it.
- **Topology of each head's write cloud** (R2-a set, 33 × R points): y divided by the layer's stream RMS norm, PCA to
  20 dimensions (variance kept is reported), exact duplicates removed. TwoNN intrinsic dimension (Facco et al. 2017,
  MLE d = N / Σ log(r₂/r₁)), Rips H0 total persistence (sum of the minimum-spanning-tree edge lengths, which are the
  finite H0 death times), and the PCA participation ratio (Σλ)²/Σλ² on all 64 dimensions as a global dimension. Each
  related to BOS share by Spearman across the 144 heads, point estimates only. Caveat fixed in advance: within a row the
  33 sequences differ only at τ, so nearest neighbours (TwoNN, short H0 bars) mostly measure within-row variation, the
  part caused by the country.
- **Write norms:** RMS ‖w‖ and ‖w̄‖ over S, divided by the stream RMS norm. These exclude each head's folded constant
  (β_ln1 W_V + b_V) W_O, which TransformerLens moves into b_O. A constant does not change spread but does change norms,
  so norms here are of the variable part plus the BOS-value part only.
- Notable heads quoted individually: L0H2, L4H11, L9H8, L10H3, L11H2, L11H7, L11H8, plus the extremes of each scatter.

## Caveats fixed in advance

- The same 11 country–capital pairs throughout: a CI beyond 0 means consistent over rows, not general over pairs.
- BOS's stream is identical in every sequence (position 0 attends only to itself), so a head whose BOS share is
  exactly 1 writes an exactly constant vector. The hypothesis is therefore partly mechanical at the extreme; the
  test is whether spread falls with BOS share across the whole range.
