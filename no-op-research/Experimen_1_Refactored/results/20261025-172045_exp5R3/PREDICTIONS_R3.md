# R3 operational addendum: distance dependence of the stream at T

Written 2026-10-25, before `r3_distance.py` ran on the full data (developed on 2-row smoke data only). Hypotheses R3-a and R3-b and
their pass rules are exactly those of `exp5/PREDICTIONS.md`; nothing below changes them.

## Operational choices

- **Data:** `means_<b>.npz` for b in {common, rare}, rows R = 100. Layers l = 1..12 (state 0 is zero by construction).
  m_l(τ) = mean over rows of [resid(own, d) − ctrl_resid] with d = 32 − τ, d = 1..31.
- **Normalisation (primary): one scale per layer from the controls**, s_l = sqrt(mean over rows of ‖ctrl_resid_l‖²) (the
  own-capital control for own and wrong, `rnd_ctrl_resid` for random). Reason: the stored states are averages over 11
  cells, so a per-state RMS is an RMS of an average, not LayerNorm's; and scaling the country and control states by their
  own RMS before subtracting adds a pure scale-change term to m. A single scale per layer is the same for all τ and rows.
  Note D_l is a ratio, so any per-layer constant cancels: the primary D_l equals D_l on raw (mean-centred) states.
  Only the heatmap of ‖m_l‖ uses s_l.
- **Robustness: per-state RMS.** Each state (country run and control, per row) is divided by its own RMS over the 768
  dimensions before subtracting. Reported for all three conditions, same tests.
- **D_l:** ‖mean_{d=1..4} m_l − mean_{d=16..31} m_l‖ / mean_{d=1..31} ‖m_l(d)‖.
- **Bootstrap:** 2000 row resamples (rows drawn with replacement; m and D recomputed from the resampled mean), seed 0,
  percentile 95% CI (2.5, 97.5). "CI below 0" means the 97.5th percentile < 0. Statistics per resample: Spearman(l, D_l)
  over l = 1..12 and mean D(l 9–12) − mean D(l 1–4).
- **Verdict:** R3-a PASS for a filler if the primary Spearman point estimate < 0 and its CI is below 0; same for R3-b.
  The hypothesis passes overall only if it passes in both fillers (own capital, primary normalisation).
- **Other conditions (wrong capital, random final token):** same statistics, descriptive only.
- **Trajectory descriptives** (point estimates on the row-mean m_l(d), own capital primary, also the other conditions in the
  npz): ‖m_l(d)‖/s_l by distance; path length (Σ_d ‖m(d+1) − m(d)‖, d = 1..30) over end-to-end ‖m(31) − m(1)‖; PCA
  participation ratio (Σλ)²/Σλ² of the 31 points (centred); Rips H0 (ripser) of the 31 points in the raw space: the
  largest and second-largest finite death times, their ratio, and the 2-cluster cut of the single-linkage tree (equal to
  cutting the H0 merge tree at its largest death): sizes and mean distance of the two clusters, and whether the near
  distances d ≤ 4 and the far distances d ≥ 16 fall in different clusters.
- **Caveats stated in the report:** the same 11 country–capital pairs; m is a mean over rows and 11 cells, so noise in the
  per-τ points is that of 100 rows; D compares few-vs-many τ pooled blocks (4 vs 16 distances); a norm-ratio D can fall
  with depth simply because ‖m‖ grows or its distance-independent component grows; no multiplicity correction (two pre-registered
  tests, each with a fixed rule).
