# Pre-registered predictions: TDA 1, persistent homology of head behaviour profiles

Written 2026-10-25, before any topology was computed on these data. Only label counts and a check that the per-row
arrays rebuild the saved D(d) means (exact for Exp 4, ≤ 3e-5 for Exp 4b's float16 arrays) were looked at. The question:
are the six head types from Exp 4's hand-set thresholds real, persistent clusters of heads? And do continuous families
(loops) exist? Code: `exp4/tda1_heads.py`. It uses saved results only; no model is run.

## Data

- Exp 4 (random final token): `results/20261024-000548_exp4ctrl/controlled.npz`.
- Exp 4b (capital as the final token, own and wrong): `results/20261024-150407_exp4cap/capital.npz`.
- The same 200 rows per filler type in both runs. Per-row contrasts are rebuilt with
  `exp4.run_controlled.row_differences` ([R = 200, 31 distances, 12 layers, 12 heads]). Distance index k = d − 1,
  τ = 32 − d.

## Head profile

- One point per head: 144 points.
- Each point is the concatenation of 8 D(d) curves, d = 1..31 (248 features), in this order. For each filler type
  (common, then rare):
  1. country arm, random final token (Exp 4);
  2. country arm, its own capital as the final token (Exp 4b own);
  3. country arm, a wrong capital (Exp 4b wrong);
  4. placebo-word arm, random final token (Exp 4).
- D(d) is the mean over rows of a(variant) − a(control) on the attention from the final token to τ, as in Exp 4.
- prev_score is not in the profile. The profile describes how attention to τ changes, not where the head attends at
  rest, so it cannot tell "plain previous-token" from "no country effect" (labels 5 and 6).

## Metrics (variants)

- **raw (main):** Euclidean distance between profiles in raw attention units. Every feature has equal weight, so heads
  with large effects dominate the geometry.
- **corr (robustness):** correlation distance 1 − Pearson r between two heads' 248-feature profiles. It ignores
  magnitude and compares shape only. For heads whose profile is mostly noise, r ≈ 0 against everyone.
- **std (robustness):** each of the 248 features is z-scored across the 144 heads (ddof 0), then Euclidean distance is
  used. This gives every distance and condition equal weight, but it also amplifies features where every head is near
  zero.

Verdicts come from raw. A verdict counts as robust when corr and std give the same verdict.

## Persistence

- Vietoris–Rips persistent homology, H0 and H1, with ripser on the 144 × 144 distance matrix (`distance_matrix=True`,
  `maxdim=1`).
- H0 bars are born at 0 and die at single-linkage merge heights. A cut at scale ε gives 1 + #(finite bars with death
  > ε) components, which are the single-linkage flat clusters at ε.
- H1 features are summarised by persistence (death − birth) and by the scale-free ratio death / birth (multiplicative
  persistence).

## Stability (bootstrap over rows)

- B = 500 resamples of the 200 rows, drawn independently per filler type. Within a filler type, the same resampled rows
  are used in all three final-token conditions and both arms, because they are the same rows.
- Profiles are recomputed from the per-row arrays, then the distances and diagrams.
- **Fasy et al. (2014) band:** c_k = 95th percentile over resamples of the bottleneck distance between the resample's
  Hk diagram and the main Hk diagram (gudhi; the infinite H0 bar is dropped from both). A feature is distinguishable
  from estimation error when its persistence exceeds 2·c_k, i.e. its L∞ distance to the diagonal exceeds c_k.
- **Per-cluster recurrence (Hennig 2007):** each resample is cut at the same fixed ε*. For each component of the main
  partition, recurrence is the mean over resamples of its best Jaccard overlap with a resample component.
  - ≥ 0.75: stable.
  - < 0.5: dissolved.

## Null models (each 500 draws)

- **N1, row sign-flip (noise null; primary for H0).** Each row's contrasts are multiplied by a random sign s_r = ±1,
  and profiles are recomputed as means over rows.
  - Within a filler type, the same sign is used for all heads, distances, conditions and arms of a row. This keeps the
    noise correlations of the real profiles (shared control rows, shared rows across runs, rows that move many heads
    together) but sets every true effect to zero.
  - This is the randomisation null of a paired design, the same unit as Exp 4's Wilcoxon tests.
  - A bar that outlives N1's maximum is a separation that estimation noise alone does not produce.
- **N2, moment-matched Gaussian (shape null; primary for H1).** 144 points drawn from a Gaussian with the real profiles'
  mean and covariance (μ + Z·(P − μ)/√143, Z standard normal [144 × 144]), then transformed by each variant's metric.
  - It is a unimodal cloud of the same size, scale and shape, with no clusters and no holes.
  - A loop that such a cloud produces as often by sparse sampling is not evidence of a continuous family.
- Null statistics per draw: the largest finite H0 death, and the largest H1 death / birth ratio (and the largest H1
  persistence).

## Persistent clusters

- **Cut:** ε* = max(q95 of N1's largest finite H0 death, 2·c_0). Every bar dying above ε* is beyond the noise null and
  outside the bootstrap band.
- **Cluster:** a component at ε* with ≥ 2 heads. Components with one head are reported as isolated heads.
- **Labels for comparison:** Exp 4's six labels (random final token), per filler type.

## Predictions

| id | prediction (raw metric; PASS criterion) | expectation and why |
|---|---|---|
| T1-a | **The head population splits into ≥ 3 real clusters.** At ε*, at least 3 components with ≥ 2 heads each have bootstrap recurrence ≥ 0.75 | Expected **FAIL**. Most heads have small, graded effects, and single linkage will chain them into one large component within noise. Strong heads (L0H2, L4H11, L11H8, the late trackers) should be isolated singletons, not groups: their profiles differ from one another by much more than the noise. |
| T1-b | **The same-country heads form their own cluster.** L0H2, L4H10, L3H0 and L1H11 are in one component at ε*. It has ≤ 8 heads and contains none of the 11 Exp 1 previous-token heads (L1H0, L2H2, L2H3, L2H5, L2H8, L2H9, L3H2, L3H6, L3H7, L3H8, L4H11). The same holds in ≥ 75% of bootstrap resamples | Expected **FAIL in raw**: L0H2's own-capital effect (~0.4 at every d) is several times the others'. That puts it far from L4H10, L3H0 and L1H11. Expected closer to passing under corr, which compares shape only. |
| T1-c | **The threshold labels are real clusters.** ARI between the ε* partition and Exp 4's labels is ≥ 0.5 in both filler types. Bands: ≥ 0.5 means the labels are real; 0.2–0.5, partly; < 0.2, the labels do not match persistent clusters | Expected **FAIL (ARI < 0.2)**. The labels split small effects by significance thresholds (q ≤ 0.05, \|D\| ≥ 0.02, 15 of 30 distances). Such cuts fall inside a continuum, not at gaps. Reference ceiling: ARI between the common and rare labels themselves. |
| T1-d | **No continuous family.** No H1 feature has both persistence > 2·c_1 and death / birth above N2's 95th percentile of the largest ratio | Expected **PASS**. No known mechanism makes a cyclic family of head behaviours. The effects vary along a few graded axes: magnitude, locality, own-capital association and suppression. A 144-point cloud in 248 dimensions gives short-lived loops. |

## Also reported (not predictions)

- **ARI variants.**
  - With labels 5 and 6 merged, because the profile cannot separate them.
  - A 95% bootstrap interval of each ARI, using the resample partitions at ε* against fixed labels.
  - For comparison, the ARI of a six-component single-linkage cut and of Ward k = 6, both against the labels.
- **H0 and H1 under the other null.** H0 bars against N2, and H1 against N1.
- **Focus heads.** The component of each:
  - Exp 1 previous-token heads;
  - the late any-distance heads L9H8, L10H3, L11H7, L11H2;
  - L11H8 and L4H11.
- **Checks.**
  - Rebuilt per-row means equal the saved means (Exp 4 max |Δ| = 0; Exp 4b ≤ 1e-4, float16 storage).
  - ripser's finite H0 deaths equal scipy's single-linkage heights.
  - N1's mean profile is ≈ 0.

## Caveats known in advance

- **144 points is a small cloud.**
  - H0 can find gaps, but density-based structure (modes inside the large component) is invisible to single linkage.
  - H1 on 144 points in 248 dimensions has low power. A real loop would need several heads spread around it, so
    "no H1" does not rule out families that are thinly sampled.
- **The metric decides what "close" means.** raw is magnitude-dominated; corr and std are noise-sensitive for near-zero
  heads.
- **The labels are per filler type and use prev_score**, which the profile lacks.
