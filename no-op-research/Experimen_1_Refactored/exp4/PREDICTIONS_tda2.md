# Pre-registered predictions: TDA 2, a Mapper graph of head behaviour across depth

Written 2026-10-25, before any Mapper graph was built. Data: the saved per-row arrays of Exp 4
(`results/20261024-000548_exp4ctrl/controlled.npz`, random final token) and Exp 4b
(`results/20261024-150407_exp4cap/capital.npz`, its capital or a wrong capital as the final token). No model runs.
Code: `exp4/tda2_mapper.py`.

**Question.** Does a head's range (local vs global response to a country token) change smoothly with depth, or does it
split into separate branches? The Exp 4 log grouped layers as 2–3 / 5–8 / 9–11 after seeing the data, and the verifier
flagged that grouping as post hoc. Here the depth lens uses uniform, pre-declared windows, and the earlier grouping
is tested as a hypothesis (M1c), not assumed.

## Point cloud

- **One point per head (144 points).** The features are both filler types concatenated: 6 curves × 31 distances = 186
  numbers.
- **Curves.** The country arm's D(d), d = 1..31 (attention units), under three final tokens: its capital (own), a wrong
  capital (wrong), and a random final token (Exp 4). Each is computed for common and rare filler.
- **Why one point per head.** The hypotheses are about heads, and every earlier prediction had to hold in both filler
  types. Concatenation makes a head's position reflect both. Two points per head (288) would let filler-type
  magnitude differences, such as L11H8's +0.065 vs +0.039, create structure. That version is run as a secondary
  analysis.
- **Transform: shape with shrinkage.** For a head's 186-vector v, f = v / (‖v‖ + λ), with λ = 0.02 · √31 ≈ 0.111.
  This is a margin-sized effect sustained over one full curve.
  - A head with a clear response maps near the unit sphere, so its position reflects the shape of the response:
    range, sign, and which final token drives it.
  - A head with no response (‖v‖ ≪ λ) maps near the origin.
  - Raw attention units would let L0H2 (0.43) dominate every distance, and the graph would mostly show magnitude.
    The question is about range, so shape is what should drive the graph.
  - `asinh(D / 0.02)` features are a secondary analysis.
- **Metric:** Euclidean distance on f.

## Mapper

- **Main lens: layer index (0..11).** It is discrete, so the cover is a set of windows of w consecutive layers with
  step s. Consecutive windows share w − s layers, so the overlaps are never empty.
  - This is exactly kmapper's uniform `Cover(n_cubes = (12 − w)/s + 1, perc_overlap = 1 − s/w)` with limits
    [(w − 1 − s)/2, (w − 1 − s)/2 + n·s]. Every window boundary falls on a half-integer.
- **Alternative lenses (secondary, descriptive):**
  - **Locality lens.** F = far / (near + far + κ), with κ = 0.005:
    - near = the mean over the 6 curves of |mean D(d), d 2–4|;
    - far = the same over d 16–31.
    - F is a bounded version of the log's "share of the near effect left at d 16–31": with r = far / near,
      F ≈ r / (1 + r). κ pulls heads with no response toward 0.
  - **PC1 lens:** the first principal component of f.
  - Both use kmapper's uniform cover over [min, max] with the same (n, p) pairs as the depth grid.
- **Clustering:** single linkage within each window, cut at ε. This is DBSCAN with min_samples = 1, so every head
  appears in every window it falls in and no head is dropped as noise.
  - ε = m · ε0.
  - ε0 = the median, over responsive heads, of the distance from a head to its 3rd-nearest other head in the full
    cloud. It is recomputed on every bootstrap resample.
  - A head is **responsive** if it has at least one meaningful cell (q ≤ 0.05 and |D| ≥ 0.02) in the country arm in
    any of the 6 saved analyses (3 final tokens × 2 filler types).
- **Graph.** One node per cluster per window. Two nodes are joined when they share at least one head. All nodes
  holding a given head are therefore in one connected component.
- **Main setting:** w = 4, s = 2 (5 windows: layers 0–3, 2–5, 4–7, 6–9, 8–11; kmapper n = 5, overlap 0.5), m = 1.
- **Parameter grid (21 settings).** Seven covers (w, s), each with m ∈ {0.75, 1, 1.5}:

  | (w, s) | (2,1) | (3,1) | (4,1) | (5,1) | (4,2) | (6,2) | (6,3) |
  |---|---|---|---|---|---|---|---|
  | kmapper (n, p) | (11, 0.50) | (10, 0.67) | (9, 0.75) | (8, 0.80) | (5, 0.50) | (4, 0.67) | (3, 0.50) |

- **Bootstrap.** 200 resamples of the 200 rows. Each filler type is resampled independently. Within a filler, the same
  row indices are used for the own, wrong and random arrays, because they are the same rows.
  - Each resample rebuilds the six D curves from the per-row contrasts (`exp4.run_controlled.row_differences`), then
    f, ε0 and all 21 graphs.
  - Head sets are fixed from the saved full-data results and never re-derived.

## Head sets (fixed)

- **𝓛, near-only:** labelled "local country-tracking" in at least one filler type in Exp 4 (random final token) and
  "global" in neither.
- **𝓖, any-distance:** labelled "global country-tracking" in at least one filler type in Exp 4. This includes L9H8,
  L10H3, L11H7, L11H2 and the 13 common-only global heads.
- **S_same, same-country association heads:** L0H2, L1H11, L3H0, L4H10.
- **S_prev, Exp 1 previous-token heads:** L1H0, L2H2, L2H3, L2H5, L2H8, L2H9, L3H2, L3H6, L3H7, L3H8, L4H11.
- **S_late, late any-distance heads:** L9H8, L10H3, L11H7, L11H2.

## Operational definitions (per graph)

- **co(S), co-clustering.** Count the pairs of heads in S that fall in the same window, counted per window.
  co(S) is the fraction of those pairs that share a node. It is undefined if there are no such pairs, and the test
  then fails.
- **pur(S), purity.** Over the nodes that contain any S head: Σ |node ∩ S| / Σ |node|.
- **S forms a distinct branch (clusters together):** co(S) ≥ 0.8 and pur(S) ≥ 0.5.
  - Together, S heads share nodes, and those nodes are mostly S, not the bulk of other heads.
  - Whether the branch hangs off the main body (a flare) or is a separate component is reported, not tested.
- **Main component for M1:** the connected component that holds the most heads of 𝓛 ∪ 𝓖.
- **Split window.** A window that holds at least one 𝓛 and one 𝓖 head, and in which some node has 𝓛 heads but no 𝓖
  heads while another node has 𝓖 heads but no 𝓛 heads. In that window, local and global heads sit on separate
  branches.
- **Node locality.** F computed from the node's mean curves (the mean over its member heads).

## Hypotheses

The depth lens is used throughout. Prior expectations are given in brackets.

| id | hypothesis | holds in a graph when | prior |
|---|---|---|---|
| M1a | **One structure.** Near-only and any-distance heads are connected, not separate components | the main component holds ≥ 80% of 𝓛 ∪ 𝓖 heads, including at least one of each | likely; the bulk of null heads can link them |
| M1b | **Smooth, not branched.** Where local and global heads share a depth window, they are not split into separate nodes | no split window | unlikely: Exp 4 finding 5 says the late effect is carried by four heads |
| M1c | **The earlier grouping.** The local end sits in layers 2–3 and the global end in 9–11 | Take the main component's nodes that contain an 𝓛 ∪ 𝓖 head (at least 3 such nodes), sorted by node F; a third is ⌊n/3⌋ nodes, and a set's layer is the mean, over its nodes, of each node's mean member layer. The lowest-F third has a layer in [2, 4), and the highest-F third has a layer ≥ 9 | unlikely (post hoc in Exp 4) |
| M2 | **The same-country heads form a distinct branch** | S_same forms a distinct branch | likely: a large, own-capital-only, all-distance signature |
| M3 | **The previous-token heads cluster together** | S_prev forms a distinct branch | uncertain to unlikely: Exp 4 finding 3 shows heterogeneous content responses |
| M4 | **The late any-distance heads form their own branch at the deep end.** On the layer lens, sitting at the deep end is automatic; the test is whether they separate from the other late heads | S_late forms a distinct branch | uncertain: the 13 common-only global heads may merge with them |

## Verdicts

Each hypothesis is scored on three things:
- the main setting;
- the share of the 21 grid settings in which it holds on the full data (s_grid);
- the share of the 200 × 21 bootstrap-by-setting graphs in which it holds (s_boot).

| verdict | rule |
|---|---|
| PASS | min(s_grid, s_boot) ≥ 0.70 |
| FAIL | max(s_grid, s_boot) < 0.30 |
| UNSTABLE | otherwise |

The share of the 200 bootstrap graphs at the main setting is also reported.

## Secondary analyses (descriptive, no verdicts)

Each is run on the 21-setting grid on the full data.
- Locality lens and PC1 lens: M1a, M2, M3 and M4. M1b and M1c are depth-specific. For the PC1 lens, the report also
  says whether all four S_late heads fall in an end window.
- 288-point cloud: one point per head and filler type, 3 curves × 31 features, the same transform. The head sets
  contain both copies of each head.
- `asinh(D / 0.02)` features.

## Checks (not predictions)

- The six recomputed D curves (mean over rows of `row_differences`) match the saved `mean` arrays. The capital
  arrays are float16, so the tolerance is 1e-3.
- The hand-rolled Mapper used for the grid and bootstrap gives the same nodes and edges as kmapper 2.1.0 at the main
  setting.
- Every window in every depth cover holds w layers, and consecutive windows share w − s layers.

## Caveats known in advance

- Mapper is sensitive to its parameters, which is why the grid and the bootstrap exist.
- With 144 points, single heads decide co and pur for the small sets: S_same has 4 heads and S_late has 4.
- The head sets come from earlier analyses of the same data. The Mapper graph is not independent evidence about
  which heads respond, only about how their response shapes are arranged.
