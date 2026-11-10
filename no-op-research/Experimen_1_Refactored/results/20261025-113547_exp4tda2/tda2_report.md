# TDA 2: Mapper graph of head behaviour across depth

Run `20261025-113547_exp4tda2`. Pre-registration: `PREDICTIONS_tda2.md` (copied from `exp4/PREDICTIONS_tda2.md`, written before any graph). Data: `controlled.npz` (Exp 4, random final token) and `capital.npz` (Exp 4b), no model.

Point cloud: 144 heads × 186 features (country-arm D(d), d = 1..31, with its capital / a wrong capital / a random final token, common and rare filler), f = v / (‖v‖ + 0.111). Depth lens = layer index, windows of w layers with step s; single-linkage clustering per window cut at ε = m · ε0 (ε0 = median 3-NN distance of responsive heads). Main setting w = 4, s = 2, m = 1. Grid: 7 covers × 3 m = 21 settings; bootstrap 200 row resamples × 21 settings.

Head sets (fixed): 𝓛 (near-only) 30 heads, 𝓖 (any-distance) 17 heads, responsive 121 of 144. ε0 (full data) = 0.320.

## Checks

| check | result |
|---|---|
| recomputed D curves vs saved means (tol 1e-3) | PASS (max |Δ| 2.7e-05; own/common 1.4e-05, wrong/common 1.4e-05, random/common 0.0e+00, own/rare 1.3e-05, wrong/rare 2.7e-05, random/rare 0.0e+00) |
| depth windows hold w layers and share w − s | PASS |
| hand-rolled Mapper = kmapper 2.1.0 (main setting) | PASS (108 nodes / 51 edges (kmapper) vs 108 / 51 (ours)) |

## Pre-registered verdicts

s_grid = share of the 21 settings where it holds on the full data; s_boot = share of the 200 × 21 bootstrap graphs; main boot = share of the 200 bootstrap graphs at the main setting. PASS: min(s_grid, s_boot) ≥ 0.70; FAIL: max < 0.30; otherwise UNSTABLE.

| id | hypothesis | main setting | s_grid | s_boot | main boot | verdict |
|---|---|---|---|---|---|---|
| M1a | near-only and any-distance heads in one connected structure | fails | 0.33 | 0.32 | 0.00 | **UNSTABLE** |
| M1b | no depth window splits local and global heads into separate nodes | fails | 0.24 | 0.24 | 0.00 | **FAIL** |
| M1c | local end in layers 2–3 and global end in 9–11 | fails | 0.00 | 0.01 | 0.00 | **FAIL** |
| M2 | same-country heads (L0H2, L1H11, L3H0, L4H10) form a distinct branch | fails | 0.00 | 0.00 | 0.00 | **FAIL** |
| M3 | Exp 1 previous-token heads cluster together | fails | 0.00 | 0.00 | 0.00 | **FAIL** |
| M4 | late any-distance heads (L9H8, L10H3, L11H7, L11H2) form their own branch | fails | 0.00 | 0.00 | 0.00 | **FAIL** |

## Numbers behind the verdicts (main setting; bootstrap median [2.5%, 97.5%] over the main-setting resamples)

- M1a: main component holds 0.45 of 𝓛 ∪ 𝓖 heads; bootstrap 0.45 [0.40, 0.47].
- M1b: split windows 5 of 5 windows holding both 𝓛 and 𝓖 heads; bootstrap split count 5.00 [5.00, 5.00].
- M1c: layer of the lowest-F third 5.27, of the highest-F third 8.15 (needs [2, 4) and ≥ 9); bootstrap 5.24 [3.67, 6.02] and 7.59 [6.14, 8.15].
- M2: co 0.25, pur 0.60 (needs ≥ 0.8 and ≥ 0.5); bootstrap co 0.25 [0.25, 0.25], pur 0.60 [0.60, 0.86].
- M3: co 0.02, pur 1.00 (needs ≥ 0.8 and ≥ 0.5); bootstrap co 0.02 [0.00, 0.13], pur 1.00 [0.72, 1.00].
- M4: co 0.17, pur 1.00 (needs ≥ 0.8 and ≥ 0.5); bootstrap co 0.17 [0.00, 0.17], pur 1.00 [0.83, 1.00].
- Same-country heads in the component with the most heads (flare rather than separate component): 0.00 of the four (main setting); share over the 21 settings 0.44.

## Stability by setting (s_boot per setting; * = holds on the full data)

| w | s | m | ε | nodes | edges | components | M1a | M1b | M1c | M2 | M3 | M4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 1 | 0.75 | 0.240 | 215 | 108 | 107 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 2 | 1 | 1.0 | 0.320 | 150 | 87 | 64 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 2 | 1 | 1.5 | 0.480 | 59 | 42 | 17 | 0.68* | 0.00 | 0.32 | 0.00 | 0.00 | 0.00 |
| 3 | 1 | 0.75 | 0.240 | 274 | 262 | 99 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 3 | 1 | 1.0 | 0.320 | 177 | 186 | 58 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 3 | 1 | 1.5 | 0.480 | 54 | 84 | 7 | 1.00* | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 4 | 1 | 0.75 | 0.240 | 312 | 413 | 97 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 4 | 1 | 1.0 | 0.320 | 191 | 278 | 54 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 4 | 1 | 1.5 | 0.480 | 41 | 87 | 5 | 1.00* | 1.00* | 0.00 | 0.00 | 0.00 | 0.00 |
| 5 | 1 | 0.75 | 0.240 | 330 | 532 | 93 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 5 | 1 | 1.0 | 0.320 | 191 | 337 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 5 | 1 | 1.5 | 0.480 | 34 | 75 | 5 | 1.00* | 1.00* | 0.00 | 0.00 | 0.00 | 0.00 |
| 4 | 2 | 0.75 | 0.240 | 174 | 75 | 99 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 4 | 2 | 1.0 | 0.320 | 108 | 51 | 57 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 4 | 2 | 1.5 | 0.480 | 21 | 18 | 5 | 1.00* | 1.00* | 0.00 | 0.00 | 0.00 | 0.00 |
| 6 | 2 | 0.75 | 0.240 | 190 | 135 | 91 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 6 | 2 | 1.0 | 0.320 | 106 | 79 | 49 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 6 | 2 | 1.5 | 0.480 | 16 | 16 | 5 | 1.00* | 1.00* | 0.00 | 0.00 | 0.00 | 0.00 |
| 6 | 3 | 0.75 | 0.240 | 145 | 51 | 94 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 6 | 3 | 1.0 | 0.320 | 80 | 30 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 6 | 3 | 1.5 | 0.480 | 11 | 6 | 5 | 1.00* | 1.00* | 0.00 | 0.00 | 0.00 | 0.00 |

## Secondary analyses (descriptive, full data, share of the 21 settings where it holds)

| analysis | M1a | M1b | M1c | M2 | M3 | M4 | other |
|---|---|---|---|---|---|---|---|
| locality lens | 0.33 | n/a | n/a | 0.05 | 0.00 | 0.00 | late heads all in an end window: 0.29 |
| PC1 lens | 0.33 | n/a | n/a | 0.00 | 0.00 | 0.00 | late heads all in an end window: 0.86 |
| 288 points (head × filler), depth lens | 0.33 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |  |
| asinh(D / 0.02) features, depth lens | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |  |

## Exploratory, not pre-registered

**ε sweep** (main cover, full data, m = 0.50..2.50 in steps of 0.05; `figures/eps_sweep.png`). The grid's m values straddle the scale at which the graph goes from mostly singletons to one blob per window, so this asks whether any scale in between gives the predicted structure.

| m | components | largest component (share of heads) | same co / pur | prev co / pur | late co / pur |
|---|---|---|---|---|---|
| 0.50 | 138 | 0.02 | 0.00 / 1.00 | 0.00 / 1.00 | 0.00 / 1.00 |
| 0.75 | 99 | 0.09 | 0.00 / 1.00 | 0.00 / 1.00 | 0.00 / 1.00 |
| 1.00 | 57 | 0.46 | 0.25 / 0.60 | 0.02 / 1.00 | 0.17 / 1.00 |
| 1.25 | 26 | 0.75 | 0.50 / 0.60 | 0.34 / 0.50 | 0.50 / 0.06 |
| 1.50 | 5 | 0.97 | 1.00 / 0.07 | 0.74 / 0.17 | 1.00 / 0.05 |
| 2.00 | 1 | 1.00 | 1.00 / 0.04 | 1.00 / 0.15 | 1.00 / 0.05 |
| 2.50 | 1 | 1.00 | 1.00 / 0.04 | 1.00 / 0.15 | 1.00 / 0.05 |

- M1a holds at m = 1.30–2.50.
- M1b holds at m = 1.40–2.50.
- M1c holds at m = never.
- M2 holds at m = never.
- M3 holds at m = never.
- M4 holds at m = never.

**PC1 of the features** explains 0.44 of the variance. Spearman with layer -0.17, with locality F -0.61. The four late any-distance heads rank L9H8 22, L10H3 5, L11H7 2, L11H2 26 of 144 from the low end; the 8 heads at that end: L11H8, L11H7, L10H7, L4H7, L10H3, L0H10, L11H3, L1H5.

**Nearest heads in feature space** (full data, Euclidean on f; fact = mean own − wrong):
- L0H2: L4H10 0.15 (fact +0.230), L1H11 0.27 (fact +0.110), L0H10 0.31 (fact +0.055), L1H6 0.32 (fact +0.040)
- L1H11: L0H10 0.27 (fact +0.055), L0H2 0.27 (fact +0.351), L4H10 0.31 (fact +0.230), L0H1 0.39 (fact +0.018)
- L3H0: L4H10 0.36 (fact +0.230), L0H2 0.40 (fact +0.351), L1H6 0.43 (fact +0.040), L0H1 0.51 (fact +0.018)
- L4H10: L0H2 0.15 (fact +0.351), L0H10 0.27 (fact +0.055), L1H11 0.31 (fact +0.110), L1H6 0.31 (fact +0.040)
- L9H8: L6H4 0.32 (fact +0.019), L8H11 0.35 (fact +0.007), L10H0 0.37 (fact +0.005), L7H7 0.44 (fact -0.003)
- L10H3: L11H7 0.29 (fact -0.000), L11H2 0.44 (fact -0.049), L7H7 0.44 (fact -0.003), L6H2 0.44 (fact +0.001)
- L11H7: L10H3 0.29 (fact -0.021), L7H6 0.32 (fact -0.009), L2H7 0.33 (fact -0.001), L7H7 0.34 (fact -0.003)
- L11H2: L7H6 0.37 (fact -0.009), L1H9 0.39 (fact -0.005), L5H9 0.39 (fact -0.023), L10H3 0.44 (fact -0.021)

## Main graph (auto description)

- 108 nodes, 51 edges, 57 connected components; heads per component (largest first): 66, 8, 6, 4, 3, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1.
- window 0 (layers 0–3): 34 nodes, sizes [8, 6, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
- window 1 (layers 2–5): 31 nodes, sizes [7, 5, 3, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
- window 2 (layers 4–7): 17 nodes, sizes [24, 7, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
- window 3 (layers 6–9): 12 nodes, sizes [37, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
- window 4 (layers 8–11): 14 nodes, sizes [30, 4, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
- same-country: L0H2 (comp 10, smallest node 6: L0H1, L0H5, L0H10, L1H6, L1H11); L1H11 (comp 10, smallest node 6: L0H1, L0H2, L0H5, L0H10, L1H6); L3H0 (comp 16, smallest node 1: alone); L4H10 (comp 37, smallest node 1: alone)
- previous-token: L1H0 (comp 15, smallest node 1: alone); L2H2 (comp 4, smallest node 2: L3H7); L2H3 (comp 29, smallest node 1: alone); L2H5 (comp 5, smallest node 1: alone); L2H8 (comp 9, smallest node 1: alone); L2H9 (comp 30, smallest node 1: alone); L3H2 (comp 7, smallest node 1: alone); L3H6 (comp 32, smallest node 1: alone); L3H7 (comp 4, smallest node 2: L2H2); L3H8 (comp 6, smallest node 1: alone); L4H11 (comp 34, smallest node 1: alone)
- late any-distance: L9H8 (comp 46, smallest node 1: alone); L10H3 (comp 52, smallest node 2: L11H7); L11H7 (comp 52, smallest node 2: L10H3); L11H2 (comp 56, smallest node 1: alone)

## Interpretation (written after viewing `figures/mapper_main.png`; not generated by the script)

**Shape of the main graph: one spine and many islands.**
- The spine is a chain of large nodes (24, 37 and 30 heads in the windows for layers 4–7, 6–9 and 8–11). It holds 66
  of the 144 heads. Its nodes have small responses (pale in the size panel).
- Off the spine, almost every head with a distinctive response is an island of its own. The windows for layers 0–3
  and 2–5 have 34 and 31 nodes, most of them singletons.
- Only three islands contain more than one head and include a named head:
  - {L0H2, L1H11, L0H1, L0H5, L0H10, L1H6}, an early same-country family (light red in the fact panel);
  - {L2H2, L3H7};
  - {L10H3, L11H7}.
- There is no fork. Local and global heads do not form two coherent tracks along depth; each distinctive head stands
  apart.

**Why no setting shows the predicted branches (ε sweep).**
- As ε grows, heads join the bulk one at a time. The largest component holds 0.09, 0.46, 0.75 and 0.97 of the heads
  at m = 0.75, 1, 1.25 and 1.5.
- Every named set joins the bulk before its members join each other. For example, the late heads have co 0.50 with
  pur 0.06 at m = 1.25.
- So at this resolution, range neither changes along one smooth path with depth nor splits into two coherent
  branches. The responsive heads are individually distinct (roughly equidistant in shape space), and the only
  connected structure is the low-response mid and late heads.
- **M1a** holds only at m = 1.5, where the largest component holds 0.97 of the heads (main cover). The link there
  runs through the bulk, not through heads of intermediate range.
- **M1b** holds only at m = 1.5 with windows of 4 or more layers, which is the same collapse.

**Range does lean deeper, but not at the earlier ends.** The highest-F third of 𝓛 ∪ 𝓖 nodes sits at a mean layer
of 8.15 [bootstrap 6.14, 8.15], against 5.27 [3.67, 6.02] for the lowest-F third. That is not 2–3 vs 9–11.

**What the named sets look like in feature space.**
- L0H2's nearest head is L4H10 (0.15). The early same-country family also includes L0H10, L1H6 and L0H1, which have
  smaller own − wrong effects (+0.02 to +0.06).
- L10H3 and L11H7 are each other's nearest heads (0.29). L9H8's nearest heads are the common-only any-distance heads
  L6H4, L8H11 and L10H0. L11H2's are L7H6, L1H9 and L5H9.
- The previous-token heads are singletons except for L2H2 with L3H7, which fits Exp 4 finding 3: their content
  responses differ.

## Caveats

- **The depth lens works against M2.** L0H2 and L4H10 are nearest neighbours in feature space, but only windows of at
  least 5 layers (3 of the 7 covers) contain both layer 0 and layer 4. M2 also fails without depth: it holds in
  0 of 21 settings under the PC1 lens and 1 of 21 under the locality lens.
- **The features are 186-dimensional and there are only 144 points.** Pairwise distances are fairly uniform, so single
  linkage switches abruptly from mostly singletons to one blob. The asinh features show the same.
- Mapper is sensitive to the cover and the clustering scale; the verdicts use the pre-declared grid and the row bootstrap, but a different feature transform or clustering rule could change the picture (see the secondary rows).
- 144 points. For the 4-head sets a single head moving between nodes flips co and pur.
- The bootstrap resamples rows (the stimulus noise); it does not resample heads, countries or the 11 fixed country–capital pairs.
- The head sets 𝓛, 𝓖 and the named heads come from earlier analyses of the same data, so the graph tests how their response shapes are arranged, not which heads respond.
- Locality F is ill-defined for heads with no response (κ pulls them toward 0); read the locality panel together with the response-size panel.
- The depth lens is discrete (12 values); the windows are aligned to whole layers by construction.

## Files

- `figures/mapper_main.png`: main graph, 4 colourings, node size = member count, named heads labelled.
- `figures/stability.png`: s_boot per setting and hypothesis.
- `figures/eps_sweep.png`: exploratory ε sweep at the main cover.
- `mapper_main.html`: kmapper interactive graph of the main setting (local only).
- `main_graph.json`: nodes (window, members), edges and node attributes of the main graph.
- `stability.csv`: one row per analysis and setting, with full-data holds, bootstrap shares and graph sizes.
- `verdicts.json`.
