# Pre-registered predictions: TDA analysis 4, topology of the full attention graph

Written 2026-10-25, before any full attention pattern was extracted. What I had seen when writing this: `exp4/LOG.md`
(the verified Exp 4 / 4b findings quoted below), `exp4/PREDICTIONS_capital.md`, `configs/exp4_capital.yaml` and the
code. I had not looked at any query row other than the final one, in any run. Code: `exp4/tda4_graph.py`. Config:
`configs/exp4_tda4.yaml`, which adds the TDA settings to `configs/exp4_capital.yaml` (same stimuli, rows and seeds).

The question: does a matching country–capital pair change the **topology** of the attention network? For example, the
capital's attention to its country plus the previous-token chain could close a loop (a persistent 1-cycle) that is
absent with a wrong capital.

## Design

**Rows.** The same 200 cached rows per filler type (common ids 256–999, rare 1000–39999), with BOS at 0 and tokens at
1..32. They come from `exp2.design.cached_base_block("exp3rel_<filler>", …)` with the stimuli and demonstration ids
excluded, exactly as `exp4/run_capital.py` does. All 200 rows are used. The forward passes are about 97k sequences
(~10 min), so there is no runtime reason to cut rows.

**Conditions.** Each row is used 11 times, once per country c; a row's graph is the mean of its 11 sequences.

| condition | position τ | position 32 |
|---|---|---|
| control | the row's own token | capital of c (independent of τ) |
| own | country c | capital of c |
| wrong | country c | capital of wrong(c); `latin_derangements(11, 1, 5)[0]`, France→Tokyo, … |
| random | country c | the row's own token (as in Exp 4) |

- τ ∈ {31, 30, 28, 24, 16, 8, 1}.
- Each capital appears once per row in own and in wrong. So the control set is the same 11 sequences whichever capital
  assignment is used.

**Structural fact (not a prediction).** Attention is causal, so query rows 0..31 depend only on tokens 0..31. Own,
wrong and random share those tokens, so their graphs are identical except for the 32 edges at node 32 (the capital's
query row). This is checked numerically. So every own-vs-wrong topological difference comes from how node 32 attaches
to an otherwise fixed weighted graph. Topology adds one thing to the raw weights: whether the strengthened edges
close loops or only pull node 32 in tighter. Own vs control also differs on query rows τ..31.

**Graphs.** Each graph is a 33 × 33 causal attention matrix A[q, k], mean over a row's 11 sequences and then over
rows. One graph per unit:
- **Layer means:** the mean over the 12 heads of each layer (12 units).
- **Named heads:**
  - same-country heads L0H2, L1H11, L3H0, L4H10;
  - previous-token heads L2H2, L2H9, L3H7, L4H11.
- **Pooled (descriptive):** mean over layers 0–4, and mean over all layers. A loop made of a layer-0 country edge plus
  a layer-2–4 chain can only show up when layers are pooled.

**Undirected weight.** w(i, j) = max(A[i, j], A[j, i]). Under the causal mask the upper triangle is 0, so this equals
A[i, j] + A[j, i] = A[max(i,j), min(i,j)]. No weight is lost or double-counted. Self-attention (the diagonal) is not
an edge.

**Filtration.** Distance = 1 − w. Vertices are born at 0, and edges enter as the weight threshold decreases.
- The complex is the flag (clique) complex, up to triangles, built with gudhi (`SimplexTree.create_from_array`,
  `expansion(2)`). H0 and H1 are computed over Z/11.
- −log w gives the same ordering, so the same persistence pairs; only the filtration values differ. 1 − w is used
  because it is bounded.
- ripser on the same distance matrices is a cross-check.

**BOS.** BOS is a hub: in most layers every token's edge to BOS is far heavier than any content edge. Once a cycle's
nodes all have their BOS edges, the cone over BOS fills the cycle. Two variants:
- **noBOS (primary):** node 0 removed, raw weights kept (no renormalisation). The late BOS drop with the matching
  capital (Exp 4b) then shows up only as extra weight on node 32's content edges.
- **withBOS:** all 33 nodes. This is expected to have almost no H1 beyond layer 0 (see B1).

**Features per graph.**
- H0 and H1 diagrams (H0 without its infinite bar).
- Specific edges, with weights (birth = 1 − w) and bootstrap 95% CIs over rows:
  - E1 = {32, τ}, the capital → country edge;
  - E2 = {τ+1, τ}, the previous-token edge into the country (E2 = E1 at τ = 31);
  - E3 = {32, 31}, the capital's previous-token edge.
- **Role of E1** in the filtration:
  - **merge**: it is an H0 death edge and joins two components;
  - **cycle**: it births an H1 class of positive persistence;
  - **instant**: it births a class that a triangle fills at once, because 32 and τ already share a neighbour.
- **Representative cycle** of an H1 class born at edge (u, v) with filtration b: that edge plus a hop-shortest path
  from u to v among the edges with filtration < b. This represents the new class modulo older classes, which is the
  elder-rule meaning of "new".
- **Persistent:** death − birth ≥ 0.002, and death weight ≤ 0.75 × birth weight. The class survives a quarter of its own
  weight scale, and at least 0.002 absolute (about 2 row-SEs of a layer-mean edge).
- **Matching-pair cycle (MPC):** a persistent H1 class whose representative cycle contains the edge {32, τ}.

**Comparison statistic.**
- **D:** 1-Wasserstein distance (internal L∞) between H0 diagrams plus that between H1 diagrams (gudhi.hera).
- **Bottleneck:** the max over the two dimensions. Reported, not tested.

**Null (primary): paired label swap.** In each of 200 swaps, every row's own and wrong graphs trade labels with
probability ½. The two condition means are rebuilt and D recomputed.
- This is the exact test of "own and wrong graphs are exchangeable" at the full sample size, and it respects the
  pairing.
- p = (1 + #{null ≥ observed}) / 201. "Beyond the null" means p ≤ 0.05.
- The same swap null is built for own vs random and own vs control (descriptive).

**Split-half scale (descriptive).** Over 100 random splits of the rows into halves, D(own on half A, own on half B).
These are 100-row means, so the scale is about √2 noisier than full-sample noise. It is shown as context, not used in
a pass rule.

**Stability bootstrap.** 100 row resamples with replacement, with the same indices for every condition. Each
resample's averaged graphs are rebuilt, and I report how often each one has an MPC and how often E1 takes each role.

## Hypotheses and pass rules

Primary variant: noBOS. A rule that names both filler types must hold in each of them.

| id | prediction | pass rule | expectation, and why |
|---|---|---|---|
| G1 | **At τ = 31, the early layers' topology differs between own and wrong.** | Own-vs-wrong D beyond the swap null (p ≤ 0.05) in each of layers 0–4, both fillers. This is a conjunction, so there is no multiplicity correction. | Likely PASS. E1 = E3 at τ = 31. Same-country heads raise it (L0H2 0.43 vs 0.03) and previous-token heads lower it (L3H7 0.51 → 0.24, L4H11 0.98 → 0.86–0.91), with freed mass to key 30 and self. Layers 3–4 could partly cancel, which is the main risk. |
| G2 | **A matching-pair cycle exists with its own capital but not with a wrong capital.** | At least one cell (layer-mean unit L0–L11, τ) where, in both fillers, own has an MPC in the full-sample graph with bootstrap frequency ≥ 0.8, and wrong has none in the full-sample graph with frequency ≤ 0.2. | PASS expected in layers 0–4 at τ ≤ 24: the country edge enters early with its own capital, after the local chain τ..32 is in place. Risk: a secondary sink (e.g. position 1) or strong 2-back chords give 32 and τ a common neighbour first, so E1 is "instant" or a "merge". There are 84 cells per filler, so I also report the reverse count (wrong has a stable MPC, own does not) as a false-positive gauge. |
| G3a | **At far τ, early layers carry the difference on the country edge alone.** | At τ ∈ {16, 8, 1}, for every (filler, τ, layer 0–4) cell that is beyond the null: set own's A[32, τ] to wrong's value (transplant), and D(own′, wrong) must fall to ≤ the null's 95th percentile. Needs at least one eligible cell. | Likely PASS: early same-country heads move one edge, and their other mass comes mostly from BOS, which is outside the noBOS graph. |
| G3b | **At far τ, late layers do not.** | At τ ∈ {16, 8, 1}, layers 6–11: in at least half of the cells beyond the null, the transplant leaves D(own′, wrong) above the null's 95th percentile. Needs at least one eligible cell. | Uncertain-to-likely PASS. Exp 4b: the matching capital frees about 0.03 of late BOS mass; about 40% goes to key τ+1 and the rest is diffuse, which moves node 32's other edges. Late own-vs-wrong differences may also simply be inside the null, leaving no eligible cell (then FAIL). |
| B1 | **Methods expectation: BOS fills cycles.** | withBOS: no MPC in the full-sample graph in any layer 1–11, any τ, either filler. | Near-certain PASS. A[32, 0] and A[τ, 0] are much heavier than A[32, τ], so the triangle {0, 32, τ} enters with E1. This confirms that noBOS is the variant where the question can be asked at all. |

Not predicted, reported:
- E1/E2/E3 weights and their own − wrong / country − control CIs.
- The role of E1 in each condition.
- MPCs in the named-head and pooled units.
- MPCs in the control and random conditions at the G2 cells, as baselines.
- Own vs random and own vs control distances against their own swap nulls.
- The number of persistent H1 classes per graph.

A per-head graph has a cycle through {32, τ} only if that head also carries a path from τ to 32. A same-country head
alone may therefore show E1 as a "merge", and a previous-token head alone has E1 ≈ 0. Such loops, if any, are expected
mostly in the layer means and pooled units, where heads combine.

## Checks (not predictions)

- **Design asserts per τ and condition:** the variant equals its row except at τ and 32, with the right ids there; no
  repeats or BOS; no stimulus id in a base row; the derangement has no fixed point; each capital once per row.
- **Attention rows sum to 1** for every query of every head (asserted per batch, atol 1e-4). The upper triangle is 0
  (causal).
- **Prefix identity:** own, wrong and random per-row graphs have identical query rows 0..31 (max |Δ| ≤ 1e-6).
- **Determinism:** recomputing the first batch of one condition gives max |Δ| = 0.
- **Consistency with saved runs:** query-32 weights on τ, BOS and 31 for the named heads and layer means match Exp 4b's
  `capital.npz` (own, wrong; float16, atol 2e-3) and Exp 4's `controlled.npz` (random), at every τ.
- **gudhi vs ripser:** bottleneck distance between their diagrams ≤ 1e-6 for every full-sample graph.

## Caveats known in advance

- Averaging 11 × 200 sequences gives one graph per condition, so the topology is that of the **mean** attention, not
  a per-sequence topology. The null and the bootstrap address sampling noise, not that choice.
- A layer mean dilutes single heads 12-fold, and the noBOS raw weights are small in late layers. The persistence
  thresholds (0.002 absolute, 25% relative) are a judgement call made here, before the data.
- Representative cycles from shortest paths are one valid choice among many. "Contains {32, τ}" is a property of
  that representative, not an invariant of the class.
