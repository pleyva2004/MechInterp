# Experiment 5 log: topology of the residual stream

## Setup (2026-10-25)

**Why.** Exp 4 / 4b measured attention: where a head at the final token T looks. Exp 5 measures what moves and what is
carried: each head's write h_{l,i}(T) and the stream state x_l(T) at T. The motivation and the original proposals
R1–R5 are in `onepager/residual_topology_notes.pdf`. Pre-registration: `exp5/PREDICTIONS.md`, plus one operational
addendum per analysis (`exp5/PREDICTIONS_<id>.md`), each written before that analysis read the full data.

**Design change from Exp 4b: capital pairing.**
- Cell k has capital k at T in both conditions. own = country k at τ (France … Paris). wrong = the country whose
  deranged capital is k (China … Paris), using the Exp 4b derangement, seed 5. control = the row's own token at τ.
- With the Exp 4b pairing (same country, different capital), every position before T would be identical in the two
  runs, so patching T's stream would reproduce the whole effect by construction.
- The capital pairing also gives a built-in null: x_0(T), the capital's embedding plus position, is identical for own
  and wrong.
- Also corrected from the notes: the capital's own logit at T is not a meaningful measure, so P measures late resting
  on BOS instead.

**Extraction** (`exp5/extract.py`, `configs/exp5_residual.yaml`, run `results/20261025-170021_exp5res`, 20 min, 1.5 GB):
- 100 rows per filler type, 11 cells.
- Conditions: own, wrong, random final token, and controls.
- Per-sequence states at τ ∈ {31, 16, 1}; per-row means at all 31 τ; raw and value-weighted attention graphs at 7 τ.
- All checks pass: design asserts, rows sum to 1, determinism 0, built-in null exactly 0. The Hugging Face cross-check
  matches to within 7.3e-6 on stream states, the final LayerNorm, z and per-head writes.
- **TransformerLens processing notes** (attention is unaffected, but these vectors are):
  - centred writing weights make every state and write mean-free, so a TL state equals the HF state minus its mean;
  - folded biases mean each head's z and write exclude the constant (β_ln1 W_V + b_V) W_O, which is moved into b_O;
  - the value norms in the graphs exclude the same constant.

**Analyses:**
- R1–R5 used the saved extraction; P ran the model itself.
- Six agents ran in parallel.
- Three independent verifiers followed. R5 and P had also each been re-implemented by a second agent.

## Pre-registered verdicts

| id | verdict | short |
|---|---|---|
| R1-a | PASS | Own and wrong x_0(T) identical (max \|Δ\| 0). |
| R1-b | PASS | A consistent own − wrong direction from l = 1, every τ, both fillers. The report's per-δ null was anti-conservative: within a cell the δ's are near-identical, so the effective n is 11 pairs. The exact cell-level sign-flip null (2^11 patterns) still gives p = 1/1024 everywhere, with a margin of 1.3–1.6×, not "≥ 9.7×". |
| R1-c | FAIL | Two-component Rips H0 gap. The null is degenerate (gap > 0). Early failures come from the Ottawa cell (largest L0H2 gain). Late failures come from the single-linkage gap being fragile in sparse clouds, not from specific outlier rows. |
| R2-a | PASS | ρ(BOS share, write spread) = −0.673 / −0.696. Holds within layers and controlling for layer. |
| R2-b | PASS | ρ(Δ share, Δ spread), layers 6–11 = −0.785 / −0.815. Almost entirely dilution (see findings). |
| R3-a/b | FAIL, opposite direction | Distance dependence rises with depth (ρ +0.53 / +0.82). |
| R4-a/b | FAIL | Mapper l* rule fails. At the main setting only the final window (layers 11–12) mixes; it is not the only failure across the full grid. |
| R5-a | FAIL | The layer-0 own-only loop survives value weighting in rare (≈1.00 / 0.00) but not common (0.000 / 0.170 / 0.159). The FAIL holds for every cutoff 0.70–1.0. |
| P-a | PASS | F(l) rises with l (ρ +0.87–0.96). The rise survives every fixed-window variant, and is partly confounded with the shrinking window. |
| P-b | FAIL, fragile | Four-head transfer at τ = 16 is 0.180 [0.134, 0.228] in common and 0.359 in rare. Dropping Cairo or Beijing passes it; the pair-level bootstrap is [−0.06, 0.38]. |

## Findings, as worded after verification

1. **The match enters T's stream at layer 1, as a "matching country attended" signal.**
   - From layer 1, x(T) carries a consistent, match-gated own-vs-wrong direction shared by all 11 pairs.
   - At l = 1, 37–46% of ‖mean δ‖² comes from attention, and 54–63% from MLP0 (roughly orthogonal to the attention
     part).
   - The attention part is L0H2 (25–37%), L0H9 and L0H10. It is a shared BOS→country write that says nothing about
     which country: the country-identity part is about 0% of the mean.
   - The write is gated by the match. L0H2 gives τ 0.34–0.43 of its attention with its own capital, 0.03–0.04 with a
     wrong capital, and 0.02–0.03 for the filler control. L0H9 does the reverse. A mismatched country looks like
     filler at T.
   - A placebo (wrong vs a second derangement) gives S = 0.06–0.28; own vs wrong2 replicates (0.77–0.84).
2. **It persists through the stream, diluted at the end.**
   - Own/wrong pairs stay separable at every layer: a paired held-out-capital classifier scores ≥ 0.998.
   - Mapper shows two clean strands from right after layer 0 to about layer 6–8, then many small pure nodes; an own
     node and a wrong node never connect.
   - At x_12 the shared direction is about 10% of δ's energy. It is diluted by a large shared block-11 write (raw δ
     grows 1.35–1.57× while ‖x‖ grows 2.4–2.6×), not erased. Late separation is local, not two clusters.
3. **Causally, the effect is built into T's stream mostly in layers 3–7** (patching, P).
   - F(l), the fraction of the late-BOS effect carried by T's stream, is 0.12–0.37 at l = 1–3. It steps up from layer
     3 to 4 and reaches 0.86–0.94 by l = 7.
   - For l ≥ 8, T's stream reproduces the late BOS effect within ±6% on layers l–11. Per layer, context keys still
     contribute 2–17% at the patched layer, offset by overshoot later; at layers 6–7 they contribute about 25–70%.
   - The four same-country heads' writes carry 18% (common) / 36% (rare) at d = 16, fragile across pairs, and more at
     d = 31. L3H0 supplies about 60% of the increase.
4. **The no-op, measured in writes: BOS attention adds a constant** (R2).
   - Write variety falls with BOS share and rises when the capital pulls attention off BOS. Reweighting accounts for
     both effects.
   - The BOS term is effectively a fixed vector, about 2% of write variance, so variety scales with non-BOS attention.
   - Per unit of non-BOS attention, resting heads write as much variety as active heads or more. Correcting for
     dilution moves R2-a to +0.13…+0.36 and shrinks R2-b to |ρ| ≤ 0.12.
   - So R2 supports the no-op only in its mechanical form: attention spent on BOS contributes an input-independent
     constant. It is not evidence that resting heads read less informative content.
5. **Distance: the shift at T is mostly distance-independent, and distance dependence does not fall with depth** (R3).
   - The pre-registered D rises with depth, much of it from the adjacent (d = 1) spike.
   - Excluding d = 1, the rise holds in rare (ρ ≈ +0.70). In common it is weak and non-monotone (+0.19, peaking at
     layers 6–7). About 70% of the common difference is the d = 1 term.
   - A further slow decline from d 2 to 16 (about 25%) and an extra effect at d = 31 (the first token after BOS) make
     "flat floor" an overstatement.
   - With a random final token, D falls with depth in every variant.
6. **Value weighting turns the country into a hub** (R5).
   - The layer-0 matching loop survives in rare but not common. At τ 16 and 8 a shortcut edge (e.g. 30→16) closes it
     too soon. At τ 1 the filling triangle is {32, 29, 28}.
   - "Lost" at τ 8 and τ 1 depends on the 0.75 cutoff, which was fixed a priori, not calibrated. At 0.80, own is
     0.772 / 0.761. No cutoff passes all three cells.
   - Own-only cells go from 4 to 1 (common) and 6 to 4 (rare); a new own-only loop appears at L4 τ 28 in both fillers.
     "Shifts toward layer 4" is overstated.
   - BOS edges collapsing under value weighting explain at least 73% of the edge boost from layer 5 on, and 35–50% in
     layers 2–4.

## Process notes

- **R1:** the per-δ null in the report is anti-conservative; the verifier's exact cell-level null replaces it. The
  `--posthoc` mode was added after the main run; the main path is unchanged.
- **R2:** final run `20261025-173349_exp5R2`. It supersedes `172046`, which has the same pre-registered numbers. The
  dilution analysis is post hoc and labelled.
- **R3:** the agent handed back before the data existed and was resumed. The addendum was unchanged.
- **R4:** the post-hoc checks are in `posthoc_checks.py`, and the report has a labelled hand-written summary.
- **R5:**
  - folders `173403`, `181131` and `181253` (final) have identical arrays;
  - the addendum's "Ties" paragraph was added after the extraction finished but before any full-data read, and its
    fallback fired 0 times;
  - "calibrated" in the report should read "fixed a priori".
- **P:** the addendum's descriptive F_rev sentence was fixed after smoke test 1, before the full run. The smoke runs'
  arrays equal the full run's.
- **Machine load:** several analyses ran 50–90 minutes because all of them ran at once.
- **Data:** the run folders hold about 2 GB of arrays, untracked.

## Files

- **Code:** `exp5/extract.py`, `r1_match.py`, `r2_noop.py`, `r3_distance.py`, `r4_flow.py`, `r5_routes.py`,
  `patch.py`. Each script's run folder holds its report.
- **Runs:**
  - `results/20261025-170021_exp5res` (extraction)
  - `results/20261025-172035_exp5R1`
  - `results/20261025-173349_exp5R2`
  - `results/20261025-172045_exp5R3`
  - `results/20261025-172418_exp5R4`
  - `results/20261025-181253_exp5R5`
  - `results/20261025-174409_exp5P`
- **Verifier scripts:** `/private/tmp/claude-501/-Users-pabloleyva/cf17c3a1-b0ec-440e-916a-f5ea33bfe8a3/scratchpad/verify_exp5/`

## Open next steps

- Cities (Lyon) and demonyms at τ, to tell the "capital-of" relation from same-country relatedness in R1's signal.
- Split MLP0's share of the layer-1 signal (54–63%) into neurons.
- A cell-level (pair-level) null as the default for every future paired test.
