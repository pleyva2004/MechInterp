# Operational addendum for analysis P (activation patching), Experiment 5

Written 2026-10-25, before `exp5/patch.py` was run on the full data (only a 2–3-row smoke test under the scratchpad
preceded it). It makes operational choices only. The hypotheses P-a and P-b and their pass rules are those of
`exp5/PREDICTIONS.md` and are not changed.

## Stimuli and runs

- Config `configs/exp5_residual.yaml` (the extraction's): GPT-2 small, TransformerLens with default weight processing,
  CPU, float32, `set_determinism(0)`. Rows: the first 100 of the 200 cached rows per filler type, built with
  `exp2.design.cached_base_block` exactly as `exp5/extract.py`'s main does. Cells from `exp5.extract.cells` (11 cells;
  wrong country = the one whose deranged capital, seed 5, is capital k). τ ∈ {16, 1} (d = 16, 31).
- own sequence = `variants(rows, own countries, τ, capitals)`, wrong sequence = `variants(rows, wrong countries, τ,
  capitals)`; both pass `check_variants` (asserted), so they share every token except the one at τ.
- New forward passes (the patch needs live runs; nothing is read from the extraction). For each filler type, τ and
  block of 30 rows (30 × 11 = 330 sequences, the config's batch size; the last block has 10 rows), every run below uses
  the same batch layout (block rows × 11 cells, in the same order), so a run and its patched version differ only by the
  patch:
  1. **own** and **wrong**, clean: cache `blocks.{l}.hook_resid_pre` at T for l = 0..11 and `blocks.{l}.attn.hook_z`
     at T.
  2. **P-a forward:** own run with `blocks.{l}.hook_resid_pre[:, T]` replaced by the wrong run's value (same row,
     cell, τ), l = 0..11 (l = 0 is the built-in null: x_0(T) is the same token and position in both runs).
  3. **P-a reverse** (descriptive): wrong run with own's x_l(T), l = 1..11.
  4. **P-b forward:** own run with `blocks.{l}.attn.hook_z[:, T, h]` replaced by the wrong run's value for
     (l, h) ∈ {L0H2, L1H11, L3H0, L4H10} together, and for each head alone. A head's write at T is z_{l,h}(T) W_O^{l,h}
     (TransformerLens's `hook_result` for that head, which also has no bias: b_O is added after the sum over heads),
     so patching z at T for a head is the same as patching its `hook_result` at T. Writes at positions < T are not
     patched (as pre-registered: "the writes at T").
  5. **P-b reverse** (descriptive): wrong run with own's z at T, the four heads together and each alone.
- Measured in every run: the attention from T (query position 32) to BOS (key 0) of every layer and head, from
  `blocks.{l}.attn.hook_pattern`. The full attention row at T is checked to sum to 1 (atol 1e-4) in every run.

## Quantities

- Per sequence and layer m: b_m = mean over the 12 heads of the attention from T to BOS. Late BOS over a layer set S
  = mean of b_m over m ∈ S. Every per-row quantity below is first averaged over the 11 cells of the row.
- **P-a:** S(l) = {max(l, 6), …, 11}. Numerator N_r(l) = mean over cells of [late(patched_l) − late(own)] on S(l);
  denominator D_r(l) = mean over cells of [late(wrong) − late(own)] on S(l).
  **F(l) = mean_r N_r(l) / mean_r D_r(l)** — a ratio of means over the 100 rows, not a mean of per-row ratios. Per-row
  or per-cell denominators can be near 0 (and change sign), which would make a mean of ratios unstable; the ratio of
  means weights each row by its share of the effect. The same holds for every fraction below.
- **P-a statistic:** Spearman ρ between l = 1..11 and F(l). Row bootstrap: 5000 resamples of the 100 rows with
  replacement (seed 0; the same row indices for every l within a resample), recomputing all 11 F(l) as ratios of the
  resampled means and then ρ. 95% percentile CIs for ρ and for each F(l). **Pass (as pre-registered):** ρ > 0 and the
  CI's lower bound > 0, in both filler types and at both τ (4 of 4).
- **P-b:** S = {6, …, 11}. Fraction transferred G = mean_r [late(patched) − late(own)] / mean_r [late(wrong) −
  late(own)], with the four heads patched together. **Pass (as pre-registered):** G ≥ 0.20 at τ = 16 on the point
  estimate, in both filler types ("each must hold in both filler types unless stated otherwise"); the row-bootstrap CI
  (5000, seed 0) is reported. τ = 1, each head alone, and the reverse direction are descriptive.
- **Denominator reporting:** for each F(l) and G the denominator mean_r D_r and its row-bootstrap CI are reported. If
  that CI includes 0 for some l (possible for S = {11} or {10, 11}), F(l) at that l is flagged as unreliable in the
  report; the pass rule is still computed exactly as above, with no l dropped.
- **Reverse:** F_rev(l) = mean_r [late(rev_l) − late(wrong)] / mean_r [late(own) − late(wrong)] on S(l); same for the
  heads. If the outcome were additive in the patched site and the rest (own = s_o + c_o, wrong = s_w + c_w), both
  fractions would equal Δs / (Δs + Δc), so F ≈ F_rev indicates additivity; F − F_rev is reported. (Corrected after
  the 3-row smoke test and before the full run: the first version of this line said "F + F_rev ≈ 1", an algebra
  error in a descriptive statement; no hypothesis or pass rule is affected.)
- **Per-layer curves (descriptive):** for each measured layer m = 6..11 and l = 1..11, F_m(l) = mean_r [b_m(patched_l)
  − b_m(own)] / mean_r [b_m(wrong) − b_m(own)]. For l > m it must be exactly 0 (see checks).

## Checks

- **Design:** `check_variants` on every own and wrong block (asserted); the final token is capital k in both.
- **Attention rows sum to 1** at T in every run (asserted, atol 1e-4).
- **Built-in null:** own and wrong x_0(T) identical (max |Δ| = 0), and the l = 0 patch leaves every BOS attention at T
  equal to the own run's (max |Δ| = 0).
- **Identity patch:** on the first block of each filler type × τ, patching the own run with its own cached x_l(T)
  (each l = 1..11) and its own z at T (the four heads) changes nothing: max |Δ| = 0 over every layer and head.
- **Causal locality:** patching x_l(T) or a head in layer l leaves the BOS attention of layers < l exactly unchanged
  (max |Δ| = 0 over all patched runs).
- **Determinism:** rows 0–1 (22 sequences, a separate smaller batch) are recomputed at τ = 16 for own, wrong, the
  P-a l = 6 patch and the four-head patch, per filler type. PASS if max |Δ| = 0. CPU matrix products can round
  differently with a different batch size; if max |Δ| is non-zero but ≤ 1e-6 it is reported as such and counted as
  PASS (batch-layout rounding), and anything larger is a FAIL.
- **Unpatched effect:** late BOS (6–11), own − wrong, row-bootstrap CI. Expected: CI below 0 (own < wrong), and
  magnitude comparable to Exp 4b at the same distance — operationally, within a factor of 2 of Exp 4b's late BOS fact
  (d = 16: −0.0291 common / −0.0257 rare; d = 31: −0.0403 / −0.0332). Exp 4b paired by country (its capital vs a wrong
  capital); here the pairs share the capital, so equality is not expected.

## Outputs

`results/<timestamp>_exp5P/`: `p_report.md`, `p_results.json` (all numbers in the report), `p_arrays.npz` (per row,
cell and run, the head-mean BOS attention of each layer, float32; per row the per-head own and wrong BOS attention),
`figures/` (F(l) with CIs per τ and filler, forward and reverse; per-layer F_m(l); per-head transfer bars), copies of
the config, `PREDICTIONS.md` and this addendum, `versions.json`, `DONE`.

## Caveats fixed in advance

- The same 11 pairs throughout, so a CI over rows means consistent over contexts, not general over pairs.
- Patching T's stream leaves positions τ..T−1 different, so F(l) < 1 is expected even at late l; F(l) measures what
  the stream at T carries relative to what later layers can still read from those positions.
- S(l) shrinks for l > 6, so F(l) at different l is measured on different layer sets (as pre-registered).
