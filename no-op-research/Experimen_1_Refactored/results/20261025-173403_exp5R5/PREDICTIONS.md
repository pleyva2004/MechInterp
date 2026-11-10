# Pre-registered predictions: Experiment 5, topology of the residual stream

Written 2026-10-25, before any data from this experiment. It follows `onepager/residual_topology_notes.pdf`: where
Exp 4 / 4b measured attention (where a head at the final token T looks), Exp 5 measures what moves and what is
carried:
- each head's write h_{l,i}(T) = z_{l,i}(T) W_O^{l,i};
- the stream state x_l(T) at T before each layer.

Extraction: `exp5/extract.py`, `configs/exp5_residual.yaml`. Analyses R1–R5 and P each get their own script and an
operational addendum `exp5/PREDICTIONS_<id>.md`, written before that analysis looks at the full data. An addendum may
make operational choices but must not change the hypotheses or pass rules below.

## Design shared by all analyses

- **Model and stimuli:** GPT-2 small (TransformerLens with default weight processing as the primary backend; a Hugging
  Face subset as the cross-check). The first 100 of the 200 cached rows per filler type (`exp3rel_{common,rare}`).
  The 11 country–capital pairs of Exp 4. Positions: BOS 0, tokens 1..32, T = 32.
- **Capital pairing (changed from Exp 4b).** Cell k = capital k at T in both conditions:
  - **own:** country k at τ (France … Paris);
  - **wrong:** the country whose deranged capital is k (China … Paris), using the Exp 4b derangement, seed 5;
  - **control:** the row's own token at τ and capital k at T. The same sequence serves both own and wrong.

  The two conditions have the same token at T and differ only at τ.

  This pairing is what makes the residual analyses non-trivial:
  - x_0(T), the capital's embedding plus position, is identical for own and wrong. This is a built-in null.
  - Patching T's stream from wrong into own is not identical to the whole effect by construction, because positions
    τ..T−1 still differ.
- **Random final token:** country c at τ, with the row's own random final token, and the base row as control. This is
  the Exp 4 condition.
- **τ sets:**
  - per-sequence point clouds use τ ∈ {31, 16, 1} (distance d = 1, 16, 31);
  - per-row means (mean over the 11 cells) use all τ = 1..31;
  - attention graphs use τ ∈ {31, 30, 28, 24, 16, 8, 1}, the Exp 4 TDA set.
- **Stored at T:**
  - x_l(T) for l = 0..11 (`hook_resid_pre`) and x_12 (`blocks.11.hook_resid_post`);
  - z_{l,i}(T) (`hook_z`; writes are rebuilt with the saved processed W_O);
  - final-query attention to BOS, τ and T−1.

  Graphs are attention summed over heads with each weight multiplied by ‖v_k W_O^h‖ (value-weighted), plus raw
  attention (mean over heads).
- **Normalisation before any distance:** per layer, centre and scale each state as LayerNorm does, without its gain
  (TransformerLens's processed stream is already mean-centred). Reduce dimension with PCA to 50 (check 20 / 100).
  Bootstrap over rows.
- **Global caveat:** the same 11 pairs throughout, so a result beyond a null means consistent, not general.

## Hypotheses

Each must hold in both filler types unless stated otherwise.

| id | prediction | why |
|---|---|---|
| R1-a | **Built-in null.** The own and wrong states at T are identical at l = 0 (max \|Δ\| < 1e-5 after normalisation). | Same token and position. |
| R1-b | **The stream knows the match after layer 0.** Let δ_l = x_l(T, own) − x_l(T, wrong) per (row, capital). From l = 1 to 12, δ_l has a consistent direction beyond a sign-flip null: ‖mean δ‖ / RMS(δ) above the null's 95th percentile, BH over 12 layers × 3 τ, at every layer 1–12 and every τ ∈ {31, 16, 1}. | L0H2 attends from the capital to its own country at any distance in layer 0. |
| R1-c | **Topological separation.** On the symmetrised cloud {δ_l} ∪ {−δ_l}, Rips H0 has a persistent 2-component split (the gap between the second and third largest death times beats the sign-flip null) at every layer from the first layer where R1-b holds. A linear probe is reported as the non-topological baseline. | If the match is a direction, the ± clouds separate. |
| R2-a | **Resting heads write little variety (the no-op test).** Across the 144 heads, under the random final token, the Spearman correlation between a head's BOS share and the spread of its writes (RMS distance of writes from their mean, divided by the RMS norm of x_l(T)) is negative, with a 95% row-bootstrap CI below 0. | The no-op hypothesis: attending to BOS makes the write nearly constant. |
| R2-b | **Waking up adds variety.** For layers 6–11, the per-head change in BOS share (own − wrong) and the change in write spread (own − wrong) correlate negatively (Spearman, CI below 0). | Exp 4b: the matching capital lowers late resting. |
| R2-c | Descriptive: the median write spread of heads with BOS share ≥ 0.8 vs ≤ 0.3. | Effect size. |
| R3-a | **Distance matters less with depth (the advisor's hypothesis, on content).** Let m_l(τ) be the mean paired shift at T (country at τ minus control, per-layer normalised, own capital). Distance dependence D_l = ‖mean m_l(d 1–4) − mean m_l(d 16–31)‖ / mean ‖m_l(τ)‖ falls with depth: Spearman(l, D_l) < 0 over l = 1..12, with a row-bootstrap CI below 0. | Attention gave only a weak depth order; the content may differ. |
| R3-b | **Late vs early.** Mean D over l = 9–12 < mean D over l = 1–4 (row-bootstrap CI of the difference below 0). | Same. |
| R4-a | **Own and wrong split and stay split.** Mapper is run on the capital-centred per-sequence states at T (each state minus the mean of its (row, capital) own/wrong pair, so token identity is removed), with layer as the lens and average linkage. There is a layer l* from which every node holding ≥ 5 points is condition-pure (own share ≥ 0.8 or ≤ 0.2) through l = 12, for τ = 16 and τ = 1. | If the match is carried, it should not be overwritten. |
| R4-b | l* ≤ 4. | The early same-country heads write it. |
| R5-a | **The layer-0 loop survives value weighting.** Using exp4/tda4_graph's filtration and persistence rules on row-normalised value-weighted graphs (BOS removed), the matching-pair cycle seen in layer 0 at τ ∈ {16, 8, 1} with raw attention is present with its capital in ≥ 80% of row bootstraps and in ≤ 20% with a wrong capital, at all three τ. | L0H2's attention is large; its write may be too. |
| R5-b | Exploratory: whether the layers 1, 3 and 4 loops that also form with a wrong capital survive value weighting. | — |
| P-a | **How much of the effect T's stream carries (activation patching).** Patch x_l(T) from the wrong run into the own run at layer l (same row and capital), for l = 1..11, at τ ∈ {16, 1}. Measure late BOS mass at T over layers max(l, 6)..11. The fraction transferred F(l) = (patched − own) / (wrong − own), on the same layers, rises with l: Spearman > 0 with a row-bootstrap CI above 0. | Later states have read more of the context. |
| P-b | **The same-country heads' writes carry part of it.** Patching only the writes at T of L0H2, L1H11, L3H0 and L4H10 (`hook_result`) from wrong into own transfers ≥ 20% of the late BOS fact (own − wrong; layers 6–11) at τ = 16. | They attend from the capital to its own country. Uncertain. |

## Checks (not predictions)

- **Design:** each variant equals its control except at τ, with no repeats or BOS.
- **Attention:** rows sum to 1.
- **Determinism:** recomputing 2 rows gives identical numbers.
- **Built-in null:** the own and wrong x_0(T) are identical.
- **Hugging Face cross-check on 2 rows:**
  - x_l(T) vs HF `hidden_states[l]` minus its mean, for l ≤ 11;
  - x_12 after `ln_final` vs HF `hidden_states[12]`;
  - per-head writes vs the HF `c_proj` input split by head times its weight slice, mean-centred;
  - tolerance 1e-3 relative.
