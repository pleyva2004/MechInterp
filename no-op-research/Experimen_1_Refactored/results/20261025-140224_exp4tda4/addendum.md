# TDA analysis 4: verification and addendum

Verifies the saved run `20261025-113614_exp4tda4` (pre-registration `exp4/PREDICTIONS_tda4.md`, code `exp4/tda4_graph.py`). Written by `exp4/tda4_finalize.py` into this new folder (`20261025-140224_exp4tda4`); the run folder is unchanged. The original report is `tda4_report.md` in the run folder; this file adds the independent checks, a plain-language reading, and the pre-registered 'reported, not predicted' tables that the report left out.

## 1. Completeness and provenance

- Coverage: fillers common, rare; conditions control, own, wrong, random; τ ∈ {31, 30, 28, 24, 16, 8, 1}; 22 units (12 layer means, 8 named heads, 2 pooled); 200 rows per filler (the whole cached block, base_cache_n = 200). Saved graph arrays [7, 22, 33, 33] per filler × condition; swap null [3, 200, 22, 2] [pair, swap, unit, variant]; split-half [100, 22, 2]; 14/14 (filler, τ) cells have TDA arrays. Bootstrap: 100 resamples.
- Runtime: extraction 6.0 min, TDA 3.0 min, total 9.1 min (CPU).
- Code: `tda4_graph.py` last modified 11:35:46, before the run started at 11:36:14. Files in exp1/exp1ext/exp2/exp3/exp4 modified after the start: exp4/tda1_heads.py (not imported by `tda4_graph.py`). The run's recorded code hash equals that of the smoke run started at 11:35:48. The HF re-extraction and the re-derived distances below reproduce the saved arrays, so the run used the final code.
- The run's `PREDICTIONS.md` copy is identical to `exp4/PREDICTIONS_tda4.md`; its `config.yaml` is identical to `configs/exp4_tda4.yaml`.

## 2. Checks

Run's own checks (`checks.json`):

| check | result |
|---|---|
| design (ids at τ and 32, row elsewhere, no repeats/BOS/stimulus ids; control = row + capital) | PASS (asserted per τ and condition) |
| wrong-capital derangement has no fixed point; each capital once per row | PASS |
| attention rows sum to 1 (every query, every head) | PASS (max abs(sum − 1) 4.6e-07, atol 0.0001, asserted per batch) |
| causal: upper triangle of every mean graph is 0 | PASS (max 0.0e+00) |
| prefix identity: wrong / random query rows 0..31 equal own's | PASS (max abs Δ 0.0e+00) |
| determinism: first batch (50 rows, common, own, τ 31) recomputed | PASS (max abs Δ 0.0e+00) |
| query-32 weights on τ / BOS / 31 match Exp 4b (own, wrong, control) and Exp 4 (random) | PASS (max abs Δ 2.4e-04, atol 0.002) |
| gudhi vs ripser diagrams (bottleneck, every full-sample graph) | PASS (max 3.0e-08) |

Independent checks here (HuggingFace GPT-2, eager attention, sequences built in this file; own and wrong at τ ∈ {31, 24, 16, 8, 1}, 200 rows per filler):

| check | result |
|---|---|
| design (country at τ, capital at 32, row elsewhere, no repeats, no BOS after 0, no stimulus ids in rows) | PASS (asserted) |
| wrong capitals | France→Tokyo, Germany→Ottawa, Italy→Moscow, Spain→Dublin, Russia→Madrid, Ireland→Cairo, Ukraine→Berlin, Japan→Rome, China→Paris, Egypt→Kiev, Canada→Beijing |
| attention rows sum to 1 | max abs(sum − 1) 4.8e-07 |
| causal (upper triangle) | max 0.0e+00 |
| prefix identity, own vs wrong query rows 0..31 | max abs Δ 0.0e+00 |
| HF per-filler mean graphs vs saved `graphs.npz` (TransformerLens), layer means | max abs Δ 3.8e-07 |
| every saved distance D (3 pairs × 22 units × 2 variants × 14 cells) re-derived from `graphs.npz` with MST + ripser + assignment W1 | max abs Δ 3.2e-05, max relative Δ 3.0e-04 (the run's gudhi.hera W1 uses its default delta = 0.01, i.e. it is exact only to within 1%; ripser works in float32) |
| every saved transplant distance re-derived | max abs Δ 3.2e-06 |
| E1 role, all 2464 full-sample graphs | 0 disagreements |
| persistent H1 count, all full-sample graphs | 0 disagreements |
| run's MPC between strict (E1 on every shortest representative) and lenient (on some) | 0 violations; 62/62 run MPCs hold under the strict reading |
| H1 birth edges matched by value | 0 ambiguous |

## 3. Headline statistic reproduced: own vs wrong D at τ = 31 (noBOS, layer means)

D(run) is the saved value; D(here) is recomputed on the HF graphs. The swap null here uses 200 new paired swaps (seed 20261025), and the split-half uses 100 new splits. The split-half is descriptive in the pre-registration. It compares two 100-row halves that differ in all 528 edges, so it is not the null for own vs wrong.

| filler | layer | D run | D here | null95 run | null95 here | p run | p here | split-half median run | split-half median here | rows with own−wrong E1 of the mean's sign |
|---|---|---|---|---|---|---|---|---|---|---|
| common | L0 | 0.0495 | 0.0495 | 0.0065 | 0.0071 | 0.005 | 0.005 | 0.0491 | 0.0491 | 1.00 |
| common | L1 | 0.0039 | 0.0039 | 0.0006 | 0.0006 | 0.005 | 0.005 | 0.0479 | 0.0478 | 0.99 |
| common | L2 | 0.0473 | 0.0473 | 0.0063 | 0.0071 | 0.005 | 0.005 | 0.0788 | 0.0742 | 1.00 |
| common | L3 | 0.0402 | 0.0402 | 0.0052 | 0.0057 | 0.005 | 0.005 | 0.0815 | 0.0813 | 1.00 |
| common | L4 | 0.0200 | 0.0200 | 0.0027 | 0.0028 | 0.005 | 0.005 | 0.0466 | 0.0462 | 0.99 |
| common | L5 | 0.0028 | 0.0028 | 0.0008 | 0.0009 | 0.005 | 0.005 | 0.0407 | 0.0385 | 0.67 |
| common | L6 | 0.0096 | 0.0096 | 0.0016 | 0.0017 | 0.005 | 0.005 | 0.0392 | 0.0419 | 0.94 |
| common | L7 | 0.0048 | 0.0048 | 0.0008 | 0.0009 | 0.005 | 0.005 | 0.0329 | 0.0323 | 0.93 |
| common | L8 | 0.0042 | 0.0042 | 0.0007 | 0.0009 | 0.005 | 0.005 | 0.0315 | 0.0310 | 0.86 |
| common | L9 | 0.0136 | 0.0136 | 0.0017 | 0.0021 | 0.005 | 0.005 | 0.0294 | 0.0280 | 1.00 |
| common | L10 | 0.0209 | 0.0209 | 0.0030 | 0.0033 | 0.005 | 0.005 | 0.0273 | 0.0275 | 1.00 |
| common | L11 | 0.0028 | 0.0028 | 0.0008 | 0.0007 | 0.005 | 0.005 | 0.0477 | 0.0465 | 0.59 |
| rare | L0 | 0.0480 | 0.0480 | 0.0072 | 0.0067 | 0.005 | 0.005 | 0.0378 | 0.0379 | 1.00 |
| rare | L1 | 0.0027 | 0.0027 | 0.0004 | 0.0004 | 0.005 | 0.005 | 0.0374 | 0.0382 | 0.97 |
| rare | L2 | 0.0432 | 0.0432 | 0.0072 | 0.0065 | 0.005 | 0.005 | 0.0694 | 0.0698 | 1.00 |
| rare | L3 | 0.0369 | 0.0369 | 0.0056 | 0.0049 | 0.005 | 0.005 | 0.0588 | 0.0607 | 1.00 |
| rare | L4 | 0.0056 | 0.0056 | 0.0020 | 0.0019 | 0.005 | 0.005 | 0.0434 | 0.0459 | 0.75 |
| rare | L5 | 0.0086 | 0.0086 | 0.0016 | 0.0013 | 0.005 | 0.005 | 0.0426 | 0.0444 | 0.89 |
| rare | L6 | 0.0037 | 0.0037 | 0.0010 | 0.0009 | 0.005 | 0.005 | 0.0378 | 0.0382 | 0.77 |
| rare | L7 | 0.0085 | 0.0085 | 0.0013 | 0.0012 | 0.005 | 0.005 | 0.0349 | 0.0334 | 0.99 |
| rare | L8 | 0.0011 | 0.0011 | 0.0006 | 0.0005 | 0.005 | 0.005 | 0.0303 | 0.0306 | 0.62 |
| rare | L9 | 0.0086 | 0.0086 | 0.0015 | 0.0013 | 0.005 | 0.005 | 0.0299 | 0.0295 | 0.99 |
| rare | L10 | 0.0095 | 0.0095 | 0.0017 | 0.0014 | 0.005 | 0.005 | 0.0389 | 0.0397 | 0.98 |
| rare | L11 | 0.0008 | 0.0008 | 0.0004 | 0.0004 | 0.005 | 0.005 | 0.0639 | 0.0639 | 0.60 |

## 4. Pre-registered verdicts (primary variant noBOS)

| id | rule (short) | run | re-derived here | numbers |
|---|---|---|---|---|
| G1 | τ 31, own-vs-wrong D beyond the swap null (p ≤ 0.05) in each of L0–L4, both fillers | PASS | PASS | all 10 cells p = 0.005 in the run (observed above all 200 swaps); here max p 0.005; D / null95 from 2.8 (rare L4) to 7.7 |
| G2 | some layer-mean cell where, in both fillers, own has a stable MPC (full sample, bootstrap ≥ 0.8) and wrong has none (bootstrap ≤ 0.2) | PASS | PASS | run: 4 cells (L4 τ 24, L0 τ 16, L0 τ 8, L0 τ 1; own bootstrap 1.00, wrong 0.00 in both fillers); reverse 0. Here strict: L4 τ 24, L0 τ 16, L0 τ 8, L0 τ 1; lenient: L4 τ 24, L0 τ 16, L0 τ 8, L0 τ 1; reverse 0. τ 30 and 28 have no own MPC in any layer mean: True |
| G3a | τ ∈ {16, 8, 1}, L0–L4: every cell beyond the null falls to ≤ null95 after transplanting own's A[32, τ] to wrong's value | FAIL | FAIL | run 10/30 fall within the null (common τ16 L0, common τ8 L0, common τ1 L0, common τ1 L1, rare τ16 L0, rare τ8 L0, rare τ8 L1, rare τ1 L0, rare τ1 L1, rare τ1 L4); here 10/30 |
| G3b | τ ∈ {16, 8, 1}, L6–L11: ≥ half of the cells beyond the null stay above null95 after the transplant | PASS | PASS | run 34/36; here 34/36 |
| B1 | withBOS: no MPC in any layer 1–11, any τ, either filler | PASS | PASS | run none (all four conditions); here, lenient reading, 0 |

Transplant cells here (filler, τ, layer: D → D after transplant, null95):

- early (G3a): common τ16 L0: 0.0264 → 0.0005 (0.0052); common τ16 L1: 0.0037 → 0.0011 (0.0006); common τ16 L2: 0.0033 → 0.0033 (0.0007); common τ16 L3: 0.0141 → 0.0094 (0.0024); common τ16 L4: 0.0240 → 0.0094 (0.0045); common τ8 L0: 0.0252 → 0.0005 (0.0049); common τ8 L1: 0.0044 → 0.0007 (0.0007); common τ8 L2: 0.0030 → 0.0030 (0.0005); common τ8 L3: 0.0194 → 0.0084 (0.0030); common τ8 L4: 0.0286 → 0.0094 (0.0043); common τ1 L0: 0.0296 → 0.0005 (0.0045); common τ1 L1: 0.0050 → 0.0000 (0.0008); common τ1 L2: 0.0024 → 0.0024 (0.0005); common τ1 L3: 0.0176 → 0.0082 (0.0028); common τ1 L4: 0.0288 → 0.0059 (0.0048); rare τ16 L0: 0.0187 → 0.0006 (0.0044); rare τ16 L1: 0.0013 → 0.0005 (0.0003); rare τ16 L2: 0.0049 → 0.0049 (0.0007); rare τ16 L3: 0.0196 → 0.0125 (0.0040); rare τ16 L4: 0.0167 → 0.0049 (0.0028); rare τ8 L0: 0.0172 → 0.0006 (0.0045); rare τ8 L1: 0.0005 → 0.0003 (0.0003); rare τ8 L2: 0.0045 → 0.0045 (0.0007); rare τ8 L3: 0.0286 → 0.0119 (0.0054); rare τ8 L4: 0.0187 → 0.0052 (0.0029); rare τ1 L0: 0.0188 → 0.0005 (0.0043); rare τ1 L1: 0.0004 → 0.0002 (0.0003); rare τ1 L2: 0.0041 → 0.0041 (0.0006); rare τ1 L3: 0.0302 → 0.0122 (0.0050); rare τ1 L4: 0.0206 → 0.0027 (0.0031)
- late (G3b): common τ16 L6: 0.0096 → 0.0096 (0.0026); common τ16 L7: 0.0218 → 0.0218 (0.0041); common τ16 L8: 0.0080 → 0.0080 (0.0014); common τ16 L9: 0.0090 → 0.0110 (0.0023); common τ16 L10: 0.0153 → 0.0153 (0.0029); common τ16 L11: 0.0050 → 0.0000 (0.0010); common τ8 L6: 0.0091 → 0.0100 (0.0025); common τ8 L7: 0.0252 → 0.0252 (0.0042); common τ8 L8: 0.0089 → 0.0089 (0.0016); common τ8 L9: 0.0083 → 0.0111 (0.0027); common τ8 L10: 0.0191 → 0.0191 (0.0034); common τ8 L11: 0.0018 → 0.0000 (0.0009); common τ1 L6: 0.0065 → 0.0072 (0.0011); common τ1 L7: 0.0327 → 0.0327 (0.0049); common τ1 L8: 0.0081 → 0.0091 (0.0016); common τ1 L9: 0.0125 → 0.0151 (0.0038); common τ1 L10: 0.0309 → 0.0309 (0.0056); common τ1 L11: 0.0103 → 0.0103 (0.0030); rare τ16 L6: 0.0114 → 0.0114 (0.0025); rare τ16 L7: 0.0173 → 0.0173 (0.0029); rare τ16 L8: 0.0089 → 0.0089 (0.0014); rare τ16 L9: 0.0124 → 0.0124 (0.0021); rare τ16 L10: 0.0157 → 0.0157 (0.0028); rare τ16 L11: 0.0091 → 0.0078 (0.0016); rare τ8 L6: 0.0129 → 0.0131 (0.0031); rare τ8 L7: 0.0169 → 0.0169 (0.0031); rare τ8 L8: 0.0090 → 0.0090 (0.0017); rare τ8 L9: 0.0103 → 0.0103 (0.0023); rare τ8 L10: 0.0165 → 0.0165 (0.0034); rare τ8 L11: 0.0114 → 0.0093 (0.0018); rare τ1 L6: 0.0104 → 0.0113 (0.0023); rare τ1 L7: 0.0211 → 0.0211 (0.0039); rare τ1 L8: 0.0091 → 0.0091 (0.0018); rare τ1 L9: 0.0099 → 0.0109 (0.0028); rare τ1 L10: 0.0246 → 0.0246 (0.0042); rare τ1 L11: 0.0129 → 0.0129 (0.0026)

## 5. What the topology shows

- **What can differ.** Own, wrong and random share tokens 0..31, so their graphs are identical except for node 32's 32 edges (prefix identity, max abs Δ 0). Every own-vs-wrong topological difference is about how the capital's node attaches to a fixed network of the other 31 tokens.
- **BOS.** With BOS kept (withBOS), each token's heavy edge to BOS cones off any loop as soon as the loop forms. No matching-pair cycle exists in layers 1–11 in any condition (B1 PASS). So the question is asked on the noBOS graph: node 0 is deleted and the remaining weights are kept raw (no renormalisation). The capital's late drop in BOS attention then shows up only as extra weight on its content edges.
- **Adjacent country (τ = 31): no loop.** E1 = E3 is the capital's previous-token edge. In almost every layer mean it is an H0 merge edge, so node 32 hangs off the network as a leaf. The exception is L2 own, where it is 'instant': the matching capital's weaker previous-token edge lets node 32 attach through another edge first. The topological change is how tightly node 32 attaches: tighter with the matching capital in L0/L1/L4 (the same-country heads; L0 E1 +0.050 common, +0.048 rare) and looser in L2/L3 (the previous-token heads look away; L2 −0.048, L3 −0.040/−0.037). G1 passes on these weight changes at node 32, not on a cycle. The only τ = 31 loop through E1 is in common L10.
- **Distant country, layer 0: a loop that only the matching capital closes.** At τ ≤ 16 (both fillers) and at τ = 24 (rare), the matching capital's heavier edge to its country (L0 E1 own − wrong about +0.03) joins node 32 to the network through the country first (E1 is a merge edge). When the capital's own previous-token edge {31, 32} enters, it closes a long loop: 32 → τ, then a chain of short back edges (mostly 1–5 back) up to 31, then 32. At τ = 1 the hop from position 1 is a long edge. These are 7- to 13-node cycles with persistence 0.010–0.018 and bootstrap frequency ≥ 0.99. With a wrong capital, E1 enters only after 32 and τ are already linked, so it is filled at once or makes a short-lived cycle; bootstrap 0.00. Random and control never have this loop.
- **Layer 4, τ = 24: a short loop closed by E1 itself.** The cycle is 24–26–28–30–32 over layer 4's 2-back chain (persistence 0.017 common, 0.011 rare). Own has it in both fillers and wrong in neither.
- **Layers 1, 3, 4 (and pooled L0–4) at τ ≤ 16: the loop exists with both capitals.** A wrong capital still attends to the country token. In L1 it attends as much as the matching one, or more (E1 own − wrong −0.003 to −0.005 at τ ≤ 16, common), so the loop marks 'a country is present', not the match. In L3/L4 the match makes the loop longer-lived (e.g. common L4 τ 8: persistence 0.030 vs 0.009; L3 τ 8: 0.024 vs 0.013). The single head L3H0 has a loop through E1 at τ 8/1 in own, wrong and also random and/or control, so it is not specific to the capital.
- **Late layers (5–11).** There are no stable matching-pair cycles, except common L10 at τ 31 (one filler only, so not a G2 cell). Own-vs-wrong distances are beyond the null, but transplanting the country edge leaves them unchanged in most far-τ cells (G3b PASS). The late difference sits on node 32's other content edges, where the BOS mass freed by the matching capital goes, not on the country edge.
- **G3a fails because only layer 0 is the country edge alone.** In L0 the transplant removes the whole difference (D 0.017–0.030 → 0.0005, 6/6 cells). In L2 the transplant changes nothing, because E1 is not a critical edge there and the difference is in the capital's other edges. In L3/L4 the transplant removes 33–87% of D (run values) but not all of it.

## 6. Caveats added by this verification

- **The swap null is very tight, so 'beyond the null' means 'consistent', not 'large'.** 295/308 noBOS own-vs-wrong cells (every unit, τ and filler) have p ≤ 0.05. D / null95 has quartiles 5.2 / 6.2 / 6.9, which is close to √200 / 1.96 ≈ 7.2. That is the value expected when the per-row own − wrong difference has the same sign in nearly every row (see the last column of section 3). Each row graph averages the same 11 country–capital pairs, so the effect is nearly deterministic across rows. The null tests exchangeability across rows given these 11 pairs; it says nothing about other countries. G1 passing therefore shows a consistent difference, and G3a/G3b compare against a very small yardstick.
- In absolute terms the own-vs-wrong D is small next to the row-sampling variation of the whole graph: the split-half median is of similar size or larger for most layer means. That comparison is not like for like (split-half differs in all 528 edges, own vs wrong in 32), which is why the pre-registration kept it descriptive.
- Representative cycles: 62/62 of the run's full-sample MPCs keep E1 on every hop-shortest representative. The G2 cells are re-derived here under both the strict and the lenient reading (section 4).
- The G2 cells are bootstrap-stable over rows, but the 11 country–capital pairs are fixed. Cycle lengths and persistences come from mean graphs over 2,200 sequences, not from per-sequence topology.

## 7. Pre-registered 'reported' items missing from the original report

### 7a. Stability bootstrap: E1 role frequencies (noBOS; own / wrong; modal role and its frequency over 100 resamples)

**common**

| unit | τ 31 | τ 30 | τ 28 | τ 24 | τ 16 | τ 8 | τ 1 |
|---|---|---|---|---|---|---|---|
| L0 | M1.00/M1.00 | M1.00/M1.00 | M1.00/I0.97 | M1.00/I1.00 | M1.00/I1.00 | M1.00/I0.77 | M1.00/C1.00 |
| L1 | M1.00/M1.00 | M1.00/M1.00 | M1.00/M1.00 | M1.00/M1.00 | C1.00/C0.86 | C1.00/C1.00 | C1.00/C0.98 |
| L2 | I0.54/M1.00 | I1.00/I1.00 | I1.00/I1.00 | I1.00/I1.00 | I1.00/I1.00 | I1.00/I1.00 | I1.00/I1.00 |
| L3 | M1.00/M1.00 | I1.00/I1.00 | I1.00/I1.00 | I1.00/I1.00 | C1.00/C1.00 | C1.00/C1.00 | C1.00/C1.00 |
| L4 | M1.00/M1.00 | I1.00/I1.00 | C1.00/I1.00 | C1.00/I0.54 | C1.00/C1.00 | C1.00/C1.00 | C1.00/C1.00 |
| L5 | M1.00/M1.00 | I1.00/I1.00 | I1.00/I1.00 | I1.00/I1.00 | I1.00/I1.00 | I1.00/I1.00 | I1.00/I0.99 |
| L6 | M1.00/M1.00 | I1.00/I1.00 | I1.00/I1.00 | I1.00/I1.00 | I1.00/I1.00 | I1.00/I1.00 | I1.00/I1.00 |
| L7 | M1.00/M1.00 | I1.00/I1.00 | I1.00/I1.00 | I1.00/I1.00 | I1.00/I1.00 | I1.00/I1.00 | I1.00/I1.00 |
| L8 | M1.00/M1.00 | I1.00/I1.00 | I1.00/I1.00 | I1.00/I1.00 | I1.00/I1.00 | I1.00/I1.00 | I1.00/I1.00 |
| L9 | M1.00/M1.00 | I1.00/I1.00 | I1.00/I1.00 | I1.00/I1.00 | I1.00/I1.00 | I1.00/I1.00 | I1.00/I1.00 |
| L10 | M1.00/M1.00 | M0.79/M0.81 | I1.00/I1.00 | I1.00/I0.99 | I1.00/I0.97 | I1.00/I0.94 | I1.00/I1.00 |
| L11 | M1.00/M1.00 | M1.00/M1.00 | M0.67/M1.00 | M0.86/M1.00 | I0.75/M1.00 | I1.00/M1.00 | I1.00/M1.00 |
| L0-4 | M1.00/M1.00 | I1.00/I1.00 | I1.00/I1.00 | C1.00/I1.00 | C1.00/C1.00 | C1.00/C1.00 | C1.00/C1.00 |

**rare**

| unit | τ 31 | τ 30 | τ 28 | τ 24 | τ 16 | τ 8 | τ 1 |
|---|---|---|---|---|---|---|---|
| L0 | M1.00/M1.00 | M1.00/I1.00 | M1.00/I1.00 | M1.00/I1.00 | M1.00/I1.00 | M1.00/I1.00 | M1.00/I1.00 |
| L1 | M1.00/M1.00 | M1.00/M1.00 | M0.68/C0.82 | C1.00/C1.00 | C1.00/C1.00 | C1.00/C1.00 | C1.00/C1.00 |
| L2 | I0.97/M1.00 | I1.00/I1.00 | I1.00/I1.00 | I1.00/I1.00 | I1.00/I1.00 | I1.00/I1.00 | I1.00/I1.00 |
| L3 | M1.00/M1.00 | I1.00/I1.00 | I1.00/I1.00 | I1.00/I1.00 | C1.00/C0.99 | C1.00/C1.00 | C1.00/C1.00 |
| L4 | M1.00/M1.00 | I1.00/I1.00 | I1.00/I1.00 | C1.00/I0.99 | C1.00/C1.00 | C1.00/C1.00 | C1.00/C1.00 |
| L5 | M1.00/M1.00 | I1.00/I1.00 | I1.00/I1.00 | I1.00/I1.00 | I1.00/I1.00 | I1.00/I1.00 | I1.00/I1.00 |
| L6 | M1.00/M1.00 | I1.00/I1.00 | I1.00/I1.00 | I1.00/I1.00 | I1.00/I1.00 | I1.00/I1.00 | I1.00/I1.00 |
| L7 | M1.00/M1.00 | I1.00/I1.00 | I1.00/I1.00 | I1.00/I1.00 | I1.00/I1.00 | I1.00/I1.00 | I1.00/I1.00 |
| L8 | M1.00/M1.00 | I1.00/I1.00 | I1.00/I1.00 | I1.00/I1.00 | I1.00/I1.00 | I1.00/I1.00 | I1.00/I1.00 |
| L9 | M1.00/M1.00 | I1.00/I1.00 | I1.00/I1.00 | I1.00/I1.00 | I1.00/I1.00 | I1.00/I1.00 | I1.00/I1.00 |
| L10 | M1.00/M1.00 | I0.98/I0.96 | I1.00/I1.00 | I1.00/I1.00 | I1.00/I1.00 | I1.00/I1.00 | I1.00/I1.00 |
| L11 | M1.00/M1.00 | I0.52/M1.00 | I1.00/I1.00 | I1.00/I0.93 | I1.00/I0.99 | I1.00/I0.98 | I1.00/I0.81 |
| L0-4 | M1.00/M1.00 | I1.00/I1.00 | I1.00/I1.00 | C0.99/I1.00 | C1.00/I1.00 | C1.00/C1.00 | C1.00/C1.00 |

### 7b. Role of E1 in the random and control conditions (full sample, noBOS; random / control)

**common**

| unit | τ 31 | τ 30 | τ 28 | τ 24 | τ 16 | τ 8 | τ 1 |
|---|---|---|---|---|---|---|---|
| L0 | M/M | I/I | I/I | I/I | I/I | I/I | I/I |
| L1 | M/M | I/I | I/I | I/I | I/I | I/I | I/I |
| L2 | M/M | I/I | I/I | I/I | I/I | I/I | I/I |
| L3 | M/M | I/I | I/I | I/I | I/I | I/I | I/C |
| L4 | M/M | I/I | I/I | I/I | I/I | I/I | I/I |
| L5 | M/M | I/I | I/I | I/I | I/I | I/I | I/I |
| L6 | M/M | I/I | I/I | I/I | I/I | I/I | I/I |
| L7 | M/M | I/I | I/I | I/I | I/I | I/I | I/I |
| L8 | M/M | M/I | I/I | I/I | I/I | I/I | I/I |
| L9 | M/M | M/I | M/I | M/I | M/I | C/I | C/C |
| L10 | M/M | M/I | M/I | M/I | M/I | C/I | I/C |
| L11 | M/C | M/I | M/I | M/I | M/I | M/I | C/M |
| L0-4 | M/M | I/I | I/I | I/I | I/I | I/I | I/I |

**rare**

| unit | τ 31 | τ 30 | τ 28 | τ 24 | τ 16 | τ 8 | τ 1 |
|---|---|---|---|---|---|---|---|
| L0 | M/M | I/I | I/I | I/I | I/I | I/I | I/I |
| L1 | M/M | I/I | I/I | I/I | I/I | I/I | I/I |
| L2 | M/M | I/I | I/I | I/I | I/I | I/I | I/I |
| L3 | M/M | I/I | I/I | I/I | I/I | I/I | I/C |
| L4 | M/M | I/I | I/I | I/I | I/I | I/I | I/I |
| L5 | M/M | I/I | I/I | I/I | I/I | I/I | I/I |
| L6 | M/M | I/I | I/I | I/I | I/I | I/I | I/I |
| L7 | M/M | I/I | I/I | I/I | I/I | I/I | I/I |
| L8 | M/M | I/I | I/I | I/I | I/I | I/I | I/I |
| L9 | M/M | M/I | I/I | I/I | I/I | I/I | I/I |
| L10 | M/M | M/I | M/I | I/I | I/I | I/I | I/I |
| L11 | M/M | M/I | M/I | M/I | I/I | I/I | I/I |
| L0-4 | M/M | I/I | I/I | I/I | I/I | I/I | I/I |

### 7c. E1 own−wrong with 95% bootstrap CI over rows (layer means L0–L4 and the same-country heads)

**common**

| unit | τ 31 | τ 30 | τ 28 | τ 24 | τ 16 | τ 8 | τ 1 |
|---|---|---|---|---|---|---|---|
| L0 | +0.0495 [+0.0494, +0.0497] | +0.0458 [+0.0457, +0.0459] | +0.0410 [+0.0409, +0.0411] | +0.0364 [+0.0363, +0.0365] | +0.0333 [+0.0332, +0.0334] | +0.0322 [+0.0321, +0.0323] | +0.0320 [+0.0319, +0.0321] |
| L1 | +0.0039 [+0.0036, +0.0041] | +0.0030 [+0.0028, +0.0033] | +0.0018 [+0.0015, +0.0020] | -0.0001 [-0.0004, +0.0001] | -0.0028 [-0.0030, -0.0026] | -0.0040 [-0.0042, -0.0038] | -0.0050 [-0.0051, -0.0048] |
| L2 | -0.0479 [-0.0483, -0.0475] | -0.0309 [-0.0316, -0.0303] | -0.0121 [-0.0125, -0.0117] | -0.0018 [-0.0019, -0.0017] | +0.0003 [+0.0003, +0.0004] | +0.0003 [+0.0003, +0.0003] | -0.0001 [-0.0001, -0.0000] |
| L3 | -0.0401 [-0.0408, -0.0394] | -0.0210 [-0.0218, -0.0201] | -0.0066 [-0.0073, -0.0060] | +0.0002 [-0.0003, +0.0008] | +0.0051 [+0.0044, +0.0057] | +0.0110 [+0.0103, +0.0116] | +0.0094 [+0.0088, +0.0100] |
| L4 | +0.0200 [+0.0189, +0.0213] | +0.0147 [+0.0140, +0.0154] | +0.0160 [+0.0154, +0.0166] | +0.0176 [+0.0169, +0.0182] | +0.0196 [+0.0190, +0.0203] | +0.0219 [+0.0212, +0.0226] | +0.0265 [+0.0261, +0.0270] |
| L0H2 | +0.4012 [+0.4000, +0.4023] | +0.3962 [+0.3950, +0.3972] | +0.3907 [+0.3896, +0.3918] | +0.3806 [+0.3794, +0.3817] | +0.3590 [+0.3579, +0.3601] | +0.3458 [+0.3447, +0.3468] | +0.3278 [+0.3267, +0.3288] |
| L1H11 | +0.2321 [+0.2303, +0.2340] | +0.2188 [+0.2163, +0.2213] | +0.1968 [+0.1940, +0.1994] | +0.1624 [+0.1596, +0.1652] | +0.1119 [+0.1096, +0.1142] | +0.0825 [+0.0809, +0.0842] | +0.0664 [+0.0654, +0.0674] |
| L3H0 | +0.1355 [+0.1324, +0.1387] | +0.1413 [+0.1375, +0.1452] | +0.1497 [+0.1452, +0.1541] | +0.1738 [+0.1682, +0.1794] | +0.2356 [+0.2283, +0.2431] | +0.3285 [+0.3204, +0.3362] | +0.3050 [+0.2996, +0.3100] |
| L4H10 | +0.2774 [+0.2719, +0.2831] | +0.2493 [+0.2430, +0.2557] | +0.2393 [+0.2328, +0.2460] | +0.2390 [+0.2318, +0.2464] | +0.2463 [+0.2391, +0.2533] | +0.2614 [+0.2540, +0.2695] | +0.3021 [+0.2970, +0.3073] |

**rare**

| unit | τ 31 | τ 30 | τ 28 | τ 24 | τ 16 | τ 8 | τ 1 |
|---|---|---|---|---|---|---|---|
| L0 | +0.0480 [+0.0478, +0.0481] | +0.0441 [+0.0439, +0.0443] | +0.0394 [+0.0392, +0.0395] | +0.0346 [+0.0345, +0.0348] | +0.0315 [+0.0313, +0.0316] | +0.0303 [+0.0302, +0.0305] | +0.0301 [+0.0300, +0.0303] |
| L1 | +0.0027 [+0.0025, +0.0029] | +0.0024 [+0.0022, +0.0026] | +0.0026 [+0.0024, +0.0027] | +0.0021 [+0.0020, +0.0023] | +0.0008 [+0.0006, +0.0009] | +0.0002 [+0.0001, +0.0004] | -0.0002 [-0.0004, -0.0001] |
| L2 | -0.0476 [-0.0480, -0.0473] | -0.0282 [-0.0287, -0.0276] | -0.0094 [-0.0096, -0.0091] | -0.0012 [-0.0013, -0.0012] | +0.0003 [+0.0003, +0.0003] | +0.0003 [+0.0003, +0.0003] | -0.0000 [-0.0001, -0.0000] |
| L3 | -0.0369 [-0.0376, -0.0362] | -0.0151 [-0.0161, -0.0142] | +0.0007 [+0.0001, +0.0013] | +0.0077 [+0.0071, +0.0082] | +0.0135 [+0.0130, +0.0142] | +0.0236 [+0.0228, +0.0243] | +0.0219 [+0.0213, +0.0224] |
| L4 | +0.0056 [+0.0037, +0.0075] | +0.0095 [+0.0090, +0.0101] | +0.0118 [+0.0113, +0.0123] | +0.0136 [+0.0131, +0.0141] | +0.0146 [+0.0141, +0.0151] | +0.0150 [+0.0144, +0.0156] | +0.0209 [+0.0205, +0.0214] |
| L0H2 | +0.3750 [+0.3733, +0.3768] | +0.3700 [+0.3682, +0.3718] | +0.3644 [+0.3626, +0.3662] | +0.3540 [+0.3522, +0.3558] | +0.3333 [+0.3315, +0.3351] | +0.3205 [+0.3188, +0.3223] | +0.3037 [+0.3021, +0.3053] |
| L1H11 | +0.1589 [+0.1569, +0.1611] | +0.1477 [+0.1456, +0.1499] | +0.1364 [+0.1343, +0.1385] | +0.1155 [+0.1132, +0.1177] | +0.0840 [+0.0823, +0.0858] | +0.0685 [+0.0671, +0.0699] | +0.0624 [+0.0617, +0.0632] |
| L3H0 | +0.1339 [+0.1314, +0.1363] | +0.1351 [+0.1310, +0.1390] | +0.1480 [+0.1430, +0.1532] | +0.1780 [+0.1716, +0.1844] | +0.2531 [+0.2460, +0.2609] | +0.3763 [+0.3675, +0.3855] | +0.3668 [+0.3613, +0.3723] |
| L4H10 | +0.2563 [+0.2505, +0.2618] | +0.2223 [+0.2158, +0.2286] | +0.2074 [+0.2013, +0.2133] | +0.2021 [+0.1967, +0.2077] | +0.2011 [+0.1948, +0.2072] | +0.1943 [+0.1875, +0.2013] | +0.2490 [+0.2442, +0.2541] |

### 7c. E1 own−control with 95% bootstrap CI over rows (layer means L0–L4 and the same-country heads)

**common**

| unit | τ 31 | τ 30 | τ 28 | τ 24 | τ 16 | τ 8 | τ 1 |
|---|---|---|---|---|---|---|---|
| L0 | +0.0579 [+0.0564, +0.0594] | +0.0526 [+0.0512, +0.0540] | +0.0497 [+0.0487, +0.0506] | +0.0456 [+0.0450, +0.0462] | +0.0416 [+0.0412, +0.0420] | +0.0399 [+0.0395, +0.0402] | +0.0393 [+0.0388, +0.0398] |
| L1 | +0.0576 [+0.0553, +0.0598] | +0.0530 [+0.0508, +0.0549] | +0.0473 [+0.0455, +0.0488] | +0.0411 [+0.0402, +0.0421] | +0.0331 [+0.0323, +0.0338] | +0.0294 [+0.0288, +0.0300] | +0.0301 [+0.0296, +0.0307] |
| L2 | -0.0380 [-0.0440, -0.0319] | -0.0318 [-0.0370, -0.0268] | -0.0090 [-0.0123, -0.0057] | +0.0035 [+0.0024, +0.0045] | +0.0028 [+0.0022, +0.0034] | +0.0018 [+0.0014, +0.0022] | +0.0007 [+0.0003, +0.0011] |
| L3 | -0.0158 [-0.0236, -0.0081] | -0.0098 [-0.0153, -0.0048] | +0.0061 [+0.0032, +0.0087] | +0.0158 [+0.0147, +0.0167] | +0.0212 [+0.0205, +0.0219] | +0.0293 [+0.0285, +0.0300] | +0.0262 [+0.0256, +0.0268] |
| L4 | +0.0204 [+0.0157, +0.0253] | +0.0216 [+0.0183, +0.0248] | +0.0255 [+0.0234, +0.0275] | +0.0301 [+0.0289, +0.0314] | +0.0309 [+0.0298, +0.0319] | +0.0333 [+0.0324, +0.0343] | +0.0367 [+0.0359, +0.0375] |
| L0H2 | +0.4155 [+0.4139, +0.4171] | +0.4100 [+0.4085, +0.4114] | +0.4037 [+0.4021, +0.4051] | +0.3922 [+0.3907, +0.3937] | +0.3701 [+0.3685, +0.3715] | +0.3577 [+0.3562, +0.3591] | +0.3396 [+0.3381, +0.3411] |
| L1H11 | +0.2472 [+0.2452, +0.2493] | +0.2333 [+0.2306, +0.2362] | +0.2104 [+0.2074, +0.2132] | +0.1755 [+0.1724, +0.1785] | +0.1230 [+0.1205, +0.1256] | +0.0931 [+0.0913, +0.0950] | +0.0789 [+0.0778, +0.0801] |
| L3H0 | +0.1380 [+0.1348, +0.1413] | +0.1438 [+0.1399, +0.1479] | +0.1526 [+0.1480, +0.1572] | +0.1779 [+0.1721, +0.1836] | +0.2432 [+0.2356, +0.2512] | +0.3438 [+0.3349, +0.3524] | +0.3095 [+0.3028, +0.3157] |
| L4H10 | +0.3109 [+0.3042, +0.3179] | +0.2778 [+0.2705, +0.2853] | +0.2639 [+0.2563, +0.2714] | +0.2641 [+0.2556, +0.2730] | +0.2724 [+0.2637, +0.2806] | +0.2928 [+0.2839, +0.3019] | +0.3458 [+0.3390, +0.3528] |

**rare**

| unit | τ 31 | τ 30 | τ 28 | τ 24 | τ 16 | τ 8 | τ 1 |
|---|---|---|---|---|---|---|---|
| L0 | +0.0460 [+0.0448, +0.0471] | +0.0439 [+0.0429, +0.0449] | +0.0398 [+0.0388, +0.0406] | +0.0370 [+0.0364, +0.0376] | +0.0332 [+0.0327, +0.0338] | +0.0322 [+0.0317, +0.0326] | +0.0320 [+0.0315, +0.0325] |
| L1 | +0.0262 [+0.0242, +0.0283] | +0.0249 [+0.0234, +0.0263] | +0.0213 [+0.0199, +0.0227] | +0.0209 [+0.0200, +0.0216] | +0.0168 [+0.0160, +0.0176] | +0.0159 [+0.0153, +0.0166] | +0.0170 [+0.0162, +0.0179] |
| L2 | -0.0565 [-0.0619, -0.0512] | -0.0353 [-0.0392, -0.0316] | -0.0095 [-0.0114, -0.0076] | +0.0014 [+0.0007, +0.0022] | +0.0011 [+0.0007, +0.0014] | +0.0002 [-0.0001, +0.0004] | +0.0003 [+0.0001, +0.0005] |
| L3 | -0.0274 [-0.0344, -0.0208] | -0.0076 [-0.0111, -0.0039] | +0.0090 [+0.0075, +0.0105] | +0.0154 [+0.0147, +0.0161] | +0.0211 [+0.0204, +0.0218] | +0.0324 [+0.0316, +0.0332] | +0.0308 [+0.0300, +0.0316] |
| L4 | +0.0091 [+0.0050, +0.0132] | +0.0186 [+0.0163, +0.0209] | +0.0228 [+0.0216, +0.0241] | +0.0243 [+0.0234, +0.0252] | +0.0238 [+0.0229, +0.0246] | +0.0234 [+0.0225, +0.0243] | +0.0291 [+0.0284, +0.0298] |
| L0H2 | +0.3828 [+0.3804, +0.3852] | +0.3774 [+0.3750, +0.3797] | +0.3720 [+0.3695, +0.3744] | +0.3624 [+0.3602, +0.3649] | +0.3394 [+0.3370, +0.3417] | +0.3275 [+0.3252, +0.3297] | +0.3117 [+0.3093, +0.3141] |
| L1H11 | +0.1778 [+0.1755, +0.1803] | +0.1660 [+0.1637, +0.1685] | +0.1532 [+0.1508, +0.1555] | +0.1305 [+0.1280, +0.1331] | +0.0958 [+0.0939, +0.0978] | +0.0796 [+0.0779, +0.0813] | +0.0753 [+0.0742, +0.0764] |
| L3H0 | +0.1365 [+0.1338, +0.1390] | +0.1378 [+0.1337, +0.1419] | +0.1512 [+0.1460, +0.1565] | +0.1824 [+0.1758, +0.1892] | +0.2614 [+0.2538, +0.2695] | +0.3966 [+0.3867, +0.4067] | +0.3843 [+0.3761, +0.3924] |
| L4H10 | +0.3025 [+0.2947, +0.3100] | +0.2625 [+0.2542, +0.2705] | +0.2438 [+0.2362, +0.2511] | +0.2369 [+0.2304, +0.2439] | +0.2348 [+0.2268, +0.2424] | +0.2290 [+0.2205, +0.2375] | +0.3034 [+0.2970, +0.3100] |

### 7d. Units beyond their swap null, per pair (noBOS, count out of 22 units)

| filler | τ | own–wrong | own–random | own–control |
|---|---|---|---|---|
| common | 31 | 22 | 21 | 22 |
| common | 30 | 22 | 21 | 20 |
| common | 28 | 22 | 21 | 21 |
| common | 24 | 20 | 19 | 21 |
| common | 16 | 20 | 20 | 21 |
| common | 8 | 20 | 20 | 22 |
| common | 1 | 20 | 20 | 21 |
| rare | 31 | 22 | 22 | 21 |
| rare | 30 | 22 | 21 | 21 |
| rare | 28 | 21 | 16 | 21 |
| rare | 24 | 21 | 20 | 21 |
| rare | 16 | 21 | 20 | 21 |
| rare | 8 | 21 | 20 | 21 |
| rare | 1 | 21 | 20 | 22 |

### 7e. Persistent H1 classes per graph, wrong condition (layer means; noBOS / withBOS); own is in the report

| filler | τ | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | L11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| common | 31 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| common | 30 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| common | 28 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 1/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| common | 24 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 1/0 | 0/0 | 1/0 | 0/0 | 0/0 | 1/0 | 0/0 |
| common | 16 | 0/0 | 1/0 | 0/0 | 1/0 | 1/0 | 1/0 | 0/0 | 1/0 | 1/0 | 0/0 | 1/0 | 0/0 |
| common | 8 | 0/0 | 1/0 | 0/0 | 1/0 | 1/0 | 1/0 | 0/0 | 1/0 | 1/0 | 0/0 | 0/0 | 0/0 |
| common | 1 | 0/0 | 1/0 | 0/0 | 1/0 | 1/0 | 1/0 | 0/0 | 1/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| rare | 31 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| rare | 30 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| rare | 28 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 1/0 | 0/0 | 1/0 | 1/0 | 0/0 | 0/0 | 0/0 |
| rare | 24 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 1/0 | 0/0 | 1/0 | 0/0 | 1/0 | 0/0 | 0/0 |
| rare | 16 | 0/0 | 0/0 | 0/0 | 0/0 | 1/0 | 1/0 | 0/0 | 1/0 | 1/0 | 1/0 | 1/0 | 0/0 |
| rare | 8 | 0/0 | 0/0 | 0/0 | 1/0 | 1/0 | 1/0 | 0/0 | 1/0 | 1/0 | 0/0 | 1/0 | 0/0 |
| rare | 1 | 0/0 | 1/0 | 0/0 | 1/0 | 1/0 | 1/0 | 0/0 | 1/0 | 0/0 | 0/0 | 1/0 | 0/0 |

## Files

- `verify.json`: provenance, coverage, independent checks, re-derived verdicts and null-tightness summary.
- `verify_arrays.npz`: per (filler, τ ∈ {31, 24, 16, 8, 1}) the HF-based observed D, transplant D, swap null [swap, layer], p, null95, split-half (τ 31 only) and per-row E1 sign consistency, for layer means L0–L11.
- Original run: `results/20261025-113614_exp4tda4/` (`tda4_report.md`, `figures/`, arrays).
