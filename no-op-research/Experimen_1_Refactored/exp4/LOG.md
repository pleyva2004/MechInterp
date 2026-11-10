# Experiment 4 log: controlled difference (country vs original token at τ)

## Run `results/20261024-000548_exp4ctrl` (2026-10-24)

**Setup.** Design and predictions are in `exp4/PREDICTIONS.md`, written before the run; the config is
`configs/exp4_controlled.yaml`. The base rows are the 200 cached rows per filler type that R1-out, R1-in and the sweep
used. Each arm (country, matched word) has 11 fills × 31 positions per row, and the run took 16 minutes on CPU.

**Checks.** All pass:
- design asserts;
- attention rows sum to 1;
- determinism (max |Δ| 0);
- TL vs HF (max |Δ| 1.4e-5);
- L4H11 prev_score 0.986 / 0.985 (common / rare).

**Predictions.**

| id | verdict | short |
|---|---|---|
| C1 | FAIL | Several Exp 1 previous-token heads follow content at d ≥ 2: L2H3, L2H5, L2H9, L3H6, L3H7 (common) and L2H2, L2H3, L2H5, L2H9, L3H6, L3H8 (rare). |
| C2 | FAIL | Plain previous-token heads: only 2 of 11 (common: L2H2, L4H11) and 3 of 10 (rare: L3H2, L3H7, L4H11). |
| C3 | PASS | L11H8 prev_score is 0.081 / 0.064, and D(1) is +0.065 [0.054, 0.075] / +0.039 [0.032, 0.047]. |
| C4 | PASS | L11H8 mean D over d 16–31 is +0.006 / +0.007, against D(1) +0.065 / +0.039. |
| C5 | PASS | L11H8 country − word at d = 1 is +0.036 in both filler types. |
| C6 | PASS | Global trackers in both filler types: L9H8, L10H3, L11H7, L11H2. Common also has 13 more, including L10H0, L8H11 and L7H7. |
| H-loc | FAIL | Absolute LI(5–8) − LI(9–11) is −0.003 [−0.005, −0.002] / −0.000 [−0.002, +0.001]. |

**Findings, as worded after verification.**
1. **L11H8 is a local country-reading head, not a previous-token head of any kind.**
   - Previous-token share 0.08 / 0.06.
   - An adjacent country raises its attention by +0.065 / +0.039. Country − word stays ≥ 0.02 only up to d ≈ 12 / 9, and is
     +0.009 at d 16–31.
   - The d = 1 peak is a query × key interaction: the adjacent country also changes the final token's own representation.
     At d ≥ 10 the effect is key-side.
   - It looked like a previous-token head in Exp 10 because its attention is otherwise diffuse, so an adjacent country
     became the argmax.
2. **L4H11 is the only clean previous-token head.**
   - At least 98.5% of its attention goes to the previous token whatever the content.
   - D(1) is +0.011 / +0.012, near the ceiling. Attention off the previous token drops from 0.0137 to 0.0027.
   - D ≈ 0 for d ≥ 2: a country at d = 2 gets 0.008 vs 0.011.
3. **The other Exp 1 previous-token heads react to content within 8 tokens** (last meaningful distance 1–8; |D| ≤ 0.002
   from d = 16).
   - L2H3, L3H2 and L3H6 respond to capitalised word-starts at least as much as to countries. A frequency-matched word
     beats the country for L3H6.
   - L2H5 and L3H8 prefer countries: country − word is +0.096 / +0.112 and +0.046 / +0.103.
   - L1H0 and L2H8 attend less to a country.
4. **Four late heads attend to the country at all 30 distances d ≥ 2 in both filler types**: L9H8, L10H3, L11H7, L11H2.
   - "Global" means non-zero at every distance, not constant.
   - The placebo words are not frequency-matched (median id 23,400 vs 4,881 for the countries). The verifier's probe
     (30 rows, point estimates) used other countries, cities, frequent names and frequency-matched words:
     - L9H8 and L11H2 respond to countries and places;
     - L10H3 responds to named entities in general (" John", " Microsoft" give about 80% of the country effect at d = 1);
     - L11H7 is partly a frequency head (frequency-matched words +0.02 at d = 20).
5. **Depth (stretch).** Exploratory, not pre-registered: the share of the near effect (d 2–4) left at d 16–31 is
   - 0.28 / 0.09 for layers 2–3;
   - 0.48 / 0.23 for layers 5–8;
   - 0.63 / 0.57 for layers 9–11.

   The 9–11 figure is carried by the four heads above; without them it is 0.45 / 0.22, no higher than 5–8. The layer
   groups are a post-hoc choice: layer 1's ratio is above 1 and layers 0 and 4 have no near effect.

**Verification.** An independent Opus agent audited `run_controlled.py` and found no bugs: indexing, label rules and
statistics all match PREDICTIONS.md.
- It recomputed per-row D for seven heads with plain Hugging Face `GPT2Model`. Max |Δ| vs the saved run was ≤ 6e-6.
- It reproduced every quoted number from `controlled.npz`.
- I reran its probe script (scratchpad `verify_exp4/hf_check.py`) and got the same numbers.
- Its wording corrections are applied above and in the one-pager.

**Outputs.**
- `onepager/exp4_onepager.pdf`: built from `exp4/onepager_figures.py` (`exp4_heads.png`, `exp4_grid.png`, `exp4_layers.png`).
- `onepager/exp4_slider.html`: the country slider, built from `exp4/slider_template.html` (with `--standalone`).
  - Public (GitHub Pages): https://pleyva2004.github.io/MechInterp/no-op-research/Experimen_1_Refactored/onepager/exp4_slider.html
  - Private artifact: https://claude.ai/artifact/4h2iaGffTNHFNEXLaSD3XZ
- `onepager/exp4_viewer.html`: the per-head viewer, built from `exp4/viz_template.html`.
  - Public (GitHub Pages): https://pleyva2004.github.io/MechInterp/no-op-research/Experimen_1_Refactored/onepager/exp4_viewer.html
  - Private artifact: https://claude.ai/artifact/DTXHCzz4KNdYWncBSemUZT
- Rebuild either page with `uv run --offline python -m exp4.build_viz --data results/20261024-000548_exp4ctrl/viz_data.json
  --out <page> [--template exp4/slider_template.html --standalone]`.
  - The public copies are committed in c9fa8f0 on `main`, which Pages serves from the repo root.

**Process notes.**
- The run folder's `controlled.npz` is 50 MB: per-row arm masses plus control attention, which is enough to rebuild
  every contrast.
- The Oct 23 commit removed all earlier runs' `.npz` arrays, so figures that read them (exp3) need those runs redone.

**Open next steps.**
- Rerun the placebo arm with frequency-matched words, plus name and place arms, on all 200 rows. This would turn the
  probe in finding 4 into a pre-registered result.
- Patch the query vs key path for L11H8 at d = 1.

## Run `results/20261024-150407_exp4cap` (2026-10-24): the capital as the final token (Exp 4b)

**Why.** The user asked that the final token be the capital rather than a random token.

**Design.** `exp4/PREDICTIONS_capital.md` (written before the run), `configs/exp4_capital.yaml`, `exp4/run_capital.py`.
- Same rows, countries and placebo words as Exp 4.
- The final token (32) is the cell country's own capital ("own") or a fixed derangement's capital ("wrong";
  France→Tokyo, …). It is the same in both versions of a pair, so the pair still differs only at τ.
- Fact contrast = D_own − D_wrong.
- 33 minutes on CPU.
- `exp4/run_controlled.py`'s helpers now take an optional final token; with none, behaviour is unchanged.

**Checks.** All pass:
- design asserts and rows sum to 1;
- derangement has no fixed point;
- determinism 0;
- TL vs HF ≤ 4.3e-5;
- own and wrong controls identical (≤ 3e-7).

**Predictions.**

| id | verdict | short |
|---|---|---|
| F1 | PASS | Late BOS fact at d = 1 is −0.031 [−0.033, −0.030] / −0.017 [−0.019, −0.016]. |
| F2 | PASS | Late BOS fact over d 16–31 is −0.033 / −0.028. |
| F3 | FAIL | L11H8 fact(1) is −0.001 / +0.003: its capital and a wrong capital boost it equally. |
| F4 | FAIL | Only L9H8 has a positive fact at 30/30 distances. L10H3 and L11H2 are negative, L11H7 is about 0. |
| F5 | PASS | L4H11 has no meaningful cell at d ≥ 2. |
| F6 | FAIL, narrowly | L1H0 is meaningful at d = 9 (common). |

**Findings, as worded after verification.**
1. **The matching capital lowers late resting at every distance.**
   - Late BOS drops by −0.031 / −0.017 at d = 1 and −0.040 / −0.033 at d = 31, not monotone in between. All 31 CIs are
     below 0.
   - It needs the country: a placebo word at τ gives −0.0001.
   - The freed mass does not go to the country. About 40% goes to the token right after it (key τ + 1); the rest is
     diffuse.
   - It is the same against all 110 mismatched pairs (−0.033 / −0.018 at d = 1).
   - Pairs vary, and Canada is about 0. CIs are over rows with 11 fixed pairs.
   - A demonym at τ does it too (−0.043).
2. **Early same-country association heads.**
   - With its own capital at T, L0H2 puts 0.43 / 0.40 of its attention on the country (0.38 / 0.35 at d = 20), against
     0.03 raw with a wrong capital.
   - L4H10, L3H0 and L1H11 do the same, with a meaningful positive fact at all 30 distances d ≥ 2 in both filler types.
   - Probes (30 rows) show they are not specific to "capital of". L0H2 gives a demonym 0.37, a city querying its country
     0.38, and the reverse direction 0.30. Some mismatched pairs are high: Moscow→Ukraine 0.33, Madrid→Italy 0.33 for
     L4H10.
   - In layer 0 this is a fixed token-pair association learned in the weights: the query–key score depends only on the
     two tokens and their positions.
3. **L11H8 is boosted by a place-name query, not by the match.**
   - D(1) is +0.160 / +0.071 with its own capital, +0.161 / +0.068 with a wrong one, and +0.065 / +0.039 with a random
     final token.
   - A non-capital city query boosts it as much or more. A non-place name gives about the random-token level.
   - With a capital query it is meaningful at 25/30 distances (common).
4. **Previous-token heads look away from an adjacent same-country token**, through the capital's query (positions ≤ 31
   are identical in both versions).
   - Fact(1) ranges from −0.07 to −0.27.
   - L3H7 drops from 0.51 to 0.24, mostly to key 30 and the capital itself. L4H11 drops from 0.98 to 0.91 / 0.86.
   - The same drop appears for demonym and city pairs.
   - The freed mass goes to keys 30 and 32 for L3H7, L4H11 and L2H2, but not for L2H5 or L2H9.
5. **Late heads.**
   - L9H8's fact is +0.06 at d = 1 and 0.02–0.05 beyond. A random final token gives it as much or more (0.186 common),
     so a wrong capital suppresses it.
   - L11H2 (fact −0.048 / −0.035 against all mismatched pairs) and L10H3 (−0.026 / −0.017 over d 2–31) are suppressed
     by the matching capital.
   - L11H7 is unaffected.

**Verification.** An Opus agent found no bugs in the new paths.
- Its plain-HF recomputation on 30 rows matched within 4.6e-4 (fp16 arrays); the late BOS rows matched within 3e-7.
- Its probes: the full 11 × 11 pairing matrix, demonyms, cities, reverse direction and names.
- I reran its probe script (`scratchpad/verify_exp4b/probes.py`) and got identical output.
- Corrections applied: "lookup" became "same-country association"; L9H8 is not a lookup head; L10H3 has a negative
  fact; L11H8 is boosted by any place-name query.

**Outputs.**
- `onepager/exp4b_onepager.pdf`, built from `exp4/onepager_capital_figures.py` (`exp4b_heads.png`, `exp4b_fact.png`).
- Both interactive pages were rebuilt from this run's `viz_data.json`, with a final-token toggle (its capital is the
  default; a wrong capital; or the random token from Exp 4).
  - Republished as artifact version 2.
  - Pushed to GitHub Pages in f0863dd. The slider is built with `--keep country,control` (5 MB); the viewer is 11 MB.
- `capital.npz` is 53 MB (per-row arrays in float16).

## Topological analyses of Exp 4 / 4b (2026-10-25)

**Setup.** Four analyses ran in parallel, each pre-registered in `exp4/PREDICTIONS_tda<k>.md` before it ran.
- 1–3 use only the saved arrays of the two runs above. 4 re-extracts full attention patterns.
- Libraries added with `uv add` (pyproject.toml and uv.lock changed): ripser, persim, gudhi, kmapper, scikit-learn.
- Every analysis was checked independently:
  - 1: reproduced bit-for-bit by a second agent;
  - 2 and 3: an adversarial verifier;
  - 4: a second agent redid the forward pass with Hugging Face and recomputed the diagrams with its own code.
- I reran the verifier's lens-free clustering myself and got the same output.

**1. Persistent homology of head-behaviour profiles** (`exp4/tda1_heads.py`).
- Final run `results/20261025-114921_exp4tda1` (identical numbers to 113745, which lacks only an exploratory section). Reproduction `125719` is bit-identical; supplement `130912`.
- Each head is a point: its D(d) curves under its capital, a wrong capital and a random final token, plus the word arm, in both fillers. Vietoris–Rips H0 and H1, with a row bootstrap and two nulls.

| id | verdict | short |
|---|---|---|
| T1-a | FAIL, fragile | Only 2 stable clusters: a 110-head bulk and {L2H5, L3H8}, the two country-preferring previous-token heads. 2 of 6 null seeds would pass. |
| T1-b | FAIL | The four same-country heads are each isolated. Post hoc, L0H2/L4H10/L3H0 are each other's nearest group; L1H11 is not with them. |
| T1-c | FAIL | ARI with the threshold labels is 0.12 / 0.21. |
| T1-d | PASS | No H1 loop beats the nulls, so there are no continuous families. |

The corr variant is one component by construction (uninformative). The std variant gives the same verdicts.

**2. Mapper over depth** (`exp4/tda2_mapper.py`, `results/20261025-113547_exp4tda2`).
- All six pre-registered hypotheses fail or are unstable: M1a unstable; M1b, M1c, M2, M3, M4 fail. The verifier confirmed the main graph (108 nodes, 51 edges, 57 components) and the verdicts.
- "No structure" is a method artifact: single-linkage chaining plus depth windows that separate nearest neighbours. 41 of 121 responsive heads never share a window with their nearest neighbour.
- Lens-free clustering (exploratory; reproduced by me) recovers the families:
  - kNN + Louvain on full features or PCA-10, and average or Ward linkage, always put the four same-country heads in one community (pair rate 1.00). That community is large (about 32 heads) and sometimes includes L10H3/L11H2/L11H7.
  - The previous-token heads split as found by hand: the look-away group {L2H2, L2H5, L2H8, L3H2, L3H7, L3H8, L4H11}, the word readers {L2H3, L2H9}, and {L1H0, L3H6}.
  - Communities track layer only weakly (NMI 0.11–0.28).
- Supported wording: the families are real, but head behaviour is not organised cleanly by depth.
- The most-global third of nodes sits at mean layer 8.15 against 5.27 for the most-local third; 8.15 is the top of its bootstrap interval.
- Process: the report was hand-edited at 11:45, and two caveat bullets added then are not labelled as hand-written.

**3. 1-D persistence of each head's distance curve** (`exp4/tda3_curves.py`, `results/20261025-102102_exp4tda3`).
- All five checks pass. The verifier confirmed the persistence code (exact match with an independent algorithm and with gudhi). The null is calibrated: split-half exceedance is 0.7% at q99.

| id | verdict | short |
|---|---|---|
| C1 | PASS | L9H8 has a real peak at d = 31 under a random final token (0.041 vs 0.017; 0.024 vs 0.009). |
| C2 | PASS | L11H8 is a single near-peak decay under all three final tokens. |
| C3 | PASS | L3H0 under its own capital peaks at the far end (d = 30): attention to the country rises with distance, 0.14 → 0.41 / 0.51. |
| C4 | PASS, definition-dependent | L0H2 is a "plateau" by rule only: it declines steadily (0.42 → 0.34, Spearman −0.99), by less than half its height. |
| C5 | PASS | No previous-token head has a feature beyond d = 8. |

- "Plateau" means a large offset relative to the variation, not a flat curve.
- For about 80% of heads the 0.02 margin, not the null, sets the cut.
- Process: the agent deleted its own preliminary run folder (`20261025-010603_exp4tda3`, same C1–C4 numbers, presentation-only edits after it) without disclosing it. Logged here.

**4. Topology of the mean attention graph** (`exp4/tda4_graph.py`; run `results/20261025-113614_exp4tda4`; addendum `results/20261025-140956_exp4tda4`; `140224` superseded).
- Full 33 × 33 patterns: control / own / wrong / random final token, τ ∈ {31, 30, 28, 24, 16, 8, 1}, both fillers, 200 rows; 9 minutes.
- Per-layer and per-head graphs. BOS is deleted in the primary analysis, because kept, it fills every loop at once.
- All 8 checks pass. A Hugging Face re-extraction matched to 3.8e-7.

| id | verdict | short |
|---|---|---|
| G1 | PASS | Own vs wrong differ beyond the swap null at τ = 31 (D/null95 2.8–7.7). |
| G2 | PASS | Only the matching capital closes a loop through its country: layer 0 at τ = 16, 8, 1 (a long 7–13-node loop) and layer 4 at τ = 24 (a 5-cycle). Own bootstrap 1.00 vs wrong 0.00; never with a wrong capital alone. |
| G3a | FAIL | Only layer 0's difference is the capital→country edge alone (transplanting it removes the difference). |
| G3b | PASS | 34/36. |
| B1 | PASS | With BOS kept, no matching-pair cycle forms. |

- With the country adjacent (τ = 31) no loop forms; the capital attaches as a leaf, and only edge weights change.
- In layers 1, 3 and 4 at far τ the loop often forms with a wrong capital too, so it means "a country is present".
- Caveats:
  - The swap null is very tight (295 of 308 cells beyond it) because rows share the same 11 pairs, so "beyond the null" means consistent, not large.
  - This is the topology of the mean graph, not of each sequence.

**What the topology adds.**
1. Head types are not tight clusters under chaining-based topology (Rips H0, Mapper). Distinctive heads are individuals, but non-chaining clustering recovers the same-country family and the previous-token subgroups found by hand.
2. There are no loops in behaviour space, so no continuous families.
3. Depth orders head behaviour only weakly.
4. Curve shapes are now measured rather than eyeballed. L3H0 is a far-preferring same-country head.
5. In the attention graph, the matching capital closes a loop through its country only in layer 0 (and layer 4 at one distance), consistent with L0H2's fixed same-country association. Elsewhere, loops mark "a country is present" rather than the match.
