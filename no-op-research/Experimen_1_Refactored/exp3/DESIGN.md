# Experiment 3: natural-language injection and head behaviour

Class 1 (injection form). Class 2 (relational structure: country → capital, correct vs wrong) is in `DESIGN_relations.md`.

Status: designed, not run (2026-10-15); revised after an adversarial review the same day. Predictions are written before any Experiment 3 data.

## 1. Why this experiment

What Experiments 1–2 established (GPT-2 small, BOS + 32 random tokens, final query at position 32):

- 74% of heads park on BOS at the final token on random input, more in later layers (Exp 1).
- A coherent phrase right before the query makes layer 6–11 heads park on BOS *more*, beyond its tokens' single
  effects and beyond its scrambled order (Exp 2; three templates in Test C).
- Test A (pronoun–verb–object only) split it into adjacent pairs. The pair ending at 31 (VO) carried 46% on common
  bases (the pre-registered ≥ 50% failed) and 61% on rare; PV 34% / 22%. Test C did not decompose pairs. Test A's
  PV sat at 29–30 with a random token at 31: PV ending at 31 was never measured.

What they did not measure:

1. **Where attention goes instead.** Every test so far tracked BOS mass. We never asked which *job* a head switches
   to (previous token, phrase content, a copy target, a delimiter) or from which job it came.
2. **The query's role.** The query was always a random token *after* the phrase. We never let the query be part of
   the phrase, so we cannot tell "the phrase is complete, nothing to do" from "coherent text before any query".
3. **Length and distance.** Only 2–3 token phrases at two distances (positions 29–31 and 14–16).
4. **A positive control.** We have no case where a head is *known* to change job, so we have not shown our metrics
   would detect a switch.

**Goal:** inject natural language in controlled ways and map how individual heads change role: which heads switch,
from what to what, and under which conditions. Five sub-experiments (E3.1–E3.5), one shared measurement.

## 2. Shared measurement

**Rules shared with `DESIGN_relations.md`:**

1. **Position masks.** Category masks are defined by position, taken from the treated variant, and applied
   identically to both arms of every contrast. No category is empty by definition in a control. Also report
   **phrase-position mass** as its own measure: all injected positions, including 31. It overlaps with Previous.
2. **Role switch.** Per-row labels and a paired McNemar test, BH-FDR. A switch also needs a shift of ≥ 15
   percentage points in the share of rows carrying the destination label.
   - Diffuse: max weight < 0.2.
   - A mass-based label (the category with the largest summed mass) is reported next to the argmax-key label,
     because content reading spread over several keys otherwise shows up as Diffuse.
   - The placebo contrast must yield ≤ 1 switching head.
   - Every "≥ N heads" prediction means ≥ N heads more than the placebo count.
3. **Unit of analysis.** The base row is the unit. Average per row over phrases (or pairs/passages) first, then
   Wilcoxon over rows. Generalization across phrases uses per-phrase means (sign count plus bootstrap over phrases).
   The BH family is 144 heads × one metric × one contrast × one base class.
4. **Rows and power.** ≥ 500 rows per base class for E3.1 and E3.4. A power calculation from iteration 3's
   per-head SDs runs before launch.

**Applied here.**

Key categories (disjoint, masses sum to 1 at the final query q = 32; positions from the treated variant):

| category | keys |
|---|---|
| BOS | 0 |
| Previous | q − 1 (31) |
| Self | q (32) |
| Copy | E3.4: the position after the earlier occurrence of the query token |
| Dup | E3.4: the earlier occurrence of the query token |
| Delimiter | E3.5: the delimiter position (29) |
| Span | other injected positions before 31 |
| Filler | everything else |

- Phrase-position mass = every injected key before the query (31 when injected), so it overlaps Previous, Copy,
  Dup and Delimiter.
- Argmax label: Diffuse, else the argmax key's category (Sink / Previous / Self / Copy / Dup / Delimiter / Span /
  Filler). Exp 1's "Other" is the union of the last five, so Exp 1 labels stay recoverable.
- With several phrases per row, labels are taken on the row's attention averaged over its phrase instances (key
  positions align under rule 1), so each row has one label per arm, as McNemar needs.
- A head's role is its most common label over rows. A switch's source is the label whose row share falls most; the
  transition matrix counts switching heads by (source, destination).
- Mass tests: paired Wilcoxon per head on per-row category masses; meaningful = BH q ≤ 0.05 and |Δ| ≥ 0.02
  (Exp 2). Late = layers 6–11, pooled as the mean over their 72 heads.
- Margin m = 0.005 for "≈ 0" (about half the smallest pooled late effect in Exp 2, +0.009). Equivalence: TOST, the
  90% CI lies inside ±m. Non-inferiority: the one-sided 95% bound lies above −m.
- Placebo per run: two independent random draws over that run's treated positions (`matched` vs `matched_b`, as in
  Exp 2), named in each sub-experiment.

**Outputs** per contrast: from → to transition matrix (argmax and mass labels), per-head Δ of every category mass
and of phrase-position mass, TV distance between category profiles, layer profile. Figures: `attention_maps.py`-style
grids with the new categories, transition matrices, per-head curves for the sweeps (E3.2, E3.3).

**Inherited from Exp 2:** paired within-row contrasts, bootstrap CIs, logit(sink) check, TL and HF on identical ids,
determinism, replication across both base classes (rare 1000–39999, common 256–999), predictions in a file and
auto-checked, GPT-2 small only.

## 3. Sub-experiments

Phrases: the 12 pronoun–verb–object phrases of iteration 3 (P V O, e.g. " I like sushi" = 314 588 36324), unless
stated. `r` = the base row's own random token at that position.

### E3.1 Query inside vs outside the phrase (core)

**Question.** Does a late head park on BOS because the phrase is finished and the query has nothing to do with it?
If the query *completes* the phrase, do the same heads leave BOS and read the phrase? Is that about completion, or
only about the query being a real word?

**Design.** Context at 30–31 × query at 32, plus order and sanity variants:

| variant | 30 | 31 | 32 (query) | role |
|---|---|---|---|---|
| PV+O | P | V | O | query completes the phrase |
| rr+O | r | r | O | same query, no context (control) |
| PV+O′ | P | V | O′ | real object, implausible (" she ate music") |
| rr+O′ | r | r | O′ | control |
| PV+M | P | V | M | same-form random word, matched to O |
| rr+M | r | r | M | control |
| PV+r | P | V | r | same context, unrelated query |
| rr+r | r | r | r | none |
| VP+O | V | P | O | scrambled context, completing query |
| VP+r | V | P | r | scrambled context, unrelated query |
| PVO+r (at 29–31) | P V O | | r | iteration 3's `nl`, sanity |
| matched / matched_b (30–32) | | | | placebo |

- O′: another phrase's object, one per phrase, chosen as implausible and listed in the config before running. All
  12 objects are rare-class ids with a leading space, so O′ matches O in form and class.
- M: drawn per row from O's matched pool (Exp 2's `matched`), so it is a real word of O's form and class.

**Contrasts.** Context effect given query x: C(x) = PV+x − rr+x, x ∈ {O, O′, M, r}. Interaction: C(O) − C(r).
Completion vs any real word: C(O) − C(M). Completion vs plausibility: C(O) − C(O′). Order × query:
(PV+O − VP+O) − (PV+r − VP+r); PV+O − VP+O alone also changes the token at q − 1 (V vs P), which the difference
removes.

**Predictions** (each in both base classes, late layers 6–11, per row averaged over phrases):

- E1a (sanity): PVO+r − rr+r late sink > 0, as in iteration 3.
- E1b: C(O) − C(r) on late sink < 0. A completing query takes away the BOS boost.
- E1c: PV+O vs rr+O has ≥ 5 late heads (beyond placebo) that switch Sink → Span or Previous.
- E1d: order × query on late phrase-position mass (30–31) > 0, pooled 95% CI over phrases excludes 0. Coherent
  order plus a completing query makes heads read the phrase.
- E1e: C(O) − C(M) on late sink < 0 (completion, not any real word). Bet, low confidence: C(O) − C(O′) < 0.

**Reading.** E1b–c pass: parking depends on how the query relates to the context ("nothing to integrate"). That is
not a query-side result in Test B's sense: a query–key match with the phrase is key-side too, so Test B's q/k
patching is still needed to localize it. E1e separates a completion signal from "the query is a real word". E1b
equivalent to 0 (±m): coherent text before the query drives parking whatever the query is; consistent with Test B's
key-side story (B2), not evidence for it.

### E3.2 Span length: from one word to natural text (core)

**Question.** In Test A (one template) the pair before the query carries about half of the effect. Does more
coherent text keep adding, saturate, or switch heads into content roles once the context is natural text?

**Design.** 12 hand-written English passages of ≥ 33 tokens (narrative, dialogue, news, instructions), each with its
true next token t. For k ∈ {1, 2, 3, 4, 6, 8, 12, 16, 24, 31}, the passage's last k tokens before t are written at
positions 32 − k … 31; random filler before; query random.

| variant | what is written |
|---|---|
| nl (per k) | the passage's last k tokens |
| shuffled (k ≥ 3) | the same k tokens, one fixed random order per passage, the token at 31 kept (k ≤ 2 would equal nl) |
| matched (per k) | same-form random words; punctuation stays in place, and each distinct word id maps to one draw, so the repeat pattern is kept |
| passage-rr | k = 31 with 30–31 back to the row's random tokens: long context without the coherent final pair |
| 2 × 2 at k = 31 | context (passage vs none) × query (t vs w); w = a same-form wrong word, drawn per row from t's matched pool; `natural` = passage + t |
| matched_b (k = 8, 31) | placebo |

- Passages are written so every word id lies in 256–39999 and t is a word. `token_class_of` raises on byte ids
  (0–255) and ids ≥ 40000, so matched pools cannot be built for them; punctuation is never redrawn.
- Exclusion per passage, not global: rows are shared across passages, and a filler id that occurs in that passage
  is redrawn from the pool minus the passage and the row. Excluding all 12 passages' ids at once would remove many
  common-range ids (256–999).
- Recorded per passage × k: repeated ids inside the written span (count, shortest repeat distance), since repeats
  recruit duplicate-token and induction heads. Each passage's final-pair type (e.g. det–noun, verb–object,
  noun–punctuation) is assigned before data.

**Predictions** (E(k) = late pooled nl − none sink):

- E2a (saturation): E(3) > 0 (CI excludes 0, so a flat-at-zero curve cannot pass), and E(8) − E(3) and
  E(31) − E(3) are each equivalent to 0 within ±m.
- E2b: nl − shuffled late sink > 0 for every k ≥ 3.
- E2c (bet, low confidence): (passage+t − none+t) − (passage+w − none+w) on late sink < 0, with ≥ 5 late heads
  (beyond placebo) switching Sink → Span in passage+t vs passage+w. This connects to E3.1 at full length.
- E2d: passage-rr − none ≤ 50% of E(31): E(31) − 2 · (passage-rr − none) > 0, CI excludes 0.

**Outputs (no prediction).** A saturation fit E(k) = E∞ · (1 − e^(−k/κ)) with a bootstrap CI on κ. E(k) per
final-pair type (descriptive, 12 passages). Per head, the "switch length": the smallest k from which its role is
switched (§2 rule 2) at that k and every larger k, as a heatmap over layer × head.

### E3.3 Distance sweep

**Question.** How fast does the effect fade as random tokens separate the phrase from the query, and do any heads
keep responding at a distance?

**Design.** Phrase P V O ending at 31 − g, g ∈ {0, 1, 2, 3, 4, 6, 8, 12, 15, 24}; random filler between phrase and
query. g = 15 is iteration 1's far slot (14–16). Main contrast nl − shuffled (shuffled = V P O, O stays last, as in
iteration 3): same tokens, same last token. none is kept for the total effect. Placebo: matched vs matched_b at g = 0.

Moving the phrase away from the query also moves it toward BOS, so g confounds distance with absolute position.
Length control: rows cut to their first 24 random tokens (query at 24), phrase at 21–23. That has the distance of
g = 0 and the absolute position of g = 8 at full length; nl − shuffled is compared within each length.

**Predictions** (E(g) = late pooled nl − shuffled sink):

- E3a: E(1) ≤ 50% of E(0) (the pair at q − 1 matters most): E(0) − 2 · E(1) > 0, CI excludes 0.
- E3b (equivalence): E(24) within ±m (TOST), both base classes.
- E3c (replication): at g = 15, late pooled nl − none > 0, CI excludes 0, both base classes. Far was not a no-op in
  iteration 1: the total effect was meaningful in 32/72 (common) and 24/72 (rare) late heads, late mean +0.014 /
  +0.009; the order effect there was +0.005 / +0.004 (14/72 and 2/72 heads).

**Outputs (no prediction).** Per-head half-distance (the g at which the effect halves); heads with half-distance ≥ 4
are listed as long-range candidates. Length control: E at (24 tokens, g = 0) against (32, g = 0) and (32, g = 8).

### E3.4 Repeated phrase: induction as the positive control (run first)

**Question.** Can our measurement see a head clearly change job? Induction is a known role in GPT-2 small: on a
repeat, the head attends to the token *after* the earlier occurrence of the current token. And: are the late heads
that park on BOS for a coherent phrase the same heads that switch to copying when a repeat gives them something to
copy?

**Design.** Phrase A B C (= P V O) early at 10–12; query = B at 32, so Copy is position 12 (holds C) and Dup is 11
(the earlier B).

| variant | 10–12 | 31 | 32 (query) |
|---|---|---|---|
| repeat-nl | A B C | A | B |
| norepeat-nl (control) | r r r | A | B |
| other-nl (control) | another phrase | A | B |
| repeat-nl-1 | A B C | r | B |
| repeat-matched | A′ B′ C′ | A′ | B′ |
| repeat-rand | X Y Z | X | Y |
| norepeat-rand (control) | r r r | X | Y |
| norepeat-rand_b (placebo) | r′ r′ r′ | X | Y |

- X Y Z: random ids drawn per row from the base pool, excluded from that row's filler. r′: an independent draw.
- A′ B′ C′: drawn per row from A B C's matched pools (same form and class), excluded from the filler.
- other-nl: another of the 12 phrases sharing no id with A B C, fixed per phrase in the config.
- Each control differs from its treated variant only at 10–12, and repeat-nl-1 from repeat-nl only at 31, so the
  query token is identical within a contrast. repeat-matched is compared with repeat-nl on copy-target mass only.

**Induction heads** are defined from our own data: heads with mean copy-target mass ≥ 0.3 in repeat-rand. (The IOI
work's L5H5 and L6H9 should be among them, as a check.)

**Predictions:**

- E4a: ≥ 3 heads in layers 5–7 (beyond placebo) switch Sink → Copy in repeat-rand vs norepeat-rand.
- E4b: in the induction heads, BOS loses more than its proportional share of the copy-target gain: BOS loss −
  Δcopy · m_BOS / (1 − m_copy), masses from norepeat-rand, is > 0 (per row, pooled over induction heads, CI
  excludes 0). Proportional renormalization alone already takes > 50% of the gain from BOS in any head with
  m_BOS > 0.5, so "≥ 50% from BOS" would pass with no BOS-specific effect.
- E4c (non-inferiority): induction-head copy-target mass in repeat-nl ≥ repeat-matched − 0.05 (one-sided 95% bound
  of the paired per-row difference). Coherent content does not suppress copying. repeat-rand reported alongside.
- E4d: of the 10 late heads with the largest VO interaction in Test A, ≥ 3 lose BOS mass (meaningful, Δ < 0) in
  repeat-nl vs other-nl, and that count exceeds the 95th percentile of the same count over 1,000 random sets of 10
  other late heads (the same heads can park or copy).
- E4e: induction-head copy-target mass, repeat-nl − repeat-nl-1 > 0, CI excludes 0: a two-token match (A B … A B)
  adds to a one-token match.

**Fallback (pre-registered).** If E4a fails, run once: a 15-id random block at 2–16 and its first 13 ids again at
19–31, query = its 14th id (Copy 16, Dup 15), against the same row with 2–16 left random. The role metrics
(categories, label rules, switch rule, thresholds) are frozen once E3.4 is analysed, before E3.1 runs; the fallback
tests them and does not change them. If the fallback also fails, stop: any metric change is a new pre-registration.

### E3.5 Punctuation: does a delimiter replace BOS as the resting place?

**Question.** Is the no-op "attend to BOS" specifically, or "attend to a boundary"? In larger models attention
also parks on periods and newlines.

**Design.** The delimiter sits off q − 1, so its mass is not previous-token mass: phrase at 26–28, position 29 = d,
30–31 = the row's random tokens. d ∈ {"." (13), "," (11), "\n" (198)}, non-delimiter byte controls {"x" (87),
"7" (22)}, or r. Context: phrase vs random (the row's own tokens at 26–28). Full 2 × 6 grid; none = random + r.
Placebo: random + r vs the same with 26–28 redrawn.

- The five byte d are ids 0–255, excluded from bases since Exp 1, so each is the row's only byte token. "x" and "7"
  separate "a delimiter" from "an out-of-pool byte token".
- 31 holds r in every arm, so Exp 1's previous-token heads (L1H0, L2H2, L2H3, L2H5, L2H8, L2H9, L3H2, L3H6, L3H7,
  L3H8, L4H11) should keep Previous throughout; a change there flags a confound.

**Predictions** (late pooled; Delimiter = position 29 in every arm, §2 rule 1):

- E5a: the phrase × d interaction on delimiter-position mass,
  (phrase+"." − phrase+r) − (random+"." − random+r), is > 0 (CI excludes 0), and ≥ 5 late heads (beyond placebo)
  switch Sink → Delimiter in phrase+"." vs phrase+r.
- E5b: the same interaction on BOS + delimiter-position mass is ≥ −m (non-inferiority): the resting place moves;
  resting does not shrink.
- E5c: after the phrase, the delimiter-position gain (phrase+d − phrase+r) is larger for "." than for "x" and for
  "7" (CI excludes 0).

"," and "\n" are reported without a prediction.

**Parked:** meaning vs syntax in the context (grammatical-but-nonsense phrases such as " I fold rain"; E3.1's O′
covers only the query); gpt2-medium replication.

## 4. Order and size

1. Power calculation from iteration 3's per-head SDs (`results/20261004-125831_exp2`); it can raise any row count
   below.
2. E3.4 first: the positive control validates the role metrics, which are then frozen (fallback in E3.4).
3. E3.1, then E3.2: the core questions.
4. E3.3, E3.5.

| run | variants | phrases | rows per base class | sequences (2 bases) |
|---|---|---|---|---|
| E3.4 | 8 | 12 | 500 | 96k |
| E3.1 | 13 | 12 | 500 | 156k |
| E3.2 | 36 | 12 passages | 200 | 173k |
| E3.3 | 25 | 12 | 200 | 120k |
| E3.5 | 13 | 12 | 300 | 94k |
| E3.4 fallback (only if E4a fails) | 3 | — | 500 | 3k |

Sequences = variants × phrases × rows × 2, phrase-independent variants counted per phrase (an upper bound).
E3.2: nl 10, shuffled 8, matched 10, passage-rr, 2 × 2 adds 4, none, matched_b 2. E3.3: nl 10, shuffled 10, none,
length control 2, placebo 2. E3.5: 2 × 6 grid plus placebo.

Iteration 3 (108k sequences, TL + HF) ran past the 10-minute limit, so every run goes detached (nohup + Monitor).

## 5. Code changes

- `exp2/design.py`: an `allow_query` option so writes can land on the query position (E3.1, E3.2's 2 × 2, E3.4);
  a repeat-tolerant check for E3.4 and passages that still requires filler to exclude every written id; per-passage
  exclusion by redrawing colliding filler ids; matched draws with fixed punctuation and a consistent id map; row
  truncation (E3.3 length control). `token_class_of` raises on byte ids and ids ≥ 40000, hence E3.2's passage rule.
- `exp3/roles.py` (new): position masks from the treated variant, argmax and mass labels on phrase-averaged rows,
  the McNemar switch test with the 15-point rule, transition matrices, TOST and non-inferiority, the
  proportional-renormalization baseline (E4b); reuses `exp2.analysis.paired_test` and BH-FDR.
- `exp3/run.py`: a thin wrapper over `exp2.run` adding the role outputs and the pre-launch power check;
  `configs/exp3_*.yaml`; `exp3/PREDICTIONS.md`; tests in `tests/test_exp3_*.py`.

## 6. What the outcomes mean for the thesis

| outcome | reading |
|---|---|
| E3.4 passes (E4a) | The role metrics detect a known switch |
| E4b passes | Copying draws on BOS beyond proportional renormalization: the sink is where these heads wait when no copy target exists |
| E3.1 interaction < 0 (E1b) | Parking depends on the query's relation to the context: heads park when the context offers the query nothing. Not localized: a query–key match is key-side too (Test B decides) |
| E1e passes | The release from BOS is a completion signal, not "the query is a real word" |
| E3.1 interaction ≈ 0 (within ±m) | Parking is driven by the context alone (coherent text before the query), independent of the query |
| E2a and E2d pass | The no-op signal is local (the last pair), matching Test A; long context adds nothing |
| E3.2 natural lowers sink (E2c) | Natural text with its true next token puts heads to work; the random-token regime overstates the no-op |
| E5a–c pass | After a phrase, resting mass moves onto the delimiter, more than onto a non-delimiter byte, and resting does not shrink: the no-op is "rest on a boundary", and BOS is one boundary among several |
| any prediction inconclusive, or base classes disagree | Reported per base class; no general claim |
