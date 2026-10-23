# Pre-registered predictions: Experiment 3, E3.2 (span length: from one word to natural text)

Written 2026-10-15, before any E3.2 attention data. The E3.2 section below is copied verbatim from `exp3/DESIGN.md`
§3. Added, all fixed before any E3.2 run: the 12 passages and their final-pair types, and the operational details
that make each prediction computable. None of them changes a prediction or a threshold. Config:
`configs/exp3_e32.yaml`. Shared measurement rules: `exp3/DESIGN.md` §2.

## E3.2 as designed (verbatim from `exp3/DESIGN.md` §3)

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

Each prediction must hold in both base classes (rare 1000–39999, common 256–999), 200 rows per base class.

## Passages and final-pair types (fixed before any E3.2 run)

Text, gpt2 token ids and t are in `configs/exp3_e32.yaml` (the run checks that each text tokenizes to its ids and t to
one id). Every passage has ≥ 32 tokens before t; every word id in its last 31 tokens lies in 256–39999; t is a single
word token that does not occur in the passage.

| passage | genre | final pair | type | t |
|---|---|---|---|---|
| fisherman | narrative | ' would be' | aux–verb | ' better' |
| party | dialogue | ' from the' | prep–det | ' meeting' |
| bridge | news | ' the daily' | det–adj | ' commute' |
| tea | instructions | ' to five' | prep–num | ' minutes' |
| plants | science | ' of the' | prep–det | ' sun' |
| moon | history | ' in human' | prep–adj | ' history' |
| game | sports | ' they could' | pron–aux | ' win' |
| garage | diary | ' looking at' | verb–prep | ' them' |
| storm | weather | ' is absolutely' | aux–adv | ' necessary' |
| pizza | review | ' definitely go' | adv–verb | ' back' |
| password | tech support | ' a new' | det–adj | ' password' |
| rabbit | children's story | ' out to' | particle–prep | ' hunt' |

## Operational details (fixed before data)

- **Rows and units.**
  - Each row is BOS + 32 random ids from the base class range, with no repeats. The rows are shared across passages.
  - Per passage, any filler id that occurs anywhere in that passage's text, or equals its t, is redrawn uniformly
    from the class pool minus the passage's ids, t and the row's ids (fixed seed per row and passage).
  - The units are the 12 passages, and every row carries every variant once per passage: 36 variants × 12 passages
    × 200 rows × 2 base classes = 172,800 sequences.
  - Attention at the query (position 32) is averaged over a row's 12 passages before any mass or label is taken
    (§2 rule 3).
- **Positions.**
  - The passage's last k tokens sit at 32 − k … 31, so a passage token sits at the same position for every k.
  - The query (32) is the row's own random token, except in the 2 × 2.
- **shuffled (k ≥ 3).**
  - Per passage and k, one fixed permutation of the first k − 1 span tokens; the token at 31 is kept.
  - It is drawn from a fixed seed, is the same in both base classes, and must change the id sequence.
  - The orders are recorded in `design_summary.json`.
- **matched (per k) and matched_b.**
  - Punctuation means byte ids (0–255) and any token whose stripped text is not alphabetic (here '.', ',', ' "',
    '?"', ',"'). Punctuation stays in place.
  - Each distinct word id of the passage's last 31 tokens gets one draw per row and passage, from its matched pool
    (same `leading_space`, `is_alpha`, `is_capitalized` and id class; Exp 2's rule). The pool excludes every id of
    the passage and t; the draw avoids the row's ids, and distinct words get distinct draws. The repeat pattern of
    nl is therefore kept.
  - The draw is made once for the 31-token span; matched at k is its last k tokens, nested like nl.
  - matched_b is an independent second draw by the same rule.
- **passage-rr.** nl at k = 31 with 30–31 left as the row's own tokens (passage tokens at 1–29).
- **2 × 2 at k = 31.**
  - Cells: passage+t (`natural`: passage at 1–31, t at 32), passage+w, none+t (the row with t at 32) and none+w.
  - w is drawn per row and passage from t's matched pool, minus the passage's ids and the row's ids. The same w
    appears in passage+w and none+w.
- **Masks** come from each contrast's treated (first) arm and apply to both arms (§2 rule 1).
  - nl at k vs none: Span 32 − k … 30, Previous 31, Self 32, Filler the rest.
  - passage+t vs passage+w: Span 1–30, Previous 31, Self 32.
  - Phrase-position mass covers every written key before the query.
- **E(k)**, per row: late pooled BOS mass (mean over the 72 heads of layers 6–11) in nl at k minus none, on the
  row's passage-averaged attention.
- **E2a.** All three parts must hold.
  - E(3): the 95% CI over rows lies above 0.
  - E(8) − E(3), per row nl_8 − nl_3: TOST, the 90% CI lies inside (−0.005, +0.005).
  - E(31) − E(3), per row nl_31 − nl_3: the same TOST.
- **E2b.** For each k ∈ {3, 4, 6, 8, 12, 16, 24, 31}: per row nl_k − shuffled_k late BOS, 95% CI above 0. All eight
  must hold. There is no multiplicity correction: each k needs its own CI above 0.
- **E2c.** Both parts must hold.
  - (i) Interaction per row: (passage+t − none+t) − (passage+w − none+w) late BOS, 95% CI below 0.
  - (ii) Count the heads in layers 6–11 that switch Sink → Span (argmax labels) in passage+t vs passage+w. A switch
    needs McNemar's exact test on discordant rows at BH q ≤ 0.05 (BH over the 144 heads, per destination label) and
    a ≥ 15-point rise in Span's row share; the source is the label whose share falls most. The count minus the
    placebo count must be ≥ 5.
  - The placebo count is the larger of the two placebo contrasts' counts (distinct heads switching, any labels,
    either direction, argmax labels).
- **E2d.** Per row E(31) − 2 · (passage-rr − none) = nl_31 − 2 · passage-rr + none, late BOS, 95% CI above 0.
- **Placebo.** matched_b − matched at k = 8 and at k = 31, masks from matched at that k. Rule check: each yields
  ≤ 1 switching head (§2 rule 2).
- **Statistics.**
  - CIs come from 2,000 bootstrap resamples of rows (percentile intervals).
  - The per-passage CI (bootstrap over the 12 per-passage means, plus a sign count) is reported next to every
    pooled value; it does not gate a prediction.
  - Each prediction must hold in both base classes.
- **Outputs without a prediction.**
  - Saturation fit: least squares of E∞ · (1 − e^(−k/κ)) on the 10 row-mean E(k), with κ ∈ [0.05, 1000] and
    E∞ ∈ [−1, 1]. 95% percentile CIs on κ and E∞ come from refitting 2,000 row resamples; the share of resamples
    that end on a bound is reported.
  - E(k) per final-pair type: per passage, the mean over rows of late BOS nl_k − none, then the mean over the
    passages of each type.
  - Switch length, per head: the smallest k in the grid such that the head's role switches (§2 rule 2, any
    destination) in nl_k vs none at that k and at every larger k in the grid. It is computed for argmax and mass
    labels. A head that does not switch at k = 31 has no switch length.
  - Recorded per passage × k, for the nl and shuffled spans: the number of repeated ids, the number of extra
    occurrences, and the shortest repeat distance.

## Checks (must pass for the predictions to count)

- Design:
  - Every variant equals its passage's base row outside its written positions.
  - The filler holds no written id, and no sequence contains BOS.
  - Repeats occur only between positions that hold the same passage id (declared per passage), in every row.
  - Each base row holds no id of its passage or t, and differs from the shared row only where that row held one.
  - nl, shuffled, matched and matched_b differ only on the span. shuffled keeps the token at 31 and changes the order.
  - matched keeps punctuation in place and draws each word from its pool with one id per distinct word, nested
    across k. passage-rr differs from nl at k = 31 only at 30–31.
  - Within the 2 × 2, the cells differ only where the table says: passage vs none at 1–31, t vs w at 32.
- Attention rows sum to 1 (atol 1e-4).
- Determinism: a second TL pass on the first chunk of rows matches exactly.
- Cross-library: TL vs HF on identical ids for the first 10 rows per class (all variants and passages), with max
  |Δ attention| < 1e-4.
- Each placebo yields ≤ 1 switching head (§2 rule 2).

## Reported without a prediction

- Per contrast and measure (each category mass and phrase-position mass): the late pooled per-row value with 95%
  and 90% CIs, a TOST verdict (m = 0.005), and the CI pooled over passages.
- Per-head Wilcoxon with BH, and the number of meaningful heads.
- Role switches (argmax and mass labels), transition matrices, and TV distance between category profiles.
- nl − matched per k (coherent text vs same-form words); nl at 31 − passage-rr (the final pair given long context);
  the 2 × 2's simple effects.
- Per-head ΔBOS curves over k.
