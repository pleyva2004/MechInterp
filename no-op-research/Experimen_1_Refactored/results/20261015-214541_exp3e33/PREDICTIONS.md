# Pre-registered predictions: Experiment 3, E3.3 (distance sweep)

Written 2026-10-15, before any E3.3 attention data. The E3.3 section below is copied verbatim from `exp3/DESIGN.md`
§3. Below it are the operational details that make each prediction and output computable, fixed before any E3.3 run;
none of them changes a prediction or a threshold. Config: `configs/exp3_e33.yaml`. Shared measurement rules:
`exp3/DESIGN.md` §2 (m = 0.005; equivalence = TOST, the 90% CI lies inside ±m).

## E3.3 as designed (verbatim from `exp3/DESIGN.md` §3)

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

Each prediction must hold in both base classes (rare 1000–39999, common 256–999), 200 rows per base class
(`exp3/DESIGN.md` §4).

## Operational details (fixed before data)

- **Rows and units.**
  - Each row is BOS + 32 random ids with no repeats, and no phrase id appears in the filler (as in Exp 2). The rows
    are a new cached base block per class (`exp3e33_<class>`, seed 0).
  - The units are the 12 pronoun–verb–object phrases of iteration 3, with iteration 3's shuffled ids (V P O).
  - Every row carries every variant for every phrase. `none` is the same sequence for each of a row's 12 phrase
    slots (DESIGN.md §4 counts it per phrase). That gives 25 × 12 × 200 × 2 = 120k sequences.
  - Attention at the query is averaged over a row's 12 instances before any mass or label is taken (§2 rule 3).
- **Variants** (model positions, BOS = 0; the filler is always the row's own tokens):
  - `nl_g<g>`: P V O at 29 − g, 30 − g, 31 − g, for each g in the sweep. g = 0 is 29–31; g = 15 is 14–16; g = 24 is 5–7.
  - `shuffled_g<g>`: V P O at the same positions, so O sits at 31 − g in both.
  - `none`: the row unchanged.
  - `matched`, `matched_b`: at 29–31, two independent per-row draws from each phrase token's matched pool (Exp 2's
    `matched`: same leading-space / alphabetic / capitalised features and the same id class). Each draw avoids every
    id of the row.
  - `nl_len24`, `shuffled_len24`: the row's first 24 random tokens only (query = the row's own token at 24), with P V O
    or V P O at 21–23. DESIGN.md counts the length control as 2 variants, so there is no `none` at length 24, and no
    total effect at length 24.
- **Masks** come from each contrast's treated (first) arm and apply to both arms (§2 rule 1). Categories, given the
  treated arm's own query q (32 at full length, 24 in the length control):
  - g = 0: BOS 0, Span 29–30, Previous 31 (holds O), Self 32, Filler the rest. Phrase-position mass: 29–31.
  - g ≥ 1: BOS 0, Span = the three phrase positions, Previous 31 (a filler token), Self 32, Filler the rest.
    Phrase-position mass: the three phrase positions.
  - Length control: BOS 0, Span 21–22, Previous 23 (holds O), Self 24, Filler 1–20. Keys 25–32 do not exist.
    Phrase-position mass: 21–23.
  - Placebo: the masks of `nl_g0` (the same positions as `matched`).
  - Copy, Dup and Delimiter are not defined in E3.3, so their masses are 0 by construction. The measures analysed are
    BOS, Previous, Self, Span, Filler and phrase-position mass.
- **Contrasts** (treated − control, paired per row):
  - `order_g<g>` = `nl_g<g>` − `shuffled_g<g>`, for every g: E(g).
  - `total_g<g>` = `nl_g<g>` − `none`, for every g: the total effect.
  - `len24` = `nl_len24` − `shuffled_len24`: E within the 24-token rows.
  - `placebo` = `matched` − `matched_b`.
  - No contrast compares variants of different lengths. The length-control comparison is between per-row E values,
    each computed within its own length.
- **E(g), per row:** sink = BOS mass. Average the row's attention over its 12 phrases, take nl − shuffled BOS mass
  per head, then the mean over the 72 heads of layers 6–11.
- **E3a.** Per row, x = E_row(0) − 2 · E_row(1). Pass if the 95% CI over rows lies above 0. E(0) and E(1) are reported
  next to it.
- **E3b.** Per row, E_row(24). Pass if the 90% CI over rows (the 5th–95th percentiles of the row bootstrap) lies inside
  (−0.005, +0.005).
- **E3c.** Per row, the late pooled `nl_g15` − `none` BOS mass. Pass if the 95% CI over rows lies above 0. The number of
  late heads with a meaningful positive total effect at g = 15 is reported next to it (iteration 1's comparison).
- **Statistics.**
  - CIs come from 2,000 bootstrap resamples of rows, using percentile intervals.
  - The CI pooled over phrases (bootstrap over the 12 per-phrase means, plus a sign count) is reported next to every
    prediction. It does not gate the verdict.
  - Each prediction must hold in both base classes.
- **Per-head half-distance** (output, no prediction):
  - E_h(g) is head h's row-mean Δ BOS mass in `order_g<g>`.
  - It is defined for heads whose g = 0 effect is meaningful: Wilcoxon BH q ≤ 0.05 over the 144 heads, and |Δ| ≥ 0.02.
  - Half-distance = the smallest g > 0 in the sweep with E_h(g) / E_h(0) ≤ 0.5. The ratio is sign-aware, so a
    negative E_h(0) halves the same way. A head whose ratio stays above 0.5 through g = 24 is reported as "> 24".
  - Heads with half-distance ≥ 4 (including "> 24") are listed as long-range candidates.
  - The same is reported for the total effect (`total_g<g>`), marked as supplementary.
- **Length control** (output, no prediction): E(24 tokens, g = 0) = late pooled `len24`, next to E(0) and E(8) at full
  length. The per-row paired differences E_len24 − E(0) and E_len24 − E(8) are reported with 95% CIs.
- **Labels and switches** (reported, no prediction): argmax and mass labels, McNemar with BH, and the ≥ 15-point
  shift rule, for every contrast, together with transition matrices and TV distance.

## Checks (must pass for the predictions to count)

- Design:
  - Every variant equals its base row (the first 24 tokens, in the length control) outside its written positions.
  - The filler holds no written id, and no sequence contains BOS or a repeated id.
  - `nl_g<g>` and `shuffled_g<g>` differ only at 29 − g and 30 − g. Both hold O at 31 − g, and shuffled is V P O.
  - `nl_g<g>`, `shuffled_g<g>`, `matched` and `matched_b` differ from `none` only at their written positions.
  - `matched` and `matched_b` come from their matched pools.
  - `nl_len24` and `shuffled_len24` differ only at 21–22.
  - The query token is the row's own token at 32 (at 24 in the length control) in every variant.
- Attention rows sum to 1 (atol 1e-4).
- Determinism: a second TL pass on the first chunk of rows matches exactly.
- Cross-library: TL vs HF on identical ids for the first 10 rows per class (all 25 variants and 12 phrases), with max
  |Δ attention| < 1e-4.
- The placebo yields ≤ 1 switching head (§2 rule 2).

## Reported without a prediction

- Per contrast and measure: the late pooled per-row value with 95% and 90% CIs, a TOST verdict (m = 0.005), and the
  CI pooled over phrases.
- Per-head Wilcoxon with BH, and the number of meaningful heads.
- Role switches (argmax and mass labels), transition matrices, and TV distance between category profiles.
- The sweep curves E(g) and the total effect over g, pooled and per head.
