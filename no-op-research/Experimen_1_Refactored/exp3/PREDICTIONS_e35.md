# Pre-registered predictions: Experiment 3, E3.5 (punctuation: does a delimiter replace BOS as the resting place?)

Written 2026-10-15, before any E3.5 attention data. The E3.5 section below is copied verbatim from `exp3/DESIGN.md`
§3. Added below it, fixed before any E3.5 run, are the operational details that make each prediction computable.
They change no prediction and no threshold. Config: `configs/exp3_e35.yaml`. Shared measurement rules:
`exp3/DESIGN.md` §2. The role metrics (categories, label rules, switch rule, thresholds) are the ones E3.4 uses.

## E3.5 as designed (verbatim from `exp3/DESIGN.md` §3)

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

Each prediction must hold in both base classes (rare 1000–39999, common 256–999), 300 rows per base class
(`exp3/DESIGN.md` §4).

Reading, from `exp3/DESIGN.md` §6: "E5a–c pass | After a phrase, resting mass moves onto the delimiter, more than
onto a non-delimiter byte, and resting does not shrink: the no-op is "rest on a boundary", and BOS is one boundary
among several".

## Operational details (fixed before data)

- **Rows and units.**
  - Each row is BOS + 32 random ids with no repeats, and no phrase id appears in the filler.
  - The units are the 12 pronoun–verb–object phrases of iteration 3. Every row carries every phrase variant for
    every phrase.
  - Attention at the query (position 32) is averaged over a row's 12 phrase instances before any mass or label is
    taken (§2 rule 3).
- **Variants** (13). Names are `<context>+<d>`, with d ∈ {period ".", comma ",", newline "\n", x "x", seven "7",
  r}.
  - phrase+d: P V O at 26–28 and d at 29.
  - random+d: the row's own tokens at 26–28 and d at 29. random+r is the base row itself ("none").
  - random_b+r (the placebo): random+r with 26–28 redrawn. The 3 ids are drawn once per row from the base pool,
    excluding every id of the row.
  - With d = r, position 29 keeps the row's own token. Positions 30–32 always hold the row's own tokens, so 31 is r
    in every arm.
- **Byte ids** are checked at run time with the gpt2 tokenizer: "." → 13, "," → 11, "\n" → 198, "x" → 87,
  "7" → 22. In every variant, the byte at 29 is the only id ≤ 255 in the row; the r arms have none.
- **Phrase-independent variants.**
  - The six random+d variants and the placebo do not depend on the phrase. Each holds one sequence per row,
    identical in all 12 phrase slots.
  - Each is computed once per row and counts for every phrase slot. The attention is identical, so the row average
    is unchanged.
  - Sequences computed: 6 × 12 × 300 × 2 + 7 × 300 × 2 = 47,400. DESIGN.md's 94k counts these variants once per
    phrase, as an upper bound.
- **Masks** (categories by position, taken from the contrast's treated arm, applied to both arms).
  - Delimiter is position 29 in every variant, whatever token it holds.
  - phrase+byte: BOS 0, Filler 1–25 and 30, Span 26–28, Delimiter 29, Previous 31, Self 32. Phrase-position mass
    covers 26–29.
  - phrase+r: the same categories; phrase-position mass covers 26–28.
- **Contrasts** (treated − control, per row).
  - gain_d = phrase+d − phrase+r, for each of the 5 byte d. Masks come from phrase+d.
  - rgain_d = random+d − random+r. Masks also come from phrase+d, so all four arms of the interaction share one set
    of masks. Span 26–28 then holds the row's random tokens in the random arms.
  - ctx_d = phrase+d − random+d, for each of the 6 d (ctx_r = phrase+r − none). Masks come from phrase+d.
  - placebo = random_b+r − random+r. Masks come from phrase+period, the masks of E5a's switch contrast.
  - Interactions (difference of two per-row contrasts with identical masks):
    - x_d = gain_d − rgain_d;
    - period_vs_x = gain_period − gain_x, and period_vs_seven = gain_period − gain_seven.
- **Measures.** Each category mass (BOS, Previous, Self, Copy, Dup, Delimiter, Span, Filler), phrase-position mass,
  and BOS+Delimiter (the sum of the BOS and Delimiter masses). Delimiter-position mass is the Delimiter mass, the
  weight on key 29.
- **Late pooled.** Per row, the mean over the 72 heads of layers 6–11 of that row's Δ. CIs come from 2,000
  bootstrap resamples of rows (percentile intervals). Per phrase: the bootstrap over the 12 per-phrase means, plus a
  sign count (reported).
- **Labels and switches.**
  - E5a uses argmax-key labels (Diffuse when the max weight is < 0.2). Mass labels are reported next to them.
  - A head switches to label X when McNemar's exact test on discordant rows gives BH q ≤ 0.05 (BH over the 144
    heads, per destination label) and X's row share rises by ≥ 15 points.
  - The switch's source is the label whose share falls most.
- **Placebo count.** The number of distinct heads, out of all 144, that switch in the placebo contrast (any labels,
  either direction, argmax labels). "Beyond placebo" means count − placebo count. Rule check: placebo ≤ 1.
- **E5a.** Both parts must hold.
  - (i) x_period on Delimiter mass: the 95% CI of the late pooled per-row value lies above 0.
  - (ii) In gain_period, count the heads in layers 6–11 that switch with source Sink and destination Delimiter.
    Pass needs count − placebo count ≥ 5.
- **E5b.** x_period on BOS+Delimiter mass: the one-sided 95% lower bound (the 5th percentile of the row bootstrap)
  is > −0.005 (m).
- **E5c.** period_vs_x and period_vs_seven on Delimiter mass: both 95% CIs of the late pooled per-row value lie
  above 0.
- **Statistics.** Each prediction must hold in both base classes.

## Checks (must pass for the predictions to count)

- Design:
  - Every variant equals its base row outside its written positions.
  - The filler holds no written id; there are no repeats within a row; no sequence contains BOS.
  - phrase+d and phrase+r differ only at 29, and so do random+d and random+r.
  - phrase+d and random+d differ only at 26–28, and so do the placebo and random+r.
  - random+r is the base row.
  - The random variants and the placebo are identical across phrase slots.
  - The byte at 29 is the row's only id ≤ 255.
  - The placebo's ids come from the base pool and differ from the row's own.
- Attention rows sum to 1 (atol 1e-4).
- Determinism: a second TL pass on the first chunk of rows matches exactly, for the phrase-dependent and the shared
  variants.
- Cross-library: TL vs HF on identical ids for the first 25 rows per class (all variants and phrases), with max
  |Δ attention| < 1e-4.
- The placebo yields ≤ 1 switching head (§2 rule 2).
- Exp 1's previous-token heads keep Previous throughout.
  - In each of the 13 variants, each head's role (its most common argmax label over rows, under that variant's own
    masks) is Previous.
  - No listed head has an argmax-label switch whose source is Previous, in any contrast.
  - DESIGN.md: "a change there flags a confound". A FAIL is reported as that flag, next to the verdicts.

## Reported without a prediction

- "," and "\n": gain_comma, gain_newline, x_comma, x_newline.
- The context effect at each d (ctx_d), and the gains on random context (rgain_d).
- Per contrast, interaction and measure: the late pooled per-row value with 95% and 90% CIs, a TOST verdict
  (m = 0.005), and the CI pooled over phrases.
- Per-head Wilcoxon with BH, and the number of meaningful heads (q ≤ 0.05, |Δ| ≥ 0.02).
- Per contrast: role switches (argmax and mass labels), transition matrices, and TV distance between category
  profiles.
