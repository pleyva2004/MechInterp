# Pre-registered predictions: Experiment 3, E3.4 (repeated phrase: induction as the positive control)

Written 2026-10-15, before any E3.4 attention data. The E3.4 section below is copied verbatim from `exp3/DESIGN.md`
§3. Two things are added, both fixed before any E3.4 run: the concrete E4d head list (chosen from Test A's saved
tables) and the operational details that make each prediction computable. Neither changes a prediction or a
threshold. Config: `configs/exp3_e34.yaml`. Shared measurement rules: `exp3/DESIGN.md` §2.

## E3.4 as designed (verbatim from `exp3/DESIGN.md` §3)

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

Each prediction must hold in both base classes (rare 1000–39999, common 256–999), 500 rows per base class.

## E4d head list (fixed before any E3.4 run)

**L7H0, L6H0, L8H7, L7H11, L8H5, L6H11, L9H3, L6H8, L10H2, L8H11.**

How they were chosen: Test A's run `results/20261004-152303_exp2`, table `additivity_heads.parquet`, rows with
`combo == "VO"`, `metric == "sink_mass"` and layer 6–11. The VO interaction per head is the row-mean
(VO − none) − (V − none) − (O − none) on BOS mass, one value per phrase and base class. Each head's 24 values
(12 phrases × 2 base classes) are averaged, and the 10 heads with the largest mean are kept. Larger means more BOS
parking beyond additivity. Ranking by |mean| gives the same 10 heads.

| head | mean VO interaction (sink mass) |
|---|---|
| L7H0 | +0.0590 |
| L6H0 | +0.0578 |
| L8H7 | +0.0496 |
| L7H11 | +0.0325 |
| L8H5 | +0.0305 |
| L6H11 | +0.0287 |
| L9H3 | +0.0262 |
| L6H8 | +0.0258 |
| L10H2 | +0.0224 |
| L8H11 | +0.0201 (11th: L9H11, +0.0200) |

## Operational details (fixed before data)

- **Rows and units.** Each row is BOS + 32 random ids with no repeats, and no phrase id appears in the filler. The
  units are the 12 pronoun–verb–object phrases of iteration 3, and every row carries every variant for every phrase.
  The phrase-independent variants (repeat-rand, norepeat-rand, norepeat-rand_b) get a fresh X Y Z (and r′) for each
  row and phrase slot, so every variant has 12 instances per row. That gives 8 × 12 × 500 × 2 = 96k sequences, the
  DESIGN.md table's count. Attention at the query (position 32) is averaged over a row's 12 instances before any mass
  or label is taken (§2 rule 3).
- **Masks** come from each contrast's treated (first) arm and are applied to both arms. In the repeat variants they
  are BOS 0, Span 10, Dup 11, Copy 12, Previous 31, Self 32, and Filler for the rest. Phrase-position mass covers
  every written key before the query (10, 11, 12, 31). The placebo uses repeat-rand's masks, the masks of the
  contrast it controls, so both count switches in the same label space.
- **Contrasts** (treated − control):
  - E4a and E4b: repeat-rand − norepeat-rand.
  - E4c: repeat-nl − repeat-matched; repeat-nl − repeat-rand is reported alongside. Both hold other tokens at
    31–32 than repeat-nl, so both are compared on copy-target mass only (as DESIGN.md says), with no labels or
    switches.
  - E4d: repeat-nl − other-nl.
  - E4e: repeat-nl − repeat-nl-1.
  - Placebo: norepeat-rand_b − norepeat-rand.
- **Labels and switches.**
  - E4a uses argmax-key labels (Diffuse when the max weight is < 0.2). Mass labels are reported next to them.
  - A head switches to label X when McNemar's exact test on discordant rows gives BH q ≤ 0.05 (BH over the 144
    heads, per destination label) and X's row share rises by ≥ 15 points.
  - The switch's source is the label whose share falls most.
- **Placebo count.** The number of distinct heads that switch in the placebo contrast (any labels, either
  direction, argmax labels). "Beyond placebo" means count − placebo count. Rule check: placebo ≤ 1.
- **Induction heads**, per base class: mean copy-target mass (position 12) in repeat-rand ≥ 0.3, averaged over rows
  and phrases.
- **E4b.**
  - Use the masses of each induction head in each row: c = norepeat-rand, t = repeat-rand.
  - BOS loss = m_BOS(c) − m_BOS(t), and Δcopy = m_copy(t) − m_copy(c).
  - Excess = BOS loss − Δcopy · m_BOS(c) / (1 − m_copy(c)), averaged over the induction heads in each row.
  - Pass if the 95% CI over rows lies above 0.
- **E4c.** Per row, take the induction-head mean of copy-target mass in repeat-nl − repeat-matched. Pass if the
  one-sided 95% lower bound (the 5th percentile of the row bootstrap) is > −0.05.
- **E4d.**
  - Run a paired Wilcoxon per head on per-row BOS mass (repeat-nl − other-nl), with BH over the 144 heads. A head
    "loses BOS mass" if q ≤ 0.05 and Δ ≤ −0.02.
  - Count the listed heads that lose BOS mass. Pass needs count ≥ 3 + the placebo count, where the placebo count is
    how many of the 10 heads are meaningful on BOS mass (q ≤ 0.05, |Δ| ≥ 0.02, either sign) in the placebo contrast.
    This applies §2's "every ≥ N heads prediction means beyond placebo".
  - The count must also be > the 95th percentile of the same count over 1,000 random sets of 10 heads, drawn without
    replacement from the other 62 late heads (fixed seed).
- **E4e.** Per row, take the induction-head mean of copy-target mass in repeat-nl − repeat-nl-1. Pass if the 95% CI
  over rows lies above 0.
- **Statistics.**
  - CIs come from 2,000 bootstrap resamples of rows, using percentile intervals.
  - If a base class has no induction heads, E4b, E4c and E4e fail in that class.
  - Each prediction must hold in both base classes.
- **Fallback** (`fallback: true` in the config; only if E4a fails).
  - Variants: fb-repeat; fb-norepeat (2–16 left as the row's own tokens); and fb-norepeat_b (2–16 replaced by an
    independent random draw: the placebo).
  - One instance per row, 500 rows per base class.
  - E4a's rule is applied with Copy 16 and Dup 15 (masks from fb-repeat). Induction heads are reported from
    fb-repeat.

## Checks (must pass for the predictions to count)

- Design:
  - Every variant equals its base row outside its written positions.
  - The filler holds no written id, and no sequence contains BOS.
  - Repeats occur only at the declared positions (10↔31 and 11↔32 in repeat-nl, repeat-matched and repeat-rand;
    11↔32 in repeat-nl-1), and those repeats are present in every row.
  - Each control differs from its treated variant only at 10–12, and repeat-nl-1 from repeat-nl only at 31.
  - other-nl shares no id with its phrase.
  - A′ B′ C′ come from their matched pools, and X Y Z and r′ from the base pool.
- Attention rows sum to 1 (atol 1e-4).
- Determinism: a second TL pass on the first chunk of rows matches exactly.
- Cross-library: TL vs HF on identical ids for the first 25 rows per class (all variants and phrases), with max
  |Δ attention| < 1e-4.
- The placebo yields ≤ 1 switching head (§2 rule 2).
- L5H5 and L6H9 are among the induction heads (reported as a check; it does not gate the predictions).

## Reported without a prediction

- Per contrast and measure (each category mass and phrase-position mass): the late pooled per-row value with 95%
  and 90% CIs, a TOST verdict (m = 0.005), and the CI pooled over phrases (bootstrap over the 12 per-phrase means,
  plus a sign count).
- Per-head Wilcoxon with BH, and the number of meaningful heads.
- Role switches (argmax and mass labels), transition matrices, and TV distance between category profiles.
- repeat-nl − norepeat-nl (the repeat effect on natural text).
