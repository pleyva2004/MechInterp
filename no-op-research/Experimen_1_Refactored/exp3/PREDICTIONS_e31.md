# Pre-registered predictions: Experiment 3, E3.1 (query inside vs outside the phrase)

Written 2026-10-15, before any E3.1 attention data. The E3.1 section below is copied verbatim from `exp3/DESIGN.md`
§3. Two things are added, both fixed before any E3.1 run: the concrete O′ list (in the config, with a one-line
justification per phrase) and the operational details that make each prediction computable. Neither changes a
prediction or a threshold. Config: `configs/exp3_e31.yaml`. Shared measurement rules: `exp3/DESIGN.md` §2. The role
metrics (categories, label rules, switch rule, thresholds) are the ones frozen after E3.4
(`results/20261015-211838_exp3e34`, E4a PASS).

## E3.1 as designed (verbatim from `exp3/DESIGN.md` §3)

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

## O′ list (fixed before any E3.1 run; `oprime` in the config)

The list is a derangement: every object serves exactly once as O′. So per row, PV+O and PV+O′ hold the same 12 query
ids, and only their pairing with the context differs. rr+O and rr+O′ then hold the same 12 sequences in a different
phrase order.

How the objects were assigned:
- Restrictive verbs get an object that the verb's meaning excludes.
- The four preference verbs (like, love, hate, want) accept almost any noun. They got the four objects left over,
  assigned to minimise their summed log P subject to log P(O′) < log P(O). For these four phrases O′ is "less
  probable", not "impossible".

Check: GPT-2 small next-token log P on BOS + " P V" (clean context), computed once to choose and verify the list.
This is not attention data. Every O′ is less probable than O.

| phrase | O′ | text | why | log P(O′) | log P(O) |
|---|---|---|---|---|---|
| sushi | songs | " I like songs" | like takes any noun; leftover object | −9.6 | −8.8 |
| water | chess | " we need chess" | a board game is not a necessity | −13.0 | −8.7 |
| pizza | music | " she ate music" | music is not edible (DESIGN.md's example) | −10.7 | −7.0 |
| music | cars | " they love cars" | love takes any noun; leftover object | −8.7 | −7.4 |
| coffee | money | " he drinks money" | money is not a drink | −8.1 | −4.6 |
| chess | coffee | " you play coffee" | coffee is not a game or an instrument | −11.9 | −7.7 |
| rain | rice | " I hate rice" | hate takes any noun; leftover object | −10.2 | −8.7 |
| cars | rain | " we sell rain" | rain cannot be sold | −9.7 | −7.3 |
| songs | water | " she wrote water" | water is not something written | −12.6 | −10.9 |
| rice | pizza | " they grow pizza" | pizza is a dish, not a crop | −11.5 | −8.9 |
| bikes | sushi | " he fixed sushi" | food is not repaired | −12.6 | −11.0 |
| money | bikes | " you want bikes" | want takes any noun; leftover object | −13.7 | −8.3 |

GPT-2 does not rank O first in every context: for example, " he fixed" prefers " money" (−8.7) to " bikes" (−11.0).
So "O completes the phrase" is a semantic label, not "O is GPT-2's top prediction". The run reports log P of every
query in its E3.1 context (below), without a threshold.

## Operational details (fixed before data)

- **Variant names in code.** O′ is written `Op`: PV+Op and rr+Op.
- **Rows and units.**
  - Each row is BOS + 32 random ids with no repeats, and no phrase id appears in the filler.
  - 500 rows per base class: rare (1000–39999) and common (256–999).
  - The units are the 12 phrases, and every row carries every variant for every phrase.
  - rr+r is the base row itself, the same sequence in all 12 phrase slots. It is kept per phrase so every variant
    has 12 instances per row. That gives 13 × 12 × 500 × 2 = 156k sequences, the DESIGN.md table's count.
  - Attention at the query (position 32) is averaged over a row's 12 instances before any mass or label is taken
    (§2 rule 3).
- **M.** One id per row and phrase, drawn uniformly from O's matched pool (same leading space, is_alpha,
  is_capitalized, same id class; minus the phrase's own ids), never an id of that row. The same M is used in PV+M and
  rr+M, so the query is identical within C(M).
- **Placebo.** matched and matched_b each write, per row and phrase, one id per phrase token at 30, 31 and 32. Each
  id comes from that token's matched pool and is never an id of the row. The two draws are independent. The placebo
  contrast is matched_b − matched.
- **Masks** (§2 rule 1; positions from the treated variant, applied to both arms):
  - Every E3.1 contrast uses PV+O's masks: BOS 0, Span 30, Previous 31, Self 32, Filler the rest. Phrase-position mass
    = keys 30–31.
  - The placebo also uses PV+O's masks, the masks of the contrast it controls (E1c). matched writes the same positions.
  - The sanity contrast uses PVO+r's masks: Span 29–30, Previous 31, Self 32. Phrase-position mass = keys 29–31.
  - Copy, Dup and Delimiter are empty in every E3.1 variant, so they are not tested.
- **Contrasts** (treated − control):
  - Two-arm: C(O) = PV+O − rr+O; C(O′) = PV+Op − rr+Op; C(M) = PV+M − rr+M; C(r) = PV+r − rr+r;
    PV+O − VP+O and PV+r − VP+r (reported alone); sanity = PVO+r − rr+r; placebo = matched_b − matched.
  - Differences of two contrasts, per row on the row's phrase-averaged attention, with PV+O's masks on all four arms:
    - interaction = C(O) − C(r);
    - completion = C(O) − C(M);
    - plausibility = C(O) − C(O′);
    - order × query = (PV+O − VP+O) − (PV+r − VP+r).
  - McNemar needs two arms, so the differences get measures only (no labels, switches or TV).
- **Labels and switches** (as frozen after E3.4).
  - Argmax-key labels (Diffuse when the max weight is < 0.2) decide E1c. Mass labels are reported next to them.
  - A head switches to label X when McNemar's exact test on discordant rows gives BH q ≤ 0.05 (BH over the 144 heads,
    per destination label) and X's row share rises by ≥ 15 points.
  - The switch's source is the label whose share falls most.
- **Placebo count.** The number of distinct heads that switch in the placebo contrast (argmax labels, any
  destination, either direction, all 144 heads). "Beyond placebo" means count − placebo count. Rule check: placebo
  count ≤ 1.
- **Late pooled value.** Per row, the mean over the 72 heads of layers 6–11 of the measure's difference.
- **E1a.** Per row, the late pooled BOS mass of PVO+r − rr+r. Pass if the 95% CI over rows lies above 0.
- **E1b.** Per row, the late pooled BOS mass of the interaction C(O) − C(r). Pass if the 95% CI over rows lies below 0.
- **E1c.**
  - Contrast PV+O vs rr+O, argmax labels.
  - Count the distinct heads in layers 6–11 that switch with source Sink and destination Span or Previous. A head that
    passes for both destinations counts once.
  - Pass if count − placebo count ≥ 5.
- **E1d.**
  - Order × query on phrase-position mass (keys 30–31), late pooled.
  - Per phrase: the mean over rows of the late-head mean of the difference.
  - Bootstrap over the 12 per-phrase means. Pass if the 95% CI lies above 0. This is the "pooled 95% CI over
    phrases" of DESIGN.md. The per-row 95% CI is reported alongside and does not decide.
- **E1e.** Per row, the late pooled BOS mass of C(O) − C(M). Pass if the 95% CI over rows lies below 0.
- **E1e bet (low confidence).** Per row, the late pooled BOS mass of C(O) − C(O′). Pass if the 95% CI over rows lies
  below 0. DESIGN.md writes the bet inside E1e; it is scored as its own row (`E1e-bet`) and does not change E1e's
  verdict.
- **Statistics.**
  - CIs come from 2,000 bootstrap resamples of rows (or of phrases for E1d), using percentile intervals.
  - Each prediction must hold in both base classes.
- **Row count.** 500 per base class, as DESIGN.md's table says. The §2 rule 4 power calculation is not part of this
  runner (the E3.4 runner has none either).

## Checks (must pass for the predictions to count)

- Design:
  - Every variant equals its base row outside its written positions. The written positions include the query (32)
    in PV+O, rr+O, PV+Op, rr+Op, PV+M, rr+M, VP+O, matched and matched_b.
  - The filler holds no written id, and no sequence contains BOS.
  - No repeats are declared, so any repeated id within a row fails, the query included.
- Layout:
  - rr+x differs from PV+x only at 30–31 (x ∈ O, O′, M, r), and VP+O / VP+r from PV+O / PV+r only at 30–31. So the
    query is identical within C(x) and within each order contrast.
  - The PV+x variants differ from each other only at 32, and so do the rr+x variants.
  - PVO+r differs from rr+r only at 29–31; matched and matched_b differ from rr+r only at 30–32.
  - The ids sit where the table says.
  - O′ is another phrase's object, shares no id with the phrase, and lies in O's matched pool (same form and class).
  - M and the matched / matched_b ids lie in their matched pools.
- Attention rows sum to 1 (atol 1e-4).
- Determinism: a second TL pass on the first chunk of rows matches exactly.
- Cross-library: TL vs HF on identical ids for the first 25 rows per class (all variants and phrases), with max
  |Δ attention| < 1e-4.
- The placebo yields ≤ 1 switching head (§2 rule 2).

## Reported without a prediction

- Per contrast (two-arm and differences) and measure (each non-empty category mass and phrase-position mass):
  - the late pooled per-row value with 95% and 90% CIs;
  - a TOST verdict (m = 0.005); E1b equivalent to 0 is the "context alone" reading of DESIGN.md;
  - the CI pooled over phrases (bootstrap over the 12 per-phrase means, plus a sign count).
- Per-head Wilcoxon with BH, and the number of meaningful heads.
- Role switches (argmax and mass labels), transition matrices, TV distance and layer profiles.
- logit(BOS) check (Exp 2's bounded-scale check). For each difference contrast: per row and head, the same difference
  of logit(BOS mass), with mass clipped to [1e-6, 1 − 1e-6]. Then the late mean per row, with a 95% CI.
- Query log P (manipulation). On the first 25 rows per class, log P of O, O′, M and r at position 32 under the PV, VP
  and rr contexts, from the logits at position 31.
