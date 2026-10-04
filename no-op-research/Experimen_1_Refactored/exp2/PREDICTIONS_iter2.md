# exp2 predictions, iteration 2 (recorded 2026-10-03, after iteration 1's results, before any iteration-2 data)

Question: is the phrase's effect on late heads (layers 6–11, where it raised sink_mass) explained by its tokens'
single-position effects adding up (H2), or does the combination add something that tracks how predictable the
context is to the model (H1)?

Design: the iteration-1 bases (same cache, same seeds, so `none`, `nl`, `shuffled`, `matched`, `matched_b` are
the identical rows), `near` placement only (29–31). New variants: each phrase token alone at each slot it occupies
in nl or shuffled: I@29, like@30, sushi@31 (nl's parts) and like@29, I@30 (shuffled's; sushi@31 is shared). Per
row, interaction = (combo − none) − Σ(part − none). Surprisal = the model's summed −log p of the tokens at
positions 29–31 given everything before them (TL).

Metric: sink_mass at the final query. Late = layers 6–11, early = layers 0–5; depth values are means over that
group's 72 heads per row, with 95% bootstrap CIs over rows. All predictions must hold in both base classes.

1. **Q1 (H1): the grammatical combination adds sink in late layers.** The late-layer mean interaction of `nl` is
   positive, CI excluding 0.
2. **Q2 (H1): more than the scrambled one does.** Late-layer mean of (interaction of nl − interaction of shuffled)
   is positive, CI excluding 0.
3. **Q3 (H1 over H2): the order effect is not additive.** Across the 144 heads, the additive prediction of
   nl − shuffled from the single-token variants explains less than half its variance (r² < 0.5).
4. **Q4 (H1): late heads rest on BOS more when the slot tokens are more predictable.** Within rows, across all 10
   variants, the late-layer mean sink_mass falls as slot surprisal rises (slope < 0, CI excluding 0).
5. **Q5 (H1 is about late heads): the surprisal slope is steeper late than early.** |late slope| > |early slope|.
6. **Q6 (sanity): controls hold and the placebo stays empty.** As P5 and P6 of iteration 1.

What would count against H1: Q1/Q2 near zero or negative with Q3 r² high (H2: additive token-position effects),
or Q4's slope ≥ 0 (predictability is not what late heads track). Iteration 1 showed same-form random words
*lower* late-layer sink (form −); if those words are also more predictable than the base's random tokens, Q4 will
fail, and that would be informative rather than a nuisance.
