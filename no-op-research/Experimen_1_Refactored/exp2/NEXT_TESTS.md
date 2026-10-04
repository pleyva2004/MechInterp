# Next tests (after exp2 iterations 1–3)

Starting point (details in `exp2/LOOP_LOG.md`): a coherent 3-token phrase at positions 29–31, ending right before
the query, raises late-layer (L6–11) sink_mass beyond the sum of its single tokens. The scrambled order (same
tokens, same last token) doesn't, and surprisal doesn't explain it. This held for 12/12 phrases, both base classes,
TL and HF.

Every test: write predictions before the run, reuse the exp2 pipeline (paired rows, interaction = combo − none −
Σ(parts − none)), and launch any run longer than 10 minutes detached (`nohup`).

## Test A: which part of the phrase carries the effect?

**Goal.** Find out whether the extra sink comes from the verb–object pair (" like sushi"), the subject–verb pair
(" I like"), or only from the full triple, and whether pair familiarity predicts it.

**Procedure.**
1. Reuse the 12 phrases and bases from iteration 3. For each phrase PRON VERB OBJ, add three pair variants (the
   third slot keeps its base token):
   PRON@29 + VERB@30, VERB@30 + OBJ@31, PRON@29 + OBJ@31.
2. Pair interaction = pair − none − (single_a − none) − (single_b − none), per row; average over the late heads.
3. Triple term = full nl interaction − sum of the three pair interactions.
4. Pair familiarity = the model's −log p(second token | first token) right after BOS.
5. Across phrases, regress each pair interaction on pair familiarity.

**Prediction.** VERB@30 + OBJ@31 carries most of the effect: the object at q−1 reads the verb through
previous-token heads. If the triple term dominates instead, the heads respond to the 3-token structure, not to a
word pair.

## Test B: which components carry the signal? (activation patching)

**Goal.** Find the path from the phrase to the late heads' sink at the query: which positions, layers and heads.

**Procedure.**
1. Take paired runs of the same base row: nl and shuffled. They differ only at positions 29–30.
2. Metric: late-layer mean sink_mass at the query. Effect = nl − shuffled.
3. Position × layer sweep: for each layer L and position p in {29, 30, 31, 32}, patch nl's `resid_pre` at (L, p)
   into the shuffled run. Recovered fraction = (patched − shuffled) / (nl − shuffled).
4. Head sweep: at the positions that recover the most, patch single head outputs (`hook_z`) from nl into shuffled
   for layers 0–5, and rank heads by recovered fraction.
5. Confirm: mean-ablate the top heads in the nl run and check that the nl − shuffled gap shrinks. Read their
   attention patterns, e.g. whether position 31 attends to 30.
6. Use TL only (HF has no hooks), 200 rows per base, 4–6 phrases.

**Prediction.** The signal runs through position 31's residual stream and is written by early previous-token heads
(L2H2, L4H11 and similar).

## Test C: does it generalize beyond pronoun–verb–object?

**Goal.** Check whether "coherent phrase → late heads rest on BOS" holds for other phrase types.

**Procedure.**
1. Two new templates, 12 phrases each, single tokens with leading spaces, ids in 256–39999:
   - adjective–noun–verb, e.g. " big dogs bark"
   - preposition–determiner–noun, e.g. " on the table"
2. shuffled = swap the first two tokens; the last token stays put. Add the same five single-token variants per
   phrase.
3. Run the iteration-3 config with the new phrase list. Config only; no code changes. About 25 min per template,
   detached.
4. Check: pooled late nl interaction > 0 and pooled late nl − shuffled > 0 (CI over phrases), per template and base.

**Prediction.** Both hold for both templates; effects may be smaller for prepositional phrases, whose pairs
(" on the") are frequent in any order.

## Later (not now)

- Replicate the strongest result on gpt2-medium.
