# exp2 predictions, iteration 3 (recorded 2026-10-04, after iteration 2's results, before any iteration-3 data)

Question: does iteration 2's finding generalize beyond " I like sushi", and is it a real effect rather than an
artifact of the bounded probability scale? The finding: a grammatical pronoun–verb–object phrase ending right
before the query raises late-layer sink_mass beyond the sum of its single tokens, and more than its scrambled
version does.

Design: 12 phrases " PRON VERB OBJ" (" I like sushi", " we need water", " she ate pizza", " they love music",
" he drinks coffee", " you play chess", " I hate rain", " we sell cars", " she wrote songs", " they grow rice",
" he fixed bikes", " you want money"), each written at positions 29–31. shuffled = " VERB PRON OBJ" (last token
kept; reliably ungrammatical). Five single-token variants per phrase (PRON@29, VERB@30, OBJ@31, VERB@29, PRON@30).
Fresh base rows (500 rare, 500 common), since excluding 36 phrase ids changes the pool. Late = layers 6–11.
Interaction per row = (combo − none) − Σ(part − none), averaged over the late heads. Pooled = mean over phrases,
95% CI by bootstrapping phrases. Everything must hold in both base classes.

1. **R0 (replication on new rows).** For " I like sushi": the late nl interaction and the late nl − shuffled
   interaction are both positive, CI over rows excluding 0.
2. **R1 (H1′ generalizes).** Pooled over phrases, the late nl interaction is positive (CI over phrases excludes 0).
3. **R2 (order generalizes).** Pooled late nl − shuffled interaction is positive (CI over phrases excludes 0).
4. **R3 (not driven by a few phrases).** The late nl − shuffled interaction is positive with its own row-bootstrap
   CI excluding 0 in at least 8 of the 12 phrases.
5. **R4 (not a scale artifact).** R1 and R2 also hold for logit(sink_mass).
6. **R5 (surprisal does not set the size; low power).** Across the 12 phrases, the Spearman correlation between the
   late nl − shuffled interaction and the surprisal gap NLL(shuffled) − NLL(nl) stays below 0.58 (the p < 0.05
   cutoff for n = 12). n = 12 gives little power, so a pass is weak evidence; a strong positive ρ would revive H1.
7. **R6 (sanity).** Controls hold and the placebo stays empty in every phrase.

What would count against H1′: R1/R2 failing (the sushi result does not generalize), or R4 failing (it was the
probability scale). A pass on R1–R4 with R3 marginal would mean a real but phrase-dependent effect.
