# Test A predictions: which part of the phrase carries the effect? (recorded 2026-10-04, before any Test A data)

Design: the 12 iteration-3 phrases PRON VERB OBJ on the iteration-3 bases (same cache, same rows), positions 29–31.
New pair variants per phrase, with the third slot keeping its base token: PV = PRON@29 + VERB@30,
VO = VERB@30 + OBJ@31, PO = PRON@29 + OBJ@31 (non-adjacent). Per row, with f(S) = value with tokens S written
minus `none`:
- pair interaction, e.g. VO = f(VO) − f(V) − f(O)
- triple = f(PVO) − f(PV) − f(VO) − f(PO) + f(P) + f(V) + f(O) = nl interaction − PV − VO − PO
Familiarity of an adjacent pair (a, b) = the model's −log p(b | BOS a) (TL); lower = more familiar.

Metric: late-layer (L6–11) mean sink_mass at the query; pooled = mean over the 12 phrases, CI by bootstrapping
phrases. Each prediction must hold in both base classes.

1. **A1: the verb–object pair adds sink on its own.** Pooled VO interaction > 0, CI excluding 0.
2. **A2: it is the main pair.** Pooled VO − PV > 0 and VO − PO > 0, CIs over phrases excluding 0.
3. **A3: it carries at least half the phrase's effect.** Pooled VO / pooled nl interaction ≥ 0.5.
4. **A4: little is left for the full triple.** |pooled triple| < pooled VO.
5. **A5 (low power, n = 12; sign only): more familiar verb–object pairs add more.** Spearman ρ between VO
   interaction and VO familiarity (−log p) is negative.
6. **A6 (sanity).** Controls hold (L4H11 prev_mass ≥ 0.94, L5H1 sink_mass ≥ 0.90, L0H1 Self) and the placebo has at
   most 1 meaningful head per (base, phrase).

Two thresholds in A6 were set after iteration 3, which they would have passed: the L4H11 floor drops from 0.95
to 0.94 (iteration 3 saw 0.949 in 1 of 240 cells), and the placebo allows 1 head per group (iteration 3: 2 of
3,456 tests at 500 rows per base).

Basis: in iteration 3 the order effect compared " I like sushi" with " like I sushi"; that swap changes both
adjacent pairs, including the pair ending at q−1 (" like sushi" vs " I sushi"). Previous-token heads let the token at
31 read the token at 30, so the pair ending at q−1 is the most direct route to the query. If the triple term
dominates instead (A4 fails), the late heads respond to the 3-token structure, not to a word pair.
