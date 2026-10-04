# exp2 predictions, iteration 1 (recorded 2026-10-03, before any exp2 data was generated)

Design: " I like sushi" (ids 314, 588, 36324) written over 3 random tokens of each base row, at positions 29–31
(`near`, ending right before the query) or 14–16 (`far`). Controls: the same ids as " like I sushi" (`shuffled`,
last id unchanged) and same-form random words (`matched`). Bases: rare (1000–39999) and common (256–999).
Metric: sink_mass at the final query unless stated. "Meaningful" = BH-significant and |Δ| ≥ 0.02.

Basis, from exp1ext (results/20260928-135243_exp1ext):
- One substituted token at position 16 barely moves any head: per-head sd of the per-token effect has median
  0.005 (rare bases) / 0.007 (common), max 0.020 / 0.033.
- At position 31 (right before the query) the same sd has median 0.029 / 0.030, max 0.092 / 0.116; 20 heads
  exceed 0.05 on rare bases. " like" at 31 alone moved L7H7 +0.20, L7H0 −0.17, L5H2 −0.16.
- Late heads needed most of the 31 context tokens changed before their rare→common effect appeared.

1. **P1: the far phrase is close to a no-op.** At positions 14–16, no head's total effect (nl − none) reaches
   |Δ| 0.10, and the median head's |Δ| is ≤ 0.015, in both base classes.
2. **P2: position matters more than content.** Median |Δ total| over heads is at least 3× larger near than far,
   in both base classes.
3. **P3: near, the position-31-sensitive heads move.** At least half of the 20 heads with exp1ext position-31
   sd ≥ 0.05 shift by |Δ| ≥ 0.05 on rare bases.
4. **P4: word order barely matters.** nl − shuffled is meaningful for at most 14 heads (10%) near and at most 2
   far, per base class. The position-31 token is the same in both, and exp1ext found heads tracking token form,
   not combinations of tokens.
5. **P5: controls hold in every cell.** L4H11 mean prev_mass ≥ 0.95, L5H1 mean sink_mass ≥ 0.90, L0H1 mode label
   Self.
6. **P6 (sanity): the placebo is empty.** Two independent same-form draws (matched − matched_b) give 0
   meaningful heads.

Not predicted (exploratory): whether identity (shuffled − matched) or form (matched − none) carries more of the
total; whether effects agree in sign across base classes; whether the final token attends more to the phrase
slots in nl than in shuffled (slot_mass, order contrast).
