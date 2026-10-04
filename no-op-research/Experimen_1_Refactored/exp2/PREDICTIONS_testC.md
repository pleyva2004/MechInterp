# Test C predictions: does it generalize beyond pronoun–verb–object? (recorded 2026-10-04, before any Test C data)

Design: two new templates, 12 phrases each, positions 29–31, run separately with the iteration-3 pipeline:
- `adj`: adjective–noun–verb (e.g. " big dogs bark"), config `configs/exp2_testC_adj.yaml`
- `prep`: preposition–determiner–noun (e.g. " on the table"), config `configs/exp2_testC_prep.yaml`
shuffled = the first two tokens swapped, last token kept in place (e.g. " dogs big bark", " the on table"). The same
five single-token variants per phrase; fresh bases per template (the excluded phrase ids differ).

Metric: late-layer (L6–11) mean sink_mass at the query; interaction per row = (combo − none) − Σ(part − none);
pooled = mean over phrases, CI by bootstrapping phrases. Each prediction must hold in both base classes, and is
checked separately for each template.

1. **C1: the grammatical phrase adds sink beyond its tokens.** Pooled late nl interaction > 0, CI excluding 0.
2. **C2: more than the scrambled one.** Pooled late nl − shuffled interaction > 0, CI excluding 0.
3. **C3: across most phrases.** nl − shuffled late interaction > 0 with its own CI excluding 0 in ≥ 8 of 12.
4. **C4: not a scale artifact.** C2 also holds for logit(sink_mass).
5. **C5 (sanity).** Controls hold (L4H11 prev_mass ≥ 0.94, L5H1 sink_mass ≥ 0.90, L0H1 Self) and the placebo has at
   most 1 meaningful head per (base, phrase). Thresholds as in Test A, set after iteration 3.

Expectation beyond the checks: `prep` may give smaller effects than `adj`. Preposition–determiner and
determiner–noun pairs are frequent and short, and " the" is a single shared token, so the phrases differ only in
their first and last words.

What would count against generality: C1/C2 failing for either template means the iteration-3 effect is specific to
pronoun–verb–object (or to a subject followed by its verb).
