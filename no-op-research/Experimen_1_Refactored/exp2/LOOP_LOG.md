# exp2 loop log

Each iteration: write predictions → run → check predictions → state hypotheses → decide the next design.
Predictions for an iteration are written before its first run and never edited afterwards; a later iteration
gets a new predictions file. Capped at 3 iterations unless the user extends it.

Starting point (exp1ext, results/20260928-135243_exp1ext):
- Late heads' rare→common sink increase is driven by the whole context (needs >16 of 31 tokens changed), not
  by the final token alone.
- Single-token substitutions barely move heads at position 16 but move many heads at position 31 (q−1).
- Per-token effects track surface form (leading space, byte fragments, brackets), not the model's familiarity
  with the token.

## Iteration 1: " I like sushi", near vs far, order and form controls

- Config: `configs/exp2.yaml`; predictions: `exp2/PREDICTIONS.md` (P1–P6), recorded 2026-10-03 before data.
- Process note: after the predictions were written, a 20-row smoke test (TL only, 50 bootstrap draws, cache in
  the scratchpad) was run to check the pipeline. Its numbers were seen; the predictions were not changed.
- Run: `results/20261003-151225_exp2` (`results/20261003-150538_exp2` crashed after the analysis step on a
  config-iteration bug in the replication check, fixed; marked CRASHED.txt, no results in it).
- Checks: all PASS. `none` rows replicate exp1 (per-head mean sink r = 0.9998 rare, 0.9999 common). TL vs HF:
  max |Δ| 5.3e-05, labels agree 100%, meaningful decisions agree 100% (max |Δ rank-biserial| 0.054, on heads whose
  per-row differences are at float-noise level; no decision changes). Placebo: 0 meaningful heads everywhere.

### Prediction outcomes

| id | result | what happened |
|---|---|---|
| P1 far ≈ no-op | FAIL | median |Δ| fine (0.005 rare, 0.008 common), but common bases have outliers: L8H2 +0.123; 37 heads meaningful far (common), 25 (rare) |
| P2 near ≥ 3× far | FAIL | rare 5.1×, common 2.6× |
| P3 pos-31-sensitive heads move | PASS (barely) | 10/20, exactly the threshold |
| P4 word order barely matters | FAIL | order meaningful in 61 heads (common, near), 27 (rare, near), 14 (common, far) |
| P5 controls | PASS | L4H11 prev ≥ 0.98, L5H1 sink ≥ 0.97, L0H1 Self everywhere |
| P6 placebo empty | PASS | 0 |

### What the data show

1. Near ≫ far, but far is not null in common bases.
2. Identity (the phrase's tokens vs same-form random words) is the largest component: meaningful in 76 / 68 heads
   near (rare / common); form in 54 / 47.
3. Word order matters, against P4. Largest: L6H8 +0.163, L7H0, L5H2, L4H0 (common, near).
4. Direction is structured by depth. In layers 6–11 the phrase raises sink_mass: meaningful total effects + vs −
   are 48:4 (common near), 40:9 (rare near), 24:8 (common far), 23:1 (rare far). Layers 0–5 are mixed. Late-layer
   mean signed effect: identity +0.009 to +0.038, order +0.040 (common near) / +0.005 (rare near), form −0.004
   to −0.019 (same-form random words make late heads *less* sink-bound).
5. The exp1ext "sink rises" late heads (L7H11, L11H9, L11H1, L10H4, L11H3) move the way exp1's rare→common shift
   moved them: +0.10 to +0.14 in common bases near.

### Hypotheses after iteration 1

- **H1, predictability → late-head no-op.** Late heads fall back to BOS as the context gets more predictable to the
  model (common tokens, a real phrase, grammatical order) and stay active when it is unpredictable. This would
  unify exp1 (late heads diffuse on rare tokens), exp1ext (driven by the aggregate context, not the final token)
  and iteration 1 (identity +, order +). It does not obviously explain form −.
- **H2, additive position-specific token effects.** The order effect is just which token sits at 29 vs 30; each
  token's single-position effect adds up and the combination adds nothing. Single-token effects can share a sign
  across heads (exp1ext: " like" at 31 raised L7H7 by +0.20), so the consistent sign alone doesn't rule H2 out.
- **H3, the q−1 route.** The phrase acts mainly through the token at 31 and what it reads from 30. Fits near ≫ far;
  compatible with both H1 and H2.

### Decision

Iteration 2 tests H1 against H2 on the same bases and phrase, near only: add each phrase token alone at each slot
it occupies in nl or shuffled, so nl and shuffled can each be compared with the sum of their single-token parts
(an interaction = what the combination adds), and record the model's own surprisal of the slot tokens so sink_mass
can be regressed on predictability within each row.

## Iteration 2: additivity and surprisal (near only)

- Config: `configs/exp2_iter2.yaml`; predictions: `exp2/PREDICTIONS_iter2.md` (Q1–Q6), recorded after iteration 1's
  results and before any iteration-2 data. Same bases and seeds as iteration 1, so the shared variants are the
  identical rows.
- Code added: single-token custom variants (`exp2.design.build_custom`), `exp2.analysis.additivity_tables` and
  `surprisal_slopes`, slot surprisal from TL logits (`exp2.run.slot_nll`), new prediction kinds, report §7–8.
- Process note: a 20-row smoke run (scratchpad) checked the new pipeline after the predictions were written; its
  numbers were not used.
- Run: `results/20261003-230841_exp2`. Checks all PASS (same values as iteration 1). The 7,200 contrasts shared with
  iteration 1 are identical (max |Δ mean| = 0.0), as they must be with the same rows.

### Prediction outcomes

| id | result | what happened |
|---|---|---|
| Q1 nl interaction > 0, late | PASS | +0.028 [+0.024, +0.031] common, +0.013 [+0.011, +0.015] rare |
| Q2 nl − shuffled interaction > 0, late | PASS | +0.043 [+0.040, +0.046] common, +0.009 [+0.007, +0.011] rare |
| Q3 order effect not additive (r² < 0.5) | PASS | r² 0.001 common, 0.007 rare: the single-token parts predict nothing about the order effect across heads |
| Q4 late sink falls with slot surprisal | PASS, but small | −0.0014 / −0.0011 per nat; see below |
| Q5 slope steeper late than early | PASS | early slopes are slightly positive |
| Q6 controls, placebo | PASS | |

### What the data show

1. The combination matters: with " I like" before it, " sushi" adds late-layer sink beyond the sum of the three
   single tokens; with " like I" before it, it subtracts (common, −0.015) or adds less (rare, +0.004).
2. " sushi" alone at position 31 raises late-layer mean sink by +0.030 (common) / +0.022 (rare). In rare bases
   that single token carries almost all of the phrase's late effect (late mean 0.626 vs nl 0.628).
3. Slot surprisal is a weak account even though Q4 passed. The slope predicts +0.005 of the +0.040 order effect
   (common), and two variants go against it: " sushi"@31 is *less* predictable than the base tokens (+5.3 nats,
   common) yet raises sink; same-form random words in rare bases are *more* predictable (−5.5 nats) yet lower it.

### Hypotheses after iteration 2

- **H2 (additive token-position effects): rejected for the order effect** (r² ≈ 0).
- **H1 (scalar predictability): downgraded.** Right sign on average, but it explains about a tenth of the order
  effect and is contradicted by two variants.
- **H1′, phrase completion at q−1.** Late heads rest on BOS more when the token right before the query completes a
  coherent phrase with what precedes it, beyond what each token does alone.
- **H4, q−1 token identity.** Some tokens at q−1 (here " sushi") raise late sink on their own, whatever their
  predictability; exp1ext already showed q−1 identity has large per-head effects.
- **Open threat: scale.** sink_mass is bounded, so effects need not add on the probability scale. A pure saturation
  artifact should hit nl and shuffled alike (their parts are similar in size), yet their interactions have
  opposite signs in common bases; still, it should be checked on the logit scale.
- **Open threat: one phrase.** Everything so far is about " I like sushi".

### Decision

Iteration 3 generalizes: 12 three-token pronoun–verb–object phrases (shuffled = verb pronoun object, last token
kept, so the scrambled order is reliably ungrammatical), the same five single-token variants per phrase, fresh
bases (the exclusion list changes, so the base rows are new: " I like sushi" doubles as a replication on new rows),
500 rows per base, near only. Additivity on both sink_mass and logit(sink_mass). Phrases become the unit for the
pooled tests (bootstrap over phrases), and the across-phrase correlation between the order interaction and the
surprisal gap tests H1′ against H1 more directly.

## Iteration 3: 12 phrases, logit scale (near only)

- Config: `configs/exp2_iter3.yaml`; predictions: `exp2/PREDICTIONS_iter3.md` (R0–R6), recorded after iteration 2's
  results and before any iteration-3 data. Fresh bases (500 per class): excluding all 36 phrase ids changes the pool.
- Code added: combos with a `variant` field (one "nl" per phrase), `exp2.analysis.phrase_level` (phrases as the
  unit), sink_logit as an additivity metric, prediction kinds pooled_term_sign / phrase_count /
  phrase_spearman_below, `exp2.plots.plot_phrase_terms`, report §9 (every phrase listed).
- Process notes: the loop was paused once at the user's request (a report edit was declined); the user then asked
  for iteration 3 to run. The report shows every phrase in its summary tables; long per-head tables only for
  " I like sushi", with full per-head results in the saved parquet files. A 20-row smoke run (scratchpad) checked
  the pipeline after the predictions were written; its numbers were not used.
- Run: `results/20261004-125831_exp2` (detached with nohup; the first attempt, `results/20261004-124512_exp2`, was killed
  by the 10-minute background-task limit during TL extraction and is marked CRASHED.txt). Checks all PASS: `none` rows
  replicate exp1 (r = 0.9995 rare, 0.9997 common); TL vs HF max |Δ| 6.4e-05, labels 100%, meaningful decisions 100%.

### Prediction outcomes

| id | result | what happened |
|---|---|---|
| R0 sushi replicates on new rows | PASS | nl late +0.028 / +0.012, nl−shuffled +0.044 / +0.012 (common / rare); iteration 2 had +0.028 / +0.013 and +0.043 / +0.009 |
| R1 pooled nl interaction > 0 | PASS | +0.027 [+0.022, +0.032] common, +0.018 [+0.014, +0.022] rare |
| R2 pooled nl − shuffled > 0 | PASS | +0.025 [+0.020, +0.031] common, +0.010 [+0.008, +0.013] rare |
| R3 ≥ 8 of 12 phrases | PASS | 12/12 in both bases, none with the opposite sign |
| R4 holds on the logit scale | PASS | nl +0.15 / +0.11, nl−shuffled +0.14 / +0.06, all CIs exclude 0 |
| R5 order effect doesn't track the surprisal gap | PASS (weak, n = 12) | ρ +0.19 (p 0.56) common, +0.07 (p 0.83) rare. " he fixed bikes" has almost no gap (+0.2 / +0.8 nats) yet a clear order interaction |
| R6a controls | FAIL (marginal) | L4H11 prev_mass 0.949 in 1 of 240 cells (common, " play you chess"); floor 0.95 |
| R6b placebo empty | FAIL (marginal) | 2 of 3,456 placebo tests meaningful (rare: bikes L4H0 −0.024, rice L7H0 −0.021), just over the 0.02 bar with wide CIs. 500 rows per base (vs 1000 before) makes that bar less protective. Real total contrast: median 71.5 meaningful heads per phrase |

### Conclusions after three iterations

1. A short grammatical phrase written right before the query makes late heads (layers 6–11) rest on BOS more:
   12/12 phrases, both base classes, both libraries.
2. The effect is non-additive and order-sensitive. The phrase does more than the sum of its tokens placed alone, and
   the scrambled order (same tokens, same last token) does much less or nothing. It holds on the logit scale, so it
   is not a probability-ceiling artifact.
3. The model's surprisal of the phrase does not explain it: within rows the slope covers about a tenth of the order
   effect (iteration 2), and across phrases the order effect does not track the surprisal gap (iteration 3).
4. Position matters: far placements (14–16) give much smaller effects than near ones (iteration 1).
5. Link to exp1/exp1ext: the late heads that were diffuse on rare-token context and sink-bound on common-token
   context also move toward the sink when the context ends in a coherent phrase. Working hypothesis (H1′): late heads
   default to BOS ("no-op") when the recent context forms a locally coherent phrase, and stay active when it doesn't.
   Effects are small in absolute terms (late-layer mean +0.01 to +0.04 sink mass) but consistent.

### Open, not tested here

- Local bigram familiarity vs syntax: the scrambled orders contain rare bigrams (" like I", " need we"); a control
  with frequent-but-ungrammatical bigrams would separate the two.
- Mechanism: which earlier heads write the "phrase continues" signal into position 31 (path patching from that
  residual stream would test it).
- Other templates (adjective–noun, prepositional phrases), and gpt2-medium.

Loop stopped after iteration 3 (the cap).
