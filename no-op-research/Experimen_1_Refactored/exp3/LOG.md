# Experiment 3 log

## Setup
- Designs: `DESIGN.md` (class 1: injection form) and `DESIGN_relations.md` (class 2: relational structure). Both were
  revised after adversarial reviews on 2026-10-15, before any data.
- Predictions were pre-registered in `PREDICTIONS_*.md` and checked automatically.
- Model: GPT-2 small. Attention comes from TransformerLens; Hugging Face recomputes a subset as a cross-check.

## Process notes
- **Role-switch rule (2026-10-15, before E3.2 ran).** The rule "row share rises by ≥ 0.15" compared floats
  exactly, so a rise of exactly 15 points (e.g. 30/200 rows) could round to 0.1499… and be missed. Both
  `engine.role_switches` and `relations.role_switches` now compare with a 1e-9 tolerance. Runs before E3.2 used the
  strict comparison.
  - Every earlier run was rechecked from its saved `labels.npz`. 11 contrasts gain one head (two in one city
    contrast), all exactly at the threshold.
  - None is a predicted switch type. Rin-3: the rare placebo mean goes from 0.45 to 0.50, still FAIL.
  - No prediction verdict changes.
- **E3.5's previous-token check** was registered more strictly than DESIGN ("Previous in every variant" vs "a change
  flags a confound"). It was left as registered, and its result is read as a confound flag (see E3.5).

## Class 2, R1-out: country (30) + capital or city (31), with the row's own random token as the final query

Run `results/20261015-190957_exp3rel`. 200 rows per base class. Each row has 2 × 121 grid cells plus none.

**Checks: all PASS.**

| check | result |
|---|---|
| determinism | exact |
| TL vs HF | max \|Δ\| 2.6e-5 |
| manipulation | 11/11 capitals preferred in context, both base classes |

**Predictions** (1/4 held):

| id | prediction | result | values |
|---|---|---|---|
| Rout-1 | correct capital raises late BOS | FAIL | equivalent on rare (−0.0009); opposite sign on common (−0.0037) |
| Rout-2 | correct city raises late BOS | FAIL | holds on rare (+0.0057); equivalent on common (+0.0009) |
| Rout-3 | ≥ 9/11 countries positive | FAIL | 3/11 rare, 2/11 common |
| Rout-4 | placebo stays quiet | PASS | ≤ 2/20 derangements significant; 0 switches |

**What happened:**
- Any country + place pair before the query raises late BOS a lot relative to none: +0.013 rare, +0.05 common. The
  *pairing* (true vs false) adds almost nothing to BOS.
- The robust pairing effect is a redistribution inside the pair, and it holds in all four base × grid cells. With a
  correct pair, late attention moves from the partner (31, q−1) to the country (30): partner −0.004 to −0.008,
  country +0.001 to +0.004.
  - Heads that gain country mass: L9H8, L6H8, L9H3, L5H6.
  - Heads that lose partner mass: L9H8, L1H0, L4H0, L7H0.
- Among wrong pairs, more associated pairs carry slightly more BOS (positive slope in every cell). Even so, the
  correct capital carries less BOS than the city does (capital − city −0.0065 rare, −0.0047 common).
- No role switches. Effects are 0.004–0.008 of mass pooled and up to 0.07 per head, too small to change labels.

**Reading.** Exp 2's rule ("a coherent pair right before the query makes late heads park") does not extend to factual
coherence. The model does register truth in attention, but by shifting weight within the pair, not by changing
no-op/BOS mass.

## Class 2, R1-in: country (31) + capital or city AT the final query (32)

Run `results/20261015-203150_exp3rel_in`. 200 rows per base class. Each row has the grids, none, 11 anchor cells
and 11 capital-alone cells.

**Checks: all PASS.**

| check | result |
|---|---|
| determinism | exact |
| TL vs HF | max \|Δ\| 5.0e-5 |
| manipulation | 11/11 |

**Predictions** (3/7 held):

| id | prediction | result | values |
|---|---|---|---|
| Rin-1 | correct capital lowers late BOS | **PASS** | −0.018 [−0.019, −0.017] rare, −0.034 [−0.035, −0.032] common; perm p < 0.0001 |
| Rin-2 | the freed BOS mass goes to the country | FAIL | country gains only +0.002 / +0.006; filler gains +0.015 / +0.029 |
| Rin-3 | ≥ 3 Sink→Country switches beyond placebo | FAIL | 3 (rare) and 4 (common) switches against placebo means of 0.45 / 0.70, i.e. 2.55 / 3.3 beyond placebo; rare is short |
| Rin-4 | capital lowers BOS more than city | FAIL | −0.018 on common; equivalent on rare (−0.0026) |
| Rin-5 | more associated wrong pairs carry less BOS | PASS | slope < 0 in both base classes |
| Rin-6 | ≥ 9/11 countries negative | PASS | 11/11 in both |
| Rin-7 | within correct cells, a more expected capital means less BOS | FAIL | sign reversed: ρ(surprisal, BOS) −0.28 rare, −0.23 common |

- The city grid shows the same drop: −0.015 / −0.016.

**What happened:**
- On a token that completes a known relation, late heads come off BOS. The drop is 2–4× the size of Exp 2's
  coherent-phrase BOS rise, in the opposite direction, and it holds for capitals and own-country cities alike.
- The freed mass spreads over the random context (filler), not specifically onto the country. Heads broaden; they
  do not lock onto the relation token.
- The effect is graded across pairings: more associated wrong pairs also lower BOS (Rin-5).
- Within a fixed correct pairing, rows where the context made the capital *less* expected also have *less* BOS.
  Across pairings, expected means less BOS; within a pairing, unexpected means less BOS. BOS is not a monotone
  function of predictability. The within-cell variation is driven by the random context, which can move both
  quantities, so this is not causal.
- The anchor cells (a country followed by a random capitalised word) carry far more BOS than any place-name query:
  −0.08 to −0.14 relative to correct, wrong and capital-alone. This compares different query tokens, so it is
  descriptive only.

**Reading.** Completion at the query lifts the no-op, as `DESIGN.md` E1b predicted for phrases. Factual coherence
*before* an unrelated query does not raise it (R1-out).

## Class 2, position sweep: country at i, capital at j, every 1 ≤ i < j ≤ 32

Run `results/20261015-210238_exp3pos`. 496 placements; 50 rows per base class; 11 correct, 22 wrong and 11 placebo
cells per placement and row.

**Checks: all PASS.**

| check | result |
|---|---|
| determinism | exact |
| TL vs HF | max \|Δ\| 4.3e-5 |
| (30, 31) vs R1-out | same sign (−0.0007 / −0.0032 here; −0.0009 / −0.0037 in R1-out) |
| (31, 32) vs R1-in | same sign (−0.015 / −0.033 here; −0.018 / −0.034 in R1-in) |

**Predictions** (3/5 held):

| id | prediction | result | values |
|---|---|---|---|
| S1 | capital at the query lowers late BOS for every country position 26–31 | PASS | −0.015 to −0.033 |
| S2 | that effect weakens with distance | FAIL, **reversed** on rare | the farther the country, the *larger* the drop: rare −0.034 at i = 1 vs −0.015 at i = 31; common −0.039 at i = 1, −0.022 at i = 30, then −0.033 at the adjacent i = 31 |
| S3 | no BOS effect before a random query (≤ 10% of j ≤ 31 placements significant) | FAIL on rare | rare 72/465 (15.5%), common 4/465 |
| S4 | within-pair shift (country up, capital down) for adjacent pairs near the query | PASS | grows toward the query: +0.002 at (27, 28) to +0.006 / +0.012 at (30, 31) |
| S5 | placebo significant in ≤ 5% of placements | PASS | 0/496 in both base classes |

**What happened:**
- **Completion at the query works over the whole context.** With the capital as the final token, every one of the 31
  country positions gives a significant BOS drop, in both base classes. About 70 of 144 heads are meaningful at
  (1, 32). The drop is largest when the country is farthest away. The relation is retrieved across the whole window,
  and a distant match releases parking at least as much as an adjacent one.
- **Before a random query, a correct pair slightly *lowers* late BOS rather than raising it.**
  - Mean −0.002 to −0.004 across the triangle.
  - Strongest when the country is early and the capital is late (rare: up to −0.007 with the capital at 31 and the
    country at 1–5). It approaches 0 when the pair is adjacent (30, 31), which is R1-out's placement.
  - The significant placements on rare bases cluster at j = 26–31 with i ≤ 20.
  - Exp 2's "coherent pair → park" does not hold for factual pairs at any placement.
- **Attention to the capital itself drops when the pairing is correct** (common bases: −0.006 to −0.008 whenever the
  capital sits at 31, for every country position). **Attention to the country rises when the pair is near the
  query.**

## Class 1, E3.4: repeated phrase (the induction positive control)

Run `results/20261015-211838_exp3e34`. 500 rows per base class; 8 variants × 12 phrases.

**Checks: all PASS.**

| check | result |
|---|---|
| determinism (20 rows) | exact |
| TL vs HF | max \|Δ\| 3.6e-5 |
| placebo | 0 switching heads |
| L5H5 and L6H9 among the induction heads | yes |

| base | induction heads (copy-target mass ≥ 0.3) |
|---|---|
| rare | 10 |
| common | 4 |

**Predictions** (3/5 held):

| id | prediction | result | values |
|---|---|---|---|
| E4a | heads switch Sink → Copy beyond placebo | **PASS** | 7 (rare) and 5 (common), placebo 0: L5H0, L5H1, L5H5, L6H9, L7H1, L7H2, L7H10 |
| E4b | induction heads take more from BOS than proportional renormalization | PASS | excess +0.085 (rare), +0.040 (common) |
| E4c | coherent content does not suppress copying (non-inferiority to repeat-matched) | FAIL | copy mass on the NL phrase is 0.11 (rare) and 0.18 (common) *below* same-form random words; 0/12 phrases above. Against random ids, NL is equal (rare) or higher (common, +0.066) |
| E4d | Test A's top-10 "parking" heads leave BOS on repeats | FAIL | 3/10 and 4/10 lose BOS; random sets of other late heads average 7/10 |
| E4e | a two-token match adds to a one-token match | PASS | +0.23 / +0.21 copy mass, 12/12 phrases |

**What happened:**
- The metric detects a known switch, so the role metrics are frozen from here on (`DESIGN.md` E3.4) and the fallback
  is not needed.
- Copying draws on BOS beyond renormalization: BOS is where these heads wait when there is nothing to copy.
- Natural-language content reduces induction copying relative to same-form words. A plausible reading is that a
  predictable continuation needs less copying.
- The heads that parked on BOS for coherent phrases in Exp 2 are *less* involved in the repeat response than typical
  late heads. Parking and copying use partly different head populations.

## Class 1, E3.3: distance sweep (phrase ending at 31 − g)

Run `results/20261015-214541_exp3e33`. 200 rows per base class.

**Checks: all PASS.**

| check | result |
|---|---|
| determinism | exact |
| TL vs HF | max \|Δ\| 2.3e-5 |
| placebo | 0 switching heads |

**Predictions** (3/3 held):

| id | prediction | result | values |
|---|---|---|---|
| E3a | the order effect at g = 1 is at most half of g = 0 | PASS | rare +0.0128 → +0.0049; common +0.0235 → +0.0101 |
| E3b | the order effect is equivalent to 0 at g = 24 | PASS | +0.0003 rare, +0.0004 common |
| E3c | the total effect at g = 15 is still positive (iteration 1's far slot) | PASS | +0.0060 rare, +0.0040 common |

**What happened:**
- **Order is local; content is not.**
  - The order effect (nl − shuffled) falls by more than half per step for the first 1–2 steps. By g ≥ 3 it is a
    small residual (+0.001–0.006), and it is gone by g = 24.
  - The total effect (nl − none) stays at +0.006–0.009 on rare bases out to g = 15. "Real words somewhere in
    context" acts at long range, while "coherent order next to the query" acts only locally.
- **Distance, not absolute position.** 24-token rows with the phrase at 21–23 (g = 0) give the same order effect as
  32-token rows at g = 0 (rare +0.0130 vs +0.0128), and far more than g = 8 at 32 tokens.
- **Role switches at g = 0, total effect (beyond a placebo of 0).**
  - Late heads switch Previous → Sink: L4H3, L5H6, L6H0, L6H8, L7H0. When the previous token is a word that belongs
    to a phrase, they stop reading it and rest.
  - Early heads switch Span → Previous: L2H2, L2H5, L2H9, L3H2, L3H6, L3H7.
  - L2H8 goes the other way, Previous → Span (0.6–0.7 of rows): it reads the verb one step back instead of the object.
  - The switches vanish by g ≈ 2.
- **Long-range heads.**
  - Order effect: L10H8 and L11H5.
  - Total effect: L9H2, L10H1, L11H6, L10H2, L10H10, with half-distance > 24.

## Class 1, E3.5: punctuation at 29 (phrase at 26–28, random tokens at 30–31)

Run `results/20261015-220224_exp3e35`. 300 rows per base class.

**Checks:**
- Design, determinism, TL vs HF (max \|Δ\| 3.4e-5) and the placebo (0 switching heads) all PASS.
- The registered check that Exp 1's previous-token heads keep Previous in every variant FAILS in both base classes:
  - L2H5 switches Previous → Delimiter (0.6–0.76 of rows) whenever "." or "," is present, with or without the phrase.
  - On common bases, L1H0 and L3H8 are not mostly Previous even in `none`. That part of the check fails at
    baseline; it was left as registered.

**Predictions** (2/3 held):

| id | prediction | result | values |
|---|---|---|---|
| E5a | after a phrase, a period draws mass and ≥ 5 late heads switch Sink → Delimiter | FAIL | the interaction on delimiter mass is ≈ 0 (rare +0.0008) or small (common +0.0031); 1 and 0 late heads switch |
| E5b | resting (BOS + delimiter) does not shrink | PASS | non-inferior |
| E5c | a period draws more than the byte controls "x" and "7" | PASS | +0.012 / +0.008 rare, +0.012 / +0.007 common; 12/12 phrases |

**What happened:**
- In GPT-2 small at these positions, punctuation does **not** replace BOS as the late heads' resting place:
  - A period three tokens back draws +0.007 (rare) and +0.013 (common) of late attention, with or without a phrase
    before it.
  - Late BOS moves by about ±0.01, in opposite directions on rare (−0.008) and common (+0.010) bases.
- The delimiter response is an **early-head** behaviour: L2H5 (and L3H8 on common bases) acts as a "most recent
  punctuation" head. This is also what the failed previous-token check caught.
- "Real words at 26–28 vs random" raises late BOS for every d (+0.006 to +0.017). That replicates Exp 2's phrase
  effect at gap 3.
- "," after a phrase *lowers* delimiter mass on rare bases (interaction −0.009). It is not a symmetric partner of ".".

## Class 1, E3.1: query inside vs outside the phrase

Run `results/20261015-221059_exp3e31`. 500 rows per base class; 13 variants × 12 phrases.

**Checks: all PASS.** Placebo: 0 switching heads.

**Predictions** (3/6 held; per base class):

| id | prediction | result | rare | common |
|---|---|---|---|---|
| E1a | sanity: PVO+r − rr+r late BOS > 0 (replicates iteration 3) | PASS | +0.029 | +0.033 |
| E1b | the interaction C(O) − C(r) < 0 | FAIL | −0.055 | −0.0004 (equivalent) |
| E1c | ≥ 5 late heads switch Sink → Span/Previous | FAIL | 1 (L6H0 → Previous) | 0 |
| E1d | order × query on phrase-position mass > 0 | FAIL | n.s. | −0.007 |
| E1e | completion: C(O) − C(M) < 0 | **PASS** | −0.054 | −0.026 (10–11/12 phrases) |
| E1e-bet | plausibility: C(O) − C(O′) < 0 | PASS | −0.037 | −0.027 (9/12) |

**Context effect C(x) = PV+x − rr+x on late BOS, by query x:**

| query x | rare | common |
|---|---|---|
| r (the row's random token) | +0.012 | +0.009 |
| M (same-form random word) | +0.010 | +0.034 |
| O′ (implausible object) | −0.007 | +0.035 |
| **O (completing object)** | **−0.044** | **+0.008** |

**What happened:**
- A coherent context raises late BOS when the query does not continue it, and much less (rare: it reverses) when
  the query completes it.
- The completion contrast is negative in both base classes: against a same-form word (E1e) and against an
  implausible object (E1e-bet).
- E1b fails only on common bases, because the random-token query baseline C(r) is itself small there.
- The freed mass goes mostly to filler (rare C(O): filler +0.028), with small gains on previous and self. As in
  R1-in, there are no late role switches: the change is spread across heads, not a few heads flipping job.
- This agrees with R1-in (country → capital): when the token at the query completes what came before, late heads
  rest less than when it does not. Exp 2's "coherent context → park" holds for unrelated queries only.

## Class 1, E3.2: span length, from one word to natural text

Run `results/20261015-223621_exp3e32`. 200 rows per base class; 12 hand-written passages.

**Checks: all PASS.** Two placebos, 0 switching heads each.

**Predictions** (1/4 held):

| id | prediction | result | values |
|---|---|---|---|
| E2a | saturation by k = 3 | FAIL | no saturation (see below) |
| E2b | nl − shuffled > 0 at every k ≥ 3 | PASS | grows with k: +0.006 → +0.117 (rare), +0.021 → +0.114 (common) |
| E2c | natural next token vs same-form wrong word, crossed with context: interaction < 0, plus ≥ 5 late Sink → Span switches | FAIL | the interaction holds strongly (−0.063 rare, −0.062 common; 12/12 passages); 0 late Sink → Span switches |
| E2d | long context without the final pair carries ≤ 50% of the k = 31 effect | FAIL | it carries more than half |

**E(k) = nl − none, late BOS:**

| base | k = 1–4 | k = 8 | k = 12 | k = 16 | k = 24 | k = 31 |
|---|---|---|---|---|---|---|
| rare | ≈ 0 | +0.021 | +0.043 | +0.058 | +0.092 | +0.133 |
| common | +0.014 → +0.043 | +0.074 | +0.092 | +0.105 | +0.135 | +0.159 |

**What happened:**
- **Parking grows with the amount of coherent text and does not saturate.**
  - E(k) keeps rising roughly linearly up to a fully natural context.
  - The saturation fit puts κ ≈ 16 tokens on common bases and fails to saturate at all on rare bases (κ on the
    bound).
  - Same-form random words do nothing comparable: nl − matched = +0.21 / +0.23 at k = 31.
  - The effect needs real, ordered language (E2b).
- **The final pair is not required.**
  - With the passage's last two tokens put back to random (passage-rr), most of the k = 31 effect remains (E2d
    fails, 0/12 passages).
  - Test A's "the pair before the query does most of it" holds for 3-token phrases, not for long text.
- **Completion releases parking at full length too.** With a natural context, the true next token at the query
  gives 0.06 less late BOS than a same-form wrong word, relative to the same contrast without context. The released
  mass is spread out (0 late heads switch Sink → Span).
- **Late role switches accumulate with length.** Many late heads switch Diffuse/Span → Sink in nl_k vs none, and a
  large share first switch only at k ≥ 8–24 (switch-length table in the report).
