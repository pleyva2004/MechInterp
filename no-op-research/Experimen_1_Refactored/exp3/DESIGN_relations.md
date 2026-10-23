# Experiment 3, class 2: relational structure (country → capital, correct vs wrong)

Status: designed, not run (2026-10-15); revised after an adversarial review the same day. Predictions are written
before any attention data. Two pieces of work ran first, and neither computed attention:

- stimulus selection (tokenization, plus GPT-2's knowledge in a framed prompt);
- a manipulation-check pilot (next-token probabilities only).

Class 1 (injection form) is in `DESIGN.md`. Class 2 uses its measurement rules (§2), adapted to the grid in §3.

## 1. Idea

Put a country second-to-last (position 31) and a capital last (position 32, the final query).

- **Correct:** " France Paris" is a coherent fact.
- **Wrong:** " France Tokyo" uses the same kinds of tokens in the same places, with the relation broken.

Then ask which heads behave differently.

**Why it matters for the thesis.** In Exp 2, coherent phrases differed from their controls in *form*: a scrambled
order, or different words. Here every cell is a country followed by a place name, drawn from one fixed set of
tokens, and the analysis removes each token's own effect. What is left is the effect of the *pairing*.

**What the pairing effect is.** For a language model, "France goes with Paris" is learned from co-occurrence.
Truth and association cannot be fully separated. The design therefore measures both:

- a **capital grid**: every country with every capital;
- a **city grid**: every country with a famous non-capital city of each country (" France Lyon").

An own-country city is associated with its country about as strongly as the capital is (pilot below), but it is
not the capital. The capital effect minus the city effect is the part specific to the capital relation.

**Where the effect can show up.** Attention is causal, so a capital can only change attention at its own position or
after it. That gives two placements:

| placement | 30 | 31 | 32 (query) | question |
|---|---|---|---|---|
| **in** | r | country | capital / city | What does a head do when the token it sits on completes a fact, vs breaks it? |
| **out** | country | capital / city | r | Does a coherent pair right before an unrelated token make late heads park on BOS, as Exp 2's coherent pairs did? |

`r` is the base row's own random token.

## 2. Stimuli

Selection rules, fixed before any attention data:

- country, capital and city are each a single GPT-2 token with a leading space;
- ids lie in 1000–39999, so there are no byte tokens and the same-form machinery applies;
- GPT-2 small ranks the capital 1st among all candidate capitals after " The capital of {country} is";
- in that prompt it prefers the capital over the city;
- the city is the first listed candidate that passes these rules.

Results of the check:

- 26 countries were checked.
- Dropped because no city passed: Iraq (Mosul is preferred over Baghdad), Australia (Sydney and Melbourne are
  preferred over Canberra), Syria (Aleppo).
- Dropped because no city is a single token: Poland, Netherlands, Greece, Sweden, Denmark, Pakistan, Afghanistan,
  Iran, Thailand, Philippines, Chile.
- Dropped because GPT-2 does not know the capital (rank 12): Switzerland (Bern).

That leaves 11 countries:

| country | id | capital | id | city | id | framed log p(capital) − log p(city) |
|---|---|---|---|---|---|---|
| France | 4881 | Paris | 6342 | Lyon | 35193 | +4.24 |
| Germany | 4486 | Berlin | 11307 | Munich | 22418 | +1.90 |
| Italy | 8031 | Rome | 10598 | Milan | 21574 | +1.06 |
| Spain | 8602 | Madrid | 14708 | Barcelona | 15142 | +0.44 |
| Russia | 3284 | Moscow | 9070 | Petersburg | 27051 | +6.07 |
| Ireland | 7517 | Dublin | 18220 | Cork | 25567 | +2.74 |
| Ukraine | 7049 | Kiev | 22761 | Donetsk | 31819 | +0.94 |
| Japan | 2869 | Tokyo | 11790 | Kyoto | 36298 | +5.42 |
| China | 2807 | Beijing | 11618 | Shanghai | 21865 | +1.11 |
| Egypt | 6365 | Cairo | 23732 | Alexandria | 27872 | +3.16 |
| Canada | 3340 | Ottawa | 14074 | Montreal | 12871 | +0.36 |

The 11 dropped pairs that are otherwise eligible (Netherlands, Sweden, Denmark, Greece, Poland, Pakistan,
Afghanistan, Iran, Thailand, Philippines, Chile) never appear in the grids. They serve as R3's demonstrations.

**Bases.** Rare (1000–39999) and common (256–999) classes as in Exp 2, with all 33 stimulus ids and all demonstration
ids excluded from the filler. 200 rows per class (see §5 for power).

**Manipulation-check pilot.** 60 rows per class on separate random bases. Next-token probabilities at 31, with the
country at 31. The demonstration capitals are never among the candidates.

| base | frame | own capital − mean of the other 10 | own city − mean of the other 10 | own capital ranked 1st | own city ranked 1st |
|---|---|---|---|---|---|
| rare | bare | +2.43 nats (11/11 countries) | +2.21 (11/11) | 81% of rows | 73% |
| common | bare | +2.67 (11/11) | +2.49 (11/11) | 79% | 86% |
| rare | list | +2.19 (11/11) | +2.08 (10/11) | 76% | 57% |
| common | list | +2.48 (11/11) | +2.40 (10/11) | 81% | 65% |

What the pilot shows:

- GPT-2 small knows both relations inside random tokens.
- The city association is nearly as strong as the capital association (within 0.2 nats), so the city grid is a fair
  control for association.
- Two demonstration pairs do not strengthen either relation, so the list frame (R3) is secondary.

## 3. Measurement

**Per base row**, for each head and each metric (category masses at the query), the capital grid gives an 11 × 11
array A[country, capital]. The city grid gives A_city likewise.

**Removing token identity.** Double-centre each grid within the row: R = A − row mean − column mean + grand mean. This
removes the main effect of every country and every capital (or city).

- **Fact score** T = mean of the diagonal residuals R[j, j]. It is one number per row, and it is what is left when
  the tokens are held constant.
- **T_cap** comes from the capital grid and **T_city** from the city grid.
- **Capital-specific** = T_cap − T_city.

**Inference.** Two levels, and both must agree.

- **Rows:** Wilcoxon on T over the 200 rows, with a bootstrap CI over rows.
- **Items:** a permutation test over pairings. Relabel the capitals by 10,000 random permutations (the same
  permutation for every row) and recompute T. p is the share of permuted T at least as extreme as the observed T, in
  the pre-registered direction. This asks whether the correct pairing is special among all pairings, so item
  variance is accounted for.
- **Per item:** each country's diagonal residual, averaged over rows. The sign count is over 11 countries:
  ≥ 9/11 gives one-sided p = 0.033.

**Categories at the query** use position masks, identical in every cell:

- **in:** BOS, 31 (country = previous), 32 (self), filler.
- **out:** BOS, 30 (country), 31 (capital/city = previous), self, filler.

"Late" means layers 6–11, pooled as the mean over their 72 heads. Per-head tests use BH-FDR over the 144 heads, for
one metric, one contrast and one base class.

**Role switch** (DESIGN.md §2 rule 2). Per row, take labels on attention averaged over the 11 diagonal cells (the
correct arm) and over the 110 off-diagonal cells (the wrong arm). The key positions line up because the masks are
positional. That gives each row one label per arm. Run a paired McNemar test with BH-FDR, and require a shift of
≥ 15 percentage points in the share of rows carrying the destination label.

**Placebo.** Draw 20 random derangements in advance, with a fixed seed. Score each as if it were the correct pairing
(its 11 cells against the other 110 wrong cells). Expect ≤ 2 of the 20 to reach permutation p < 0.05 on late BOS,
and a mean of ≤ 1 switching head. Role-switch predictions count heads beyond that placebo mean.

**Three-way decision rule** for every pooled score, with margin m = 0.005 (about half the smallest pooled late effect
in Exp 2, which was +0.009):

| verdict | condition |
|---|---|
| **direction** | the row CI and the item permutation test both exclude 0 on the same side |
| **equivalent** | the 90% row CI lies inside ±m |
| **inconclusive** | anything else; no claim is made |

A prediction passes only on "direction" with the pre-registered sign. Every prediction must hold in both base
classes. If the base classes disagree, the per-class results are reported and no general claim is made, which is what
happened in every Exp 2 run.

**Association covariate.** Use the bare-context association of each cell: log p(capital | BOS country), computed
once per cell. Among off-diagonal cells only, regress the late-BOS residual on it. Variation among the wrong pairs
identifies how much association alone moves heads. Correct cells are left out, because by construction they are
always the most expected.

**Dose within a cell.** For a fixed correct cell, the capital's surprisal still varies across rows with the random
context. Compute the Spearman correlation between surprisal and late BOS across rows, within each correct cell, then
pool over the 11 cells. This is identified, unlike a model across cells, where correctness and surprisal are almost
collinear.

**Figures.**

- `attention_maps.py`-style grids: columns correct / wrong / own city / anchor; rows are mean BOS, the most-attended
  category, and Δ vs wrong.
- A per-head scatter of Δ BOS against Δ country mass.
- The 11 × 11 residual grid of late BOS, for the capitals and the cities.

## 4. Experiments

### R1-in: the capital is the query (the experiment as proposed)

Variants per row:

- capital grid: 121 cells (11 correct, 110 wrong);
- city grid: 121;
- **anchor**: country at 31 and a same-form random word at 32, matched to that country's capital through
  `exp2.design.matched_pools` and drawn per row: 11;
- capital alone (r at 31): 11;
- none: 1.

**Direction.** The directional prediction follows `DESIGN.md` E1b, which says that a query completing a phrase lowers
sink because heads read the context. Exp 2 never put the query inside the phrase, so the opposite reading (H-rest: a
completed fact leaves nothing to do, so the head parks) is an extrapolation of Exp 2. It is listed as the
alternative.

**Predictions** (each in both base classes):

- **Rin-1:** late BOS T_cap < 0. A correct capital takes heads off BOS.
- **Rin-2:** the BOS mass lost goes to the country: T_cap on position-31 mass > 0, with at least half the size of
  Rin-1. Self and filler change by less than m.
- **Rin-3:** at least 3 heads beyond the placebo mean switch Sink → Previous (country) on correct vs wrong cells.
- **Rin-4 (capital-specific, low confidence):** late BOS T_cap − T_city < 0. Heads respond to the capital relation
  beyond country–city association.
- **Rin-5 (association):** among wrong pairs, the late-BOS residual falls as association rises (slope < 0). More
  associated wrong pairs look more like correct ones.
- **Rin-6 (generalization):** ≥ 9/11 countries' diagonal residuals have Rin-1's sign.
- **Rin-7 (dose):** within correct cells, rows where the capital was more expected have lower late BOS (pooled
  Spearman between surprisal and BOS > 0, CI excluding 0).
- **Rin-8 (which arm moved; descriptive):** correct − anchor and wrong − anchor, each judged by the three-way rule.
  Both arms have a country at 31 and a capitalised word at 32.
  - If correct departs from the anchor and wrong does not, the fact changes heads.
  - If wrong departs and correct does not, the contradiction does.

### R1-out: the pair is followed by a random query (link to Exp 2)

Variants per row: the capital grid and the city grid at 30–31 (242 cells), plus none (1).

**Predictions** (both base classes). These come from Exp 2's working hypothesis: late heads park when the tokens
before the query form a coherent pair. Note that association alone predicts the same sign.

- **Rout-1:** late BOS T_cap > 0.
- **Rout-2:** late BOS T_city > 0. A coherent pair need not be a capital fact.
- **Rout-3 (generalization):** ≥ 9/11 countries share Rout-1's sign.
- **Rout-4:** the placebo, as in §3.

### R3: list frame (secondary; run only if R1-in gives "direction" on Rin-1)

Two demonstration pairs at 27–30, drawn per row from the 11 dropped pairs, so no token repeats. Then the
capital-grid cell at 31–32.

**Question:** with a pattern on screen, do heads at the capital look at the demonstration capitals (28, 30), and does
that depend on the test pair being correct?

**Prediction R3-1:** T_cap on the mass at positions 28 + 30 is > 0. Among the late heads, at least 3 beyond the
placebo mean pass the per-head test.

### R4: other relations (later)

The grid design works for any set of (x, y) pairs where yⱼ is xⱼ's correct partner. Candidate relations follow; their
tokens are not yet verified, and the selection rules of §2 apply:

- country → nationality adjective;
- day → next day;
- number → next number;
- antonyms.

Each would separate factual, sequential and lexical structure.

## 5. Size, power, order

| run | cells per row | rows per base class | sequences (2 bases) |
|---|---|---|---|
| R1-in | 265 | 200 | 106k |
| R1-out | 243 | 200 | 97k |
| R3 | 121 | 200 | 48k |

- **Engines:** TL for all rows, attention and logits. HF runs on identical ids for a 50-row subset per class, as a
  cross-library check; earlier runs agreed to < 1e-4.
- **Power:** before launch, simulate the power to detect T = 0.01 on late BOS, using iteration 3's per-head SDs and
  the 11-cell averaging. If power is below 80%, raise the row count.
- **Order (updated 2026-10-15):** R1-out runs first, as the main run. It keeps the final token as the row's own
  random token, the rule every earlier experiment followed, so correct and wrong cells differ only in context.
  R1-in follows. Every contrast there holds the final token fixed (same capital, different country at 31), and the
  grid's row and column balance cancels each capital's own query effect. Then R3 if triggered, then R4.
- **Power (computed before launch):** iteration 3's late-head per-row SDs (order contrast, one-cell differences:
  median 0.038 rare / 0.054 common) give SD(T) ≈ 0.009 / 0.013 per row after the 11-vs-110-cell averaging. At 200
  rows, power is > 99% for a pooled T of 0.01, and for a per-head T of 0.02 at Bonferroni-144. The binding limit is
  the 11 items (permutation test), which row count does not fix.

## 6. Code changes

- **Shared with class 1:** writes at the query position (`allow_query`).
- **`exp3/relations.py` (new):**
  - the stimulus table;
  - the grid builder, with checks that every cell has the right ids at 31/32 (or 30/31), that filler excludes
    every stimulus and demonstration id, and that no row repeats a token;
  - double-centring and the fact scores;
  - the pairing-permutation test, the derangement placebo, the association covariate and the within-cell dose;
  - the role-share switch test.
- **Also new:** `configs/exp3_relations.yaml`, `exp3/PREDICTIONS_relations.md`, and `tests/test_exp3_relations.py`.
  The tests include double-centring on a synthetic grid with a known diagonal effect, and the permutation test's
  null calibration.

## 7. What the outcomes mean for the thesis

| outcome | reading |
|---|---|
| Rin-1 and Rin-2 pass | On a token that completes a fact, heads stop resting and read the country: the no-op is lifted when the current token has something to bind. This agrees with E1b |
| Rin-1 opposite sign (T_cap > 0) | A completed fact makes heads park (H-rest), so completion and an unrelated query both lead to rest; E1b's logic fails |
| Rin-4 passes | Heads separate "capital of" from mere association: relational knowledge reaches attention |
| Rin-4 equivalent | Heads respond to association (co-occurrence) and treat "France Lyon" like "France Paris" |
| Rout-1 and Rout-2 pass | A coherent pair before an unrelated token frees later heads, whether it is a fact or an association, which extends Exp 2 beyond syntax |
| Rout-1 and Rout-2 equivalent | Late-head parking does not respond to relational coherence; Exp 2's effect is about syntactic form |
| Rin-7 passes | The effect scales with how strongly the model expected the capital in that context. This is how the knowledge acts, not an alternative to it |
| any inconclusive, or base classes disagree | Reported as such; no claim |

**Limit.** For an LM, truth and co-occurrence are entangled. The city grid bounds that confound but does not remove
it: in bare context the capital is still slightly more expected than the city (+0.2 nats in the pilot).

## 8. Changes after review

- **Stimuli and contrasts:** 16 region-split pairs became 11 country / capital / city triples, and the four fixed
  permutations became full 11 × 11 grids with double-centring. A city grid was added as the association control.
- **Item-level inference:** a pairing-permutation test, with random-derangement placebos. Before, items were treated
  as fixed.
- **Decision rule:** a three-way rule. Rin-1 now has a sign consistent with E1b, where before "CI excludes 0" could
  pass either way.
- **Surprisal:** the identified within-cell dose (Rin-7) replaced the across-cell surprisal model, which was
  collinear with correctness.
- **Anchor:** a same-form anchor with the country at 31 was added, to tell which arm moved.
- **R3:** demonstrations now come from pairs outside the grids, and in-context capitals are removed from the pilot's
  candidate set.
- **Smaller fixes:** role switch made well defined with 11 cells per row (labels on arm-averaged attention, as in
  DESIGN.md); per-base-class readings; "none" counted once.
