# Test B: how does a coherent phrase make late heads fall back to BOS?

Status: designed, not run. Predictions below were written 2026-10-04, before any Test B data.

## 1. The question this project is asking

Experiment 1's goal was to "characterize attention behavior on random-token input and establish a quantitative
no-op baseline": at the final token, which heads do real work, and which park their attention on BOS (the "sink",
a no-op)? Everything since asks what moves a head between those two states. The specific claim we are building
toward: a late head's BOS attention is not a fixed default; it is set by the context, and certain contexts switch
heads into the no-op state.

## 2. What we found, step by step

| step | what we did | what we learned |
|---|---|---|
| Exp 1 | 1000 random-token sequences, rare (ids 1000–39999) vs common (256–999) vs with byte tokens; TL and HF | 74% of heads park on BOS at the final token, and more so in later layers. Rare vs common tokens barely change the share of BOS heads, but individual heads shift in opposite directions: late heads are more active on rare tokens, early/mid heads on common ones |
| exp1ext | 2×2 of context vs final token, how many context tokens must change, single-token swaps | The late-head shift comes from the whole context, not the final token. One changed token barely matters, except at position 31, right before the final token |
| Exp 2, iteration 1 | " I like sushi" written into random context, near (29–31) or far (14–16), with scrambled and same-form controls | Near the final token the phrase makes layer 6–11 heads park on BOS more. Word order matters, specific tokens matter more than surface form, and far placement does little |
| Iteration 2 | each phrase token alone, plus the model's surprisal of the phrase | The phrase does more than its tokens added up; the scrambled order doesn't. Single-token effects explain none of the order effect (r² ≈ 0). Surprisal explains about a tenth |
| Iteration 3 | 12 pronoun–verb–object phrases, logit scale | Holds for 12/12 phrases in both base classes, and on the logit scale (not a ceiling artifact). The size doesn't track surprisal |
| Test A | word pairs: PV " I like", VO " like sushi", PO " I … sushi" | The effect is mostly pairwise between adjacent words. The pair right before the final token (VO) carries about half, PV a quarter to a third, PO and the three-word term little |
| Test C | adjective–noun–verb and preposition–determiner–noun, 12 phrases each | Generalizes to all three templates (pooled CIs exclude 0 in all six template × base cells). Weak order effect for adjective phrases on rare bases; which base class shows more depends on the template (unexplained) |

**Working hypothesis now:** late heads park on BOS when the tokens right before the final token form a locally
coherent pair, the pair ending at position 31 most of all. It is not explained by how surprising the tokens are.

## 3. Why Test B matters

Everything above is behavioural: we change the input and watch the attention. We know *which* inputs switch late
heads into the no-op state, but not *how*. The no-op thesis needs that answer, for three reasons:

1. **Default, or computed?** If a specific upstream circuit detects the coherent pair and that signal is what makes
   late heads park, the no-op state is computed: the model decides a head has nothing useful to do. If no
   localized path exists, the effect is diffuse, and "no-op" is closer to a passive fallback.
2. **"Nothing to look at" vs "nothing to ask".** BOS's own key cannot change (causal masking: BOS never sees later
   tokens), so more BOS attention means either the final token's *query* changed, or the phrase tokens' *keys*
   became less attractive. Key side would mean the phrase tokens advertise themselves as already handled. Query side
   would mean the final token itself stops asking. These are different stories about what a no-op head is.
3. **A causal handle for later experiments.** Identified writer heads can be switched off to remove the no-op shift
   (or patched in to create it). That is the tool Experiment 3 (patching, natural-language injection) needs.

## 4. Design

Reuses the exp2 pipeline: the same bases, phrases and variant builder; TransformerLens only, since HF has no hooks
(iterations 1–3 and Tests A/C showed TL and HF agree to < 1e-4).

- **Rows:** 200 rare + 200 common base rows; the 12 pronoun–verb–object phrases from Test A. If time is tight,
  6 phrases with the largest VO effect, chosen from Test A's saved tables before running.
- **Clean contrast:** "pair" = VERB@30 + OBJ@31 vs "object alone" = OBJ@31, with the base's random token at 30. The
  two runs differ only at position 30.
- **Target heads:** from the saved Test A + C per-head tables (no new run), the late heads (layers 6–11) whose VO
  interaction is positive with q ≤ 0.05 in at least 2/3 of phrases in both base classes. Listed in the config before
  running.
- **Metric:** the target heads' mean `sink_mass` at the final token. Effect = pair − object alone.
  Recovered fraction of a patch = (patched − object alone) / (pair − object alone).

**Steps**

1. **Residual sweep.** For every layer L and position p in {30, 31, 32}, copy the pair run's `resid_pre` at (L, p)
   into the object-alone run and record the recovered fraction. This shows where the signal sits at each depth.
2. **Query vs key.** For the target heads only, patch `hook_q` at position 32, or `hook_k` (and `hook_v`, as a
   check) at positions 30–31, from pair into object-alone.
3. **Writer heads.** At the position step 1 points to (expected: 31), patch each early head's output (`hook_z`,
   layers 0–5) one at a time, and rank heads by recovered fraction.
4. **Ablation.** Mean-ablate the top 3 writer heads in the pair run (their mean output over object-alone rows) and
   measure how much of the effect remains. Repeat with 3 random early heads as a control.
5. **Secondary contrast.** Repeat steps 1–2 for nl vs shuffled (they differ at 29–30) to check the same path carries
   the order effect.

## 5. Predictions (pre-registered)

Each must hold in both base classes.

1. **B1: the signal moves from 30 to 31 early.** Patching position 31 recovers ≥ 50% of the effect from layer 4 on,
   while patching position 30 recovers < 50% from layer 4 on.
2. **B2: key side over query side.** Patching the target heads' keys at 30–31 recovers more than patching their
   query at 32.
3. **B3: previous-token heads write it.** The top 3 early writer heads at position 31 include at least one head
   labelled Previous in Exp 1 (L1H0, L2H2, L2H3, L2H5, L2H8, L2H9, L3H2, L3H6, L3H7, L3H8, L4H11).
4. **B4: the writers are necessary.** Ablating the top 3 writer heads removes ≥ 50% of the effect; ablating 3 random
   early heads removes < 20%.
5. **B5 (sanity).** Patching position 30 at layer 0 recovers ~100% (the only difference between the runs), and
   patching an untouched position (10) recovers ~0%.

## 6. What the outcomes would mean for the thesis

| outcome | reading |
|---|---|
| B1–B4 pass | A small early circuit (previous-token heads at 31) computes "the pair is complete", and late heads respond by parking on BOS: the no-op state is computed, key-driven |
| B1 passes, B2 fails (query side) | The signal reaches the final token's own state; late heads stop asking rather than finding nothing to read |
| B3/B4 fail (no small writer set) | The signal is distributed. Late-head no-op is a broad response to context statistics rather than a dedicated circuit; Experiment 3 should use residual-level patching, not head ablation |
| B1 fails (signal stays at 30 or spreads) | The pair ending at 31 is not special mechanistically, and Test A's pairwise result needs another explanation |

## 7. To implement

- `exp2/patching.py`: TL hook-based patching (`resid_pre`, `hook_q` / `hook_k` / `hook_v`, `hook_z`) and
  mean-ablation over paired runs, reusing `exp2.design` and `exp2.run.build_design`.
- `configs/exp2_testB.yaml`: rows, phrases, target heads (filled from the saved tables before the run), the
  predictions above.
- Scale: roughly 3 positions × 12 layers + 2 q/k sets + 72 heads + ablations ≈ 120 patched forward passes per row
  set. Estimate before launch; run detached if over 10 minutes.
