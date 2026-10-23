# Experiment 3: what the eight runs show, and where to go next

Written 2026-10-15/16, after all Experiment 3 runs finished, and corrected after an independent fact-check against
the run reports. The per-run details and numbers are in `LOG.md`, and each run folder has a full report.

The model is GPT-2 small throughout, with BOS + 32 tokens and attention measured at the final token. "Late" means
layers 6–11. "BOS mass" is the late heads' attention on the first token: the "rest" / no-op state. "Rare" and
"common" are the id ranges of the random filler tokens.

## 1. The answer in one paragraph

Late heads rest on BOS more as the context contains more coherent language that the current token does not engage
with. They rest less when the current token continues or completes that context: a phrase's object, a country's
capital, a passage's true next word.

For country → capital, the release held wherever the country sat in the window. In aggregate, most of the released
attention spreads over the rest of the context rather than landing on the related token. Per token, though, the
related token gains several times more than an average filler token.

A correct country–capital pair before an unrelated token does not raise resting, and it slightly lowers it. Truth
matters mainly through completion.

Copying (induction) is a separate route off BOS, run by a partly separate set of heads.

## 2. Scorecard

The pre-registered predictions, all checked automatically:

| run | question | held |
|---|---|---|
| R1-out | does a true fact before a random final token raise BOS? | 1/4 |
| R1-in | does a correct capital at the final token lower BOS? | 3/7 |
| position sweep | where in the window does the pairing matter? | 3/5 |
| E3.4 | can the metrics see a known role switch (induction)? | 3/5 |
| E3.3 | how fast does the effect fade with distance? | 3/3 |
| E3.5 | does punctuation replace BOS as the resting place? | 2/3 |
| E3.1 | does a completing final token lift resting? | 3/6 |
| E3.2 | does more coherent text keep adding, or saturate? | 1/4 |
| **total** | | **19/37** |

All runs passed their integrity checks, with one exception: E3.5's registered previous-token-head check, which flags
a real confound (L2H5 tracks punctuation; see F6).

- Determinism was exact.
- TL and HF agreed to within 5e-5.
- Every placebo produced ≤ 1 switching head, and 0 of 496 placements were significant in the sweep.

Of the 18 failed predictions, many failed with a tight CI in the other direction. Seven were nulls or shortfalls,
such as too few switching heads or a single base class missing.

## 3. Findings

### F1. Resting grows with the amount of coherent context; long text does not depend on the final pair (E3.2, E3.3)

| context before the final token (late BOS vs random tokens, E(k)) | rare | common |
|---|---|---|
| 1–4 natural tokens | ≈ 0 | +0.014 to +0.043 |
| 8 | +0.021 | +0.074 |
| 16 | +0.058 | +0.105 |
| 31 (fully natural) | +0.133 | +0.159 |

- It does not saturate within 31 tokens on rare bases. On common bases the fitted κ is ≈ 16 tokens, so it is
  flattening.
- **It needs ordered language.** nl − shuffled grows from +0.006 to +0.117 on rare bases and from +0.021 to +0.114 on
  common.
- **Caveat on the same-form control.** nl − same-form random words reaches +0.21 / +0.23, but about a third of that
  is the control falling *below* random tokens (−0.08 / −0.07 at k = 31). E(k) above is the cleaner number.
- **Text does not need its final pair.** Removing the passage's last two words leaves most of the effect (E2d failed,
  0/12 passages). Test A's "pair before the query" describes 3-word phrases, not text.
- **Distance (E3.3).**
  - Word order counts mostly next to the query: it more than halves from g = 0 to g = 1, leaves a small but
    significant residual out to g = 15, and is gone at g = 24.
  - The presence of real words still counts 15 tokens away.
- **Absolute position.** On rare bases it does not matter: 24-token rows match 32-token rows. On common bases the
  24-token rows give slightly more (+0.004).

### F2. Completion at the current token reduces resting (R1-in, E3.1, E3.2, sweep)

| completing token at the query | late BOS change (rare / common) | contrast |
|---|---|---|
| correct capital after its country (R1-in) | −0.018 / −0.034 | correct − wrong pairing, query tokens balanced |
| own-country city (R1-in) | −0.015 / −0.016 | same |
| completing object vs same-form word (E3.1) | −0.054 / −0.026 | C(O) − C(M) |
| completing object vs implausible object (E3.1) | −0.037 / −0.027 | C(O) − C(O′) |
| true next word vs wrong word, natural context (E3.2) | −0.063 / −0.062 | 2 × 2 interaction, 12/12 passages |

**Completion shrinks the context's BOS boost rather than always reversing it.**
- The boost reverses only in R1-in and in E3.1 on rare bases (C(O) = −0.044).
- On common bases in E3.1, C(O) is still +0.008, vs +0.034 for a same-form word.
- In E3.2 the context still raises BOS with the true next word (+0.064 / +0.101), just less than with a wrong word.

**The sweep (country → capital only).** With the capital as the final token, all 31 country positions give
significant drops in both base classes, and about half of all heads are individually meaningful at (1, 32).
- On rare bases the drop grows with distance: −0.034 at position 1 vs −0.015 at 31.
- On common bases it is flat: near − far −0.002, CI including 0.

### F3. In aggregate, the freed attention lands mostly on the rest of the context (R1-in, E3.1, E3.2)

- In R1-in the filler gains +0.015 / +0.029 in total, the country +0.002 / +0.006. Per token, the country gains about
  5× an average filler token, and in E3.1 the phrase tokens gain 5–9×.
- **Late role switches are few.**
  - R1-in: 3 (rare) and 4 (common) heads switch Sink → Country, including L6H8 in every cell. Against placebo means
    of 0.45 / 0.70, the rare count falls just short of the registered threshold.
  - E3.1: 1 head and 0 heads.
  - E3.2: 0 on argmax labels for Sink → Span, but 13 late Sink → Span switches on mass labels in the query × context
    contrast.
- **Reading:** late heads mostly stop resting and spread attention over the context, with a per-token preference for
  the related tokens. A few heads, such as L6H8, switch outright.

### F4. Truth before an unrelated token does not raise resting for capitals (R1-out, sweep)

- **Capitals.** Correct vs wrong country–capital pairs before a random final token change BOS by −0.0009 / −0.0037 at
  (30, 31). Across the sweep's triangle the change is slightly negative, down to −0.007 when the country is early and
  the capital late (rare).
- **Cities.** A correct country–city pair *does* raise BOS on rare bases (+0.0057, 11/11 countries) but not on
  common.
- **What the pairing does change** is attention inside the pair: correct pairs move attention from the partner to the
  country. This holds in R1-out's four base × grid cells, and for adjacent pairs near the query in the sweep (S4).
- Any country + place pair raises BOS a lot vs none (+0.013 / +0.05).

### F5. Induction is a separate way off BOS, with partly separate heads (E3.4)

- On a repeat, 5–7 heads switch Sink → Copy: L5H0, L5H1, L5H5, L6H9, L7H1, L7H2, L7H10.
- They take more from BOS than proportional renormalization would (E4b): BOS is where these heads wait when there
  is nothing to copy.
- **Partial overlap with Exp 2's heads.** The late heads that rested most on coherent phrases in Exp 2 leave BOS on
  repeats less often than random late heads (E4d failed: 3–4/10 vs a null mean of 7/10). The overlap is not zero:
  - L6H8 and L6H11 do leave BOS on repeats.
  - L5H5 is among the top release heads in R1-in.
- Natural-language content gets copied *less* than same-form random words (E4c failed: −0.11 / −0.18 copy mass).

### F6. Punctuation is not an alternative sink in GPT-2 small (E3.5)

- A period draws only +0.007 / +0.013 of late attention, and 0–1 late heads switch to it.
- The punctuation response belongs to early heads, mainly L2H5 ("most recent punctuation"). That is why the
  registered previous-token check failed.

## 4. Revised working hypothesis

**Old (after Exp 2, H1′):** late heads park on BOS when the tokens right before the query form a locally coherent
pair.

**New (H3, "nothing to integrate"):** a late head's BOS mass at a token reflects how much coherent context exists that
the token does not engage with.

- More unconsumed coherent context means more resting (F1).
- A token that continues or completes the context consumes some of it: heads rest less and spread over the context
  (F2, F3). For country → capital, this holds at any distance.
- Unrelated real words raise resting (F4). Their truth adds little unless the current token completes them.
- Repeats offer something to copy and recruit a partly separate head set (F5).

H1′ is the special case of a short phrase next to an unrelated token. It fails for long text, where the effect is not
local, and completing tokens shrink the effect, reversing it only in some cells.

**What H3 does not yet explain:**

1. **The base-class splits.** Several effects differ between rare and common bases:
   - E1b holds on rare only;
   - S3 is 15% vs 1%;
   - E3.2's small-k curve is ≈ 0 on rare bases;
   - the sweep's distance effect is rare-only.
2. **The within-cell dose in R1-in (Rin-7, sign reversed).** Within a correct pairing, rows where the capital was
   *less* expected have *less* BOS. Across pairings, a more expected (correct) capital means less BOS.
   Predictability alone does not order resting.
3. **The inverse distance on rare bases:** why a far country releases more than an adjacent one.
4. **What the spread-out attention actually does** to the residual stream.

## 5. How this relates to known work

These pointers are for orientation; check details before citing.

- **BOS / first-token sinks:** Xiao et al. 2023 (StreamingLLM); Gu et al. 2024 (when sinks emerge); Barbero et al.
  2025 (why LLMs attend to the first token).
- **"No-op" attention to special tokens:**
  - Clark et al. 2019 found BERT heads attend to [SEP] and punctuation when they have nothing to do.
  - Kobayashi et al. 2020 showed that such attention carries tiny value-vector norms.
  - Sun et al. 2024 (massive activations) tie sinks to huge residual activations on those tokens.
  - Bondarenko et al. 2023 (quantizable transformers) describe heads learning "no update" by attending to
    low-information tokens.
- **Induction heads:** Olsson et al. 2022. IOI circuit heads in GPT-2 small: Wang et al. 2022. Our L5H5 and L6H9
  match.
- **Factual recall:** Geva et al. 2023 locate attribute extraction in upper attention heads reading the subject.

**What is new here:**

1. Resting is *graded by context*: more unconsumed coherent text means more rest.
2. Completion reduces it; for country → capital, at any distance.
3. Truth acts mainly through completion.
4. Punctuation does not play the [SEP] role in GPT-2 small at these lengths.

## 6. Research directions, in priority order

### D1. Is resting actually a no-op? (functional test; highest priority)

Everything so far measures attention weights. The thesis needs the *effect* on computation.

1. Measure value-weighted attention (α · ‖v‖, and the head's output norm) for resting vs released heads. Prediction:
   BOS contributes little output (following Kobayashi et al.), so "resting" heads write almost nothing.
2. Run an ablation test. Mean-ablate the late heads that rest in a "nothing to integrate" context and in a
   completion context, and measure the change in next-token loss. If resting is a no-op, ablation costs ~0 when
   heads rest and costs more when they are released.

### D2. What signals completion? (mechanism; adapts Test B)

R1-in gives a clean contrast: the same capital at the query, with only the country at 31 differing between correct
and wrong.

- Patch the residual stream (per layer, at the country position and at the query) from correct cells into wrong cells.
- Patch the late heads' q vs k / v.
- Patch MLP outputs: factual recall runs through MLPs at the subject in the factual-recall literature.

Predictions: the signal is written at the country's position by mid-layer MLPs and read key-side by late heads.
The sweep says it must work at any distance.

### D3. The natural-text regime

E3.2 gives the largest effects of anything tested. Move to real text, for example 1–5k OpenWebText windows.

- Per token, relate late-head BOS mass to (a) the token's surprisal, (b) whether the token continues a span seen
  earlier (an n-gram or entity match), and (c) the amount of preceding context.
- H3 predicts lower BOS mass on tokens that complete earlier structure than on tokens that start new material,
  controlling for position and surprisal. Rin-7 says to model surprisal non-monotonically.

### D4. The base-class splits (a dedicated test)

Vary filler frequency in steps (ids 256–999, 1k–5k, 5k–20k, 20k–40k) and rerun the core contrasts (E3.1's completion
contrast, E3.2 at k = 8, R1-in). This tells whether filler frequency moderates resting or is a nuisance.

### D5. Generality

- **Models.** Repeat the core contrasts in GPT-2 medium and a Pythia model. GPT-2 medium was deferred at the start of
  the project and needs your go-ahead.
- **Relations.** Run the position sweep for E3.1-style phrases and for other relations (R4: day → next day, number
  → next number, antonyms) and the R3 list frame. Only country → capital has been swept so far.

### Smaller follow-ups

- Why a distant country releases more on rare bases (sweep S2 reversed).
- Why natural phrases are copied less than same-form words (E4c).
- L2H5 as a punctuation tracker, and whether it matters downstream.

## 7. Caveats

- One model, GPT-2 small.
- Most contexts are random tokens with inserted language. E3.2 shows the natural regime behaves more strongly, not
  differently in sign.
- **Effect sizes.** Pooled late effects range from 0.002 (pairing before a random query) to 0.13–0.16 (31 natural
  tokens). Single heads reach 0.7–0.8 (E3.4 L6H9; E3.2 L8H6).
- **Role switches** are common for some contrasts: induction, long text, and the phrase right at the query (E3.3
  g = 0). They are rare for completion and truth contrasts.
- The verdicts above use the pre-registered label type (argmax). Mass labels give more switches in some contrasts.
