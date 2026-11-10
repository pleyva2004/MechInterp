# Pre-registered predictions: Experiment 4, controlled difference (country vs original token at τ)

Written 2026-10-24, before any data from this experiment. It follows the advisor meeting: the Exp 9–10 maps showed
several heads (L11H8 among them) whose most-attended key became the country only when the country was the token
right before the final token. Those maps cannot tell a previous-token head (attends to position T − 1 whatever sits
there) from a head that tracks the country. This experiment separates the two. Config: `configs/exp4_controlled.yaml`.

## Design

- **Ensemble P:** the 200 cached random-token rows per filler type that R1-out, R1-in and the sweep used
  (`exp3rel_{common,rare}`; common ids 256–999, rare ids 1000–39999). Each row is BOS + T = 32 tokens with no repeats
  and no stimulus ids. Positions are model positions (BOS 0, tokens 1..32). The final token, at T = 32, is the
  row's own random token and is never edited.
- **Matched variants per row and position τ = 1..31:**
  - **control:** the row unchanged, with its own random token at τ.
  - **country:** the same row with country c at τ, for each of the 11 countries. Nothing else changes.
  - **word (placebo):** the same row with a matched word at τ. Per row and country, one random id is drawn from the
    country's id class with the same leading space, alphabetic and capitalised flags, never a stimulus,
    demonstration or row token. The same word is used at every τ.
- **Measure:** the attention from the final token (query 32) to key τ, for every head. a(variant) − a(control) is
  averaged over the 11 countries within a row. The row is the unit. D_h(d) is the mean over rows at distance
  d = T − τ (1 = adjacent, 31 = farthest).
- **Statistics:** 95% bootstrap CI over rows (2000 resamples) and a two-sided Wilcoxon test over rows. BH is applied
  over the 31 × 144 head-distance cells within a filler type and arm. A cell is **meaningful** when q ≤ 0.05 and
  |D| ≥ 0.02.
- **prev_score:** the control attention from the final token to position 31, the previous token.
- **Head labels** (per filler type, first matching rule wins; pos(d) = meaningful and D > 0):
  1. **global country-tracking:** pos(d) at 15 or more of the 30 distances d = 2..31.
  2. **local country-tracking:** pos(d) at some d ≥ 2, but fewer than 15.
  3. **content-gated previous-token:** pos(1) only (none at d ≥ 2). The head attends more to the previous position
     when the country is there, but does not follow the country elsewhere.
  4. **negative:** some meaningful D, all of it negative.
  5. **plain previous-token:** no meaningful D at any d, and prev_score ≥ 0.2.
  6. **no country effect:** no meaningful D at any d, and prev_score < 0.2.
- **Focus heads:**
  - Exp 1's previous-token heads (`configs/exp3_e35.yaml`): L1H0, L2H2, L2H3, L2H5, L2H8, L2H9, L3H2, L3H6,
    L3H7, L3H8, L4H11.
  - Heads whose most-attended key became the country only when it was adjacent in the Exp 10 maps: L4H7, L5H6,
    L6H8, L11H8.
  - L0H2: most-attended = country at every Exp 10 position, so it is the candidate country tracker.
  - All 144 heads are measured.

## Predictions

Each must hold in both filler types.

| id | prediction | why |
|---|---|---|
| C1 | **Previous-token heads do not follow the country.** Every Exp 1 previous-token head with prev_score ≥ 0.2 has no meaningful positive D at any d ≥ 2 | a positional head reads T − 1 whatever is elsewhere (advisor's expected readout) |
| C2 | **Most of them are plain.** At least half of those heads are labelled plain previous-token, i.e. no meaningful D at d = 1 either | weaker; some may be content-modulated at T − 1 |
| C3 | **L11H8 is not a plain previous-token head.** prev_score < 0.2 and D(1) ≥ 0.02, meaningful | Exp 10: L11H8 was Diffuse with the country at 1 or 16 and Country only when adjacent |
| C4 | **L11H8's effect is local.** The mean of D(d) over d = 16–31 is below a quarter of D(1) | same maps: no country preference when the country was far |
| C5 | **L11H8 is country-specific at d = 1.** D_country(1) − D_word(1) > 0, with a 95% row CI excluding 0 | Exp 9: content at the adjacent position mattered |
| C6 | **Some head tracks the country globally.** At least one head is labelled global country-tracking (candidate L0H2) | L0H2 was Country at positions 1, 16 and 31 |
| H-loc | **Locality by depth (stretch, advisor's hypothesis).** Let LI be mean D over d = 2–4 minus mean D over d = 16–31, averaged over a layer group's heads. Two parts: (a) LI(layers 5–8) − LI(layers 9–11) > 0, with a 95% row CI excluding 0; (b) the 95% CI of LI(9–11) includes 0 | layers 5–8 act locally and 9–11 globally |

d = 1 is excluded from LI, so a previous-token head cannot drive it.

## Checks (not predictions)

- **Design:** each variant differs from its row only at τ; τ holds the intended id; there are no repeated tokens,
  no BOS, and no stimulus ids in the rows.
- **Attention:** rows sum to 1 (asserted per batch).
- **Cross-library:** TL vs HF max |Δ| < 1e-4, on 5 rows at τ = 1, 16 and 31.
- **Determinism:** recomputing 2 rows gives identical numbers.
- **Measurement sanity:** L4H11's prev_score ≥ 0.9. E3.5 measured it at about 0.996 on rare filler.
