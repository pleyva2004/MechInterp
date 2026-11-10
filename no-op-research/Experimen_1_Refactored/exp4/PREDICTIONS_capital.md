# Pre-registered predictions: Experiment 4b, controlled difference with the capital as the final token

Written 2026-10-24, before any data from this run. Exp 4 (`results/20261024-000548_exp4ctrl`) kept the row's own
random token at T = 32. Here the final token is a capital, and it is the same in both versions of each pair, so the
pair still differs only at τ. Config: `configs/exp4_capital.yaml`.

## Design

Everything not listed here is as in `exp4/PREDICTIONS.md`: the 200 cached rows per filler type, the 11 countries,
the matched placebo words (same seed, so the same words), τ = 1..31, and the statistics and labels.

- **Final token (position 32), two conditions.** In both, the token replaces the row's own final token in every
  version, control included.
  - **own:** the capital of the country in that cell (France … Paris).
  - **wrong:** the capital that a fixed derangement assigns to that country (seed 5; it never assigns a country its own
    capital).
  - Each capital appears exactly once per row in each condition. The control (no country at τ) is therefore the same
    set of 11 sequences in both, averaged within row.
- **D_own(d), D_wrong(d):** attention from the final token to τ, country (or word) at τ minus the original token,
  with that condition's capital at T.
- **Fact contrast:** D_own(d) − D_wrong(d) = a(country, own capital) − a(country, wrong capital), per row. The controls
  cancel, and each country and capital token appears equally often on both sides.
- **Late BOS fact:** the late-layer (6–11) mean BOS mass with the country at τ, own minus wrong. This repeats the
  sweep's capital-at-the-query placements (country at i, capital at 32) with 200 rows instead of 50.
- **Comparison with random T (descriptive, not predicted):** per-row D_own − D_random for the same rows and countries,
  using Exp 4's saved arrays.

## Predictions

Each must hold in both filler types.

| id | prediction | why |
|---|---|---|
| F1 | **The fact lowers resting at the adjacent position.** Late BOS fact < 0 at d = 1, with a 95% row CI below 0 | sweep: −0.033 / −0.015 at (31, 32); R1-in |
| F2 | **And at a distance.** The late BOS fact averaged over d = 16–31 has a 95% row CI below 0 | sweep: all 31 capital-at-query placements significant; S2 (decay) failed |
| F3 | **L11H8 attends more to the country when the query is its capital, at d = 1.** Fact contrast meaningful and > 0 at d = 1 | Exp 10: with Paris at T, L11H8's top key was the adjacent country |
| F4 | **The any-distance heads respond to the fact far away.** At least two of L9H8, L10H3, L11H7 and L11H2 have a meaningful positive fact contrast at 15 or more of the 30 distances d ≥ 2 | uncertain: on average the sweep's late country mass barely moved (+0.0005 / +0.0006) |
| F5 | **The pure previous-token head stays blind.** L4H11 has no meaningful D_own, D_wrong or fact contrast at any d ≥ 2 | Exp 4: Δ ≈ 0 for d ≥ 2 with random T |
| F6 | **Previous-token heads stay local.** No Exp 1 previous-token head has a meaningful D_own beyond d = 8 | Exp 4: last meaningful distance 1–8 with random T |

## Checks (not predictions)

- **Design:** each variant equals its row with that condition's capital at 32, except at τ. There are no repeats or
  BOS, and the derangement has no fixed point.
- **Attention:** rows sum to 1.
- **Cross-library:** TL vs HF max |Δ| < 1e-4, on 5 rows at τ = 1, 16 and 31 (own condition).
- **Determinism:** recomputing 2 rows gives identical numbers.
- **Controls identical:** the per-row control means are identical in the own and wrong conditions (by construction).
