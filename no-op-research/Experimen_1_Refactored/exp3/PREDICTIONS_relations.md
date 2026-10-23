# Pre-registered predictions: Experiment 3, class 2, R1-out (country → capital / city before a random query)

Written 2026-10-15, before any attention data for these stimuli. The stimulus selection and the manipulation-check
pilot used next-token probabilities only. Config: `configs/exp3_relations.yaml`. Design:
`exp3/DESIGN_relations.md`.

## Layout

| position | 30 | 31 | 32 (query) |
|---|---|---|---|
| capital grid | country c | capital k | the row's own random token |
| city grid | country c | city k | the row's own random token |

There are 11 countries, so each grid has 121 cells: 11 correct, 110 wrong. Each base row also has one "none" cell.
200 rows per base class, rare (1000–39999) and common (256–999).

## Definitions

- **T** = mean over correct cells − mean over wrong cells, per row. Every country and every partner appears once in
  a correct cell and 10 times in wrong ones, so each token's own effect cancels. T equals 11/10 × the mean diagonal
  residual of the double-centred grid.
- **Late** = layers 6–11, the mean over their 72 heads. **Metric** = BOS mass at the final query.
- **Verdict** (three-way, margin m = 0.005 on T):
  - positive / negative: the 95% bootstrap CI over rows and the one-sided permutation test over pairings
    (10,000 random permutations, p < 0.05) agree on the side;
  - equivalent: the 90% row CI lies inside ±m;
  - inconclusive: anything else.
- **Item residual** = a country's diagonal residual after double-centring the row-mean grid.
- **Placebo:** 20 random derangements (seed 2), each scored as if it were the correct pairing.

## Predictions

Each must hold in both base classes.

| id | prediction | why |
|---|---|---|
| Rout-1 | late BOS, capital grid: verdict **positive** | Exp 2's working hypothesis: late heads park more when the tokens before the query form a coherent pair. Association predicts the same sign |
| Rout-2 | late BOS, city grid: verdict **positive** | A coherent pair need not be a capital fact. Country–city association is nearly as strong as country–capital (pilot) |
| Rout-3 | capital grid, late BOS: ≥ 9/11 countries have a positive item residual | Generalizes across items (one-sided binomial p = 0.033) |
| Rout-4 | per base class and grid: ≤ 2 of 20 derangements reach two-sided permutation p < 0.05 on late BOS, and the mean number of switching heads over derangements is ≤ 1 | The pipeline does not manufacture pairing effects |

## Checks (must pass for the predictions to count)

- Every cell holds the right ids at 30 and 31 and equals its base row elsewhere. No repeats, no BOS, no stimulus or
  demonstration id in any base row.
- Attention rows sum to 1 (atol 1e-4).
- Determinism: a second TL pass on 5 rows per class matches exactly.
- Cross-library: TL vs HF on 50 rows per class, max |Δ attention| < 1e-4.
- Manipulation: at position 30 the model prefers each country's own capital over the mean of the other 10 in
  ≥ 9/11 countries, in each base class (the pilot gave 11/11).

## Reported without a prediction

- T for country, capital/city (previous) and self/filler mass.
- Per-head T maps with BH-FDR.
- Role switches (correct vs wrong, McNemar, ≥ 15-point shift) against the placebo mean.
- The capital − city difference.
- The association slope among wrong cells.
- Correct − none and wrong − none (the total effect, comparable to Exp 2).
