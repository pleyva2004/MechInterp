# Pre-registered predictions: Experiment 3, class 2, R1-in (the capital / city is the final query)

The predictions were written into `exp3/DESIGN_relations.md` §4 on 2026-10-15, before any attention data, and are
copied here unchanged. R1-out ran before R1-in, but none of these predictions or thresholds was edited after it.
Config: `configs/exp3_relations_in.yaml`.

## Layout

| cells | 31 | 32 (query) |
|---|---|---|
| capital grid (121) | country c | capital k |
| city grid (121) | country c | city k |
| anchor (11) | country c | a same-form random word, matched to c's capital, drawn per row |
| capital alone (11) | the row's own token | capital k |
| none (1) | the row's own token | the row's own token |

There are 200 rows per base class, rare and common.

## Definitions

These are as in R1-out (`PREDICTIONS_relations.md`):

- T = mean(correct cells) − mean(wrong cells), per row.
- The verdict uses three outcomes (margin m = 0.005): positive / negative requires the 95% row CI and the one-sided
  pairing-permutation test to agree; equivalent means the 90% CI lies inside ±m; anything else is inconclusive.
- The placebo is 20 derangements.

The direction comes from `DESIGN.md` E1b: a query that completes a phrase lowers sink, because heads read the context.
The alternative, H-rest (a completed fact leaves nothing to do, so the head parks), extrapolates from Exp 2.

## Predictions

Each prediction must hold in both base classes.

| id | prediction | how it is checked |
|---|---|---|
| Rin-1 | A correct capital takes heads off BOS: late BOS T_cap < 0 | verdict "negative" |
| Rin-2 | The BOS mass lost goes to the country | T_cap on country mass (31) > 0 (95% CI), with ≥ 0.5 × \|T_cap BOS\|; \|T\| < m for the query's own mass (32) and for filler |
| Rin-3 | ≥ 3 heads beyond the placebo mean switch Sink → Country on correct vs wrong cells | McNemar, BH, ≥ 15-point shift; argmax labels |
| Rin-4 | Capital-specific (low confidence): T_cap − T_city on late BOS < 0 | verdict "negative"; the permutation test applies the same relabelling to both grids |
| Rin-5 | Among wrong pairs, late-BOS falls as association rises | slope of the cell mean on log p(capital \| BOS country) over the 110 wrong cells < 0, 95% CI over rows |
| Rin-6 | ≥ 9/11 countries' diagonal residuals have Rin-1's sign | late BOS, capital grid, sign negative |
| Rin-7 | Dose: within correct cells, rows where the capital was more expected have lower late BOS | pooled Spearman(surprisal, late BOS) across rows within each correct cell, averaged over the 11 cells, > 0 with its 95% CI over rows excluding 0 |

## Reported without a prediction

- **Rin-8 (descriptive):** correct − anchor and wrong − anchor, each judged by the three-way rule on the row CI.
  - If correct departs from the anchor and wrong does not, the fact changes heads.
  - If wrong departs from the anchor and correct does not, the contradiction does.
- Capital-alone vs anchor.
- Per-head maps.
- T for every category.
- The manipulation check.

## Checks

These are as in R1-out:

- design integrity (the extra cells are checked too);
- attention rows sum to 1;
- determinism;
- TL vs HF on 50 rows per class, |Δ| < 1e-4;
- manipulation: the own capital is preferred at 31 in ≥ 9/11 countries.
