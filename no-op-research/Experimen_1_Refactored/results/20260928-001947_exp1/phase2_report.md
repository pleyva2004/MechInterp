# Experiment 1, Phase 2: condition comparison

- Run: `20260928-001947_exp1`; conditions run here: `B_common`, `C_common_clean`; reused from earlier runs: `A_rare` from `results/20260927-235708_exp1`.
- 1000 sequences per condition; statistics on the TL backend unless noted; the other backend is re-tested in section 2 as a cross-library check.
- Id ranges: `A_rare` [1000, 39999], `B_common` [0, 999], `C_common_clean` [256, 999].

## 1. Head types with 95% bootstrap CIs

% of the 144 heads whose mode label is each type; CI from 1000 resamples of sequences (percentile method).

| mode_label | A_rare | B_common | C_common_clean |
|---|---|---|---|
| Diffuse | 13.2 [13.2, 13.9] | 10.4 [10.4, 11.1] | 10.4 [10.4, 11.1] |
| Sink | 73.6 [72.9, 73.6] | 73.6 [72.9, 74.3] | 75.7 [74.3, 75.7] |
| Previous | 7.6 [6.9, 7.6] | 9.0 [7.6, 9.7] | 6.2 [5.6, 8.3] |
| Self | 4.9 [4.9, 4.9] | 4.9 [4.9, 4.9] | 4.9 [4.9, 4.9] |
| Other | 0.7 [0.7, 1.4] | 2.1 [1.4, 2.8] | 2.8 [2.1, 3.5] |

## 2. Per-head sink_mass comparisons (Mann-Whitney U, BH-FDR q ≤ 0.05 over 432 tests)

effect_size = rank-biserial r = P(first > second) − P(first < second); r > 0 means the first condition puts more mass on the sink.

| comparison | significant heads | sig. with r > 0 | sig. with r < 0 | median |r| | largest |r| |
|---|---|---|---|---|---|
| A_rare vs B_common | 134/144 | 74 | 60 | 0.282 | L2H10 (r = -0.983, Δ = -0.009) |
| A_rare vs C_common_clean | 135/144 | 67 | 68 | 0.331 | L2H10 (r = -0.989, Δ = -0.009) |
| B_common vs C_common_clean | 111/144 | 35 | 76 | 0.103 | L11H2 (r = -0.688, Δ = -0.259) |

Top significant heads by |r|, A_rare vs B_common:

| layer | head | mean_diff | effect_size | p | q |
|---|---|---|---|---|---|
| 2 | 10 | -0.009 | -0.983 | 0.0e+00 | 0.0e+00 |
| 1 | 2 | -0.004 | -0.885 | 3.5e-257 | 3.8e-255 |
| 2 | 0 | 0.108 | 0.803 | 1.9e-212 | 1.2e-210 |
| 2 | 4 | 0.347 | 0.748 | 2.6e-184 | 1.0e-182 |
| 10 | 9 | 0.336 | 0.733 | 2.8e-177 | 1.0e-175 |
| 5 | 2 | 0.307 | 0.724 | 4.5e-173 | 1.5e-171 |
| 7 | 7 | -0.306 | -0.712 | 2.3e-167 | 6.3e-166 |
| 7 | 8 | 0.228 | 0.697 | 2.7e-160 | 6.6e-159 |
| 1 | 8 | 0.037 | 0.656 | 3.6e-142 | 7.1e-141 |
| 1 | 3 | -0.024 | -0.639 | 4.3e-135 | 6.9e-134 |

Top significant heads by |r|, A_rare vs C_common_clean:

| layer | head | mean_diff | effect_size | p | q |
|---|---|---|---|---|---|
| 2 | 10 | -0.009 | -0.989 | 0.0e+00 | 0.0e+00 |
| 1 | 2 | -0.004 | -0.935 | 6.4e-287 | 9.2e-285 |
| 1 | 7 | -0.091 | -0.838 | 7.1e-231 | 6.1e-229 |
| 2 | 0 | 0.107 | 0.805 | 2.5e-213 | 1.8e-211 |
| 10 | 9 | 0.355 | 0.803 | 5.2e-212 | 2.8e-210 |
| 0 | 11 | -0.071 | -0.788 | 2.9e-204 | 1.4e-202 |
| 5 | 2 | 0.336 | 0.763 | 1.0e-191 | 4.4e-190 |
| 7 | 7 | -0.308 | -0.717 | 1.3e-169 | 4.1e-168 |
| 3 | 4 | -0.176 | -0.714 | 5.2e-168 | 1.5e-166 |
| 2 | 4 | 0.346 | 0.711 | 5.5e-167 | 1.4e-165 |

Top significant heads by |r|, B_common vs C_common_clean:

| layer | head | mean_diff | effect_size | p | q |
|---|---|---|---|---|---|
| 11 | 2 | -0.259 | -0.688 | 2.2e-156 | 5.0e-155 |
| 10 | 0 | -0.245 | -0.620 | 1.9e-127 | 2.5e-126 |
| 3 | 0 | -0.109 | -0.435 | 1.4e-63 | 7.1e-63 |
| 3 | 4 | -0.101 | -0.401 | 2.1e-54 | 8.5e-54 |
| 11 | 9 | -0.136 | -0.393 | 3.5e-52 | 1.4e-51 |
| 1 | 7 | -0.038 | -0.388 | 5.1e-51 | 2.0e-50 |
| 5 | 9 | -0.080 | -0.381 | 3.9e-49 | 1.4e-48 |
| 4 | 4 | -0.046 | -0.366 | 1.5e-45 | 5.0e-45 |
| 11 | 8 | -0.000 | -0.365 | 2.0e-45 | 6.6e-45 |
| 8 | 2 | 0.142 | 0.361 | 2.2e-44 | 7.4e-44 |

Cross-library check (HF vs TL): significance decisions agree on 432/432 tests; max |Δ effect_size| = 1.00e-05.

## 3. Split-half noise floor

Each condition's sequences are split into two random halves (500 vs 500) and compared with the same test; BH-FDR over all 432 split-half tests. With no real difference, any 'significant' head here is noise. The halves have half the sample size of the cross-condition tests, so this floor is slightly conservative in power.

| condition | significant (q ≤ 0.05) | uncorrected p < 0.05 | median |r| | max |r| |
|---|---|---|---|---|
| A_rare | 0/144 | 4 | 0.033 | 0.092 |
| B_common | 0/144 | 3 | 0.017 | 0.094 |
| C_common_clean | 0/144 | 2 | 0.020 | 0.079 |

For reference, 5% of 144 heads = 7.2 uncorrected false positives expected per condition under the null.

## 4. Monotonicity: does head activity increase A_rare < B_common < C_common_clean?

Overall (mean over all heads and sequences):

| measure | A_rare | B_common | C_common_clean | increasing A_rare < B_common < C_common_clean |
|---|---|---|---|---|
| mean (1 − sink_mass) | 0.5322 | 0.5574 | 0.5393 | no |
| mean entropy (nats) | 1.7223 | 1.7916 | 1.7590 | no |

Per layer (mean over the layer's heads and all sequences):

| layer | 1−sink A_rare | 1−sink B_common | 1−sink C_common_clean | 1−sink increasing | entropy A_rare | entropy B_common | entropy C_common_clean | entropy increasing |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.932 | 0.925 | 0.927 | no | 2.135 | 2.153 | 2.171 | yes |
| 1 | 0.844 | 0.841 | 0.833 | no | 2.702 | 2.678 | 2.660 | no |
| 2 | 0.791 | 0.832 | 0.826 | no | 2.105 | 2.135 | 2.124 | no |
| 3 | 0.632 | 0.636 | 0.590 | no | 1.606 | 1.610 | 1.508 | no |
| 4 | 0.472 | 0.516 | 0.508 | no | 1.363 | 1.484 | 1.445 | no |
| 5 | 0.343 | 0.373 | 0.353 | no | 1.270 | 1.335 | 1.268 | no |
| 6 | 0.391 | 0.448 | 0.433 | no | 1.543 | 1.716 | 1.672 | no |
| 7 | 0.296 | 0.336 | 0.326 | no | 1.186 | 1.346 | 1.325 | no |
| 8 | 0.347 | 0.361 | 0.383 | yes | 1.441 | 1.533 | 1.619 | yes |
| 9 | 0.319 | 0.329 | 0.312 | no | 1.390 | 1.387 | 1.376 | no |
| 10 | 0.445 | 0.498 | 0.414 | no | 1.729 | 1.876 | 1.712 | no |
| 11 | 0.574 | 0.596 | 0.567 | no | 2.199 | 2.248 | 2.229 | no |

Layers where the order holds: 1/12 for 1 − sink_mass, 2/12 for entropy.

## 5. Figures

- [Mode label, conditions side by side](figures/phase2_mode_label_panels.png)
- [Δ mean sink_mass per head, FDR-significant heads outlined](figures/phase2_sink_mass_diff.png)
- [Head-type % by condition with 95% CIs](figures/phase2_head_types.png)
- Per-condition figures and validation checks for the conditions run here: `validation_report.md`
