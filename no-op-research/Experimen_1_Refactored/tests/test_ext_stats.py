"""Tests for exp1extended.stats."""
from __future__ import annotations

import warnings
from itertools import product

import numpy as np
import pytest
from scipy.stats import kstest, wilcoxon

from exp1extended.stats import (
    DIAGONAL_TERMS, YATES_EFFECTS, bootstrap_twoway_ci, diagonal_shares, icc_oneway, paired_test,
    split_half_reliability, twoway_effects, within_background_permutation_p, yates_effects,
    yates_effects_from_cells,
)

CODES = ["".join(levels) for levels in product("rc", repeat=3)]  # Q, P, X order


def _sign(level: str) -> float:
    return 1.0 if level == "c" else -1.0


def _missing_mask(rng: np.random.Generator, n_tok: int, n_bg: int, frac: float) -> np.ndarray:
    """Random mask with every token and background keeping at least one observed cell."""
    missing = rng.random((n_tok, n_bg)) < frac
    missing[np.arange(n_tok), rng.integers(0, n_bg, size=n_tok)] = False
    missing[rng.integers(0, n_tok, size=n_bg), np.arange(n_bg)] = False
    return missing


def _additive(rng, n_tok, n_bg, n_cols, token_sd, bg_sd, noise_sd):
    a = rng.normal(0.0, token_sd, size=(n_tok, 1, n_cols))
    b = rng.normal(0.0, bg_sd, size=(1, n_bg, n_cols))
    return a + b + rng.normal(0.0, noise_sd, size=(n_tok, n_bg, n_cols)), a[:, 0], b[0]


def _lstsq_reference(y: np.ndarray, missing: np.ndarray):
    """Token effects from an explicit dummy-coded least-squares fit (single column)."""
    n_tok, n_bg = missing.shape
    rows, cols = np.nonzero(~missing)
    design = np.zeros((rows.size, n_tok + n_bg))
    design[np.arange(rows.size), rows] = 1.0
    design[np.arange(rows.size), n_tok + cols] = 1.0
    coef = np.linalg.lstsq(design, y[rows, cols], rcond=None)[0]
    fitted = design @ coef
    levels = coef[:n_tok] + coef[n_tok:].mean()  # token level at the average background (gauge-free)
    return levels - levels.mean(), fitted


# ---- paired_test --------------------------------------------------------------------------------------------

def test_paired_test_shifted_normal_small_p_and_ci_contains_shift():
    rng = np.random.default_rng(0)
    shift = 0.5
    noise = rng.normal(size=(200, 3, 4))
    noise -= noise.mean(axis=0)  # sample mean difference is exactly the shift
    y = rng.normal(size=(200, 3, 4))
    x = y + shift + noise

    result = paired_test(x, y, iters=500, seed=1, ci=0.95)

    assert result["n"] == 200
    for key in ("mean_diff", "lo", "hi", "p"):
        assert result[key].shape == (3, 4)
    np.testing.assert_allclose(result["mean_diff"], shift)
    assert np.all(result["lo"] < shift) and np.all(shift < result["hi"])
    assert np.all(result["p"] < 1e-4)


def test_paired_test_identical_inputs_p_one_without_warnings():
    x = np.random.default_rng(1).normal(size=(40, 12, 12))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = paired_test(x, x.copy(), iters=100, seed=0, ci=0.95)
    np.testing.assert_array_equal(result["p"], 1.0)
    np.testing.assert_array_equal(result["mean_diff"], 0.0)
    np.testing.assert_array_equal(result["lo"], 0.0)
    np.testing.assert_array_equal(result["hi"], 0.0)


@pytest.mark.parametrize("n", [30, 120])  # per-column scipy calls below the asymptotic cutoff, vectorized above
def test_paired_test_p_matches_scipy_per_column(n):
    rng = np.random.default_rng(2)
    x = rng.normal(size=(n, 5))
    y = x - rng.normal(0.2, 1.0, size=(n, 5))
    y[:, 1] = x[:, 1]  # all differences zero
    y[: n // 4, 3] = x[: n // 4, 3]  # some zeros

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = paired_test(x, y, iters=50, seed=0, ci=0.9)

    assert result["p"][1] == 1.0
    for j in (0, 2, 3, 4):
        ref = wilcoxon(x[:, j] - y[:, j], zero_method="wilcox").pvalue
        assert result["p"][j] == pytest.approx(ref)


def test_paired_test_bootstrap_shares_resamples_across_columns_and_is_deterministic():
    rng = np.random.default_rng(3)
    base = rng.normal(size=(60, 1))
    x = np.concatenate([base, 2.0 * base, base + 7.0], axis=1)
    y = np.zeros_like(x)

    r1 = paired_test(x, y, iters=300, seed=5, ci=0.95)
    r2 = paired_test(x, y, iters=300, seed=5, ci=0.95)
    for key in ("mean_diff", "lo", "hi", "p"):
        np.testing.assert_array_equal(r1[key], r2[key])
    # the same resampled pairs in every column -> CIs are exact affine images of column 0's
    np.testing.assert_allclose(r1["lo"][1], 2.0 * r1["lo"][0])
    np.testing.assert_allclose(r1["hi"][2], r1["hi"][0] + 7.0)


def test_paired_test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        paired_test(np.zeros((5, 3)), np.zeros((5, 4)), iters=10, seed=0, ci=0.95)


# ---- Yates effects -----------------------------------------------------------------------------------------

def test_yates_additive_model_has_zero_interactions_and_recovers_mains():
    rng = np.random.default_rng(4)
    n = 50
    base = rng.normal(size=(n, 12, 12))
    beta = {"Q": rng.normal(size=(12, 12)), "P": rng.normal(size=(12, 12)), "X": rng.normal(size=(12, 12))}
    cells = {
        code: base + sum(beta[f] * (level == "c") for f, level in zip("QPX", code)) for code in CODES
    }

    effects = yates_effects(cells)

    assert tuple(effects) == YATES_EFFECTS
    for f in "QPX":
        np.testing.assert_allclose(effects[f], np.broadcast_to(beta[f], (n, 12, 12)), atol=1e-12)
    for name in ("QP", "QX", "PX", "QPX"):
        np.testing.assert_allclose(effects[name], 0.0, atol=1e-12)


def test_yates_diagonal_identity_holds_exactly_per_pair():
    rng = np.random.default_rng(5)
    cells = {code: rng.normal(size=(30, 7)) for code in CODES}  # arbitrary, interactions included

    effects = yates_effects(cells)

    total = cells["ccc"] - cells["rrr"]
    np.testing.assert_allclose(sum(effects[t] for t in DIAGONAL_TERMS), total, rtol=0, atol=1e-12)
    # the two-way terms are not part of the diagonal and are nonzero here
    assert np.abs(effects["QP"]).max() > 0.1


def test_yates_matches_definitions():
    rng = np.random.default_rng(6)
    cells = {code: rng.normal(size=(20, 3)) for code in CODES}
    effects = yates_effects(cells)

    mean_q_c = np.mean([cells[c] for c in CODES if c[0] == "c"], axis=0)
    mean_q_r = np.mean([cells[c] for c in CODES if c[0] == "r"], axis=0)
    np.testing.assert_allclose(effects["Q"], mean_q_c - mean_q_r)

    q_at = {p: np.mean([cells["c" + p + x] - cells["r" + p + x] for x in "rc"], axis=0) for p in "rc"}
    np.testing.assert_allclose(effects["QP"], 0.5 * (q_at["c"] - q_at["r"]))

    three_way = sum(_sign(c[0]) * _sign(c[1]) * _sign(c[2]) * cells[c] for c in CODES) / 4.0
    np.testing.assert_allclose(effects["QPX"], three_way)


def test_yates_accepts_cell_names_ignores_other_cells_and_factor_order():
    rng = np.random.default_rng(7)
    cells = {code: rng.normal(size=(10, 2)) for code in CODES}
    named = {f"Q{c[0]}_P{c[1]}_X{c[2]}": v for c, v in cells.items()}
    named["Qm_Pm_Xm"] = rng.normal(size=(10, 2))  # not an r/c cell -> ignored
    named["mmm"] = rng.normal(size=(10, 2))

    ref = yates_effects(cells)
    got = yates_effects(named)
    for name in YATES_EFFECTS:
        np.testing.assert_array_equal(got[name], ref[name])

    # codes written in (X, P, Q) order give the same effects under the matching names
    reordered = {c[2] + c[1] + c[0]: v for c, v in cells.items()}
    rev = yates_effects_from_cells(reordered, factor_order=("X", "P", "Q"))
    for name, rev_name in [("Q", "Q"), ("QP", "PQ"), ("PX", "XP"), ("QPX", "XPQ")]:
        np.testing.assert_allclose(rev[rev_name], ref[name])


def test_yates_missing_cell_raises():
    cells = {code: np.zeros((4, 2)) for code in CODES if code != "rcr"}
    with pytest.raises(ValueError, match="rcr"):
        yates_effects(cells)


def test_diagonal_shares_sum_to_one_and_match_means():
    rng = np.random.default_rng(8)
    cells = {code: rng.normal(size=(100, 4)) + 0.5 * code.count("c") for code in CODES}
    effects = yates_effects(cells)

    shares = diagonal_shares(effects)

    assert tuple(shares) == DIAGONAL_TERMS
    gap = (cells["ccc"] - cells["rrr"]).mean(axis=0)
    np.testing.assert_allclose(sum(shares.values()), 1.0)
    np.testing.assert_allclose(shares["Q"], effects["Q"].mean(axis=0) / gap)


def test_diagonal_shares_nan_when_gap_is_zero():
    cells = {code: np.ones((5, 2)) for code in CODES}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        shares = diagonal_shares(yates_effects(cells))
    assert all(np.all(np.isnan(v)) for v in shares.values())


# ---- twoway_effects ----------------------------------------------------------------------------------------

def test_twoway_no_missing_equals_row_and_column_means():
    rng = np.random.default_rng(9)
    y = rng.normal(size=(40, 16, 3, 2))
    missing = np.zeros((40, 16), dtype=bool)

    token_eff, bg_eff, grand = twoway_effects(y, missing)

    np.testing.assert_allclose(grand, y.mean(axis=(0, 1)), rtol=1e-13)
    np.testing.assert_allclose(token_eff, y.mean(axis=1) - y.mean(axis=(0, 1)), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(bg_eff, y.mean(axis=0) - y.mean(axis=(0, 1)), rtol=1e-12, atol=1e-14)
    levels, _, _ = twoway_effects(y, missing, center=False)
    np.testing.assert_allclose(levels, y.mean(axis=1), rtol=1e-13)


def test_twoway_missing_matches_explicit_least_squares():
    rng = np.random.default_rng(10)
    y, _, _ = _additive(rng, 12, 5, 1, 1.0, 2.0, 0.5)
    missing = _missing_mask(rng, 12, 5, 0.25)
    y[missing] = np.nan  # missing cells may hold anything

    token_eff, bg_eff, grand = twoway_effects(y, missing)

    ref_eff, ref_fitted = _lstsq_reference(y[..., 0], missing)
    np.testing.assert_allclose(token_eff[:, 0], ref_eff, atol=1e-10)
    rows, cols = np.nonzero(~missing)
    fitted = grand[0] + token_eff[rows, 0] + bg_eff[cols, 0]
    np.testing.assert_allclose(fitted, ref_fitted, atol=1e-10)
    assert abs(token_eff.sum()) < 1e-10 and abs(bg_eff.sum()) < 1e-10


def test_twoway_recovers_planted_token_effects_with_missing_cells():
    rng = np.random.default_rng(11)
    y, a, b = _additive(rng, 300, 32, 4, 1.0, 3.0, 0.2)  # background effects much larger than token effects
    missing = _missing_mask(rng, 300, 32, 0.10)

    token_eff, bg_eff, _ = twoway_effects(y, missing)

    truth = a - a.mean(axis=0)
    # noise SE per token ~ 0.2 / sqrt(29) ~ 0.04; naive masked row means would be off by ~3 / sqrt(29)
    assert np.abs(token_eff - truth).max() < 0.2
    for j in range(4):
        assert np.corrcoef(token_eff[:, j], truth[:, j])[0, 1] > 0.99
    np.testing.assert_allclose(bg_eff, b - b.mean(axis=0), atol=0.1)


def test_twoway_empty_token_is_nan_and_disconnected_raises():
    rng = np.random.default_rng(12)
    y = rng.normal(size=(6, 4, 2))
    missing = np.zeros((6, 4), dtype=bool)
    missing[2] = True
    missing[0, 1] = True
    token_eff, _, _ = twoway_effects(y, missing)
    assert np.all(np.isnan(token_eff[2]))
    assert np.all(np.isfinite(np.delete(token_eff, 2, axis=0)))
    np.testing.assert_allclose(np.delete(token_eff, 2, axis=0).sum(axis=0), 0.0, atol=1e-12)

    blocks = np.ones((6, 4), dtype=bool)
    blocks[:3, :2] = False
    blocks[3:, 2:] = False
    with pytest.raises(ValueError, match="disconnected"):
        twoway_effects(y, blocks)


# ---- split_half_reliability --------------------------------------------------------------------------------

def test_split_half_reliability_high_for_strong_effects_low_for_noise():
    rng = np.random.default_rng(13)
    strong, _, _ = _additive(rng, 150, 16, 3, 1.0, 1.0, 0.1)
    noise, _, _ = _additive(rng, 150, 16, 3, 0.0, 1.0, 1.0)
    missing = _missing_mask(rng, 150, 16, 0.10)

    rel_strong = split_half_reliability(strong, missing, seed=0, n_splits=10)
    rel_noise = split_half_reliability(noise, missing, seed=0, n_splits=10)

    assert rel_strong.shape == (3,)
    assert np.all(rel_strong > 0.98)
    assert np.all(np.abs(rel_noise) < 0.3)
    np.testing.assert_array_equal(rel_strong, split_half_reliability(strong, missing, seed=0, n_splits=10))


# ---- within_background_permutation_p ------------------------------------------------------------------------

def test_permutation_p_roughly_uniform_under_null():
    rng = np.random.default_rng(14)
    p = []
    for rep in range(200):
        # large background effects, no token effect: only a within-background shuffle keeps this null exact
        y, _, _ = _additive(rng, 8, 5, 1, 0.0, 5.0, 1.0)
        missing = _missing_mask(rng, 8, 5, 0.10)
        p.append(within_background_permutation_p(y, missing, n_perm=49, seed=rep)[0])
    p = np.asarray(p)
    assert np.all((p >= 1 / 50) & (p <= 1.0))
    assert kstest(p, "uniform").pvalue > 0.001
    assert 0.35 < p.mean() < 0.65


def test_permutation_p_small_under_planted_effect_and_ignores_missing_values():
    rng = np.random.default_rng(15)
    y, _, _ = _additive(rng, 30, 8, 3, 1.0, 2.0, 1.0)
    y[..., 2] = rng.normal(size=(30, 8))  # column 2 has no token effect
    missing = _missing_mask(rng, 30, 8, 0.10)

    p = within_background_permutation_p(y, missing, n_perm=199, seed=0)
    assert p.shape == (3,)
    assert p[0] == p[1] == pytest.approx(1 / 200)
    assert p[2] > 0.01

    garbage = y.copy()
    garbage[missing] = 1e6  # values in missing cells must never enter the statistic or the null
    np.testing.assert_array_equal(within_background_permutation_p(garbage, missing, n_perm=199, seed=0), p)


def test_permutation_p_constant_column_is_one():
    y = np.full((10, 4, 1), 3.7)
    missing = np.zeros((10, 4), dtype=bool)
    missing[0, 0] = True
    assert within_background_permutation_p(y, missing, n_perm=20, seed=0)[0] == 1.0


# ---- icc_oneway -----------------------------------------------------------------------------------------

def test_icc_oneway_balanced_hand_calculation():
    # group means 1.5, 3.5, 5.5, grand 3.5: MSB = 2 * (4 + 0 + 4) / 2 = 8, MSW = 6 * 0.25 / 3 = 0.5, n0 = 2
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    groups = np.array([0, 0, 1, 1, 2, 2])
    assert icc_oneway(y, groups) == pytest.approx((8.0 - 0.5) / (8.0 + 0.5))

    # a singleton group is dropped; trailing axes are vectorized
    y_single = np.append(y, 100.0)
    groups_single = np.append(groups, 9)
    assert icc_oneway(y_single, groups_single) == pytest.approx(7.5 / 8.5)
    stacked = np.stack([y, -2.0 * y], axis=1)
    np.testing.assert_allclose(icc_oneway(stacked, groups), [7.5 / 8.5, 7.5 / 8.5])


def test_icc_oneway_unbalanced_matches_formula():
    y = np.array([1.0, 3.0, 2.0, 6.0, 5.0, 7.0, 0.0])
    groups = np.array(["a", "a", "a", "b", "b", "b", "b"])
    sizes = np.array([3, 4])
    means = np.array([2.0, 4.5])
    grand = y.mean()
    msb = (sizes * (means - grand) ** 2).sum() / 1
    msw = (((y[:3] - 2.0) ** 2).sum() + ((y[3:] - 4.5) ** 2).sum()) / (7 - 2)
    n0 = (7 - (9 + 16) / 7) / 1
    assert icc_oneway(y, groups) == pytest.approx((msb - msw) / (msb + (n0 - 1) * msw))


def test_icc_oneway_needs_two_groups():
    with pytest.raises(ValueError):
        icc_oneway(np.arange(4.0), np.array([0, 0, 0, 1]))


# ---- bootstrap_twoway_ci -----------------------------------------------------------------------------------

def test_bootstrap_twoway_ci_brackets_point_estimate():
    rng = np.random.default_rng(16)
    y, _, _ = _additive(rng, 60, 12, 2, 1.0, 1.0, 0.5)
    y[:30] += np.array([0.8, -0.3])  # first half of the tokens shifted, per column
    missing = _missing_mask(rng, 60, 12, 0.10)
    group = np.arange(60) < 30

    def group_gap(v, m, tok_idx, bg_idx):
        eff = twoway_effects(v, m)[0]
        g = group[tok_idx]
        return np.nanmean(eff[g], axis=0) - np.nanmean(eff[~g], axis=0)

    point, lo, hi = bootstrap_twoway_ci(y, missing, group_gap, iters=100, seed=0, ci=0.95, with_index=True)

    assert point.shape == lo.shape == hi.shape == (2,)
    assert np.all(lo <= point) and np.all(point <= hi)
    assert lo[0] > 0 and hi[1] < 0.5
    again = bootstrap_twoway_ci(y, missing, group_gap, iters=100, seed=0, ci=0.95, with_index=True)
    for got, ref in zip(again, (point, lo, hi)):
        np.testing.assert_array_equal(got, ref)


def test_bootstrap_twoway_ci_plain_stat_fn():
    rng = np.random.default_rng(17)
    y, _, _ = _additive(rng, 40, 8, 1, 1.0, 1.0, 0.5)
    missing = _missing_mask(rng, 40, 8, 0.10)

    def token_sd(v, m):
        return np.nanstd(twoway_effects(v, m)[0], axis=0)

    point, lo, hi = bootstrap_twoway_ci(y, missing, token_sd, iters=100, seed=1, ci=0.9)
    assert lo[0] <= point[0] <= hi[0]
