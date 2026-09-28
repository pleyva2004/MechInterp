"""Tests for exp1.compare."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import false_discovery_control, mannwhitneyu

from exp1.compare import (
    bh_fdr, bootstrap_label_percentages, bootstrap_mean_ci, compare_conditions, label_shift, split_half,
)

N_LAYERS, N_HEADS = 12, 12


# ---- compare_conditions -----------------------------------------------------

def test_compare_conditions_matches_scipy_loop():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(9, N_LAYERS, N_HEADS))
    y = rng.normal(loc=0.3, size=(11, N_LAYERS, N_HEADS))

    result = compare_conditions(x, y)

    for l in range(N_LAYERS):
        for h in range(N_HEADS):
            ref = mannwhitneyu(
                x[:, l, h], y[:, l, h],
                alternative="two-sided", method="asymptotic", use_continuity=True,
            )
            assert result["U"][l, h] == pytest.approx(ref.statistic)
            assert result["p"][l, h] == pytest.approx(ref.pvalue)


def test_compare_conditions_effect_size_extremes():
    n_x, n_y = 6, 8
    x = np.full((n_x, N_LAYERS, N_HEADS), 10.0)
    y = np.full((n_y, N_LAYERS, N_HEADS), 0.0)

    result = compare_conditions(x, y)
    np.testing.assert_allclose(result["effect_size"], 1.0)

    result_rev = compare_conditions(y, x)
    np.testing.assert_allclose(result_rev["effect_size"], -1.0)


def test_compare_conditions_effect_size_near_zero_same_distribution():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2000, N_LAYERS, N_HEADS))
    y = rng.normal(size=(2000, N_LAYERS, N_HEADS))

    result = compare_conditions(x, y)
    # loose tolerance: same distribution, large n -> effect size near 0
    assert np.abs(result["effect_size"]).mean() < 0.05


def test_compare_conditions_sign_convention():
    rng = np.random.default_rng(2)
    y = rng.normal(size=(200, N_LAYERS, N_HEADS))
    x = y + rng.normal(size=(200, N_LAYERS, N_HEADS)) * 0.01 + 1.0  # x shifted up

    result = compare_conditions(x, y)
    assert np.all(result["effect_size"] > 0)
    assert np.all(result["mean_diff"] > 0)


def test_compare_conditions_shape_and_dtype():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(5, N_LAYERS, N_HEADS))
    y = rng.normal(size=(7, N_LAYERS, N_HEADS))
    result = compare_conditions(x, y)
    for key in ("U", "p", "effect_size", "mean_diff"):
        assert result[key].shape == (N_LAYERS, N_HEADS)
        assert result[key].dtype == np.float64


def test_compare_conditions_unequal_n_works():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(3, 5))
    y = rng.normal(size=(19, 5))
    result = compare_conditions(x, y)
    assert result["U"].shape == (5,)
    assert np.all(result["U"] >= 0)
    assert np.all(result["U"] <= 3 * 19)


def test_compare_conditions_shape_mismatch_raises():
    x = np.zeros((5, N_LAYERS, N_HEADS))
    y = np.zeros((5, N_LAYERS, N_HEADS - 1))
    with pytest.raises(ValueError):
        compare_conditions(x, y)


# ---- bh_fdr ------------------------------------------------------------------

def test_bh_fdr_matches_scipy_1d():
    rng = np.random.default_rng(5)
    p = rng.uniform(size=200)
    q = bh_fdr(p)
    q_ref = false_discovery_control(p, method="bh")
    np.testing.assert_allclose(q, q_ref)


def test_bh_fdr_matches_scipy_with_ties():
    p = np.array([0.01, 0.2, 0.2, 0.03, 0.2, 0.9, 0.001, 0.2])
    q = bh_fdr(p)
    q_ref = false_discovery_control(p, method="bh")
    np.testing.assert_allclose(q, q_ref)


def test_bh_fdr_matches_scipy_2d():
    rng = np.random.default_rng(6)
    p = rng.uniform(size=(N_LAYERS, N_HEADS))
    q = bh_fdr(p)
    q_ref = false_discovery_control(p.ravel(), method="bh").reshape(p.shape)
    np.testing.assert_allclose(q, q_ref)
    assert q.shape == p.shape


def test_bh_fdr_q_at_least_p_and_at_most_one():
    rng = np.random.default_rng(7)
    p = rng.uniform(size=500)
    q = bh_fdr(p)
    assert np.all(q >= p - 1e-12)
    assert np.all(q <= 1.0)


def test_bh_fdr_known_hand_example():
    p = np.array([0.01, 0.04, 0.03])
    q = bh_fdr(p)
    np.testing.assert_allclose(q, [0.03, 0.04, 0.04])


# ---- split_half ----------------------------------------------------------

def test_split_half_deterministic_for_seed():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(20, N_LAYERS, N_HEADS))
    r1 = split_half(x, seed=42)
    r2 = split_half(x, seed=42)
    for key in r1:
        np.testing.assert_array_equal(r1[key], r2[key])


def test_split_half_differs_across_seeds():
    rng = np.random.default_rng(9)
    x = rng.normal(size=(40, N_LAYERS, N_HEADS))
    r1 = split_half(x, seed=1)
    r2 = split_half(x, seed=2)
    assert not np.array_equal(r1["mean_diff"], r2["mean_diff"])


def test_split_half_matches_reference_permutation_and_handles_odd_n():
    n = 7  # odd -> halves of n//2 = 3, last sequence dropped
    rng = np.random.default_rng(10)
    x = rng.normal(size=(n, 4, 3))
    seed = 123

    result = split_half(x, seed)

    perm = np.random.default_rng(seed).permutation(n)
    half = n // 2
    expected = compare_conditions(x[perm[:half]], x[perm[half : 2 * half]])
    for key in expected:
        np.testing.assert_array_equal(result[key], expected[key])
    # sanity: U is bounded by half*half for a 3-vs-3 comparison
    assert np.all(result["U"] <= half * half)


def test_split_half_uses_given_compare():
    rng = np.random.default_rng(12)
    labels = rng.integers(0, 5, size=(9, 4, 3))
    seed = 5

    result = split_half(labels, seed, compare=lambda a, b: label_shift(a, b, n_labels=5))

    perm = np.random.default_rng(seed).permutation(9)
    expected = label_shift(labels[perm[:4]], labels[perm[4:8]], n_labels=5)
    for key in expected:
        np.testing.assert_array_equal(result[key], expected[key])


# ---- label_shift -------------------------------------------------------------

def test_label_shift_identical_mix_is_zero():
    rng = np.random.default_rng(13)
    x = rng.integers(0, 5, size=(10, N_LAYERS, N_HEADS))
    result = label_shift(x, x[::-1], n_labels=5)
    np.testing.assert_allclose(result["tv"], 0.0)
    np.testing.assert_allclose(result["frac_diff"], 0.0)


def test_label_shift_disjoint_labels_is_one():
    x = np.zeros((6, 2, 2), dtype=np.int8)
    y = np.full((4, 2, 2), 3, dtype=np.int8)
    result = label_shift(x, y, n_labels=5)
    np.testing.assert_allclose(result["tv"], 1.0)


def test_label_shift_hand_example_and_sign():
    # x: 3/4 label 1, 1/4 label 2; y: 1/4 label 1, 3/4 label 2 -> TV = 0.5
    x = np.array([1, 1, 1, 2])[:, None]
    y = np.array([1, 2, 2, 2])[:, None]
    result = label_shift(x, y, n_labels=3)
    np.testing.assert_allclose(result["frac_diff"][:, 0], [0.0, 0.5, -0.5])
    np.testing.assert_allclose(result["tv"], [0.5])
    assert result["frac_diff"].shape == (3, 1)


def test_label_shift_shape_mismatch_raises():
    with pytest.raises(ValueError):
        label_shift(np.zeros((3, 2, 2), dtype=int), np.zeros((3, 2, 3), dtype=int), n_labels=5)


# ---- bootstrap_label_percentages -----------------------------------------

def test_bootstrap_pct_sums_to_100():
    rng = np.random.default_rng(11)
    labels = rng.integers(0, 5, size=(30, N_LAYERS, N_HEADS)).astype(np.int8)
    result = bootstrap_label_percentages(labels, n_labels=5, iters=50, seed=0, ci=0.95)
    assert result["pct"].sum() == pytest.approx(100.0)
    assert np.all(result["boot"].sum(axis=1) == pytest.approx(100.0))


def test_bootstrap_constant_labels_gives_zero_width_ci():
    n_seq = 15
    labels = np.zeros((n_seq, 3, 4), dtype=np.int8)
    labels[:, 0, 0] = 2  # one head always labelled 2, rest always labelled 0
    result = bootstrap_label_percentages(labels, n_labels=5, iters=100, seed=0, ci=0.95)

    np.testing.assert_allclose(result["lo"], result["pct"])
    np.testing.assert_allclose(result["hi"], result["pct"])
    np.testing.assert_allclose(result["boot"], np.broadcast_to(result["pct"], result["boot"].shape))


def test_bootstrap_deterministic_for_seed():
    rng = np.random.default_rng(12)
    labels = rng.integers(0, 5, size=(25, 3, 4)).astype(np.int8)
    r1 = bootstrap_label_percentages(labels, n_labels=5, iters=40, seed=99, ci=0.9)
    r2 = bootstrap_label_percentages(labels, n_labels=5, iters=40, seed=99, ci=0.9)
    for key in r1:
        np.testing.assert_array_equal(r1[key], r2[key])


def test_bootstrap_tie_rule_lowest_code_wins():
    # single head (1 layer x 1 head), 4 sequences split evenly between codes 1 and 3
    labels = np.array([1, 3, 1, 3], dtype=np.int8).reshape(4, 1, 1)
    result = bootstrap_label_percentages(labels, n_labels=5, iters=10, seed=0, ci=0.95)
    expected_pct = np.zeros(5)
    expected_pct[1] = 100.0  # tie between codes 1 and 3 -> lowest code (1) wins
    np.testing.assert_allclose(result["pct"], expected_pct)


def test_bootstrap_ci_contains_pct():
    rng = np.random.default_rng(13)
    labels = rng.integers(0, 5, size=(50, 3, 4)).astype(np.int8)
    result = bootstrap_label_percentages(labels, n_labels=5, iters=200, seed=7, ci=0.9)
    assert np.all(result["lo"] <= result["pct"] + 1e-9)
    assert np.all(result["pct"] <= result["hi"] + 1e-9)


def test_bootstrap_shapes():
    rng = np.random.default_rng(14)
    n_labels = 5
    iters = 30
    labels = rng.integers(0, n_labels, size=(10, 3, 4)).astype(np.int8)
    result = bootstrap_label_percentages(labels, n_labels=n_labels, iters=iters, seed=0, ci=0.95)
    assert result["pct"].shape == (n_labels,)
    assert result["lo"].shape == (n_labels,)
    assert result["hi"].shape == (n_labels,)
    assert result["boot"].shape == (iters, n_labels)


def test_bootstrap_mean_ci_brackets_grand_mean_and_is_deterministic():
    rng = np.random.default_rng(3)
    x = rng.normal(0.5, 0.1, size=(400, 12, 12))
    mean, lo, hi = bootstrap_mean_ci(x, iters=500, seed=1, ci=0.95)
    assert mean == pytest.approx(x.mean())
    assert lo < mean < hi
    assert (mean, lo, hi) == bootstrap_mean_ci(x, iters=500, seed=1, ci=0.95)


def test_bootstrap_mean_ci_constant_input_has_zero_width():
    mean, lo, hi = bootstrap_mean_ci(np.full((50, 3, 3), 0.25), iters=100, seed=0, ci=0.95)
    assert mean == lo == hi == pytest.approx(0.25)
