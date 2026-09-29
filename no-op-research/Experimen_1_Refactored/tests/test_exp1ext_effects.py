"""Tests for exp1ext.effects."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from exp1ext.effects import (
    attention_received, bootstrap_mean_per_unit, decompose_2x2, per_token_effects, split_reliability,
)


# ---- bootstrap_mean_per_unit -------------------------------------------------

def test_bootstrap_mean_per_unit_matches_nanmean():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(50, 4))
    x[3, 2] = np.nan
    mean, lo, hi = bootstrap_mean_per_unit(x, iters=200, seed=0, ci=0.95)
    np.testing.assert_allclose(mean, np.nanmean(x, axis=0))
    assert mean.shape == (4,)
    assert np.all(lo <= mean + 1e-9) and np.all(mean <= hi + 1e-9)


def test_bootstrap_mean_per_unit_constant_data_zero_width_ci():
    x = np.full((30, 3), 2.5)
    mean, lo, hi = bootstrap_mean_per_unit(x, iters=50, seed=0, ci=0.95)
    np.testing.assert_allclose(mean, 2.5)
    np.testing.assert_allclose(lo, 2.5)
    np.testing.assert_allclose(hi, 2.5)


def test_bootstrap_mean_per_unit_deterministic_for_seed():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(40, 2))
    r1 = bootstrap_mean_per_unit(x, iters=80, seed=5, ci=0.9)
    r2 = bootstrap_mean_per_unit(x, iters=80, seed=5, ci=0.9)
    for a, b in zip(r1, r2):
        np.testing.assert_array_equal(a, b)


# ---- decompose_2x2 -----------------------------------------------------------

def test_decompose_2x2_exact_on_constant_cells():
    n = 20
    a = np.array([1.0, 2.0])
    b = np.array([1.5, 2.5])
    c = np.array([1.2, 2.2])
    d = np.array([2.0, 3.0])
    cells = {
        "AA": np.tile(a, (n, 1)), "AC": np.tile(b, (n, 1)),
        "CA": np.tile(c, (n, 1)), "CC": np.tile(d, (n, 1)),
    }
    result = decompose_2x2(cells, iters=50, seed=0, ci=0.95)

    np.testing.assert_allclose(result["cell_AA"]["est"], a)
    np.testing.assert_allclose(result["total"]["est"], d - a)
    np.testing.assert_allclose(result["query"]["est"], b - a)
    np.testing.assert_allclose(result["context"]["est"], c - a)
    np.testing.assert_allclose(result["interaction"]["est"], (d - a) - (b - a) - (c - a))

    # constant cells -> every bootstrap resample reproduces the same means -> zero-width CI
    for key in result:
        np.testing.assert_allclose(result[key]["lo"], result[key]["est"], atol=1e-10)
        np.testing.assert_allclose(result[key]["hi"], result[key]["est"], atol=1e-10)


def test_decompose_2x2_total_equals_sum_of_parts():
    rng = np.random.default_rng(7)
    n = 100
    cells = {k: rng.normal(size=(n, 4)) for k in ("AA", "AC", "CA", "CC")}
    result = decompose_2x2(cells, iters=100, seed=1, ci=0.9)
    np.testing.assert_allclose(
        result["total"]["est"],
        result["query"]["est"] + result["context"]["est"] + result["interaction"]["est"],
    )


def test_decompose_2x2_ci_brackets_est_on_random_data():
    rng = np.random.default_rng(8)
    n = 150
    cells = {k: rng.normal(size=(n, 3)) for k in ("AA", "AC", "CA", "CC")}
    result = decompose_2x2(cells, iters=200, seed=2, ci=0.9)
    for key, vals in result.items():
        assert np.all(vals["lo"] <= vals["est"] + 1e-9), key
        assert np.all(vals["est"] <= vals["hi"] + 1e-9), key


def test_decompose_2x2_paired_resampling_gives_zero_width_query_ci():
    # AC - AA is constant per row (query effect) even though AA/AC vary a lot row to row;
    # CA and CC are independent of AA, so total/context/interaction stay wide. Joint (paired)
    # resampling must recover the exact per-row constant for query, with zero-width CI.
    rng = np.random.default_rng(42)
    n = 200
    shift = np.array([0.5, -0.3, 1.0])
    aa = rng.normal(scale=5.0, size=(n, 3))
    ac = aa + shift  # AC - AA == shift for every row, regardless of which rows are resampled
    ca = rng.normal(scale=5.0, size=(n, 3))
    cc = rng.normal(scale=5.0, size=(n, 3))
    cells = {"AA": aa, "AC": ac, "CA": ca, "CC": cc}

    result = decompose_2x2(cells, iters=300, seed=3, ci=0.95)

    np.testing.assert_allclose(result["query"]["est"], shift, atol=1e-9)
    np.testing.assert_allclose(result["query"]["lo"], shift, atol=1e-9)
    np.testing.assert_allclose(result["query"]["hi"], shift, atol=1e-9)
    # sanity: an effect that isn't row-paired-constant keeps a non-trivial CI width
    assert np.all(result["context"]["hi"] - result["context"]["lo"] > 1e-3)


def test_decompose_2x2_missing_cell_raises():
    with pytest.raises(ValueError):
        decompose_2x2({"AA": np.zeros((5, 2)), "AC": np.zeros((5, 2)), "CA": np.zeros((5, 2))}, 10, 0, 0.95)


def test_decompose_2x2_shape_mismatch_raises():
    cells = {"AA": np.zeros((5, 2)), "AC": np.zeros((5, 3)), "CA": np.zeros((5, 2)), "CC": np.zeros((5, 2))}
    with pytest.raises(ValueError):
        decompose_2x2(cells, 10, 0, 0.95)


# ---- per_token_effects --------------------------------------------------------

def _values_with_allnan_column(seed):
    rng = np.random.default_rng(seed)
    K, T, H = 30, 8, 2
    values = rng.normal(size=(K, T, H))
    values[:, 3, :] = np.nan  # column 3 entirely invalid
    scatter = rng.random(size=(K, T, H)) < 0.1
    scatter[:, 3, :] = False
    return np.where(scatter, np.nan, values)


def test_per_token_effects_shapes_and_dtype():
    values = _values_with_allnan_column(0)
    K, T, H = values.shape
    result = per_token_effects(values, iters=60, seed=0, ci=0.95)
    for key in ("mean", "lo", "hi", "n"):
        assert result[key].shape == (T, H)
    assert np.issubdtype(result["n"].dtype, np.integer)


def test_per_token_effects_n_counts_match_manual():
    values = _values_with_allnan_column(1)
    result = per_token_effects(values, iters=40, seed=0, ci=0.95)
    expected_n = np.sum(~np.isnan(values), axis=0)
    np.testing.assert_array_equal(result["n"], expected_n)


def test_per_token_effects_mean_matches_nanmean():
    values = _values_with_allnan_column(2)
    result = per_token_effects(values, iters=40, seed=0, ci=0.95)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # reference nanmean, same all-NaN column
        expected = np.nanmean(values, axis=0)
    np.testing.assert_allclose(result["mean"], expected, equal_nan=True)


def test_per_token_effects_allnan_column_is_nan_with_zero_n_and_no_warnings_leak():
    values = _values_with_allnan_column(3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any leaked RuntimeWarning becomes a test failure
        result = per_token_effects(values, iters=100, seed=0, ci=0.95)

    assert np.all(np.isnan(result["mean"][3]))
    assert np.all(np.isnan(result["lo"][3]))
    assert np.all(np.isnan(result["hi"][3]))
    assert np.all(result["n"][3] == 0)


def test_per_token_effects_ci_brackets_mean():
    values = _values_with_allnan_column(4)
    result = per_token_effects(values, iters=150, seed=0, ci=0.9)
    finite = result["n"] > 0
    assert np.all(result["lo"][finite] <= result["mean"][finite] + 1e-9)
    assert np.all(result["mean"][finite] <= result["hi"][finite] + 1e-9)


def test_per_token_effects_deterministic_for_seed():
    values = _values_with_allnan_column(5)
    r1 = per_token_effects(values, iters=50, seed=9, ci=0.9)
    r2 = per_token_effects(values, iters=50, seed=9, ci=0.9)
    for key in r1:
        np.testing.assert_allclose(r1[key], r2[key], equal_nan=True)


# ---- split_reliability ---------------------------------------------------

def test_split_reliability_identical_halves_gives_r_one():
    rng = np.random.default_rng(9)
    K, T = 40, 25
    per_token = rng.normal(size=T)  # value depends only on the token, not the context
    values = np.tile(per_token, (K, 1))
    r = split_reliability(values, seed=0)
    assert r.shape == ()
    assert r == pytest.approx(1.0, abs=1e-9)


def test_split_reliability_pure_noise_is_small():
    rng = np.random.default_rng(11)
    values = rng.normal(size=(40, 300))
    r = split_reliability(values, seed=1)
    assert abs(r) < 0.3


def test_split_reliability_shape_with_trailing_dims():
    rng = np.random.default_rng(12)
    values = rng.normal(size=(20, 15, 3))
    r = split_reliability(values, seed=2)
    assert r.shape == (3,)


def test_split_reliability_insufficient_tokens_gives_nan():
    rng = np.random.default_rng(13)
    values = rng.normal(size=(10, 2))  # only 2 tokens, below the minimum of 3
    r = split_reliability(values, seed=0)
    assert np.isnan(r)


def test_split_reliability_zero_variance_gives_nan():
    values = np.full((20, 5), 3.0)  # identical across tokens in both halves -> zero variance
    r = split_reliability(values, seed=0)
    assert np.isnan(r)


def test_split_reliability_deterministic_for_seed():
    rng = np.random.default_rng(14)
    values = rng.normal(size=(30, 20))
    r1 = split_reliability(values, seed=6)
    r2 = split_reliability(values, seed=6)
    np.testing.assert_allclose(r1, r2, equal_nan=True)


# ---- attention_received -------------------------------------------------

def test_attention_received_hand_computed():
    bos, query_id = 50256, 99
    tokens = np.array([
        [bos, 5, 7, query_id],
        [bos, 5, 5, query_id],
        [bos, 9, 7, query_id],
    ], dtype=np.int64)

    rows = np.zeros((3, 2, 4), dtype=np.float64)
    # head 0: no guard triggered
    rows[0, 0] = [0.5, 0.2, 0.1, 0.2]
    rows[1, 0] = [0.4, 0.3, 0.2, 0.1]
    rows[2, 0] = [0.6, 0.1, 0.2, 0.1]
    # head 1: row 0's sink is ~1 -> its two occurrences are excluded from the share average
    rows[0, 1] = [0.9999995, 0.0000003, 0.0000001, 0.0000001]
    rows[1, 1] = [0.5, 0.3, 0.1, 0.1]
    rows[2, 1] = [0.5, 0.2, 0.2, 0.1]

    key_positions = np.array([1, 2])
    result = attention_received(tokens, rows, key_positions)

    np.testing.assert_array_equal(result["ids"], [5, 7, 9])
    np.testing.assert_array_equal(result["count"], [3, 2, 1])

    expected_mean_attn_h0 = [(0.2 + 0.3 + 0.2) / 3, (0.1 + 0.2) / 2, 0.1]
    expected_mean_attn_h1 = [(0.0000003 + 0.3 + 0.1) / 3, (0.0000001 + 0.2) / 2, 0.2]
    np.testing.assert_allclose(result["mean_attn"][:, 0], expected_mean_attn_h0, rtol=1e-6)
    np.testing.assert_allclose(result["mean_attn"][:, 1], expected_mean_attn_h1, rtol=1e-6)

    expected_mean_share_h0 = [
        (0.2 / 0.5 + 0.3 / 0.6 + 0.2 / 0.6) / 3,  # id5: row0 pos1, row1 pos1, row1 pos2
        (0.1 / 0.5 + 0.2 / 0.4) / 2,               # id7: row0 pos2, row2 pos2
        0.1 / 0.4,                                  # id9: row2 pos1
    ]
    # head1: row0's occurrences of id5 (pos1) and id7 (pos2) are dropped by the sink guard
    expected_mean_share_h1 = [
        (0.3 / 0.5 + 0.1 / 0.5) / 2,  # id5: only row1 pos1, pos2 (row0 excluded)
        0.2 / 0.5,                     # id7: only row2 pos2 (row0 excluded)
        0.2 / 0.5,                     # id9: row2 pos1
    ]
    np.testing.assert_allclose(result["mean_share"][:, 0], expected_mean_share_h0, rtol=1e-6)
    np.testing.assert_allclose(result["mean_share"][:, 1], expected_mean_share_h1, rtol=1e-6)

    # mean_logratio = mean over occurrences of log(attn_k) - log(sink); no guard/exclusion here,
    # unlike mean_share, since flooring (tested separately) is enough to keep it finite.
    expected_mean_logratio_h0 = [
        np.mean([np.log(0.2) - np.log(0.5), np.log(0.3) - np.log(0.4), np.log(0.2) - np.log(0.4)]),  # id5
        np.mean([np.log(0.1) - np.log(0.5), np.log(0.2) - np.log(0.6)]),                              # id7
        np.log(0.1) - np.log(0.6),                                                                    # id9
    ]
    expected_mean_logratio_h1 = [
        np.mean([
            np.log(0.0000003) - np.log(0.9999995), np.log(0.3) - np.log(0.5), np.log(0.1) - np.log(0.5),
        ]),  # id5
        np.mean([np.log(0.0000001) - np.log(0.9999995), np.log(0.2) - np.log(0.5)]),  # id7
        np.log(0.2) - np.log(0.5),                                                    # id9
    ]
    np.testing.assert_allclose(result["mean_logratio"][:, 0], expected_mean_logratio_h0, rtol=1e-6)
    np.testing.assert_allclose(result["mean_logratio"][:, 1], expected_mean_logratio_h1, rtol=1e-6)


def test_attention_received_logratio_floors_exact_zeros():
    bos, query_id = 50256, 99
    tokens = np.array([[bos, 5, 7, query_id]], dtype=np.int64)
    rows = np.zeros((1, 1, 4))
    rows[0, 0] = [0.0, 0.0, 1.0, 0.0]  # sink and id5's attention are exactly 0

    result = attention_received(tokens, rows, key_positions=np.array([1, 2]))

    eps = np.finfo(np.float32).tiny
    assert np.all(np.isfinite(result["mean_logratio"]))  # no -inf from log(0)
    id5, id7 = 0, 1  # ids sorted: [5, 7]
    # both attn and sink floored to eps -> they cancel exactly
    assert result["mean_logratio"][id5, 0] == pytest.approx(0.0)
    # attn=1.0 (unaffected by flooring), sink floored to eps -> log(1) - log(eps) = -log(eps)
    assert result["mean_logratio"][id7, 0] == pytest.approx(-np.log(eps))


def test_attention_received_excludes_bos_and_query_positions():
    tokens = np.array([[999, 5, 7, 5]], dtype=np.int64)  # id 5 also sits at BOS-adjacent query slot
    rows = np.zeros((1, 1, 4))
    rows[0, 0] = [0.1, 0.2, 0.3, 0.4]
    result = attention_received(tokens, rows, key_positions=np.array([1, 2]))
    np.testing.assert_array_equal(result["ids"], [5, 7])
    np.testing.assert_array_equal(result["count"], [1, 1])  # id 5's query-position occurrence doesn't count
