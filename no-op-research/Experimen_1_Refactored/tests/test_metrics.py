"""Tests for exp1.metrics.compute_metrics."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from exp1.metrics import compute_metrics


@pytest.mark.parametrize("n", [2, 5, 8, 33, 100])
def test_entropy_uniform(n):
    attn = np.full(n, 1.0 / n)
    out = compute_metrics(attn)
    assert out["entropy"].shape == ()
    assert out["entropy"] == pytest.approx(np.log(n))


@pytest.mark.parametrize("q", [1, 3, 7])
def test_entropy_onehot_no_warnings(q):
    key_len = 8
    attn = np.zeros(key_len)
    attn[q] = 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = compute_metrics(attn, query_index=q)
    assert out["entropy"] == 0.0


def test_hand_made_row_picks_right_indices():
    # key_len = 6, q defaults to key_len - 1 = 5.
    attn = np.array([0.1, 0.2, 0.05, 0.05, 0.3, 0.3])
    out = compute_metrics(attn)
    assert out["sink_mass"] == pytest.approx(0.1)  # index 0
    assert out["prev_mass"] == pytest.approx(0.3)  # index q - 1 = 4
    assert out["self_mass"] == pytest.approx(0.3)  # index q = 5
    assert out["max_weight"] == pytest.approx(0.3)
    expected_entropy = -np.sum(attn * np.log(attn))
    assert out["entropy"] == pytest.approx(expected_entropy)


def test_hand_made_row_explicit_query_index():
    # key_len = 8, explicit q = 3.
    attn = np.array([0.5, 0.0, 0.1, 0.2, 0.0, 0.0, 0.0, 0.2])
    out = compute_metrics(attn, query_index=3)
    assert out["sink_mass"] == pytest.approx(0.5)
    assert out["prev_mass"] == pytest.approx(0.1)  # index q - 1 = 2
    assert out["self_mass"] == pytest.approx(0.2)  # index q = 3
    assert out["max_weight"] == pytest.approx(0.5)


def test_4d_batch_shape():
    rng = np.random.default_rng(0)
    n_seq, n_layers, n_heads, key_len = 4, 3, 2, 8
    raw = rng.random((n_seq, n_layers, n_heads, key_len))
    attn = raw / raw.sum(axis=-1, keepdims=True)
    out = compute_metrics(attn)
    for name, arr in out.items():
        assert arr.shape == (n_seq, n_layers, n_heads), name
        assert arr.dtype == np.float64, name


def test_query_index_none_equals_last_equals_negative_one():
    rng = np.random.default_rng(1)
    key_len = 33
    raw = rng.random((5, key_len))
    attn = raw / raw.sum(axis=-1, keepdims=True)

    out_none = compute_metrics(attn, query_index=None)
    out_last = compute_metrics(attn, query_index=key_len - 1)
    out_neg = compute_metrics(attn, query_index=-1)

    for name in out_none:
        np.testing.assert_array_equal(out_none[name], out_last[name])
        np.testing.assert_array_equal(out_none[name], out_neg[name])


def test_query_index_zero_raises():
    attn = np.full(8, 1.0 / 8)
    with pytest.raises(ValueError):
        compute_metrics(attn, query_index=0)


def test_query_index_out_of_range_raises():
    attn = np.full(8, 1.0 / 8)
    with pytest.raises(ValueError):
        compute_metrics(attn, query_index=8)
