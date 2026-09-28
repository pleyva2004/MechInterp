"""Tests for exp1.aggregate.aggregate_heads."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from exp1.aggregate import aggregate_heads
from exp1.metrics import LABELS, METRICS, classify, compute_metrics

N_SEQ, N_LAYERS, N_HEADS, KEY_LEN = 4, 2, 3, 6
THRESHOLD = 0.2
Q = KEY_LEN - 1  # 5


def _row(argmax_idx, max_val=0.6, key_len=KEY_LEN):
    """A row with its max mass at argmax_idx and the rest spread thinly (< threshold)."""
    rest = (1.0 - max_val) / (key_len - 1)
    row = np.full(key_len, rest)
    row[argmax_idx] = max_val
    return row


def _diffuse_row(key_len=KEY_LEN):
    return np.full(key_len, 1.0 / key_len)  # max = 1/6 < 0.2


def _build_attn():
    """4 sequences x 2 layers x 3 heads, with fully controlled per-head labels.

    head (layer=0, head=0): labels [Sink, Sink, Sink, Previous]      -> mode Sink, consistency 0.75
    head (layer=0, head=1): labels [Self, Self, Other, Other]        -> tie -> mode Self (earlier in LABELS)
    head (layer=0, head=2): labels [Diffuse]*4                       -> mode Diffuse, consistency 1.0
    head (layer=1, head=0): labels [Previous]*4                      -> mode Previous, consistency 1.0
    head (layer=1, head=1): labels [Other]*4                         -> mode Other, consistency 1.0
    head (layer=1, head=2): labels [Sink, Previous, Self, Other]     -> 4-way tie -> mode Sink
    """
    attn = np.zeros((N_SEQ, N_LAYERS, N_HEADS, KEY_LEN))

    # (0, 0): Sink, Sink, Sink, Previous
    attn[0, 0, 0] = _row(0)
    attn[1, 0, 0] = _row(0)
    attn[2, 0, 0] = _row(0)
    attn[3, 0, 0] = _row(Q - 1)

    # (0, 1): Self, Self, Other, Other
    attn[0, 0, 1] = _row(Q)
    attn[1, 0, 1] = _row(Q)
    attn[2, 0, 1] = _row(2)  # not 0, Q-1, or Q -> Other
    attn[3, 0, 1] = _row(2)

    # (0, 2): Diffuse x4
    for s in range(N_SEQ):
        attn[s, 0, 2] = _diffuse_row()

    # (1, 0): Previous x4
    for s in range(N_SEQ):
        attn[s, 1, 0] = _row(Q - 1)

    # (1, 1): Other x4
    for s in range(N_SEQ):
        attn[s, 1, 1] = _row(2)

    # (1, 2): Sink, Previous, Self, Other (one of each)
    attn[0, 1, 2] = _row(0)
    attn[1, 1, 2] = _row(Q - 1)
    attn[2, 1, 2] = _row(Q)
    attn[3, 1, 2] = _row(2)

    return attn


def test_row_count_and_order():
    attn = _build_attn()
    df = aggregate_heads(attn, THRESHOLD)
    assert len(df) == N_LAYERS * N_HEADS
    expected_pairs = [(l, h) for l in range(N_LAYERS) for h in range(N_HEADS)]
    assert list(zip(df["layer"], df["head"])) == expected_pairs


def test_column_order_no_id_cols():
    attn = _build_attn()
    df = aggregate_heads(attn, THRESHOLD)
    expected = ["layer", "head", "mode_label", "consistency",
                "frac_diffuse", "frac_sink", "frac_previous", "frac_self", "frac_other"]
    for m in METRICS:
        expected += [f"mean_{m}", f"std_{m}"]
    assert list(df.columns) == expected


def test_id_cols_prepended():
    attn = _build_attn()
    id_cols = {"model": "gpt2-small", "backend": "transformer_lens", "condition": "A_rare"}
    df = aggregate_heads(attn, THRESHOLD, id_cols=id_cols)
    assert list(df.columns)[:3] == ["model", "backend", "condition"]
    assert (df["model"] == "gpt2-small").all()
    assert (df["backend"] == "transformer_lens").all()
    assert (df["condition"] == "A_rare").all()
    assert list(df.columns)[3:5] == ["layer", "head"]


def test_mode_label_and_consistency():
    attn = _build_attn()
    df = aggregate_heads(attn, THRESHOLD)

    row = df[(df["layer"] == 0) & (df["head"] == 0)].iloc[0]
    assert row["mode_label"] == "Sink"
    assert row["consistency"] == pytest.approx(0.75)

    row = df[(df["layer"] == 0) & (df["head"] == 2)].iloc[0]
    assert row["mode_label"] == "Diffuse"
    assert row["consistency"] == pytest.approx(1.0)

    row = df[(df["layer"] == 1) & (df["head"] == 0)].iloc[0]
    assert row["mode_label"] == "Previous"
    assert row["consistency"] == pytest.approx(1.0)

    row = df[(df["layer"] == 1) & (df["head"] == 1)].iloc[0]
    assert row["mode_label"] == "Other"
    assert row["consistency"] == pytest.approx(1.0)


def test_mode_label_tie_breaks_to_earliest_label():
    attn = _build_attn()
    df = aggregate_heads(attn, THRESHOLD)

    # (0, 1): 2x Self, 2x Other -> tie; Self precedes Other in LABELS.
    row = df[(df["layer"] == 0) & (df["head"] == 1)].iloc[0]
    assert row["mode_label"] == "Self"
    assert row["consistency"] == pytest.approx(0.5)

    # (1, 2): one of each of Sink/Previous/Self/Other -> 4-way tie (Diffuse absent);
    # Sink is earliest among the tied labels.
    row = df[(df["layer"] == 1) & (df["head"] == 2)].iloc[0]
    assert row["mode_label"] == "Sink"
    assert row["consistency"] == pytest.approx(0.25)


def test_frac_columns_sum_to_one_and_match_counts():
    attn = _build_attn()
    df = aggregate_heads(attn, THRESHOLD)
    frac_cols = ["frac_diffuse", "frac_sink", "frac_previous", "frac_self", "frac_other"]
    sums = df[frac_cols].sum(axis=1)
    np.testing.assert_allclose(sums.to_numpy(), 1.0)

    row = df[(df["layer"] == 0) & (df["head"] == 0)].iloc[0]
    assert row["frac_sink"] == pytest.approx(0.75)
    assert row["frac_previous"] == pytest.approx(0.25)
    assert row["frac_diffuse"] == pytest.approx(0.0)
    assert row["frac_self"] == pytest.approx(0.0)
    assert row["frac_other"] == pytest.approx(0.0)


def test_mean_std_match_numpy():
    attn = _build_attn()
    df = aggregate_heads(attn, THRESHOLD)

    metrics = compute_metrics(attn)  # {name: [n_seq, n_layers, n_heads]}
    for name in METRICS:
        values = metrics[name]
        expected_mean = values.mean(axis=0)
        expected_std = values.std(axis=0, ddof=1)
        for l in range(N_LAYERS):
            for h in range(N_HEADS):
                row = df[(df["layer"] == l) & (df["head"] == h)].iloc[0]
                assert row[f"mean_{name}"] == pytest.approx(expected_mean[l, h])
                assert row[f"std_{name}"] == pytest.approx(expected_std[l, h])


def test_pure_function_no_mutation():
    attn = _build_attn()
    attn_copy = attn.copy()
    aggregate_heads(attn, THRESHOLD)
    np.testing.assert_array_equal(attn, attn_copy)


def test_matches_classify_directly():
    # Cross-check: consistency/mode_label derived independently via classify().
    attn = _build_attn()
    df = aggregate_heads(attn, THRESHOLD)
    labels = classify(attn, THRESHOLD)  # [n_seq, n_layers, n_heads] int8

    for l in range(N_LAYERS):
        for h in range(N_HEADS):
            codes, counts = np.unique(labels[:, l, h], return_counts=True)
            best = codes[np.argmax(counts)]
            expected_label = LABELS[best]
            expected_consistency = counts.max() / N_SEQ
            row = df[(df["layer"] == l) & (df["head"] == h)].iloc[0]
            assert row["mode_label"] == expected_label
            assert row["consistency"] == pytest.approx(expected_consistency)
