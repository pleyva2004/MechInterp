"""Tests for exp1.metrics.classify."""
from __future__ import annotations

import numpy as np

from exp1.metrics import LABELS, classify

DIFFUSE = LABELS.index("Diffuse")
SINK = LABELS.index("Sink")
PREVIOUS = LABELS.index("Previous")
SELF = LABELS.index("Self")
OTHER = LABELS.index("Other")

THRESHOLD = 0.2
KEY_LEN = 8  # q defaults to KEY_LEN - 1 = 7


def test_diffuse():
    # Every entry below threshold -> Diffuse, regardless of argmax location.
    attn = np.full(KEY_LEN, 1.0 / KEY_LEN)  # max = 0.125 < 0.2
    assert classify(attn, THRESHOLD) == DIFFUSE


def test_sink():
    attn = np.zeros(KEY_LEN)
    attn[0] = 0.5
    attn[1:] = 0.5 / (KEY_LEN - 1)
    assert classify(attn, THRESHOLD) == SINK


def test_previous():
    q = KEY_LEN - 1
    attn = np.zeros(KEY_LEN)
    attn[q - 1] = 0.5
    attn[0] = 0.1
    attn[q] = 0.1
    attn[1] = 0.3
    assert classify(attn, THRESHOLD) == PREVIOUS


def test_self():
    q = KEY_LEN - 1
    attn = np.zeros(KEY_LEN)
    attn[q] = 0.5
    attn[0] = 0.1
    attn[q - 1] = 0.1
    attn[1] = 0.3
    assert classify(attn, THRESHOLD) == SELF


def test_other():
    q = KEY_LEN - 1
    attn = np.zeros(KEY_LEN)
    attn[2] = 0.5  # not 0, not q-1, not q
    attn[0] = 0.1
    attn[q - 1] = 0.1
    attn[q] = 0.1
    attn[1] = 0.2
    assert classify(attn, THRESHOLD) == OTHER


def test_max_weight_exactly_threshold_is_not_diffuse():
    q = KEY_LEN - 1
    attn = np.full(KEY_LEN, (1.0 - THRESHOLD) / (KEY_LEN - 1))  # each < THRESHOLD
    attn[q] = THRESHOLD  # the max entry, exactly equal to the threshold
    assert attn.argmax() == q
    assert attn.max() == THRESHOLD
    assert classify(attn, THRESHOLD) == SELF  # strict < means this is NOT Diffuse


def test_diffuse_precedence_over_sink():
    # argmax is index 0, but max_weight < threshold -> Diffuse wins (precedence).
    attn = np.full(KEY_LEN, 1.0 / KEY_LEN)
    attn[0] += 1e-6  # nudge argmax to 0 without crossing threshold
    assert attn.argmax() == 0
    assert attn.max() < THRESHOLD
    assert classify(attn, THRESHOLD) == DIFFUSE


def test_tie_index0_and_qminus1_resolves_to_sink():
    q = KEY_LEN - 1
    attn = np.zeros(KEY_LEN)
    attn[0] = 0.5
    attn[q - 1] = 0.5  # tie with index 0; np.argmax picks the lower index (0)
    assert classify(attn, THRESHOLD) == SINK


def test_non_default_query_index_previous_and_self():
    # Causal-mask-like row: length 8, q = 5, entries beyond q are 0.
    key_len = 8
    q = 5

    attn_prev = np.zeros(key_len)
    attn_prev[q - 1] = 0.6
    attn_prev[0] = 0.4
    assert classify(attn_prev, THRESHOLD, query_index=q) == PREVIOUS

    attn_self = np.zeros(key_len)
    attn_self[q] = 0.6
    attn_self[0] = 0.4
    assert classify(attn_self, THRESHOLD, query_index=q) == SELF


def test_int8_dtype_and_batch_shape():
    rng = np.random.default_rng(0)
    n_seq, n_layers, n_heads, key_len = 4, 2, 3, 8
    raw = rng.random((n_seq, n_layers, n_heads, key_len))
    attn = raw / raw.sum(axis=-1, keepdims=True)
    out = classify(attn, THRESHOLD)
    assert out.shape == (n_seq, n_layers, n_heads)
    assert out.dtype == np.int8
