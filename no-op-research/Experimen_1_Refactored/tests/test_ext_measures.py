"""Tests for exp1extended.measures and exp1extended.extract.

Synthetic tests check the math against hand-derived/closed-form values. The real-model tests
load GPT-2 small via TransformerLens (from the local HF cache Exp 1 already populated) on <= 8
sequences and check run_final/extract's validation helpers (V1-V4) against Exp 1's saved
attention; they are skipped cleanly (not failed) if the model can't be loaded.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
import yaml

from exp1 import backends, sequences
from exp1.metrics import classify
from exp1extended import extract, measures

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "configs" / "exp1extended.yaml").read_text())
BOS_ID = CONFIG["bos_token_id"]
DIFFUSE_THRESHOLD = CONFIG["diffuse_threshold"]
SAVED_RUN_DIR = ROOT / CONFIG["exp1_runs"]["C_common_clean"]

N_REAL = 8
TARGET_HEADS_SAMPLE = [(0, 6), (7, 7), (10, 11)]  # byte_specific, rare_active x2
ATOL = 1e-5


def _softmax(s: np.ndarray) -> np.ndarray:
    s64 = s.astype(np.float64)
    e = np.exp(s64 - s64.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# Synthetic: sink_logit / prev_logit / logit_split
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scale", [1.0, 20.0, 50.0])
def test_sink_logit_matches_log_odds(scale):
    # At these scales double-precision softmax doesn't saturate to exact 0/1, so log(p/(1-p))
    # computed straight from probabilities is itself well-defined and an independent check.
    rng = np.random.default_rng(0)
    s = rng.normal(size=(7, 33)) * scale
    p = _softmax(s)
    expected = np.log(p[..., 0] / (1.0 - p[..., 0]))
    got = measures.sink_logit(s)
    assert got.dtype == np.float32
    assert np.isfinite(got).all()
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)


def test_sink_logit_extreme_scores_finite():
    # Finite but wildly separated scores: naive softmax-then-log would over/underflow.
    s = np.array([
        [1e6, -1e6, 0.0, 0.0, 0.0],
        [-1e6, 1e6, 1e6, 1e6, 1e6],
        [0.0, 1e-300, -1e-300, 5.0, -5.0],
    ])
    got = measures.sink_logit(s)
    assert np.isfinite(got).all()
    assert not np.isnan(got).any()


def test_sink_logit_matches_lse_definition_at_saturating_scale():
    # At this scale, float64 softmax(s) rounds some probabilities to exact 0.0/1.0, so
    # log(p0 / (1 - p0)) is -inf/nan even though the true log-odds is finite -- exactly why
    # sink_logit works from scores via logsumexp, not from softmax probabilities. Check against
    # the LSE definition directly (independent of measures.logsumexp's implementation).
    from scipy.special import logsumexp as scipy_lse

    rng = np.random.default_rng(0)
    s = rng.normal(size=(7, 33)) * 500.0
    expected = (s[..., 0] - scipy_lse(s[..., 1:].astype(np.float64), axis=-1)).astype(np.float32)
    got = measures.sink_logit(s)
    assert np.isfinite(got).all()
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-3)


@pytest.mark.parametrize("scale", [1.0, 20.0, 50.0])
def test_prev_logit_matches_log_odds(scale):
    rng = np.random.default_rng(1)
    key_len = 10
    q = key_len - 1
    s = rng.normal(size=(6, key_len)) * scale
    p = _softmax(s)
    expected = np.log(p[..., q - 1] / (1.0 - p[..., q - 1]))
    got = measures.prev_logit(s)
    assert got.dtype == np.float32
    assert np.isfinite(got).all()
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)


def test_prev_logit_extreme_scores_finite():
    s = np.array([[1e6, -1e6, 1e6, 0.0, -1e6], [0.0, 0.0, 1e8, 0.0, 0.0]])
    got = measures.prev_logit(s)
    assert np.isfinite(got).all()


def test_prev_logit_matches_lse_definition_at_saturating_scale():
    from scipy.special import logsumexp as scipy_lse

    rng = np.random.default_rng(1)
    key_len = 10
    q = key_len - 1
    s = rng.normal(size=(6, key_len)) * 500.0
    rest = np.delete(s.astype(np.float64), q - 1, axis=-1)
    expected = (s[..., q - 1] - scipy_lse(rest, axis=-1)).astype(np.float32)
    got = measures.prev_logit(s)
    assert np.isfinite(got).all()
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-3)


def test_logit_split_reconstructs_sink_logit():
    rng = np.random.default_rng(2)
    s = rng.normal(size=(5, 4, 33)) * 100
    s0, lse_rest = measures.logit_split(s)
    reconstructed = (s0 - lse_rest).astype(np.float32)
    np.testing.assert_array_equal(reconstructed, measures.sink_logit(s))


# ---------------------------------------------------------------------------
# Synthetic: attractor_scores / other_gap / other_argmax
# ---------------------------------------------------------------------------

def test_attractor_scores_shape_and_values():
    key_len = 8  # q = 7, context range j in [1, q - 2] = [1, 5]
    s = np.arange(key_len, dtype=np.float64)[None, :]  # [1, 8]: 0, 1, ..., 7
    out = measures.attractor_scores(s)
    assert out.shape == (1, key_len - 3)  # q - 2 = 5
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, [[1.0, 2.0, 3.0, 4.0, 5.0]])


def test_other_gap_and_argmax_hand_example():
    # key_len = 6 -> q = 5, "other" range is j in [1, 3].
    s = np.array([0.0, 1.0, 5.0, -2.0, 3.0, 0.5])
    gap = measures.other_gap(s)
    argmax = measures.other_argmax(s)
    assert gap == pytest.approx(5.0 - 0.0)  # max(1, 5, -2) - s0
    assert int(argmax) == 2  # position of the 5.0
    assert argmax.dtype == np.int8


def test_other_argmax_ties_resolve_to_lowest_index():
    s = np.array([0.0, 3.0, 3.0, -1.0, 0.0, 0.0])  # tie between j=1 and j=2
    assert int(measures.other_argmax(s)) == 1


# ---------------------------------------------------------------------------
# Synthetic: qk_scores
# ---------------------------------------------------------------------------

def test_qk_scores_matches_manual_einsum():
    rng = np.random.default_rng(3)
    d_head, key_len = 64, 33
    q_vec = rng.normal(size=(4, 3, d_head))
    k_vec = rng.normal(size=(4, 3, key_len, d_head))
    expected = np.einsum("...d,...kd->...k", q_vec, k_vec) / np.sqrt(d_head)
    got = measures.qk_scores(q_vec, k_vec, d_head)
    assert got.dtype == np.float32
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Synthetic: summarize (entropy with exact zeros, labels, shapes/dtypes)
# ---------------------------------------------------------------------------

def test_summarize_entropy_with_exact_zeros_no_warnings():
    key_len = 33
    q = key_len - 1
    pattern = np.zeros((2, 1, 1, key_len), dtype=np.float32)
    pattern[0, 0, 0, 0] = 1.0  # all mass on sink -> entropy 0, many exact zeros elsewhere
    pattern[1, 0, 0, q] = 0.5
    pattern[1, 0, 0, q - 1] = 0.5
    scores = np.zeros_like(pattern)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = measures.summarize(scores, pattern, diffuse_threshold=0.2)

    assert out["entropy"][0, 0, 0] == pytest.approx(0.0)
    assert out["entropy"][1, 0, 0] == pytest.approx(-2 * 0.5 * np.log(0.5))
    assert out["entropy"].dtype == np.float32


def test_summarize_shapes_dtypes_and_labels_match_classify():
    rng = np.random.default_rng(4)
    n, n_layers, n_heads, key_len = 3, 2, 2, 12
    raw = rng.random((n, n_layers, n_heads, key_len))
    pattern = (raw / raw.sum(axis=-1, keepdims=True)).astype(np.float32)
    scores = (rng.normal(size=(n, n_layers, n_heads, key_len)) * 10).astype(np.float32)

    out = measures.summarize(scores, pattern, diffuse_threshold=0.2)

    for name in measures.METRICS:
        assert out[name].shape == (n, n_layers, n_heads), name
        assert out[name].dtype == np.float32, name
    assert out["label"].shape == (n, n_layers, n_heads)
    assert out["label"].dtype == np.int8
    assert out["other_argmax"].shape == (n, n_layers, n_heads)
    assert out["other_argmax"].dtype == np.int8
    np.testing.assert_array_equal(out["label"], classify(pattern, 0.2, key_len - 1))


# ---------------------------------------------------------------------------
# Real model: GPT-2 small via TransformerLens, <= 8 sequences.
# Skipped cleanly (not failed) if the model can't be loaded.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def gpt2_small():
    backends.set_determinism(0)
    try:
        return backends.load_model("transformer_lens", "gpt2-small", "cpu", "float32")
    except Exception as exc:  # pragma: no cover - depends on local environment/cache
        pytest.skip(f"could not load gpt2-small via TransformerLens: {exc}")


@pytest.fixture(scope="module")
def real_data():
    seqs = np.load(ROOT / "data" / "sequences" / "C_common_clean.npy")[:N_REAL]
    tokens = sequences.prepend_bos(seqs, BOS_ID)
    saved_attn = np.load(SAVED_RUN_DIR / "C_common_clean_attn.npy")[:N_REAL]
    return tokens, saved_attn


@pytest.fixture(scope="module")
def real_run(gpt2_small, real_data):
    tokens, _ = real_data
    backends.set_determinism(0)
    return extract.run_final(
        gpt2_small, tokens, batch_size=CONFIG["batch_size"], target_heads=TARGET_HEADS_SAMPLE,
        diffuse_threshold=DIFFUSE_THRESHOLD, keep_full=True, keep_qk=True,
    )


def test_real_pattern_bitmatches_saved_attn(real_run, real_data):
    # V1. Saved with batch_size 250 (1000 sequences); here batch_size == CONFIG["batch_size"]
    # (250) but only N_REAL=8 rows, so it's a single batch of 8 -- see
    # test_real_batch_size_does_not_affect_bitmatch below for the composition check this
    # anticipates: on this CPU + deterministic-algorithms setup, results are bit-identical
    # regardless of batch size/composition, so a small isolated batch still bit-matches the
    # 250-row batches Exp 1 saved.
    _, saved_attn = real_data
    ok, max_abs_diff = extract.check_bitmatch(real_run["pattern"], saved_attn)
    assert ok, f"max abs diff {max_abs_diff}"


def test_real_batch_size_does_not_affect_bitmatch(gpt2_small, real_data):
    # Explicit batch-size-sensitivity check: re-extract the same 8 rows split across 3 batches
    # (sizes 3, 3, 2) instead of 1, and confirm the result is still bit-identical to both the
    # single-batch extraction and Exp 1's saved (batch_size=250) attention.
    tokens, saved_attn = real_data
    backends.set_determinism(0)
    multi_batch = extract.run_final(
        gpt2_small, tokens, batch_size=3, target_heads=TARGET_HEADS_SAMPLE,
        diffuse_threshold=DIFFUSE_THRESHOLD, keep_full=True,
    )
    ok, max_abs_diff = extract.check_bitmatch(multi_batch["pattern"], saved_attn)
    assert ok, f"batch_size=3 vs saved: max abs diff {max_abs_diff}"


def test_real_check_softmax(real_run):
    ok, max_abs_diff = extract.check_softmax(real_run["scores"], real_run["pattern"], atol=ATOL)
    assert ok, f"max abs diff {max_abs_diff}"


def test_real_check_qk(real_run, gpt2_small):
    ok, max_abs_diff = extract.check_qk(
        real_run["q"], real_run["k"], real_run["target_scores"], gpt2_small.cfg.d_head, atol=ATOL
    )
    assert ok, f"max abs diff {max_abs_diff}"


def test_real_check_bos_key(real_run):
    ok, max_abs_diff = extract.check_bos_key(real_run["k"], atol=ATOL)
    assert ok, f"max abs diff {max_abs_diff}"


def test_real_summarize_labels_match_exp1_classify(real_run, real_data):
    _, saved_attn = real_data
    expected = classify(saved_attn, DIFFUSE_THRESHOLD, -1)
    np.testing.assert_array_equal(real_run["label"], expected)
