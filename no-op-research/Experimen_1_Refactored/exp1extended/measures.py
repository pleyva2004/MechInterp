"""Attention-score measures for exp1extended: log-odds ("logit") statistics computed directly
from pre-softmax attention scores.

We work from scores rather than from the post-softmax pattern because saved Exp 1 attention
contains exact zeros (very common for a 33-key row): log(0) is -inf, so any "logit" built out of
probabilities would blow up. Scores never hit those exact zeros, so log-odds computed from them
(via logsumexp) stay finite.

All functions take `s`, the final-query row of pre-softmax attention scores, shape [..., key_len],
already scaled by 1/sqrt(d_head) (this is exactly what TransformerLens's hook_attn_scores holds,
after the causal mask -- which is a no-op on the final query row, since it sees every key). The
query's own key position is q = key_len - 1.

Dtype convention: every function promotes its input to float64 internally for numerical stability
(matters most for `logsumexp` with extreme scores), matching "logits are computed from scores in
float64 and returned as float32". `logit_split` is the one exception: it returns the float64
decomposition (s0, lse_rest) so `sink_logit` can be reconstructed by subtracting them without an
intermediate rounding step. `other_argmax` returns int8 key positions, not a logit.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp as _scipy_logsumexp

from exp1.metrics import classify, compute_metrics

METRICS: tuple[str, ...] = ("sink_logit", "prev_logit", "other_gap", "entropy", "sink_mass", "prev_mass")


def logsumexp(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """log(sum(exp(x))) along `axis`, float64 internally; finite for any finite x (no overflow)."""
    return _scipy_logsumexp(np.asarray(x, dtype=np.float64), axis=axis)


def logit_split(s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(s0, lse_rest), float64, with s0 = s[..., 0] and lse_rest = logsumexp(s[..., 1:]).

    sink_logit(s) == (s0 - lse_rest).astype(float32) exactly (same subtraction, done once).
    """
    s = np.asarray(s, dtype=np.float64)
    if s.shape[-1] < 2:
        raise ValueError(f"s needs key_len >= 2 for logit_split, got shape {s.shape}")
    return s[..., 0], logsumexp(s[..., 1:], axis=-1)


def sink_logit(s: np.ndarray) -> np.ndarray:
    """log(p0 / (1 - p0)) where p = softmax(s): s0 - LSE(s[1:]). float32."""
    s0, lse_rest = logit_split(s)
    return (s0 - lse_rest).astype(np.float32)


def prev_logit(s: np.ndarray) -> np.ndarray:
    """log(p_{q-1} / (1 - p_{q-1})): s[q-1] - LSE(s[j != q-1]). float32."""
    s = np.asarray(s, dtype=np.float64)
    key_len = s.shape[-1]
    q = key_len - 1
    if q - 1 < 0:
        raise ValueError(f"key_len={key_len} too small for prev_logit (need q - 1 >= 0)")
    rest = np.delete(s, q - 1, axis=-1)
    return (s[..., q - 1] - logsumexp(rest, axis=-1)).astype(np.float32)


def other_gap(s: np.ndarray) -> np.ndarray:
    """max_{1<=j<=q-2} s_j - s_0. float32. Requires key_len >= 4 (q - 2 >= 1)."""
    s = np.asarray(s, dtype=np.float64)
    sub = _other_range(s)
    return (sub.max(axis=-1) - s[..., 0]).astype(np.float32)


def other_argmax(s: np.ndarray) -> np.ndarray:
    """Absolute key position j in [1, q-2] attaining max s_j (ties -> lowest j, np.argmax semantics). int8."""
    s = np.asarray(s)
    sub = _other_range(s)
    return (np.argmax(sub, axis=-1) + 1).astype(np.int8)


def attractor_scores(s: np.ndarray) -> np.ndarray:
    """s[..., 1:q-1] - s[..., :1]: keys 1..q-2 (context X) relative to BOS. float32, shape [..., q-2]."""
    s = np.asarray(s, dtype=np.float64)
    sub = _other_range(s)
    return (sub - s[..., :1]).astype(np.float32)


def _other_range(s: np.ndarray) -> np.ndarray:
    """s[..., 1:q-1], the "other" context range (keys 1..q-2); raises if that range is empty."""
    key_len = s.shape[-1]
    q = key_len - 1
    sub = s[..., 1 : q - 1]
    if sub.shape[-1] < 1:
        raise ValueError(f"key_len={key_len} too small: need q - 2 >= 1 (q = key_len - 1)")
    return sub


def qk_scores(q_vec: np.ndarray, k_vec: np.ndarray, d_head: int) -> np.ndarray:
    """q.k / sqrt(d_head): q_vec [..., d_head], k_vec [..., key_len, d_head] -> [..., key_len]. float32."""
    q64 = np.asarray(q_vec, dtype=np.float64)
    k64 = np.asarray(k_vec, dtype=np.float64)
    if q64.shape[-1] != d_head or k64.shape[-1] != d_head:
        raise ValueError(f"expected last dim d_head={d_head}, got q_vec {q64.shape}, k_vec {k64.shape}")
    scores = (q64[..., None, :] * k64).sum(axis=-1) / np.sqrt(d_head)
    return scores.astype(np.float32)


def summarize(scores: np.ndarray, pattern: np.ndarray, diffuse_threshold: float) -> dict[str, np.ndarray]:
    """scores, pattern: [n, L, H, key_len], final-query rows (scores pre-softmax, pattern post-softmax).

    Returns float32 [n, L, H] arrays for each name in METRICS, plus int8 "label" (from
    exp1.metrics.classify, so labels match Exp 1 exactly) and int8 "other_argmax".
    """
    if scores.shape != pattern.shape:
        raise ValueError(f"scores {scores.shape} and pattern {pattern.shape} must have the same shape")
    key_len = scores.shape[-1]
    q = key_len - 1

    out: dict[str, np.ndarray] = {
        "sink_logit": sink_logit(scores),
        "prev_logit": prev_logit(scores),
        "other_gap": other_gap(scores),
    }
    pattern_metrics = compute_metrics(pattern, query_index=q)  # 0 log 0 := 0, handled there
    out["entropy"] = pattern_metrics["entropy"].astype(np.float32)
    out["sink_mass"] = pattern_metrics["sink_mass"].astype(np.float32)
    out["prev_mass"] = pattern_metrics["prev_mass"].astype(np.float32)
    out["label"] = classify(pattern, diffuse_threshold, q)
    out["other_argmax"] = other_argmax(scores)
    return out
