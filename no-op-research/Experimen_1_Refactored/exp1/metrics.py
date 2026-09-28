"""Per-(sequence, layer, head) attention metrics and categorical labels.

Operates on the attention row of the final query position: attn[..., k] is the
post-softmax weight paid to key position k (key axis is last; any leading shape,
e.g. [n_seq, n_layers, n_heads, key_len]).
"""
from __future__ import annotations

import numpy as np

LABELS: tuple[str, ...] = ("Diffuse", "Sink", "Previous", "Self", "Other")  # label code i == LABELS[i]
METRICS: tuple[str, ...] = ("sink_mass", "prev_mass", "self_mass", "max_weight", "entropy")


def _resolve_query_index(query_index: int | None, key_len: int) -> int:
    """None defaults to the last key position; negative query_index wraps modulo key_len."""
    q = key_len - 1 if query_index is None else query_index
    if q < 0:
        q %= key_len
    if q < 1 or q >= key_len:
        raise ValueError(f"query_index must resolve to 1 <= q < key_len ({key_len}); got q={q}")
    return q


def compute_metrics(attn: np.ndarray, query_index: int | None = None) -> dict[str, np.ndarray]:
    """attn: [..., key_len] post-softmax attention weights (last axis sums to 1).

    Returns {metric: array of shape attn.shape[:-1]}, float64. query_index q defaults
    to key_len - 1 (the final token attending to itself); a negative q wraps modulo
    key_len. Raises ValueError if q resolves outside [1, key_len - 1).
    """
    key_len = attn.shape[-1]
    q = _resolve_query_index(query_index, key_len)
    p = attn.astype(np.float64)

    # 0 * log(0) := 0: compute log/product with FP warnings suppressed, then
    # overwrite the p == 0 entries (which are -inf/nan) with 0 before summing.
    with np.errstate(divide="ignore", invalid="ignore"):
        plogp = p * np.log(p)
    plogp = np.where(p > 0, plogp, 0.0)
    entropy = -plogp.sum(axis=-1)

    return {
        "sink_mass": p[..., 0],
        "prev_mass": p[..., q - 1],
        "self_mass": p[..., q],
        "max_weight": p.max(axis=-1),
        "entropy": entropy,
    }


def classify(attn: np.ndarray, diffuse_threshold: float, query_index: int | None = None) -> np.ndarray:
    """attn: [..., key_len] post-softmax attention weights.

    Returns int8 label codes (indices into LABELS), shape attn.shape[:-1], by precedence:
      Diffuse  if max_weight <  diffuse_threshold (strict)
      Sink     elif argmax == 0
      Previous elif argmax == q - 1
      Self     elif argmax == q
      Other    otherwise
    Ties in argmax resolve to the lowest key index (np.argmax semantics), so e.g. a
    tie between key 0 and key q - 1 resolves to Sink.
    """
    key_len = attn.shape[-1]
    q = _resolve_query_index(query_index, key_len)

    max_weight = attn.max(axis=-1)
    argmax = attn.argmax(axis=-1)

    codes = np.select(
        [
            max_weight < diffuse_threshold,
            argmax == 0,
            argmax == q - 1,
            argmax == q,
        ],
        [
            LABELS.index("Diffuse"),
            LABELS.index("Sink"),
            LABELS.index("Previous"),
            LABELS.index("Self"),
        ],
        default=LABELS.index("Other"),
    )
    return codes.astype(np.int8)
