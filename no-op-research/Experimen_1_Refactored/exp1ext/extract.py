"""Run a model over many sequences and keep only what exp1ext needs.

The full final-query attention for every head ([n, 12, 12, 33] float32) is ~19 KB per sequence, so
for the 10^5-sequence sweeps it is reduced chunk by chunk to per-head metrics, the weight on a few
chosen key positions, and full rows for a handful of heads.
"""
from __future__ import annotations

import numpy as np
import torch

from exp1.backends import check_rows_sum_to_one, get_final_attention
from exp1.metrics import METRICS, classify, compute_metrics


def run_metrics(
    model: torch.nn.Module, tokens: np.ndarray, batch_size: int, chunk_size: int, query_position: int,
    diffuse_threshold: float, row_sum_atol: float, keep_heads: list[tuple[int, int]] | None = None,
    key_positions: list[int] | None = None,
) -> dict[str, object]:
    """tokens: int64 [n, key_len] model input. Returns
    - "metrics": {name: float32 [n, n_layers, n_heads]} for every name in exp1.metrics.METRICS
    - "labels": int8 [n, n_layers, n_heads]
    - "rows": float32 [n, len(keep_heads), key_len] full final rows for keep_heads (if given)
    - "at": float32 [n, len(key_positions), n_layers, n_heads] weight on each key position (if given)
    - "row_sum_dev": max |row sum - 1| seen (every chunk is checked; a failure raises).
    """
    key_len = tokens.shape[1]
    query_index = query_position % key_len
    metrics: dict[str, list[np.ndarray]] = {name: [] for name in METRICS}
    labels, rows, at = [], [], []
    row_sum_dev = 0.0
    for start in range(0, len(tokens), chunk_size):
        attn = get_final_attention(model, tokens[start : start + chunk_size], batch_size, query_position)
        row_sum_dev = max(row_sum_dev, check_rows_sum_to_one(attn, row_sum_atol))
        for name, values in compute_metrics(attn, query_index).items():
            metrics[name].append(values.astype(np.float32))
        labels.append(classify(attn, diffuse_threshold, query_index))
        if keep_heads:
            rows.append(np.stack([attn[:, layer, head, :] for layer, head in keep_heads], axis=1))
        if key_positions:
            at.append(np.stack([attn[..., p] for p in key_positions], axis=1))
    return {
        "metrics": {name: np.concatenate(v) for name, v in metrics.items()},
        "labels": np.concatenate(labels),
        "rows": np.concatenate(rows) if keep_heads else None,
        "at": np.concatenate(at) if key_positions else None,
        "row_sum_dev": row_sum_dev,
    }
