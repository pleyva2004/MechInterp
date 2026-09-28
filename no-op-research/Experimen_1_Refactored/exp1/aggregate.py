"""Per-(layer, head) aggregation of attention metrics across sequences.

Reduces the [n_seq, n_layers, n_heads, key_len] attention tensor to one row per
(layer, head): the categorical label distribution from classify, plus mean/std
of the per-sequence metrics from compute_metrics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from exp1.metrics import LABELS, METRICS, classify, compute_metrics


def aggregate_heads(
    attn: np.ndarray,
    diffuse_threshold: float,
    query_index: int | None = None,
    id_cols: dict[str, object] | None = None,
) -> pd.DataFrame:
    """attn: [n_seq, n_layers, n_heads, key_len]. One output row per (layer, head),
    sorted by layer then head.

    mode_label ties (multiple labels sharing the max sequence count) resolve to
    whichever label appears earliest in LABELS, via np.argmax over per-label counts.
    """
    n_seq, n_layers, n_heads, _key_len = attn.shape

    metrics = compute_metrics(attn, query_index)  # each value: [n_seq, n_layers, n_heads]
    labels = classify(attn, diffuse_threshold, query_index)  # [n_seq, n_layers, n_heads] int8

    # counts[i] = number of sequences labelled LABELS[i], per (layer, head).
    counts = np.stack([(labels == code).sum(axis=0) for code in range(len(LABELS))])
    mode_idx = counts.argmax(axis=0)  # [n_layers, n_heads]; ties -> earliest label in LABELS
    mode_label = np.asarray(LABELS)[mode_idx]
    consistency = counts.max(axis=0) / n_seq

    layer_idx, head_idx = np.meshgrid(np.arange(n_layers), np.arange(n_heads), indexing="ij")

    data: dict[str, np.ndarray] = {}
    if id_cols:
        n_rows = n_layers * n_heads
        for key, value in id_cols.items():
            data[key] = np.full(n_rows, value)
    data["layer"] = layer_idx.ravel()
    data["head"] = head_idx.ravel()
    data["mode_label"] = mode_label.ravel()
    data["consistency"] = consistency.ravel()

    for code, label in enumerate(LABELS):
        data[f"frac_{label.lower()}"] = (counts[code] / n_seq).ravel()

    for name in METRICS:
        values = metrics[name]  # [n_seq, n_layers, n_heads]
        data[f"mean_{name}"] = values.mean(axis=0).ravel()
        data[f"std_{name}"] = values.std(axis=0, ddof=1).ravel()

    return pd.DataFrame(data)
