"""Condition comparisons and resampling statistics for per-(layer, head) metrics.

Two conditions are compared per head with a vectorized two-sided Mann-Whitney U
test; bh_fdr controls the false discovery rate across the resulting p-values;
label_shift measures how far each head's label mix moves between conditions (total
variation distance); split_half gives a noise-floor null by comparing random halves
of one condition;
bootstrap_label_percentages gives a CI on the % of heads whose mode label is
each code (see exp1.aggregate for the mode/tie convention this mirrors).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.stats import mannwhitneyu


def compare_conditions(x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    """x: [n_x, ...], y: [n_y, ...] per-sequence values, compared along axis 0
    independently for every trailing index (e.g. per layer/head).
    """
    if x.shape[1:] != y.shape[1:]:
        raise ValueError(f"trailing shapes differ: x{x.shape[1:]} vs y{y.shape[1:]}")

    n_x, n_y = x.shape[0], y.shape[0]
    result = mannwhitneyu(
        x, y, alternative="two-sided", method="asymptotic", use_continuity=True, axis=0
    )
    u = np.asarray(result.statistic, dtype=np.float64)
    p = np.asarray(result.pvalue, dtype=np.float64)
    effect_size = 2.0 * u / (n_x * n_y) - 1.0
    mean_diff = x.mean(axis=0) - y.mean(axis=0)

    return {
        "U": u,
        "p": p,
        "effect_size": effect_size,
        "mean_diff": np.asarray(mean_diff, dtype=np.float64),
    }


def bh_fdr(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values), over all entries of p."""
    p = np.asarray(p, dtype=np.float64)
    shape = p.shape
    flat = p.ravel()
    m = flat.size

    order = np.argsort(flat)
    ranks = np.arange(1, m + 1, dtype=np.float64)
    q_sorted = flat[order] * m / ranks
    # cumulative min from the largest rank down, so q is monotone non-decreasing in p
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)

    q = np.empty(m, dtype=np.float64)
    q[order] = q_sorted
    return q.reshape(shape)


def label_shift(x: np.ndarray, y: np.ndarray, n_labels: int) -> dict[str, np.ndarray]:
    """x: [n_x, ...], y: [n_y, ...] int label codes, compared along axis 0 for every trailing index.

    frac_diff[k] = share of x labelled k minus share of y labelled k (same x - y sign as
    compare_conditions' mean_diff); tv = total variation distance = half the L1 norm of
    frac_diff, 0 for an identical label mix and 1 when the two share no label.
    """
    if x.shape[1:] != y.shape[1:]:
        raise ValueError(f"trailing shapes differ: x{x.shape[1:]} vs y{y.shape[1:]}")

    frac_diff = np.stack([(x == k).mean(axis=0) - (y == k).mean(axis=0) for k in range(n_labels)])
    return {"tv": 0.5 * np.abs(frac_diff).sum(axis=0), "frac_diff": frac_diff}


def split_half(
    x: np.ndarray,
    seed: int,
    compare: Callable[[np.ndarray, np.ndarray], dict[str, np.ndarray]] = compare_conditions,
) -> dict[str, np.ndarray]:
    """Noise-floor null: compare two random, non-overlapping halves of one sample with `compare`."""
    n = x.shape[0]
    half = n // 2
    perm = np.random.default_rng(seed).permutation(n)
    return compare(x[perm[:half]], x[perm[half : 2 * half]])


def bootstrap_mean_ci(x: np.ndarray, iters: int, seed: int, ci: float) -> tuple[float, float, float]:
    """(mean, lo, hi) of x's grand mean, with a percentile CI from resampling along axis 0 (sequences)."""
    per_seq = x.reshape(x.shape[0], -1).mean(axis=1)
    rng = np.random.default_rng(seed)
    boot = per_seq[rng.integers(0, len(per_seq), size=(iters, len(per_seq)))].mean(axis=1)
    tail = (1.0 - ci) / 2.0 * 100.0
    return float(per_seq.mean()), float(np.percentile(boot, tail)), float(np.percentile(boot, 100.0 - tail))


def bootstrap_label_percentages(
    labels: np.ndarray, n_labels: int, iters: int, seed: int, ci: float
) -> dict[str, np.ndarray]:
    """labels: int codes [n_seq, n_layers, n_heads].

    Statistic: % of heads (over all layers x heads) whose mode label (argmax of
    per-head label counts, ties -> lowest code) is each label code.
    """
    n_seq, n_layers, n_heads = labels.shape
    n_units = n_layers * n_heads
    flat = labels.reshape(n_seq, n_units)

    # one-hot [n_seq, n_units * n_labels], built once so each resample is a matmul.
    onehot = np.zeros((n_seq, n_units, n_labels), dtype=np.float64)
    onehot[np.arange(n_seq)[:, None], np.arange(n_units)[None, :], flat] = 1.0
    onehot = onehot.reshape(n_seq, n_units * n_labels)

    def label_pct(weights: np.ndarray) -> np.ndarray:
        counts = (weights @ onehot).reshape(n_units, n_labels)
        mode = counts.argmax(axis=1)  # ties -> lowest code
        return np.bincount(mode, minlength=n_labels).astype(np.float64) / n_units * 100.0

    pct = label_pct(np.ones(n_seq, dtype=np.float64))

    rng = np.random.default_rng(seed)
    boot = np.empty((iters, n_labels), dtype=np.float64)
    for i in range(iters):
        idx = rng.integers(0, n_seq, size=n_seq)
        weights = np.bincount(idx, minlength=n_seq).astype(np.float64)
        boot[i] = label_pct(weights)

    tail = (1.0 - ci) / 2.0 * 100.0
    lo = np.percentile(boot, tail, axis=0)
    hi = np.percentile(boot, 100.0 - tail, axis=0)

    return {"pct": pct, "lo": lo, "hi": hi, "boot": boot}
