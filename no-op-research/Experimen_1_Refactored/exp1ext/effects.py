"""Effect sizes and bootstrap CIs for the exp1ext factorial, sweep, and observational stages.

bootstrap_mean_per_unit is the shared building block: a percentile bootstrap over axis 0 (rows),
nan-aware, vectorized across trailing indices (e.g. heads). decompose_2x2 reuses that CI machinery
but resamples row indices *jointly* across the four (context class x final-token class) cells,
since a cell's rows are paired: row i of every cell is built from the same sampled components.
per_token_effects and split_reliability instead resample/split the K base contexts of a per-token
sweep (values indexed [K, T, ...]), since there the tokens are fixed and the contexts are the
random draws. attention_received is a purely observational aggregate (no resampling): per key
token id, the mean attention and mean share of non-sink attention it receives, from a single set
of sequences.
"""
from __future__ import annotations

import warnings

import numpy as np


def _percentile_ci(boot: np.ndarray, ci: float) -> tuple[np.ndarray, np.ndarray]:
    """boot: [iters, ...]. Percentile CI bounds along axis 0, nan-aware."""
    tail = (1.0 - ci) / 2.0 * 100.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN slice -> NaN bound, silently
        lo = np.nanpercentile(boot, tail, axis=0)
        hi = np.nanpercentile(boot, 100.0 - tail, axis=0)
    return lo, hi


def bootstrap_mean_per_unit(x: np.ndarray, iters: int, seed: int, ci: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """x: [n, ...]. Per trailing index: nanmean over axis 0, with a percentile CI from resampling
    rows (axis 0) with replacement -- the same row indices are reused for every trailing index in
    a given iteration, so the resample is coherent across (e.g.) heads. Returns (mean, lo, hi), each
    x.shape[1:].
    """
    n = x.shape[0]
    trailing_shape = x.shape[1:]
    mean = np.nanmean(x, axis=0)

    rng = np.random.default_rng(seed)
    boot = np.empty((iters,) + trailing_shape, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN column at some resample -> NaN, silently
        for i in range(iters):
            idx = rng.integers(0, n, size=n)
            boot[i] = np.nanmean(x[idx], axis=0)

    lo, hi = _percentile_ci(boot, ci)
    return mean, lo, hi


def decompose_2x2(cells: dict[str, np.ndarray], iters: int, seed: int, ci: float) -> dict[str, dict[str, np.ndarray]]:
    """cells: {"AA", "AC", "CA", "CC"} -> per-sequence values [n, ...], same shape, index-aligned
    (row i of every cell is built from the same sampled components).

    Effects on the mean (per trailing index):
      total       = CC - AA   (rare context + rare final token -> common context + common final token)
      query       = AC - AA   (final token only, rare -> common, context stays rare)
      context     = CA - AA   (context only, rare -> common, final token stays rare)
      interaction = total - query - context

    CIs are a percentile bootstrap resampling row indices jointly across all four cells (one index
    vector per iteration, shared by all cells), since the cells are paired: independently resampling
    each cell would break that pairing and overstate the CI width of the query/context effects.
    """
    required = ("AA", "AC", "CA", "CC")
    missing = [k for k in required if k not in cells]
    if missing:
        raise ValueError(f"cells missing {missing}")
    aa = cells["AA"]
    for name in required[1:]:
        if cells[name].shape != aa.shape:
            raise ValueError(f"cell {name} shape {cells[name].shape} != AA shape {aa.shape}")

    n = aa.shape[0]
    trailing_shape = aa.shape[1:]

    def effects(means: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        total = means["CC"] - means["AA"]
        query = means["AC"] - means["AA"]
        context = means["CA"] - means["AA"]
        interaction = total - query - context
        return {
            "cell_AA": means["AA"], "cell_AC": means["AC"], "cell_CA": means["CA"], "cell_CC": means["CC"],
            "total": total, "query": query, "context": context, "interaction": interaction,
        }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        est = effects({name: np.nanmean(arr, axis=0) for name, arr in cells.items()})

        rng = np.random.default_rng(seed)
        keys = list(est.keys())
        boot = {k: np.empty((iters,) + trailing_shape, dtype=np.float64) for k in keys}
        for i in range(iters):
            idx = rng.integers(0, n, size=n)  # shared across all four cells: preserves the row pairing
            means = {name: np.nanmean(arr[idx], axis=0) for name, arr in cells.items()}
            e = effects(means)
            for k in keys:
                boot[k][i] = e[k]

    out: dict[str, dict[str, np.ndarray]] = {}
    for k in keys:
        lo, hi = _percentile_ci(boot[k], ci)
        out[k] = {"est": est[k], "lo": lo, "hi": hi}
    return out


def per_token_effects(values: np.ndarray, iters: int, seed: int, ci: float) -> dict[str, np.ndarray]:
    """values: [K, T, ...] (K base contexts x T tokens x trailing, e.g. heads); may contain NaN for
    invalid (context, token) cells.

    Per token (and trailing index): nanmean over the K contexts, a percentile CI from resampling
    contexts (axis 0) with replacement (the same context indices are reused for every token in a
    given iteration), and n_valid = number of non-NaN contexts. A token column with no valid
    contexts yields NaN mean/lo/hi and n = 0; the "Mean of empty slice"/"All-NaN slice" RuntimeWarnings
    that would otherwise raise from that case are suppressed locally.
    """
    K = values.shape[0]
    T = values.shape[1]
    trailing_shape = values.shape[2:]
    n_valid = np.sum(~np.isnan(values), axis=0).astype(np.int64)  # [T, ...]

    rng = np.random.default_rng(seed)
    boot = np.empty((iters, T) + trailing_shape, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(values, axis=0)
        for i in range(iters):
            idx = rng.integers(0, K, size=K)
            boot[i] = np.nanmean(values[idx], axis=0)

    lo, hi = _percentile_ci(boot, ci)
    return {"mean": mean, "lo": lo, "hi": hi, "n": n_valid}


def split_reliability(values: np.ndarray, seed: int) -> np.ndarray:
    """values: [K, T, ...]. Split the K contexts into two random halves
    (np.random.default_rng(seed).permutation(K); first K // 2 vs next K // 2), nanmean each half over
    contexts -> [T, ...], then the Pearson r across TOKENS between the two halves, per trailing index,
    using only tokens finite in both halves. Returns r with shape values.shape[2:] (a scalar-shaped
    array when there are no trailing dims). NaN if fewer than 3 usable tokens or either half has zero
    variance over those tokens.
    """
    K = values.shape[0]
    T = values.shape[1]
    trailing_shape = values.shape[2:]
    perm = np.random.default_rng(seed).permutation(K)
    half = K // 2

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        half1 = np.nanmean(values[perm[:half]], axis=0).reshape(T, -1)  # [T, n_units]
        half2 = np.nanmean(values[perm[half : 2 * half]], axis=0).reshape(T, -1)

    n_units = half1.shape[1]
    r = np.full(n_units, np.nan, dtype=np.float64)
    for u in range(n_units):
        a, b = half1[:, u], half2[:, u]
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() < 3:
            continue
        a_m, b_m = a[mask], b[mask]
        if np.std(a_m) == 0.0 or np.std(b_m) == 0.0:
            continue
        r[u] = np.corrcoef(a_m, b_m)[0, 1]

    return r.reshape(trailing_shape)


def attention_received(tokens: np.ndarray, rows: np.ndarray, key_positions: np.ndarray) -> dict[str, np.ndarray]:
    """tokens: int64 [n, key_len] model input (BOS at 0); rows: float [n, n_heads_sel, key_len]
    final-query attention rows for the selected heads; key_positions: the positions to treat as
    keys (the caller passes 1..key_len - 2, excluding BOS and the query itself).

    For every distinct token id occurring at those positions: its occurrence count, the mean
    attention it receives per selected head, the mean share of the non-sink mass it receives,
    share = attn / (1 - attn_on_position_0), computed per (sequence, head) and averaged; rows where
    1 - sink < 1e-6 for that head are excluded from the share average (dividing by ~0 would blow up
    an otherwise-tiny attention value into a meaningless share); and mean_logratio, the mean of
    log(attn_k) - log(attn_on_position_0) = log(p_k / p_sink) per occurrence. Unlike the raw
    probability, this log-ratio cancels the softmax normalizer (it's s_k - s_0 in logit space), so it
    scores a token free of the coupling every p_k has with every other key through the softmax sum.
    Both probabilities are floored at np.finfo(np.float32).tiny before the log so an exact zero
    doesn't produce -inf.
    """
    key_positions = np.asarray(key_positions, dtype=np.int64)
    n_heads_sel = rows.shape[1]

    sub_tokens = tokens[:, key_positions]  # [n, n_pos]
    sub_attn = rows[:, :, key_positions].astype(np.float64)  # [n, n_heads_sel, n_pos]
    sink = rows[:, :, 0].astype(np.float64)  # [n, n_heads_sel]

    flat_ids = sub_tokens.ravel()
    ids, inverse = np.unique(flat_ids, return_inverse=True)  # inverse: [n * n_pos], flat_ids order
    n_ids = ids.shape[0]
    count = np.bincount(inverse, minlength=n_ids)

    valid_rh = (1.0 - sink) >= 1e-6  # [n, n_heads_sel]
    denom = np.where(valid_rh, 1.0 - sink, 1.0)[:, :, None]  # dummy 1.0 where invalid, discarded below
    share = sub_attn / denom  # [n, n_heads_sel, n_pos]
    share_mask = np.broadcast_to(valid_rh[:, :, None], share.shape)
    share = np.where(share_mask, share, 0.0)

    eps = np.finfo(np.float32).tiny  # floor before log so an exact-zero probability isn't -inf
    logratio = np.log(np.maximum(sub_attn, eps)) - np.log(np.maximum(sink, eps))[:, :, None]  # [n, n_heads_sel, n_pos]

    mean_attn = np.empty((n_ids, n_heads_sel), dtype=np.float64)
    mean_share = np.empty((n_ids, n_heads_sel), dtype=np.float64)
    mean_logratio = np.empty((n_ids, n_heads_sel), dtype=np.float64)
    for h in range(n_heads_sel):
        sum_attn = np.bincount(inverse, weights=sub_attn[:, h, :].ravel(), minlength=n_ids)
        mean_attn[:, h] = sum_attn / count

        sum_share = np.bincount(inverse, weights=share[:, h, :].ravel(), minlength=n_ids)
        n_share = np.bincount(inverse, weights=share_mask[:, h, :].astype(np.float64).ravel(), minlength=n_ids)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_share[:, h] = np.where(n_share > 0, sum_share / n_share, np.nan)

        sum_logratio = np.bincount(inverse, weights=logratio[:, h, :].ravel(), minlength=n_ids)
        mean_logratio[:, h] = sum_logratio / count

    return {
        "ids": ids.astype(np.int64),
        "count": count.astype(np.int64),
        "mean_attn": mean_attn,
        "mean_share": mean_share,
        "mean_logratio": mean_logratio,
    }
