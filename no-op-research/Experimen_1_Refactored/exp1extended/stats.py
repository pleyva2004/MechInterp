"""Statistics for Stage 1 (paired position factorial) and Stage 2 (per-token sweep).

Every function is vectorized over trailing axes, so one call covers all heads ([..., 12, 12]) or the target
heads ([..., T]). Axis 0 is the replication unit (the pair i) unless stated otherwise.

Stage 1: paired_test (Wilcoxon signed-rank + paired percentile bootstrap) on any two cells; yates_effects splits
the 2^3 (Q, P, X) x (rare, common) factorial into per-pair main effects and interactions, and diagonal_shares
reports each term's share of the all-common minus all-rare gap.

Stage 2: values are [T tokens, K backgrounds, ...] with a bool missing mask [T, K] (a token that already occurs in
a background is not placed there). twoway_effects fits y_tk = mu + a_t + b_k by exact least squares over the
observed cells; split_half_reliability, within_background_permutation_p and bootstrap_twoway_ci are built on it.
icc_oneway is the one-way ICC(1) used by the Stage 0 "which position drives the head" table.

bh_fdr, label_shift, bootstrap_mean_ci and compare_conditions are re-exported from exp1.compare, not duplicated.
"""
from __future__ import annotations

from itertools import combinations, product
from typing import Callable

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import wilcoxon

from exp1.compare import bh_fdr, bootstrap_mean_ci, compare_conditions, label_shift

__all__ = [
    "FACTORS", "LEVEL_SIGN", "YATES_EFFECTS", "DIAGONAL_TERMS",
    "paired_test", "yates_effects", "yates_effects_from_cells", "diagonal_shares",
    "twoway_effects", "split_half_reliability", "within_background_permutation_p", "icc_oneway",
    "bootstrap_twoway_ci",
    "bh_fdr", "bootstrap_mean_ci", "compare_conditions", "label_shift",
]

FACTORS: tuple[str, ...] = ("Q", "P", "X")
LEVEL_SIGN: dict[str, float] = {"r": -1.0, "c": 1.0}  # sign convention: effects are common minus rare
YATES_EFFECTS: tuple[str, ...] = ("Q", "P", "X", "QP", "QX", "PX", "QPX")
DIAGONAL_TERMS: tuple[str, ...] = ("Q", "P", "X", "QPX")  # the terms that sum to y(ccc) - y(rrr)

_BOOT_CHUNK = 256  # bootstrap iterations per matmul in paired_test; bounds memory at chunk x n
_WILCOXON_ASYMPTOTIC_N = 50  # scipy's method="auto" is asymptotic for every column above this n


def _trailing_2d(x: np.ndarray) -> np.ndarray:
    """[n, ...] -> float64 [n, prod(...)]."""
    x = np.asarray(x, dtype=np.float64)
    return x.reshape(x.shape[0], -1)


# ---- Stage 1: paired comparisons ---------------------------------------------------------------------------

def paired_test(x: np.ndarray, y: np.ndarray, iters: int, seed: int, ci: float) -> dict[str, np.ndarray]:
    """Paired comparison of x and y ([n, ...] each, pairs along axis 0), independently per trailing index.

    Returns float64 arrays of the trailing shape:
      mean_diff  mean of d = x - y over pairs;
      lo, hi     percentile bootstrap CI (level `ci`) of mean_diff; each iteration resamples the n pair indices
                 once and applies them to every trailing column, so heads share resamples;
      p          two-sided Wilcoxon signed-rank p, scipy.stats.wilcoxon(d, axis=0, zero_method="wilcox");
                 p = 1 where every difference is exactly 0 (scipy would return NaN with a warning);
    and n (int), the number of pairs.
    """
    x, y = np.asarray(x), np.asarray(y)
    if x.shape != y.shape:
        raise ValueError(f"paired arrays differ in shape: x{x.shape} vs y{y.shape}")
    if x.ndim < 1 or x.shape[0] < 1:
        raise ValueError(f"need at least one pair along axis 0; got shape {x.shape}")
    trailing = x.shape[1:]
    d = _trailing_2d(x) - _trailing_2d(y)  # [n, P]
    n, n_cols = d.shape

    mean_diff = d.mean(axis=0)

    rng = np.random.default_rng(seed)
    boot = np.empty((iters, n_cols), dtype=np.float64)
    for start in range(0, iters, _BOOT_CHUNK):
        chunk = min(_BOOT_CHUNK, iters - start)
        idx = rng.integers(0, n, size=(chunk, n))
        # resample counts per pair, so a chunk of bootstrap means is one [chunk, n] @ [n, P] matmul
        flat = (idx + n * np.arange(chunk)[:, None]).ravel()
        counts = np.bincount(flat, minlength=chunk * n).reshape(chunk, n).astype(np.float64)
        boot[start : start + chunk] = counts @ d / n
    tail = (1.0 - ci) / 2.0 * 100.0
    lo = np.percentile(boot, tail, axis=0)
    hi = np.percentile(boot, 100.0 - tail, axis=0)

    p = np.ones(n_cols, dtype=np.float64)
    testable = np.flatnonzero(np.any(d != 0, axis=0))
    if testable.size:
        if n > _WILCOXON_ASYMPTOTIC_N:
            p[testable] = wilcoxon(d[:, testable], axis=0, zero_method="wilcox").pvalue
        else:
            # below the cutoff scipy picks exact vs asymptotic from zeros/ties anywhere in the array, so a
            # vectorized call would make one head's p depend on the others; test each column on its own
            for j in testable:
                p[j] = wilcoxon(d[:, j], zero_method="wilcox").pvalue

    return {
        "mean_diff": mean_diff.reshape(trailing),
        "lo": lo.reshape(trailing),
        "hi": hi.reshape(trailing),
        "p": p.reshape(trailing),
        "n": n,
    }


def _cell_code(key: str, factor_order: tuple[str, ...]) -> str | None:
    """A cell key as an r/c code in factor_order ("rcc"), or None if it is not an r/c cell.

    Accepts the code itself or design.cell_name's form ("Qr_Pc_Xc", any factor order).
    """
    k = len(factor_order)
    if len(key) == k and set(key) <= set(LEVEL_SIGN):
        return key
    parts = key.split("_")
    if len(parts) != k:
        return None
    levels: dict[str, str] = {}
    for part in parts:
        name, level = part[:-1], part[-1:]
        if name not in factor_order or name in levels or level not in LEVEL_SIGN:
            return None
        levels[name] = level
    return "".join(levels[f] for f in factor_order)


def _effect_terms(factor_order: tuple[str, ...]) -> list[tuple[str, tuple[int, ...]]]:
    """(name, factor indices) for every main effect and interaction, lower orders first."""
    k = len(factor_order)
    return [
        ("".join(factor_order[i] for i in combo), combo)
        for order in range(1, k + 1)
        for combo in combinations(range(k), order)
    ]


def yates_effects_from_cells(
    cells: dict[str, np.ndarray], factor_order: tuple[str, ...] = FACTORS
) -> dict[str, np.ndarray]:
    """Per-pair Yates effects of a 2^k factorial (k = len(factor_order); k = 3 for (Q, P, X)).

    cells: {code: [n, ...]} paired over axis 0. A code has one letter per factor, in factor_order, each "r"
    (rare, sign -1) or "c" (common, sign +1): "rcc" = Q r, P c, X c. design.cell_name keys ("Qr_Pc_Xc") are
    also accepted. Keys that are not pure r/c cells (e.g. "mmm", "Qm_Pc_Xc") are ignored; all 2^k r/c cells
    are required.

    With s_F(cell) the sign of factor F in that cell, each effect is contrast / 2^(k-1), where the contrast
    is the sum over cells of (product of that effect's factor signs) * y. For (Q, P, X):
      Q   = (1/4) sum sQ * y          = mean(Q=c cells) - mean(Q=r cells)
      QP  = (1/4) sum sQ * sP * y     = half the difference of the Q effect at P=c and at P=r
      QPX = (1/4) sum sQ * sP * sX * y
    and, exactly, per pair:
      y(ccc) - y(rrr) = Q + P + X + QPX
    because in the saturated model y = mean + (1/2) sum_E s_E * E, the all-common and all-rare cells have
    equal signs on every even-order term (QP, QX, PX cancel since (+1)(+1) = (-1)(-1)) and opposite signs
    on every odd-order term. (In general the diagonal is the sum of the odd-order effects.)

    Returns {name: float64 [n, ...]} for "Q", "P", "X", "QP", "QX", "PX", "QPX" (names join factor names in
    factor_order).
    """
    factor_order = tuple(factor_order)
    k = len(factor_order)
    if k < 1 or len(set(factor_order)) != k:
        raise ValueError(f"factor_order must be distinct factor names; got {factor_order}")

    by_code: dict[str, np.ndarray] = {}
    for key, value in cells.items():
        code = _cell_code(key, factor_order)
        if code is None:
            continue
        if code in by_code:
            raise ValueError(f"cell {code} given twice (key {key!r})")
        by_code[code] = np.asarray(value, dtype=np.float64)

    codes = ["".join(levels) for levels in product(("r", "c"), repeat=k)]
    absent = [code for code in codes if code not in by_code]
    if absent:
        raise ValueError(f"missing cells {absent} (factor order {factor_order}); got keys {sorted(cells)}")
    shapes = {by_code[code].shape for code in codes}
    if len(shapes) != 1:
        raise ValueError(f"cells differ in shape: {sorted(shapes)}")

    stacked = np.stack([by_code[code] for code in codes])  # [2^k, n, ...]
    signs = np.array([[LEVEL_SIGN[level] for level in code] for code in codes])  # [2^k, k]
    divisor = 2.0 ** (k - 1)
    return {
        name: np.tensordot(signs[:, list(combo)].prod(axis=1), stacked, axes=1) / divisor
        for name, combo in _effect_terms(factor_order)
    }


def yates_effects(cells: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Yates effects over (Q, P, X); see yates_effects_from_cells for keys, signs and the exact
    decomposition y(ccc) - y(rrr) = Q + P + X + QPX."""
    return yates_effects_from_cells(cells, FACTORS)


def diagonal_shares(
    effects: dict[str, np.ndarray], factor_order: tuple[str, ...] = FACTORS
) -> dict[str, np.ndarray]:
    """Each diagonal term's share of the mean gap y(all c) - y(all r).

    effects: per-pair arrays [n, ...] from yates_effects_from_cells. The diagonal terms are the odd-order
    effects (Q, P, X, QPX for three factors); share_E = mean_i(E_i) / mean_i(gap_i) with
    gap = sum of the diagonal terms, so the shares sum to 1. NaN where the mean gap is exactly 0; a share
    is only meaningful where |gap| is large and its CI excludes 0, which the caller checks.

    Returns {term: float64 [...]}.
    """
    terms = [name for name, combo in _effect_terms(tuple(factor_order)) if len(combo) % 2 == 1]
    means = {term: np.asarray(effects[term], dtype=np.float64).mean(axis=0) for term in terms}
    gap = sum(means.values())
    with np.errstate(divide="ignore", invalid="ignore"):
        return {term: np.where(gap != 0, mean / gap, np.nan) for term, mean in means.items()}


# ---- Stage 2: token x background two-way layout ------------------------------------------------------------

def _prep_twoway(values: np.ndarray, missing: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    """-> (y float64 [T, K, P] with 0 in missing cells, missing bool [T, K], trailing shape)."""
    values = np.asarray(values)
    missing = np.asarray(missing)
    if missing.dtype != np.bool_:
        raise ValueError(f"missing must be bool, got {missing.dtype}")
    if values.ndim < 2 or values.shape[:2] != missing.shape:
        raise ValueError(f"values {values.shape} must start with missing's shape {missing.shape}")
    trailing = values.shape[2:]
    y = values.reshape(values.shape[0], values.shape[1], -1).astype(np.float64)
    if not np.all(np.isfinite(y[~missing])):
        raise ValueError("values has non-finite entries in observed (non-missing) cells")
    # missing cells may hold anything (NaN, the unswept value); zero them so sums only see observed cells
    y[missing] = 0.0
    return y, missing, trailing


class _TwoWayFit:
    """Exact least squares for y_tk = m_t + b_k over the observed cells of a fixed missing mask.

    Tokens / backgrounds with no observed cell are not estimable and are excluded. Eliminating m_t from the
    normal equations leaves a K x K system L b = C - W^T (R / n) with L = diag(c) - W^T diag(1/n) W (the
    Laplacian of the token-background graph, null vector 1); sum(b) = 0 fixes the gauge. Adding
    alpha * 1 1^T to L makes it positive definite when the observed graph is connected and leaves the
    sum-zero solution unchanged, so one Cholesky factor serves every fit (and every permutation) of the mask.
    """

    def __init__(self, missing: np.ndarray):
        observed = ~missing
        n_tok = observed.sum(axis=1)
        n_bg = observed.sum(axis=0)
        self.tok_ok = n_tok > 0
        self.bg_ok = n_bg > 0
        if not self.tok_ok.any():
            raise ValueError("no observed cells")
        self.w = observed[self.tok_ok][:, self.bg_ok].astype(np.float64)  # [Tv, Kv]
        self.n_tok = n_tok[self.tok_ok].astype(np.float64)
        n_bg_v = n_bg[self.bg_ok].astype(np.float64)

        t_idx, k_idx = np.nonzero(self.w)
        n_nodes = self.w.shape[0] + self.w.shape[1]
        graph = coo_matrix((np.ones(t_idx.size), (t_idx, self.w.shape[0] + k_idx)), shape=(n_nodes, n_nodes))
        n_components, _ = connected_components(graph, directed=False)
        if n_components != 1:
            raise ValueError(
                f"observed token-background cells form {n_components} disconnected groups; "
                "token and background effects are not identifiable"
            )

        laplacian = np.diag(n_bg_v) - self.w.T @ (self.w / self.n_tok[:, None])
        alpha = n_bg_v.mean() / n_bg_v.size  # puts the added eigenvalue at mean(c), comparable to L's
        self.cho = cho_factor(laplacian + alpha)

    def solve(self, row_sums: np.ndarray, col_sums: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """row_sums [T, P], col_sums [K, P] over observed cells -> (m [Tv, P], b [Kv, P], sum(b) = 0)."""
        rows = row_sums[self.tok_ok]
        rhs = col_sums[self.bg_ok] - self.w.T @ (rows / self.n_tok[:, None])
        b = cho_solve(self.cho, rhs)
        b -= b.mean(axis=0)  # remove rounding drift along the null vector; m absorbs it exactly below
        m = (rows - self.w @ b) / self.n_tok[:, None]
        return m, b


def _scatter(valid: np.ndarray, part: np.ndarray) -> np.ndarray:
    """Place rows of `part` at the True entries of `valid`; NaN elsewhere."""
    out = np.full((valid.size,) + part.shape[1:], np.nan)
    out[valid] = part
    return out


def twoway_effects(
    values: np.ndarray, missing: np.ndarray, center: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Additive token + background fit y_tk = mu + a_t + b_k, least squares over the observed cells.

    values: [T, K, ...] (anything in missing cells is ignored); missing: bool [T, K].
    Constraints: sum_t a_t = 0 and sum_k b_k = 0 (unweighted). With no missing cell this is exactly
    a_t = row mean - grand mean, b_k = column mean - grand mean, mu = grand mean; with missing cells it is
    the exact least-squares solution (one K x K solve shared by every trailing column), not a
    missing-at-random average, so unbalanced backgrounds do not leak into token effects.

    Returns (token_eff [T, ...], bg_eff [K, ...], grand [...]), float64. token_eff is a_t, or mu + a_t
    (the token's level at the average background) when center=False. A token or background with no
    observed cell gets NaN and is left out of the constraints. Raises ValueError if the observed cells do
    not connect all tokens and backgrounds (effects not identifiable).
    """
    y, missing, trailing = _prep_twoway(values, missing)
    n_tok, n_bg = missing.shape

    if not missing.any():
        grand = y.mean(axis=(0, 1))
        levels = y.mean(axis=1)
        bg_eff = y.mean(axis=0) - grand
    else:
        fit = _TwoWayFit(missing)
        m, b = fit.solve(y.sum(axis=1), y.sum(axis=0))
        grand = m.mean(axis=0)
        levels = _scatter(fit.tok_ok, m)
        bg_eff = _scatter(fit.bg_ok, b)

    token_eff = levels - grand if center else levels
    return (
        token_eff.reshape((n_tok,) + trailing),
        bg_eff.reshape((n_bg,) + trailing),
        grand.reshape(trailing),
    )


def split_half_reliability(
    values: np.ndarray, missing: np.ndarray, seed: int, n_splits: int = 20
) -> np.ndarray:
    """Reliability of the token-effect profile across backgrounds, per trailing index.

    Each split randomly halves the backgrounds (with odd K the second half gets the extra one), fits
    twoway_effects in each half, takes the Pearson r of the two token-effect vectors (over tokens
    observed in both halves) and Spearman-Brown corrects it to full length, 2r / (1 + r). Returns the mean
    over splits, float64 [...]; about 1 for a stable profile, about 0 for noise.
    """
    y, missing, trailing = _prep_twoway(values, missing)
    n_bg = missing.shape[1]
    if n_bg < 2:
        raise ValueError(f"need at least 2 backgrounds to split; got {n_bg}")
    if n_splits < 1:
        raise ValueError(f"n_splits must be >= 1; got {n_splits}")

    rng = np.random.default_rng(seed)
    corrected = np.empty((n_splits, y.shape[2]), dtype=np.float64)
    for s in range(n_splits):
        perm = rng.permutation(n_bg)
        halves = (perm[: n_bg // 2], perm[n_bg // 2 :])
        eff = [twoway_effects(y[:, half], missing[:, half])[0] for half in halves]
        both = np.isfinite(eff[0][:, 0]) & np.isfinite(eff[1][:, 0])
        a = eff[0][both] - eff[0][both].mean(axis=0)
        b = eff[1][both] - eff[1][both].mean(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = (a * b).sum(axis=0) / np.sqrt((a * a).sum(axis=0) * (b * b).sum(axis=0))
            corrected[s] = 2.0 * r / (1.0 + r)
    return corrected.mean(axis=0).reshape(trailing)


def within_background_permutation_p(
    values: np.ndarray, missing: np.ndarray, n_perm: int, seed: int
) -> np.ndarray:
    """Omnibus test of "no token effect", per trailing index.

    Statistic: variance over tokens (ddof = 0) of the twoway_effects token effects. Null: within each
    background, token labels are permuted among that background's non-missing tokens only (independently
    per background), which keeps the missing mask and each background's values, so background effects and
    collisions cannot create a token effect. p = (1 + #{null >= observed}) / (1 + n_perm), float64 [...].
    """
    y, missing, trailing = _prep_twoway(values, missing)
    n_tok, n_bg, n_cols = y.shape
    fit = _TwoWayFit(missing)
    col_sums = y.sum(axis=0)  # a within-column permutation leaves these unchanged

    def token_variance(row_sums: np.ndarray) -> np.ndarray:
        m, _ = fit.solve(row_sums, col_sums)
        return m.var(axis=0)

    observed = token_variance(y.sum(axis=1))
    # ties with the observed statistic (e.g. a constant column) must count as >=, despite rounding
    tol = 1e-10 * (observed + (y * y).sum(axis=0).sum(axis=0) / (~missing).sum())

    obs_rows = [np.flatnonzero(~missing[:, k]) for k in range(n_bg)]
    obs_vals = [y[rows, k] for k, rows in enumerate(obs_rows)]  # [n_k, P] per background
    rng = np.random.default_rng(seed)
    exceed = np.zeros(n_cols, dtype=np.int64)
    row_sums = np.empty((n_tok, n_cols), dtype=np.float64)
    for _ in range(n_perm):
        row_sums.fill(0.0)
        for rows, vals in zip(obs_rows, obs_vals):
            row_sums[rows] += vals[rng.permutation(rows.size)]
        exceed += token_variance(row_sums) >= observed - tol
    return ((1.0 + exceed) / (1.0 + n_perm)).reshape(trailing)


def icc_oneway(y: np.ndarray, groups: np.ndarray) -> float | np.ndarray:
    """One-way random-effects ICC(1) of y over groups (e.g. sequences sharing the token at one position).

    y: [n] or [n, ...] (vectorized over trailing axes); groups: [n] labels. Groups with fewer than 2
    members are dropped. With g groups, N observations, group sizes n_j:
      ICC = (MSB - MSW) / (MSB + (n0 - 1) MSW),  n0 = (N - sum n_j^2 / N) / (g - 1)
    (n0 = the common group size when balanced). Returns a float for 1-D y, else float64 [...].
    NaN where both mean squares are 0.
    """
    y = np.asarray(y, dtype=np.float64)
    groups = np.asarray(groups)
    if groups.ndim != 1 or groups.shape[0] != y.shape[0]:
        raise ValueError(f"groups must be 1-D with len(y) = {y.shape[0]} entries; got {groups.shape}")
    trailing = y.shape[1:]

    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    keep = counts[inverse] >= 2
    _, inverse, sizes = np.unique(groups[keep], return_inverse=True, return_counts=True)
    vals = y[keep].reshape(int(keep.sum()), -1)
    n_groups, n_obs = sizes.size, vals.shape[0]
    if n_groups < 2:
        raise ValueError(f"need at least 2 groups with >= 2 members; got {n_groups}")

    sums = np.zeros((n_groups, vals.shape[1]), dtype=np.float64)
    np.add.at(sums, inverse, vals)
    means = sums / sizes[:, None]
    grand = vals.mean(axis=0)
    msb = (sizes[:, None] * (means - grand) ** 2).sum(axis=0) / (n_groups - 1)
    msw = ((vals - means[inverse]) ** 2).sum(axis=0) / (n_obs - n_groups)
    n0 = (n_obs - (sizes.astype(np.float64) ** 2).sum() / n_obs) / (n_groups - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        icc = (msb - msw) / (msb + (n0 - 1.0) * msw)
    return float(icc[0]) if not trailing else icc.reshape(trailing)


def bootstrap_twoway_ci(
    values: np.ndarray,
    missing: np.ndarray,
    stat_fn: Callable[..., np.ndarray],
    iters: int,
    seed: int,
    ci: float,
    with_index: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Percentile CI of stat_fn under a two-way bootstrap: tokens and backgrounds resampled independently
    with replacement (T tokens, then K backgrounds, per iteration), so the CI reflects both sources of
    variation.

    values: [T, K, ...]; missing: bool [T, K]. stat_fn(values_resampled, missing_resampled) -> array.
    With with_index=True it is called as stat_fn(values, missing, tok_idx, bg_idx), where tok_idx [T] and
    bg_idx [K] index the originals (arange for the point estimate), so per-token covariates (e.g. token
    features for a regression) can be resampled alongside. Returns (point, lo, hi), float64 arrays of
    stat_fn's output shape. NaN in any resample propagates to lo / hi.
    """
    values = np.asarray(values)
    missing = np.asarray(missing)
    if missing.dtype != np.bool_ or values.ndim < 2 or values.shape[:2] != missing.shape:
        raise ValueError(f"values {values.shape} must start with bool missing's shape {missing.shape}")
    n_tok, n_bg = missing.shape

    def call(tok_idx: np.ndarray, bg_idx: np.ndarray, v: np.ndarray, m: np.ndarray) -> np.ndarray:
        out = stat_fn(v, m, tok_idx, bg_idx) if with_index else stat_fn(v, m)
        return np.asarray(out, dtype=np.float64)

    point = call(np.arange(n_tok), np.arange(n_bg), values, missing)
    rng = np.random.default_rng(seed)
    boot = np.empty((iters,) + point.shape, dtype=np.float64)
    for i in range(iters):
        tok_idx = rng.integers(0, n_tok, size=n_tok)
        bg_idx = rng.integers(0, n_bg, size=n_bg)
        cells = np.ix_(tok_idx, bg_idx)
        boot[i] = call(tok_idx, bg_idx, values[cells], missing[cells])
    tail = (1.0 - ci) / 2.0 * 100.0
    return point, np.percentile(boot, tail, axis=0), np.percentile(boot, 100.0 - tail, axis=0)
