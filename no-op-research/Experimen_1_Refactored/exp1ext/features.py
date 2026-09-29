"""Character-level features of GPT-2 token ids, and a simple OLS regression of a per-token effect
(e.g. exp1ext.effects.per_token_effects output) onto those features.

Decoding a byte-fallback token that is not valid UTF-8 on its own (id < 256, no matching character)
yields the replacement character U+FFFD; it is not alphanumeric, so such a token is scored is_punct
like any other non-alphanumeric core -- it is not tracked separately from ordinary punctuation.
"""
from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import t

FEATURES: tuple[str, ...] = (
    "is_byte", "leading_space", "is_alpha", "is_digit", "is_punct", "is_capitalized", "n_chars", "log_id",
)


def _capitalized(core: str) -> bool:
    """True iff core's first alphabetic character is uppercase; False if core has none."""
    first_alpha = next((ch for ch in core if ch.isalpha()), "")
    return first_alpha.isupper()


def token_features(ids: np.ndarray, tokenizer) -> pd.DataFrame:
    """ids: [n] token ids. One row per id, in input order. `text`/`bpe` are for display/lookup;
    the boolean/numeric features are derived from `core` = the decoded text with whitespace
    stripped (so e.g. "\\n" and " " have an empty core and score False on every text-shape flag).
    """
    ids = np.asarray(ids, dtype=np.int64)
    texts = [tokenizer.decode([int(i)]) for i in ids]
    bpe = tokenizer.convert_ids_to_tokens([int(i) for i in ids])
    cores = [text.strip() for text in texts]

    return pd.DataFrame({
        "id": ids,
        "text": texts,
        "bpe": bpe,
        "is_byte": ids < 256,
        "leading_space": np.array([b.startswith("Ġ") for b in bpe]),
        "is_alpha": np.array([bool(c) and c.isalpha() for c in cores]),
        "is_digit": np.array([bool(c) and c.isdigit() for c in cores]),
        "is_punct": np.array([bool(c) and not any(ch.isalnum() for ch in c) for c in cores]),
        "is_capitalized": np.array([_capitalized(c) for c in cores]),
        "n_chars": np.array([len(c) for c in cores], dtype=int),
        "log_id": np.log1p(ids.astype(np.float64)),
    })


def fit_feature_model(effect: np.ndarray, features: pd.DataFrame, columns: Sequence[str], ci: float) -> pd.DataFrame:
    """OLS of effect [T] on the given feature columns (+ intercept), dropping rows where effect is
    non-finite; booleans are cast to 0/1. A column left constant after row filtering is dropped (it
    would make the design matrix rank-deficient) and reported via a warning.

    Returns one row per surviving term ("intercept" first, then kept columns in `columns` order):
    term, coef, se, lo, hi, plus n (rows used) and r2, repeated on every row. CIs use classic OLS
    standard errors and a Student-t interval with n - p degrees of freedom at level `ci`.
    """
    effect = np.asarray(effect, dtype=np.float64)
    if len(features) != len(effect):
        raise ValueError(f"features has {len(features)} rows but effect has {len(effect)}")

    finite = np.isfinite(effect)
    y = effect[finite]
    n = y.shape[0]

    kept: list[str] = []
    cols: list[np.ndarray] = []
    for name in columns:
        col = features[name].to_numpy().astype(np.float64)[finite]
        if np.all(col == col[0]):
            warnings.warn(f"fit_feature_model: dropping constant column {name!r} after row filtering", stacklevel=2)
            continue
        kept.append(name)
        cols.append(col)

    X = np.column_stack([np.ones(n), *cols]) if cols else np.ones((n, 1))
    terms = ["intercept"] + kept
    p = X.shape[1]
    dof = n - p
    if dof <= 0:
        raise ValueError(f"not enough rows ({n}) for {p} parameters (dof={dof})")

    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    sigma2 = np.sum(resid ** 2) / dof
    xtx_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(xtx_inv) * sigma2)

    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - np.sum(resid ** 2) / ss_tot if ss_tot > 0 else np.nan

    t_crit = t.ppf(1.0 - (1.0 - ci) / 2.0, dof)

    return pd.DataFrame({
        "term": terms,
        "coef": coef,
        "se": se,
        "lo": coef - t_crit * se,
        "hi": coef + t_crit * se,
        "n": n,
        "r2": r2,
    })
