"""Per-token surface features of GPT-2 ids, joint strata over them, and balance diagnostics.

Experiment 1 compared id ranges, but the rare range (1000-39999) differs from the common range
(256-999) in surface form too: more word-start (Ġ) tokens and longer tokens. These features let
Stage 1 draw a rare pool matched to the common pool's surface form and let Stage 2 regress
per-token effects on form. The tokenizer is used only to decode ids for features and display;
model input is always the raw id array.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from exp1.show_sequences import _display_piece

FEATURES = ("word_start", "char_len", "len_bin", "is_alpha", "is_cap", "is_digit", "is_byte", "log_id", "embed_norm")

# stratum_codes packs each column into its own base-_RADIX digit, so a code reads as the values it
# encodes (e.g. word_start=1, len_bin=5, is_alpha=1 -> 1_005_001) and never depends on row order.
_RADIX = 1000
_MAX_STRATUM_COLUMNS = 6  # _RADIX ** 6 = 1e18 still fits in int64


def _len_bin(char_len: np.ndarray, len_bins: list[int]) -> np.ndarray:
    """Label of the bin that max(char_len, 1) falls in. Each len_bins entry is its bin's lower
    edge and label; the last bin takes that value or more, values below the first edge go to
    the first bin."""
    edges = np.asarray(len_bins, dtype=np.int64)
    if edges.ndim != 1 or edges.size == 0 or np.any(np.diff(edges) <= 0):
        raise ValueError(f"len_bins must be a non-empty strictly increasing list, got {len_bins}")
    idx = np.searchsorted(edges, np.maximum(char_len, 1), side="right") - 1
    return edges[np.clip(idx, 0, edges.size - 1)]


def token_features(tokenizer, ids: np.ndarray, len_bins: list[int], embed: np.ndarray | None = None) -> pd.DataFrame:
    """ids: [n] token ids -> one row per id, in input order, columns `id` + FEATURES.

    text = the id decoded on its own; word_start = text starts with a space (GPT-2's Ġ); core =
    text without that space; char_len = len(core) (so ' ' has 0); len_bin = bin of max(char_len, 1);
    is_alpha = core is non-empty ASCII letters; is_cap = core starts uppercase; is_digit = core is
    all digits; is_byte = id < 256; log_id = log10(id + 1); embed_norm = ||embed[id]|| (embed = W_E,
    [vocab, d_model]) or NaN without embed. Byte tokens that are not valid UTF-8 alone decode to
    U+FFFD, which is none of alpha/cap/digit.
    """
    ids = np.asarray(ids, dtype=np.int64)
    if ids.ndim != 1:
        raise ValueError(f"ids must be 1D, got shape {ids.shape}")
    # clean_up_tokenization_spaces=False is the gpt2 default already; pinned so a tokenizer config
    # that enables it cannot strip the space from tokens like ' .' and flip their word_start.
    texts = [tokenizer.decode([int(i)], clean_up_tokenization_spaces=False) for i in ids]
    word_start = np.array([t.startswith(" ") for t in texts], dtype=bool)
    cores = [t[1:] if ws else t for t, ws in zip(texts, word_start)]
    char_len = np.array([len(c) for c in cores], dtype=np.int64)

    if embed is None:
        embed_norm = np.full(ids.shape, np.nan)
    else:
        embed = np.asarray(embed)
        if embed.ndim != 2 or (ids.size and ids.max() >= embed.shape[0]):
            raise ValueError(f"embed must be [vocab, d_model] covering every id, got shape {embed.shape}")
        embed_norm = np.linalg.norm(embed[ids].astype(np.float64), axis=1)

    return pd.DataFrame({
        "id": ids,
        "word_start": word_start,
        "char_len": char_len,
        "len_bin": _len_bin(char_len, len_bins),
        "is_alpha": np.array([c.isascii() and c.isalpha() for c in cores], dtype=bool),
        "is_cap": np.array([c[:1].isupper() for c in cores], dtype=bool),
        "is_digit": np.array([c.isdigit() for c in cores], dtype=bool),
        "is_byte": ids < 256,
        "log_id": np.log10(ids.astype(np.float64) + 1.0),
        "embed_norm": embed_norm,
    })


def stratum_codes(features: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """int64 [n] code for each row's joint combination of `columns` (bool or non-negative int).

    The code is a fixed mixed-radix packing of the values themselves, not an order-of-appearance
    factorization, so two frames (e.g. the rare and common pools) give the same code to the same
    combination.
    """
    columns = list(columns)
    if not 1 <= len(columns) <= _MAX_STRATUM_COLUMNS:
        raise ValueError(f"need 1..{_MAX_STRATUM_COLUMNS} stratum columns, got {columns}")
    codes = np.zeros(len(features), dtype=np.int64)
    for name in columns:
        col = features[name].to_numpy()
        if not (np.issubdtype(col.dtype, np.integer) or col.dtype == bool):
            raise ValueError(f"stratum column {name!r} must be bool or integer, got dtype {col.dtype}")
        col = col.astype(np.int64)
        if col.size and (col.min() < 0 or col.max() >= _RADIX):
            raise ValueError(f"stratum column {name!r} has values outside [0, {_RADIX}): "
                             f"[{col.min()}, {col.max()}]")
        codes = codes * _RADIX + col
    return codes


def display_text(tokenizer, token_id: int) -> str:
    """Printable text for one id (the BPE symbol, e.g. 'Ċ', when it decodes to nothing printable);
    '$' is escaped for matplotlib, as in the Exp 1 sequence figures."""
    return _display_piece(tokenizer, int(token_id))[0]


def balance_table(features_a: pd.DataFrame, features_b: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Standardized mean difference (a - b) / pooled SD per feature, pooled SD = sqrt((var_a + var_b) / 2)
    with sample variances (ddof=1); booleans count as 0/1 and NaNs are skipped.

    Columns: feature, mean_a, mean_b, sd_a, sd_b, smd. When both SDs are 0 the SMD is 0 if the means
    agree and +-inf if they differ: a feature constant in each group but different between them is
    the worst possible imbalance and must not pass a |SMD| <= threshold check.
    """
    rows = []
    for name in columns:
        a = features_a[name].to_numpy().astype(np.float64)
        b = features_b[name].to_numpy().astype(np.float64)
        a, b = a[~np.isnan(a)], b[~np.isnan(b)]
        if a.size < 2 or b.size < 2:  # e.g. embed_norm computed without an embedding
            rows.append({"feature": name, "mean_a": np.nan, "mean_b": np.nan, "sd_a": np.nan, "sd_b": np.nan,
                         "smd": np.nan})
            continue
        mean_a, mean_b = a.mean(), b.mean()
        sd_a, sd_b = a.std(ddof=1), b.std(ddof=1)
        pooled = np.sqrt((sd_a ** 2 + sd_b ** 2) / 2.0)
        diff = mean_a - mean_b
        if pooled > 0:
            smd = diff / pooled
        elif diff == 0:
            smd = 0.0
        else:
            smd = np.copysign(np.inf, diff)
        rows.append({"feature": name, "mean_a": mean_a, "mean_b": mean_b, "sd_a": sd_a, "sd_b": sd_b, "smd": smd})
    return pd.DataFrame(rows, columns=["feature", "mean_a", "mean_b", "sd_a", "sd_b", "smd"])
