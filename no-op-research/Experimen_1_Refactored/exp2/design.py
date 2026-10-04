"""Build the exp2 inputs: random base rows with a phrase (or a control) written over a few slots.

Arrays hold the 32 random tokens WITHOUT BOS (array index i = model position i + 1); the caller prepends BOS.
Every variant of a base row equals that row outside the slots, so every contrast can be paired per row.
Variants, per phrase x placement:
  none       the base row unchanged
  nl         the phrase's ids in order
  shuffled   the same ids in a fixed scrambled order
  matched    per row, one random id per phrase token with the same surface form and id class as that token
  matched_b  an independent second matched draw: a placebo whose contrast with `matched` is pure noise
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from exp1.sequences import load_or_create_sequences, stable_index

VARIANTS: tuple[str, ...] = ("none", "nl", "shuffled", "matched", "matched_b")


def derive_seed(base_seed: int, name: str) -> int:
    return base_seed + stable_index(name)


def id_pool(id_range: Sequence[int], exclude: Sequence[int]) -> np.ndarray:
    """Every id in the inclusive id_range except those in `exclude`, ascending."""
    ids = np.arange(int(id_range[0]), int(id_range[1]) + 1, dtype=np.int64)
    return ids[~np.isin(ids, np.asarray(list(exclude), dtype=np.int64))]


def cached_base_block(name: str, id_range: Sequence[int], exclude: Sequence[int], n: int, seq_len: int,
                      base_seed: int, bos_id: int, data_dir: str | Path) -> np.ndarray:
    """n rows of seq_len ids from id_range minus `exclude`, no repeats within a row.

    Drawn as indices into the pool through exp1's hash-checked cache, so a stale cache raises instead of being
    regenerated. The cache name carries a hash of the pool: a block drawn under a different exclusion list can
    never be picked up by mistake.
    """
    pool = id_pool(id_range, exclude)
    tag = hashlib.sha256(np.ascontiguousarray(pool, dtype="<i8").tobytes()).hexdigest()[:10]
    idx = load_or_create_sequences(f"{name}_{tag}", [0, len(pool) - 1], n, seq_len, base_seed, True, bos_id, data_dir)
    return pool[idx]


def token_class_of(token_id: int, token_classes: dict[str, Sequence[int]]) -> str:
    for name, (lo, hi) in token_classes.items():
        if lo <= token_id <= hi:
            return name
    raise ValueError(f"token id {token_id} is in none of the token classes {token_classes}")


def matched_pools(phrase_ids: Sequence[int], features: pd.DataFrame, match_features: Sequence[str],
                  token_classes: dict[str, Sequence[int]]) -> list[np.ndarray]:
    """For each phrase token, the ids in its class range whose match_features all equal its own, minus the
    phrase ids themselves. features: exp1ext.features.token_features output covering every class range."""
    feats = features.set_index("id")
    pools = []
    for tok in phrase_ids:
        lo, hi = token_classes[token_class_of(int(tok), token_classes)]
        cand = feats.loc[(feats.index >= lo) & (feats.index <= hi)]
        same = np.ones(len(cand), dtype=bool)
        for col in match_features:
            same &= cand[col].to_numpy() == feats.at[int(tok), col]
        pool = cand.index.to_numpy(dtype=np.int64)[same]
        pool = pool[~np.isin(pool, np.asarray(phrase_ids, dtype=np.int64))]
        if len(pool) == 0:
            raise ValueError(f"no token matches phrase token {tok} on {list(match_features)}")
        pools.append(pool)
    return pools


def draw_matched(bases: np.ndarray, slots: np.ndarray, pools: Sequence[np.ndarray], seed: int) -> np.ndarray:
    """bases: [n, L]; slots: array indices to fill; pools: one candidate id array per slot.

    Returns [n, len(slots)]: slot j's id drawn uniformly from pools[j], never equal to a base id outside the
    slots or to an id already drawn for the row, so no row can end up with a repeated token. Row i uses
    default_rng([seed, i]), so a row's draw does not depend on how many rows there are.
    """
    keep = np.ones(bases.shape[1], dtype=bool)
    keep[slots] = False
    out = np.empty((len(bases), len(slots)), dtype=np.int64)
    for i, row in enumerate(bases):
        rng = np.random.default_rng([seed, i])
        taken = set(row[keep].tolist())
        for j, pool in enumerate(pools):
            for _ in range(10_000):
                tok = int(pool[rng.integers(len(pool))])
                if tok not in taken:
                    break
            else:
                raise RuntimeError(f"row {i}: could not draw a non-repeating id for slot {j} from {len(pool)} candidates")
            out[i, j] = tok
            taken.add(tok)
    return out


def positions_to_slots(positions: Sequence[int], seq_len: int) -> np.ndarray:
    """Model positions (BOS = 0) -> array indices. The query (model position seq_len) is never a slot."""
    pos = np.asarray(positions, dtype=np.int64)
    if pos.min() < 1 or pos.max() > seq_len - 1:
        raise ValueError(f"slot positions must lie in [1, {seq_len - 1}] (the query, {seq_len}, stays random); got {positions}")
    if len(set(pos.tolist())) != len(pos):
        raise ValueError(f"slot positions repeat: {positions}")
    return pos - 1


def build_variants(bases: np.ndarray, slots: np.ndarray, phrase_ids: Sequence[int], shuffled_ids: Sequence[int],
                   pools: Sequence[np.ndarray], seed: int) -> dict[str, np.ndarray]:
    """{variant: [n, L]} for one phrase x placement; `none` is a copy of the bases."""
    if len(phrase_ids) != len(slots) or sorted(phrase_ids) != sorted(shuffled_ids):
        raise ValueError("phrase and shuffled ids must be permutations of each other, one per slot")

    def write(tokens: np.ndarray) -> np.ndarray:
        seqs = bases.copy()
        seqs[:, slots] = tokens
        return seqs

    return {
        "none": bases.copy(),
        "nl": write(np.asarray(phrase_ids, dtype=np.int64)),
        "shuffled": write(np.asarray(shuffled_ids, dtype=np.int64)),
        "matched": write(draw_matched(bases, slots, pools, seed)),
        "matched_b": write(draw_matched(bases, slots, pools, seed + 1)),
    }


def build_custom(bases: np.ndarray, writes: Sequence[Sequence[int]], seq_len: int) -> np.ndarray:
    """The bases with each [model position, id] in `writes` written in (e.g. one phrase token alone at one slot)."""
    pos = positions_to_slots([w[0] for w in writes], seq_len)
    seqs = bases.copy()
    seqs[:, pos] = np.asarray([w[1] for w in writes], dtype=np.int64)
    return seqs


def check_custom(seqs: np.ndarray, bases: np.ndarray, writes: Sequence[Sequence[int]], seq_len: int, bos_id: int) -> None:
    """Raise unless the written ids sit at their positions, the rest equals the base, and no row repeats a token."""
    pos = positions_to_slots([w[0] for w in writes], seq_len)
    keep = np.ones(bases.shape[1], dtype=bool)
    keep[pos] = False
    if not (seqs[:, pos] == np.asarray([w[1] for w in writes])).all():
        raise AssertionError(f"custom variant {writes}: written ids missing")
    if not (seqs[:, keep] == bases[:, keep]).all():
        raise AssertionError(f"custom variant {writes}: differs from the base outside its positions")
    sorted_rows = np.sort(seqs, axis=1)
    if (sorted_rows[:, 1:] == sorted_rows[:, :-1]).any() or (seqs == bos_id).any():
        raise AssertionError(f"custom variant {writes}: a row repeats a token or contains BOS")


def check_variants(variants: dict[str, np.ndarray], bases: np.ndarray, slots: np.ndarray, phrase_ids: Sequence[int],
                   shuffled_ids: Sequence[int], pools: Sequence[np.ndarray], bos_id: int) -> None:
    """Raise unless: no row repeats a token or contains BOS; every variant equals the base outside the slots; the
    slots hold exactly what the variant says; and no phrase id occurs in a base row outside the slots."""
    keep = np.ones(bases.shape[1], dtype=bool)
    keep[slots] = False
    if np.isin(bases[:, keep], np.asarray(phrase_ids)).any():
        raise AssertionError("a phrase id occurs in a base row outside the slots")
    for name, seqs in variants.items():
        if seqs.shape != bases.shape or seqs.dtype != np.int64:
            raise AssertionError(f"{name}: shape {seqs.shape} / dtype {seqs.dtype} differs from the bases")
        if (seqs == bos_id).any():
            raise AssertionError(f"{name}: BOS id among the random tokens")
        sorted_rows = np.sort(seqs, axis=1)
        if (sorted_rows[:, 1:] == sorted_rows[:, :-1]).any():
            raise AssertionError(f"{name}: a row repeats a token")
        if not (seqs[:, keep] == bases[:, keep]).all():
            raise AssertionError(f"{name}: differs from the base outside the slots")
    if not (variants["nl"][:, slots] == np.asarray(phrase_ids)).all():
        raise AssertionError("nl slots do not hold the phrase ids")
    if not (variants["shuffled"][:, slots] == np.asarray(shuffled_ids)).all():
        raise AssertionError("shuffled slots do not hold the shuffled ids")
    for name in ("matched", "matched_b"):
        for j, pool in enumerate(pools):
            if not np.isin(variants[name][:, slots[j]], pool).all():
                raise AssertionError(f"{name}: slot {j} holds an id outside its matched pool")
