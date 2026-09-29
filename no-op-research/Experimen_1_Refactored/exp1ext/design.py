"""Build the exp1ext input sequences from cached random blocks.

Every random block is sampled and cached by exp1.sequences.load_or_create_sequences (SHA-256
verified, never silently regenerated); the functions here only recombine blocks, so every
design is a deterministic function of the cache. Arrays hold the 32 random tokens WITHOUT BOS;
the caller prepends BOS. Array index i therefore sits at model position i + 1.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from exp1.sequences import check_sequences, load_or_create_sequences

CELLS = ("AA", "AC", "CA", "CC")  # first letter = context class, second = final-token class (A rare, C common)


def cached_block(name: str, id_range: list[int], n: int, length: int, base_seed: int, bos_id: int,
                 data_dir: str | Path) -> np.ndarray:
    """n rows of `length` ids from id_range, no repeats within a row, cached under data_dir/{name}."""
    return load_or_create_sequences(name, id_range, n, length, base_seed, True, bos_id, data_dir)


def build_factorial(rare: np.ndarray, common: np.ndarray) -> dict[str, np.ndarray]:
    """rare, common: [n, seq_len] rows without repeats. Returns the four 2x2 cells, index-aligned:
    row i of every cell reuses row i's context and final token, so effects can be paired per row.

    The rare and common ranges are disjoint, so swapping in the other class's final token can
    never create a repeated token.
    """
    if rare.shape != common.shape:
        raise ValueError(f"rare {rare.shape} and common {common.shape} blocks must have the same shape")
    return {
        "AA": rare.copy(),
        "AC": np.concatenate([rare[:, :-1], common[:, -1:]], axis=1),
        "CA": np.concatenate([common[:, :-1], rare[:, -1:]], axis=1),
        "CC": common.copy(),
    }


def build_dose_response(rare: np.ndarray, replacements: np.ndarray, ks: list[int], seed: int) -> dict[int, np.ndarray]:
    """rare: [n, seq_len]; replacements: [n, seq_len - 1] common ids (disjoint from the rare range).

    For each k, replace k of the seq_len - 1 context positions of every rare row with that row's first k
    replacement ids; the final token stays rare. Nested by construction: the positions and ids replaced
    at k are a subset of those replaced at any larger k, so the curve over k is paired within each row.
    """
    n, seq_len = rare.shape
    if replacements.shape != (n, seq_len - 1):
        raise ValueError(f"replacements must be [{n}, {seq_len - 1}], got {replacements.shape}")
    if max(ks) > seq_len - 1 or min(ks) < 0:
        raise ValueError(f"ks must lie in [0, {seq_len - 1}], got {ks}")
    order = np.stack([np.random.default_rng([seed, i]).permutation(seq_len - 1) for i in range(n)])
    out = {}
    for k in ks:
        seqs = rare.copy()
        rows = np.repeat(np.arange(n), k)
        seqs[rows, order[:, :k].ravel()] = replacements[:, :k].ravel()
        out[k] = seqs
    return out


def position_sweep(bases: np.ndarray, tokens: np.ndarray, index: int) -> tuple[np.ndarray, np.ndarray]:
    """Every base with array position `index` replaced by every token.

    bases: [K, seq_len]; tokens: [T]. Returns (seqs [K * T, seq_len] in base-major order, so row k * T + j
    is base k with token j, and invalid [K, T]: True where token j already occurs elsewhere in base k, i.e.
    the substitution would create a repeated token; those rows are still built but must be masked out).
    """
    K, seq_len = bases.shape
    if not 0 <= index < seq_len:
        raise ValueError(f"index {index} outside [0, {seq_len})")
    seqs = np.repeat(bases, len(tokens), axis=0)
    seqs[:, index] = np.tile(tokens, K)
    others = np.delete(bases, index, axis=1)  # [K, seq_len - 1]
    invalid = (others[:, None, :] == tokens[None, :, None]).any(axis=2)
    return seqs, invalid


def check_design(seqs: np.ndarray, allowed_ranges: list[list[int]], bos_id: int) -> None:
    """Every id lies in one of the allowed ranges and BOS never appears among the random tokens."""
    in_range = np.zeros(seqs.shape, dtype=bool)
    for lo, hi in allowed_ranges:
        in_range |= (seqs >= lo) & (seqs <= hi)
    if not in_range.all():
        row, col = np.argwhere(~in_range)[0]
        raise AssertionError(f"id {seqs[row, col]} at row {row}, col {col} is outside {allowed_ranges}")
    check_sequences(seqs, [int(seqs.min()), int(seqs.max())], without_replacement=False, bos_id=bos_id)
