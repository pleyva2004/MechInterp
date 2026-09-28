"""Sample and cache reproducible random-token-ID sequences for GPT-2 small.

Sequences are generated with numpy's Generator API (never the stdlib `random` or
`hash()`, both of which are process- or seed-state dependent in ways that break
reproducibility across runs/machines) and cached on disk per condition. A cached
array is only ever read back and re-validated, never silently regenerated or
overwritten, so a stale or corrupted cache surfaces as a loud error instead of
quietly biasing results.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


class SequenceCacheMismatch(RuntimeError):
    """Raised when an on-disk sequence cache doesn't match the requested config
    or fails an integrity check. The caller must move/delete the offending
    files themselves; this module will not overwrite an existing cache.
    """


def stable_index(name: str) -> int:
    """Deterministic non-negative int derived from `name`.

    Uses sha256 rather than Python's `hash()`, which is randomized per process
    (PYTHONHASHSEED) and therefore not reproducible across runs or machines.
    """
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def derive_seed(base_seed: int, condition: str) -> int:
    """Per-condition seed that stays reproducible but keeps conditions independent."""
    return base_seed + stable_index(condition)


def sample_sequences(
    id_range: tuple[int, int] | list[int],
    n: int,
    seq_len: int,
    seed: int,
    without_replacement: bool = True,
) -> np.ndarray:
    """n random-token-id sequences of length seq_len, ids drawn from inclusive [lo, hi].

    without_replacement=True samples each row without repeats: repeated tokens within
    a row trigger induction / duplicate-token heads and would bias the very attention
    patterns this experiment measures. Repeats across different rows are fine.
    """
    lo, hi = int(id_range[0]), int(id_range[1])
    if lo > hi:
        raise ValueError(f"id_range invalid: lo={lo} > hi={hi}")
    range_size = hi - lo + 1
    if without_replacement and seq_len > range_size:
        raise ValueError(
            f"seq_len={seq_len} exceeds id_range size {range_size} (range=[{lo}, {hi}]); "
            "cannot sample without replacement"
        )

    rng = np.random.default_rng(seed)
    if without_replacement:
        seqs = np.empty((n, seq_len), dtype=np.int64)
        for i in range(n):
            seqs[i] = rng.choice(range_size, size=seq_len, replace=False) + lo
    else:
        seqs = rng.integers(lo, hi + 1, size=(n, seq_len), dtype=np.int64)

    check_sequences(seqs, id_range, without_replacement, bos_id=None)
    return seqs


def check_sequences(
    seqs: np.ndarray,
    id_range: tuple[int, int] | list[int],
    without_replacement: bool,
    bos_id: int | None,
) -> None:
    """Validate dtype/shape/range/duplicate/BOS invariants on a sequence array.

    Uses explicit `raise AssertionError` (not bare `assert`) so these checks still
    run under `python -O`.
    """
    if seqs.dtype != np.int64:
        raise AssertionError(f"expected dtype int64, got {seqs.dtype}")
    if seqs.ndim != 2:
        raise AssertionError(f"expected a 2D array, got ndim={seqs.ndim} (shape={seqs.shape})")

    lo, hi = int(id_range[0]), int(id_range[1])
    out_of_range = (seqs < lo) | (seqs > hi)
    if np.any(out_of_range):
        row, col = (int(x) for x in np.argwhere(out_of_range)[0])
        raise AssertionError(
            f"value {int(seqs[row, col])} at row {row}, col {col} is outside id_range [{lo}, {hi}]"
        )

    if without_replacement:
        for row in range(seqs.shape[0]):
            vals, counts = np.unique(seqs[row], return_counts=True)
            dup_idx = np.flatnonzero(counts > 1)
            if dup_idx.size > 0:
                raise AssertionError(
                    f"row {row} has duplicate token id {int(vals[dup_idx[0]])} "
                    "but without_replacement=True"
                )

    if bos_id is not None:
        matches = np.argwhere(seqs == bos_id)
        if matches.size > 0:
            row, col = (int(x) for x in matches[0])
            raise AssertionError(f"bos_id {bos_id} found at row {row}, col {col}")


def sequences_hash(seqs: np.ndarray) -> str:
    """Hex sha256 over shape-prefixed, contiguous int64 bytes (so a reshape changes it)."""
    payload = f"{seqs.shape}".encode() + np.ascontiguousarray(seqs, dtype="<i8").tobytes()
    return hashlib.sha256(payload).hexdigest()


def load_or_create_sequences(
    condition: str,
    id_range: tuple[int, int] | list[int],
    n: int,
    seq_len: int,
    base_seed: int,
    without_replacement: bool,
    bos_id: int,
    data_dir: str | Path,
) -> np.ndarray:
    """Load condition's cached sequences from data_dir, creating them on first use.

    The array is always cheap to resample, so we always regenerate it in memory from
    derive_seed(base_seed, condition) and compare against whatever is on disk. A mismatch
    (either in the recorded config or in the hashes) means the cache is stale or corrupt;
    we raise rather than overwrite so nobody silently loses or mutates already-cached data.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    npy_path = data_dir / f"{condition}.npy"
    json_path = data_dir / f"{condition}.json"

    seed = derive_seed(base_seed, condition)
    expected = sample_sequences(id_range, n, seq_len, seed, without_replacement)
    expected_hash = sequences_hash(expected)
    lo, hi = int(id_range[0]), int(id_range[1])
    expected_meta = {
        "condition": condition,
        "id_range": [lo, hi],
        "n": n,
        "seq_len": seq_len,
        "base_seed": base_seed,
        "seed": seed,
        "without_replacement": without_replacement,
    }

    npy_exists = npy_path.exists()
    json_exists = json_path.exists()

    if not npy_exists and not json_exists:
        np.save(npy_path, expected)
        meta = dict(expected_meta, sha256=expected_hash)
        json_path.write_text(json.dumps(meta, indent=2))
        result = expected
    elif npy_exists and json_exists:
        found_meta = json.loads(json_path.read_text())
        mismatches = [
            f"{key}: expected {exp_val!r}, found {found_meta.get(key, '<missing>')!r}"
            for key, exp_val in expected_meta.items()
            if found_meta.get(key, "<missing>") != exp_val
        ]

        loaded = np.load(npy_path)
        loaded_hash = sequences_hash(loaded)
        found_sha = found_meta.get("sha256", "<missing>")
        if loaded_hash != found_sha:
            mismatches.append(
                f"sha256 (file integrity): {json_path.name} says {found_sha!r}, "
                f"but {npy_path.name} actually hashes to {loaded_hash!r}"
            )
        if loaded_hash != expected_hash:
            mismatches.append(
                f"sha256 (reproducibility): cached array hashes to {loaded_hash!r}, "
                f"freshly sampled array hashes to {expected_hash!r}"
            )

        if mismatches:
            raise SequenceCacheMismatch(
                f"Cached sequences for condition {condition!r} in {data_dir} do not match the "
                "requested configuration or are corrupted:\n  "
                + "\n  ".join(mismatches)
                + f"\nMove or delete {npy_path} and {json_path} explicitly to regenerate them."
            )
        result = loaded
    else:
        existing = npy_path if npy_exists else json_path
        missing = json_path if npy_exists else npy_path
        raise SequenceCacheMismatch(
            f"Incomplete sequence cache for condition {condition!r} in {data_dir}: "
            f"{existing.name} exists but {missing.name} is missing. "
            f"Move or delete {existing} explicitly before regenerating."
        )

    check_sequences(result, id_range, without_replacement, bos_id)
    return result


def prepend_bos(seqs: np.ndarray, bos_id: int) -> np.ndarray:
    """Prepend bos_id as column 0, shape (n, seq_len) -> (n, seq_len + 1).

    Done directly on the id array: we never decode ids to text and re-tokenize (that
    doesn't round-trip), and never rely on a library's default/implicit BOS handling.
    """
    n, seq_len = seqs.shape
    out = np.empty((n, seq_len + 1), dtype=np.int64)
    out[:, 0] = bos_id
    out[:, 1:] = seqs
    return out
