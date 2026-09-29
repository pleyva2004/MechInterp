"""Input designs for Stages 1 and 2: paired triples, position cells, nested dose masks, token sweeps.

Stage 1 draws, for each pair i, three rows of 32 random tokens (no BOS): r_i uniform over the rare
pool, m_i from the rare pool matched position-by-position to the common pool's surface-form strata,
and c_i uniform over the common pool. A cell takes each segment (X = context, P = previous,
Q = final query; see SEGMENTS) from one member of the triple, so every contrast between cells is
paired over i. Stage 2 places single tokens at one position of fixed backgrounds and marks
(token, background) pairs that would repeat a token as missing instead of building them.

Every array is regenerated deterministically in memory (numpy Generator API, seeds from
exp1.sequences.derive_seed) and compared against the on-disk cache by load_or_create_array, which,
like exp1.sequences.load_or_create_sequences, raises on any mismatch and never overwrites.
Rows never repeat a token: repeats trigger induction / duplicate-token heads and would bias the
attention being measured, so every builder here raises rather than emit one.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from exp1.sequences import SequenceCacheMismatch, derive_seed, sample_sequences, sequences_hash
from exp1extended.tokens import stratum_codes, token_features

SEQ_LEN = 32
# Column ranges in a seqs array (no BOS), whose column j is model position j + 1.
SEGMENTS = {"X": slice(0, 30), "P": slice(30, 31), "Q": slice(31, 32)}
SWEEP_GROUPS = ("byte", "common", "rare_uniform", "rare_matched", "rare_transition")

_MAX_ROW_REDRAWS = 1000
_RESERVED_META_KEYS = ("sha256", "dtype", "shape")


def _first_duplicate(seqs: np.ndarray) -> tuple[int, int] | None:
    """(row, token id) of the first within-row repeat, or None."""
    ordered = np.sort(seqs, axis=1)
    repeats = ordered[:, 1:] == ordered[:, :-1]
    if not repeats.any():
        return None
    row, col = (int(x) for x in np.argwhere(repeats)[0])
    return row, int(ordered[row, col])


def _check_seqs_shape(shape: tuple[int, ...], name: str) -> None:
    if len(shape) != 2 or shape[1] != SEQ_LEN:
        raise ValueError(f"{name} must be [n, {SEQ_LEN}], got shape {shape}")


def _seed(cfg: dict, name: str) -> int:
    return derive_seed(int(cfg["seed"]), f"exp1extended/{name}")


# ---- sampling -----------------------------------------------------------------------------------

def sample_pool(pool: tuple[int, int], n: int, seq_len: int, seed: int) -> np.ndarray:
    """int64 [n, seq_len], ids uniform over the inclusive pool, no repeats within a row."""
    return sample_sequences(pool, n, seq_len, seed, without_replacement=True)


def matched_target_probs(pool_strata: np.ndarray, target_strata: np.ndarray) -> tuple[dict[int, float], float]:
    """Stratum distribution of a target pool in which every token is equally likely, restricted to
    strata that occur in the sampling pool.

    pool_strata: [N_pool] codes of the pool we sample from; target_strata: [N_target] codes of the pool
    to match. Returns ({code: prob} over the kept strata, renormalized to sum to 1, sorted by code;
    dropped_mass = the target probability of strata absent from the pool, before renormalizing).
    """
    target_strata = np.asarray(target_strata, dtype=np.int64)
    if target_strata.size == 0:
        raise ValueError("target_strata is empty")
    codes, counts = np.unique(target_strata, return_counts=True)
    probs = counts / counts.sum()
    present = np.isin(codes, np.asarray(pool_strata, dtype=np.int64))
    if not present.any():
        raise ValueError("no target stratum occurs in the pool")
    kept = probs[present] / probs[present].sum()
    return {int(c): float(p) for c, p in zip(codes[present], kept)}, float(probs[~present].sum())


def _strata_members(pool_ids: np.ndarray, pool_strata: np.ndarray, target_probs: dict[int, float]
                    ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """(stratum codes with p > 0 sorted, their probs, sorted pool ids per stratum), validated."""
    pool_ids = np.asarray(pool_ids, dtype=np.int64)
    pool_strata = np.asarray(pool_strata, dtype=np.int64)
    if pool_ids.ndim != 1 or pool_ids.shape != pool_strata.shape:
        raise ValueError(f"pool_ids {pool_ids.shape} and pool_strata {pool_strata.shape} must be matching 1D arrays")
    if np.unique(pool_ids).size != pool_ids.size:
        raise ValueError("pool_ids contains repeated ids; sampling would not be uniform over tokens")

    items = sorted((int(k), float(v)) for k, v in target_probs.items())
    probs = np.array([v for _, v in items])
    if probs.size == 0 or not np.all(np.isfinite(probs)) or np.any(probs < 0):
        raise ValueError(f"target_probs must be non-empty, finite and non-negative, got {target_probs}")
    if abs(probs.sum() - 1.0) > 1e-6:
        raise ValueError(f"target_probs must sum to 1, got {probs.sum()}")
    keep = probs > 0
    strata = np.array([k for k, _ in items], dtype=np.int64)[keep]
    probs = probs[keep] / probs[keep].sum()

    members = [np.sort(pool_ids[pool_strata == s]) for s in strata]
    empty = [int(s) for s, m in zip(strata, members) if m.size == 0]
    if empty:
        raise ValueError(f"target strata {empty} have no pool tokens; drop them first (matched_target_probs)")
    return strata, probs, members


def sample_matched(pool_ids: np.ndarray, pool_strata: np.ndarray, target_probs: dict[int, float],
                   n: int, seq_len: int, seed: int) -> np.ndarray:
    """int64 [n, seq_len] rows whose positions each independently draw a stratum from target_probs,
    then a token uniformly from the pool tokens in that stratum, redrawing on a within-row repeat.

    Drawing the token without replacement within its stratum is the same distribution as redrawing
    on a repeat, and leaves every position's stratum exactly target-distributed. If a row draws a
    stratum more often than it has tokens (possible for tiny strata), that row's strata are redrawn,
    i.e. rows are conditioned on being fillable.
    """
    strata, probs, members = _strata_members(pool_ids, pool_strata, target_probs)
    capacity = np.array([m.size for m in members])
    if seq_len > capacity.sum():
        raise ValueError(f"seq_len={seq_len} exceeds the {capacity.sum()} pool tokens in the target strata")

    rng = np.random.default_rng(seed)
    out = np.empty((n, seq_len), dtype=np.int64)
    for i in range(n):
        for _ in range(_MAX_ROW_REDRAWS):
            row_strata = rng.choice(strata.size, size=seq_len, p=probs)
            counts = np.bincount(row_strata, minlength=strata.size)
            if np.all(counts <= capacity):
                break
        else:
            raise ValueError(f"row {i}: could not draw a fillable stratum sequence in {_MAX_ROW_REDRAWS} tries; "
                             f"strata capacities {dict(zip(strata.tolist(), capacity.tolist()))} are too small")
        for s in np.flatnonzero(counts):
            out[i, row_strata == s] = members[s][rng.choice(capacity[s], size=counts[s], replace=False)]

    dup = _first_duplicate(out)
    if dup is not None:
        raise AssertionError(f"sample_matched produced row {dup[0]} with repeated token {dup[1]}")
    return out


def _capped_quotas(probs: np.ndarray, capacity: np.ndarray, n: int) -> np.ndarray:
    """Integer per-stratum counts summing to n, proportional to probs where capacity allows: strata
    whose share exceeds their capacity are filled completely and the rest is re-split over the others
    (largest-remainder rounding, ties to the lower index)."""
    quota = np.zeros(probs.size, dtype=np.int64)
    free = np.ones(probs.size, dtype=bool)
    remaining = n
    while free.any() and remaining > 0:
        ideal = np.zeros(probs.size)
        ideal[free] = remaining * probs[free] / probs[free].sum()
        over = free & (ideal > capacity)
        if over.any():
            quota[over] = capacity[over]
            remaining -= int(capacity[over].sum())
            free &= ~over
            continue
        base = np.floor(ideal).astype(np.int64)
        extra = remaining - int(base[free].sum())
        frac = np.where(free, ideal - base, -1.0)
        base[np.argsort(-frac, kind="stable")[:extra]] += 1
        quota[free] = base[free]
        remaining = 0
    return quota


def _matched_set(pool_ids: np.ndarray, pool_strata: np.ndarray, target_probs: dict[int, float], n: int,
                 seed: int) -> tuple[np.ndarray, float, list[int]]:
    """n distinct pool ids whose stratum composition follows target_probs as closely as the strata
    allow (fixed quotas, not iid draws: a set of distinct tokens cannot oversample a tiny stratum).

    Returns (sorted ids [n], total-variation distance between the achieved and target composition,
    codes of strata taken whole yet still short of their target share).
    """
    strata, probs, members = _strata_members(pool_ids, pool_strata, target_probs)
    capacity = np.array([m.size for m in members])
    if n > capacity.sum():
        raise ValueError(f"n={n} exceeds the {capacity.sum()} pool tokens in the target strata")
    quota = _capped_quotas(probs, capacity, n)
    if quota.sum() != n or np.any(quota > capacity):
        raise AssertionError(f"quota allocation failed: {quota} for n={n}, capacity {capacity}")
    rng = np.random.default_rng(seed)
    ids = np.concatenate([m[rng.choice(m.size, size=q, replace=False)] for m, q in zip(members, quota)])
    tv = 0.5 * float(np.abs(quota / n - probs).sum())
    capped = [int(s) for s, q, c, p in zip(strata, quota, capacity, probs) if q == c and q < n * p]
    return np.sort(ids), tv, capped


def matching_setup(cfg: dict, tokenizer, embed: np.ndarray | None = None) -> dict:
    """Features and strata for matching the rare pool to the common pool, as configured.

    Returns dict: pool_ids [N_rare], pool_features, pool_strata [N_rare] (the rare pool, sampled from);
    target_features, target_strata [N_common] (the common pool, matched to); target_probs and
    dropped_mass from matched_target_probs. Callers report dropped_mass and a balance_table of
    sampled-vs-target features.
    """
    columns = list(cfg["matching"]["features"])
    len_bins = list(cfg["matching"]["len_bins"])
    (r_lo, r_hi), (c_lo, c_hi) = cfg["pools"]["rare"], cfg["pools"]["common"]
    pool_ids = np.arange(int(r_lo), int(r_hi) + 1, dtype=np.int64)
    target_ids = np.arange(int(c_lo), int(c_hi) + 1, dtype=np.int64)
    pool_features = token_features(tokenizer, pool_ids, len_bins, embed)
    target_features = token_features(tokenizer, target_ids, len_bins, embed)
    pool_strata = stratum_codes(pool_features, columns)
    target_strata = stratum_codes(target_features, columns)
    target_probs, dropped_mass = matched_target_probs(pool_strata, target_strata)
    return {
        "pool_ids": pool_ids, "pool_features": pool_features, "pool_strata": pool_strata,
        "target_features": target_features, "target_strata": target_strata,
        "target_probs": target_probs, "dropped_mass": dropped_mass,
    }


def paired_triples(cfg: dict, tokenizer, embed: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Stage 1 triples {"r", "m", "c"}, each int64 [n_pairs, 32]; row i of each is pair i.

    r: uniform over the rare pool; m: rare pool matched to the common pool's joint distribution of
    cfg["matching"]["features"] (independently per position); c: uniform over the common pool.
    Seeds "exp1extended/stage1/{r,m,c}". The tokenizer only decodes ids for the matching features.
    """
    n, seq_len = int(cfg["stage1"]["n_pairs"]), int(cfg["seq_len"])
    if seq_len != SEQ_LEN:
        raise ValueError(f"SEGMENTS assume seq_len {SEQ_LEN}, config has {seq_len}")
    match = matching_setup(cfg, tokenizer, embed)
    triples = {
        "r": sample_pool(cfg["pools"]["rare"], n, seq_len, _seed(cfg, "stage1/r")),
        "m": sample_matched(match["pool_ids"], match["pool_strata"], match["target_probs"], n, seq_len,
                            _seed(cfg, "stage1/m")),
        "c": sample_pool(cfg["pools"]["common"], n, seq_len, _seed(cfg, "stage1/c")),
    }
    pool_of = {"r": cfg["pools"]["rare"], "m": cfg["pools"]["rare"], "c": cfg["pools"]["common"]}
    for member, seqs in triples.items():
        check_design(seqs, {seg: pool_of[member] for seg in SEGMENTS})
    return triples


# ---- recombination and dose ---------------------------------------------------------------------

def cell_name(spec: dict[str, str]) -> str:
    """{"Q": "r", "P": "c", "X": "c"} -> "Qr_Pc_Xc"."""
    return f"Q{spec['Q']}_P{spec['P']}_X{spec['X']}"


def recombine(triples: dict[str, np.ndarray], spec: dict[str, str]) -> np.ndarray:
    """int64 [n, 32] cell whose segment Q / P / X is copied from triples[spec[segment]].

    Raises ValueError on a within-row repeat: r and m share the rare id range, so a cell mixing
    them (e.g. Q from m, X from r) can repeat a token; the configured cells never mix r and m.
    """
    if set(spec) != set(SEGMENTS):
        raise ValueError(f"spec must map exactly {sorted(SEGMENTS)} to members, got {spec}")
    missing = sorted({m for m in spec.values() if m not in triples})
    if missing:
        raise ValueError(f"spec {spec} uses members {missing} not in triples {sorted(triples)}")
    shapes = {triples[m].shape for m in spec.values()}
    if len(shapes) != 1:
        raise ValueError(f"members used by {spec} have different shapes {shapes}")
    shape = shapes.pop()
    _check_seqs_shape(shape, "triples members")

    out = np.empty(shape, dtype=np.int64)
    for seg, member in spec.items():
        out[:, SEGMENTS[seg]] = triples[member][:, SEGMENTS[seg]]
    dup = _first_duplicate(out)
    if dup is not None:
        raise ValueError(f"cell {cell_name(spec)}: row {dup[0]} repeats token {dup[1]}")
    return out


def dose_masks(n: int, levels: list[int], seed: int, n_positions: int = 30) -> np.ndarray:
    """bool [len(levels), n, n_positions]: mask[l, i] marks the first levels[l] entries of one random
    permutation of row i's positions, so each mask has exactly levels[l] True per row and a mask
    for a smaller level is a subset of every mask for a larger one (dose curves are paired per row).
    """
    levels = [int(k) for k in levels]
    bad = [k for k in levels if not 0 <= k <= n_positions]
    if bad:
        raise ValueError(f"levels {bad} outside [0, {n_positions}]")
    rng = np.random.default_rng(seed)
    order = rng.permuted(np.tile(np.arange(n_positions), (n, 1)), axis=1)
    rank = np.argsort(order, axis=1)  # rank[i, j] = where position j falls in row i's permutation
    return rank[None, :, :] < np.array(levels, dtype=np.int64)[:, None, None]


def dose_sequences(base: np.ndarray, donor: np.ndarray, mask_level: np.ndarray, qp_member: np.ndarray) -> np.ndarray:
    """int64 [n, 32]: X positions where mask_level [n, 30] is True come from donor, the others from
    base; Q and P come from qp_member. Raises ValueError on a within-row repeat."""
    for name, arr in (("base", base), ("donor", donor), ("qp_member", qp_member)):
        _check_seqs_shape(arr.shape, name)
    if not base.shape == donor.shape == qp_member.shape:
        raise ValueError(f"shape mismatch: base {base.shape}, donor {donor.shape}, qp_member {qp_member.shape}")
    x = SEGMENTS["X"]
    if mask_level.dtype != bool or mask_level.shape != (base.shape[0], x.stop - x.start):
        raise ValueError(f"mask_level must be bool [{base.shape[0]}, {x.stop - x.start}], "
                         f"got {mask_level.dtype} {mask_level.shape}")
    out = np.array(qp_member, dtype=np.int64, copy=True)
    out[:, x] = np.where(mask_level, donor[:, x], base[:, x])
    dup = _first_duplicate(out)
    if dup is not None:
        raise ValueError(f"dose row {dup[0]} repeats token {dup[1]}")
    return out


# ---- Stage 2 sweep ------------------------------------------------------------------------------

def _uniform_excluding(id_range: tuple[int, int], n: int, exclude: np.ndarray, seed: int) -> np.ndarray:
    lo, hi = int(id_range[0]), int(id_range[1])
    candidates = np.setdiff1d(np.arange(lo, hi + 1, dtype=np.int64), exclude)
    if n > candidates.size:
        raise ValueError(f"need {n} ids from [{lo}, {hi}] but only {candidates.size} remain after exclusions")
    return np.sort(np.random.default_rng(seed).choice(candidates, size=n, replace=False))


def sweep_tokens(cfg: dict, tokenizer, embed: np.ndarray | None = None) -> pd.DataFrame:
    """Stage 2 token set: DataFrame with columns id (int64, all distinct) and group, in SWEEP_GROUPS
    order, ids sorted within a group.

    byte / common: every id of cfg["stage2"]["tokens"]["all"], grouped by pool. rare_uniform: uniform
    over the rare pool. rare_matched: rare pool tokens with the common pool's stratum composition
    (fixed quotas; a stratum with too few rare tokens is taken whole and its shortfall re-split over
    the others). rare_transition: uniform over its range. Each later group excludes ids already chosen.
    Seeds "exp1extended/stage2/tokens/<group>". df.attrs records the matching quality:
    rare_matched_dropped_mass, rare_matched_composition_tv, rare_matched_capped_strata.
    """
    tok_cfg = cfg["stage2"]["tokens"]
    pools = cfg["pools"]
    a_lo, a_hi = (int(v) for v in tok_cfg["all"])
    all_ids = np.arange(a_lo, a_hi + 1, dtype=np.int64)
    in_byte = (all_ids >= pools["byte"][0]) & (all_ids <= pools["byte"][1])
    in_common = (all_ids >= pools["common"][0]) & (all_ids <= pools["common"][1])
    if not np.all(in_byte | in_common):
        raise ValueError(f"stage2.tokens.all {tok_cfg['all']} reaches outside the byte and common pools")

    parts = {"byte": all_ids[in_byte], "common": all_ids[in_common]}
    chosen = np.concatenate(list(parts.values()))

    parts["rare_uniform"] = _uniform_excluding(pools["rare"], int(tok_cfg["rare_uniform"]), chosen,
                                               _seed(cfg, "stage2/tokens/rare_uniform"))
    chosen = np.concatenate([chosen, parts["rare_uniform"]])

    match = matching_setup(cfg, tokenizer, embed)
    keep = ~np.isin(match["pool_ids"], chosen)
    probs, dropped = matched_target_probs(match["pool_strata"][keep], match["target_strata"])
    parts["rare_matched"], tv, capped = _matched_set(match["pool_ids"][keep], match["pool_strata"][keep], probs,
                                                     int(tok_cfg["rare_matched"]),
                                                     _seed(cfg, "stage2/tokens/rare_matched"))
    chosen = np.concatenate([chosen, parts["rare_matched"]])

    trans = tok_cfg["rare_transition"]
    t_lo, t_hi = (int(v) for v in trans["range"])
    if not pools["rare"][0] <= t_lo <= t_hi <= pools["rare"][1]:
        raise ValueError(f"rare_transition range {trans['range']} is not inside the rare pool {pools['rare']}")
    parts["rare_transition"] = _uniform_excluding((t_lo, t_hi), int(trans["n"]), chosen,
                                                  _seed(cfg, "stage2/tokens/rare_transition"))

    df = pd.DataFrame({
        "id": np.concatenate([parts[g] for g in SWEEP_GROUPS]).astype(np.int64),
        "group": np.concatenate([np.full(parts[g].size, g, dtype=object) for g in SWEEP_GROUPS]),
    })
    if df["id"].duplicated().any():
        raise AssertionError("sweep_tokens produced a repeated id")
    df.attrs.update(rare_matched_dropped_mass=dropped, rare_matched_composition_tv=tv,
                    rare_matched_capped_strata=capped)
    return df


def build_sweep(backgrounds: np.ndarray, column: int, token_ids: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Place every token at `column` of every background.

    backgrounds: int64 [K, L] rows without repeats; token_ids: [T]. A (t, k) pair whose token already
    occurs at another column of background k would repeat a token, so it is not built and is marked
    missing[t, k] (a token equal to background k's own token at `column` is fine: that row is the
    background itself). Returns (seqs [n_valid, L], tok_idx [n_valid], bg_idx [n_valid], missing
    bool [T, K]) with rows ordered by (token, background).
    """
    backgrounds = np.asarray(backgrounds)
    token_ids = np.asarray(token_ids, dtype=np.int64)
    if backgrounds.dtype != np.int64 or backgrounds.ndim != 2:
        raise ValueError(f"backgrounds must be int64 [K, L], got {backgrounds.dtype} {backgrounds.shape}")
    if token_ids.ndim != 1:
        raise ValueError(f"token_ids must be 1D, got shape {token_ids.shape}")
    if not 0 <= column < backgrounds.shape[1]:
        raise ValueError(f"column {column} outside [0, {backgrounds.shape[1]})")
    dup = _first_duplicate(backgrounds)
    if dup is not None:
        raise ValueError(f"background {dup[0]} already repeats token {dup[1]}")

    others = np.delete(backgrounds, column, axis=1)
    missing = (token_ids[:, None, None] == others[None, :, :]).any(axis=2)
    tok_idx, bg_idx = np.nonzero(~missing)  # row-major, so ordered by token then background
    seqs = backgrounds[bg_idx].copy()
    seqs[:, column] = token_ids[tok_idx]
    if _first_duplicate(seqs) is not None:
        raise AssertionError("build_sweep produced a repeated token despite the collision mask")
    return seqs, tok_idx.astype(np.int64), bg_idx.astype(np.int64), missing


# ---- cache and checks ---------------------------------------------------------------------------

def _json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"meta value {obj!r} of type {type(obj).__name__} is not JSON serializable")


def load_or_create_array(name: str, array: np.ndarray, data_dir: Path, meta: dict) -> np.ndarray:
    """Cache `array` as data_dir/{name}.npy + {name}.json, or verify it against an existing cache.

    The caller regenerates `array` deterministically in memory every run. If neither file exists both
    are written (json = meta + dtype, shape and sha256 via exp1.sequences.sequences_hash). If both
    exist, any difference in meta, dtype, shape, the file's own hash or the regenerated array's hash
    raises SequenceCacheMismatch; the cached array is returned only when everything agrees. Existing
    files are never overwritten, so a stale cache has to be moved away by hand.
    """
    array = np.asarray(array)
    # sequences_hash hashes int64 bytes, which is lossless only for integer and bool arrays.
    if not (np.issubdtype(array.dtype, np.integer) or array.dtype == bool):
        raise ValueError(f"only integer or bool arrays can be cached, got dtype {array.dtype}")
    clash = sorted(set(meta) & set(_RESERVED_META_KEYS))
    if clash:
        raise ValueError(f"meta keys {clash} are reserved")

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    npy_path, json_path = data_dir / f"{name}.npy", data_dir / f"{name}.json"
    expected_hash = sequences_hash(array)
    # Round-trip through JSON so tuples and numpy scalars compare equal to what is read back.
    expected_meta = json.loads(json.dumps(dict(meta, dtype=str(array.dtype), shape=list(array.shape)),
                                          default=_json_default))

    npy_exists, json_exists = npy_path.exists(), json_path.exists()
    if not npy_exists and not json_exists:
        np.save(npy_path, array)
        json_path.write_text(json.dumps(dict(expected_meta, sha256=expected_hash), indent=2))
        return array
    if npy_exists != json_exists:
        existing, absent = (npy_path, json_path) if npy_exists else (json_path, npy_path)
        raise SequenceCacheMismatch(f"Incomplete cache for {name!r} in {data_dir}: {existing.name} exists but "
                                    f"{absent.name} is missing. Move or delete {existing} explicitly.")

    found = json.loads(json_path.read_text())
    found_sha = found.pop("sha256", "<missing>")
    mismatches = [f"{key}: expected {expected_meta.get(key, '<missing>')!r}, found {found.get(key, '<missing>')!r}"
                  for key in sorted(set(expected_meta) | set(found))
                  if expected_meta.get(key, "<missing>") != found.get(key, "<missing>")]
    loaded = np.load(npy_path)
    if str(loaded.dtype) != found.get("dtype") or list(loaded.shape) != found.get("shape"):
        mismatches.append(f"{npy_path.name} is {loaded.dtype} {list(loaded.shape)}, json records "
                          f"{found.get('dtype')} {found.get('shape')}")
    loaded_hash = sequences_hash(loaded)
    if loaded_hash != found_sha:
        mismatches.append(f"sha256 (file integrity): {json_path.name} says {found_sha!r}, "
                          f"{npy_path.name} hashes to {loaded_hash!r}")
    if loaded_hash != expected_hash:
        mismatches.append(f"sha256 (reproducibility): cached {loaded_hash!r}, regenerated {expected_hash!r}")
    if mismatches:
        raise SequenceCacheMismatch(f"Cached array {name!r} in {data_dir} does not match:\n  "
                                    + "\n  ".join(mismatches)
                                    + f"\nMove or delete {npy_path} and {json_path} explicitly to regenerate.")
    return loaded


def _as_ranges(pools) -> list[tuple[int, int]]:
    arr = np.asarray(pools, dtype=np.int64)
    if arr.shape == (2,):
        return [(int(arr[0]), int(arr[1]))]
    if arr.ndim == 2 and arr.shape[1] == 2:
        return [(int(lo), int(hi)) for lo, hi in arr]
    raise ValueError(f"a segment pool must be (lo, hi) or a list of (lo, hi), got {pools!r}")


def check_design(seqs: np.ndarray, segment_pools: dict[str, tuple[int, int]]) -> None:
    """Raise AssertionError if seqs is not int64 [n, 32], a row repeats a token, or a segment holds
    an id outside its pool. segment_pools maps "Q" / "P" / "X" to an inclusive (lo, hi), or to a
    list of them (e.g. dose X mixes the rare and common pools); unlisted segments are not checked.
    """
    if seqs.dtype != np.int64 or seqs.ndim != 2 or seqs.shape[1] != SEQ_LEN:
        raise AssertionError(f"expected int64 [n, {SEQ_LEN}], got {seqs.dtype} {seqs.shape}")
    dup = _first_duplicate(seqs)
    if dup is not None:
        raise AssertionError(f"row {dup[0]} repeats token {dup[1]}")
    for seg, pools in segment_pools.items():
        if seg not in SEGMENTS:
            raise ValueError(f"unknown segment {seg!r}; expected one of {sorted(SEGMENTS)}")
        vals = seqs[:, SEGMENTS[seg]]
        ok = np.zeros(vals.shape, dtype=bool)
        for lo, hi in _as_ranges(pools):
            ok |= (vals >= lo) & (vals <= hi)
        if not ok.all():
            row, col = (int(x) for x in np.argwhere(~ok)[0])
            raise AssertionError(f"segment {seg}: id {int(vals[row, col])} at row {row} is outside {pools}")
