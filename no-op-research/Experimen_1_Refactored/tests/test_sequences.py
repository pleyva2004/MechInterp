"""Tests for exp1.sequences: sampling, validation, caching, and BOS prepending."""

import json

import numpy as np
import pytest

from exp1.sequences import (
    SequenceCacheMismatch,
    check_sequences,
    derive_seed,
    load_or_create_sequences,
    prepend_bos,
    sample_sequences,
    sequences_hash,
    stable_index,
)

# The three real conditions from configs/exp1.yaml.
CONDITIONS = {
    "A_rare": (1000, 39999),
    "B_common": (0, 999),
    "C_common_clean": (256, 999),
}


# --- range / dtype / shape -------------------------------------------------


@pytest.mark.parametrize("condition,id_range", CONDITIONS.items())
def test_range_real_conditions(condition, id_range):
    lo, hi = id_range
    seqs = sample_sequences(id_range, n=1000, seq_len=32, seed=derive_seed(0, condition))
    assert seqs.min() >= lo
    assert seqs.max() <= hi


def test_dtype_and_shape():
    seqs = sample_sequences((0, 999), n=50, seq_len=32, seed=1)
    assert seqs.dtype == np.int64
    assert seqs.shape == (50, 32)


# --- no duplicates -----------------------------------------------------------


def test_no_duplicates_within_row():
    seqs = sample_sequences((0, 999), n=200, seq_len=32, seed=2, without_replacement=True)
    for row in seqs:
        assert len(set(row.tolist())) == len(row)


def test_tight_range_every_row_is_a_permutation():
    lo, hi = 256, 999
    seq_len = hi - lo + 1
    seqs = sample_sequences((lo, hi), n=20, seq_len=seq_len, seed=3, without_replacement=True)
    expected = set(range(lo, hi + 1))
    for row in seqs:
        assert set(row.tolist()) == expected


def test_seq_len_exceeds_range_raises_value_error():
    with pytest.raises(ValueError):
        sample_sequences((0, 9), n=5, seq_len=11, seed=0, without_replacement=True)


# --- determinism -------------------------------------------------------------


def test_same_seed_same_array():
    a = sample_sequences((0, 999), n=100, seq_len=16, seed=42)
    b = sample_sequences((0, 999), n=100, seq_len=16, seed=42)
    np.testing.assert_array_equal(a, b)


def test_different_seed_different_array():
    a = sample_sequences((0, 999), n=100, seq_len=16, seed=42)
    b = sample_sequences((0, 999), n=100, seq_len=16, seed=43)
    assert not np.array_equal(a, b)


def test_different_condition_via_derive_seed_different_array():
    a = sample_sequences((0, 999), n=100, seq_len=16, seed=derive_seed(0, "cond_a"))
    b = sample_sequences((0, 999), n=100, seq_len=16, seed=derive_seed(0, "cond_b"))
    assert not np.array_equal(a, b)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("A_rare", 1424353956),
        ("B_common", 332731810),
        ("C_common_clean", 3519401512),
        ("cond_a", 3904752551),
        ("cond_b", 682782701),
    ],
)
def test_stable_index_known_values(name, expected):
    # Hard-coded expected values (first 4 bytes of sha256, big-endian) so switching
    # to hash() (process-randomized) or a different digest would be caught.
    assert stable_index(name) == expected


def test_derive_seed_is_base_plus_stable_index():
    assert derive_seed(10, "A_rare") == 10 + stable_index("A_rare")


# --- check_sequences ----------------------------------------------------------


def test_check_sequences_raises_on_out_of_range():
    seqs = np.array([[0, 1, 2], [3, 4, 1000]], dtype=np.int64)
    with pytest.raises(AssertionError):
        check_sequences(seqs, (0, 999), without_replacement=False, bos_id=None)


def test_check_sequences_raises_on_duplicate_within_row():
    seqs = np.array([[0, 1, 2], [3, 3, 5]], dtype=np.int64)
    with pytest.raises(AssertionError):
        check_sequences(seqs, (0, 999), without_replacement=True, bos_id=None)


def test_check_sequences_raises_on_bos_present():
    seqs = np.array([[0, 1, 2], [3, 4, 50256]], dtype=np.int64)
    with pytest.raises(AssertionError):
        check_sequences(seqs, (0, 60000), without_replacement=False, bos_id=50256)


def test_check_sequences_passes_on_valid_input():
    seqs = sample_sequences((0, 999), n=10, seq_len=16, seed=5)
    check_sequences(seqs, (0, 999), without_replacement=True, bos_id=50256)


# --- load_or_create_sequences --------------------------------------------------


def _kwargs(tmp_path, **overrides):
    kwargs = dict(
        condition="A_rare",
        id_range=(1000, 39999),
        n=50,
        seq_len=32,
        base_seed=0,
        without_replacement=True,
        bos_id=50256,
        data_dir=tmp_path,
    )
    kwargs.update(overrides)
    return kwargs


def test_load_or_create_first_call_writes_both_files(tmp_path):
    seqs = load_or_create_sequences(**_kwargs(tmp_path))
    npy_path = tmp_path / "A_rare.npy"
    json_path = tmp_path / "A_rare.json"
    assert npy_path.exists()
    assert json_path.exists()
    assert seqs.shape == (50, 32)

    meta = json.loads(json_path.read_text())
    assert meta["condition"] == "A_rare"
    assert meta["id_range"] == [1000, 39999]
    assert meta["n"] == 50
    assert meta["seq_len"] == 32
    assert meta["base_seed"] == 0
    assert meta["seed"] == derive_seed(0, "A_rare")
    assert meta["without_replacement"] is True
    assert meta["sha256"] == sequences_hash(seqs)


def test_load_or_create_second_call_returns_identical_array_and_does_not_rewrite(tmp_path):
    first = load_or_create_sequences(**_kwargs(tmp_path))
    npy_path = tmp_path / "A_rare.npy"
    json_path = tmp_path / "A_rare.json"
    npy_mtime = npy_path.stat().st_mtime_ns
    json_mtime = json_path.stat().st_mtime_ns

    second = load_or_create_sequences(**_kwargs(tmp_path))

    np.testing.assert_array_equal(first, second)
    assert npy_path.stat().st_mtime_ns == npy_mtime
    assert json_path.stat().st_mtime_ns == json_mtime


def test_load_or_create_mismatched_n_raises(tmp_path):
    load_or_create_sequences(**_kwargs(tmp_path))
    with pytest.raises(SequenceCacheMismatch):
        load_or_create_sequences(**_kwargs(tmp_path, n=51))


def test_load_or_create_mismatched_seed_raises(tmp_path):
    load_or_create_sequences(**_kwargs(tmp_path))
    with pytest.raises(SequenceCacheMismatch):
        load_or_create_sequences(**_kwargs(tmp_path, base_seed=1))


def test_load_or_create_tampered_npy_raises(tmp_path):
    load_or_create_sequences(**_kwargs(tmp_path))
    npy_path = tmp_path / "A_rare.npy"
    arr = np.load(npy_path).copy()
    # Range is [1000, 39999] (39000 values) but seq_len is 32, so there's always an
    # unused in-range value to swap in without creating a duplicate within the row.
    used = set(arr[0].tolist())
    replacement = next(v for v in range(1000, 40000) if v not in used)
    arr[0, 0] = replacement
    np.save(npy_path, arr)
    with pytest.raises(SequenceCacheMismatch):
        load_or_create_sequences(**_kwargs(tmp_path))


def test_load_or_create_only_npy_present_raises(tmp_path):
    load_or_create_sequences(**_kwargs(tmp_path))
    (tmp_path / "A_rare.json").unlink()
    with pytest.raises(SequenceCacheMismatch):
        load_or_create_sequences(**_kwargs(tmp_path))


def test_load_or_create_only_json_present_raises(tmp_path):
    load_or_create_sequences(**_kwargs(tmp_path))
    (tmp_path / "A_rare.npy").unlink()
    with pytest.raises(SequenceCacheMismatch):
        load_or_create_sequences(**_kwargs(tmp_path))


# --- prepend_bos ---------------------------------------------------------------


def test_prepend_bos_shape_column_and_dtype():
    seqs = sample_sequences((0, 999), n=10, seq_len=32, seed=7)
    out = prepend_bos(seqs, bos_id=50256)
    assert out.shape == (10, 33)
    assert out.dtype == np.int64
    assert np.all(out[:, 0] == 50256)
    np.testing.assert_array_equal(out[:, 1:], seqs)
