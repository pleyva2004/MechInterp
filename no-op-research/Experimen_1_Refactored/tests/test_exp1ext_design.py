import numpy as np
import pytest

from exp1.sequences import sample_sequences
from exp1ext.design import build_dose_response, build_factorial, cached_block, check_design, position_sweep

RARE, COMMON = [1000, 39999], [256, 999]


def blocks(n=50, seq_len=32):
    return sample_sequences(RARE, n, seq_len, 1), sample_sequences(COMMON, n, seq_len, 2)


def no_repeats(seqs):
    return all(len(set(row)) == len(row) for row in seqs.tolist())


def test_factorial_cells_classes_and_pairing():
    rare, common = blocks()
    cells = build_factorial(rare, common)
    in_range = lambda x, r: ((x >= r[0]) & (x <= r[1])).all()
    for name, (ctx, fin) in {"AA": (RARE, RARE), "AC": (RARE, COMMON), "CA": (COMMON, RARE), "CC": (COMMON, COMMON)}.items():
        assert in_range(cells[name][:, :-1], ctx) and in_range(cells[name][:, -1], fin), name
        assert no_repeats(cells[name]), name
    # index-aligned: AA and AC share the context, AA and CA share the final token
    assert (cells["AA"][:, :-1] == cells["AC"][:, :-1]).all()
    assert (cells["AA"][:, -1] == cells["CA"][:, -1]).all()
    assert (cells["CC"][:, :-1] == cells["CA"][:, :-1]).all()


def test_dose_response_nested_counts_and_final_token_kept():
    rare, common = blocks()
    repl = sample_sequences(COMMON, len(rare), 31, 3)
    ks = [0, 1, 2, 4, 8, 16, 31]
    dose = build_dose_response(rare, repl, ks, seed=0)
    for k in ks:
        n_common = ((dose[k] >= COMMON[0]) & (dose[k] <= COMMON[1])).sum(axis=1)
        assert (n_common == k).all()
        assert (dose[k][:, -1] == rare[:, -1]).all()
        assert no_repeats(dose[k])
    assert (dose[0] == rare).all()
    # nested: every position replaced at k=4 is also replaced (identically) at k=8
    changed4 = dose[4] != rare
    assert (dose[8][changed4] == dose[4][changed4]).all()
    assert (build_dose_response(rare, repl, ks, seed=0)[8] == dose[8]).all()


def test_position_sweep_changes_only_that_position_and_flags_repeats():
    rare, _ = blocks(n=3)
    tokens = np.array([5, int(rare[0, 3]), 7000, int(rare[1, -1])])
    seqs, invalid = position_sweep(rare, tokens, index=31)
    assert seqs.shape == (3 * 4, 32)
    for k in range(3):
        for j, t in enumerate(tokens):
            row = seqs[k * 4 + j]
            assert row[31] == t
            assert (np.delete(row, 31) == np.delete(rare[k], 31)).all()
    # token rare[0, 3] already sits at index 3 of base 0 -> invalid there only
    assert invalid[0, 1] and not invalid[1, 1]
    # putting base 1's own final token back at the final position is not a repeat
    assert not invalid[1, 3]


def test_position_sweep_rejects_bad_index():
    rare, _ = blocks(n=2)
    with pytest.raises(ValueError):
        position_sweep(rare, np.array([1, 2]), index=32)


def test_check_design_flags_out_of_range_and_bos():
    rare, _ = blocks(n=4)
    check_design(rare, [RARE], bos_id=50256)
    bad = rare.copy()
    bad[0, 0] = 50
    with pytest.raises(AssertionError):
        check_design(bad, [RARE], bos_id=50256)
    with pytest.raises(AssertionError):
        check_design(np.full((2, 3), 50256, dtype=np.int64), [[0, 50256]], bos_id=50256)


def test_cached_block_reuses_file(tmp_path):
    a = cached_block("blk", RARE, 5, 32, 0, 50256, tmp_path)
    b = cached_block("blk", RARE, 5, 32, 0, 50256, tmp_path)
    assert (a == b).all() and (tmp_path / "blk.npy").exists()
