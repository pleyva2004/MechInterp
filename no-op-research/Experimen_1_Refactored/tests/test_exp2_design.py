"""Tests for exp2.design."""
from __future__ import annotations

import numpy as np
import pytest
from transformers import AutoTokenizer

from exp1ext.features import token_features
from exp2.design import (
    build_variants,
    cached_base_block,
    check_variants,
    draw_matched,
    id_pool,
    matched_pools,
    positions_to_slots,
    token_class_of,
)

RARE, COMMON = [1000, 39999], [256, 999]
PHRASE = [314, 588, 36324]
SHUFFLED = [588, 314, 36324]
BOS = 50256
SEQ_LEN = 32


def no_repeats(seqs):
    return all(len(set(row)) == len(row) for row in seqs.tolist())


def make_pools():
    return [np.arange(2000, 2300), np.arange(5000, 5300), np.arange(9000, 9300)]


def make_bases(n=20):
    return cached_base_block("rare", RARE, PHRASE, n, SEQ_LEN, 0, BOS, make_bases.dir)


@pytest.fixture
def bases(tmp_path):
    make_bases.dir = tmp_path
    return make_bases(20)


# ---- id_pool / cached_base_block ----------------------------------------

def test_id_pool_excludes_and_keeps_rest():
    pool = id_pool([10, 20], [12, 15, 99])
    assert pool.tolist() == [10, 11, 13, 14, 16, 17, 18, 19, 20]


def test_cached_base_block_properties_and_reload(tmp_path):
    a = cached_base_block("rare", RARE, PHRASE, 30, SEQ_LEN, 0, BOS, tmp_path)
    assert a.shape == (30, SEQ_LEN)
    assert no_repeats(a)
    assert not np.isin(a, PHRASE).any()
    assert ((a >= RARE[0]) & (a <= RARE[1])).all()
    files = sorted(p.name for p in tmp_path.iterdir())
    b = cached_base_block("rare", RARE, PHRASE, 30, SEQ_LEN, 0, BOS, tmp_path)
    assert (a == b).all()
    assert sorted(p.name for p in tmp_path.iterdir()) == files


def test_cached_base_block_different_exclude_uses_different_file(tmp_path):
    cached_base_block("rare", RARE, PHRASE, 10, SEQ_LEN, 0, BOS, tmp_path)
    n_files = len(list(tmp_path.glob("*.npy")))
    other = cached_base_block("rare", RARE, [1234, 5678], 10, SEQ_LEN, 0, BOS, tmp_path)
    assert len(list(tmp_path.glob("*.npy"))) == n_files + 1
    assert not np.isin(other, [1234, 5678]).any()


# ---- positions_to_slots --------------------------------------------------

def test_positions_to_slots_maps_to_array_indices():
    assert positions_to_slots([29, 30, 31], SEQ_LEN).tolist() == [28, 29, 30]
    assert positions_to_slots([1], SEQ_LEN).tolist() == [0]


@pytest.mark.parametrize("positions", [[0, 1, 2], [30, 31, 32], [5, 6, 5]])
def test_positions_to_slots_rejects_bad(positions):
    with pytest.raises(ValueError):
        positions_to_slots(positions, SEQ_LEN)


# ---- draw_matched --------------------------------------------------------

def test_draw_matched_in_pool_and_no_repeats(bases):
    slots = positions_to_slots([29, 30, 31], SEQ_LEN)
    pools = make_pools()
    # make repeats likely: pools overlapping the base ids' range
    drawn = draw_matched(bases, slots, pools, seed=3)
    assert drawn.shape == (len(bases), 3)
    for j, pool in enumerate(pools):
        assert np.isin(drawn[:, j], pool).all()
    keep = np.ones(SEQ_LEN, dtype=bool)
    keep[slots] = False
    for i in range(len(bases)):
        row = np.concatenate([bases[i, keep], drawn[i]])
        assert len(set(row.tolist())) == len(row)


def test_draw_matched_avoids_base_ids_when_pool_is_small():
    base = np.array([[1, 2, 3, 4, 5, 6]])
    slots = np.array([0, 1])
    pools = [np.array([3, 4, 7]), np.array([5, 6, 8])]
    for seed in range(20):
        d = draw_matched(base, slots, pools, seed)
        assert d.tolist() == [[7, 8]]


def test_draw_matched_deterministic_and_row_independent(bases):
    slots = positions_to_slots([29, 30, 31], SEQ_LEN)
    pools = make_pools()
    a = draw_matched(bases, slots, pools, 7)
    assert (a == draw_matched(bases, slots, pools, 7)).all()
    assert (draw_matched(bases[:5], slots, pools, 7) == draw_matched(bases[:20], slots, pools, 7)[:5]).all()
    assert not (a == draw_matched(bases, slots, pools, 8)).all()


def test_draw_matched_raises_when_no_candidate():
    base = np.array([[1, 2, 3]])
    with pytest.raises(RuntimeError):
        draw_matched(base, np.array([0]), [np.array([2, 3])], 0)


# ---- matched_pools -------------------------------------------------------

@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("gpt2")


def test_token_class_of():
    classes = {"common": [256, 999], "rare": [1000, 39999]}
    assert token_class_of(314, classes) == "common"
    assert token_class_of(1000, classes) == "rare"
    with pytest.raises(ValueError):
        token_class_of(100, classes)
    with pytest.raises(ValueError):
        token_class_of(50000, classes)


def test_matched_pools_share_features_class_and_exclude_phrase(tokenizer):
    feats = token_features(np.arange(256, 40000), tokenizer)
    match = ["leading_space", "is_alpha", "is_capitalized"]
    classes = {"common": [256, 999], "rare": [1000, 39999]}
    pools = matched_pools(PHRASE, feats, match, classes)
    assert len(pools) == 3
    f = feats.set_index("id")
    for tok, pool in zip(PHRASE, pools):
        lo, hi = classes[token_class_of(tok, classes)]
        assert len(pool) > 0
        assert ((pool >= lo) & (pool <= hi)).all()
        assert not np.isin(pool, PHRASE).any()
        for col in match:
            assert (f.loc[pool, col].to_numpy() == f.at[tok, col]).all()


def test_matched_pools_raises_when_empty(tokenizer):
    feats = token_features(np.arange(256, 1000), tokenizer)
    # a class range holding only the phrase id itself leaves nothing
    with pytest.raises(ValueError):
        matched_pools([314], feats, ["leading_space"], {"only": [314, 314]})


# ---- build_variants / check_variants ------------------------------------

def build(bases, positions):
    slots = positions_to_slots(positions, SEQ_LEN)
    pools = make_pools()
    variants = build_variants(bases, slots, PHRASE, SHUFFLED, pools, seed=5)
    return slots, pools, variants


@pytest.mark.parametrize("positions", [[29, 30, 31], [14, 15, 16]])
def test_build_variants_pass_check_and_structure(bases, positions):
    slots, pools, v = build(bases, positions)
    assert set(v) == {"none", "nl", "shuffled", "matched", "matched_b"}
    check_variants(v, bases, slots, PHRASE, SHUFFLED, pools, BOS)
    keep = np.ones(SEQ_LEN, dtype=bool)
    keep[slots] = False
    for name, seqs in v.items():
        assert (seqs[:, keep] == bases[:, keep]).all(), name
    assert (v["none"] == bases).all() and v["none"] is not bases
    assert (v["nl"][:, slots] == PHRASE).all()
    assert (v["shuffled"][:, slots] == SHUFFLED).all()
    assert not (v["matched"][:, slots] == v["matched_b"][:, slots]).all()
    # bases untouched
    assert not np.isin(bases[:, slots], PHRASE).all()


def test_build_variants_rejects_non_permutation(bases):
    slots = positions_to_slots([29, 30, 31], SEQ_LEN)
    with pytest.raises(ValueError):
        build_variants(bases, slots, PHRASE, [588, 314, 1], make_pools(), 0)
    with pytest.raises(ValueError):
        build_variants(bases, slots, PHRASE, [588, 314], make_pools(), 0)


def tampered(bases, positions, edit):
    slots, pools, v = build(bases, positions)
    b = bases.copy()
    edit(v, b, slots)
    return v, b, slots, pools


def check_raises(v, b, slots, pools):
    with pytest.raises(AssertionError):
        check_variants(v, b, slots, PHRASE, SHUFFLED, pools, BOS)


def test_check_variants_flags_row_repeat(bases):
    def edit(v, b, slots):
        v["nl"][0, 0] = v["nl"][0, 5]
    check_raises(*tampered(bases, [29, 30, 31], edit))


def test_check_variants_flags_non_slot_change(bases):
    def edit(v, b, slots):
        v["shuffled"][2, 0] = 39000
    check_raises(*tampered(bases, [29, 30, 31], edit))


def test_check_variants_flags_phrase_id_in_base_outside_slots(bases):
    def edit(v, b, slots):
        for s in v.values():
            s[1, 0] = 314
        b[1, 0] = 314
    check_raises(*tampered(bases, [29, 30, 31], edit))


def test_check_variants_flags_out_of_pool_matched_id(bases):
    def edit(v, b, slots):
        v["matched"][3, slots[1]] = 39999
    check_raises(*tampered(bases, [14, 15, 16], edit))


def test_check_variants_flags_bos_and_wrong_dtype(bases):
    def edit(v, b, slots):
        v["none"] = v["none"].astype(np.int32)
    check_raises(*tampered(bases, [14, 15, 16], edit))
