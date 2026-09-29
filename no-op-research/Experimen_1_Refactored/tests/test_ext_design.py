"""Tests for exp1extended.tokens and exp1extended.design."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy.stats import chisquare
from transformers import AutoTokenizer

from exp1.sequences import SequenceCacheMismatch, derive_seed, sequences_hash
from exp1extended.design import (
    SEGMENTS, _matched_set, build_sweep, cell_name, check_design, dose_masks, dose_sequences, load_or_create_array,
    matched_target_probs, matching_setup, paired_triples, recombine, sample_matched, sample_pool, sweep_tokens,
)
from exp1extended.tokens import FEATURES, balance_table, display_text, stratum_codes, token_features

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "exp1extended.yaml"
RARE, COMMON = (1000, 39999), (256, 999)
LEVELS = [0, 1, 2, 4, 8, 15, 22, 30]


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("gpt2")


@pytest.fixture(scope="module")
def cfg():
    return yaml.safe_load(CONFIG.read_text())


@pytest.fixture(scope="module")
def match(cfg, tokenizer):
    return matching_setup(cfg, tokenizer)


class FakeTokenizer:
    """Maps ids to fixed strings, so feature rules can be tested on hand-picked text."""

    def __init__(self, texts: dict[int, str]):
        self.texts = texts

    def decode(self, ids, **kwargs):
        return "".join(self.texts[i] for i in ids)


def _no_row_repeats(seqs: np.ndarray) -> bool:
    return all(len(set(row.tolist())) == len(row) for row in seqs)


def _triples(n: int, seed: int) -> dict[str, np.ndarray]:
    return {"r": sample_pool(RARE, n, 32, seed), "m": sample_pool(RARE, n, 32, seed + 1),
            "c": sample_pool(COMMON, n, 32, seed + 2)}


# ---- tokens.token_features --------------------------------------------------------------------

def test_token_features_known_ids(tokenizer):
    df = token_features(tokenizer, np.array([262, 13, 220]), [1, 2, 3, 4, 5]).set_index("id")
    assert list(df.columns) == list(FEATURES)

    the = df.loc[262]  # ' the'
    assert the["word_start"] and the["char_len"] == 3 and the["len_bin"] == 3
    assert the["is_alpha"] and not the["is_cap"] and not the["is_digit"] and not the["is_byte"]

    dot = df.loc[13]  # '.'
    assert not dot["word_start"] and not dot["is_alpha"] and dot["char_len"] == 1 and dot["is_byte"]

    space = df.loc[220]  # ' '
    assert space["word_start"] and space["char_len"] == 0 and space["len_bin"] == 1

    np.testing.assert_allclose(df["log_id"], np.log10(np.array([263, 14, 221])))
    assert df["embed_norm"].isna().all()


def test_token_features_rules_on_synthetic_text():
    texts = {0: "", 1: "a", 2: " Abcd", 3: " abcde", 4: "abcdefghijkl", 5: " 1999", 6: "é", 7: " Ĳ", 8: " ."}
    ids = np.array(sorted(texts))
    df = token_features(FakeTokenizer(texts), ids, [1, 2, 3, 4, 5]).set_index("id")
    assert df["char_len"].tolist() == [0, 1, 4, 5, 12, 4, 1, 1, 1]
    assert df["len_bin"].tolist() == [1, 1, 4, 5, 5, 4, 1, 1, 1]
    assert df["word_start"].tolist() == [False, False, True, True, False, True, False, True, True]
    assert df["is_alpha"].tolist() == [False, True, True, True, True, False, False, False, False]  # ASCII only
    assert df["is_cap"].tolist() == [False, False, True, False, False, False, False, True, False]
    assert df["is_digit"].tolist() == [False, False, False, False, False, True, False, False, False]


def test_token_features_embed_norm():
    embed = np.random.default_rng(0).normal(size=(10, 4)).astype(np.float32)
    ids = np.array([3, 7])
    df = token_features(FakeTokenizer({3: "x", 7: " y"}), ids, [1, 2], embed=embed)
    np.testing.assert_allclose(df["embed_norm"], np.linalg.norm(embed[ids].astype(np.float64), axis=1))


def test_display_text(tokenizer):
    assert display_text(tokenizer, 262) == "the"
    assert display_text(tokenizer, 198) == "Ċ"  # newline has no printable text: BPE symbol


# ---- tokens.stratum_codes / balance_table ---------------------------------------------------------

def test_stratum_codes_stable_across_frames():
    a = pd.DataFrame({"word_start": [True, False, True], "len_bin": [5, 1, 5], "is_alpha": [True, True, False]})
    b = a.iloc[::-1].reset_index(drop=True)
    cols = ["word_start", "len_bin", "is_alpha"]
    codes_a, codes_b = stratum_codes(a, cols), stratum_codes(b, cols)
    np.testing.assert_array_equal(codes_a, codes_b[::-1])
    assert codes_a.tolist() == [1_005_001, 1_001, 1_005_000]
    assert len(set(codes_a.tolist())) == 3


def test_stratum_codes_rejects_non_integer_column():
    with pytest.raises(ValueError):
        stratum_codes(pd.DataFrame({"log_id": [0.5, 1.5]}), ["log_id"])


def test_balance_table_values():
    a = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "same": [1, 1, 1, 1], "const": [True] * 4})
    b = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "same": [1, 1, 1, 1], "const": [False] * 4})
    t = balance_table(a, b, ["x", "same", "const"]).set_index("feature")
    sd = np.std([0, 1, 2, 3], ddof=1)
    assert t.loc["x", "smd"] == pytest.approx(-1.0 / sd)
    assert t.loc["same", "smd"] == 0.0
    assert t.loc["const", "smd"] == np.inf  # constant in each group but different: worst imbalance


# ---- matched sampling ---------------------------------------------------------------------------

def test_matched_target_probs_drops_and_renormalizes():
    probs, dropped = matched_target_probs(np.array([1, 2, 2]), np.array([1, 1, 2, 3]))
    assert dropped == pytest.approx(0.25)
    assert probs == pytest.approx({1: 2 / 3, 2: 1 / 3})


def test_sample_matched_hits_target_proportions_synthetic():
    # stratum 2 mimics the real tiny stratum: 7 tokens at p = 0.06.
    pool_ids = np.arange(1000, 1257)
    pool_strata = np.repeat([0, 1, 2], [50, 200, 7])
    target = {0: 0.54, 1: 0.40, 2: 0.06}
    seqs = sample_matched(pool_ids, pool_strata, target, n=625, seq_len=32, seed=0)  # 20k draws

    assert seqs.shape == (625, 32) and seqs.dtype == np.int64
    assert _no_row_repeats(seqs)
    strata = pool_strata[seqs.ravel() - 1000]
    observed = np.bincount(strata, minlength=3) / strata.size
    np.testing.assert_allclose(observed, [0.54, 0.40, 0.06], atol=0.01)
    # Positions are exchangeable: the last position (Q) alone is also on target.
    q_strata = pool_strata[seqs[:, -1] - 1000]
    np.testing.assert_allclose(np.bincount(q_strata, minlength=3) / 625, [0.54, 0.40, 0.06], atol=0.05)
    # Uniform within a stratum.
    counts = np.bincount(seqs.ravel() - 1000, minlength=257)[:50]
    assert chisquare(counts).pvalue > 0.001


def test_sample_matched_is_deterministic_and_raises_on_empty_stratum():
    pool_ids, pool_strata = np.arange(100, 200), np.repeat([0, 1], 50)
    a = sample_matched(pool_ids, pool_strata, {0: 0.5, 1: 0.5}, 20, 16, seed=3)
    b = sample_matched(pool_ids, pool_strata, {0: 0.5, 1: 0.5}, 20, 16, seed=3)
    np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError):
        sample_matched(pool_ids, pool_strata, {0: 0.5, 9: 0.5}, 20, 16, seed=3)


def test_sample_matched_real_pools_hit_common_strata(cfg, tokenizer, match):
    probs = match["target_probs"]
    seqs = sample_matched(match["pool_ids"], match["pool_strata"], probs, n=625, seq_len=32, seed=1)
    assert _no_row_repeats(seqs)
    assert seqs.min() >= RARE[0] and seqs.max() <= RARE[1]

    strata = match["pool_strata"][seqs.ravel() - RARE[0]]
    codes = np.array(sorted(probs))
    observed = np.array([(strata == c).sum() for c in codes])
    expected = np.array([probs[int(c)] for c in codes]) * strata.size
    assert np.isin(strata, codes).all()
    np.testing.assert_allclose(observed / strata.size, expected / strata.size, atol=0.01)
    assert chisquare(observed, expected).pvalue > 0.001


def test_matched_balance_real_pools(cfg, tokenizer, match):
    """V5-style balance of a Stage 1-sized matched rare sample against the common pool."""
    assert match["dropped_mass"] == 0.0  # every common-pool stratum occurs in the rare pool
    seqs = sample_matched(match["pool_ids"], match["pool_strata"], match["target_probs"], 1000, 32, seed=2)
    feats = token_features(tokenizer, seqs.ravel(), cfg["matching"]["len_bins"])
    common = match["target_features"]
    cols = ["word_start", "char_len", "len_bin", "is_alpha"]
    table = balance_table(feats, common, cols).set_index("feature")
    uniform = token_features(tokenizer, sample_pool(RARE, 1000, 32, 2).ravel(), cfg["matching"]["len_bins"])
    print("\nmatched rare vs common:\n", table.to_string())
    print("uniform rare vs common:\n", balance_table(uniform, common, cols).to_string())

    for col in ("word_start", "len_bin", "is_alpha"):
        assert abs(table.loc[col, "smd"]) <= cfg["validation"]["max_smd"], col
    # char_len is only matched up to the open 5+ bin; below it the match is exact.
    short = balance_table(feats[feats["len_bin"] < 5], common[common["len_bin"] < 5], ["char_len"])
    assert abs(short["smd"].iloc[0]) <= cfg["validation"]["max_smd"]


# ---- paired triples ----------------------------------------------------------------------------

def test_paired_triples(cfg, tokenizer):
    small = copy.deepcopy(cfg)
    small["stage1"]["n_pairs"] = 200
    triples = paired_triples(small, tokenizer)
    assert set(triples) == {"r", "m", "c"}
    for member, pool in (("r", RARE), ("m", RARE), ("c", COMMON)):
        seqs = triples[member]
        assert seqs.shape == (200, 32) and seqs.dtype == np.int64
        assert seqs.min() >= pool[0] and seqs.max() <= pool[1]
        assert _no_row_repeats(seqs)
    np.testing.assert_array_equal(triples["r"], sample_pool(RARE, 200, 32, derive_seed(0, "exp1extended/stage1/r")))
    np.testing.assert_array_equal(triples["c"], sample_pool(COMMON, 200, 32, derive_seed(0, "exp1extended/stage1/c")))
    again = paired_triples(small, tokenizer)
    for member in triples:
        np.testing.assert_array_equal(triples[member], again[member])


# ---- recombine ---------------------------------------------------------------------------------

def test_recombine_copies_segments():
    triples = _triples(50, seed=10)
    spec = {"Q": "c", "P": "r", "X": "c"}
    cell = recombine(triples, spec)
    np.testing.assert_array_equal(cell[:, :30], triples["c"][:, :30])
    np.testing.assert_array_equal(cell[:, 30], triples["r"][:, 30])
    np.testing.assert_array_equal(cell[:, 31], triples["c"][:, 31])
    assert cell_name(spec) == "Qc_Pr_Xc"


def test_configured_cells_have_no_repeats(cfg):
    triples = _triples(300, seed=11)
    names = set()
    for spec in cfg["stage1"]["cells"]:
        cell = recombine(triples, spec)
        assert _no_row_repeats(cell)
        pools = {seg: (RARE if spec[seg] in "rm" else COMMON) for seg in SEGMENTS}
        check_design(cell, pools)
        names.add(cell_name(spec))
    assert len(names) == len(cfg["stage1"]["cells"])
    np.testing.assert_array_equal(recombine(triples, {"Q": "r", "P": "r", "X": "r"}), triples["r"])


def test_recombine_raises_on_r_m_collision():
    triples = _triples(5, seed=12)
    triples["m"][3, 31] = triples["r"][3, 7]  # m's query token also sits in r's context
    with pytest.raises(ValueError, match="row 3"):
        recombine(triples, {"Q": "m", "P": "r", "X": "r"})


def test_recombine_rejects_bad_spec():
    triples = _triples(5, seed=13)
    with pytest.raises(ValueError):
        recombine(triples, {"Q": "r", "P": "r"})
    with pytest.raises(ValueError):
        recombine(triples, {"Q": "z", "P": "r", "X": "r"})


# ---- dose --------------------------------------------------------------------------------------

def test_dose_masks_nested_with_exact_counts():
    masks = dose_masks(500, LEVELS, seed=4)
    assert masks.shape == (len(LEVELS), 500, 30) and masks.dtype == bool
    for level, mask in zip(LEVELS, masks):
        assert np.all(mask.sum(axis=1) == level)
    for lo, hi in zip(masks[:-1], masks[1:]):
        assert np.all(~lo | hi)  # lo implies hi
    np.testing.assert_array_equal(masks, dose_masks(500, LEVELS, seed=4))
    # Every position is equally likely to be switched: at k = 15 each is on in about half the rows.
    np.testing.assert_allclose(masks[LEVELS.index(15)].mean(axis=0), 0.5, atol=0.1)
    with pytest.raises(ValueError):
        dose_masks(5, [31], seed=0)


def test_dose_sequences():
    triples = _triples(40, seed=14)
    r, c = triples["r"], triples["c"]
    masks = dose_masks(40, LEVELS, seed=5)
    np.testing.assert_array_equal(dose_sequences(r, c, masks[0], r), r)  # k = 0 is the all-rare cell
    full = dose_sequences(r, c, masks[-1], r)
    np.testing.assert_array_equal(full, recombine(triples, {"Q": "r", "P": "r", "X": "c"}))
    for level_mask, qp in ((masks[3], r), (masks[5], c)):
        seqs = dose_sequences(r, c, level_mask, qp)
        np.testing.assert_array_equal(seqs[:, :30][level_mask], c[:, :30][level_mask])
        np.testing.assert_array_equal(seqs[:, :30][~level_mask], r[:, :30][~level_mask])
        np.testing.assert_array_equal(seqs[:, 30:], qp[:, 30:])
        assert _no_row_repeats(seqs)
        check_design(seqs, {"X": [RARE, COMMON], "P": RARE if qp is r else COMMON, "Q": RARE if qp is r else COMMON})


# ---- Stage 2 sweep -----------------------------------------------------------------------------

def test_build_sweep_hand_example():
    backgrounds = np.array([[10, 11, 12, 13], [20, 21, 22, 23]], dtype=np.int64)
    token_ids = np.array([11, 99, 23, 20])
    seqs, tok_idx, bg_idx, missing = build_sweep(backgrounds, 3, token_ids)

    # 11 already sits in background 0; 20 in background 1. 23 is background 1's own column-3 token: valid.
    np.testing.assert_array_equal(missing, [[True, False], [False, False], [False, False], [False, True]])
    np.testing.assert_array_equal(tok_idx, [0, 1, 1, 2, 2, 3])
    np.testing.assert_array_equal(bg_idx, [1, 0, 1, 0, 1, 0])
    np.testing.assert_array_equal(seqs, [[20, 21, 22, 11], [10, 11, 12, 99], [20, 21, 22, 99],
                                         [10, 11, 12, 23], [20, 21, 22, 23], [10, 11, 12, 20]])


def test_build_sweep_random_backgrounds_never_repeat():
    backgrounds = np.concatenate([sample_pool(RARE, 8, 32, 20), sample_pool(COMMON, 8, 32, 21)])
    token_ids = np.arange(0, 1200)
    seqs, tok_idx, bg_idx, missing = build_sweep(backgrounds, 31, token_ids)
    assert _no_row_repeats(seqs)
    assert len(seqs) == (~missing).sum()
    expected_missing = np.array([[t in set(bg[:31].tolist()) for bg in backgrounds] for t in token_ids])
    np.testing.assert_array_equal(missing, expected_missing)
    np.testing.assert_array_equal(seqs[:, 31], token_ids[tok_idx])
    np.testing.assert_array_equal(seqs[:, :31], backgrounds[bg_idx, :31])


def test_sweep_tokens_groups(cfg, tokenizer):
    df = sweep_tokens(cfg, tokenizer)
    assert list(df.columns) == ["id", "group"]
    assert df["id"].is_unique
    sizes = df.groupby("group").size().to_dict()
    assert sizes == {"byte": 256, "common": 744, "rare_uniform": 400, "rare_matched": 400, "rare_transition": 200}
    bounds = {"byte": (0, 255), "common": (256, 999), "rare_uniform": RARE, "rare_matched": RARE,
              "rare_transition": (1000, 2999)}
    for group, (lo, hi) in bounds.items():
        ids = df.loc[df["group"] == group, "id"]
        assert ids.min() >= lo and ids.max() <= hi, group
    assert df.attrs["rare_matched_dropped_mass"] == 0.0
    pd.testing.assert_frame_equal(df, sweep_tokens(cfg, tokenizer))
    print("\nsweep rare_matched:", df.attrs)


def test_matched_set_caps_tiny_stratum():
    pool_ids = np.arange(0, 110)
    pool_strata = np.repeat([0, 1, 2], [3, 57, 50])
    ids, tv, capped = _matched_set(pool_ids, pool_strata, {0: 0.2, 1: 0.4, 2: 0.4}, n=50, seed=0)
    assert len(ids) == 50 and len(set(ids.tolist())) == 50
    counts = np.bincount(pool_strata[ids], minlength=3)
    assert counts[0] == 3  # target 10, only 3 exist: take them all
    assert counts.tolist() == [3, 24, 23] or counts.tolist() == [3, 23, 24]
    assert capped == [0]
    assert tv == pytest.approx(0.5 * (abs(3 / 50 - 0.2) + abs(counts[1] / 50 - 0.4) + abs(counts[2] / 50 - 0.4)))


# ---- cache -------------------------------------------------------------------------------------

def _files(tmp_path, name="cells"):
    return tmp_path / f"{name}.npy", tmp_path / f"{name}.json"


def test_cache_writes_then_reloads(tmp_path):
    arr = sample_pool(RARE, 20, 32, 30)
    meta = {"stage": "stage1", "id_range": (1000, 39999), "seed": np.int64(5)}
    out = load_or_create_array("cells", arr, tmp_path, meta)
    np.testing.assert_array_equal(out, arr)
    npy, js = _files(tmp_path)
    stored = json.loads(js.read_text())
    assert stored["sha256"] == sequences_hash(arr) and stored["id_range"] == [1000, 39999]
    assert stored["dtype"] == "int64" and stored["shape"] == [20, 32]

    mtimes = (npy.stat().st_mtime_ns, js.stat().st_mtime_ns)
    again = load_or_create_array("cells", arr.copy(), tmp_path, meta)
    np.testing.assert_array_equal(again, arr)
    assert (npy.stat().st_mtime_ns, js.stat().st_mtime_ns) == mtimes


def test_cache_bool_masks_round_trip(tmp_path):
    masks = dose_masks(10, LEVELS, seed=1)
    load_or_create_array("masks", masks, tmp_path, {"levels": LEVELS})
    back = load_or_create_array("masks", masks, tmp_path, {"levels": LEVELS})
    assert back.dtype == bool
    np.testing.assert_array_equal(back, masks)


@pytest.mark.parametrize("change", ["meta", "array", "extra_meta_key"])
def test_cache_mismatch_raises_and_leaves_files(tmp_path, change):
    arr = sample_pool(RARE, 20, 32, 31)
    meta = {"stage": "stage1", "n": 20}
    load_or_create_array("cells", arr, tmp_path, meta)
    npy, js = _files(tmp_path)
    before = (npy.read_bytes(), js.read_bytes(), npy.stat().st_mtime_ns, js.stat().st_mtime_ns)

    new_arr, new_meta = arr, meta
    if change == "meta":
        new_meta = dict(meta, n=21)
    elif change == "array":
        new_arr = arr.copy()
        new_arr[0, 0] = next(v for v in range(*RARE) if v not in set(arr[0].tolist()))
    else:
        new_meta = dict(meta, extra=1)
    with pytest.raises(SequenceCacheMismatch):
        load_or_create_array("cells", new_arr, tmp_path, new_meta)
    assert (npy.read_bytes(), js.read_bytes(), npy.stat().st_mtime_ns, js.stat().st_mtime_ns) == before


def test_cache_tampered_or_incomplete_raises(tmp_path):
    arr = sample_pool(RARE, 20, 32, 32)
    load_or_create_array("cells", arr, tmp_path, {})
    npy, js = _files(tmp_path)
    tampered = arr.copy()
    tampered[1, 2] = next(v for v in range(*RARE) if v not in set(arr[1].tolist()))
    np.save(npy, tampered)
    with pytest.raises(SequenceCacheMismatch, match="file integrity"):
        load_or_create_array("cells", arr, tmp_path, {})
    js.unlink()
    with pytest.raises(SequenceCacheMismatch, match="Incomplete"):
        load_or_create_array("cells", arr, tmp_path, {})
    assert not js.exists()


def test_cache_rejects_float_arrays_and_reserved_keys(tmp_path):
    with pytest.raises(ValueError):
        load_or_create_array("f", np.zeros((2, 2)), tmp_path, {})
    with pytest.raises(ValueError):
        load_or_create_array("i", np.zeros((2, 2), dtype=np.int64), tmp_path, {"sha256": "x"})
    assert not any(tmp_path.iterdir())


# ---- check_design ------------------------------------------------------------------------------

def test_check_design():
    triples = _triples(10, seed=40)
    check_design(triples["r"], {"Q": RARE, "P": RARE, "X": RARE})
    with pytest.raises(AssertionError):
        check_design(triples["r"], {"Q": COMMON})
    bad = triples["r"].copy()
    bad[2, 5] = bad[2, 9]
    with pytest.raises(AssertionError):
        check_design(bad, {"X": RARE})
    with pytest.raises(AssertionError):
        check_design(triples["r"][:, :31], {})
