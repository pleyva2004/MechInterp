"""Tests for exp3.e34: the E3.4 builder, its layout checks, the fallback, and the analysis on synthetic attention."""
from __future__ import annotations

import numpy as np
import pytest

from exp3 import e34
from exp3 import engine as eng

BOS_ID = 0
LAYOUT = {"phrase": [10, 11, 12], "previous": 31, "query": 32, "dup": 11, "copy": 12}
FB_LAYOUT = {"block": [2, 16], "repeat": [19, 31], "query": 32, "dup": 15, "copy": 16}
PHRASES = {"p0": [1, 2, 3], "p1": [4, 5, 6], "p2": [1, 7, 8]}  # p0 and p2 share id 1
OTHER = {"p0": "p1", "p1": "p2", "p2": "p1"}
POOL = np.arange(1000, 3000, dtype=np.int64)


def _base(rows: int = 6) -> np.ndarray:
    rng = np.random.default_rng(0)
    return np.stack([rng.choice(POOL, 32, replace=False) for _ in range(rows)]).astype(np.int64)


def _matched() -> dict[str, list[np.ndarray]]:
    return {n: [np.arange(5000 + 100 * (3 * k + j), 5000 + 100 * (3 * k + j) + 40) for j in range(3)]
            for k, n in enumerate(PHRASES)}


def _build(rows: int = 6, seed: int = 11) -> eng.Design:
    return e34.build_e34(_base(rows), PHRASES, OTHER, _matched(), POOL, seed, LAYOUT)


# ---- builder -------------------------------------------------------------------------------------------------

def test_build_passes_generic_and_layout_checks():
    d = _build()
    assert d.variants == e34.VARIANTS and d.units == tuple(PHRASES)
    for s in d.seqs.values():
        assert s.shape == (6, 3, 32) and s.dtype == np.int64
    eng.check_design(d, BOS_ID)
    e34.check_e34(d, PHRASES, OTHER, _matched(), POOL, LAYOUT)


def test_layout_matches_the_design_table():
    d = _build()
    base = _base()
    s = d.seqs
    for u, name in enumerate(d.units):
        A, B, Cc = PHRASES[name]
        np.testing.assert_array_equal(s["repeat-nl"][:, u, [9, 10, 11, 30, 31]], np.tile([A, B, Cc, A, B], (6, 1)))
        np.testing.assert_array_equal(s["norepeat-nl"][:, u, 9:12], base[:, 9:12])
        np.testing.assert_array_equal(s["norepeat-nl"][:, u, [30, 31]], np.tile([A, B], (6, 1)))
        np.testing.assert_array_equal(s["other-nl"][:, u, 9:12], np.tile(PHRASES[OTHER[name]], (6, 1)))
        np.testing.assert_array_equal(s["repeat-nl-1"][:, u, 30], base[:, 30])  # 31 keeps the row's own token
        assert (s["repeat-nl-1"][:, u, 31] == B).all()
        rr = s["repeat-rand"][:, u]
        np.testing.assert_array_equal(rr[:, 30], rr[:, 9])    # X at 10 and 31
        np.testing.assert_array_equal(rr[:, 31], rr[:, 10])   # Y at 11 and 32
        for v in ("norepeat-rand", "norepeat-rand_b"):
            np.testing.assert_array_equal(s[v][:, u, 30:32], rr[:, 30:32])
        np.testing.assert_array_equal(s["norepeat-rand"][:, u, 9:12], base[:, 9:12])
        alt = s["norepeat-rand_b"][:, u, 9:12]
        assert not any(np.isin(alt[i], base[i]).any() or np.isin(alt[i], rr[i, 9:12]).any() for i in range(6))
        rm = s["repeat-matched"][:, u]
        np.testing.assert_array_equal(rm[:, 30], rm[:, 9])
        np.testing.assert_array_equal(rm[:, 31], rm[:, 10])
        for j in range(3):
            assert np.isin(rm[:, 9 + j], _matched()[name][j]).all()


def test_random_draws_fresh_per_row_and_unit_and_avoid_the_row():
    d = _build(rows=20)
    base = _base(20)
    xyz = d.seqs["repeat-rand"][:, :, 9:12]  # [rows, units, 3]
    for u in range(3):
        assert not any(np.isin(xyz[i, u], base[i]).any() for i in range(20))
    assert not np.array_equal(xyz[:, 0], xyz[:, 1])
    assert len({tuple(x) for x in xyz[:, 0]}) == 20


def test_build_is_deterministic_and_row_stable():
    a, b = _build(rows=6), _build(rows=6)
    for v in e34.VARIANTS:
        np.testing.assert_array_equal(a.seqs[v], b.seqs[v])
    small = _build(rows=3)
    for v in e34.VARIANTS:
        np.testing.assert_array_equal(small.seqs[v], a.seqs[v][:3])
    other_seed = _build(seed=12)
    assert not np.array_equal(other_seed.seqs["repeat-rand"], a.seqs["repeat-rand"])


def test_query_identical_within_contrasts():
    d = _build()
    q = d.seqs
    for v in ("norepeat-nl", "other-nl", "repeat-nl-1"):
        np.testing.assert_array_equal(q[v][..., 31], q["repeat-nl"][..., 31])
    for v in ("norepeat-rand", "norepeat-rand_b"):
        np.testing.assert_array_equal(q[v][..., 31], q["repeat-rand"][..., 31])


def test_check_e34_rejects_other_phrase_sharing_an_id():
    d = _build()
    with pytest.raises(AssertionError, match="shares an id"):
        e34.check_e34(d, PHRASES, {"p0": "p2", "p1": "p2", "p2": "p1"}, _matched(), POOL, LAYOUT)


def test_checks_catch_tampering():
    d = _build()
    d.seqs["norepeat-nl"][0, 0, 20] += 1  # control differs from repeat-nl off 10-12
    with pytest.raises(AssertionError):
        e34.check_e34(d, PHRASES, OTHER, _matched(), POOL, LAYOUT)
    d = _build()
    d.seqs["norepeat-rand"][0, 1, 31] = d.seqs["norepeat-rand"][0, 1, 9]  # Y := the row's token at 10: accidental repeat
    with pytest.raises(AssertionError):
        eng.check_design(d, BOS_ID)


def test_meta_categories_and_repeats():
    d = _build()
    assert d.meta["repeat-nl"].categories == {"Copy": (12,), "Dup": (11,)}
    assert d.meta["repeat-nl"].repeats == ((10, 31), (11, 32))
    assert d.meta["repeat-nl-1"].repeats == ((11, 32),)
    assert d.meta["norepeat-rand"].written == (31, 32)
    c = eng.make_contrast(d, "plc", "norepeat-rand_b", "norepeat-rand", mask_from="repeat-rand")
    assert c.codes[12] == eng.C["Copy"] and c.codes[11] == eng.C["Dup"] and c.codes[10] == eng.C["Span"]


def test_bad_layout_raises():
    with pytest.raises(ValueError):
        e34.build_e34(_base(), PHRASES, OTHER, _matched(), POOL, 1, {**LAYOUT, "copy": 11})


# ---- fallback ------------------------------------------------------------------------------------------------

def test_fallback_layout():
    base = _base()
    d = e34.build_fallback(base, POOL, 5, FB_LAYOUT)
    eng.check_design(d, BOS_ID)
    e34.check_fallback(d, POOL, FB_LAYOUT)
    r = d.seqs["fb-repeat"][:, 0]
    np.testing.assert_array_equal(r[:, 18:31], r[:, 1:14])   # first 13 block ids again at 19-31
    np.testing.assert_array_equal(r[:, 31], r[:, 14])        # query = the 14th block id, at Dup 15
    assert d.meta["fb-repeat"].categories == {"Copy": (16,), "Dup": (15,)}
    np.testing.assert_array_equal(d.seqs["fb-norepeat"][:, 0, 1:16], base[:, 1:16])
    np.testing.assert_array_equal(d.seqs["fb-norepeat"][:, 0, 18:], r[:, 18:])
    alt = d.seqs["fb-norepeat_b"][:, 0, 1:16]
    assert not any(np.isin(alt[i], base[i]).any() or np.isin(alt[i], r[i, 1:16]).any() for i in range(len(base)))


# ---- analysis on synthetic attention -------------------------------------------------------------------------

CFG = {
    "diffuse_threshold": 0.2,
    "e4d_heads": ["L7H0", "L6H0", "L8H7", "L7H11", "L8H5", "L6H11", "L9H3", "L6H8", "L10H2", "L8H11"],
    "analysis": {"late_layers": [6, 11], "margin": 0.005, "bootstrap": {"iters": 200, "seed": 0},
                 "meaningful": {"q": 0.05, "min_abs": 0.02}, "switch": {"q": 0.05, "min_shift": 0.15},
                 "e4d_null": {"sets": 200, "seed": 3}},
}
CONTRASTS = {
    "rand": {"a": "repeat-rand", "b": "norepeat-rand"},
    "content": {"a": "repeat-nl", "b": "repeat-matched", "measures": ["Copy"]},
    "nl_vs_rand": {"a": "repeat-nl", "b": "repeat-rand", "measures": ["Copy"]},
    "park": {"a": "repeat-nl", "b": "other-nl"},
    "two_tok": {"a": "repeat-nl", "b": "repeat-nl-1"},
    "placebo": {"a": "norepeat-rand_b", "b": "norepeat-rand", "mask_from": "repeat-rand"},
}
SPECS = [
    {"id": "E4a", "kind": "switch_count", "contrast": "rand", "labels": "argmax", "from": "Sink", "to": "Copy",
     "layers": [5, 7], "min_beyond_placebo": 3},
    {"id": "E4b", "kind": "renorm_excess", "contrast": "rand", "target": "BOS", "gain": "Copy"},
    {"id": "E4c", "kind": "noninferior", "contrast": "content", "measure": "Copy", "margin": 0.05, "alongside": "nl_vs_rand"},
    {"id": "E4d", "kind": "head_set_loss", "contrast": "park", "measure": "BOS", "min_count": 3, "null_quantile": 95},
    {"id": "E4e", "kind": "positive", "contrast": "two_tok", "measure": "Copy"},
]
INDUCTION = [(5, 1), (5, 5), (6, 9)]


def _synthetic_extraction(rows: int = 40) -> eng.Extraction:
    """Every head parks on BOS (0.7). In the repeat variants the three induction heads move most of BOS onto the copy
    target (key 12). repeat-nl-1 copies less. Four of the E4d heads park less in repeat-nl than in other-nl. Small
    per-row noise; rows sum to 1."""
    rng = np.random.default_rng(0)
    V, K = len(e34.VARIANTS), 33
    att = np.zeros((rows, V, 12, 12, K))
    att[..., 0] = 0.7
    att[..., 1:] = 0.3 / 32
    vi = {v: i for i, v in enumerate(e34.VARIANTS)}
    for v, copy in (("repeat-rand", 0.6), ("repeat-nl", 0.6), ("repeat-matched", 0.6), ("repeat-nl-1", 0.4)):
        for layer, head in INDUCTION:
            att[:, vi[v], layer, head, 12] = copy
            att[:, vi[v], layer, head, 0] = 0.7 - copy  # BOS gives up the copy mass
    for layer, head in [(7, 0), (6, 0), (8, 7), (7, 11)]:
        att[:, vi["repeat-nl"], layer, head, 0] -= 0.1
        att[:, vi["repeat-nl"], layer, head, 20] += 0.1
    att += rng.uniform(0, 0.002, att.shape)
    att /= att.sum(-1, keepdims=True)
    units = 3
    late = att[:, :, 6:12].mean(axis=(2, 3))
    return eng.Extraction(e34.VARIANTS, ("p0", "p1", "p2"), np.full(V, K), (6, 11), att.astype(np.float32),
                          np.repeat(late[:, None], units, axis=1).astype(np.float32),
                          np.repeat(att.mean(axis=0)[None], units, axis=0).astype(np.float32), 0.0, 0, 1.0)


def test_analyse_detects_the_planted_induction_switch():
    d = _build(rows=40)
    res = e34.analyse(_synthetic_extraction(40), d, CFG, CONTRASTS, {"variant": "repeat-rand", "threshold": 0.3}, SPECS)
    assert sorted(res["induction"]["heads"]) == sorted(["L5H1", "L5H5", "L6H9"])
    assert res["placebo_switching"]["argmax"] == []
    p = res["predictions"]
    assert p["E4a"]["ok"] and p["E4a"]["stats"]["count"] == 3
    assert p["E4b"]["ok"] and p["E4b"]["stats"]["mean"] > 0
    assert p["E4c"]["ok"]
    assert p["E4d"]["ok"] and p["E4d"]["stats"]["count"] == 4
    assert p["E4e"]["ok"]
    t = res["transitions"][("rand", "argmax")]
    assert t[eng.LABELS.index("Sink"), eng.LABELS.index("Copy")] == 3
    # the copy-only contrasts get no other measures, labels or switches
    assert ("content", "Copy") in res["heads"] and ("content", "BOS") not in res["heads"]
    assert ("content", "argmax") not in res["switches"] and "content" not in res["tv"]
    assert {p["measure"] for p in res["pooled"] if p["contrast"] == "content"} == {"Copy"}
    preds = e34.check_predictions(SPECS, {"rare": res, "common": res})
    assert [x["verdict"] for x in preds] == ["PASS"] * 5


def test_analyse_without_induction_heads_fails_dependent_predictions():
    d = _build(rows=40)
    ext = _synthetic_extraction(40)
    res = e34.analyse(ext, d, CFG, CONTRASTS, {"variant": "repeat-rand", "threshold": 0.99}, SPECS)
    assert res["induction"]["heads"] == []
    for pid in ("E4b", "E4c", "E4e"):
        assert not res["predictions"][pid]["ok"]
