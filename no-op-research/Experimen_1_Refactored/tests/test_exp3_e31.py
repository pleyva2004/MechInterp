"""Tests for exp3.e31: the E3.1 builder, its layout checks, the config, and the analysis on synthetic attention."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from exp3 import e31
from exp3 import engine as eng

BOS_ID = 0
LAYOUT = {"context": [30, 31], "query": 32, "sanity": [29, 30, 31]}
PHRASES = {"p0": [1, 2, 3], "p1": [4, 5, 6], "p2": [1, 7, 8]}  # p0 and p2 share P, as the real phrases share pronouns
OPRIME = {"p0": "p1", "p1": "p2", "p2": "p0"}  # a derangement: O' = 6, 8, 3
POOL = np.arange(1000, 3000, dtype=np.int64)
CONFIG = Path(__file__).resolve().parents[1] / "configs" / "exp3_e31.yaml"


def _base(rows: int = 6) -> np.ndarray:
    rng = np.random.default_rng(0)
    return np.stack([rng.choice(POOL, 32, replace=False) for _ in range(rows)]).astype(np.int64)


def _matched() -> dict[str, list[np.ndarray]]:
    """P and V pools of 40 ids each; O's pool also holds the other phrases' objects (same form and class)."""
    objects = [ph[2] for ph in PHRASES.values()]
    out = {}
    for k, (n, ph) in enumerate(PHRASES.items()):
        pools = [np.arange(5000 + 100 * (3 * k + j), 5040 + 100 * (3 * k + j)) for j in range(3)]
        pools[2] = np.concatenate([[o for o in objects if o != ph[2]], pools[2]]).astype(np.int64)
        out[n] = pools
    return out


def _build(rows: int = 6, seed: int = 11) -> eng.Design:
    return e31.build_e31(_base(rows), PHRASES, OPRIME, _matched(), seed, LAYOUT)


# ---- builder -------------------------------------------------------------------------------------------------

def test_build_passes_generic_and_layout_checks():
    d = _build()
    assert d.variants == e31.VARIANTS and d.units == tuple(PHRASES)
    for s in d.seqs.values():
        assert s.shape == (6, 3, 32) and s.dtype == np.int64
    eng.check_design(d, BOS_ID)
    e31.check_e31(d, PHRASES, OPRIME, _matched(), LAYOUT)


def test_layout_matches_the_design_table():
    d = _build()
    base = _base()
    s = d.seqs
    for u, name in enumerate(d.units):
        P, V, O = PHRASES[name]
        Op = PHRASES[OPRIME[name]][2]
        rows = np.ones((6, 1), dtype=np.int64)
        np.testing.assert_array_equal(s["PV+O"][:, u, 29:32], rows * [P, V, O])
        np.testing.assert_array_equal(s["PV+Op"][:, u, 29:32], rows * [P, V, Op])
        np.testing.assert_array_equal(s["VP+O"][:, u, 29:32], rows * [V, P, O])
        np.testing.assert_array_equal(s["PV+r"][:, u, 29:31], rows * [P, V])
        np.testing.assert_array_equal(s["VP+r"][:, u, 29:31], rows * [V, P])
        np.testing.assert_array_equal(s["PVO+r"][:, u, 28:31], rows * [P, V, O])
        for v in ("PV+r", "VP+r", "PVO+r", "rr+r"):
            np.testing.assert_array_equal(s[v][:, u, 31], base[:, 31])  # the row's own query
        for x in e31.QUERIES:  # rr+x: the row's own tokens at 30-31, PV+x's query at 32
            np.testing.assert_array_equal(s[f"rr+{x}"][:, u, 29:31], base[:, 29:31])
            np.testing.assert_array_equal(s[f"rr+{x}"][:, u, 31], s[f"PV+{x}"][:, u, 31])
        np.testing.assert_array_equal(s["rr+r"][:, u], base)
        np.testing.assert_array_equal(s["PV+M"][:, u, 29:31], rows * [P, V])
        assert np.isin(s["PV+M"][:, u, 31], _matched()[name][2]).all()
        for v in ("matched", "matched_b"):
            for j in range(3):
                assert np.isin(s[v][:, u, 29 + j], _matched()[name][j]).all()
        np.testing.assert_array_equal(s["PVO+r"][:, u, :28], base[:, :28])


def test_m_and_placebo_draws_avoid_the_row_and_vary():
    d = _build(rows=20)
    base = _base(20)
    m = d.seqs["PV+M"][:, :, 31]  # [rows, units]
    assert not any(np.isin(m[i], base[i]).any() for i in range(20))
    assert len(set(m[:, 0].tolist())) > 5  # drawn per row, not one fixed id
    for v in ("matched", "matched_b"):
        ids = d.seqs[v][:, :, 29:32]
        for u in range(3):
            assert not any(np.isin(ids[i, u], base[i]).any() for i in range(20))
    assert not np.array_equal(d.seqs["matched"][:, :, 29:32], d.seqs["matched_b"][:, :, 29:32])


def test_build_is_deterministic_and_row_stable():
    a, b = _build(rows=6), _build(rows=6)
    for v in e31.VARIANTS:
        np.testing.assert_array_equal(a.seqs[v], b.seqs[v])
    small = _build(rows=3)
    for v in e31.VARIANTS:
        np.testing.assert_array_equal(small.seqs[v], a.seqs[v][:3])
    assert not np.array_equal(_build(seed=12).seqs["PV+M"], a.seqs["PV+M"])


def test_meta_written_positions_include_the_query():
    d = _build()
    assert d.meta["PV+O"].written == (30, 31, 32)
    assert d.meta["rr+O"].written == (32,)
    assert d.meta["PV+r"].written == (30, 31)
    assert d.meta["rr+r"].written == ()
    assert d.meta["PVO+r"].written == (29, 30, 31)
    assert d.meta["matched_b"].written == (30, 31, 32)
    assert all(not m.categories and not m.repeats for m in d.meta.values())


def test_query_writes_pass_but_repeats_and_undeclared_writes_are_caught():
    d = _build()
    eng.check_design(d, BOS_ID)  # nine variants write the query
    d.seqs["PV+O"][0, 0, 31] = d.seqs["PV+O"][0, 0, 4]  # the query repeats a filler id
    with pytest.raises(AssertionError, match="filler position"):
        eng.check_design(d, BOS_ID)
    d = _build()
    d.seqs["PV+M"][2, 1, 31] = d.seqs["PV+M"][2, 1, 29]  # M := P: a repeat among written positions
    with pytest.raises(AssertionError, match="repeat"):
        eng.check_design(d, BOS_ID)
    d = _build()
    d.meta["rr+M"] = eng.VariantMeta(())  # the query write left undeclared
    with pytest.raises(AssertionError, match="outside the written"):
        eng.check_design(d, BOS_ID)


def test_check_e31_catches_tampering():
    d = _build()
    d.seqs["rr+O"][0, 0, 31] = 999_999  # the query differs within C(O)
    with pytest.raises(AssertionError):
        e31.check_e31(d, PHRASES, OPRIME, _matched(), LAYOUT)
    d = _build()
    d.seqs["VP+O"][1, 2, 10] += 1  # VP+O differs from PV+O off 30-31
    with pytest.raises(AssertionError):
        e31.check_e31(d, PHRASES, OPRIME, _matched(), LAYOUT)
    d = _build()
    d.seqs["matched"][0, 0, 29] = 999_999  # a placebo id outside its pool
    with pytest.raises(AssertionError, match="matched pool"):
        e31.check_e31(d, PHRASES, OPRIME, _matched(), LAYOUT)
    d = _build()
    pools = _matched()
    pools["p0"][2] = pools["p0"][2][pools["p0"][2] != 6]  # O' (6) no longer in O's pool
    with pytest.raises(AssertionError, match="matched pool"):
        e31.check_e31(d, PHRASES, OPRIME, pools, LAYOUT)


def test_oprime_validation():
    with pytest.raises(ValueError, match="another phrase"):
        e31.check_oprime(PHRASES, {"p0": "p0", "p1": "p2", "p2": "p0"})
    with pytest.raises(ValueError, match="one O' per phrase"):
        e31.check_oprime(PHRASES, {"p0": "p1", "p1": "p2"})
    with pytest.raises(ValueError, match="own ids"):
        e31.check_oprime({"a": [1, 2, 3], "b": [4, 5, 3]}, {"a": "b", "b": "a"})


def test_bad_layout_raises():
    with pytest.raises(ValueError):
        e31.build_e31(_base(), PHRASES, OPRIME, _matched(), 1, {**LAYOUT, "context": [29, 30]})
    with pytest.raises(ValueError):
        e31.build_e31(_base(), PHRASES, OPRIME, _matched(), 1, {**LAYOUT, "sanity": [28, 29, 30]})


def test_masks_from_pv_o_and_sanity():
    d = _build()
    c = eng.make_contrast(d, "C_O", "PV+O", "rr+O")
    assert c.codes[30] == eng.C["Span"] and c.codes[31] == eng.C["Previous"] and c.codes[32] == eng.C["Self"]
    assert c.phrase == (30, 31)
    assert e31.present_measures(c) == ("BOS", "Previous", "Self", "Span", "Filler", "Phrase")
    s = eng.make_contrast(d, "sanity", "PVO+r", "rr+r")
    assert s.codes[29] == eng.C["Span"] and s.codes[30] == eng.C["Span"] and s.phrase == (29, 30, 31)
    plus, minus = e31.difference_pair(d, "interaction", {"plus": "C_O", "minus": "C_r", "mask_from": "PV+O"},
                                      {"C_O": {"a": "PV+O", "b": "rr+O"}, "C_r": {"a": "PV+r", "b": "rr+r"}})
    assert minus.mask_from == "PV+O" and minus.a == "PV+r" and minus.b == "rr+r"
    np.testing.assert_array_equal(plus.codes, minus.codes)


# ---- the config ----------------------------------------------------------------------------------------------

def test_config_is_consistent():
    cfg = yaml.safe_load(CONFIG.read_text())
    phrases = {n: ph["ids"] for n, ph in cfg["phrases"].items()}
    assert len(phrases) == 12 and cfg["n_bases"] == 500
    e31.check_oprime(phrases, cfg["oprime"])
    assert sorted(cfg["oprime"].values()) == sorted(phrases)  # a derangement: every object once as O'
    for spec in cfg["contrasts"].values():
        assert {spec["a"], spec["b"], spec.get("mask_from", spec["a"])} <= set(e31.VARIANTS)
    for spec in cfg["differences"].values():
        assert spec["plus"] in cfg["contrasts"] and spec["minus"] in cfg["contrasts"]
    assert [p["id"] for p in cfg["predictions"]] == ["E1a", "E1b", "E1c", "E1d", "E1e", "E1e-bet"]
    e31._layout(cfg["layout"])


# ---- analysis on synthetic attention -------------------------------------------------------------------------

CFG = {
    "diffuse_threshold": 0.2,
    "analysis": {"late_layers": [6, 11], "margin": 0.005, "bootstrap": {"iters": 200, "seed": 0},
                 "meaningful": {"q": 0.05, "min_abs": 0.02}, "switch": {"q": 0.05, "min_shift": 0.15},
                 "logit_eps": 1e-6},
}
CONTRASTS = {
    "C_O": {"a": "PV+O", "b": "rr+O"}, "C_Op": {"a": "PV+Op", "b": "rr+Op"}, "C_M": {"a": "PV+M", "b": "rr+M"},
    "C_r": {"a": "PV+r", "b": "rr+r"}, "order_O": {"a": "PV+O", "b": "VP+O"}, "order_r": {"a": "PV+r", "b": "VP+r"},
    "sanity": {"a": "PVO+r", "b": "rr+r"}, "placebo": {"a": "matched_b", "b": "matched", "mask_from": "PV+O"},
}
DIFFERENCES = {
    "interaction": {"plus": "C_O", "minus": "C_r", "mask_from": "PV+O"},
    "completion": {"plus": "C_O", "minus": "C_M", "mask_from": "PV+O"},
    "plausibility": {"plus": "C_O", "minus": "C_Op", "mask_from": "PV+O"},
    "order_x_query": {"plus": "order_O", "minus": "order_r", "mask_from": "PV+O"},
}
SPECS = [
    {"id": "E1a", "kind": "sign", "contrast": "sanity", "measure": "BOS", "sign": "positive"},
    {"id": "E1b", "kind": "sign", "contrast": "interaction", "measure": "BOS", "sign": "negative"},
    {"id": "E1c", "kind": "switch_count", "contrast": "C_O", "labels": "argmax", "from": "Sink",
     "to": ["Span", "Previous"], "layers": [6, 11], "min_beyond_placebo": 5},
    {"id": "E1d", "kind": "unit_sign", "contrast": "order_x_query", "measure": "Phrase", "sign": "positive"},
    {"id": "E1e", "kind": "sign", "contrast": "completion", "measure": "BOS", "sign": "negative"},
    {"id": "E1e-bet", "kind": "sign", "contrast": "plausibility", "measure": "BOS", "sign": "negative",
     "note": "bet, low confidence"},
]
READERS = [(6, 1, 30), (7, 2, 31), (8, 3, 30), (9, 4, 31), (10, 5, 30), (11, 6, 31)]  # (layer, head, key read)


def _synthetic_extraction(rows: int = 40, null: bool = False) -> eng.Extraction:
    """Every head parks on BOS (0.7), the rest spread evenly. Unless null:
    - a coherent context with an unrelated or non-completing query (PV+r, PV+M, PV+Op, PVO+r) raises late BOS to 0.75
      and VP+r to 0.72, as in Exp 2;
    - in PV+O six late heads leave BOS (0.1) and read the context (0.6 on key 30 or 31); the other heads stay at 0.7.
    Small per-row and per-unit noise unless null; rows sum to 1."""
    rng = np.random.default_rng(0)
    V, K, U = len(e31.VARIANTS), 33, 3
    vi = {v: i for i, v in enumerate(e31.VARIANTS)}
    bos = np.full((V, 12, 12), 0.7)
    if not null:
        for v in ("PV+r", "PV+M", "PV+Op", "PVO+r"):
            bos[vi[v], 6:] = 0.75
        bos[vi["VP+r"], 6:] = 0.72
    att = np.repeat(((1.0 - bos) / 32)[None, ..., None], rows, axis=0).repeat(K, axis=-1)
    att[..., 0] = bos[None]
    if not null:
        for layer, head, key in READERS:
            att[:, vi["PV+O"], layer, head] = 0.3 / 32
            att[:, vi["PV+O"], layer, head, 0] = 0.1
            att[:, vi["PV+O"], layer, head, key] += 0.6
        att += rng.uniform(0, 0.002, att.shape)
    att /= att.sum(-1, keepdims=True)
    late = att[:, :, 6:12].mean(axis=(2, 3))  # [rows, V, K]
    late_unit = np.repeat(late[:, None], U, axis=1)
    if not null:
        late_unit = late_unit + rng.uniform(0, 0.001, late_unit.shape)
        late_unit /= late_unit.sum(-1, keepdims=True)
    return eng.Extraction(e31.VARIANTS, tuple(PHRASES), np.full(V, K), (6, 11), att.astype(np.float32),
                          late_unit.astype(np.float32), np.repeat(att.mean(axis=0)[None], U, axis=0).astype(np.float32),
                          0.0, 0, 1.0)


def test_analyse_detects_the_planted_completion_effects():
    d = _build(rows=40)
    res = e31.analyse(_synthetic_extraction(40), d, CFG, CONTRASTS, DIFFERENCES, SPECS)
    assert res["placebo_switching"]["argmax"] == []
    p = res["predictions"]
    assert p["E1a"]["ok"] and p["E1a"]["stats"]["mean"] > 0
    assert p["E1b"]["ok"] and p["E1b"]["stats"]["mean"] == pytest.approx(-0.1, abs=0.01)
    assert p["E1c"]["ok"] and p["E1c"]["stats"]["count"] == 6
    assert sorted(p["E1c"]["stats"]["heads"]) == sorted(f"L{l_}H{h_}" for l_, h_, _ in READERS)
    assert p["E1d"]["ok"] and p["E1d"]["stats"]["unit"]["lo"] > 0
    assert p["E1e"]["ok"] and p["E1e-bet"]["ok"]
    t = res["transitions"][("C_O", "argmax")]
    assert t[eng.LABELS.index("Sink"), eng.LABELS.index("Span")] == 3
    assert t[eng.LABELS.index("Sink"), eng.LABELS.index("Previous")] == 3
    # the differences get measures and the logit check, but no labels, switches or TV
    assert ("interaction", "BOS") in res["heads"] and ("interaction", "argmax") not in res["switches"]
    assert "interaction" not in res["tv"]
    logit = {x["contrast"]: x for x in res["logit"]}
    assert set(logit) == set(DIFFERENCES) and logit["interaction"]["mean"] < 0
    # empty categories are not tested
    assert ("C_O", "Copy") not in res["heads"] and ("sanity", "Span") in res["heads"]
    types = {(x["contrast"], x["type"]) for x in res["pooled"]}
    assert ("interaction", "difference") in types and ("C_O", "two-arm") in types
    preds = e31.check_predictions(SPECS, {"rare": res, "common": res})
    assert [x["verdict"] for x in preds] == ["PASS"] * 6
    assert preds[-1]["note"] == "bet, low confidence"


def test_analyse_null_fails_every_prediction():
    d = _build(rows=40)
    res = e31.analyse(_synthetic_extraction(40, null=True), d, CFG, CONTRASTS, DIFFERENCES, SPECS)
    assert not any(r["ok"] for r in res["predictions"].values())
    assert res["predictions"]["E1c"]["stats"]["count"] == 0


def test_check_predictions_needs_every_base():
    good = {"predictions": {"E1a": {"ok": True, "detail": "x"}}}
    bad = {"predictions": {"E1a": {"ok": False, "detail": "y"}}}
    out = e31.check_predictions([{"id": "E1a"}], {"rare": good, "common": bad})
    assert out[0]["verdict"] == "FAIL" and out[0]["per_base"] == {"rare": True, "common": False}


def test_switch_count_counts_a_head_once_and_subtracts_placebo():
    sw = [{"layer": 7, "head": 1, "to": "Span", "from": "Sink", "shift": 0.3, "q": 0.01},
          {"layer": 7, "head": 1, "to": "Previous", "from": "Sink", "shift": 0.2, "q": 0.01},
          {"layer": 8, "head": 0, "to": "Span", "from": "Diffuse", "shift": 0.4, "q": 0.01},  # wrong source
          {"layer": 3, "head": 0, "to": "Span", "from": "Sink", "shift": 0.4, "q": 0.01}]     # not late
    res = {"switches": {("C_O", "argmax"): sw}, "placebo_switching": {"argmax": [(2, 2)]}}
    out = e31.evaluate(SPECS[2], res, {}, CFG)
    assert out["stats"]["count"] == 1 and out["stats"]["placebo"] == 1 and out["stats"]["beyond"] == 0
    assert not out["ok"]


def test_summarize_query_logp():
    lp = {ctx: np.zeros((4, 2, 4)) for ctx in e31.CONTEXTS}
    lp["PV"][..., 0] = -1.0                        # O
    lp["PV"][..., 1] = [[-3.0, -0.5]] * 4          # O': beaten by O in phrase 0 only
    lp["PV"][..., 2] = -2.0                        # M
    lp["PV"][..., 3] = -9.0                        # r
    lp["rr"][..., 0] = -4.0
    s = e31.summarize_query_logp(lp, ("a", "b"))
    assert s["mean"]["PV"]["O"] == -1.0 and s["rows"] == 4
    assert s["pv_O_beats"] == {"Op": 0.5, "M": 1.0, "r": 1.0}
    assert s["lift_pv_minus_rr"]["O"] == pytest.approx(3.0)
    assert s["per_phrase_O_minus_Op"] == {"a": 2.0, "b": -0.5}


def test_query_candidates_order():
    d = _build()
    c = e31.query_candidates(d, 4)
    assert c.shape == (4, 3, 4)
    np.testing.assert_array_equal(c[..., 0], d.seqs["PV+O"][:4, :, 31])
    np.testing.assert_array_equal(c[..., 3], _base()[:4, None, 31].repeat(3, axis=1))
