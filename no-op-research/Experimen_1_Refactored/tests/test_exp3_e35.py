"""Tests for exp3.e35: the E3.5 builder, its layout checks, the split extraction, and the analysis on synthetic
attention. The contrasts, interactions and prediction specs come from configs/exp3_e35.yaml itself."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from exp3 import e35
from exp3 import engine as eng

CFG_FILE = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "exp3_e35.yaml").read_text())
BOS_ID = 0
LAYOUT = CFG_FILE["layout"]
BYTES = {n: b["id"] for n, b in CFG_FILE["bytes"].items()}
PHRASES = {"p0": [1001, 1002, 1003], "p1": [1004, 1005, 1006], "p2": [1001, 1007, 1008]}  # p0 and p2 share an id
POOL = np.arange(1100, 3000, dtype=np.int64)  # the base rows' pool: no phrase id, no byte id


def _base(rows: int = 6) -> np.ndarray:
    rng = np.random.default_rng(0)
    return np.stack([rng.choice(POOL, 32, replace=False) for _ in range(rows)]).astype(np.int64)


def _build(rows: int = 6, seed: int = 11) -> eng.Design:
    return e35.build_e35(_base(rows), PHRASES, BYTES, POOL, seed, LAYOUT)


# ---- config --------------------------------------------------------------------------------------------------

def test_config_matches_the_design():
    assert BYTES == {"period": 13, "comma": 11, "newline": 198, "x": 87, "seven": 22}
    assert CFG_FILE["bytes"]["newline"]["text"] == "\n"
    assert LAYOUT == {"phrase": [26, 27, 28], "delimiter": 29, "query": 32}
    assert CFG_FILE["n_bases"] == 300 and len(CFG_FILE["phrases"]) == 12
    assert all(lo > e35.MAX_BYTE_ID for lo, _ in CFG_FILE["bases"].values())
    names = set(e35.variant_names(BYTES))
    assert len(names) == 13
    for s in CFG_FILE["contrasts"].values():
        assert {s["a"], s["b"], s.get("mask_from", s["a"])} <= names
    for s in CFG_FILE["interactions"].values():
        assert {s["treated"], s["control"]} <= set(CFG_FILE["contrasts"])


# ---- builder -------------------------------------------------------------------------------------------------

def test_build_passes_generic_and_layout_checks():
    d = _build()
    assert d.variants == e35.variant_names(BYTES) and d.units == tuple(PHRASES)
    assert d.variants[:6] == tuple(f"phrase+{x}" for x in ("period", "comma", "newline", "x", "seven", "r"))
    assert d.variants[-1] == e35.PLACEBO
    for s in d.seqs.values():
        assert s.shape == (6, 3, 32) and s.dtype == np.int64
    eng.check_design(d, BOS_ID)
    e35.check_e35(d, PHRASES, BYTES, POOL, LAYOUT)


def test_layout_matches_the_design():
    d = _build()
    base = _base()
    s = d.seqs
    for u, name in enumerate(d.units):
        for x in e35.d_names(BYTES):
            ph, rd = s[f"phrase+{x}"][:, u], s[f"random+{x}"][:, u]
            np.testing.assert_array_equal(ph[:, 25:28], np.tile(PHRASES[name], (6, 1)))   # phrase at 26-28
            np.testing.assert_array_equal(rd[:, 25:28], base[:, 25:28])                   # the row's own tokens
            at29 = base[:, 28] if x == "r" else np.full(6, BYTES[x])
            np.testing.assert_array_equal(ph[:, 28], at29)
            np.testing.assert_array_equal(rd[:, 28], at29)
            for v in (ph, rd):
                np.testing.assert_array_equal(v[:, 29:], base[:, 29:])                    # 30-32 = r (31 = r)
                np.testing.assert_array_equal(v[:, :25], base[:, :25])
    np.testing.assert_array_equal(s["random+r"][:, 0], base)                              # none = the base row
    alt = s[e35.PLACEBO][:, 0, 25:28]
    assert not any(np.isin(alt[i], base[i]).any() for i in range(6))
    np.testing.assert_array_equal(s[e35.PLACEBO][:, 0, 28:], base[:, 28:])


def test_meta_written_and_delimiter_everywhere():
    d = _build()
    for v, m in d.meta.items():
        assert m.categories == {"Delimiter": (29,)}  # position 29 in every arm, also where it holds r
    assert d.meta["phrase+period"].written == (26, 27, 28, 29)
    assert d.meta["phrase+r"].written == (26, 27, 28)
    assert d.meta["random+x"].written == (29,)
    assert d.meta["random+r"].written == ()
    assert d.meta[e35.PLACEBO].written == (26, 27, 28)


def test_shared_variants_identical_across_units_and_placebo_once_per_row():
    d = _build(rows=20)
    for v in e35.shared_variants(BYTES):
        for u in range(1, 3):
            np.testing.assert_array_equal(d.seqs[v][:, u], d.seqs[v][:, 0])
    alt = d.seqs[e35.PLACEBO][:, 0, 25:28]
    assert len({tuple(x) for x in alt}) == 20
    assert set(e35.shared_variants(BYTES)) == {f"random+{x}" for x in e35.d_names(BYTES)} | {e35.PLACEBO}


def test_build_is_deterministic_and_row_stable():
    a, b = _build(rows=6), _build(rows=6)
    for v in a.variants:
        np.testing.assert_array_equal(a.seqs[v], b.seqs[v])
    small = _build(rows=3)
    for v in a.variants:
        np.testing.assert_array_equal(small.seqs[v], a.seqs[v][:3])
    assert not np.array_equal(_build(seed=12).seqs[e35.PLACEBO], a.seqs[e35.PLACEBO])


def test_masks_from_the_treated_arm():
    d = _build()
    g = eng.make_contrast(d, "gain_period", "phrase+period", "phrase+r")
    assert g.codes[29] == eng.C["Delimiter"] and g.codes[31] == eng.C["Previous"] and g.codes[32] == eng.C["Self"]
    assert all(g.codes[p] == eng.C["Span"] for p in (26, 27, 28)) and g.codes[30] == eng.C["Filler"]
    assert g.phrase == (26, 27, 28, 29)
    rg = eng.make_contrast(d, "rgain_period", "random+period", "random+r", mask_from="phrase+period")
    np.testing.assert_array_equal(rg.codes, g.codes)
    own = eng.make_contrast(d, "own", "random+period", "random+r")
    assert own.codes[29] == eng.C["Delimiter"] and own.codes[27] == eng.C["Filler"] and own.phrase == (29,)
    none = eng.make_contrast(d, "ctx_r", "phrase+r", "random+r")
    assert none.codes[29] == eng.C["Delimiter"] and none.phrase == (26, 27, 28)


def test_checks_catch_tampering():
    d = _build()
    d.seqs["phrase+r"][0, 0, 10] += 1  # the r arm differs from the d arm off 29
    with pytest.raises(AssertionError):
        e35.check_e35(d, PHRASES, BYTES, POOL, LAYOUT)
    d = _build()
    d.seqs["random+comma"][2, 1, 28] = BYTES["period"]  # one unit of a shared variant changed
    with pytest.raises(AssertionError):
        e35.check_e35(d, PHRASES, BYTES, POOL, LAYOUT)
    d = _build()
    d.base[:, :, 5] = BYTES["seven"]  # a byte token at 6 in every variant: every other check still holds
    for s in d.seqs.values():
        s[:, :, 5] = BYTES["seven"]
    with pytest.raises(AssertionError, match="byte token"):
        e35.check_e35(d, PHRASES, BYTES, POOL, LAYOUT)


def test_bad_inputs_raise():
    with pytest.raises(ValueError):
        e35.build_e35(_base(), PHRASES, BYTES, POOL, 1, {"phrase": [28, 29, 30], "delimiter": 31, "query": 32})  # q - 1
    with pytest.raises(ValueError):
        e35.build_e35(_base(), PHRASES, BYTES, POOL, 1, {**LAYOUT, "delimiter": 30})  # not right after the phrase
    with pytest.raises(ValueError):
        e35.build_e35(_base(), PHRASES, {**BYTES, "big": 300}, POOL, 1, LAYOUT)
    with pytest.raises(ValueError):
        e35.build_e35(_base(), PHRASES, {**BYTES, "r": 5}, POOL, 1, LAYOUT)


# ---- split extraction ----------------------------------------------------------------------------------------

def _fake_attention(model, tokens, batch_size, query_position=-1):
    """Deterministic per-sequence 'attention' [n, 12, 12, K] that depends on every token, rows summing to 1."""
    tokens = np.asarray(tokens, dtype=np.int64)
    coef = np.arange(1, 145, dtype=np.int64).reshape(12, 12)
    pos = np.arange(tokens.shape[1], dtype=np.int64)
    w = ((tokens[:, None, None, :] * coef[None, :, :, None] + pos * 7 + tokens.sum(axis=1)[:, None, None, None]) % 97
         + 1).astype(np.float64)
    return (w / w.sum(-1, keepdims=True)).astype(np.float32)


def test_split_extraction_equals_the_full_extraction(monkeypatch):
    monkeypatch.setattr(eng, "get_final_attention", _fake_attention)
    monkeypatch.setattr(eng, "model_shape", lambda model: (12, 12))
    d = _build(rows=7)
    full = eng.extract(None, d, BOS_ID, (6, 11), 3, 1e-4)
    merged, parts = e35.extract_split(None, d, e35.shared_variants(BYTES), BOS_ID, (6, 11),
                                      {"per_unit": 3, "shared": 5}, 1e-4)
    assert merged.variants == full.variants and merged.units == full.units
    np.testing.assert_array_equal(merged.key_len, full.key_len)
    for k in ("att_row", "att_late_unit", "att_unit"):
        np.testing.assert_allclose(getattr(merged, k), getattr(full, k), rtol=1e-6, atol=1e-8)
    n_shared = len(e35.shared_variants(BYTES))
    assert merged.n_sequences == (13 - n_shared) * 3 * 7 + n_shared * 7
    assert parts["shared"][0].units == ("shared",) and parts["per_unit"][0].units == tuple(PHRASES)
    for name, (pd_, pe) in parts.items():
        det = eng.determinism_check(None, pd_, pe, BOS_ID, {"per_unit": 3, "shared": 5}[name], 1e-4, 2)
        assert det["exact"]


def test_split_refuses_a_unit_dependent_shared_variant():
    d = _build()
    with pytest.raises(AssertionError):
        e35.split_design(d, ["phrase+period"])


# ---- analysis on synthetic attention -------------------------------------------------------------------------

CFG = {
    "diffuse_threshold": 0.2,
    "previous_heads": CFG_FILE["previous_heads"],
    "analysis": {**CFG_FILE["analysis"], "bootstrap": {"iters": 200, "seed": 0}},
}
DELIM_HEADS = [(6, 1), (7, 2), (8, 3), (9, 4), (10, 5), (11, 6)]


def _synthetic_extraction(rows: int = 40, random_absorbs: bool = False, break_prev: bool = False) -> eng.Extraction:
    """Every head parks on BOS (0.7); Exp 1's previous-token heads read key 31 (0.6). After the phrase, six late heads
    move BOS mass onto key 29 when it holds "." (0.6) and a little when it holds "x" or "7" (0.05); on random
    context "." takes nothing, unless random_absorbs (0.62, slightly more than after the phrase). break_prev moves
    L2H2 from key 31 to BOS in phrase+period. Small per-row noise; rows sum to 1."""
    rng = np.random.default_rng(0)
    names = e35.variant_names(BYTES)
    V, K = len(names), 33
    vi = {v: i for i, v in enumerate(names)}
    att = np.zeros((rows, V, 12, 12, K))
    att[..., 0] = 0.7
    att[..., 1:] = 0.3 / 32
    for l_, h_ in eng.parse_heads(CFG["previous_heads"]):
        att[..., l_, h_, 0] = 0.3
        att[..., l_, h_, 31] = 0.4 + 0.3 / 32
    moves = [("phrase+period", 0.6), ("phrase+x", 0.05), ("phrase+seven", 0.05)]
    if random_absorbs:
        moves.append(("random+period", 0.62))
    for v, amount in moves:
        for l_, h_ in DELIM_HEADS:
            att[:, vi[v], l_, h_, 29] += amount
            att[:, vi[v], l_, h_, 0] -= amount
    if break_prev:
        att[:, vi["phrase+period"], 2, 2, 31] -= 0.4
        att[:, vi["phrase+period"], 2, 2, 0] += 0.4
    att += rng.uniform(0, 0.002, att.shape)
    att /= att.sum(-1, keepdims=True)
    units = 3
    late = att[:, :, 6:12].mean(axis=(2, 3))
    return eng.Extraction(names, tuple(PHRASES), np.full(V, K), (6, 11), att.astype(np.float32),
                          np.repeat(late[:, None], units, axis=1).astype(np.float32),
                          np.repeat(att.mean(axis=0)[None], units, axis=0).astype(np.float32), 0.0, 0, 1.0)


def _analyse(ext: eng.Extraction) -> dict:
    return e35.analyse(ext, _build(rows=ext.att_row.shape[0]), CFG, CFG_FILE["contrasts"], CFG_FILE["interactions"],
                       CFG_FILE["predictions"])


def test_analyse_detects_the_planted_delimiter_shift():
    res = _analyse(_synthetic_extraction())
    assert res["placebo_switching"]["argmax"] == []
    p = res["predictions"]
    assert p["E5a"]["ok"], p["E5a"]["detail"]
    sw = p["E5a"]["stats"]["parts"][1]
    assert sw["count"] == 6 and sw["beyond"] == 6
    assert p["E5b"]["ok"], p["E5b"]["detail"]
    assert p["E5c"]["ok"], p["E5c"]["detail"]
    t = res["transitions"][("gain_period", "argmax")]
    assert t[eng.LABELS.index("Sink"), eng.LABELS.index("Delimiter")] == 6
    assert res["previous"]["ok"]
    assert all(r == "Previous" for roles in res["previous"]["roles"].values() for r in roles.values())
    # BOS + key 29 is conserved in the moving heads, so the interaction on it is about 0
    pooled = {(x["contrast"], x["measure"]): x for x in res["pooled"]}
    assert abs(pooled[("x_period", "BOS+Delimiter")]["mean"]) < 0.002
    assert pooled[("x_period", "Delimiter")]["mean"] > 0.04
    assert ("x_period", "argmax") not in res["switches"]  # interactions get no labels
    preds = e35.check_predictions(CFG_FILE["predictions"], {"rare": res, "common": res})
    assert [x["verdict"] for x in preds] == ["PASS"] * 3


def test_no_interaction_when_random_context_absorbs_too():
    res = _analyse(_synthetic_extraction(random_absorbs=True))
    p = res["predictions"]
    assert not p["E5a"]["ok"]
    assert p["E5a"]["stats"]["parts"][0]["hi"] < 0 and p["E5a"]["stats"]["parts"][1]["count"] == 6
    assert p["E5c"]["ok"]  # the "." vs "x" / "7" comparison is within the phrase context
    preds = e35.check_predictions(CFG_FILE["predictions"], {"rare": res, "common": _analyse(_synthetic_extraction())})
    assert preds[0]["verdict"] == "FAIL" and preds[0]["per_base"] == {"rare": False, "common": True}


def test_previous_check_flags_a_moved_previous_head():
    res = _analyse(_synthetic_extraction(break_prev=True))
    pv = res["previous"]
    assert not pv["ok"]
    assert {"variant": "phrase+period", "head": "L2H2", "role": "Sink"} in pv["not_previous"]
    assert any(s["contrast"] == "gain_period" and (s["layer"], s["head"]) == (2, 2) for s in pv["switches_from_previous"])
