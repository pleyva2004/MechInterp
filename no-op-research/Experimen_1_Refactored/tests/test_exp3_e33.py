"""Tests for exp3.e33: the E3.3 builder, its layout checks, masks at both lengths, half-distance, and the analysis on
synthetic attention."""
from __future__ import annotations

import numpy as np
import pytest

from exp3 import e33
from exp3 import engine as eng

BOS_ID = 0
GAPS = [0, 1, 2, 3, 4, 6, 8, 12, 15, 24]
LAYOUT = {"gaps": GAPS, "phrase_end": 31, "query": 32, "placebo_gap": 0,
          "length_control": {"seq_len": 24, "phrase": [21, 22, 23]}}
PHRASES = {"p0": [1, 2, 3], "p1": [4, 5, 6], "p2": [1, 7, 8]}  # p0 and p2 share id 1
SHUFFLED = {n: [ids[1], ids[0], ids[2]] for n, ids in PHRASES.items()}
POOL = np.arange(1000, 3000, dtype=np.int64)


def _base(rows: int = 6) -> np.ndarray:
    rng = np.random.default_rng(0)
    return np.stack([rng.choice(POOL, 32, replace=False) for _ in range(rows)]).astype(np.int64)


def _matched() -> dict[str, list[np.ndarray]]:
    return {n: [np.arange(5000 + 100 * (3 * k + j), 5000 + 100 * (3 * k + j) + 40) for j in range(3)]
            for k, n in enumerate(PHRASES)}


def _build(rows: int = 6, seed: int = 11) -> eng.Design:
    return e33.build_e33(_base(rows), PHRASES, SHUFFLED, _matched(), seed, LAYOUT)


# ---- builder -------------------------------------------------------------------------------------------------

def test_build_passes_generic_and_layout_checks():
    d = _build()
    assert len(d.variants) == 25 and d.units == tuple(PHRASES)
    for v, s in d.seqs.items():
        assert s.shape == ((6, 3, 24) if v.endswith("len24") else (6, 3, 32)) and s.dtype == np.int64
    assert d.key_len("nl_len24") == 25 and d.key_len("nl_g0") == 33
    eng.check_design(d, BOS_ID)
    e33.check_e33(d, PHRASES, SHUFFLED, _matched(), LAYOUT)


def test_layout_matches_the_design_table():
    d, base, s = _build(), _base(), _build().seqs
    np.testing.assert_array_equal(s["none"], np.repeat(base[:, None], 3, axis=1))
    for u, name in enumerate(d.units):
        P, V, O = PHRASES[name]
        for g in GAPS:
            pos = np.array([29 - g, 30 - g, 31 - g])
            np.testing.assert_array_equal(s[f"nl_g{g}"][:, u, pos - 1], np.tile([P, V, O], (6, 1)))
            np.testing.assert_array_equal(s[f"shuffled_g{g}"][:, u, pos - 1], np.tile([V, P, O], (6, 1)))
            # random filler between the phrase and the query, and the query itself, are the row's own tokens
            np.testing.assert_array_equal(s[f"nl_g{g}"][:, u, 31 - g:], base[:, 31 - g:])
        assert d.meta["nl_g15"].written == (14, 15, 16) and d.meta["nl_g24"].written == (5, 6, 7)
        short = s["nl_len24"][:, u]
        np.testing.assert_array_equal(short[:, 20:23], np.tile([P, V, O], (6, 1)))
        np.testing.assert_array_equal(short[:, :20], base[:, :20])
        np.testing.assert_array_equal(short[:, 23], base[:, 23])  # query 24 = the row's own token at 24
        np.testing.assert_array_equal(s["shuffled_len24"][:, u, 20:23], np.tile([V, P, O], (6, 1)))
        for v in ("matched", "matched_b"):
            m = s[v][:, u]
            for j in range(3):
                assert np.isin(m[:, 28 + j], _matched()[name][j]).all()
            assert not any(np.isin(m[i, 28:31], base[i]).any() for i in range(6))
        assert not np.array_equal(s["matched"][:, u, 28:31], s["matched_b"][:, u, 28:31])


def test_build_is_deterministic_and_row_stable():
    a, b = _build(rows=6), _build(rows=6)
    for v in a.variants:
        np.testing.assert_array_equal(a.seqs[v], b.seqs[v])
    small = _build(rows=3)
    for v in a.variants:
        np.testing.assert_array_equal(small.seqs[v], a.seqs[v][:3])
    assert not np.array_equal(_build(seed=12).seqs["matched"], a.seqs["matched"])


def test_checks_catch_tampering():
    d = _build()
    d.seqs["shuffled_g3"][0, 1, 31 - 3 - 1] += 1  # O no longer identical in nl and shuffled
    with pytest.raises(AssertionError):
        e33.check_e33(d, PHRASES, SHUFFLED, _matched(), LAYOUT)
    d = _build()
    d.seqs["nl_len24"][0, 0, 5] += 1  # length-control row differs from its base outside 21-23
    with pytest.raises(AssertionError):
        e33.check_e33(d, PHRASES, SHUFFLED, _matched(), LAYOUT)
    d = _build()
    d.seqs["matched"][0, 2, 29] = d.seqs["matched"][0, 2, 5]  # a matched id repeats a filler id
    with pytest.raises(AssertionError):
        eng.check_design(d, BOS_ID)
    bad = {**SHUFFLED, "p1": [6, 4, 5]}  # not V P O
    with pytest.raises(AssertionError, match="V P O"):
        e33.check_e33(e33.build_e33(_base(), PHRASES, bad, _matched(), 11, LAYOUT), PHRASES, bad, _matched(), LAYOUT)


def test_bad_layouts_raise():
    for bad in ({**LAYOUT, "gaps": [0, 1, 29]},           # phrase would reach BOS
                {**LAYOUT, "gaps": [1, 2]},                # no g = 0
                {**LAYOUT, "placebo_gap": 5},
                {**LAYOUT, "length_control": {"seq_len": 24, "phrase": [20, 21, 22]}}):
        with pytest.raises(ValueError):
            e33.build_e33(_base(), PHRASES, SHUFFLED, _matched(), 1, bad)


# ---- masks ---------------------------------------------------------------------------------------------------

def test_masks_use_each_length_own_positions():
    d = _build()
    c = eng.make_contrast(d, "len24", "nl_len24", "shuffled_len24")
    assert len(c.codes) == 33
    assert c.codes[0] == eng.C["BOS"] and c.codes[24] == eng.C["Self"] and c.codes[23] == eng.C["Previous"]
    assert c.codes[21] == c.codes[22] == eng.C["Span"] and (c.codes[1:21] == eng.C["Filler"]).all()
    assert (c.codes[25:] == -1).all() and c.phrase == (21, 22, 23)
    c0 = eng.make_contrast(d, "order_g0", "nl_g0", "shuffled_g0")
    assert c0.codes[31] == eng.C["Previous"] and c0.codes[32] == eng.C["Self"]
    assert c0.codes[29] == c0.codes[30] == eng.C["Span"] and c0.phrase == (29, 30, 31)
    c3 = eng.make_contrast(d, "total_g3", "nl_g3", "none")
    assert (c3.codes[[26, 27, 28]] == eng.C["Span"]).all() and c3.codes[31] == eng.C["Previous"]
    assert c3.codes[29] == c3.codes[30] == eng.C["Filler"] and c3.phrase == (26, 27, 28)
    plc = eng.make_contrast(d, "placebo", "matched", "matched_b", mask_from="nl_g0")
    np.testing.assert_array_equal(plc.codes, c0.codes)
    with pytest.raises(ValueError):
        eng.make_contrast(d, "x", "nl_len24", "nl_g0")  # no contrast across lengths


# ---- half-distance -------------------------------------------------------------------------------------------

def test_half_distance_rule():
    gaps = [0, 1, 2, 4, 8]
    eff = np.zeros((5, 1, 5))
    eff[:, 0, 0] = [0.10, 0.04, 0.02, 0.0, 0.0]     # halves at 1
    eff[:, 0, 1] = [0.10, 0.08, 0.06, 0.05, 0.01]   # 0.05 / 0.10 = 0.5 at 4 -> 4
    eff[:, 0, 2] = [-0.10, -0.09, -0.08, -0.07, -0.06]  # negative, never halves -> inf
    eff[:, 0, 3] = [0.10, -0.01, 0.0, 0.0, 0.0]     # sign flip counts as halved -> 1
    eff[:, 0, 4] = [0.10, 0.0, 0.0, 0.0, 0.0]       # not eligible -> nan
    elig = np.array([[True, True, True, True, False]])
    hd = e33.half_distance(eff, gaps, elig, 0.5)
    assert hd[0, 0] == 1 and hd[0, 1] == 4 and np.isinf(hd[0, 2]) and hd[0, 3] == 1 and np.isnan(hd[0, 4])
    with pytest.raises(ValueError):
        e33.half_distance(eff, [1, 2, 4, 8, 16], elig, 0.5)


# ---- analysis on synthetic attention -------------------------------------------------------------------------

CONTRASTS = {**{f"order_g{g}": {"a": f"nl_g{g}", "b": f"shuffled_g{g}"} for g in GAPS},
             **{f"total_g{g}": {"a": f"nl_g{g}", "b": "none"} for g in GAPS},
             "len24": {"a": "nl_len24", "b": "shuffled_len24"},
             "placebo": {"a": "matched", "b": "matched_b", "mask_from": "nl_g0"}}
CFG = {
    "layout": LAYOUT,
    "diffuse_threshold": 0.2,
    "sweep": {"order": "order_g", "total": "total_g"},
    "length_control": {"contrast": "len24", "same_distance": "order_g0", "same_position": "order_g8"},
    "half_distance": {"measure": "BOS", "ratio": 0.5, "long_range_min": 4},
    "analysis": {"late_layers": [6, 11], "margin": 0.005, "bootstrap": {"iters": 200, "seed": 0},
                 "meaningful": {"q": 0.05, "min_abs": 0.02}, "switch": {"q": 0.05, "min_shift": 0.15},
                 "measures": ["BOS", "Previous", "Self", "Span", "Filler", "Phrase"]},
}
SPECS = [
    {"id": "E3a", "kind": "combination_positive", "terms": [["order_g0", 1.0], ["order_g1", -2.0]], "measure": "BOS"},
    {"id": "E3b", "kind": "equivalent", "contrast": "order_g24", "measure": "BOS", "margin": 0.005},
    {"id": "E3c", "kind": "positive", "contrast": "total_g15", "measure": "BOS"},
]
SHORT = [(layer, head) for layer in range(6, 12) for head in range(3)]  # 18 late heads, fast decay
LONG = (9, 5)                                                           # one late head that halves only at g = 8


def _synthetic_extraction(d: eng.Design, rows: int, order=lambda g: 0.12 * 0.3 ** g, long_order=None,
                          shared: float = 0.03, short_len: float = 0.06) -> eng.Extraction:
    """Every head parks on BOS (0.7). In nl_g the SHORT heads gain order(g) more BOS than in shuffled_g; both phrase
    arms gain `shared` over none in the SHORT heads (so total = order + shared). LONG gains long_order(g). The 24-token
    nl arm gains short_len in the SHORT heads. BOS gains come out of the other keys; padding keys stay 0; small per-row
    noise; rows sum to 1."""
    long_order = long_order or (lambda g: 0.10 if g <= 6 else 0.04)
    rng = np.random.default_rng(0)
    V, K = len(d.variants), 33
    att = np.zeros((rows, V, 12, 12, K))
    for vi, v in enumerate(d.variants):
        kl = d.key_len(v)
        extra = np.zeros((12, 12))
        if v.startswith(("nl_g", "shuffled_g")):
            g = int(v.split("_g")[1])
            for lh in SHORT:
                extra[lh] += shared
            if v.startswith("nl_g"):
                for lh in SHORT:
                    extra[lh] += order(g)
                extra[LONG] += long_order(g)
        elif v == "nl_len24":
            for lh in SHORT:
                extra[lh] += short_len
        att[:, vi, :, :, 0] = 0.7 + extra
        att[:, vi, :, :, 1:kl] = ((0.3 - extra) / (kl - 1))[..., None]
    noise = rng.uniform(0, 0.002, att.shape)
    att += np.where(att > 0, noise, 0.0)
    att /= att.sum(-1, keepdims=True)
    units = len(d.units)
    late = att[:, :, 6:12].mean(axis=(2, 3))
    return eng.Extraction(d.variants, d.units, np.array([d.key_len(v) for v in d.variants]), (6, 11),
                          att.astype(np.float32), np.repeat(late[:, None], units, axis=1).astype(np.float32),
                          np.repeat(att.mean(axis=0)[None], units, axis=0).astype(np.float32), 0.0, 0, 1.0)


def test_analyse_recovers_the_planted_sweep():
    d = _build(rows=40)
    res = e33.analyse(_synthetic_extraction(d, 40), d, CFG, CONTRASTS, SPECS)
    assert res["placebo_switching"]["argmax"] == [] and res["placebo_switching"]["mass"] == []
    p = res["predictions"]
    assert p["E3a"]["ok"] and p["E3a"]["stats"]["mean"] > 0
    assert p["E3b"]["ok"] and abs(p["E3b"]["stats"]["mean"]) < 0.005
    assert p["E3c"]["ok"] and p["E3c"]["stats"]["late_meaningful_pos"] == 19  # SHORT (+0.03) and LONG (+0.04)
    # E(0) = (18 * 0.12 + 0.10) / 72, E(1) = (18 * 0.036 + 0.10) / 72
    assert p["E3a"]["stats"]["parts"]["order_g0"]["mean"] == pytest.approx((18 * 0.12 + 0.10) / 72 / 1.033, rel=0.05)
    sweep = {(x["effect"], x["g"]): x for x in res["sweep"]}
    assert sweep[("order", 0)]["mean"] > sweep[("order", 1)]["mean"] > sweep[("order", 24)]["mean"]
    assert sweep[("total", 15)]["mean"] > 0 and sweep[("order", 0)]["positions"] == "29-31"
    hd = res["half_distance"]["order"]
    assert hd["eligible"].sum() == 19
    assert all(hd["half_distance"][lh] == 1 for lh in SHORT) and hd["half_distance"][LONG] == 8
    assert hd["long_range_heads"] == ["L9H5"]
    assert np.isnan(hd["half_distance"][0, 0])
    lc = res["length_control"]
    assert lc["E_length_control"]["mean"] == pytest.approx(18 * 0.06 / 72 / 1.03, rel=0.05)
    assert lc["length_control_minus_same_distance"]["mean"] < 0
    # masses at the short length are measured on its own keys: Previous = 23, Self = 24, nothing beyond
    i = res["contrasts"]["len24"]
    assert i.codes[23] == eng.C["Previous"] and i.codes[24] == eng.C["Self"]
    assert {x["measure"] for x in res["pooled"] if x["contrast"] == "len24"} == set(CFG["analysis"]["measures"])
    preds = e33.check_predictions(SPECS, {"rare": res, "common": res})
    assert [x["verdict"] for x in preds] == ["PASS"] * 3


def test_analyse_fails_predictions_when_the_effect_does_not_fade():
    d = _build(rows=40)
    flat = _synthetic_extraction(d, 40, order=lambda g: 0.12, long_order=lambda g: 0.0, shared=0.0)
    res = e33.analyse(flat, d, CFG, CONTRASTS, SPECS)
    p = res["predictions"]
    assert not p["E3a"]["ok"]                     # E(0) - 2 E(1) = -E(0) < 0
    assert not p["E3b"]["ok"]                     # E(24) = E(0), far outside +-m
    assert res["half_distance"]["order"]["long_range_heads"]  # every planted head is "> 24"
    assert np.isinf(res["half_distance"]["order"]["half_distance"][SHORT[0]])


def test_check_predictions_needs_every_base_class():
    d = _build(rows=40)
    good = e33.analyse(_synthetic_extraction(d, 40), d, CFG, CONTRASTS, SPECS)
    bad = e33.analyse(_synthetic_extraction(d, 40, order=lambda g: 0.12), d, CFG, CONTRASTS, SPECS)
    preds = {p["id"]: p for p in e33.check_predictions(SPECS, {"rare": good, "common": bad})}
    assert preds["E3a"]["verdict"] == "FAIL" and preds["E3a"]["per_base"] == {"rare": True, "common": False}
