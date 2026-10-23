"""Tests for exp3.e32: the E3.2 builder, its layout checks, the per-passage exclusion, the sweep helpers and the
analysis on synthetic attention. A small layout (12-token rows, K = 11) stands in for the real one (32, K = 31)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from exp3 import e32
from exp3 import engine as eng

BOS_ID = 0
SEQ = 12
LAYOUT = {"ks": [1, 2, 3, 5, 11], "shuffled_ks": [3, 5, 11], "placebo_ks": [5, 11], "rr": [10, 11], "query": 12}
POOL = np.arange(1000, 1400, dtype=np.int64)
# ids 1 and 2 are punctuation (no matched pool); pa repeats 101 (distance 4) and the comma-like 1 (distance 7).
SPANS = {
    "pa": [101, 1, 102, 103, 101, 2, 104, 105, 1, 106, 107],
    "pb": [201, 202, 2, 203, 204, 205, 206, 207, 208, 202, 209],
    "pc": [1000, 1001, 301, 302, 1, 303, 1002, 304, 305, 306, 1003],  # ids inside the base pool: filler collisions
}
T_IDS = {"pa": 150, "pb": 250, "pc": 1004}
EXCLUDE = {"pa": SPANS["pa"] + [99], "pb": SPANS["pb"], "pc": SPANS["pc"] + [1005]}  # 99 / 1005: earlier passage ids
TYPES = {"pa": "det–noun", "pb": "det–noun", "pc": "verb–prep"}


def _word_pools() -> dict[str, dict[int, np.ndarray]]:
    """All words of a passage share one pool (as real lowercase words do), so distinct draws are tested."""
    out = {}
    for i, (name, span) in enumerate(SPANS.items()):
        pool = np.arange(5000 + 100 * i, 5000 + 100 * i + 40, dtype=np.int64)
        out[name] = {int(x): pool for x in span if x > 2}
    return out


def _t_pools() -> dict[str, np.ndarray]:
    return {name: np.arange(6000 + 100 * i, 6000 + 100 * i + 40, dtype=np.int64) for i, name in enumerate(SPANS)}


def _rows(n: int = 8) -> np.ndarray:
    """Rows from the pool minus pc's ids, then pc collisions planted in rows 0 (span id at 4) and 1 (t at the query)
    and 2 (an earlier passage id at 2)."""
    rng = np.random.default_rng(0)
    clean = POOL[~np.isin(POOL, [1000, 1001, 1002, 1003, 1004, 1005])]
    rows = np.stack([rng.choice(clean, SEQ, replace=False) for _ in range(n)]).astype(np.int64)
    rows[0, 3], rows[1, 11], rows[2, 1] = 1000, 1004, 1005
    return rows


def _orders() -> dict[str, dict[int, list[int]]]:
    return e32.shuffle_orders(SPANS, LAYOUT["shuffled_ks"], 7)


def _build(n: int = 8, seed: int = 11) -> eng.Design:
    return e32.build_e32(_rows(n), SPANS, T_IDS, EXCLUDE, _word_pools(), _t_pools(), _orders(), POOL, seed, LAYOUT)


def _check(d: eng.Design, n: int = 8) -> dict:
    eng.check_design(d, BOS_ID)
    return e32.check_e32(d, _rows(n), SPANS, T_IDS, EXCLUDE, _word_pools(), _t_pools(), _orders(), POOL, LAYOUT)


# ---- layout and builder --------------------------------------------------------------------------------------

def test_real_config_layout_gives_36_variants_and_valid_specs():
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "exp3_e32.yaml").read_text())
    names = e32.variant_names(cfg["layout"])
    assert len(names) == 36 and len(set(names)) == 36
    contrasts = e32.contrast_specs(cfg)
    e32.validate_specs(cfg["predictions"], names, contrasts)
    assert contrasts["placebo_31"] == {"a": "matched_b_31", "b": "matched_31", "mask_from": "matched_31"}
    assert {"E_1", "E_31", "order_3", "content_31", "placebo_8", "query_ctx"} <= set(contrasts)
    assert "order_2" not in contrasts


def test_bad_layout_raises():
    with pytest.raises(ValueError):
        e32.check_layout({**LAYOUT, "ks": [1, 2, 3, 5, 10]})  # must end at q - 1
    with pytest.raises(ValueError):
        e32.check_layout({**LAYOUT, "shuffled_ks": [2, 3]})
    with pytest.raises(ValueError):
        e32.check_layout({**LAYOUT, "rr": [9, 10]})


def test_build_passes_generic_and_layout_checks():
    d = _build()
    assert d.variants == e32.variant_names(LAYOUT) and d.units == tuple(SPANS)
    for s in d.seqs.values():
        assert s.shape == (8, 3, SEQ) and s.dtype == np.int64
    info = _check(d)
    assert info["redrawn"] == {"pa": 0, "pb": 0, "pc": 3}


def test_nl_spans_are_nested_at_fixed_positions():
    d = _build()
    for u, name in enumerate(d.units):
        span = np.asarray(SPANS[name])
        for k in LAYOUT["ks"]:
            nl = d.seqs[f"nl_{k}"][:, u]
            np.testing.assert_array_equal(nl[:, SEQ - 1 - k:SEQ - 1], np.tile(span[11 - k:], (8, 1)))
            np.testing.assert_array_equal(nl[:, :SEQ - 1 - k], d.base[:, u, :SEQ - 1 - k])  # filler
            np.testing.assert_array_equal(nl[:, SEQ - 1], d.base[:, u, SEQ - 1])            # random query
        assert d.meta["nl_3"].written == (9, 10, 11)


def test_repeat_groups_are_declared_per_passage():
    d = _build()
    assert d.meta["nl_11"].unit_repeats[0] == ((1, 5), (2, 9))       # pa: 101 at 1 and 5, punctuation 1 at 2 and 9
    assert d.meta["nl_11"].unit_repeats[1] == ((2, 10),)             # pb: 202
    assert d.meta["nl_3"].unit_repeats[0] == ()
    assert d.meta["matched_11"].unit_repeats == d.meta["nl_11"].unit_repeats
    assert d.meta["passage+t"].unit_repeats == d.meta["nl_11"].unit_repeats


def test_span_repeats_counts_and_distance():
    r = e32.span_repeats(SPANS["pa"])
    assert r == {"repeated_ids": 2, "extra_occurrences": 2, "min_distance": 4, "ids": [1, 101]}
    assert e32.span_repeats([5, 6, 7])["min_distance"] is None
    assert e32.span_repeats([5, 5, 5])["extra_occurrences"] == 2


def test_shuffled_keeps_the_last_token_and_is_fixed_per_passage():
    d = _build()
    orders = _orders()
    for u, name in enumerate(d.units):
        span = np.asarray(SPANS[name])
        for k in LAYOUT["shuffled_ks"]:
            sh = d.seqs[f"shuffled_{k}"][:, u, SEQ - 1 - k:SEQ - 1]
            assert (sh == sh[0]).all()                                # one order for every row
            assert (sh[:, -1] == span[-1]).all()                      # the token at q - 1 is kept
            assert not (sh[0] == span[11 - k:]).all()                 # the order changes
            np.testing.assert_array_equal(sh[0], span[11 - k:][orders[name][k]])
    other_class = e32.build_e32(_rows(), SPANS, T_IDS, EXCLUDE, _word_pools(), _t_pools(), _orders(), POOL, 99, LAYOUT)
    for k in LAYOUT["shuffled_ks"]:  # same orders under another base-class seed
        np.testing.assert_array_equal(other_class.seqs[f"shuffled_{k}"][..., SEQ - 1 - k:SEQ - 1],
                                      d.seqs[f"shuffled_{k}"][..., SEQ - 1 - k:SEQ - 1])


def test_shuffle_order_without_a_changing_order_raises():
    with pytest.raises(ValueError):
        e32.shuffle_order([5, 5, 7], 0)


def test_matched_keeps_punctuation_and_the_repeat_pattern_and_is_nested():
    d = _build()
    pools = _word_pools()
    rows = _rows()
    m11 = d.seqs["matched_11"][:, 0, :11]  # pa
    span = np.asarray(SPANS["pa"])
    np.testing.assert_array_equal(m11[:, [1, 5, 8]], np.tile([1, 2, 1], (8, 1)))  # punctuation in place
    np.testing.assert_array_equal(m11[:, 0], m11[:, 4])                         # 101 -> one draw, used twice
    words = [0, 2, 3, 6, 7, 9, 10]
    assert all(len(set(m11[i, words].tolist())) == len(words) for i in range(8))  # distinct words, distinct draws
    assert np.isin(m11[:, words], pools["pa"][101]).all()
    assert not any(np.isin(m11[i], rows[i]).any() for i in range(8))
    for k in LAYOUT["ks"]:
        np.testing.assert_array_equal(d.seqs[f"matched_{k}"][:, 0, 11 - k:11], m11[:, 11 - k:])
    mb = d.seqs["matched_b_11"][:, 0, :11]
    assert (mb[:, [1, 5, 8]] == [1, 2, 1]).all() and (mb[:, 0] == mb[:, 4]).all()
    assert not np.array_equal(mb, m11)
    np.testing.assert_array_equal(d.seqs["matched_b_5"][:, 0, 6:11], mb[:, 6:])


def test_per_passage_exclusion_redraws_only_the_collisions():
    d = _build()
    rows = _rows()
    np.testing.assert_array_equal(d.base[:, 0], rows)  # pa's ids lie outside the base pool
    np.testing.assert_array_equal(d.base[:, 1], rows)
    pc = d.base[:, 2]
    changed = np.argwhere(pc != rows)
    assert sorted(map(tuple, changed.tolist())) == [(0, 3), (1, 11), (2, 1)]
    assert not np.isin(pc, EXCLUDE["pc"] + [T_IDS["pc"]]).any()
    assert all(len(set(r.tolist())) == SEQ for r in pc)
    assert np.isin(pc, POOL).all()


def test_passage_rr_and_the_2x2():
    d = _build()
    s = d.seqs
    for u, name in enumerate(d.units):
        span, t = np.asarray(SPANS[name]), T_IDS[name]
        rr = s["passage-rr"][:, u]
        np.testing.assert_array_equal(rr[:, :9], np.tile(span[:9], (8, 1)))
        np.testing.assert_array_equal(rr[:, 9:], d.base[:, u, 9:])           # 10, 11 and the query: the row's own
        np.testing.assert_array_equal(s["passage+t"][:, u, :11], np.tile(span, (8, 1)))
        assert (s["passage+t"][:, u, 11] == t).all() and (s["none+t"][:, u, 11] == t).all()
        np.testing.assert_array_equal(s["none+t"][:, u, :11], d.base[:, u, :11])
        w = s["passage+w"][:, u, 11]
        np.testing.assert_array_equal(s["none+w"][:, u, 11], w)
        assert np.isin(w, _t_pools()[name]).all() and len(set(w.tolist())) > 1  # drawn per row
    assert d.meta["passage-rr"].written == tuple(range(1, 10))
    assert d.meta["passage+t"].written == tuple(range(1, 13))
    assert d.meta["none+w"].written == (12,)


def test_build_is_deterministic_and_row_stable():
    a, b = _build(), _build()
    small = _build(n=4)
    for v in a.variants:
        np.testing.assert_array_equal(a.seqs[v], b.seqs[v])
        np.testing.assert_array_equal(small.seqs[v], a.seqs[v][:4])
    assert not np.array_equal(_build(seed=12).seqs["matched_11"], a.seqs["matched_11"])


def test_checks_catch_tampering():
    d = _build()
    d.seqs["matched_5"][0, 0, 7] = 1  # a word slot set to a punctuation id
    with pytest.raises(AssertionError):
        _check(d)
    d = _build()
    d.seqs["matched_11"][3, 0, 4] = d.seqs["matched_11"][3, 0, 2]  # breaks 101's one-draw rule
    with pytest.raises(AssertionError):
        _check(d)
    d = _build()
    d.seqs["nl_3"][2, 1, 0] = SPANS["pb"][-1]  # filler holds a written id
    with pytest.raises(AssertionError):
        eng.check_design(d, BOS_ID)
    d = _build()
    d.seqs["passage+w"][1, 2, 11] = T_IDS["pc"]  # w := t
    with pytest.raises(AssertionError):
        e32.check_e32(d, _rows(), SPANS, T_IDS, EXCLUDE, _word_pools(), _t_pools(), _orders(), POOL, LAYOUT)


# ---- sweep helpers -------------------------------------------------------------------------------------------

def test_switch_lengths_need_every_larger_k():
    ks = [1, 2, 4, 8]
    z = np.zeros((2, 2), dtype=bool)
    sw = {1: z.copy(), 2: z.copy(), 4: z.copy(), 8: z.copy()}
    sw[8][0, 0] = sw[4][0, 0] = sw[2][0, 0] = True   # switched from 2 on
    sw[8][0, 1] = sw[2][0, 1] = sw[1][0, 1] = True   # not at 4, so only from 8
    sw[4][1, 0] = True                                # not at the largest k: none
    out = e32.switch_lengths(ks, sw)
    assert out[0, 0] == 2 and out[0, 1] == 8 and np.isnan(out[1, 0]) and np.isnan(out[1, 1])


def test_fit_saturation_recovers_a_planted_curve():
    ks = [1, 2, 3, 4, 6, 8, 12, 16, 24, 31]
    rng = np.random.default_rng(0)
    e_rows = e32.saturation(np.array(ks), 0.02, 4.0)[None, :] + rng.normal(0, 0.002, (300, len(ks)))
    fit = e32.fit_saturation(ks, e_rows, {"iters": 200, "seed": 0},
                             {"kappa_bounds": [0.05, 1000.0], "e_inf_bounds": [-1.0, 1.0], "kappa_start": 3.0})
    assert abs(fit["kappa"] - 4.0) < 0.5 and abs(fit["e_inf"] - 0.02) < 0.001
    assert fit["kappa_ci"][0] < 4.0 < fit["kappa_ci"][1] and fit["n_failed"] == 0


def test_type_sweep_groups_passages():
    unit_e = np.array([[0.1, 0.2], [0.3, 0.4], [1.0, 2.0]])
    out = e32.type_sweep(unit_e, ("pa", "pb", "pc"), TYPES)
    assert out["det–noun"]["passages"] == ["pa", "pb"]
    np.testing.assert_allclose(out["det–noun"]["E"], [0.2, 0.3])
    np.testing.assert_allclose(out["verb–prep"]["E"], [1.0, 2.0])


# ---- analysis on synthetic attention -------------------------------------------------------------------------

CFG = {
    "diffuse_threshold": 0.2,
    "layout": LAYOUT,
    "contrasts": {"rr": {"a": "passage-rr", "b": "none"}, "final_pair": {"a": "nl_11", "b": "passage-rr"},
                  "ctx_t": {"a": "passage+t", "b": "none+t"}, "ctx_w": {"a": "passage+w", "b": "none+w"},
                  "query_ctx": {"a": "passage+t", "b": "passage+w"}, "query_none": {"a": "none+t", "b": "none+w"}},
    "analysis": {"late_layers": [6, 11], "margin": 0.005, "bootstrap": {"iters": 200, "seed": 0},
                 "meaningful": {"q": 0.05, "min_abs": 0.02}, "switch": {"q": 0.05, "min_shift": 0.15},
                 "saturation": {"kappa_bounds": [0.05, 1000.0], "e_inf_bounds": [-1.0, 1.0], "kappa_start": 3.0}},
}
SPECS = [
    {"id": "E2a", "parts": [{"name": "E(3)", "test": "positive", "combo": {"nl_3": 1, "none": -1}},
                            {"name": "E(5)-E(3)", "test": "equivalent", "combo": {"nl_5": 1, "nl_3": -1}},
                            {"name": "E(11)-E(3)", "test": "equivalent", "combo": {"nl_11": 1, "nl_3": -1}}]},
    {"id": "E2b", "parts": [{"name": f"k={k}", "test": "positive", "combo": {f"nl_{k}": 1, f"shuffled_{k}": -1}}
                            for k in LAYOUT["shuffled_ks"]]},
    {"id": "E2c", "parts": [{"name": "interaction", "test": "negative",
                             "combo": {"passage+t": 1, "none+t": -1, "passage+w": -1, "none+w": 1}},
                            {"name": "switches", "test": "switch_count", "contrast": "query_ctx", "labels": "argmax",
                             "from": "Sink", "to": "Span", "layers": [6, 11], "min_beyond_placebo": 5,
                             "placebo": ["placebo_5", "placebo_11"]}]},
    {"id": "E2d", "parts": [{"name": "E2d", "test": "positive", "combo": {"nl_11": 1, "passage-rr": -2, "none": 1}}]},
]
E2C_HEADS = [(6, 0), (6, 1), (7, 2), (8, 3), (9, 4), (10, 5)]


def _focus(att: np.ndarray, key: int) -> None:
    """Set one head's attention: BOS 0.1, `key` 0.6, the rest spread over the other non-BOS keys."""
    att[...] = 0.3 / (att.shape[-1] - 2)
    att[..., 0], att[..., key] = 0.1, 0.6


def _synthetic_extraction(d: eng.Design, rows: int = 40) -> eng.Extraction:
    """Every head parks on BOS (0.7). Late heads add +0.03 BOS in nl (k >= 2; +0.01 at k = 1), +0.02 in shuffled,
    +0.005 in passage-rr and +0.03 in both passage cells. In passage+t six late heads move to a Span key (5). L3H4
    moves to key 8 in nl_5 and nl_11 only (switch length 5). Small noise; rows sum to 1."""
    rng = np.random.default_rng(0)
    V, K = len(d.variants), SEQ + 1
    vi = {v: i for i, v in enumerate(d.variants)}
    att = np.zeros((rows, V, 12, 12, K))
    att[..., 0] = 0.7
    att[..., 1:] = 0.3 / (K - 1)

    def bos(v: str, delta: float) -> None:
        att[:, vi[v], 6:12, :, 0] += delta
        att[:, vi[v], 6:12, :, 1:] -= delta / (K - 1)

    for k in LAYOUT["ks"]:
        bos(f"nl_{k}", 0.01 if k == 1 else 0.03)
    for k in LAYOUT["shuffled_ks"]:
        bos(f"shuffled_{k}", 0.02)
    bos("passage-rr", 0.005)
    bos("passage+t", 0.03)
    bos("passage+w", 0.03)
    for layer, head in E2C_HEADS:
        _focus(att[:, vi["passage+t"], layer, head], 5)
    for v in ("nl_5", "nl_11"):
        _focus(att[:, vi[v], 3, 4], 8)
    att += rng.uniform(0, 0.002, att.shape)
    att /= att.sum(-1, keepdims=True)
    late = att[:, :, 6:12].mean(axis=(2, 3))
    U = len(d.units)
    return eng.Extraction(d.variants, d.units, np.full(V, K), (6, 11), att.astype(np.float32),
                          np.repeat(late[:, None], U, axis=1).astype(np.float32),
                          np.repeat(att.mean(axis=0)[None], U, axis=0).astype(np.float32), 0.0, 0, 1.0)


def test_analyse_on_synthetic_attention():
    d = _build(n=40)
    contrasts = e32.contrast_specs(CFG)
    e32.validate_specs(SPECS, d.variants, contrasts)
    res = e32.analyse(_synthetic_extraction(d), d, CFG, contrasts, SPECS, TYPES)
    assert res["placebo_switching"]["placebo_5"]["argmax"] == []
    assert res["placebo_switching"]["placebo_11"]["argmax"] == []
    p = res["predictions"]
    assert all(p[i]["ok"] for i in ("E2a", "E2b", "E2c", "E2d")), {i: p[i]["detail"] for i in p}
    assert p["E2c"]["stats"]["switches"]["count"] == 6 and p["E2c"]["stats"]["interaction"]["mean"] < -0.04
    assert abs(p["E2d"]["stats"]["E2d"]["mean"] - 0.02) < 0.002
    t = res["transitions"][("query_ctx", "argmax")]
    assert t[eng.LABELS.index("Sink"), eng.LABELS.index("Span")] == 6
    sl = res["switch_length"]["argmax"]
    assert sl[3, 4] == 5 and np.isnan(np.delete(sl.ravel(), 3 * 12 + 4)).all()
    assert res["switch_dest"]["argmax"] == {"L3H4": "Sink→Span"}
    e = {r["k"]: r["mean"] for r in res["sweep"] if r["series"] == "E"}
    assert abs(e[1] - 0.01) < 0.001 and all(abs(e[k] - 0.03) < 0.001 for k in (2, 3, 5, 11))
    assert {r["k"] for r in res["sweep"] if r["series"] == "order"} == {3, 5, 11}
    assert res["head_curves"].shape == (5, 12, 12) and res["e_rows"].shape == (40, 5)
    assert np.isfinite(res["saturation"]["kappa"]) and res["types"]["det–noun"]["passages"] == ["pa", "pb"]
    preds = e32.check_predictions(SPECS, {"rare": res, "common": res})
    assert [x["verdict"] for x in preds] == ["PASS"] * 4


def test_switch_count_counts_beyond_the_larger_placebo():
    d = _build(n=40)
    specs = [{**SPECS[2], "parts": [{**SPECS[2]["parts"][1], "min_beyond_placebo": 7}]}]
    res = e32.analyse(_synthetic_extraction(d), d, CFG, e32.contrast_specs(CFG), specs, TYPES)
    r = res["predictions"]["E2c"]
    assert not r["ok"] and r["stats"]["switches"]["beyond"] == 6
    one = e32.check_predictions(specs, {"rare": res, "common": res})[0]
    assert one["verdict"] == "FAIL" and one["per_base"] == {"rare": False, "common": False}
