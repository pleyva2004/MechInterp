"""Tests for the exp2 iteration-2 additions: build_custom / check_custom, additivity, surprisal slopes, new prediction kinds."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from exp2.analysis import (
    additivity_tables,
    check_prediction,
    contrast_table,
    interactions,
    surprisal_slopes,
    tv_table,
)
from exp2.design import build_custom, check_custom

SEQ_LEN = 32
BOS = 50256
BOOT = {"iters": 200, "seed": 0, "ci": 0.95}
DEPTH = {"early": list(range(6)), "late": list(range(6, 12))}
N, L, H = 60, 12, 12
METRIC = "sink_mass"
MINUS = "−"

COMBOS = {
    "nl": {"phrase": "sushi", "placement": "near", "parts": ["I@29", "like@30", "sushi@31"]},
    "shuffled": {"phrase": "sushi", "placement": "near", "parts": ["like@29", "I@30", "sushi@31"]},
}
DIFFS = [["nl", "shuffled"]]
PARTS = ["I@29", "like@30", "sushi@31", "like@29", "I@30"]


# ---- build_custom / check_custom ----------------------------------------------------------------------------

@pytest.fixture
def bases():
    rng = np.random.default_rng(0)
    return np.stack([rng.permutation(np.arange(1000, 5000))[:SEQ_LEN] for _ in range(20)]).astype(np.int64)


WRITES = [[29, 314], [31, 36324]]


def test_build_custom_writes_at_positions_and_keeps_rest(bases):
    orig = bases.copy()
    seqs = build_custom(bases, WRITES, SEQ_LEN)
    assert (bases == orig).all()  # input untouched
    assert seqs.shape == bases.shape
    assert (seqs[:, 28] == 314).all() and (seqs[:, 30] == 36324).all()  # array index = position - 1
    keep = np.ones(SEQ_LEN, dtype=bool)
    keep[[28, 30]] = False
    assert (seqs[:, keep] == bases[:, keep]).all()
    assert (seqs[:, 29] == bases[:, 29]).all()


def test_build_custom_single_write(bases):
    seqs = build_custom(bases, [[1, 588]], SEQ_LEN)
    assert (seqs[:, 0] == 588).all() and (seqs[:, 1:] == bases[:, 1:]).all()


def test_build_custom_rejects_query_and_bos_positions(bases):
    with pytest.raises(ValueError):
        build_custom(bases, [[SEQ_LEN, 5]], SEQ_LEN)
    with pytest.raises(ValueError):
        build_custom(bases, [[0, 5]], SEQ_LEN)


def test_check_custom_passes_on_built(bases):
    check_custom(build_custom(bases, WRITES, SEQ_LEN), bases, WRITES, SEQ_LEN, BOS)


def test_check_custom_rejects_wrong_id(bases):
    seqs = build_custom(bases, WRITES, SEQ_LEN)
    seqs[3, 28] = 7777
    with pytest.raises(AssertionError, match="missing"):
        check_custom(seqs, bases, WRITES, SEQ_LEN, BOS)


def test_check_custom_rejects_changed_non_written_token(bases):
    seqs = build_custom(bases, WRITES, SEQ_LEN)
    seqs[5, 0] = 7777
    with pytest.raises(AssertionError, match="outside"):
        check_custom(seqs, bases, WRITES, SEQ_LEN, BOS)


def test_check_custom_rejects_repeat(bases):
    bad = bases.copy()
    bad[2, 10] = bad[2, 11]  # a repeat already in the base row, nothing else differs
    seqs = build_custom(bad, WRITES, SEQ_LEN)
    with pytest.raises(AssertionError, match="repeats"):
        check_custom(seqs, bad, WRITES, SEQ_LEN, BOS)


def test_check_custom_rejects_written_id_repeating_base_token(bases):
    writes = [[29, int(bases[0, 5])]]  # row 0 now holds that id twice
    seqs = build_custom(bases, writes, SEQ_LEN)
    with pytest.raises(AssertionError, match="repeats"):
        check_custom(seqs, bases, writes, SEQ_LEN, BOS)


def test_check_custom_rejects_bos(bases):
    writes = [[29, BOS]]
    with pytest.raises(AssertionError):
        check_custom(build_custom(bases, writes, SEQ_LEN), bases, writes, SEQ_LEN, BOS)


# ---- synthetic additive cells --------------------------------------------------------------------------------

def make_additive_cells(extra=None, bases=("rare", "common"), seed=0):
    """Every variant = none + sum of planted per-part offsets; `extra` [L, H] is added to the nl combo only."""
    rng = np.random.default_rng(seed)
    cells = {}
    for base in bases:
        offs = {p: rng.normal(size=(L, H)) * 0.05 for p in PARTS}
        none = rng.normal(size=(N, L, H)) * 0.1 + 0.3
        v = {"none": none}
        for p in PARTS:
            v[p] = none + offs[p]
        for combo, spec in COMBOS.items():
            v[combo] = none + sum(offs[p] for p in spec["parts"])
        if extra is not None:
            v["nl"] = v["nl"] + extra
        for name, arr in v.items():
            cells[(base, "sushi", "near", name)] = {METRIC: arr}
    return cells


def test_interactions_zero_under_exact_additivity():
    inter = interactions(make_additive_cells(), COMBOS, METRIC)
    assert set(inter) == {(b, "sushi", "near", c) for b in ("rare", "common") for c in COMBOS}
    for x in inter.values():
        assert x.shape == (N, L, H)
        np.testing.assert_allclose(x, 0.0, atol=1e-12)


def test_interactions_recover_planted_combo_only_offset():
    extra = np.random.default_rng(5).normal(size=(L, H)) * 0.2
    inter = interactions(make_additive_cells(extra=extra), COMBOS, METRIC)
    for base in ("rare", "common"):
        np.testing.assert_allclose(inter[(base, "sushi", "near", "nl")], np.broadcast_to(extra, (N, L, H)), atol=1e-12)
        np.testing.assert_allclose(inter[(base, "sushi", "near", "shuffled")], 0.0, atol=1e-12)


# ---- additivity_tables ---------------------------------------------------------------------------------------

def test_additivity_per_head_observed_minus_additive_is_interaction():
    extra = np.random.default_rng(5).normal(size=(L, H)) * 0.2
    cells = make_additive_cells(extra=extra)
    per_head, _, _ = additivity_tables(cells, COMBOS, DIFFS, DEPTH, METRIC, BOOT)
    assert len(per_head) == 2 * len(COMBOS) * L * H
    np.testing.assert_allclose(per_head["observed"] - per_head["additive"], per_head["interaction"], atol=1e-12)
    assert (per_head["name"] == "L" + per_head["layer"].astype(str) + "H" + per_head["head"].astype(str)).all()
    nl = per_head[(per_head["combo"] == "nl") & (per_head["base"] == "rare")].sort_values(["layer", "head"])
    np.testing.assert_allclose(nl["interaction"].to_numpy(), extra.ravel(), atol=1e-12)
    sh = per_head[per_head["combo"] == "shuffled"]
    np.testing.assert_allclose(sh["interaction"], 0.0, atol=1e-12)
    # observed is the mean over rows of (combo - none)
    r = nl.iloc[0]
    k = ("rare", "sushi", "near")
    expect = (cells[(*k, "nl")][METRIC] - cells[(*k, "none")][METRIC])[:, r["layer"], r["head"]].mean()
    assert r["observed"] == pytest.approx(expect)


def test_additivity_per_head_ci_brackets_mean_with_row_noise():
    rng = np.random.default_rng(3)
    cells = make_additive_cells()
    for k, v in cells.items():
        if k[3] == "nl":
            v[METRIC] = v[METRIC] + 0.1 + rng.normal(size=(N, L, H)) * 0.05  # planted +0.1 interaction, noisy
    per_head, _, _ = additivity_tables(cells, COMBOS, DIFFS, DEPTH, METRIC, BOOT)
    assert (per_head["lo"] <= per_head["interaction"] + 1e-12).all()
    assert (per_head["interaction"] <= per_head["hi"] + 1e-12).all()
    nl = per_head[per_head["combo"] == "nl"]
    assert (nl["lo"] > 0).mean() > 0.9 and (nl["interaction"].mean() == pytest.approx(0.1, abs=0.01))
    assert (nl["p"] < 0.05).mean() > 0.9


def test_additivity_depth_rows_and_difference_term():
    extra = np.zeros((L, H))
    extra[6:] = 0.15  # only late layers interact, only in nl
    rng = np.random.default_rng(1)
    cells = make_additive_cells(extra=extra)
    for k, v in cells.items():
        if k[3] == "nl":
            v[METRIC] = v[METRIC] + rng.normal(size=(N, 1, 1)) * 0.02  # row-level jitter -> non-degenerate CI
    _, depth, _ = additivity_tables(cells, COMBOS, DIFFS, DEPTH, METRIC, BOOT)
    terms = {"nl", "shuffled", f"nl{MINUS}shuffled"}
    assert set(depth["term"]) == terms
    assert len(depth) == 2 * len(terms) * len(DEPTH)  # each base x term x depth group
    for base in ("rare", "common"):
        for term in terms:
            assert set(depth[(depth["base"] == base) & (depth["term"] == term)]["depth"]) == set(DEPTH)
    assert (depth["lo"] <= depth["mean"] + 1e-12).all() and (depth["mean"] <= depth["hi"] + 1e-12).all()
    get = lambda term, d: depth[(depth["base"] == "rare") & (depth["term"] == term) & (depth["depth"] == d)].iloc[0]
    assert get("nl", "late")["mean"] == pytest.approx(0.15, abs=0.01)
    assert get("nl", "early")["mean"] == pytest.approx(0.0, abs=0.01)
    assert get(f"nl{MINUS}shuffled", "late")["mean"] == pytest.approx(0.15, abs=0.01)
    assert get("shuffled", "late")["mean"] == pytest.approx(0.0, abs=1e-12)
    assert get(f"nl{MINUS}shuffled", "late")["lo"] > 0
    assert get("nl", "late")["lo"] < get("nl", "late")["hi"]


def test_additivity_depth_difference_is_a_minus_b_of_means():
    rng = np.random.default_rng(2)
    extra_nl, extra_sh = rng.normal(size=(L, H)) * 0.1, rng.normal(size=(L, H)) * 0.1
    cells = make_additive_cells(extra=extra_nl)
    for k, v in cells.items():
        if k[3] == "shuffled":
            v[METRIC] = v[METRIC] + extra_sh
    _, depth, _ = additivity_tables(cells, COMBOS, DIFFS, DEPTH, METRIC, BOOT)
    for d, layers in DEPTH.items():
        row = depth[(depth["base"] == "rare") & (depth["term"] == f"nl{MINUS}shuffled") & (depth["depth"] == d)].iloc[0]
        assert row["mean"] == pytest.approx((extra_nl - extra_sh)[layers].mean(), abs=1e-12)


def test_order_fit_perfect_when_exactly_additive():
    cells = make_additive_cells()
    _, _, fit = additivity_tables(cells, COMBOS, DIFFS, DEPTH, METRIC, BOOT)
    assert len(fit) == 2 and set(fit["pair"]) == {f"nl{MINUS}shuffled"}
    np.testing.assert_allclose(fit["r2"], 1.0, atol=1e-9)
    np.testing.assert_allclose(fit["slope"], 1.0, atol=1e-9)
    np.testing.assert_allclose(fit["r"], 1.0, atol=1e-9)
    np.testing.assert_allclose(fit["mean_abs_interaction"], 0.0, atol=1e-12)
    assert (fit["mean_abs_observed"] > 0).all()


def test_order_fit_r2_drops_with_planted_interaction():
    extra = np.random.default_rng(9).normal(size=(L, H)) * 0.2  # larger than the additive spread (~0.07)
    _, _, fit = additivity_tables(make_additive_cells(extra=extra), COMBOS, DIFFS, DEPTH, METRIC, BOOT)
    assert (fit["r2"] < 0.9).all()
    np.testing.assert_allclose(fit["mean_abs_interaction"], np.abs(extra).mean(), atol=1e-9)


# ---- surprisal_slopes ----------------------------------------------------------------------------------------

VARS = ["none", "nl", "shuffled", "matched", "I@29", "like@30"]


def make_surprisal(slope, noise=0.0, seed=0, row_scale=1.0, bases=("rare",)):
    """metric = a + b_row + slope*nll (+ noise); nll = row offset + variant offset + jitter; b_row is correlated with the
    row's nll offset, which would bias a pooled regression but not the within-row one."""
    rng = np.random.default_rng(seed)
    slope = np.broadcast_to(np.asarray(slope, dtype=float), (L, H))
    cells, nll = {}, {}
    for base in bases:
        row_off = rng.normal(size=N) * 4.0 * row_scale
        b_row = 3.0 * row_off[:, None, None] + rng.normal(size=(N, 1, 1)) * row_scale
        var_off = rng.uniform(5, 20, size=len(VARS))
        for v, vo in zip(VARS, var_off):
            x = 10.0 + row_off + vo + rng.normal(size=N) * 1.0
            y = 0.5 + b_row + slope * x[:, None, None] + rng.normal(size=(N, L, H)) * noise
            nll[(base, "sushi", "near", v)] = x
            cells[(base, "sushi", "near", v)] = {METRIC: y}
    return cells, nll


def test_surprisal_slopes_exact_when_noiseless_and_row_offsets_dont_bias():
    planted = np.random.default_rng(4).uniform(-0.02, 0.0, size=(L, H))
    cells, nll = make_surprisal(planted)
    per_head, depth, _ = surprisal_slopes(cells, nll, METRIC, DEPTH, BOOT)
    ph = per_head.sort_values(["layer", "head"])
    np.testing.assert_allclose(ph["slope"].to_numpy(), planted.ravel(), atol=1e-9)
    assert (ph["lo"] <= planted.ravel() + 1e-9).all() and (planted.ravel() <= ph["hi"] + 1e-9).all()
    for d, layers in DEPTH.items():
        row = depth[depth["depth"] == d].iloc[0]
        assert row["slope"] == pytest.approx(planted[layers].mean(), abs=1e-9)
        assert row["lo"] <= row["slope"] + 1e-9 <= row["hi"] + 2e-9


def test_surprisal_slopes_recover_planted_with_noise():
    cells, nll = make_surprisal(-0.01, noise=0.05, seed=1)
    per_head, depth, _ = surprisal_slopes(cells, nll, METRIC, DEPTH, BOOT)
    assert len(per_head) == L * H and len(depth) == len(DEPTH)
    np.testing.assert_allclose(per_head["slope"], -0.01, atol=2e-3)
    covered = ((per_head["lo"] <= -0.01) & (-0.01 <= per_head["hi"])).mean()
    assert covered >= 0.85
    assert (per_head["lo"] <= per_head["slope"]).all() and (per_head["slope"] <= per_head["hi"]).all()
    for _, r in depth.iterrows():
        assert r["lo"] <= -0.01 <= r["hi"]
        assert r["lo"] < r["slope"] < r["hi"]


def test_surprisal_slopes_invariant_to_row_offsets():
    # the same within-row structure under wildly different between-row offsets gives the same slope
    cells0, nll0 = make_surprisal(-0.01, noise=0.0, seed=2, row_scale=0.0)
    cells1, nll1 = make_surprisal(-0.01, noise=0.0, seed=2, row_scale=10.0)
    s0 = surprisal_slopes(cells0, nll0, METRIC, DEPTH, BOOT)[0]["slope"].to_numpy()
    s1 = surprisal_slopes(cells1, nll1, METRIC, DEPTH, BOOT)[0]["slope"].to_numpy()
    np.testing.assert_allclose(s0, -0.01, atol=1e-9)
    np.testing.assert_allclose(s1, -0.01, atol=1e-9)

    # shifting every variant of one row by a constant (in nll and in metric) changes nothing
    cells, nll = make_surprisal(-0.01, noise=0.05, seed=3)
    base = surprisal_slopes(cells, nll, METRIC, DEPTH, BOOT)[0]["slope"].to_numpy()
    shifted_c = {k: {METRIC: v[METRIC].copy()} for k, v in cells.items()}
    shifted_n = {k: v.copy() for k, v in nll.items()}
    for k in nll:
        shifted_n[k][7] += 100.0
        shifted_c[k][METRIC][7] += 50.0
    after = surprisal_slopes(shifted_c, shifted_n, METRIC, DEPTH, BOOT)[0]["slope"].to_numpy()
    np.testing.assert_allclose(after, base, atol=1e-9)


def test_surprisal_slopes_depth_groups_differ():
    planted = np.zeros((L, H))
    planted[6:] = -0.02
    cells, nll = make_surprisal(planted, noise=0.02, seed=5)
    _, depth, _ = surprisal_slopes(cells, nll, METRIC, DEPTH, BOOT)
    get = lambda d: depth[depth["depth"] == d].iloc[0]
    assert get("late")["slope"] == pytest.approx(-0.02, abs=2e-3) and get("late")["hi"] < 0
    assert abs(get("early")["slope"]) < 2e-3 and get("early")["lo"] <= 0 <= get("early")["hi"]


def test_surprisal_variants_table_one_row_per_variant():
    cells, nll = make_surprisal(-0.01, noise=0.01, seed=6, bases=("rare", "common"))
    _, _, var = surprisal_slopes(cells, nll, METRIC, DEPTH, BOOT)
    assert len(var) == 2 * len(VARS)
    assert not var.duplicated(["base", "variant"]).any()
    assert {f"mean_{METRIC}_early", f"mean_{METRIC}_late", "mean_slot_nll"} <= set(var.columns)
    for _, r in var.iterrows():
        k = (r["base"], r["phrase"], r["placement"], r["variant"])
        assert r["mean_slot_nll"] == pytest.approx(nll[k].mean())
        assert r[f"mean_{METRIC}_late"] == pytest.approx(cells[k][METRIC][:, 6:, :].mean())
        assert r[f"mean_{METRIC}_early"] == pytest.approx(cells[k][METRIC][:, :6, :].mean())


def test_surprisal_slopes_one_group_per_base():
    cells, nll = make_surprisal(-0.01, noise=0.01, seed=7, bases=("rare", "common"))
    per_head, depth, _ = surprisal_slopes(cells, nll, METRIC, DEPTH, BOOT)
    assert set(per_head["base"]) == {"rare", "common"} and len(per_head) == 2 * L * H
    assert len(depth) == 2 * len(DEPTH)


# ---- check_prediction: the four new kinds -------------------------------------------------------------------

def depth_df(rows):
    return pd.DataFrame(rows, columns=["base", "phrase", "placement", "term", "depth", "mean", "lo", "hi"])


def surp_df(rows):
    return pd.DataFrame(rows, columns=["base", "phrase", "placement", "depth", "slope", "lo", "hi"])


def fit_df(rows):
    return pd.DataFrame(rows, columns=["base", "phrase", "placement", "pair", "slope", "r", "r2",
                                       "mean_abs_observed", "mean_abs_interaction"])


def run(spec, **extra):
    return check_prediction(spec, pd.DataFrame(), pd.DataFrame(), extra)


def test_check_depth_term_sign_pass_fail_and_sign():
    pair = f"nl{MINUS}shuffled"
    ad = depth_df([
        ("rare", "p", "near", pair, "late", 0.05, 0.02, 0.08), ("common", "p", "near", pair, "late", 0.04, 0.01, 0.07),
        ("rare", "p", "near", pair, "early", -0.05, -0.08, -0.02), ("rare", "p", "near", "nl", "late", 0.0, -0.01, 0.01),
    ])
    spec = {"id": "Q2", "kind": "depth_term_sign", "term": pair, "depth": "late", "sign": "+"}
    r = run(spec, additivity_depth=ad)
    assert r["verdict"] == "PASS" and r["id"] == "Q2" and r["kind"] == "depth_term_sign"
    assert "rare" in r["detail"] and "common" in r["detail"]
    assert run(spec | {"depth": "early", "sign": "-"}, additivity_depth=ad)["verdict"] == "PASS"
    assert run(spec | {"depth": "early"}, additivity_depth=ad)["verdict"] == "FAIL"   # wrong sign
    assert run(spec | {"term": "nl"}, additivity_depth=ad)["verdict"] == "FAIL"       # CI straddles 0
    ad_bad = ad.copy()
    ad_bad.loc[1, "lo"] = -0.01                                                       # one base not excluding 0
    assert run(spec, additivity_depth=ad_bad)["verdict"] == "FAIL"
    assert run(spec | {"bases": ["rare"]}, additivity_depth=ad_bad)["verdict"] == "PASS"


def test_check_surprisal_sign_pass_fail_and_bases():
    sd = surp_df([
        ("rare", "p", "near", "late", -0.02, -0.03, -0.01), ("common", "p", "near", "late", -0.015, -0.025, -0.005),
        ("rare", "p", "near", "early", -0.001, -0.004, 0.002), ("common", "p", "near", "early", -0.001, -0.004, 0.002),
    ])
    spec = {"id": "Q4", "kind": "surprisal_sign", "depth": "late", "sign": "-"}
    r = run(spec, surprisal_depth=sd)
    assert r["verdict"] == "PASS" and "per nat" in r["detail"]
    assert run(spec | {"sign": "+"}, surprisal_depth=sd)["verdict"] == "FAIL"
    assert run(spec | {"depth": "early"}, surprisal_depth=sd)["verdict"] == "FAIL"
    sd_bad = sd.copy()
    sd_bad.loc[1, "hi"] = 0.001
    assert run(spec, surprisal_depth=sd_bad)["verdict"] == "FAIL"
    assert run(spec | {"bases": ["rare"]}, surprisal_depth=sd_bad)["verdict"] == "PASS"


def test_check_surprisal_depth_ratio_pass_fail_and_bases():
    sd = surp_df([
        ("rare", "p", "near", "late", -0.02, -0.03, -0.01), ("rare", "p", "near", "early", -0.005, -0.01, 0.0),
        ("common", "p", "near", "late", -0.03, -0.04, -0.02), ("common", "p", "near", "early", 0.004, 0.0, 0.008),
    ])
    spec = {"id": "Q5", "kind": "surprisal_depth_ratio", "larger": "late", "smaller": "early"}
    assert run(spec, surprisal_depth=sd)["verdict"] == "PASS"
    assert run(spec | {"larger": "early", "smaller": "late"}, surprisal_depth=sd)["verdict"] == "FAIL"
    sd_bad = sd.copy()
    sd_bad.loc[3, "slope"] = 0.05  # |early| > |late| in common only
    assert run(spec, surprisal_depth=sd_bad)["verdict"] == "FAIL"
    assert run(spec | {"bases": ["rare"]}, surprisal_depth=sd_bad)["verdict"] == "PASS"


def test_check_order_fit_r2_below_pass_fail_and_bases():
    pair = f"nl{MINUS}shuffled"
    fd = fit_df([
        ("rare", "p", "near", pair, 0.7, 0.5, 0.25, 0.1, 0.08), ("common", "p", "near", pair, 0.4, 0.4, 0.16, 0.1, 0.09),
        ("rare", "p", "near", "other", 1.0, 1.0, 1.0, 0.1, 0.0),
    ])
    spec = {"id": "Q3", "kind": "order_fit_r2_below", "pair": pair, "max_r2": 0.5}
    r = run(spec, order_fit=fd)
    assert r["verdict"] == "PASS" and "r²" in r["detail"]
    assert run(spec | {"max_r2": 0.2}, order_fit=fd)["verdict"] == "FAIL"   # rare 0.25 not below 0.2
    assert run(spec | {"max_r2": 0.2, "bases": ["common"]}, order_fit=fd)["verdict"] == "PASS"
    assert run(spec | {"pair": "other"}, order_fit=fd)["verdict"] == "FAIL"  # r2 = 1


def test_check_new_kinds_end_to_end_with_real_tables():
    extra = np.zeros((L, H))
    extra[6:] = 0.15
    cells = make_additive_cells(extra=extra)
    rng = np.random.default_rng(1)
    for k, v in cells.items():
        if k[3] == "nl":
            v[METRIC] = v[METRIC] + rng.normal(size=(N, 1, 1)) * 0.02
    _, depth, fit = additivity_tables(cells, COMBOS, DIFFS, DEPTH, METRIC, BOOT)
    scells, nll = make_surprisal(np.where(np.arange(L)[:, None] >= 6, -0.02, 0.0) * np.ones((L, H)), noise=0.02,
                                 bases=("rare", "common"))
    _, sdepth, _ = surprisal_slopes(scells, nll, METRIC, DEPTH, BOOT)
    ex = {"additivity_depth": depth, "order_fit": fit, "surprisal_depth": sdepth}
    assert run({"id": "a", "kind": "depth_term_sign", "term": "nl", "depth": "late", "sign": "+"}, **ex)["verdict"] == "PASS"
    assert run({"id": "b", "kind": "surprisal_sign", "depth": "late", "sign": "-"}, **ex)["verdict"] == "PASS"
    assert run({"id": "c", "kind": "surprisal_depth_ratio", "larger": "late", "smaller": "early"}, **ex)["verdict"] == "PASS"
    assert run({"id": "d", "kind": "order_fit_r2_below", "pair": f"nl{MINUS}shuffled", "max_r2": 1.01}, **ex)["verdict"] == "PASS"


# ---- contrast_table / tv_table skip contrasts whose variants are missing ------------------------------------

CONTRASTS = {"total": ["nl", "none"], "placebo": ["matched", "matched_b"], "I@29": ["I@29", "none"]}
ALL_VARIANTS = ["none", "nl", "matched", "matched_b", "I@29"]


def test_contrast_table_skips_missing_variant_in_a_group():
    rng = np.random.default_rng(0)
    cells = {}
    for placement in ("near", "far"):
        for v in ALL_VARIANTS:
            if placement == "far" and v in ("matched_b", "I@29"):
                continue  # single-token / placebo variants defined for `near` only
            cells[("rare", "sushi", placement, v)] = {"sink_mass": rng.normal(size=(N, 2, 3)) * 0.1}
    t = contrast_table(cells, CONTRASTS, ["sink_mass"], BOOT, {"sink_mass": 0.02}, 0.05)
    far = t[t["placement"] == "far"]
    near = t[t["placement"] == "near"]
    assert set(far["contrast"]) == {"total"}
    assert set(near["contrast"]) == set(CONTRASTS)
    assert len(t) == (3 + 1) * 2 * 3
    assert t["q"].notna().all()


def test_contrast_table_skip_when_variant_missing_everywhere():
    rng = np.random.default_rng(0)
    cells = {("rare", "sushi", "near", v): {"sink_mass": rng.normal(size=(N, 2, 3))} for v in ("none", "nl")}
    t = contrast_table(cells, CONTRASTS, ["sink_mass"], BOOT, {"sink_mass": 0.02}, 0.05)
    assert set(t["contrast"]) == {"total"}


def test_tv_table_skips_missing_variant_in_a_group():
    labels = {}
    for placement in ("near", "far"):
        for v in ALL_VARIANTS:
            if placement == "far" and v in ("matched_b", "I@29"):
                continue
            labels[("rare", "sushi", placement, v)] = np.ones((10, 2, 3), dtype=int)
    tv = tv_table(labels, CONTRASTS)
    assert set(tv[tv["placement"] == "far"]["contrast"]) == {"total"}
    assert set(tv[tv["placement"] == "near"]["contrast"]) == set(CONTRASTS)
    assert len(tv) == (3 + 1) * 2 * 3
