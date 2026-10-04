"""Tests for the exp2 iteration-3 additions: per-phrase interaction variants, phrase_level, new prediction kinds and
filters, plot_phrase_terms, and the report's _counts / _sizes."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from exp2.analysis import check_prediction, interactions, phrase_level
from exp2.plots import plot_phrase_terms
from exp2.report import _counts, _sizes

BOOT = {"iters": 200, "seed": 0, "ci": 0.95}
N, L, H = 40, 4, 3
METRIC = "sink_mass"
MINUS = "−"
PHRASES = ("sushi", "tea")
SINGLES = {
    "sushi": {"nl": ["I@29", "like@30", "sushi@31"], "shuffled": ["like@29", "I@30", "sushi@31"]},
    "tea": {"nl": ["we@29", "drink@30", "tea@31"], "shuffled": ["drink@29", "we@30", "tea@31"]},
}
COMBOS = {
    f"{ph}:{v}": {"phrase": ph, "placement": "near", "parts": parts, "variant": v}
    for ph, vs in SINGLES.items() for v, parts in vs.items()
}


# ---- interactions with per-phrase variants -------------------------------------------------------------------

def make_cells(extra, bases=("rare", "common"), seed=0):
    """Additive cells for each phrase; extra[(phrase, variant)] [L, H] is added to that combo variant."""
    rng = np.random.default_rng(seed)
    cells = {}
    for base in bases:
        for ph, vs in SINGLES.items():
            parts = sorted({p for ps in vs.values() for p in ps})
            offs = {p: rng.normal(size=(L, H)) * 0.05 for p in parts}
            none = rng.normal(size=(N, L, H)) * 0.1 + 0.3
            v = {"none": none} | {p: none + offs[p] for p in parts}
            for variant, ps in vs.items():
                v[variant] = none + sum(offs[p] for p in ps) + extra.get((ph, variant), 0.0)
            for name, arr in v.items():
                cells[(base, ph, "near", name)] = {METRIC: arr}
    return cells


def test_interactions_keys_use_variant_per_phrase_group():
    inter = interactions(make_cells({}), COMBOS, METRIC)
    want = {(b, ph, "near", v) for b in ("rare", "common") for ph in PHRASES for v in ("nl", "shuffled")}
    assert set(inter) == want
    assert not any(":" in k[3] for k in inter)
    for x in inter.values():
        np.testing.assert_allclose(x, 0.0, atol=1e-12)


def test_interactions_recover_planted_per_phrase_offsets():
    rng = np.random.default_rng(7)
    extra = {(ph, v): rng.normal(size=(L, H)) * 0.2 for ph in PHRASES for v in ("nl", "shuffled")}
    inter = interactions(make_cells(extra), COMBOS, METRIC)
    for base in ("rare", "common"):
        for (ph, v), e in extra.items():
            np.testing.assert_allclose(inter[(base, ph, "near", v)], np.broadcast_to(e, (N, L, H)), atol=1e-12)


def test_interactions_planting_in_one_phrase_leaves_other_untouched():
    extra = {("sushi", "nl"): np.full((L, H), 0.3)}
    inter = interactions(make_cells(extra), COMBOS, METRIC)
    np.testing.assert_allclose(inter[("rare", "sushi", "near", "nl")], 0.3, atol=1e-12)
    for k, x in inter.items():
        if k[1:] != ("sushi", "near", "nl"):
            np.testing.assert_allclose(x, 0.0, atol=1e-12)


# ---- phrase_level --------------------------------------------------------------------------------------------

PH5 = ["a", "b", "c", "d", "e"]


def depth_rows(means, base="rare", metric=METRIC, term="nl", depth="late", half=0.02):
    return [(metric, base, ph, term, depth, m, m - half, m + half) for ph, m in zip(PH5, means)]


def mk_depth(rows):
    return pd.DataFrame(rows, columns=["metric", "base", "phrase", "term", "depth", "mean", "lo", "hi"])


def mk_variants(gaps, base="rare"):
    rows = []
    for ph, g in zip(PH5, gaps):
        rows += [(base, ph, "nl", 10.0), (base, ph, "shuffled", 10.0 + g)]
    return pd.DataFrame(rows, columns=["base", "phrase", "variant", "mean_slot_nll"])


def test_phrase_level_pooled_mean_ci_and_counts():
    means = [0.10, 0.05, 0.0, -0.06, 0.02]
    d = mk_depth(depth_rows(means))
    d["lo"], d["hi"] = d["mean"] - 0.03, d["mean"] + 0.03
    out = phrase_level(d, None, BOOT)
    assert len(out) == 1
    r = out.iloc[0]
    assert r["pooled_mean"] == pytest.approx(np.mean(means))
    assert r["lo"] <= r["pooled_mean"] <= r["hi"] and r["lo"] < r["hi"]
    assert r["n_phrases"] == 5
    assert r["n_pos"] == 2   # 0.10 and 0.05 have lo > 0; 0.02 - 0.03 < 0
    assert r["n_neg"] == 1   # only -0.06 has hi < 0


def test_phrase_level_pooled_mean_is_mean_of_phrase_means_per_group():
    d = pd.concat([mk_depth(depth_rows([0.1, 0.2, 0.3, 0.4, 0.5])),
                   mk_depth(depth_rows([-0.1, -0.2, 0.0, 0.1, 0.0], term="shuffled")),
                   mk_depth(depth_rows([1.0, 1.0, 1.0, 1.0, 3.0], metric="sink_logit"))], ignore_index=True)
    out = phrase_level(d, None, BOOT).set_index(["metric", "term"])
    assert len(out) == 3
    assert out.loc[(METRIC, "nl"), "pooled_mean"] == pytest.approx(0.3)
    assert out.loc[(METRIC, "shuffled"), "pooled_mean"] == pytest.approx(-0.04)
    assert out.loc[("sink_logit", "nl"), "pooled_mean"] == pytest.approx(1.4)


def test_phrase_level_spearman_plus_one_and_minus_one():
    gaps = [0.5, 3.0, 1.0, 7.0, 2.0]
    v = mk_variants(gaps)
    inc = mk_depth(depth_rows([np.exp(g) / 100 for g in gaps]))
    dec = mk_depth(depth_rows([-g ** 3 for g in gaps]))
    assert phrase_level(inc, v, BOOT).iloc[0]["spearman_rho"] == pytest.approx(1.0)
    r = phrase_level(dec, v, BOOT).iloc[0]
    assert r["spearman_rho"] == pytest.approx(-1.0)
    assert 0 <= r["spearman_p"] < 0.05


def test_phrase_level_spearman_is_per_base():
    gaps = [0.5, 3.0, 1.0, 7.0, 2.0]
    v = pd.concat([mk_variants(gaps, "rare"), mk_variants(gaps[::-1], "common")], ignore_index=True)
    means = [g / 10 for g in gaps]
    d = pd.concat([mk_depth(depth_rows(means, base="rare")), mk_depth(depth_rows(means, base="common"))], ignore_index=True)
    out = phrase_level(d, v, BOOT).set_index("base")
    assert out.loc["rare", "spearman_rho"] == pytest.approx(1.0)
    assert out.loc["common", "spearman_rho"] != pytest.approx(1.0)


def test_phrase_level_variants_none_gives_nan_rho():
    out = phrase_level(mk_depth(depth_rows([0.1, 0.2, 0.3, 0.4, 0.5])), None, BOOT)
    assert np.isnan(out.iloc[0]["spearman_rho"]) and np.isnan(out.iloc[0]["spearman_p"])
    assert out.iloc[0]["pooled_mean"] == pytest.approx(0.3)


# ---- check_prediction: pooled / count / spearman kinds -------------------------------------------------------

def pooled_df(rows):
    return pd.DataFrame(rows, columns=["metric", "base", "term", "depth", "n_phrases", "pooled_mean", "lo", "hi",
                                       "n_pos", "n_neg", "spearman_rho", "spearman_p"])


def run(spec, **extra):
    return check_prediction(spec, pd.DataFrame(), pd.DataFrame(), extra)


def test_check_pooled_term_sign_pass_fail():
    pp = pooled_df([
        (METRIC, "rare", "nl", "late", 12, 0.05, 0.02, 0.08, 9, 0, -0.5, 0.1),
        (METRIC, "common", "nl", "late", 12, 0.04, 0.01, 0.07, 8, 0, -0.4, 0.2),
        (METRIC, "rare", "nl", "early", 12, -0.03, -0.05, -0.01, 0, 6, 0.0, 0.9),
    ])
    spec = {"id": "P1", "kind": "pooled_term_sign", "term": "nl", "depth": "late", "sign": "+"}
    r = run(spec, phrase_pooled=pp)
    assert r["verdict"] == "PASS" and r["kind"] == "pooled_term_sign" and "12 phrases" in r["detail"]
    assert run(spec | {"sign": "-"}, phrase_pooled=pp)["verdict"] == "FAIL"
    assert run(spec | {"depth": "early", "sign": "-", "bases": ["rare"]}, phrase_pooled=pp)["verdict"] == "PASS"
    pp_bad = pp.copy()
    pp_bad.loc[1, "lo"] = -0.01
    assert run(spec, phrase_pooled=pp_bad)["verdict"] == "FAIL"
    assert run(spec | {"bases": ["rare"]}, phrase_pooled=pp_bad)["verdict"] == "PASS"


def test_check_phrase_count_pass_fail_and_sign():
    pp = pooled_df([
        (METRIC, "rare", "nl", "late", 12, 0.05, 0.02, 0.08, 9, 1, 0.0, 1.0),
        (METRIC, "common", "nl", "late", 12, 0.04, 0.01, 0.07, 7, 3, 0.0, 1.0),
    ])
    spec = {"id": "C1", "kind": "phrase_count", "term": "nl", "depth": "late", "sign": "+", "min_count": 7}
    r = run(spec, phrase_pooled=pp)
    assert r["verdict"] == "PASS" and "9/12" in r["detail"] and "7/12" in r["detail"]
    assert run(spec | {"min_count": 8}, phrase_pooled=pp)["verdict"] == "FAIL"           # common has 7
    assert run(spec | {"min_count": 8, "bases": ["rare"]}, phrase_pooled=pp)["verdict"] == "PASS"
    assert run(spec | {"sign": "-", "min_count": 3}, phrase_pooled=pp)["verdict"] == "FAIL"  # rare has 1
    assert run(spec | {"sign": "-", "min_count": 1}, phrase_pooled=pp)["verdict"] == "PASS"


def test_check_phrase_spearman_below_pass_fail():
    pp = pooled_df([
        (METRIC, "rare", "nl", "late", 12, 0.05, 0.02, 0.08, 9, 0, -0.6, 0.04),
        (METRIC, "common", "nl", "late", 12, 0.04, 0.01, 0.07, 8, 0, -0.1, 0.7),
    ])
    spec = {"id": "S1", "kind": "phrase_spearman_below", "term": "nl", "depth": "late", "max_rho": 0.0}
    r = run(spec, phrase_pooled=pp)
    assert r["verdict"] == "PASS" and "rho -0.60" in r["detail"]
    assert run(spec | {"max_rho": -0.3}, phrase_pooled=pp)["verdict"] == "FAIL"           # common -0.1
    assert run(spec | {"max_rho": -0.3, "bases": ["rare"]}, phrase_pooled=pp)["verdict"] == "PASS"
    pp_nan = pp.copy()
    pp_nan["spearman_rho"] = np.nan
    assert run(spec, phrase_pooled=pp_nan)["verdict"] == "FAIL"                           # NaN is never "below"


def test_check_pooled_kinds_filter_by_metric():
    pp = pooled_df([
        (METRIC, "rare", "nl", "late", 12, 0.05, 0.02, 0.08, 9, 0, -0.6, 0.04),
        ("sink_logit", "rare", "nl", "late", 12, -0.5, -0.8, -0.2, 0, 10, 0.5, 0.1),
    ])
    spec = {"id": "M", "kind": "pooled_term_sign", "term": "nl", "depth": "late", "sign": "+"}
    assert run(spec, phrase_pooled=pp)["verdict"] == "PASS"                                  # default sink_mass
    assert run(spec | {"metric": "sink_logit"}, phrase_pooled=pp)["verdict"] == "FAIL"
    assert run(spec | {"metric": "sink_logit", "sign": "-"}, phrase_pooled=pp)["verdict"] == "PASS"
    assert run({"id": "M2", "kind": "phrase_count", "term": "nl", "depth": "late", "sign": "-", "min_count": 10,
                "metric": "sink_logit"}, phrase_pooled=pp)["verdict"] == "PASS"


# ---- check_prediction: metric / phrase filters, per-(base, phrase) behaviour --------------------------------

def depth_df(rows):
    return pd.DataFrame(rows, columns=["metric", "base", "phrase", "placement", "term", "depth", "mean", "lo", "hi"])


def test_check_depth_term_sign_metric_filter_picks_sink_logit():
    ad = depth_df([
        (METRIC, "rare", "p", "near", "nl", "late", 0.05, 0.02, 0.08),
        ("sink_logit", "rare", "p", "near", "nl", "late", -0.4, -0.6, -0.2),
    ])
    spec = {"id": "D", "kind": "depth_term_sign", "term": "nl", "depth": "late", "sign": "+"}
    assert run(spec, additivity_depth=ad)["verdict"] == "PASS"
    r = run(spec | {"metric": "sink_logit"}, additivity_depth=ad)
    assert r["verdict"] == "FAIL" and "-0.4000" in r["detail"]
    assert run(spec | {"metric": "sink_logit", "sign": "-"}, additivity_depth=ad)["verdict"] == "PASS"


def two_phrase_depth(second_lo=-0.01):
    return depth_df([
        (METRIC, "rare", "sushi", "near", "nl", "late", 0.05, 0.02, 0.08),
        (METRIC, "rare", "tea", "near", "nl", "late", 0.03, second_lo, 0.07),
        (METRIC, "common", "sushi", "near", "nl", "late", 0.04, 0.01, 0.07),
        (METRIC, "common", "tea", "near", "nl", "late", 0.04, 0.01, 0.07),
    ])


def test_check_depth_term_sign_fails_if_either_phrase_fails():
    ad = two_phrase_depth()
    spec = {"id": "D", "kind": "depth_term_sign", "term": "nl", "depth": "late", "sign": "+"}
    r = run(spec, additivity_depth=ad)
    assert r["verdict"] == "FAIL"
    assert "rare/sushi" in r["detail"] and "rare/tea" in r["detail"] and "common/tea" in r["detail"]
    assert run(spec, additivity_depth=two_phrase_depth(second_lo=0.01))["verdict"] == "PASS"
    assert run(spec | {"bases": ["common"]}, additivity_depth=ad)["verdict"] == "PASS"


def test_check_depth_term_sign_phrase_filter_restricts():
    ad = two_phrase_depth()
    spec = {"id": "D", "kind": "depth_term_sign", "term": "nl", "depth": "late", "sign": "+"}
    r = run(spec | {"phrase": "sushi"}, additivity_depth=ad)
    assert r["verdict"] == "PASS" and "tea" not in r["detail"] and "rare/sushi" in r["detail"]
    r = run(spec | {"phrase": "tea"}, additivity_depth=ad)
    assert r["verdict"] == "FAIL" and "sushi" not in r["detail"]


def test_check_order_fit_r2_below_must_hold_for_every_phrase():
    fd = pd.DataFrame([
        (METRIC, "rare", "sushi", "near", f"nl{MINUS}shuffled", 0.7, 0.5, 0.25, 0.1, 0.08),
        (METRIC, "rare", "tea", "near", f"nl{MINUS}shuffled", 0.9, 0.9, 0.81, 0.1, 0.02),
        ("sink_logit", "rare", "tea", "near", f"nl{MINUS}shuffled", 0.1, 0.1, 0.01, 0.1, 0.09),
    ], columns=["metric", "base", "phrase", "placement", "pair", "slope", "r", "r2", "mean_abs_observed",
                "mean_abs_interaction"])
    spec = {"id": "O", "kind": "order_fit_r2_below", "pair": f"nl{MINUS}shuffled", "max_r2": 0.5}
    assert run(spec, order_fit=fd)["verdict"] == "FAIL"                                      # tea 0.81
    assert run(spec | {"max_r2": 0.9}, order_fit=fd)["verdict"] == "PASS"
    assert run(spec | {"phrase": "sushi"}, order_fit=fd)["verdict"] == "PASS"                # 0.25 < 0.5
    assert run(spec | {"phrase": "tea"}, order_fit=fd)["verdict"] == "FAIL"
    assert run(spec | {"metric": "sink_logit"}, order_fit=fd)["verdict"] == "PASS"


# ---- plot_phrase_terms ---------------------------------------------------------------------------------------

def terms_df(excl_nl=True):
    rows = []
    rng = np.random.default_rng(0)
    for metric in ("sink_mass", "sink_logit"):
        for base in ("rare", "common"):
            for i, ph in enumerate(("sushi", "tea", "rice")):
                for term in ("nl", "shuffled", f"nl{MINUS}shuffled"):
                    m = rng.normal() * 0.05 + (0.05 if term == "nl" else 0.0)
                    half = 0.01 if (term == "nl" and excl_nl) else 0.2
                    rows.append((metric, base, ph, term, "late", m, m - half, m + half))
                    rows.append((metric, base, ph, term, "early", m, m - half, m + half))
    return pd.DataFrame(rows, columns=["metric", "base", "phrase", "term", "depth", "mean", "lo", "hi"])


def assert_png(path):
    assert path.exists() and path.stat().st_size > 0
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_plot_phrase_terms_writes_png(tmp_path):
    out = tmp_path / "terms.png"
    plot_phrase_terms(terms_df(), "sink_mass", "late", out, texts={"sushi": " I like sushi", "tea": "we drink tea"})
    assert_png(out)


def test_plot_phrase_terms_mixed_ci_exclusion(tmp_path):
    d = terms_df(excl_nl=True)
    d.loc[(d["term"] == "shuffled"), ["lo", "hi"]] = [-0.1, 0.1]
    out = tmp_path / "mixed.png"
    plot_phrase_terms(d, "sink_logit", "early", out)
    assert_png(out)
    # no CI excludes 0 anywhere (all hollow markers) still renders
    out2 = tmp_path / "none.png"
    plot_phrase_terms(terms_df(excl_nl=False), "sink_mass", "late", out2)
    assert_png(out2)


# ---- report._counts / _sizes ---------------------------------------------------------------------------------

def effects_frame():
    rows = []
    # (phrase, placement) -> contrast -> head -> (mean, meaningful)
    spec = {
        ("sushi", "near"): {"total": [(0.5, True), (-0.9, True), (0.1, False), (0.02, False)],
                            "order": [(0.2, True), (0.1, False), (-0.3, True), (0.0, False)],
                            "placebo": [(0.01, False), (0.02, False), (0.03, False), (0.04, True)]},
        ("sushi", "far"): {"total": [(0.01, False), (0.02, False), (0.03, False), (0.04, False)],
                           "order": [(0.0, False)] * 4, "placebo": [(0.0, False)] * 4},
        ("tea", "near"): {"total": [(0.8, True), (0.2, True), (-0.1, True), (0.4, False)],
                          "order": [(0.1, True), (0.1, True), (0.1, True), (0.1, True)],
                          "placebo": [(0.0, False)] * 4},
        ("tea", "far"): {"total": [(-0.05, False), (0.05, False), (0.0, False), (0.2, True)],
                         "order": [(0.0, False)] * 4, "placebo": [(0.0, False)] * 4},
    }
    for (phrase, placement), cs in spec.items():
        for contrast, vals in cs.items():
            for h, (m, mf) in enumerate(vals):
                rows.append(("rare", phrase, placement, contrast, "sink_mass", f"L0H{h}", m, mf))
    # another metric that must be ignored
    rows.append(("rare", "sushi", "near", "total", "slot_mass", "L0H0", 99.0, True))
    return pd.DataFrame(rows, columns=["base", "phrase", "placement", "contrast", "metric", "name", "mean", "meaningful"])


def test_counts_one_row_per_group_columns_in_first_seen_order():
    c = _counts(effects_frame(), "sink_mass")
    assert list(c.columns) == ["base", "phrase", "placement", "total", "order", "placebo"]
    assert len(c) == 4
    assert not c.duplicated(["base", "phrase", "placement"]).any()
    row = lambda ph, pl: c[(c["phrase"] == ph) & (c["placement"] == pl)].iloc[0]
    assert (row("sushi", "near")["total"], row("sushi", "near")["order"], row("sushi", "near")["placebo"]) == (2, 2, 1)
    assert (row("sushi", "far")["total"], row("sushi", "far")["order"]) == (0, 0)
    assert (row("tea", "near")["total"], row("tea", "near")["order"], row("tea", "near")["placebo"]) == (3, 4, 0)
    assert row("tea", "far")["total"] == 1


def test_counts_uses_only_requested_metric():
    c = _counts(effects_frame(), "slot_mass")
    assert len(c) == 1 and c.iloc[0]["total"] == 1 and list(c.columns) == ["base", "phrase", "placement", "total"]


def test_sizes_medians_and_largest_head_of_first_contrast():
    s = _sizes(effects_frame(), "sink_mass")
    assert list(s.columns) == ["base", "phrase", "placement", "total median |Δ|", "order median |Δ|",
                               "placebo median |Δ|", "largest total"]
    assert len(s) == 4
    row = lambda ph, pl: s[(s["phrase"] == ph) & (s["placement"] == pl)].iloc[0]
    sn = row("sushi", "near")
    assert sn["total median |Δ|"] == pytest.approx(np.median([0.5, 0.9, 0.1, 0.02]))
    assert sn["order median |Δ|"] == pytest.approx(np.median([0.2, 0.1, 0.3, 0.0]))
    assert sn["placebo median |Δ|"] == pytest.approx(np.median([0.01, 0.02, 0.03, 0.04]))
    assert sn["largest total"] == "L0H1 (-0.900)"                      # max |mean|, sign kept
    assert row("tea", "near")["largest total"] == "L0H0 (+0.800)"
    assert row("tea", "far")["largest total"] == "L0H3 (+0.200)"
    assert row("tea", "far")["total median |Δ|"] == pytest.approx(np.median([0.05, 0.05, 0.0, 0.2]))
