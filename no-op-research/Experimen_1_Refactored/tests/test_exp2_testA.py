"""Tests for the exp2 test-A additions: labelled combos, linear_terms, phrase_level differences, familiarity_table and the
pooled_ratio_at_least / pooled_abs_less / familiarity_rho_sign prediction kinds."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from exp2.analysis import additivity_tables, check_prediction, familiarity_table, interactions, phrase_level

BOOT = {"iters": 200, "seed": 0, "ci": 0.95}
DEPTH = {"early": [0, 1], "late": [2, 3]}
N, L, H = 40, 4, 3
METRIC = "sink_mass"
MINUS = "−"
PHRASES = ("sushi", "tea")
SINGLES = ["x", "y", "z"]
PAIRS = {"PV": ("x", "y"), "VO": ("y", "z"), "PO": ("x", "z")}


def combos_for(phrases=PHRASES):
    out = {}
    for ph in phrases:
        out[f"{ph}:nl"] = {"phrase": ph, "placement": "near", "parts": SINGLES, "variant": "nl"}
        for lab, ps in PAIRS.items():
            out[f"{ph}:{lab}"] = {"phrase": ph, "placement": "near", "parts": list(ps), "variant": f"{lab}_{ph}", "label": lab}
    return out


LINEAR = {"triple": {"nl": 1.0, "PV": -1.0, "VO": -1.0, "PO": -1.0}}


def planted(seed):
    rng = np.random.default_rng(seed)
    sing = {s: rng.normal(size=(L, H)) * 0.1 for s in SINGLES}
    pair = {lab: rng.normal(size=(L, H)) * 0.2 for lab in PAIRS}
    return sing, pair, rng.normal(size=(L, H)) * 0.3


def make_cells(noise=0.01, seed=0, bases=("rare",)):
    """f(S) = none + sum of single effects + pair effects inside S + triple effect if S is all three; same for every row
    plus small row noise. Returns cells and the planted effects per phrase."""
    rng = np.random.default_rng(seed)
    cells, plants = {}, {}
    for base in bases:
        for i, ph in enumerate(PHRASES):
            sing, pair, trip = planted(100 * i + 1)
            plants[ph] = (sing, pair, trip)
            none = rng.normal(size=(N, L, H)) * 0.1 + 0.3
            f = lambda s, extra: none + sum(sing[k] for k in s) + extra + rng.normal(size=(N, L, H)) * noise
            v = {"none": none}
            for s in SINGLES:
                v[s] = f([s], 0.0)
            for lab, ps in PAIRS.items():
                v[f"{lab}_{ph}"] = f(ps, pair[lab])
            v["nl"] = f(SINGLES, sum(pair.values()) + trip)
            for name, arr in v.items():
                cells[(base, ph, "near", name)] = {METRIC: arr}
    return cells, plants


# ---- Möbius identity -----------------------------------------------------------------------------------------

def test_interactions_labels_and_planted_pair_effects():
    cells, plants = make_cells()
    inter = interactions(cells, combos_for(), METRIC)
    assert set(inter) == {("rare", ph, "near", lab) for ph in PHRASES for lab in ("nl", "PV", "VO", "PO")}
    for ph in PHRASES:
        _, pair, trip = plants[ph]
        for lab in PAIRS:
            np.testing.assert_allclose(inter[("rare", ph, "near", lab)].mean(axis=0), pair[lab], atol=0.02)
        np.testing.assert_allclose(inter[("rare", ph, "near", "nl")].mean(axis=0), sum(pair.values()) + trip, atol=0.03)


def test_mobius_identity_triple_term_and_nl_algebra():
    cells, plants = make_cells(bases=("rare", "common"))
    _, depth, _ = additivity_tables(cells, combos_for(), [], DEPTH, METRIC, BOOT, linear_terms=LINEAR)
    assert set(depth["term"]) == {"nl", "PV", "VO", "PO", "triple"}
    assert len(depth) == 2 * 2 * 5 * len(DEPTH)
    get = lambda b, ph, t, d: depth[(depth["base"] == b) & (depth["phrase"] == ph) & (depth["term"] == t) & (depth["depth"] == d)].iloc[0]
    for b in ("rare", "common"):
        for ph in PHRASES:
            _, pair, trip = plants[ph]
            for d, layers in DEPTH.items():
                assert get(b, ph, "triple", d)["mean"] == pytest.approx(trip[layers].mean(), abs=0.02)
                for lab in PAIRS:
                    assert get(b, ph, lab, d)["mean"] == pytest.approx(pair[lab][layers].mean(), abs=0.02)
                nl = get(b, ph, "nl", d)["mean"]
                rhs = sum(get(b, ph, lab, d)["mean"] for lab in PAIRS) + get(b, ph, "triple", d)["mean"]
                assert nl == pytest.approx(rhs, abs=1e-12)


def test_linear_terms_equal_coefficient_weighted_interactions():
    cells, _ = make_cells()
    inter = interactions(cells, combos_for(), METRIC)
    coefs = {"PV": 2.0, "VO": -0.5}
    _, depth, _ = additivity_tables(cells, combos_for(), [], DEPTH, METRIC, BOOT, linear_terms={"mix": coefs})
    for ph in PHRASES:
        x = sum(c * inter[("rare", ph, "near", lab)] for lab, c in coefs.items())
        for d, layers in DEPTH.items():
            row = depth[(depth["phrase"] == ph) & (depth["term"] == "mix") & (depth["depth"] == d)].iloc[0]
            assert row["mean"] == pytest.approx(x[:, layers, :].mean(), abs=1e-12)
            assert row["lo"] <= row["mean"] <= row["hi"]


def test_linear_term_skipped_when_a_label_is_missing():
    cells, _ = make_cells()
    combos = {k: v for k, v in combos_for().items() if not (k.startswith("tea:") and k.endswith("PO"))}
    _, depth, _ = additivity_tables(cells, combos, [], DEPTH, METRIC, BOOT, linear_terms=LINEAR)
    assert set(depth[depth["term"] == "triple"]["phrase"]) == {"sushi"}


def test_per_head_uses_label_and_observed_is_variant_minus_none():
    cells, _ = make_cells()
    per_head, _, _ = additivity_tables(cells, combos_for(), [], DEPTH, METRIC, BOOT)
    assert set(per_head["combo"]) == {"nl", "PV", "VO", "PO"}
    assert len(per_head) == len(PHRASES) * 4 * L * H
    for ph in PHRASES:
        for lab in PAIRS:
            sub = per_head[(per_head["phrase"] == ph) & (per_head["combo"] == lab)]
            assert len(sub) == L * H
            k = ("rare", ph, "near")
            exp = (cells[(*k, f"{lab}_{ph}")][METRIC] - cells[(*k, "none")][METRIC]).mean(axis=0)
            r = sub.iloc[5]
            assert r["observed"] == pytest.approx(exp[r["layer"], r["head"]])
    np.testing.assert_allclose(per_head["observed"] - per_head["additive"], per_head["interaction"], atol=1e-12)


def test_differences_use_labels_in_additivity_tables():
    cells, plants = make_cells()
    _, depth, fit = additivity_tables(cells, combos_for(), [["VO", "PV"]], DEPTH, METRIC, BOOT)
    assert f"VO{MINUS}PV" in set(depth["term"]) and set(fit["pair"]) == {f"VO{MINUS}PV"} and len(fit) == 2
    for ph in PHRASES:
        _, pair, _ = plants[ph]
        row = depth[(depth["phrase"] == ph) & (depth["term"] == f"VO{MINUS}PV") & (depth["depth"] == "late")].iloc[0]
        assert row["mean"] == pytest.approx((pair["VO"] - pair["PV"])[2:].mean(), abs=0.02)


# ---- backward compatibility (no label) ----------------------------------------------------------------------

def test_combos_without_label_keep_variant_names():
    combos = {"nl": {"phrase": "sushi", "placement": "near", "parts": ["x", "y"]},
              "sh": {"phrase": "sushi", "placement": "near", "parts": ["y", "z"], "variant": "shuffled"}}
    rng = np.random.default_rng(0)
    none = rng.normal(size=(N, L, H))
    eff = {s: rng.normal(size=(L, H)) * 0.1 for s in SINGLES}
    extra = rng.normal(size=(L, H)) * 0.2
    v = {"none": none, **{s: none + eff[s] for s in SINGLES},
         "nl": none + eff["x"] + eff["y"] + extra, "shuffled": none + eff["y"] + eff["z"]}
    cells = {("rare", "sushi", "near", k): {METRIC: a} for k, a in v.items()}
    inter = interactions(cells, combos, METRIC)
    assert set(inter) == {("rare", "sushi", "near", "nl"), ("rare", "sushi", "near", "shuffled")}
    np.testing.assert_allclose(inter[("rare", "sushi", "near", "nl")], np.broadcast_to(extra, (N, L, H)), atol=1e-12)
    np.testing.assert_allclose(inter[("rare", "sushi", "near", "shuffled")], 0.0, atol=1e-12)
    per_head, depth, fit = additivity_tables(cells, combos, [["nl", "shuffled"]], DEPTH, METRIC, BOOT)
    assert set(per_head["combo"]) == {"nl", "shuffled"}
    assert set(depth["term"]) == {"nl", "shuffled", f"nl{MINUS}shuffled"}
    assert set(fit["pair"]) == {f"nl{MINUS}shuffled"}


# ---- phrase_level differences -------------------------------------------------------------------------------

def depth_frame(means_by_term, phrases=("a", "b", "c", "d"), base="rare", depth="late", half=0.02):
    rows = []
    for term, means in means_by_term.items():
        rows += [(METRIC, base, ph, term, depth, m, m - half, m + half) for ph, m in zip(phrases, means)]
    return pd.DataFrame(rows, columns=["metric", "base", "phrase", "term", "depth", "mean", "lo", "hi"])


def test_phrase_level_difference_pooled_mean_and_counts():
    vo, pv = [0.30, 0.10, 0.05, 0.20], [0.10, 0.20, 0.05, 0.05]   # diffs: .2, -.1, 0, .15
    out = phrase_level(depth_frame({"VO": vo, "PV": pv}), None, BOOT, differences=[["VO", "PV"]])
    assert set(out["term"]) == {"VO", "PV", f"VO{MINUS}PV"}
    r = out[out["term"] == f"VO{MINUS}PV"].iloc[0]
    assert r["pooled_mean"] == pytest.approx(np.mean(np.array(vo) - np.array(pv)))
    assert r["n_phrases"] == 4
    assert r["n_pos"] == 2 and r["n_neg"] == 1       # the zero-difference phrase counts as neither
    assert r["lo"] <= r["pooled_mean"] <= r["hi"]
    # ordinary terms are untouched
    assert out[out["term"] == "VO"].iloc[0]["n_pos"] == 4


def test_phrase_level_difference_skips_phrases_missing_a_term():
    d = depth_frame({"VO": [0.3, 0.1, 0.2], "PV": [0.1, 0.1, 0.0]})
    d = d[~((d["term"] == "PV") & (d["phrase"] == "c"))]
    r = phrase_level(d, None, BOOT, differences=[["VO", "PV"]]).set_index("term").loc[f"VO{MINUS}PV"]
    assert r["n_phrases"] == 2 and r["pooled_mean"] == pytest.approx(0.1)


def test_phrase_level_difference_per_base_and_depth():
    d = pd.concat([depth_frame({"VO": [0.3, 0.2, 0.1], "PV": [0.0, 0.0, 0.0]}, phrases="abc", base="rare"),
                   depth_frame({"VO": [0.0, 0.0, 0.0], "PV": [0.3, 0.2, 0.1]}, phrases="abc", base="common", depth="early")])
    out = phrase_level(d, None, BOOT, differences=[["VO", "PV"]])
    t = out[out["term"] == f"VO{MINUS}PV"].set_index(["base", "depth"])
    assert t.loc[("rare", "late"), "pooled_mean"] == pytest.approx(0.2)
    assert t.loc[("common", "early"), "pooled_mean"] == pytest.approx(-0.2)
    assert t.loc[("common", "early"), "n_neg"] == 3 and t.loc[("rare", "late"), "n_pos"] == 3


# ---- familiarity_table --------------------------------------------------------------------------------------

def fam_depth(means, phrases, label="PV", base="rare", metric=METRIC, depth="late"):
    return pd.DataFrame([(metric, base, ph, label, depth, m, m - 0.01, m + 0.01) for ph, m in zip(phrases, means)],
                        columns=["metric", "base", "phrase", "term", "depth", "mean", "lo", "hi"])


def test_familiarity_rho_minus_one_and_row_granularity():
    phrases = list("abcde")
    nll = [1.0, 2.0, 3.0, 4.0, 5.0]
    fam = pd.DataFrame({"phrase": phrases * 2, "label": ["PV"] * 5 + ["VO"] * 5, "nll": nll + nll[::-1]})
    d = pd.concat([fam_depth([0.9, 0.7, 0.5, 0.2, 0.1], phrases, "PV"),
                   fam_depth([0.9, 0.7, 0.5, 0.2, 0.1], phrases, "VO"),
                   fam_depth([0.1, 0.2, 0.3, 0.4, 0.5], phrases, "PV", depth="early"),
                   fam_depth([0.1, 0.2, 0.3, 0.4, 0.5], phrases, "PV", base="common"),
                   fam_depth([0.1, 0.2, 0.3, 0.4, 0.5], phrases, "PV", metric="sink_logit"),
                   fam_depth([0.5] * 5, phrases, "nl")], ignore_index=True)   # no familiarity for nl -> no row
    out = familiarity_table(d, fam)
    assert len(out) == 5 and not out.duplicated(["metric", "base", "label", "depth"]).any()
    assert set(out["label"]) == {"PV", "VO"}
    get = lambda **kw: out[np.logical_and.reduce([out[k] == v for k, v in kw.items()])].iloc[0]
    r = get(label="PV", base="rare", depth="late", metric=METRIC)
    assert r["spearman_rho"] == pytest.approx(-1.0) and r["n_phrases"] == 5 and r["mean_nll"] == pytest.approx(3.0)
    assert 0 <= r["spearman_p"] < 0.05
    assert get(label="VO", depth="late")["spearman_rho"] == pytest.approx(1.0)      # nll reversed
    assert get(label="PV", depth="early")["spearman_rho"] == pytest.approx(1.0)
    assert get(label="PV", base="common")["spearman_rho"] == pytest.approx(1.0)
    assert get(label="PV", metric="sink_logit")["spearman_rho"] == pytest.approx(1.0)


def test_familiarity_aligns_by_phrase_name_not_order():
    phrases = list("abcd")
    fam = pd.DataFrame({"phrase": ["d", "c", "b", "a"], "label": "PV", "nll": [4.0, 3.0, 2.0, 1.0]})
    out = familiarity_table(fam_depth([0.4, 0.3, 0.2, 0.1], phrases), fam)
    assert out.iloc[0]["spearman_rho"] == pytest.approx(-1.0)


# ---- check_prediction: new kinds ----------------------------------------------------------------------------

def pooled(rows):
    return pd.DataFrame(rows, columns=["metric", "base", "term", "depth", "n_phrases", "pooled_mean", "lo", "hi", "n_pos", "n_neg",
                                       "spearman_rho", "spearman_p"])


def prow(base, term, mean, depth="late", metric=METRIC):
    return (metric, base, term, depth, 12, mean, mean - 0.01, mean + 0.01, 0, 0, 0.0, 1.0)


def run(spec, **extra):
    return check_prediction(spec, pd.DataFrame(), pd.DataFrame(), extra)


def test_check_pooled_ratio_at_least_pass_fail():
    pp = pooled([prow("rare", "VO", 0.30), prow("rare", "PV", 0.10), prow("common", "VO", 0.20), prow("common", "PV", 0.10),
                 prow("rare", "VO", 9.0, depth="early"), prow("rare", "PV", 0.01, depth="early")])
    spec = {"id": "R", "kind": "pooled_ratio_at_least", "numerator": "VO", "denominator": "PV", "depth": "late", "min_ratio": 2.0}
    r = run(spec, phrase_pooled=pp)
    assert r["verdict"] == "PASS" and r["kind"] == "pooled_ratio_at_least" and "3.00" in r["detail"] and "2.00" in r["detail"]
    assert run(spec | {"min_ratio": 2.5}, phrase_pooled=pp)["verdict"] == "FAIL"             # common ratio is 2
    assert run(spec | {"min_ratio": 2.5, "bases": ["rare"]}, phrase_pooled=pp)["verdict"] == "PASS"
    assert run(spec | {"depth": "early", "min_ratio": 100.0, "bases": ["rare"]}, phrase_pooled=pp)["verdict"] == "PASS"
    assert run(spec | {"metric": "sink_logit"}, phrase_pooled=pp.assign(metric="sink_logit"))["verdict"] == "PASS"


def test_check_with_no_matching_rows_does_not_pass_vacuously():
    pp = pooled([prow("rare", "VO", 0.30), prow("rare", "PV", 0.10)])
    spec = {"id": "R", "kind": "pooled_ratio_at_least", "numerator": "VO", "denominator": "PV", "depth": "late", "min_ratio": 2.0,
            "metric": "sink_logit"}
    r = run(spec, phrase_pooled=pp)
    assert r["verdict"] == "FAIL" and "no rows matched" in r["detail"]


def test_effects_check_with_no_matching_rows_does_not_pass_vacuously():
    effects = pd.DataFrame({"base": ["rare"], "phrase": ["p"], "placement": ["near"], "contrast": ["total"],
                            "metric": ["sink_mass"], "name": ["L0H0"], "mean": [0.0], "meaningful": [False]})
    spec = {"id": "P", "kind": "few_meaningful", "contrast": "placebo", "metric": "sink_mass", "max_heads": {"near": 0}}
    r = check_prediction(spec, effects, pd.DataFrame())
    assert r["verdict"] == "FAIL" and "no rows matched" in r["detail"]


def test_check_pooled_ratio_zero_denominator_fails():
    pp = pooled([prow("rare", "VO", 0.3), prow("rare", "PV", 0.0)])
    spec = {"id": "R", "kind": "pooled_ratio_at_least", "numerator": "VO", "denominator": "PV", "depth": "late", "min_ratio": 1.0}
    assert run(spec, phrase_pooled=pp)["verdict"] == "FAIL"


def test_check_pooled_abs_less_pass_fail():
    pp = pooled([prow("rare", "triple", -0.02), prow("rare", "VO", 0.10), prow("common", "triple", 0.03), prow("common", "VO", 0.05)])
    spec = {"id": "A", "kind": "pooled_abs_less", "term": "triple", "than": "VO", "depth": "late"}
    r = run(spec, phrase_pooled=pp)
    assert r["verdict"] == "PASS" and r["kind"] == "pooled_abs_less" and "rare" in r["detail"] and "common" in r["detail"]
    pp_bad = pp.copy()
    pp_bad.loc[2, "pooled_mean"] = 0.08                                                       # |0.08| > 0.05
    assert run(spec, phrase_pooled=pp_bad)["verdict"] == "FAIL"
    assert run(spec | {"bases": ["rare"]}, phrase_pooled=pp_bad)["verdict"] == "PASS"
    assert run(spec | {"term": "VO", "than": "triple"}, phrase_pooled=pp)["verdict"] == "FAIL"


def fam_table(rows):
    return pd.DataFrame(rows, columns=["metric", "base", "label", "depth", "n_phrases", "mean_nll", "spearman_rho", "spearman_p"])


def test_check_familiarity_rho_sign_pass_fail():
    ft = fam_table([(METRIC, "rare", "PV", "late", 12, 3.0, -0.6, 0.04), (METRIC, "common", "PV", "late", 12, 3.0, -0.2, 0.5),
                    (METRIC, "rare", "VO", "late", 12, 3.0, 0.7, 0.01), (METRIC, "rare", "PV", "early", 12, 3.0, 0.4, 0.2)])
    spec = {"id": "F", "kind": "familiarity_rho_sign", "label": "PV", "depth": "late", "sign": "-"}
    r = run(spec, familiarity=ft)
    assert r["verdict"] == "PASS" and r["kind"] == "familiarity_rho_sign" and "rho -0.60" in r["detail"]
    assert run(spec | {"sign": "+"}, familiarity=ft)["verdict"] == "FAIL"
    assert run(spec | {"label": "VO", "sign": "+"}, familiarity=ft)["verdict"] == "PASS"
    assert run(spec | {"label": "VO"}, familiarity=ft)["verdict"] == "FAIL"
    assert run(spec | {"depth": "early", "sign": "+"}, familiarity=ft)["verdict"] == "PASS"
    ft_bad = ft.copy()
    ft_bad.loc[1, "spearman_rho"] = 0.1                                                       # common flips sign
    assert run(spec, familiarity=ft_bad)["verdict"] == "FAIL"
    assert run(spec | {"bases": ["rare"]}, familiarity=ft_bad)["verdict"] == "PASS"


def test_check_new_kinds_end_to_end_with_real_tables():
    cells, plants = make_cells(bases=("rare", "common"))
    _, depth, _ = additivity_tables(cells, combos_for(), [], DEPTH, METRIC, BOOT, linear_terms=LINEAR)
    pp = phrase_level(depth.assign(metric=METRIC), None, BOOT)
    ex = {"phrase_pooled": pp}
    mean = lambda t: np.mean([plants[ph][1][t][2:].mean() for ph in PHRASES])
    want = abs(mean("PV")) < mean("VO") if mean("VO") > 0 else False
    spec = {"id": "a", "kind": "pooled_abs_less", "term": "PV", "than": "VO", "depth": "late"}
    assert (run(spec, **ex)["verdict"] == "PASS") == want
    ratio = {"id": "b", "kind": "pooled_ratio_at_least", "numerator": "nl", "denominator": "nl", "depth": "late", "min_ratio": 1.0}
    assert run(ratio, **ex)["verdict"] == "PASS"
