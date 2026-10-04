"""Tests for exp2.analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import wilcoxon

from exp1.compare import bh_fdr
from exp1.metrics import LABELS
from exp2.analysis import cell_summary, check_prediction, contrast_table, paired_test, pool_over_phrases, tv_table

CONTRASTS = {"total": ["nl", "none"], "order": ["nl", "shuffled"], "identity": ["shuffled", "matched"],
             "form": ["matched", "none"], "placebo": ["matched", "matched_b"]}
VARIANTS = ["none", "nl", "shuffled", "matched", "matched_b"]
METRICS = ["sink_mass", "entropy"]
MIN_EFFECT = {"sink_mass": 0.02, "entropy": 0.05}
BOOT = {"iters": 200, "seed": 0, "ci": 0.95}
N, L, H = 60, 2, 3


# ---- paired_test ---------------------------------------------------------

def test_paired_test_matches_scipy_with_ties_and_zeros():
    rng = np.random.default_rng(0)
    d = rng.normal(size=(60, 5))
    d[:, 1] = np.round(d[:, 1], 1)          # ties
    d[::4, 2] = 0.0                         # zeros
    d[:, 3] = rng.integers(-3, 4, size=60)  # heavy ties and zeros
    d[:, 4] += 0.3
    out = paired_test(d)
    for j in range(d.shape[1]):
        ref = wilcoxon(d[:, j], zero_method="wilcox", correction=False, method="approx").pvalue
        np.testing.assert_allclose(out["p"][j], ref, rtol=1e-6)


def test_paired_test_effect_size_signs_and_shape():
    rng = np.random.default_rng(1)
    pos = rng.uniform(0.1, 1, size=(30, 1))
    sym = np.concatenate([np.linspace(0.1, 1, 15), -np.linspace(0.1, 1, 15)])[:, None]
    d = np.concatenate([pos, -pos, sym, np.zeros((30, 1))], axis=1)
    out = paired_test(d)
    assert out["effect_size"][0] == pytest.approx(1.0)
    assert out["effect_size"][1] == pytest.approx(-1.0)
    assert out["effect_size"][2] == pytest.approx(0.0, abs=1e-12)
    assert out["p"][3] == 1.0 and out["effect_size"][3] == 0.0
    d3 = rng.normal(size=(20, 2, 4))
    o3 = paired_test(d3)
    assert o3["p"].shape == (2, 4) and o3["effect_size"].shape == (2, 4)


# ---- synthetic cells -----------------------------------------------------

def make_cells(phrases=("sushi",), seed=0, effect=0.3):
    rng = np.random.default_rng(seed)
    cells = {}
    for base in ("rare", "common"):
        for phrase in phrases:
            for placement in ("near", "far"):
                none = {m: rng.normal(size=(N, L, H)) * 0.1 for m in METRICS}
                for var in VARIANTS:
                    vals = {}
                    for m in METRICS:
                        v = none[m].copy() if var == "none" else none[m] + rng.normal(size=(N, L, H)) * 0.01
                        if var == "nl":
                            v[:, 0, 0] += effect * (10 if m == "entropy" else 1)  # large planted nl effect
                        vals[m] = v
                    cells[(base, phrase, placement, var)] = vals
    return cells


@pytest.fixture(scope="module")
def table():
    return contrast_table(make_cells(), CONTRASTS, METRICS, BOOT, MIN_EFFECT, 0.05)


def test_contrast_table_shape_and_columns(table):
    assert len(table) == 2 * 1 * 2 * len(CONTRASTS) * len(METRICS) * L * H
    for col in ["base", "phrase", "placement", "contrast", "name", "metric", "layer", "head", "mean", "lo", "hi",
                "p", "effect_size", "q", "sig", "meaningful"]:
        assert col in table.columns


def test_contrast_table_telescoping_identity(table):
    key = ["base", "placement", "metric", "layer", "head"]
    piv = table.pivot_table(index=key, columns="contrast", values="mean")
    np.testing.assert_allclose(piv["total"], piv["order"] + piv["identity"] + piv["form"], atol=1e-12)


def test_contrast_table_q_is_bh_within_metric(table):
    for metric, g in table.groupby("metric"):
        np.testing.assert_allclose(g["q"].to_numpy(), bh_fdr(g["p"].to_numpy()))


def test_contrast_table_meaningful_rule_and_ci(table):
    me = table["metric"].map(MIN_EFFECT)
    expected = (table["q"] <= 0.05) & (table["mean"].abs() >= me)
    assert (table["sig"] == (table["q"] <= 0.05)).all()
    assert (table["meaningful"] == expected).all()
    assert (table["lo"] <= table["mean"] + 1e-12).all() and (table["mean"] <= table["hi"] + 1e-12).all()


def test_contrast_table_planted_effect_and_placebo(table):
    planted = table[(table["layer"] == 0) & (table["head"] == 0)]
    for c in ("total", "order"):
        g = planted[planted["contrast"] == c]
        assert g["meaningful"].all(), c
        assert (g["mean"] > 0).all()
    assert not table.loc[table["contrast"] == "placebo", "meaningful"].any()


def test_contrast_table_name_column(table):
    r = table.iloc[0]
    assert r["name"] == f"L{r['layer']}H{r['head']}"


# ---- tv_table / cell_summary --------------------------------------------

def test_tv_table_identical_and_disjoint():
    ident = np.ones((10, L, H), dtype=int)
    labels = {("b", "p", "near", v): ident for v in VARIANTS}
    tv = tv_table(labels, CONTRASTS)
    assert (tv["tv"] == 0).all()
    labels[("b", "p", "near", "nl")] = np.full((10, L, H), 2)
    tv = tv_table(labels, CONTRASTS)
    assert (tv.loc[tv["contrast"] == "total", "tv"] == 1.0).all()
    assert (tv.loc[tv["contrast"] == "placebo", "tv"] == 0.0).all()
    assert len(tv) == len(CONTRASTS) * L * H


def test_cell_summary_mode_and_means():
    rng = np.random.default_rng(0)
    key = ("rare", "sushi", "near", "none")
    vals = {"sink_mass": rng.random((N, L, H)), "entropy": rng.random((N, L, H))}
    labels = np.ones((N, L, H), dtype=int)  # Sink everywhere
    labels[:20, 1, 2] = 4                   # minority Other at one head
    labels[:, 0, 1] = 3                     # Self at L0H1
    df = cell_summary({key: vals}, {key: labels})
    assert len(df) == L * H
    row = df[(df["layer"] == 1) & (df["head"] == 2)].iloc[0]
    assert row["mode_label"] == LABELS[1] and row["name"] == "L1H2"
    assert df[(df["layer"] == 0) & (df["head"] == 1)].iloc[0]["mode_label"] == "Self"
    for _, r in df.iterrows():
        assert r["mean_sink_mass"] == pytest.approx(vals["sink_mass"][:, r["layer"], r["head"]].mean())
        assert r["mean_entropy"] == pytest.approx(vals["entropy"][:, r["layer"], r["head"]].mean())


# ---- pool_over_phrases ---------------------------------------------------

def test_pool_over_phrases_two_phrases():
    cells = make_cells(phrases=("a", "b"))
    eff = contrast_table(cells, CONTRASTS, METRICS, BOOT, MIN_EFFECT, 0.05)
    pooled = pool_over_phrases(eff)
    assert (pooled["n_phrases"] == 2).all()
    assert len(pooled) == len(eff) // 2
    keys = ["base", "placement", "contrast", "metric", "layer", "head"]
    for _, r in pooled.iterrows():
        g = eff
        for k in keys:
            g = g[g[k] == r[k]]
        assert len(g) == 2
        assert r["mean_of_means"] == pytest.approx(g["mean"].mean())
        assert r["min_mean"] == pytest.approx(g["mean"].min())
        assert r["max_mean"] == pytest.approx(g["mean"].max())
        assert r["n_meaningful_pos"] == int((g["meaningful"] & (g["mean"] > 0)).sum())
        assert r["n_meaningful_neg"] == int((g["meaningful"] & (g["mean"] < 0)).sum())
    top = pooled[(pooled["contrast"] == "total") & (pooled["layer"] == 0) & (pooled["head"] == 0)
                 & (pooled["metric"] == "sink_mass")]
    assert (top["n_meaningful_pos"] == 2).all() and (top["n_meaningful_neg"] == 0).all()


def test_pool_over_phrases_counts_negative():
    rows = []
    for phrase, mean in (("a", -0.1), ("b", -0.2), ("c", 0.3)):
        rows.append(dict(base="rare", phrase=phrase, placement="near", contrast="total", metric="sink_mass",
                         layer=0, head=0, name="L0H0", mean=mean, meaningful=True))
    out = pool_over_phrases(pd.DataFrame(rows)).iloc[0]
    assert out["n_phrases"] == 3 and out["n_meaningful_neg"] == 2 and out["n_meaningful_pos"] == 1
    assert out["mean_of_means"] == pytest.approx(0.0)


# ---- check_prediction ----------------------------------------------------

def eff_df(rows):
    return pd.DataFrame(rows, columns=["base", "phrase", "placement", "contrast", "metric", "name", "mean", "meaningful"])


def rowsfor(placement, means, contrast="total", metric="sink_mass", base="rare", phrase="p", meaningful=None):
    return [(base, phrase, placement, contrast, metric, f"L0H{i}", m,
             bool(meaningful[i]) if meaningful is not None else False) for i, m in enumerate(means)]


def test_check_far_is_small():
    spec = {"id": "P1", "kind": "far_is_small", "contrast": "total", "metric": "sink_mass", "placement": "far",
            "max_abs": 0.1, "max_median_abs": 0.015}
    good = eff_df(rowsfor("far", [0.001, -0.002, 0.003, 0.05]))
    bad_max = eff_df(rowsfor("far", [0.001, -0.002, 0.003, 0.2]))
    bad_med = eff_df(rowsfor("far", [0.05, 0.06, 0.07, 0.08]))
    assert check_prediction(spec, good, None)["verdict"] == "PASS"
    assert check_prediction(spec, bad_max, None)["verdict"] == "FAIL"
    assert check_prediction(spec, bad_med, None)["verdict"] == "FAIL"


def test_check_median_ratio():
    spec = {"id": "P2", "kind": "median_ratio", "contrast": "total", "metric": "sink_mass", "numerator": "near",
            "denominator": "far", "min_ratio": 3.0}
    good = eff_df(rowsfor("near", [0.3, 0.4, 0.5]) + rowsfor("far", [0.01, 0.02, 0.03]))
    bad = eff_df(rowsfor("near", [0.03, 0.04, 0.05]) + rowsfor("far", [0.02, 0.03, 0.04]))
    r = check_prediction(spec, good, None)
    assert r["verdict"] == "PASS" and r["id"] == "P2" and r["kind"] == "median_ratio"
    assert check_prediction(spec, bad, None)["verdict"] == "FAIL"


def test_check_heads_move():
    heads = ["L0H0", "L0H1", "L0H2", "L0H3"]
    spec = {"id": "P3", "kind": "heads_move", "contrast": "total", "metric": "sink_mass", "placement": "near",
            "base": "rare", "heads": heads, "min_abs": 0.05, "min_fraction": 0.5}
    good = eff_df(rowsfor("near", [0.1, -0.2, 0.0, 0.0]))
    bad = eff_df(rowsfor("near", [0.1, 0.0, 0.0, 0.0]))
    other_base = eff_df(rowsfor("near", [0.5] * 4, base="common"))
    assert check_prediction(spec, good, None)["verdict"] == "PASS"
    assert check_prediction(spec, bad, None)["verdict"] == "FAIL"
    assert check_prediction(spec, pd.concat([good, other_base]), None)["verdict"] == "PASS"


def test_check_few_meaningful():
    spec = {"id": "P4", "kind": "few_meaningful", "contrast": "order", "metric": "sink_mass",
            "max_heads": {"near": 2, "far": 0}}
    mk = lambda near_m, far_m: eff_df(
        rowsfor("near", [0.1] * 4, contrast="order", meaningful=near_m)
        + rowsfor("far", [0.1] * 4, contrast="order", meaningful=far_m))
    assert check_prediction(spec, mk([1, 1, 0, 0], [0, 0, 0, 0]), None)["verdict"] == "PASS"
    assert check_prediction(spec, mk([1, 1, 1, 0], [0, 0, 0, 0]), None)["verdict"] == "FAIL"
    assert check_prediction(spec, mk([0, 0, 0, 0], [0, 1, 0, 0]), None)["verdict"] == "FAIL"


def test_check_controls():
    spec = {"id": "P5", "kind": "controls", "min_mean": {"L4H11": ["prev_mass", 0.95]},
            "mode_label": {"L0H1": "Self"}}
    cells = pd.DataFrame({"name": ["L4H11", "L4H11", "L0H1", "L0H1"], "mean_prev_mass": [0.97, 0.99, 0.0, 0.0],
                          "mode_label": ["Previous", "Previous", "Self", "Self"]})
    assert check_prediction(spec, None, cells)["verdict"] == "PASS"
    low = cells.copy()
    low.loc[0, "mean_prev_mass"] = 0.5
    assert check_prediction(spec, None, low)["verdict"] == "FAIL"
    wrong = cells.copy()
    wrong.loc[2, "mode_label"] = "Sink"
    assert check_prediction(spec, None, wrong)["verdict"] == "FAIL"


def test_check_unknown_kind_raises():
    with pytest.raises(ValueError):
        check_prediction({"id": "X", "kind": "bogus"}, eff_df([]), pd.DataFrame())
