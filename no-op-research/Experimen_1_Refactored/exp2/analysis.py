"""Paired contrasts between exp2 variants, per (layer, head), and the pre-registered prediction checks.

Every variant of a base row shares the row's tokens outside the slots, so a contrast a - b is a per-row
difference d. It is tested with a two-sided Wilcoxon signed-rank test (normal approximation with tie correction,
zeros dropped, as scipy's method="approx"); the effect size is the matched-pairs rank-biserial correlation
r = (R+ - R-) / (R+ + R-) over the nonzero |d| ranks (+1: a > b in every row with a difference). Means get a
percentile bootstrap CI over rows. BH-FDR runs over every test of one metric.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

from exp1.compare import bh_fdr, label_shift
from exp1.metrics import LABELS
from exp1ext.effects import bootstrap_mean_per_unit

Key = tuple[str, str, str, str]  # (base, phrase, placement, variant)


def head_name(layer: int, head: int) -> str:
    return f"L{layer}H{head}"


def parse_head(name: str) -> tuple[int, int]:
    layer, head = name[1:].split("H")
    return int(layer), int(head)


def paired_test(d: np.ndarray) -> dict[str, np.ndarray]:
    """d: [n, ...] per-row differences. Per trailing index: two-sided Wilcoxon p (1.0 where every d is 0) and
    the matched-pairs rank-biserial effect size (0.0 where every d is 0). Returns arrays of shape d.shape[1:]."""
    flat = np.asarray(d, dtype=np.float64).reshape(d.shape[0], -1)
    p = np.ones(flat.shape[1])
    r = np.zeros(flat.shape[1])
    for j in range(flat.shape[1]):
        x = flat[:, j]
        x = x[x != 0]
        n = len(x)
        if n == 0:
            continue
        ranks = rankdata(np.abs(x))
        r_plus, r_minus = ranks[x > 0].sum(), ranks[x < 0].sum()
        r[j] = (r_plus - r_minus) / (r_plus + r_minus)
        _, counts = np.unique(np.abs(x), return_counts=True)
        var = n * (n + 1) * (2 * n + 1) / 24.0 - (counts ** 3 - counts).sum() / 48.0
        if var > 0:
            z = (r_plus - n * (n + 1) / 4.0) / np.sqrt(var)
            p[j] = 2.0 * norm.sf(abs(z))
    return {"p": p.reshape(d.shape[1:]), "effect_size": r.reshape(d.shape[1:])}


def contrast_table(cells: dict[Key, dict[str, np.ndarray]], contrasts: dict[str, Sequence[str]], metrics: Sequence[str],
                   boot: dict[str, Any], min_effect: dict[str, float], fdr_q: float) -> pd.DataFrame:
    """cells: (base, phrase, placement, variant) -> {metric: per-row values [n, n_layers, n_heads]}, row-aligned
    across the variants of one (base, phrase, placement).

    One row per (base, phrase, placement, contrast, metric, layer, head): mean difference a - b with bootstrap CI,
    Wilcoxon p, BH q (over all rows of the metric), rank-biserial effect_size, sig (q <= fdr_q), and meaningful
    (sig and |mean| >= min_effect[metric])."""
    groups = sorted({k[:3] for k in cells})
    rows = []
    for base, phrase, placement in groups:
        for cname, (a, b) in contrasts.items():
            if (base, phrase, placement, a) not in cells or (base, phrase, placement, b) not in cells:
                continue  # e.g. a single-token variant defined for one placement only
            for metric in metrics:
                d = cells[(base, phrase, placement, a)][metric] - cells[(base, phrase, placement, b)][metric]
                mean, lo, hi = bootstrap_mean_per_unit(d, **boot)
                test = paired_test(d)
                n_layers, n_heads = d.shape[1:]
                for layer in range(n_layers):
                    for head in range(n_heads):
                        rows.append((base, phrase, placement, cname, metric, layer, head, mean[layer, head],
                                     lo[layer, head], hi[layer, head], test["p"][layer, head],
                                     test["effect_size"][layer, head]))
    df = pd.DataFrame(rows, columns=["base", "phrase", "placement", "contrast", "metric", "layer", "head", "mean",
                                     "lo", "hi", "p", "effect_size"])
    df["q"] = np.nan
    for metric, idx in df.groupby("metric").groups.items():
        df.loc[idx, "q"] = bh_fdr(df.loc[idx, "p"].to_numpy())
    df["sig"] = df["q"] <= fdr_q
    df["meaningful"] = df["sig"] & (df["mean"].abs() >= df["metric"].map(min_effect))
    df.insert(5, "name", [head_name(l, h) for l, h in zip(df["layer"], df["head"])])
    return df


def tv_table(labels: dict[Key, np.ndarray], contrasts: dict[str, Sequence[str]]) -> pd.DataFrame:
    """Label-mix total variation distance between the two variants of each contrast, per head."""
    rows = []
    for base, phrase, placement in sorted({k[:3] for k in labels}):
        for cname, (a, b) in contrasts.items():
            if (base, phrase, placement, a) not in labels or (base, phrase, placement, b) not in labels:
                continue
            tv = label_shift(labels[(base, phrase, placement, a)], labels[(base, phrase, placement, b)], len(LABELS))["tv"]
            for (layer, head), value in np.ndenumerate(tv):
                rows.append((base, phrase, placement, cname, layer, head, head_name(layer, head), float(value)))
    return pd.DataFrame(rows, columns=["base", "phrase", "placement", "contrast", "layer", "head", "name", "tv"])


def cell_summary(cells: dict[Key, dict[str, np.ndarray]], labels: dict[Key, np.ndarray]) -> pd.DataFrame:
    """Per (base, phrase, placement, variant, layer, head): mean of every metric and the mode label."""
    rows = []
    for key in sorted(cells):
        means = {m: v.mean(axis=0) for m, v in cells[key].items()}
        counts = np.stack([(labels[key] == k).sum(axis=0) for k in range(len(LABELS))])  # [label, layer, head]
        mode = counts.argmax(axis=0)  # ties -> lowest code, as exp1.aggregate
        for (layer, head), m in np.ndenumerate(mode):
            rows.append({"base": key[0], "phrase": key[1], "placement": key[2], "variant": key[3], "layer": layer,
                         "head": head, "name": head_name(layer, head), "mode_label": LABELS[m],
                         **{f"mean_{name}": float(v[layer, head]) for name, v in means.items()}})
    return pd.DataFrame(rows)


def pool_over_phrases(effects: pd.DataFrame) -> pd.DataFrame:
    """Per (base, placement, contrast, metric, head), across phrases: mean of the per-phrase mean differences, their
    range, and how many phrases give a meaningful effect of each sign. Phrases are the units here, so with few
    phrases read the counts, not the mean."""
    g = effects.groupby(["base", "placement", "contrast", "metric", "layer", "head", "name"])
    out = g.agg(n_phrases=("phrase", "nunique"), mean_of_means=("mean", "mean"), min_mean=("mean", "min"),
                max_mean=("mean", "max")).reset_index()
    pos = effects[effects["meaningful"] & (effects["mean"] > 0)].groupby(
        ["base", "placement", "contrast", "metric", "layer", "head"]).size().rename("n_meaningful_pos")
    neg = effects[effects["meaningful"] & (effects["mean"] < 0)].groupby(
        ["base", "placement", "contrast", "metric", "layer", "head"]).size().rename("n_meaningful_neg")
    keys = ["base", "placement", "contrast", "metric", "layer", "head"]
    out = out.merge(pos, left_on=keys, right_index=True, how="left").merge(neg, left_on=keys, right_index=True, how="left")
    return out.fillna({"n_meaningful_pos": 0, "n_meaningful_neg": 0}).astype({"n_meaningful_pos": int, "n_meaningful_neg": int})


# ---- additivity: does a combination do more than the sum of its single-token parts? --------------------------

def _depth_mean(x: np.ndarray, layers: Sequence[int]) -> np.ndarray:
    """x: [n, n_layers, n_heads] -> [n]: mean over the given layers' heads."""
    return x[:, list(layers), :].mean(axis=(1, 2))


def interactions(cells: dict[Key, dict[str, np.ndarray]], combos: dict[str, dict[str, Any]], metric: str) -> dict[tuple, np.ndarray]:
    """Per (base, phrase, placement, variant): per-row interaction [n, L, H] = (combo − none) − Σ_j (part_j − none).
    combos: name -> {phrase, placement, parts: [single-token variant names], variant (default: the name)}, so several
    phrases can each have an "nl" combo. Zero under exact additivity of the single-token effects (relative to the
    same base row)."""
    out = {}
    for base in sorted({k[0] for k in cells}):
        for combo, spec in combos.items():
            g = (base, spec["phrase"], spec["placement"])
            none = cells[(*g, "none")][metric]
            additive = sum(cells[(*g, part)][metric] - none for part in spec["parts"])
            variant = spec.get("variant", combo)
            out[(*g, variant)] = (cells[(*g, variant)][metric] - none) - additive
    return out


def additivity_tables(cells: dict[Key, dict[str, np.ndarray]], combos: dict[str, dict[str, Any]],
                      differences: Sequence[Sequence[str]], depth_groups: dict[str, Sequence[int]], metric: str,
                      boot: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns
    - per_head: one row per (base, phrase, placement, combo, head): observed (combo − none), additive (Σ parts),
      interaction (mean, CI, Wilcoxon p, rank-biserial), for each combo
    - depth: per (base, phrase, placement, term, depth group): mean over the group's heads, per row, then a bootstrap
      CI over rows; term = a combo's interaction, or "a−b" for a pair in `differences` (interaction of a minus b)
    - order_fit: per (base, phrase, placement, pair): across heads, the observed difference between the two combos
      (e.g. nl − shuffled, the order contrast) against its additive prediction: slope, r, r²."""
    inter = interactions(cells, combos, metric)
    rows, depth_rows, fit_rows = [], [], []
    for (base, phrase, placement, combo), x in inter.items():
        g = (base, phrase, placement)
        none = cells[(*g, "none")][metric]
        observed = (cells[(*g, combo)][metric] - none).mean(axis=0)
        mean, lo, hi = bootstrap_mean_per_unit(x, **boot)
        test = paired_test(x)
        for (layer, head), m in np.ndenumerate(mean):
            rows.append((base, phrase, placement, combo, layer, head, head_name(layer, head), observed[layer, head],
                         observed[layer, head] - m, m, lo[layer, head], hi[layer, head], test["p"][layer, head],
                         test["effect_size"][layer, head]))
        for depth, layers in depth_groups.items():
            dm, dlo, dhi = bootstrap_mean_per_unit(_depth_mean(x, layers)[:, None], **boot)
            depth_rows.append((base, phrase, placement, combo, depth, float(dm[0]), float(dlo[0]), float(dhi[0])))
    for a, b in differences:
        for base, phrase, placement in sorted({k[:3] for k in inter}):
            g = (base, phrase, placement)
            if (*g, a) not in inter or (*g, b) not in inter:
                continue
            d = inter[(*g, a)] - inter[(*g, b)]
            for depth, layers in depth_groups.items():
                dm, dlo, dhi = bootstrap_mean_per_unit(_depth_mean(d, layers)[:, None], **boot)
                depth_rows.append((base, phrase, placement, f"{a}−{b}", depth, float(dm[0]), float(dlo[0]), float(dhi[0])))
            observed = (cells[(*g, a)][metric] - cells[(*g, b)][metric]).mean(axis=0).ravel()
            additive = observed - d.mean(axis=0).ravel()
            slope = float(np.polyfit(additive, observed, 1)[0])
            r = float(np.corrcoef(additive, observed)[0, 1])
            fit_rows.append((base, phrase, placement, f"{a}−{b}", slope, r, r * r,
                             float(np.abs(observed).mean()), float(np.abs(observed - additive).mean())))
    per_head = pd.DataFrame(rows, columns=["base", "phrase", "placement", "combo", "layer", "head", "name", "observed",
                                           "additive", "interaction", "lo", "hi", "p", "effect_size"])
    depth = pd.DataFrame(depth_rows, columns=["base", "phrase", "placement", "term", "depth", "mean", "lo", "hi"])
    order_fit = pd.DataFrame(fit_rows, columns=["base", "phrase", "placement", "pair", "slope", "r", "r2",
                                                "mean_abs_observed", "mean_abs_interaction"])
    return per_head, depth, order_fit


# ---- surprisal: does sink_mass track how predictable the slot tokens are? ------------------------------------

def surprisal_slopes(cells: dict[Key, dict[str, np.ndarray]], nll: dict[Key, np.ndarray], metric: str,
                     depth_groups: dict[str, Sequence[int]], boot: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Within-row regression of `metric` on the slot tokens' summed negative log-likelihood (nats), across every
    variant of each (base, phrase, placement): both are demeaned within each row, so differences between rows
    (their own tokens) drop out and only the variant-to-variant change in a row is used.

    Returns (per_head slopes with bootstrap CI over rows; depth slopes for the mean over each depth group's heads;
    mean nll and mean metric-depth values per variant, descriptive)."""
    head_rows, depth_rows, var_rows = [], [], []
    rng_seed, iters, ci = boot["seed"], boot["iters"], boot["ci"]
    tail = (1.0 - ci) / 2.0 * 100.0
    for g in sorted({k[:3] for k in nll}):
        variants = [k[3] for k in nll if k[:3] == g]
        X = np.stack([nll[(*g, v)] for v in variants])                       # [V, n]
        Y = np.stack([cells[(*g, v)][metric] for v in variants])             # [V, n, L, H]
        Xc = X - X.mean(axis=0)
        Yc = Y - Y.mean(axis=0)
        sxx = (Xc ** 2).sum(axis=0)                                         # [n]
        sxy = np.einsum("vn,vnlh->nlh", Xc, Yc)                             # [n, L, H]
        n = len(sxx)
        idx = np.random.default_rng(rng_seed).integers(0, n, size=(iters, n))
        boot_sxx = sxx[idx].sum(axis=1)                                     # [iters]
        slope = sxy.sum(axis=0) / sxx.sum()
        boot_slope = np.stack([sxy[i].sum(axis=0) for i in idx]) / boot_sxx[:, None, None]
        lo, hi = np.percentile(boot_slope, tail, axis=0), np.percentile(boot_slope, 100 - tail, axis=0)
        for (layer, head), s in np.ndenumerate(slope):
            head_rows.append((*g, layer, head, head_name(layer, head), s, lo[layer, head], hi[layer, head]))
        for depth, layers in depth_groups.items():
            d = sxy[:, list(layers), :].mean(axis=(1, 2))                   # [n]
            s = d.sum() / sxx.sum()
            bs = d[idx].sum(axis=1) / boot_sxx
            depth_rows.append((*g, depth, float(s), float(np.percentile(bs, tail)), float(np.percentile(bs, 100 - tail))))
        for v, x, y in zip(variants, X, Y):
            var_rows.append({"base": g[0], "phrase": g[1], "placement": g[2], "variant": v, "mean_slot_nll": float(x.mean()),
                             **{f"mean_{metric}_{depth}": float(_depth_mean(y, layers).mean()) for depth, layers in depth_groups.items()}})
    cols = ["base", "phrase", "placement"]
    return (pd.DataFrame(head_rows, columns=cols + ["layer", "head", "name", "slope", "lo", "hi"]),
            pd.DataFrame(depth_rows, columns=cols + ["depth", "slope", "lo", "hi"]),
            pd.DataFrame(var_rows))


# ---- pre-registered prediction checks -------------------------------------------------------------------------

def _sel(effects: pd.DataFrame, **kw: Any) -> pd.DataFrame:
    mask = np.ones(len(effects), dtype=bool)
    for col, val in kw.items():
        mask &= (effects[col] == val).to_numpy()
    return effects[mask]


def phrase_level(depth: pd.DataFrame, variants: pd.DataFrame | None, boot: dict[str, Any]) -> pd.DataFrame:
    """Phrases as the unit. depth: additivity depth table (with a `metric` column) over several phrases; variants:
    surprisal_slopes' per-variant table (for the nl / shuffled surprisal gap), or None.

    One row per (metric, base, term, depth): mean over phrases of the per-phrase means, a percentile CI from
    bootstrapping phrases, how many phrases have a row-bootstrap CI excluding 0 on each side, and the Spearman
    correlation across phrases between the term and the surprisal gap NLL(shuffled) − NLL(nl)."""
    from scipy.stats import spearmanr

    tail = (1.0 - boot["ci"]) / 2.0 * 100.0
    gap = None
    if variants is not None:
        piv = variants.pivot_table(index=["base", "phrase"], columns="variant", values="mean_slot_nll")
        gap = (piv["shuffled"] - piv["nl"]).rename("nll_gap")
    rows = []
    for (metric, base, term, dname), g in depth.groupby(["metric", "base", "term", "depth"], sort=False):
        vals = g.set_index("phrase")["mean"]
        rng = np.random.default_rng(boot["seed"])
        bs = vals.to_numpy()[rng.integers(0, len(vals), size=(boot["iters"], len(vals)))].mean(axis=1)
        row = {"metric": metric, "base": base, "term": term, "depth": dname, "n_phrases": len(vals),
               "pooled_mean": float(vals.mean()), "lo": float(np.percentile(bs, tail)), "hi": float(np.percentile(bs, 100 - tail)),
               "n_pos": int((g["lo"] > 0).sum()), "n_neg": int((g["hi"] < 0).sum()), "spearman_rho": np.nan, "spearman_p": np.nan}
        if gap is not None and len(vals) > 2:
            x = gap.loc[base].reindex(vals.index)
            rho, pval = spearmanr(x.to_numpy(), vals.to_numpy())
            row.update(spearman_rho=float(rho), spearman_p=float(pval))
        rows.append(row)
    return pd.DataFrame(rows)


def _ci_has_sign(lo: float, hi: float, sign: str) -> bool:
    return lo > 0 if sign == "+" else hi < 0


def check_prediction(spec: dict[str, Any], effects: pd.DataFrame, cells: pd.DataFrame,
                     extra: dict[str, pd.DataFrame] | None = None) -> dict[str, Any]:
    """Evaluate one config `predictions` entry. Returns {id, kind, verdict ("PASS"/"FAIL"), detail}. A check
    that spans base classes or phrases must hold in every one of them. `extra` carries the additivity_tables
    ("additivity_depth", "order_fit") and surprisal_slopes ("surprisal_depth") outputs for the kinds that use them."""
    kind, out = spec["kind"], {"id": spec["id"], "kind": spec["kind"]}
    if kind in ("depth_term_sign", "surprisal_sign", "surprisal_depth_ratio", "order_fit_r2_below", "pooled_term_sign",
                "phrase_count", "phrase_spearman_below"):
        return out | _check_extra(spec, extra or {})
    sel = _sel(effects, contrast=spec.get("contrast"), metric=spec.get("metric")) if "contrast" in spec else effects
    parts, ok = [], True
    if kind == "far_is_small":
        for (base, phrase), g in _sel(sel, placement=spec["placement"]).groupby(["base", "phrase"]):
            a = g["mean"].abs()
            good = a.max() < spec["max_abs"] and a.median() <= spec["max_median_abs"]
            ok &= good
            parts.append(f"{base}/{phrase}: max |Δ| {a.max():.3f} ({g.loc[a.idxmax(), 'name']}), median {a.median():.4f}")
    elif kind == "median_ratio":
        for (base, phrase), g in sel.groupby(["base", "phrase"]):
            num = g.loc[g["placement"] == spec["numerator"], "mean"].abs().median()
            den = g.loc[g["placement"] == spec["denominator"], "mean"].abs().median()
            ratio = num / den if den > 0 else np.inf
            ok &= ratio >= spec["min_ratio"]
            parts.append(f"{base}/{phrase}: median |Δ| {num:.4f} vs {den:.4f}, ratio {ratio:.1f}")
    elif kind == "heads_move":
        g = _sel(sel, placement=spec["placement"], base=spec["base"])
        for phrase, gp in g.groupby("phrase"):
            gp = gp.set_index("name").loc[spec["heads"]]
            moved = gp.index[gp["mean"].abs() >= spec["min_abs"]].tolist()
            frac = len(moved) / len(spec["heads"])
            ok &= frac >= spec["min_fraction"]
            parts.append(f"{phrase}: {len(moved)}/{len(spec['heads'])} heads with |Δ| >= {spec['min_abs']} ({', '.join(moved) or 'none'})")
    elif kind == "few_meaningful":
        for (base, phrase, placement), g in sel.groupby(["base", "phrase", "placement"]):
            names = g.loc[g["meaningful"], "name"].tolist()
            limit = spec["max_heads"][placement]
            ok &= len(names) <= limit
            parts.append(f"{base}/{phrase}/{placement}: {len(names)} (limit {limit})" + (f": {', '.join(names[:12])}" if names else ""))
    elif kind == "controls":
        for name, (metric, floor) in spec.get("min_mean", {}).items():
            vals = cells.loc[cells["name"] == name, f"mean_{metric}"]
            ok &= bool((vals >= floor).all())
            parts.append(f"{name} {metric}: min over cells {vals.min():.3f} (floor {floor})")
        for name, label in spec.get("mode_label", {}).items():
            labs = cells.loc[cells["name"] == name, "mode_label"]
            ok &= bool((labs == label).all())
            parts.append(f"{name} mode {label} in {int((labs == label).sum())}/{len(labs)} cells")
    else:
        raise ValueError(f"unknown prediction kind {kind!r}")
    out.update(verdict="PASS" if ok else "FAIL", detail="; ".join(parts))
    return out


def _check_extra(spec: dict[str, Any], extra: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Prediction kinds over the additivity / surprisal / phrase-level summaries:
    - depth_term_sign: an additivity depth term (a combo's interaction, or "a−b") has CI excluding 0 with `sign`
    - surprisal_sign: the depth-group slope of the metric on slot nll has CI excluding 0 with `sign`
    - surprisal_depth_ratio: |slope| of depth `larger` exceeds |slope| of depth `smaller`
    - order_fit_r2_below: across heads, r² of observed vs additive for `pair` is below max_r2
    - pooled_term_sign: phrase_level pooled mean of a term has CI (over phrases) excluding 0 with `sign`
    - phrase_count: at least min_count phrases have the term's own CI excluding 0 with `sign`
    - phrase_spearman_below: across phrases, Spearman rho of the term vs the surprisal gap is below max_rho
    `metric` (default sink_mass) and `phrase` (default: every phrase present) filter the tables; each check must hold
    in every listed base (default: every base present) and, for per-phrase kinds, every selected phrase."""
    kind, parts, ok = spec["kind"], [], True
    if kind in ("pooled_term_sign", "phrase_count", "phrase_spearman_below"):
        t = extra["phrase_pooled"]
        t = t[(t["term"] == spec["term"]) & (t["depth"] == spec["depth"])]
    elif kind == "depth_term_sign":
        t = extra["additivity_depth"]
        t = t[(t["term"] == spec["term"]) & (t["depth"] == spec["depth"])]
    elif kind in ("surprisal_sign", "surprisal_depth_ratio"):
        t = extra["surprisal_depth"]
    else:
        t = extra["order_fit"]
        t = t[t["pair"] == spec["pair"]]
    if "metric" in t.columns:
        t = t[t["metric"] == spec.get("metric", "sink_mass")]
    if "phrase" in spec and "phrase" in t.columns:
        t = t[t["phrase"] == spec["phrase"]]
    bases = spec.get("bases", sorted(t["base"].unique()))
    for base in bases:
        gb = t[t["base"] == base]
        groups = gb.groupby("phrase", sort=False) if "phrase" in gb.columns else [("", gb)]
        for phrase, g in groups:
            tag = f"{base}/{phrase}" if phrase else base
            if kind == "depth_term_sign":
                r = g.iloc[0]
                good = _ci_has_sign(r["lo"], r["hi"], spec["sign"])
                parts.append(f"{tag}: {r['mean']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]")
            elif kind == "surprisal_sign":
                r = g[g["depth"] == spec["depth"]].iloc[0]
                good = _ci_has_sign(r["lo"], r["hi"], spec["sign"])
                parts.append(f"{tag}: slope {r['slope']:+.5f} [{r['lo']:+.5f}, {r['hi']:+.5f}] per nat")
            elif kind == "surprisal_depth_ratio":
                big = g.loc[g["depth"] == spec["larger"], "slope"].iloc[0]
                small = g.loc[g["depth"] == spec["smaller"], "slope"].iloc[0]
                good = abs(big) > abs(small)
                parts.append(f"{tag}: |{spec['larger']}| {abs(big):.5f} vs |{spec['smaller']}| {abs(small):.5f}")
            elif kind == "order_fit_r2_below":
                r = g.iloc[0]
                good = r["r2"] < spec["max_r2"]
                parts.append(f"{tag}: r² {r['r2']:.3f} (slope {r['slope']:.2f})")
            elif kind == "pooled_term_sign":
                r = g.iloc[0]
                good = _ci_has_sign(r["lo"], r["hi"], spec["sign"])
                parts.append(f"{tag}: {r['pooled_mean']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] over {r['n_phrases']} phrases")
            elif kind == "phrase_count":
                r = g.iloc[0]
                n = r["n_pos"] if spec["sign"] == "+" else r["n_neg"]
                good = n >= spec["min_count"]
                parts.append(f"{tag}: {n}/{r['n_phrases']} phrases (need {spec['min_count']}); opposite sign {r['n_neg'] if spec['sign'] == '+' else r['n_pos']}")
            else:
                r = g.iloc[0]
                good = bool(r["spearman_rho"] < spec["max_rho"])
                parts.append(f"{tag}: rho {r['spearman_rho']:+.2f} (p {r['spearman_p']:.3f})")
            ok &= bool(good)
    return {"verdict": "PASS" if ok else "FAIL", "detail": "; ".join(parts)}
