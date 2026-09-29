"""exp1ext_report.md: checks, 2x2 decomposition, dose-response, per-token effects, token features, stage 0."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from exp1.validation import md_table, verdict

BACKEND_NAMES = {"transformer_lens": "TL", "huggingface": "HF"}


def _ci(est: float, lo: float, hi: float, fmt: str = "+.3f") -> str:
    mark = "**" if lo > 0 or hi < 0 else ""  # bold when the CI excludes 0
    return f"{mark}{est:{fmt}}{mark} [{lo:{fmt}}, {hi:{fmt}}]"


def _driver(q: tuple[float, float, float], c: tuple[float, float, float]) -> str:
    """Which single change carries the rare→common shift: the final token (query) or the context."""
    q_sig, c_sig = q[1] > 0 or q[2] < 0, c[1] > 0 or c[2] < 0
    if q_sig and c_sig:
        ratio = abs(q[0]) / max(abs(c[0]), 1e-12)
        return "query" if ratio >= 2 else "context" if ratio <= 0.5 else "both"
    return "query" if q_sig else "context" if c_sig else "neither"


def _tok(text: str, token_id: int) -> str:
    shown = repr(text)[1:-1].replace("|", "\\|") or "∅"
    return f"`{shown}`({token_id})"


def write_report(path: Path, cfg: dict[str, Any], run_id: str, R: dict[str, Any]) -> None:
    heads: pd.DataFrame = R["heads"]
    decomp, tv_cells = R["decomp"], R["tv_cells"]
    ci_pct = int(round(100 * cfg["stats"]["bootstrap_ci"]))
    v, sw = cfg["validation"], cfg["sweep"]
    L = cfg["seq_len"]
    out = [
        "# exp1ext: which tokens cause the rare-vs-common head changes",
        "",
        f"- Run `{run_id}`. Rare = ids {cfg['ranges']['rare']}, common = ids {cfg['ranges']['common']} (byte tokens enter only as "
        f"swept tokens). Input = BOS + {L} random ids, no repeats within a sequence; metric at the final query (position {L}).",
        f"- Predictions were written before any data: [PREDICTIONS.md](PREDICTIONS.md). Bold = {ci_pct}% bootstrap CI excludes 0.",
        f"- Stage 1: {cfg['factorial']['n_per_cell']} sequences per 2×2 cell, dose-response {cfg['dose_response']['n']} per k. "
        f"Stage 2: {sw['n_bases_per_class']} base sequences per class; final-token sweep over ids {sw['common_tokens'][0]}–"
        f"{sw['common_tokens'][1]} + {sw['n_rare_tokens']} rare ids; substitutions at positions {sw['substitution_positions']}.",
        "",
    ]

    # ---- checks ----
    ch, cross, rep = R["checks"], R["cross"], R["replication"]
    rows = []
    for b, dev in ch["row_sum_dev"].items():
        rows.append(("Attention rows sum to 1", BACKEND_NAMES[b], verdict(dev <= v["row_sum_atol"]), f"max abs(sum − 1) = {dev:.1e}"))
    for b, (eq, diff) in ch["determinism"].items():
        rows.append((f"Determinism (first {v['determinism_n']} re-extracted)", BACKEND_NAMES[b], verdict(eq),
                     "bit-identical" if eq else f"max diff {diff:.1e}"))
    for b, c in cross.items():
        rows.append(("Cross-library, stage 1 (all cells)", f"TL vs {BACKEND_NAMES[b]}", verdict(c["stage1_max_diff"] <= v["cross_backend_atol"]),
                     f"max |Δ| probability metrics {c['stage1_max_diff']:.1e}; labels agree {100 * c['stage1_label_agreement']:.2f}%"))
        rows.append(("Cross-library, stage 2 subset", f"TL vs {BACKEND_NAMES[b]}", verdict(c["sweep_max_diff"] <= v["cross_backend_atol"]),
                     f"{sw['hf_bases_per_class']} bases per class × all tokens × all positions; max |Δ| {c['sweep_max_diff']:.1e}"))
    for cell, cond in (("AA", "A_rare"), ("CC", "C_common_clean")):
        rows.append((f"Replicates exp1 ({cell} vs {cond})", "TL", verdict(rep[cell] >= v["replication_min_r"]),
                     f"per-head mean sink_mass, r = {rep[cell]:.4f} across all heads (threshold {v['replication_min_r']})"))
    out += ["## Checks", "", md_table(pd.DataFrame(rows, columns=["check", "backend", "result", "detail"])), ""]

    # ---- tracked heads ----
    tv_cols = [c for c in heads.columns if c.startswith("tv_")]
    out += ["## 1. Tracked heads", "",
            f"Selected from exp1 by label-mix TV ≥ {cfg['head_selection']['min_tv']} (A_rare vs each common condition); group = "
            "direction of the sink_mass change from rare to common. Controls were fixed in advance.", "",
            md_table(heads[["name", "group", "both", *tv_cols, "sink_change"]]), ""]

    # ---- 2x2 ----
    sm = decomp["sink_mass"]
    at = lambda key, part, l, h: float(sm[key][part][l, h])
    rows, drivers = [], []
    tv_aacc = tv_cells[("AA", "CC")]
    for r in heads.itertuples():
        l, h = r.layer, r.head
        q = tuple(at("query", k, l, h) for k in ("est", "lo", "hi"))
        c = tuple(at("context", k, l, h) for k in ("est", "lo", "hi"))
        d = _driver(q, c)
        drivers.append({"name": r.name, "group": r.group, "both": r.both, "driver": d,
                        "tv_aacc": float(tv_aacc[l, h]), "total": at("total", "est", l, h)})
        rows.append({"head": r.name, "group": r.group,
                     **{c_: f"{at(f'cell_{c_}', 'est', l, h):.3f}" for c_ in ("AA", "AC", "CA", "CC")},
                     "total CC−AA": _ci(*(at("total", k, l, h) for k in ("est", "lo", "hi"))),
                     "query AC−AA": _ci(*q), "context CA−AA": _ci(*c),
                     "interaction": _ci(*(at("interaction", k, l, h) for k in ("est", "lo", "hi"))),
                     "driver": d, "TV AA→CC": f"{tv_aacc[l, h]:.2f}"})
    out += ["## 2. 2×2: is the change caused by the final token or by the context?", "",
            "sink_mass per cell (first letter = context, second = final token; A = rare, C = common). query = change only the final "
            "token; context = change only the 31 tokens before it. driver = the component whose CI excludes 0 (both if they are within "
            "2× of each other).", "", md_table(pd.DataFrame(rows)), ""]
    dr = pd.DataFrame(drivers)
    tracked = dr[dr["group"] != "control"]
    summary = tracked.groupby("group")["driver"].value_counts().unstack(fill_value=0)
    out += ["Driver counts by group (tracked heads):", "", md_table(summary.reset_index()), ""]
    both = dr[dr["both"]]
    exp1_sign = dict(zip(heads["name"], np.sign(heads["sink_change"])))
    replicated = both[(both["tv_aacc"] >= cfg["head_selection"]["min_tv"]) &
                      (np.sign(both["total"]) == both["name"].map(exp1_sign))]
    out += [f"Replication on fresh sequences: {len(replicated)}/{len(both)} of the heads selected in both exp1 comparisons again have "
            f"TV(AA, CC) ≥ {cfg['head_selection']['min_tv']} with the same sink direction. Split-half TV noise floor within a pure "
            f"cell: max {R['tv_floor']:.3f} over all heads.", ""]

    # ---- dose-response ----
    ks = R["ks"]
    dc = R["dose_curves"]["sink_mass"]
    rows = []
    for j, r in enumerate(heads.itertuples()):
        means = np.array([dc[k][0][j] for k in ks])
        change = means[-1] - means[0]
        frac = (means - means[0]) / change if abs(change) > 1e-9 else np.full(len(ks), np.nan)
        half_k = next((k for k, f in zip(ks, frac) if f >= 0.5), None) if np.isfinite(frac).all() else None
        rows.append({"head": r.name, "group": r.group, f"k=0": f"{means[0]:.3f}", f"k={ks[-1]}": f"{means[-1]:.3f}",
                     "change": f"{change:+.3f}", "k reaching half": half_k if half_k is not None else "—",
                     "k=1 share": f"{frac[1]:.2f}" if np.isfinite(frac[1]) else "—"})
    out += [f"## 3. Dose-response: how many common context tokens does it take?", "",
            f"sink_mass as k of the 31 context tokens of a rare sequence become common (final token stays rare; k={ks[-1]} equals the "
            "CA cell). 'k reaching half' = smallest k with at least half of the full change; a small value means a few tokens "
            "suffice, a value near the end means the aggregate context matters.", "", md_table(pd.DataFrame(rows)), ""]

    # ---- per-token sweeps ----
    eff, rel, ho = R["effects"], R["reliability"], R["heldout"]
    top_n = 5
    for kind, pos_list in (("query", [L]), ("substitution", sw["substitution_positions"])):
        for p in pos_list:
            title = ("## 4. Final-token sweep: which final tokens move each head?" if kind == "query"
                     else f"## {5 if p == sw['substitution_positions'][0] else 6}. Substitution at position {p}: which context tokens move each head?")
            what = ("mean sink_mass with that token in the final slot" if kind == "query"
                    else "Δ sink_mass vs the unsubstituted base (common tokens into rare bases; rare tokens into common bases)")
            out += [title, "", f"Per token: {what}, averaged over bases. reliability = correlation of per-token means between two "
                    "random halves of the bases (only read rankings where it is high); held-out = the top-minus-bottom gap of the "
                    f"top/bottom {sw['top_k']} tokens chosen on one half, measured on the other half (shrinkage = winner's curse).", ""]
            rows = []
            for r in heads.itertuples():
                for g in ("rare contexts", "common contexts"):
                    sel = eff[(eff["position"] == p) & (eff["metric"] == "sink_mass") & (eff["head"] == r.name) & (eff["group"] == g)]
                    sel = sel[np.isfinite(sel["mean"])]
                    if not len(sel):
                        continue
                    rr = rel[(rel["position"] == p) & (rel["metric"] == "sink_mass") & (rel["head"] == r.name) & (rel["group"] == g)]
                    hh = ho[(ho["position"] == p) & (ho["head"] == r.name) & (ho["group"] == g)]
                    s = sel.sort_values("mean")
                    rows.append({"head": r.name, "group": r.group, "bases": g.split()[0],
                                 "reliability": f"{rr['reliability_r'].iloc[0]:.2f}",
                                 "held-out gap": f"{hh['heldout_gap'].iloc[0]:.3f} / {hh['selection_gap'].iloc[0]:.3f}",
                                 "sd over tokens": f"{s['mean'].std():.3f}",
                                 f"highest {top_n}": ", ".join(_tok(t, i) for t, i in zip(s["text"][::-1][:top_n], s["token_id"][::-1][:top_n])),
                                 f"lowest {top_n}": ", ".join(_tok(t, i) for t, i in zip(s["text"][:top_n], s["token_id"][:top_n]))})
            out += [md_table(pd.DataFrame(rows)), ""]
            if kind == "query":
                cls = eff[(eff["position"] == p) & (eff["metric"] == "sink_mass") & (eff["group"] == "rare contexts")]
                by_cls = cls.groupby(["head", "class"])["mean"].mean().unstack()
                by_cls = by_cls.reindex(heads["name"]).reset_index()
                out += ["Mean sink_mass by final-token class (rare bases):", "", md_table(by_cls, ".3f"), ""]

    # ---- features ----
    fm = R["feature_models"]
    sub = fm[(fm["position"] == L) & (fm["group"] == "rare contexts") & (fm["term"] != "intercept")]
    if len(sub):
        wide = sub.assign(cell=[_ci(c, lo, hi) for c, lo, hi in zip(sub["coef"], sub["lo"], sub["hi"])]) \
            .pivot(index="head", columns="term", values="cell").reindex([n for n in heads["name"] if n in set(sub["head"])])
        r2 = sub.groupby("head")["r2"].first()
        wide.insert(0, "R²", [f"{r2[h]:.2f}" for h in wide.index])
        out += ["## 7. Which token properties explain the final-token effect?", "",
                f"OLS of each head's per-token mean sink_mass (final-token sweep, rare bases) on token features; coefficient "
                f"[{ci_pct}% CI]. log_prior = the model's mean log p(token | base) for the final slot, a frequency/familiarity "
                "proxy. Features are correlated, so read signs and CIs, not magnitudes, and do not over-read single heads.", "",
                md_table(wide.reset_index()), ""]

    # ---- stage 0 ----
    rc = R["received"]
    rows = []
    for r in heads.itertuples():
        for cond in cfg["stage0"]["received_conditions"]:
            s = rc[(rc["head"] == r.name) & (rc["condition"] == cond)].sort_values("mean_logratio", ascending=False)
            rows.append({"head": r.name, "condition": cond,
                         "top by log(p_k/p_sink)": ", ".join(_tok(t, i) for t, i in zip(s["text"][:top_n], s["token_id"][:top_n]))})
    out += ["## 8. Observational (exp1 data): which key tokens absorb attention?", "",
            "Mean log(p_k / p_sink) over each token's occurrences as a key (positions 1–31) in exp1's B and C sequences. The log-ratio "
            "cancels the softmax normalizer, so unlike raw p_k it is not pushed around by the other keys.", "", md_table(pd.DataFrame(rows)), ""]

    out += ["## Figures", "",
            "- `figures/stage1_interaction_*.png`, `figures/stage1_decomposition_*.png`, `figures/stage1_dose_*.png`: the 2×2 and dose-response",
            "- `figures/tokens/<head>_p<position>_<metric>.png`: per-token graphs (the final-token sweep is position "
            f"{L}; substitutions at {sw['substitution_positions']})",
            "- `figures/stage2_token_head_heatmap_p*.png`: tokens with the largest effects across tracked heads",
            "- `figures/overlay/<head>.png`: exp1 sequences with each token shaded by the attention it receives",
            "- `figures/received/<head>_<condition>.png`: attention received per key token (exp1 data)", "",
            "## Caveats", "",
            "- A substitution at position p < 32 changes positions p..32, so it measures the token's total effect: as a key and "
            "through earlier heads' writes into later positions. Only the final-token sweep isolates a single causal route (the query).",
            "- Per-token means average over bases of one class; a token's effect can differ between rare and common bases, which is "
            "why both are shown.",
            "- Tracked heads were chosen post hoc on exp1 data; the replication line above measures that on fresh sequences.", ""]
    path.write_text("\n".join(out))
