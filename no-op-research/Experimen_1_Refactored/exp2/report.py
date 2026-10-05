"""Write exp2_report.md from the outputs of exp2.run."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from exp1.validation import md_table, verdict

DECOMP = ("total", "order", "identity", "form")


def _fmt(mean: float, meaningful: bool) -> str:
    s = f"{mean:+.3f}"
    return f"**{s}**" if meaningful else s


def _counts(effects: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Meaningful heads per contrast (columns), one row per (base, phrase, placement)."""
    sel = effects[effects["metric"] == metric]
    table = sel.groupby(["base", "phrase", "placement", "contrast"], sort=False)["meaningful"].sum().unstack("contrast")
    return table[list(dict.fromkeys(sel["contrast"]))].reset_index()


def _sizes(effects: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Median |Δ| over heads per contrast (columns), one row per (base, phrase, placement), plus the largest head of
    the first contrast."""
    sel = effects[effects["metric"] == metric]
    contrasts = list(dict.fromkeys(sel["contrast"]))
    keys = ["base", "phrase", "placement"]
    med = sel.assign(a=sel["mean"].abs()).groupby(keys + ["contrast"], sort=False)["a"].median().unstack("contrast")[contrasts]
    med.columns = [f"{c} median |Δ|" for c in contrasts]
    first = sel[sel["contrast"] == contrasts[0]]
    top = first.loc[first.groupby(keys, sort=False)["mean"].apply(lambda s: s.abs().idxmax())].set_index(keys)
    med[f"largest {contrasts[0]}"] = [f"{r.name} ({r.mean:+.3f})" for r in top.reindex(med.index).itertuples()]
    return med.reset_index()


def _tracked(effects: pd.DataFrame, heads: pd.DataFrame, base: str, phrase: str, placement: str, metric: str) -> pd.DataFrame:
    sel = effects[(effects["base"] == base) & (effects["phrase"] == phrase) & (effects["placement"] == placement)
                  & (effects["metric"] == metric)].set_index(["name", "contrast"])
    rows = []
    for h in heads.itertuples():
        if (h.name, "total") not in sel.index:
            continue
        row = {"head": h.name, "group": h.group}
        for c in DECOMP:
            r = sel.loc[(h.name, c)]
            row[c] = _fmt(r["mean"], bool(r["meaningful"]))
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(path: Path, cfg: dict[str, Any], run_id: str, R: dict[str, Any]) -> None:
    effects, checks, heads = R["effects"], R["checks"], R["heads"]
    names = {"transformer_lens": "TL", "huggingface": "HF"}
    va = cfg["validation"]
    L: list[str] = []
    w = L.append

    w("# exp2: a natural-language phrase in random-token context\n")
    phrases = ", ".join(f"`{n}` = {ph['text']!r} {ph['ids']} (shuffled {ph['shuffled']})" for n, ph in cfg["phrases"].items())
    w(f"- Run `{run_id}`. Bases: " + ", ".join(f"{b} {r}" for b, r in cfg["bases"].items())
      + f", {cfg['n_bases']} rows each, BOS + 32 random ids, no repeats, no phrase ids. Metric at the final query (position 32).")
    w(f"- Phrases: {phrases}. Placements (model positions): "
      + ", ".join(f"{p} {pos}" for p, pos in cfg["placements"].items()) + ".")
    w(f"- Controls: `shuffled` (same ids, scrambled), `matched` / `matched_b` (per row, random ids with the same "
      f"{', '.join(cfg['match_features'])} and id class as each phrase token; pool sizes {R['info']['pool_sizes']}).")
    w("- Contrasts (paired per row, a − b): " + ", ".join(f"{c} = {a} − {b}" for c, (a, b) in cfg["contrasts"].items())
      + ". total = order + identity + form.")
    w(f"- Meaningful = BH q ≤ {cfg['stats']['fdr_q']} (Wilcoxon signed-rank, over every test of a metric) and "
      f"|Δ| ≥ the metric's threshold {cfg['min_effect']}. Bold = meaningful. Predictions were written before any data: "
      "[PREDICTIONS.md](PREDICTIONS.md).\n")

    w("## Checks\n")
    rows = []
    for b, dev in checks["row_sum_dev"].items():
        rows.append(("Attention rows sum to 1", names[b], verdict(dev <= va["row_sum_atol"]), f"max abs(sum − 1) = {dev:.1e}"))
    for b, (eq, diff) in checks["determinism"].items():
        rows.append((f"Determinism (first {va['determinism_n']} re-extracted)", names[b], verdict(eq),
                     "bit-identical" if eq else f"max diff {diff:.1e}"))
    rows.append(("Design invariants (no repeats, pairing, slot contents, pools)", "-", "PASS", "checked before extraction"))
    for b, r in checks["replication"].items():
        rows.append((f"`none` rows replicate exp1 ({b})", names[cfg["primary_backend"]],
                     verdict(r >= cfg["replication"]["min_r"]), f"per-head mean sink_mass r = {r:.4f}"))
    for b, c in checks["cross_backend"].items():
        mx = max(c["max_abs_diff"].values())
        rows.append(("Cross-library", f"TL vs {names[b]}", verdict(mx <= va["cross_backend_atol"]),
                     f"max |Δ| probability metrics {mx:.1e}; labels agree {100 * c['label_agreement']:.2f}%; "
                     f"meaningful decisions (sink_mass) agree {100 * c['meaningful_agreement']:.2f}%; "
                     f"max |Δ effect size| {c['max_effect_size_diff']:.1e}"))
    w(md_table(pd.DataFrame(rows, columns=["check", "backend", "result", "detail"])) + "\n")

    w("## 1. Pre-registered predictions\n")
    w(md_table(pd.DataFrame([(p["id"], p["kind"], p["verdict"], p["detail"]) for p in R["predictions"]],
                            columns=["id", "kind", "result", "detail"])) + "\n")

    per_phrase = cfg["figures"].get("per_phrase", list(cfg["phrases"]))
    if len(per_phrase) < len(cfg["phrases"]):
        w(f"Every phrase appears in every summary table. The long per-head tables (sections 3, 4, the head list in 7 and "
          f"the per-variant means in 8) and the per-phrase figures cover only {per_phrase}; the full per-head results "
          "for every phrase are in effects.parquet, additivity_heads.parquet and surprisal_*.parquet.\n")
    for metric in ("sink_mass", "slot_mass", "slot_logratio"):
        if metric not in cfg["metrics"]:
            continue
        w(f"## 2. Meaningful heads per contrast: {metric}\n")
        w("Out of 144 heads. The placebo row is the noise floor.\n")
        w(md_table(_counts(effects, metric), floatfmt=".0f") + "\n")
        w(f"Effect sizes ({metric}): \n")
        w(md_table(_sizes(effects, metric), floatfmt=".4f") + "\n")

    w("## 3. Tracked heads (exp1ext rule-selected + controls): Δ sink_mass\n")
    for b in cfg["bases"]:
        for name in per_phrase:
            for placement in cfg["placements"]:
                t = _tracked(effects, heads, b, name, placement, "sink_mass")
                if len(t):
                    w(f"**{b} bases, `{name}`, {placement}**\n")
                    w(md_table(t) + "\n")

    w("## 4. Does the final token attend to the phrase? Δ slot_mass, tracked heads\n")
    for b in cfg["bases"]:
        for name in per_phrase:
            t = _tracked(effects, heads, b, name, "near", "slot_mass") if "near" in cfg["placements"] else pd.DataFrame()
            if len(t):
                w(f"**{b} bases, `{name}`, near**\n")
                w(md_table(t) + "\n")

    w("## 5. Label-mix shift (TV distance)\n")
    tv = R["tv"]
    w("Split-half TV floor within the `none` rows: " + ", ".join(f"{b} {v:.3f}" for b, v in R["tv_floor"].items()) + ".\n")
    if len(cfg["phrases"]) > 1:
        above = tv.assign(above=tv["tv"] > tv["base"].map(R["tv_floor"]))
        w("Heads whose label-mix TV exceeds the floor, per contrast:\n")
        w(md_table(above.groupby(["base", "phrase", "placement", "contrast"], sort=False)["above"].sum()
                   .unstack("contrast").reset_index(), floatfmt=".0f") + "\n")
    else:
        rows = []
        for (contrast, base, phrase, placement), g in tv.groupby(["contrast", "base", "phrase", "placement"], sort=False):
            floor = R["tv_floor"][base]
            above = g[g["tv"] > floor].sort_values("tv", ascending=False)
            rows.append((contrast, base, phrase, placement, g["tv"].max(), len(above),
                         ", ".join(f"{r.name} ({r.tv:.2f})" for r in above.head(6).itertuples())))
        w(md_table(pd.DataFrame(rows, columns=["contrast", "base", "phrase", "placement", "max TV", "heads above floor", "top"])) + "\n")

    if R.get("pooled") is not None:
        w("## 6. Pooled over phrases (sink_mass, total)\n")
        p = R["pooled"]
        p = p[(p["metric"] == "sink_mass") & (p["contrast"] == "total")]
        top = p.reindex(p["mean_of_means"].abs().sort_values(ascending=False).index).head(15)
        w(md_table(top[["base", "placement", "name", "n_phrases", "mean_of_means", "min_mean", "max_mean",
                        "n_meaningful_pos", "n_meaningful_neg"]]) + "\n")

    extra = R.get("extra") or {}
    if "additivity_depth" in extra:
        ad = cfg["additivity"]
        w("## 7. Additivity: does the combination add more than its single tokens?\n")
        shown = {c: s for c, s in ad["combos"].items() if s["phrase"] in per_phrase}
        w("Per row, interaction = (combo − none) − Σ(part − none); e.g. "
          + "; ".join(f"`{c}` = {' + '.join(s['parts'])}" for c, s in shown.items())
          + ". Depth values average the group's heads within each row; CI = bootstrap over rows. Metrics: "
          + ", ".join(sorted(extra["additivity_depth"]["metric"].unique()))
          + " (sink_logit = log(p / (1 − p)), not squeezed by the probability bounds).\n")
        d = extra["additivity_depth"]
        d = d[d["phrase"].isin(per_phrase)].copy()
        d["CI excludes 0"] = (d["lo"] > 0) | (d["hi"] < 0)
        w(md_table(d[["metric", "base", "phrase", "term", "depth", "mean", "lo", "hi", "CI excludes 0"]], floatfmt="+.4f") + "\n")
        if len(cfg["phrases"]) > 1:
            w("Every phrase's late-layer values are in section 9.\n")
        w("Order contrast against its additive prediction, across the 144 heads (every phrase):\n")
        w(md_table(extra["order_fit"][["metric", "base", "phrase", "pair", "slope", "r", "r2", "mean_abs_observed",
                                       "mean_abs_interaction"]], floatfmt=".3f") + "\n")
        ah = extra["additivity_heads"]
        ah = ah[ah["metric"] == "sink_mass"]
        sig = ah[ah["q"] <= cfg["stats"]["fdr_q"]]
        rows = []
        for (base, phrase, combo), g in ah.groupby(["base", "phrase", "combo"], sort=False):
            s = sig[(sig["base"] == base) & (sig["phrase"] == phrase) & (sig["combo"] == combo)]
            late = s[s["layer"] >= 6]
            top = g.reindex(g["interaction"].abs().sort_values(ascending=False).index).head(3)
            rows.append((base, phrase, combo, len(s), int((late["interaction"] > 0).sum()), int((late["interaction"] < 0).sum()),
                         ", ".join(f"{r.name} ({r.interaction:+.3f})" for r in top.itertuples())))
        w("Heads with a BH-significant interaction on sink_mass (q over every interaction test of the metric):\n")
        w(md_table(pd.DataFrame(rows, columns=["base", "phrase", "combo", "significant", "late +", "late −",
                                               "largest |interaction|"])) + "\n")

    if "surprisal_depth" in extra:
        sp = cfg["surprisal"]
        w(f"## 8. Surprisal: does sink_mass track how predictable the slot tokens are?\n")
        w(f"Slot surprisal = −log p of the tokens at positions {sp['slot_positions']} given everything before them "
          "(TL), summed. Slope = within-row regression across every variant (both sides demeaned per row), per nat; "
          "CI = bootstrap over rows.\n")
        sd = extra["surprisal_depth"]
        w(md_table(sd[["base", "phrase", "depth", "slope", "lo", "hi"]], floatfmt="+.5f") + "\n")
        hs = extra["surprisal_heads"]
        hs = hs.assign(depth=np.where(hs["layer"] >= 6, "late", "early"),
                       neg=(hs["hi"] < 0), pos=(hs["lo"] > 0))
        w("Heads whose slope CI excludes 0, by sign (summed over phrases):\n")
        w(md_table(hs.groupby(["base", "depth"])[["neg", "pos"]].sum().reset_index().rename(
            columns={"neg": "slope < 0", "pos": "slope > 0"}), floatfmt=".0f") + "\n")
        sv = extra["surprisal_variants"]
        w("Per-variant means (descriptive):\n")
        w(md_table(sv[sv["phrase"].isin(per_phrase)].drop(columns=["placement"]), floatfmt=".3f") + "\n")

    if "phrase_pooled" in extra:
        w("## 9. Across phrases\n")
        w("Phrases as the unit: pooled_mean = mean of the per-phrase depth-group interactions, CI from bootstrapping "
          "phrases; n_pos / n_neg = phrases whose own row-bootstrap CI excludes 0 on that side; spearman = across "
          "phrases, the term against the surprisal gap NLL(shuffled) − NLL(nl).\n")
        w(md_table(extra["phrase_pooled"], floatfmt="+.4f") + "\n")
        gap = None
        if "surprisal_variants" in extra:
            piv = extra["surprisal_variants"].pivot_table(index=["base", "phrase"], columns="variant", values="mean_slot_nll")
            gap = piv["shuffled"] - piv["nl"]
        rows = []
        late = extra["additivity_depth"][extra["additivity_depth"]["depth"] == "late"]
        for (metric, base, phrase), g in late.groupby(["metric", "base", "phrase"], sort=False):
            r = {"metric": metric, "base": base, "phrase": cfg["phrases"][phrase]["text"].strip()}
            for term, gt in g.groupby("term", sort=False):
                x = gt.iloc[0]
                r[term] = f"{x['mean']:+.4f}" + ("*" if x["lo"] > 0 or x["hi"] < 0 else "")
            if gap is not None:
                r["nll gap (shuffled − nl)"] = f"{gap.loc[(base, phrase)]:+.2f}"
            rows.append(r)
        w("Late-layer interaction for every phrase (* = row-bootstrap CI excludes 0):\n")
        w(md_table(pd.DataFrame(rows)) + "\n")
        if "familiarity" in extra:
            w("Pair familiarity (−log p of the pair's second token given BOS + its first, TL; lower = more familiar) "
              "against each pair's interaction, across phrases:\n")
            w(md_table(extra["familiarity"], floatfmt="+.3f") + "\n")
            fam = extra["pair_familiarity"].assign(phrase=lambda d: d["phrase"].map(lambda n: cfg["phrases"][n]["text"].strip()))
            w(md_table(fam.pivot(index="phrase", columns="label", values="nll").reset_index(), floatfmt=".2f") + "\n")

    w("## Figures\n")
    for f in R["figures"]:
        w(f"- [{Path(f).stem}]({f})")
    w("")
    w("## Caveats\n")
    if len(cfg["phrases"]) == 1:
        w("- One phrase is one item: effects of `nl` are effects of these three tokens at these positions, not of "
          "natural language in general, until several phrases agree.")
    else:
        w(f"- {len(cfg['phrases'])} phrases from one template (pronoun verb object) are a small, hand-picked sample; "
          "CIs over phrases describe these phrases, not language in general.")
    w("- A slot at position p < 32 changes every later position's residual stream, so a contrast measures the slot "
      "tokens' total effect at the query (as keys and through earlier heads), not a single route.")
    w("- `matched` controls match surface form and id class, not frequency or meaning; `identity` therefore mixes "
      "everything else that distinguishes the phrase's tokens from same-form random words.")
    path.write_text("\n".join(L) + "\n")
