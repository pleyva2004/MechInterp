"""Phase 2: compare conditions and write comparison.parquet, split_half.parquet and phase2_report.md.

Analyses: head-type percentages with bootstrap CIs, per-head Mann-Whitney U on sink_mass for every
pair with Benjamini-Hochberg FDR across all pairs x heads, a split-half noise floor per condition,
and a monotonicity check of head activity across conditions.
"""
from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from exp1.aggregate import aggregate_heads
from exp1.compare import (
    bh_fdr, bootstrap_label_percentages, bootstrap_mean_ci, compare_conditions, label_shift, split_half,
)
from exp1.metrics import LABELS, classify, compute_metrics
from exp1.plots import heads_to_grid, plot_diff_heatmaps, plot_label_bars, plot_mode_label_panels, plot_shift_heatmaps
from exp1.validation import BACKEND_NAMES, md_table

# Config fields that must match between this run and a reference run whose attention we reuse.
SHARED_KEYS = (
    "model", "seq_len", "prepend_bos", "bos_token_id", "sample_without_replacement", "n_sequences", "seed",
    "query_position", "diffuse_threshold", "dtype", "backends",
)


def load_reference_attn(ref_dir: Path, condition: str, cfg: dict[str, Any], suffix: str) -> np.ndarray:
    """Attention for a condition computed in an earlier run, after checking that run used the same setup."""
    ref_cfg = yaml.safe_load((ref_dir / "config.yaml").read_text())
    mismatched = [key for key in SHARED_KEYS if ref_cfg.get(key) != cfg[key]]
    if ref_cfg["conditions"].get(condition) != cfg["conditions"][condition]:
        mismatched.append(f"conditions.{condition}")
    if condition not in ref_cfg["run_conditions"]:
        mismatched.append(f"run_conditions (does not include {condition})")
    if mismatched:
        raise ValueError(f"reference run {ref_dir} does not match the current config for {condition}: {mismatched}")
    return np.load(ref_dir / f"{condition}_attn{suffix}.npy")


def _pair_name(a: str, b: str) -> str:
    return f"{a} vs {b}"


# In-cell annotations of plot_label_shift, where full label names don't fit a heatmap cell.
_SHORT_LABELS = {"Diffuse": "Diff", "Sink": "Sink", "Previous": "Prev", "Self": "Self", "Other": "Other"}


def plot_label_shift(
    labels: dict[str, np.ndarray], pairs: list[tuple[str, str]], highlight_tv: float, split_half_seed: int,
    path: Path, backend_name: str,
) -> None:
    """labels: {condition: int codes [n_seq, n_layers, n_heads]}. One panel per pair: each head's TV distance
    between the two conditions' label distributions. Heads at or above highlight_tv are outlined and annotated
    with the label losing the most share -> the label gaining the most, going from the first condition to the
    second. The note reports the split-half noise floor so the cutoff can be read against it."""
    shift_of = partial(label_shift, n_labels=len(LABELS))
    short = np.vectorize(lambda lost, gained: f"{_SHORT_LABELS[LABELS[lost]]}→{_SHORT_LABELS[LABELS[gained]]}",
                         otypes=[str])
    tv, highlight, notes = {}, {}, {}
    for a, b in pairs:
        shift = shift_of(labels[a], labels[b])
        name = f"{a} → {b}"
        tv[name] = shift["tv"]
        highlight[name] = shift["tv"] >= highlight_tv
        notes[name] = short(shift["frac_diff"].argmax(axis=0), shift["frac_diff"].argmin(axis=0))

    conditions = list(dict.fromkeys(c for pair in pairs for c in pair))
    floor = max(float(split_half(labels[c], split_half_seed, shift_of)["tv"].max()) for c in conditions)
    plot_shift_heatmaps(
        tv, highlight, notes, path, f"Shift in each head's label mix between conditions [{backend_name}]",
        "TV distance between label distributions",
        note=f"outlined = TV ≥ {highlight_tv}, annotated as label losing the most share → label gaining the most "
             f"(first → second condition). Split-half noise floor: max TV = {floor:.2f} over all heads and conditions.",
    )


def _monotonic(values: list[float]) -> bool:
    return all(x < y for x, y in zip(values, values[1:]))


def run_phase2(
    cfg: dict[str, Any], run_dir: Path, attn: dict[str, dict[str, np.ndarray]], project_root: Path
) -> None:
    """attn: {backend: {condition: [n_seq, n_layers, n_heads, key_len]}} for the conditions run now;
    the rest are loaded from comparison.reference_runs."""
    comp = cfg["comparison"]
    pairs = [tuple(p) for p in comp["pairs"]]
    conditions = list(dict.fromkeys(c for pair in pairs for c in pair))
    key_len = cfg["seq_len"] + int(cfg["prepend_bos"])
    query_index = cfg["query_position"] % key_len
    primary = cfg["primary_backend"]
    alpha = comp["fdr_q"]

    for backend, by_condition in attn.items():
        for condition in conditions:
            if condition not in by_condition:
                ref_dir = project_root / comp["reference_runs"][condition]
                by_condition[condition] = load_reference_attn(ref_dir, condition, cfg, cfg["backends"][backend]["suffix"])

    metric = {b: {c: compute_metrics(a, query_index)[comp["metric"]] for c, a in by_c.items()} for b, by_c in attn.items()}

    # Per-head tests, one BH family per backend spanning every pair x head.
    comparison_rows, split_rows = [], []
    for backend in attn:
        tests = {pair: compare_conditions(metric[backend][pair[0]], metric[backend][pair[1]]) for pair in pairs}
        q = bh_fdr(np.stack([tests[pair]["p"] for pair in pairs]))
        splits = {c: split_half(metric[backend][c], comp["split_half_seed"]) for c in conditions}
        q_split = bh_fdr(np.stack([splits[c]["p"] for c in conditions]))
        n_layers, n_heads = q.shape[1:]
        layer, head = (g.ravel() for g in np.meshgrid(np.arange(n_layers), np.arange(n_heads), indexing="ij"))
        for i, pair in enumerate(pairs):
            t = tests[pair]
            comparison_rows.append(pd.DataFrame({
                "backend": backend, "layer": layer, "head": head, "comparison": _pair_name(*pair),
                "U": t["U"].ravel(), "p": t["p"].ravel(), "q": q[i].ravel(), "effect_size": t["effect_size"].ravel(),
                "mean_diff": t["mean_diff"].ravel(), "significant": q[i].ravel() <= alpha,
            }))
        for i, c in enumerate(conditions):
            s = splits[c]
            split_rows.append(pd.DataFrame({
                "backend": backend, "condition": c, "layer": layer, "head": head, "U": s["U"].ravel(),
                "p": s["p"].ravel(), "q": q_split[i].ravel(), "effect_size": s["effect_size"].ravel(),
                "significant": q_split[i].ravel() <= alpha,
            }))
    comparison = pd.concat(comparison_rows, ignore_index=True)
    split = pd.concat(split_rows, ignore_index=True)
    comparison.to_parquet(run_dir / "comparison.parquet", index=False)
    split.to_parquet(run_dir / "split_half.parquet", index=False)

    # Head-type percentages with bootstrap CIs, and the per-head tables for the side-by-side figure (primary backend).
    labels = {c: classify(attn[primary][c], cfg["diffuse_threshold"], query_index) for c in conditions}
    boot = {c: bootstrap_label_percentages(labels[c], len(LABELS), comp["bootstrap_iters"], comp["bootstrap_seed"],
                                           comp["bootstrap_ci"]) for c in conditions}
    heads = {c: aggregate_heads(attn[primary][c], cfg["diffuse_threshold"], query_index) for c in conditions}

    # Monotonicity of head activity: mean (1 - sink_mass) and mean entropy, overall and per layer.
    order = comp["monotonic_order"]
    all_metrics = {c: compute_metrics(attn[primary][c], query_index) for c in order}
    activity = {c: 1.0 - m["sink_mass"] for c, m in all_metrics.items()}
    entropy = {c: m["entropy"] for c, m in all_metrics.items()}

    fig_dir = run_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    pct_label = int(round(100 * comp["bootstrap_ci"]))
    plot_mode_label_panels(heads, LABELS, fig_dir / "phase2_mode_label_panels.png",
                           f"Mode label per head by condition [{BACKEND_NAMES[primary]}] (cell = consistency)")
    prim_cmp = comparison[comparison["backend"] == primary]
    diffs, sig = {}, {}
    for pair in pairs:
        rows = prim_cmp[prim_cmp["comparison"] == _pair_name(*pair)]
        name = f"{pair[0]} − {pair[1]}"
        diffs[name] = heads_to_grid(rows, "mean_diff")
        sig[name] = heads_to_grid(rows, "significant").astype(bool)
    plot_diff_heatmaps(diffs, sig, fig_dir / "phase2_sink_mass_diff.png",
                       f"Difference in mean sink_mass per head [{BACKEND_NAMES[primary]}]", "Δ mean sink_mass",
                       sig_note=f"outlined = BH-FDR significant (q ≤ {alpha}) across {len(pairs)} comparisons × heads")
    plot_label_shift(labels, pairs, comp["label_shift_highlight_tv"], comp["split_half_seed"],
                     fig_dir / "phase2_label_shift.png", BACKEND_NAMES[primary])
    plot_label_bars({c: (boot[c]["pct"], boot[c]["lo"], boot[c]["hi"]) for c in conditions}, LABELS,
                    fig_dir / "phase2_head_types.png",
                    f"Head types by condition [{BACKEND_NAMES[primary]}], {pct_label}% bootstrap CI "
                    f"({comp['bootstrap_iters']} resamples of sequences)")

    _write_report(run_dir / "phase2_report.md", cfg, run_dir.name, conditions, pairs, comparison, split, boot,
                  activity, entropy, heads)


def _write_report(
    path: Path, cfg: dict[str, Any], run_id: str, conditions: list[str], pairs: list[tuple[str, str]],
    comparison: pd.DataFrame, split: pd.DataFrame, boot: dict[str, dict[str, np.ndarray]],
    activity: dict[str, np.ndarray], entropy: dict[str, np.ndarray], heads: dict[str, pd.DataFrame],
) -> None:
    comp = cfg["comparison"]
    primary = cfg["primary_backend"]
    alpha = comp["fdr_q"]
    pct_label = int(round(100 * comp["bootstrap_ci"]))
    n_heads = len(next(iter(heads.values())))
    ref = ", ".join(f"`{c}` from `{d}`" for c, d in comp["reference_runs"].items()) or "none"
    out = [
        "# Experiment 1, Phase 2: condition comparison",
        "",
        f"- Run: `{run_id}`; conditions run here: {', '.join(f'`{c}`' for c in cfg['run_conditions'])}; "
        f"reused from earlier runs: {ref}.",
        f"- {cfg['n_sequences']} sequences per condition; statistics on the {BACKEND_NAMES[primary]} backend unless "
        "noted; the other backend is re-tested in section 2 as a cross-library check.",
        "- Id ranges: " + ", ".join(f"`{c}` {cfg['conditions'][c]['id_range']}" for c in conditions) + ".",
        "",
        f"## 1. Head types with {pct_label}% bootstrap CIs",
        "",
        f"% of the {n_heads} heads whose mode label is each type; CI from {comp['bootstrap_iters']} resamples of "
        "sequences (percentile method).",
        "",
    ]
    table = pd.DataFrame({"mode_label": list(LABELS)})
    for c in conditions:
        b = boot[c]
        table[c] = [f"{p:.1f} [{lo:.1f}, {hi:.1f}]" for p, lo, hi in zip(b["pct"], b["lo"], b["hi"])]
    out += [md_table(table), "",
            f"The statistic moves in steps of one head (100/{n_heads} = {100 / n_heads:.2f} points), so CIs are coarse and "
            "often asymmetric: only heads near a label boundary can flip under resampling. They capture sampling "
            "noise over sequences, not uncertainty about the label thresholds.", ""]

    prim = comparison[comparison["backend"] == primary]
    out += [
        f"## 2. Per-head sink_mass comparisons (Mann-Whitney U, BH-FDR q ≤ {alpha} over {len(prim)} tests)",
        "",
        "effect_size = rank-biserial r = P(first > second) − P(first < second); r > 0 means the first condition "
        "puts more mass on the sink. With 1000 sequences per condition almost any shift is significant, so read "
        "effect sizes (and Δ = difference in mean sink_mass) rather than the count of significant heads; a large |r| "
        "with a tiny Δ means both distributions sit near the same value but are consistently ordered.",
        "",
    ]
    rows = []
    for pair in pairs:
        r = prim[prim["comparison"] == _pair_name(*pair)]
        s = r[r["significant"]]
        top = r.loc[r["effect_size"].abs().idxmax()]
        rows.append({
            "comparison": _pair_name(*pair), "significant heads": f"{len(s)}/{len(r)}",
            "sig. with r > 0": int((s["effect_size"] > 0).sum()), "sig. with r < 0": int((s["effect_size"] < 0).sum()),
            "median |r|": float(r["effect_size"].abs().median()),
            "largest |r|": f"L{top['layer']}H{top['head']} (r = {top['effect_size']:.3f}, Δ = {top['mean_diff']:.3f})",
        })
    out += [md_table(pd.DataFrame(rows)), ""]

    for pair in pairs:
        r = prim[(prim["comparison"] == _pair_name(*pair)) & prim["significant"]]
        r = r.reindex(r["effect_size"].abs().sort_values(ascending=False).index).head(10)
        if len(r):
            out += [f"Top significant heads by |r|, {_pair_name(*pair)}:", "",
                    md_table(r[["layer", "head", "mean_diff", "effect_size", "p", "q"]].assign(
                        p=r["p"].map(lambda v: f"{v:.1e}"), q=r["q"].map(lambda v: f"{v:.1e}"))), ""]

    others = [b for b in comparison["backend"].unique() if b != primary]
    for other in others:
        o = comparison[comparison["backend"] == other]
        same = (prim["significant"].values == o["significant"].values)
        out += [f"Cross-library check ({BACKEND_NAMES[other]} vs {BACKEND_NAMES[primary]}): significance decisions agree "
                f"on {int(same.sum())}/{len(same)} tests; max |Δ effect_size| = "
                f"{np.abs(prim['effect_size'].values - o['effect_size'].values).max():.2e}.", ""]

    sp = split[split["backend"] == primary]
    out += [
        "## 3. Split-half noise floor",
        "",
        f"Each condition's sequences are split into two random halves ({cfg['n_sequences'] // 2} vs "
        f"{cfg['n_sequences'] // 2}) and compared with the same test; BH-FDR over all {len(sp)} split-half tests. "
        "With no real difference, any 'significant' head here is noise. The halves have half the sample size of the "
        "cross-condition tests, so this floor is slightly conservative in power.",
        "",
    ]
    rows = []
    for c in conditions:
        s = sp[sp["condition"] == c]
        rows.append({"condition": c, f"significant (q ≤ {alpha})": f"{int(s['significant'].sum())}/{len(s)}",
                     f"uncorrected p < {alpha}": int((s["p"] < alpha).sum()),
                     "median |r|": float(s["effect_size"].abs().median()), "max |r|": float(s["effect_size"].abs().max())})
    out += [md_table(pd.DataFrame(rows)), "",
            f"For reference, {alpha:.0%} of {n_heads} heads = {alpha * n_heads:.1f} uncorrected false positives expected "
            "per condition under the null.", ""]

    order = comp["monotonic_order"]
    rows = []
    for name, values in (("mean (1 − sink_mass)", activity), ("mean entropy (nats)", entropy)):
        cis = {c: bootstrap_mean_ci(values[c], comp["bootstrap_iters"], comp["bootstrap_seed"], comp["bootstrap_ci"])
               for c in order}
        row = {"measure": name, **{c: f"{m:.4f} [{lo:.4f}, {hi:.4f}]" for c, (m, lo, hi) in cis.items()}}
        row["increasing " + " < ".join(order)] = "yes" if _monotonic([cis[c][0] for c in order]) else "no"
        ranked = sorted(order, key=lambda c: cis[c][0])
        separated = all(cis[a][2] < cis[b][1] for a, b in zip(ranked, ranked[1:]))
        row["observed order"] = " < ".join(ranked) + (" (CIs disjoint)" if separated else " (some CIs overlap)")
        rows.append(row)
    out += [f"## 4. Monotonicity: does head activity increase {' < '.join(order)}?", "",
            f"Overall (mean over all heads and sequences; {pct_label}% bootstrap CI over sequences):", "",
            md_table(pd.DataFrame(rows)), ""]

    per_layer = pd.DataFrame({"layer": np.arange(activity[order[0]].shape[1])})
    for name, values in (("1−sink", activity), ("entropy", entropy)):
        layer_means = {c: values[c].mean(axis=(0, 2)) for c in order}
        for c in order:
            per_layer[f"{name} {c}"] = layer_means[c]
        per_layer[f"{name} increasing"] = [
            "yes" if _monotonic([layer_means[c][layer] for c in order]) else "no" for layer in per_layer["layer"]
        ]
    n_act = int((per_layer["1−sink increasing"] == "yes").sum())
    n_ent = int((per_layer["entropy increasing"] == "yes").sum())
    out += ["Per layer (mean over the layer's heads and all sequences):", "", md_table(per_layer, ".3f"), "",
            f"Layers where the order holds: {n_act}/{len(per_layer)} for 1 − sink_mass, "
            f"{n_ent}/{len(per_layer)} for entropy.", ""]

    out += ["## 5. Figures", "",
            "- [Mode label, conditions side by side](figures/phase2_mode_label_panels.png)",
            "- [Δ mean sink_mass per head, FDR-significant heads outlined](figures/phase2_sink_mass_diff.png)",
            "- [Shift in each head's label mix (TV distance), largest shifts outlined](figures/phase2_label_shift.png)",
            f"- [Head-type % by condition with {pct_label}% CIs](figures/phase2_head_types.png)",
            "- Per-condition figures and validation checks for the conditions run here: `validation_report.md`", ""]
    path.write_text("\n".join(out))
