"""Experiment 3, E3.3: distance sweep.

Run:  uv run --offline python -m exp3.run_e33 --config configs/exp3_e33.yaml

Writes results/<timestamp>_exp3e33/. Contents:
- the config, the pre-registered predictions and the library versions;
- design and attention arrays, per-row late values, plus per-head, pooled, sweep, half-distance and switch tables;
- checks.json, predictions.json, length_control.json and timing.json;
- figures and exp3_e33_report.md.
TransformerLens computes the attention. Hugging Face recomputes a subset of rows on identical ids as a cross-library
check.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # write figures to files, never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from exp1.backends import load_model, set_determinism
from exp1.runlog import make_run_dir, save_config_copy, save_versions
from exp1ext.features import token_features
from exp2.analysis import head_name
from exp2.design import cached_base_block, derive_seed, matched_pools
from exp3 import e33
from exp3 import engine as eng
from exp3 import plots_engine as plots

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def phrase_ids(cfg: dict[str, Any], tokenizer) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    """{name: [P, V, O]} and {name: [V, P, O]}; the text must tokenize to the configured ids."""
    ids, shuffled = {}, {}
    for name, ph in cfg["phrases"].items():
        enc = tokenizer.encode(ph["text"])
        if enc != ph["ids"]:
            raise ValueError(f"phrase {name}: {ph['text']!r} tokenizes to {enc}, config says {ph['ids']}")
        ids[name], shuffled[name] = list(ph["ids"]), list(ph["shuffled"])
    return ids, shuffled


def build(cfg: dict[str, Any], tokenizer, t0: float) -> tuple[dict[str, eng.Design], dict[str, Any]]:
    """Base rows per class (cached, hash-tagged, all phrase ids excluded), the E3.3 design, and every design check."""
    phrases, shuffled = phrase_ids(cfg, tokenizer)
    all_ids = sorted({i for ids in phrases.values() for i in ids})
    seq_dir = resolve(cfg["paths"]["sequences_dir"])
    lo = min(r[0] for r in cfg["token_classes"].values())
    hi = max(r[1] for r in cfg["token_classes"].values())
    features = token_features(np.arange(lo, hi + 1), tokenizer)
    pools = {n: matched_pools(ids, features, cfg["match_features"], cfg["token_classes"]) for n, ids in phrases.items()}
    designs = {}
    info = {"phrase_ids": all_ids, "design": {}, "matched_pool_sizes": {n: [len(p) for p in pl] for n, pl in pools.items()}}
    for base, id_range in cfg["bases"].items():
        rows = cached_base_block(f"exp3e33_{base}", id_range, all_ids, cfg["n_bases"], cfg["seq_len"], cfg["seed"],
                                 cfg["bos_token_id"], seq_dir)
        seed = derive_seed(cfg["seed"], f"exp3e33/{base}")
        d = e33.build_e33(rows, phrases, shuffled, pools, seed, cfg["layout"])
        info["design"][base] = eng.check_design(d, cfg["bos_token_id"])
        e33.check_e33(d, phrases, shuffled, pools, cfg["layout"])
        designs[base] = d
        log(f"{base}: design built and checked ({len(d.variants)} variants x {len(d.units)} units x {d.rows} rows)", t0)
    return designs, info


def save_tables(run: Path, designs: dict[str, eng.Design], ext: dict[str, eng.Extraction], results: dict[str, dict],
                cfg: dict[str, Any]) -> None:
    np.savez_compressed(run / "design.npz", **{f"{b}/base": d.base[:, 0] for b, d in designs.items()},
                        **{f"{b}/{v}": s for b, d in designs.items() for v, s in d.seqs.items()})
    arrays = ("att_row", "att_late_unit", "att_unit") if cfg.get("outputs", {}).get("save_att_row", True) else \
        ("att_late_unit", "att_unit")
    np.savez_compressed(run / "attention.npz",
                        **{f"{b}/{k}": getattr(e, k) for b, e in ext.items() for k in arrays},
                        **{f"{b}/key_len": e.key_len for b, e in ext.items()},
                        variants=np.array(next(iter(ext.values())).variants), units=np.array(next(iter(ext.values())).units))
    np.savez_compressed(run / "labels.npz", **{f"{b}/{k}": v for b, r in results.items() for k, v in r["labels"].items()})
    np.savez_compressed(run / "late_rows.npz",
                        **{f"{b}/{c}/{m}": x for b, r in results.items() for (c, m), x in r["late"].items()})
    head_rows, pooled, switch_rows, sweep_rows, hd_rows = [], [], [], [], []
    for b, r in results.items():
        for (name, m), h in r["heads"].items():
            for layer in range(h["mean"].shape[0]):
                for head in range(h["mean"].shape[1]):
                    head_rows.append({"base": b, "contrast": name, "measure": m, "layer": layer, "head": head,
                                      **{k: float(h[k][layer, head]) for k in ("mean", "lo", "hi", "p", "q")},
                                      "meaningful": bool(h["meaningful"][layer, head]),
                                      "tv": float(r["tv"][name][layer, head])})
        units = {(u["contrast"], u["measure"]): u for u in r["units"]}
        pooled += [{"base": b, **p, **{k: v for k, v in units[(p["contrast"], p["measure"])].items()
                                       if k not in ("contrast", "measure")}} for p in r["pooled"]]
        for (name, kind), sw in r["switches"].items():
            switch_rows += [{"base": b, "contrast": name, "labels": kind, **s} for s in sw]
        sweep_rows += [{"base": b, **row} for row in r["sweep"]]
        gaps = cfg["layout"]["gaps"]
        for eff, hd in r["half_distance"].items():
            for layer in range(hd["half_distance"].shape[0]):
                for head in range(hd["half_distance"].shape[1]):
                    hd_rows.append({"base": b, "effect": eff, "layer": layer, "head": head,
                                    "eligible": bool(hd["eligible"][layer, head]),
                                    "half_distance": float(hd["half_distance"][layer, head]),
                                    "long_range": bool(hd["long_range"][layer, head]),
                                    **{f"E_g{g}": float(hd["effect"][i, layer, head]) for i, g in enumerate(gaps)}})
    pd.DataFrame(head_rows).to_parquet(run / "heads.parquet", index=False)
    pd.DataFrame(pooled).to_csv(run / "pooled.csv", index=False)
    pd.DataFrame(switch_rows, columns=["base", "contrast", "labels", "layer", "head", "to", "from", "shift", "q"]).to_csv(
        run / "switches.csv", index=False)
    pd.DataFrame(sweep_rows).to_csv(run / "sweep.csv", index=False)
    pd.DataFrame(hd_rows).to_csv(run / "half_distance.csv", index=False)
    (run / "length_control.json").write_text(json.dumps({b: r["length_control"] for b, r in results.items()}, indent=2))


# ---- figures -------------------------------------------------------------------------------------------------

def plot_sweep_pooled(results: dict[str, dict], cfg: dict[str, Any], path: Path) -> None:
    """Late pooled sink effect over g, per base class: order (nl − shuffled) and total (nl − none) with 95% CIs, the
    ±m band, and the length-control E (24-token rows, g = 0) as a separate marker."""
    margin = cfg["analysis"]["margin"]
    fig, axes = plt.subplots(1, len(results), figsize=(6.2 * len(results), 4.4), squeeze=False)
    styles = {"order": ("tab:red", "o", "E(g) = nl − shuffled"), "total": ("tab:blue", "s", "total = nl − none")}
    for ax, (b, r) in zip(axes[0], results.items()):
        ax.axhspan(-margin, margin, color="0.9", label=f"±m ({margin})")
        ax.axhline(0, color="0.4", lw=0.8)
        for eff, (color, marker, label) in styles.items():
            rows = [x for x in r["sweep"] if x["effect"] == eff]
            g = np.array([x["g"] for x in rows])
            mean = np.array([x["mean"] for x in rows])
            err = np.array([[x["mean"] - x["lo"] for x in rows], [x["hi"] - x["mean"] for x in rows]])
            ax.errorbar(g, mean, yerr=err, color=color, marker=marker, ms=4, capsize=2, lw=1.2, label=label)
        lc = r["length_control"]["E_length_control"]
        ax.errorbar([-0.6], [lc["mean"]], yerr=[[lc["mean"] - lc["lo"]], [lc["hi"] - lc["mean"]]], color="black",
                    marker="D", ms=5, capsize=2, ls="none", label=f"E, {cfg['layout']['length_control']['seq_len']}-token rows (g = 0)")
        ax.set_xticks(cfg["layout"]["gaps"])
        ax.set_xlabel("gap g (phrase ends at 31 − g; query at 32)")
        ax.set_ylabel("late pooled Δ BOS mass (layers 6–11)")
        ax.set_title(f"{b} bases ({len(r['late'][(cfg['length_control']['contrast'], 'BOS')])} rows)", fontsize=10)
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle("E3.3: sink effect vs distance (95% CIs over rows)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_sweep_heads(r: dict[str, Any], eff: str, gaps: list[int], title: str, path: Path) -> None:
    """One panel per layer, one line per head: the row-mean Δ BOS mass over g. Heads eligible for a half-distance
    (meaningful at g = 0) are coloured and labelled with it; the others are grey."""
    hd = r["half_distance"][eff]
    effect, dist, elig = hd["effect"], hd["half_distance"], hd["eligible"]
    n_layers, n_heads = effect.shape[1:]
    vmax = float(np.abs(effect).max()) or 1.0
    fig, axes = plt.subplots(3, 4, figsize=(15, 9), sharex=True, sharey=True)
    cmap = plt.get_cmap("tab20")
    for layer, ax in enumerate(axes.ravel()[:n_layers]):
        ax.axhline(0, color="0.5", lw=0.6)
        for head in range(n_heads):
            if elig[layer, head]:
                d = dist[layer, head]
                tag = "> " + str(gaps[-1]) if np.isinf(d) else f"{d:g}"
                ax.plot(gaps, effect[:, layer, head], marker="o", ms=2.5, lw=1.2, color=cmap(head % 20),
                        label=f"H{head} (half {tag})")
            else:
                ax.plot(gaps, effect[:, layer, head], lw=0.6, color="0.75")
        ax.set_title(f"L{layer}", fontsize=9)
        ax.set_ylim(-1.05 * vmax, 1.05 * vmax)
        if elig[layer].any():
            ax.legend(fontsize=6, frameon=False, loc="upper right")
    for ax in axes[-1]:
        ax.set_xlabel("gap g", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("Δ BOS mass", fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def figures(run: Path, results: dict[str, dict], cfg: dict[str, Any], contrasts: dict[str, dict]) -> list[str]:
    out = []
    fig_dir = run / "figures"
    gaps = list(cfg["layout"]["gaps"])
    plot_sweep_pooled(results, cfg, fig_dir / "sweep_pooled.png")
    out.append("sweep_pooled.png")
    for b, r in results.items():
        for eff, prefix in cfg["sweep"].items():
            name = f"sweep_heads_{b}_{eff}.png"
            sup = " (supplementary)" if eff != "order" else ""
            plot_sweep_heads(r, eff, gaps, f"E3.3, {b} bases: per-head Δ BOS mass over g, {prefix}<g>{sup}; coloured = "
                             "meaningful at g = 0, legend = half-distance", fig_dir / name)
            out.append(name)
    panels, kinds, cbar, outline = {}, {}, {}, {}
    lc = cfg["length_control"]["contrast"]
    for b, r in results.items():
        for eff, prefix in cfg["sweep"].items():
            row = f"{b}: {eff} Δ BOS"
            kinds[row], cbar[row], panels[row] = "delta", "Δ BOS (a − b)", {}
            names = [f"{prefix}{g}" for g in gaps] + ([lc] if eff == "order" else [])
            for n in names:
                col = f"{n}\n{contrasts[n]['a']} − {contrasts[n]['b']}"
                panels[row][col] = r["heads"][(n, "BOS")]["mean"]
                outline[(row, col)] = r["heads"][(n, "BOS")]["meaningful"]
    plots.plot_head_maps(panels, kinds, cbar, "Per-head Δ BOS per contrast (boxed = meaningful: BH q ≤ 0.05, |Δ| ≥ 0.02)",
                         fig_dir / "contrast_maps.png", outline=outline)
    out.append("contrast_maps.png")
    mats = {}
    for b, r in results.items():
        for name, kind in (("order_g0", "argmax"), ("order_g0", "mass"), (lc, "argmax"), ("total_g0", "argmax"),
                           ("placebo", "argmax")):
            if (name, kind) in r["transitions"]:
                mats[f"{b}, {name} ({kind})"] = r["transitions"][(name, kind)]
    plots.plot_transitions(mats, eng.LABELS, "Role switches: source → destination (McNemar, BH, ≥ 15-point shift)",
                           fig_dir / "transitions.png")
    out.append("transitions.png")
    return out


# ---- report --------------------------------------------------------------------------------------------------

def fmt_ci(d: dict, lo: str = "lo", hi: str = "hi") -> str:
    return f"[{d[lo]:+.4f}, {d[hi]:+.4f}]"


def write_report(run: Path, cfg: dict, designs: dict, results: dict, checks: dict, preds: list[dict], timing: dict,
                 contrasts: dict, figs: list[str]) -> None:
    first = next(iter(designs.values()))
    a = cfg["analysis"]
    lay = e33.parse_layout(cfg["layout"])
    lo, hi = a["late_layers"]
    L = ["# Experiment 3, E3.3: distance sweep", "",
         f"Run `{run.name}`. TL attention at the final query (32; 24 in the length control); HF cross-check on "
         f"{cfg['checks']['hf_rows']} rows per base class. {cfg['n_bases']} rows per base class; {len(first.variants)} "
         f"variants × {len(first.units)} phrases per row. Phrase P V O at 29 − g … 31 − g, g ∈ {list(lay.gaps)}; length "
         f"control: the row's first {lay.lc_len} tokens, phrase at {lay.lc_phrase[0]}–{lay.lc_phrase[-1]}. Late = layers "
         f"{lo}–{hi}, the mean over their heads. Attention is averaged over a row's phrases before masses and labels; "
         "masks come from each contrast's treated arm and its own query position (placebo: the masks of "
         f"{contrasts['placebo'].get('mask_from', contrasts['placebo']['a'])}).", "",
         f"Extraction: {timing['sequences']} sequences in {timing['seconds']:.0f} s ({timing['seq_per_s']:.0f} seq/s, TL only).",
         "", "## Checks", "", "| check | result |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in checks.items()]
    L += ["", "## Pre-registered predictions (exp3/PREDICTIONS_e33.md)", "", "| id | verdict | detail |", "|---|---|---|"]
    L += [f"| {p['id']} | {p['verdict']} | {p['detail']} |" for p in preds]

    L += ["", "## Sweep: late pooled Δ BOS mass per gap", "",
          "E(g) = nl − shuffled; total = nl − none. 95% CI over rows; verdict as below; late heads = meaningful + / − "
          "of 72.", "",
          "| base | g | phrase at | E(g) | 95% CI | verdict | late heads + / − | total | 95% CI | verdict | late heads + / − |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for b, r in results.items():
        by = {(x["effect"], x["g"]): x for x in r["sweep"]}
        for g in lay.gaps:
            o, t = by[("order", g)], by[("total", g)]
            L.append(f"| {b} | {g} | {o['positions']} | {o['mean']:+.4f} | {fmt_ci(o)} | {o['verdict']} | "
                     f"{o['late_meaningful_pos']} / {o['late_meaningful_neg']} | {t['mean']:+.4f} | {fmt_ci(t)} | "
                     f"{t['verdict']} | {t['late_meaningful_pos']} / {t['late_meaningful_neg']} |")

    L += ["", "## Length control (no prediction)", "",
          f"E = late pooled nl − shuffled BOS mass, each computed within its own length. Differences are paired per row.",
          "", "| base | E, 24-token rows (g = 0) | E(0), 32 tokens | E(8), 32 tokens | E24 − E(0) | E24 − E(8) |",
          "|---|---|---|---|---|---|"]
    for b, r in results.items():
        c = r["length_control"]
        cells = [c[k] for k in ("E_length_control", "E_same_distance", "E_same_position",
                                "length_control_minus_same_distance", "length_control_minus_same_position")]
        L.append(f"| {b} | " + " | ".join(f"{x['mean']:+.4f} {fmt_ci(x)}" for x in cells) + " |")

    hd_cfg = cfg["half_distance"]
    L += ["", f"## Per-head half-distance (no prediction)", "",
          f"Smallest g > 0 with E_h(g) / E_h(0) ≤ {hd_cfg['ratio']} (sign-aware), for heads meaningful at g = 0; "
          f"long-range candidates have half-distance ≥ {hd_cfg['long_range_min']} (\"> {lay.gaps[-1]}\" included). "
          "The total effect is supplementary.", ""]
    for b, r in results.items():
        for eff, hd in r["half_distance"].items():
            dist = hd["half_distance"][hd["eligible"]]
            vals, counts = np.unique(np.where(np.isinf(dist), lay.gaps[-1] + 1, dist), return_counts=True)
            hist = ", ".join(f"{'> ' + str(lay.gaps[-1]) if v > lay.gaps[-1] else f'{v:g}'}: {n}"
                             for v, n in zip(vals, counts)) or "none"
            longs = []
            for name in hd["long_range_heads"]:
                l_, h_ = eng.parse_heads([name])[0]
                d = hd["half_distance"][l_, h_]
                longs.append(f"{name} (E(0) {hd['effect'][0, l_, h_]:+.3f}, half {'> ' + str(lay.gaps[-1]) if np.isinf(d) else f'{d:g}'})")
            L.append(f"- {b}, {eff}: {int(hd['eligible'].sum())} eligible heads; half-distance counts {hist}. "
                     f"Long-range candidates ({len(longs)}): {', '.join(longs) or 'none'}")

    L += ["", "## Late pooled per-row Δ (a − b), every contrast and measure", "",
          "95% / 90% CIs bootstrap rows; verdict: positive / negative (95% CI excludes 0), equivalent (TOST, 90% CI inside "
          f"±{a['margin']}), else inconclusive. Per phrase: bootstrap over the per-unit means, with a sign count.", "",
          "| base | contrast | measure | Δ | 95% CI | 90% CI | Wilcoxon p | verdict | per unit [95% CI] | units > 0 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for b, r in results.items():
        units = {(u["contrast"], u["measure"]): u for u in r["units"]}
        for p in r["pooled"]:
            u = units[(p["contrast"], p["measure"])]
            L.append(f"| {b} | {p['contrast']} | {p['measure']} | {p['mean']:+.4f} | {fmt_ci(p)} | {fmt_ci(p, 'lo90', 'hi90')} | "
                     f"{p['wilcoxon_p']:.2g} | {p['verdict']} | {u['unit_mean']:+.4f} {fmt_ci(u, 'unit_lo', 'unit_hi')} | "
                     f"{u['unit_n_pos']}/{u['unit_units']} |")
    measures = list(a.get("measures", eng.MEASURES))
    L += ["", f"## Meaningful heads per contrast (BH q ≤ {a['meaningful']['q']} over 144 heads, |Δ| ≥ "
          f"{a['meaningful']['min_abs']}; + / −)", "",
          "| base | contrast | " + " | ".join(measures) + " |", "|---|---|" + "---|" * len(measures)]
    for b, r in results.items():
        for name in contrasts:
            cells = [f"{int((h['meaningful'] & (h['mean'] > 0)).sum())} / {int((h['meaningful'] & (h['mean'] < 0)).sum())}"
                     for h in (r["heads"][(name, m)] for m in measures)]
            L.append(f"| {b} | {name} | " + " | ".join(cells) + " |")
    L += ["", f"## Role switches (McNemar, BH, ≥ {a['switch']['min_shift']:.0%} shift)", ""]
    for b, r in results.items():
        L.append(f"- {b}, placebo switching heads (either direction): argmax {len(r['placebo_switching']['argmax'])} "
                 f"{[head_name(*x) for x in r['placebo_switching']['argmax']]}, mass {len(r['placebo_switching']['mass'])}")
        for (name, kind), sw in r["switches"].items():
            if not sw:
                continue
            desc = ", ".join(f"{head_name(s['layer'], s['head'])} {s['from']}→{s['to']} ({s['shift']:+.2f})" for s in sw)
            L.append(f"- {b}, {name}, {kind} labels: {len(sw)} heads; {desc}")
        quiet = [f"{n}/{k}" for (n, k), sw in r["switches"].items() if not sw]
        L.append(f"- {b}, no switching heads: {', '.join(quiet) or '—'}")
    L += ["", "## Transition matrices (argmax labels; nonzero source → destination counts)", ""]
    for b, r in results.items():
        for name in contrasts:
            t = r["transitions"][(name, "argmax")]
            if t.any():
                cells = ", ".join(f"{eng.LABELS[i]}→{eng.LABELS[j]} {t[i, j]}" for i, j in zip(*np.nonzero(t)))
                L.append(f"- {b}, {name}: {cells}")
    L += ["", "## TV distance between category profiles (per head; late mean, max)", "",
          "| base | contrast | late mean | max (head) |", "|---|---|---|---|"]
    for b, r in results.items():
        for name in r["tv"]:
            tv = r["tv"][name]
            i = np.unravel_index(tv.argmax(), tv.shape)
            L.append(f"| {b} | {name} | {tv[lo:hi + 1].mean():.4f} | {tv[i]:.4f} ({head_name(int(i[0]), int(i[1]))}) |")
    L += ["", "## Layer profile (mean Δ BOS over rows and heads, per layer)", "",
          "| base | contrast | " + " | ".join(f"L{i}" for i in range(12)) + " |", "|---|---|" + "---|" * 12]
    for b, r in results.items():
        for name in contrasts:
            prof = r["layer_profile"][(name, "BOS")]
            L.append(f"| {b} | {name} | " + " | ".join(f"{x:+.3f}" for x in prof) + " |")
    L += ["", "## Figures", ""]
    L += [f"- `figures/{f}`" for f in figs]
    L.append("")
    (run / "exp3_e33_report.md").write_text("\n".join(L))


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Experiment 3, E3.3 (distance sweep)")
    parser.add_argument("--config", default="configs/exp3_e33.yaml", type=Path)
    args = parser.parse_args(argv)
    cfg_path = resolve(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())
    contrasts, specs = cfg["contrasts"], cfg["predictions"]
    t0 = time.time()
    run = make_run_dir(resolve(cfg["paths"]["results_dir"]), tag="exp3e33")
    save_config_copy(cfg_path, run)
    shutil.copy2(PROJECT_ROOT / "exp3" / "PREDICTIONS_e33.md", run / "PREDICTIONS.md")
    save_versions(run, PROJECT_ROOT, code_dirs=("exp1", "exp2", "exp3"))
    (run / "figures").mkdir()
    log(f"run folder {run}", t0)

    set_determinism(cfg["seed"])
    bos = cfg["bos_token_id"]
    tl = load_model("transformer_lens", cfg["backends"]["transformer_lens"]["model_name"], cfg["device"], cfg["dtype"])
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg["figures"]["tokenizer"])  # TL's tokenizer prepends BOS
    designs, info = build(cfg, tokenizer, t0)

    chunk, atol, late = cfg["extraction"]["chunk_rows"], cfg["checks"]["row_sum_atol"], cfg["analysis"]["late_layers"]
    ext = {}
    for b, d in designs.items():
        ext[b] = eng.extract(tl, d, bos, late, chunk, atol, log=lambda m, b=b: log(f"{b}: {m}", t0))
        log(f"{b}: extraction done ({ext[b].n_sequences} sequences, {ext[b].seconds:.0f} s, "
            f"{ext[b].n_sequences / ext[b].seconds:.0f} seq/s)", t0)
    n_seq = sum(e.n_sequences for e in ext.values())
    secs = sum(e.seconds for e in ext.values())
    timing = {"sequences": n_seq, "seconds": secs, "seq_per_s": n_seq / secs,
              "per_base": {b: {"sequences": e.n_sequences, "seconds": e.seconds} for b, e in ext.items()}}

    t_det = time.time()
    det = {b: eng.determinism_check(tl, d, ext[b], bos, chunk, atol, cfg["checks"]["determinism_rows"])
           for b, d in designs.items()}
    timing["determinism_seconds"] = time.time() - t_det
    hf = load_model("huggingface", cfg["backends"]["huggingface"]["model_name"], cfg["device"], cfg["dtype"],
                    cfg["backends"]["huggingface"]["attn_implementation"])
    t_hf = time.time()
    cross = {b: eng.hf_check(tl, hf, d, bos, cfg["checks"]["hf_rows"], chunk * len(d.units), atol)
             for b, d in designs.items()}
    timing["hf_check_seconds"] = time.time() - t_hf
    log("determinism and cross-library checks done", t0)

    t_an = time.time()
    results = {b: e33.analyse(ext[b], designs[b], cfg, contrasts, specs) for b in designs}
    timing["analysis_seconds"] = time.time() - t_an
    log(f"analysis done ({timing['analysis_seconds']:.0f} s)", t0)
    preds = e33.check_predictions(specs, results)

    lay = e33.parse_layout(cfg["layout"])
    max_plc = cfg["analysis"]["max_placebo_switches"]
    checks = {
        "design (base outside written positions; filler holds no written id; no BOS; no repeated id)":
            "PASS (asserted per variant and base class)",
        (f"E3.3 layout (nl_g vs shuffled_g only at 29−g, 30−g, O at 31−g in both, shuffled = V P O; every variant vs none "
         f"only at its phrase positions; matched / matched_b from their pools; {lay.lc_len}-token rows = the row's first "
         f"{lay.lc_len} tokens; query = the row's own token)"): "PASS (asserted per base class)",
        "attention rows sum to 1": f"PASS (atol {atol:g}, asserted per batch; max |sum − 1| "
                                   f"{max(e.max_row_sum_dev for e in ext.values()):.1e})",
    }
    for b in designs:
        checks[f"determinism, {b} ({det[b]['rows']} rows)"] = (
            f"{'PASS' if det[b]['exact'] else 'FAIL'} (max |Δ| {det[b]['max_abs_delta']:.1e})")
        checks[f"TL vs HF, {b} ({cross[b]['rows']} rows, {cross[b]['sequences']} sequences)"] = (
            f"{'PASS' if cross[b]['max_abs_delta'] < cfg['checks']['hf_atol'] else 'FAIL'} "
            f"(max |Δ| {cross[b]['max_abs_delta']:.1e}, argmax-key agreement {cross[b]['argmax_agreement']:.4%})")
        n_plc = len(results[b]["placebo_switching"]["argmax"])
        checks[f"placebo switching heads ≤ {max_plc}, {b}"] = (
            f"{'PASS' if n_plc <= max_plc else 'FAIL'} ({n_plc} argmax, "
            f"{len(results[b]['placebo_switching']['mass'])} mass)")

    save_tables(run, designs, ext, results, cfg)
    (run / "checks.json").write_text(json.dumps(checks, indent=2))
    pred_out = [{**p, "stats": {b: r["predictions"][p["id"]]["stats"] for b, r in results.items()}} for p in preds]
    (run / "predictions.json").write_text(json.dumps(pred_out, indent=2, default=float))
    (run / "timing.json").write_text(json.dumps(timing, indent=2))
    (run / "design_summary.json").write_text(json.dumps(info, indent=2))
    figs = figures(run, results, cfg, contrasts)
    write_report(run, cfg, designs, results, checks, preds, timing, contrasts, figs)
    log(f"done: {run / 'exp3_e33_report.md'}", t0)
    for p in preds:
        print(f"{p['id']}: {p['verdict']}  {p['detail']}")
    return run


if __name__ == "__main__":
    main()
