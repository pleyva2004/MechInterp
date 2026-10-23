"""Experiment 3, E3.2: span length, from one word to natural text.

Run:  uv run --offline python -m exp3.run_e32 --config configs/exp3_e32.yaml

Writes results/<timestamp>_exp3e32/. Contents:
- the config, the pre-registered predictions and the library versions;
- design and attention arrays, plus per-head, pooled, switch and sweep tables;
- the saturation fit, per-passage and per-type E(k), switch lengths, and span repeats;
- checks.json, predictions.json and timing.json;
- figures and exp3_e32_report.md.
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
from exp2.design import cached_base_block, derive_seed, id_pool, matched_pools, token_class_of
from exp3 import e32
from exp3 import engine as eng
from exp3 import plots_engine as plots

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAP_VARIANTS = ("none", "nl_1", "nl_3", "nl_8", "nl_31", "shuffled_31", "matched_31", "passage-rr", "passage+t",
                "passage+w")
MAP_CONTRASTS = ("E_1", "E_3", "E_8", "E_31", "order_31", "content_31", "rr", "query_ctx", "placebo_31")


def log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def passages_from_config(cfg: dict[str, Any], tokenizer, features: pd.DataFrame) -> dict[str, Any]:
    """Check every passage against the tokenizer and the passage rules, then return its span (last K ids before t),
    t, exclusion list (every id of the text), word pools (one per distinct word id of the span), t's pool, type and
    a summary. Punctuation = byte ids and tokens whose stripped text is not alphabetic: no pool, never redrawn."""
    ks, _, _, _ = e32.check_layout(cfg["layout"])
    K = ks[-1]
    out: dict[str, dict] = {"spans": {}, "t": {}, "exclude": {}, "word_pools": {}, "t_pools": {}, "types": {},
                            "summary": {}}
    for name, p in cfg["passages"].items():
        ids = [int(i) for i in p["ids"]]
        t = int(p["t"]["id"])
        if tokenizer.encode(p["text"]) != ids:
            raise ValueError(f"passage {name}: the text does not tokenize to the config's ids")
        if tokenizer.encode(p["t"]["text"]) != [t]:
            raise ValueError(f"passage {name}: t {p['t']['text']!r} does not tokenize to [{t}]")
        if tokenizer.decode(ids[-2:]) != p["final_pair"]:
            raise ValueError(f"passage {name}: the last two ids decode to {tokenizer.decode(ids[-2:])!r}, "
                             f"not {p['final_pair']!r}")
        if len(ids) < max(int(cfg["passage_min_tokens"]), K):
            raise ValueError(f"passage {name}: {len(ids)} tokens before t, need >= {cfg['passage_min_tokens']}")
        if t in ids:
            raise ValueError(f"passage {name}: t occurs in the passage")
        span = ids[-K:]
        # classified from each id's own decoded text, so a word id outside every class raises below
        alpha = token_features(np.asarray(sorted(set(span) | {t})), tokenizer).set_index("id")["is_alpha"]
        words = [i for i in dict.fromkeys(span) if i >= 256 and bool(alpha[i])]
        punct = [i for i in dict.fromkeys(span) if i not in words]
        if t < 256 or not bool(alpha[t]):
            raise ValueError(f"passage {name}: t ({t}) is not a word token")
        for i in words + [t]:
            token_class_of(i, cfg["token_classes"])  # raises outside 256-39999
        excl = sorted(set(ids) | {t})
        pools = [pl[~np.isin(pl, excl)] for pl in
                 matched_pools(words + [t], features, cfg["match_features"], cfg["token_classes"])]
        if any(len(pl) <= cfg["seq_len"] for pl in pools):  # else a row could hold the whole pool
            raise ValueError(f"passage {name}: a matched pool has <= {cfg['seq_len']} ids after exclusion")
        out["spans"][name], out["t"][name], out["exclude"][name] = span, t, excl
        out["word_pools"][name] = dict(zip(words, pools[:-1]))
        out["t_pools"][name] = pools[-1]
        out["types"][name] = p["type"]
        out["summary"][name] = {
            "genre": p["genre"], "type": p["type"], "final_pair": p["final_pair"], "t": p["t"]["text"], "t_id": t,
            "tokens_before_t": len(ids), "span_words": len(words),
            "span_punctuation": {str(i): tokenizer.decode([i]) for i in punct},
            "word_pool_sizes": [int(len(pl)) for pl in pools[:-1]], "t_pool_size": int(len(pools[-1]))}
    return out


def repeats_table(spans: dict[str, list[int]], orders: dict[str, dict[int, list[int]]], cfg: dict[str, Any]) -> list[dict]:
    ks, sks, _, _ = e32.check_layout(cfg["layout"])
    K = ks[-1]
    out = []
    for name, span in spans.items():
        for k in ks:
            ids = np.asarray(span[K - k:])
            seqs = [("nl", ids)] + ([("shuffled", ids[orders[name][k]])] if k in sks else [])
            for variant, s in seqs:
                r = e32.span_repeats(s)
                out.append({"passage": name, "k": k, "variant": variant, **{x: r[x] for x in
                            ("repeated_ids", "extra_occurrences", "min_distance")}, "ids": " ".join(map(str, r["ids"]))})
    return out


def build(cfg: dict[str, Any], tokenizer, t0: float) -> tuple[dict[str, eng.Design], dict[str, Any], dict[str, Any]]:
    """Shared base rows per class (cached, hash-tagged, no exclusion), per-passage redraws, the E3.2 design, and every
    design check."""
    _, sks, _, _ = e32.check_layout(cfg["layout"])
    lo = min(r[0] for r in cfg["token_classes"].values())
    hi = max(r[1] for r in cfg["token_classes"].values())
    features = token_features(np.arange(lo, hi + 1), tokenizer)
    P = passages_from_config(cfg, tokenizer, features)
    orders = e32.shuffle_orders(P["spans"], sks, derive_seed(cfg["seed"], "exp3e32/shuffle"))
    seq_dir = resolve(cfg["paths"]["sequences_dir"])
    designs = {}
    info: dict[str, Any] = {"passages": P["summary"], "design": {}, "redrawn": {}, "base_pool_size": {},
                            "shuffle_orders": {n: {str(k): o for k, o in od.items()} for n, od in orders.items()},
                            "repeats": repeats_table(P["spans"], orders, cfg)}
    for base, id_range in cfg["bases"].items():
        rows = cached_base_block(f"exp3e32_{base}", id_range, [], cfg["n_bases"], cfg["seq_len"], cfg["seed"],
                                 cfg["bos_token_id"], seq_dir)
        base_pool = id_pool(id_range, [])
        seed = derive_seed(cfg["seed"], f"exp3e32/{base}")
        args = (P["spans"], P["t"], P["exclude"], P["word_pools"], P["t_pools"], orders, base_pool)
        d = e32.build_e32(rows, *args, seed, cfg["layout"])
        info["design"][base] = eng.check_design(d, cfg["bos_token_id"])
        info["redrawn"][base] = e32.check_e32(d, rows, *args, cfg["layout"])["redrawn"]
        info["base_pool_size"][base] = int(len(base_pool))
        designs[base] = d
        log(f"{base}: design built and checked ({len(d.variants)} variants x {len(d.units)} passages x {d.rows} rows; "
            f"{sum(info['redrawn'][base].values())} filler ids redrawn)", t0)
    return designs, info, P


# ---- tables --------------------------------------------------------------------------------------------------

def save_arrays(run: Path, designs: dict[str, eng.Design], ext: dict[str, eng.Extraction], cfg: dict[str, Any]) -> None:
    """design.npz and attention.npz, written right after extraction so a later failure cannot lose them."""
    np.savez_compressed(run / "design.npz", **{f"{b}/base": d.base for b, d in designs.items()},
                        **{f"{b}/{v}": s for b, d in designs.items() for v, s in d.seqs.items()},
                        units=np.array(next(iter(designs.values())).units))
    arrays = ("att_row", "att_late_unit", "att_unit") if cfg.get("outputs", {}).get("save_att_row", True) else \
        ("att_late_unit", "att_unit")
    np.savez_compressed(run / "attention.npz",
                        **{f"{b}/{k}": getattr(e, k) for b, e in ext.items() for k in arrays},
                        **{f"{b}/key_len": e.key_len for b, e in ext.items()},
                        variants=np.array(next(iter(ext.values())).variants), units=np.array(next(iter(ext.values())).units))


def save_tables(run: Path, designs: dict[str, eng.Design], results: dict[str, dict], cfg: dict[str, Any],
                info: dict[str, Any], types: dict[str, str]) -> None:
    ks, _, _, _ = e32.check_layout(cfg["layout"])
    np.savez_compressed(run / "labels.npz", **{f"{b}/{k}": v for b, r in results.items() for k, v in r["labels"].items()})
    head_rows, pooled, switch_rows, sweep, unit_rows, type_rows, len_rows = [], [], [], [], [], [], []
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
        sweep += [{"base": b, **s} for s in r["sweep"]]
        for u, name in enumerate(designs[b].units):
            unit_rows += [{"base": b, "passage": name, "type": types[name], "k": k, "E": float(r["unit_e"][u, j])}
                          for j, k in enumerate(ks)]
        for ty, v in r["types"].items():
            type_rows += [{"base": b, "type": ty, "passages": " ".join(v["passages"]), "k": k, "E": e}
                          for k, e in zip(ks, v["E"])]
        for kind, sl in r["switch_length"].items():
            for layer in range(sl.shape[0]):
                for head in range(sl.shape[1]):
                    len_rows.append({"base": b, "labels": kind, "layer": layer, "head": head,
                                     "switch_length": float(sl[layer, head]),
                                     "switch_at_max_k": r["switch_dest"][kind].get(head_name(layer, head), "")})
    pd.DataFrame(head_rows).to_parquet(run / "heads.parquet", index=False)
    pd.DataFrame(pooled).to_csv(run / "pooled.csv", index=False)
    pd.DataFrame(switch_rows, columns=["base", "contrast", "labels", "layer", "head", "to", "from", "shift", "q"]).to_csv(
        run / "switches.csv", index=False)
    pd.DataFrame(sweep).to_csv(run / "sweep.csv", index=False)
    pd.DataFrame(unit_rows).to_csv(run / "passage_sweep.csv", index=False)
    pd.DataFrame(type_rows).to_csv(run / "type_sweep.csv", index=False)
    pd.DataFrame(len_rows).to_csv(run / "switch_length.csv", index=False)
    pd.DataFrame(info["repeats"]).to_csv(run / "repeats.csv", index=False)
    (run / "saturation.json").write_text(json.dumps({b: r["saturation"] for b, r in results.items()}, indent=2))


# ---- figures -------------------------------------------------------------------------------------------------

def plot_sweep(results: dict[str, dict], ks: list[int], path: Path) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(6.2 * len(results), 4.4), squeeze=False)
    styles = {"E": ("nl − none: E(k)", "tab:red"), "order": ("nl − shuffled", "tab:blue"),
              "content": ("nl − matched", "tab:green")}
    for ax, (b, r) in zip(axes[0], results.items()):
        for series, (label, color) in styles.items():
            rows = [s for s in r["sweep"] if s["series"] == series]
            x = np.array([s["k"] for s in rows], dtype=float) * {"E": 1.0, "order": 1.04, "content": 0.96}[series]
            y = np.array([s["mean"] for s in rows])
            err = np.array([[s["mean"] - s["lo"] for s in rows], [s["hi"] - s["mean"] for s in rows]])
            ax.errorbar(x, y, yerr=err, fmt="o-", color=color, ms=4, lw=1.2, capsize=2, label=label)
        sat = r["saturation"]
        if np.isfinite(sat["kappa"]):
            xx = np.geomspace(min(ks), max(ks), 200)
            ax.plot(xx, e32.saturation(xx, sat["e_inf"], sat["kappa"]), color="tab:red", ls="--", lw=1,
                    label=f"fit: E∞ {sat['e_inf']:+.4f}, κ {sat['kappa']:.2f} [{sat['kappa_ci'][0]:.2f}, "
                          f"{sat['kappa_ci'][1]:.2f}]")
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_xscale("log")
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks], fontsize=7)
        ax.set_xlabel("k (passage tokens before the query)")
        ax.set_ylabel("late pooled Δ BOS mass (95% CI over rows)")
        ax.set_title(f"{b} bases", fontsize=10)
        ax.legend(fontsize=7)
    fig.suptitle("E3.2: late-layer BOS effect against span length", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_head_curves(r: dict[str, Any], ks: list[int], late: list[int], title: str, path: Path) -> None:
    layers = list(range(int(late[0]), int(late[1]) + 1))
    fig, axes = plt.subplots(2, -(-len(layers) // 2), figsize=(4.0 * -(-len(layers) // 2), 6.4), squeeze=False,
                             sharey=True)
    curves = r["head_curves"]  # [n_k, L, H]
    cmap = plt.get_cmap("tab20")
    for ax, layer in zip(axes.ravel(), layers):
        for head in range(curves.shape[2]):
            ax.plot(ks, curves[:, layer, head], marker=".", ms=3, lw=0.9, color=cmap(head), label=f"H{head}")
        ax.axhline(0, color="gray", lw=0.7)
        ax.set_xscale("log")
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks], fontsize=6)
        ax.set_title(f"layer {layer}", fontsize=9)
    axes[0, 0].set_ylabel("Δ BOS (nl − none)")
    axes[1, 0].set_ylabel("Δ BOS (nl − none)")
    axes[0, -1].legend(fontsize=5, ncol=2)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_switch_length(results: dict[str, dict], ks: list[int], path: Path) -> None:
    kinds = e32.KINDS
    fig, axes = plt.subplots(len(results), len(kinds), figsize=(4.2 * len(kinds) + 1, 3.8 * len(results)), squeeze=False)
    cmap = plt.get_cmap("viridis", len(ks)).copy()
    cmap.set_bad("#e6e6e6")
    for i, (b, r) in enumerate(results.items()):
        for j, kind in enumerate(kinds):
            sl = r["switch_length"][kind]
            idx = np.full(sl.shape, np.nan)
            ok = np.isfinite(sl)
            idx[ok] = [ks.index(int(v)) for v in sl[ok]]
            ax = axes[i, j]
            im = ax.imshow(np.ma.masked_invalid(idx), origin="lower", cmap=cmap, vmin=-0.5, vmax=len(ks) - 0.5)
            ax.set_title(f"{b}, {kind} labels: {int(ok.sum())} heads", fontsize=9)
            ax.set_xlabel("Head", fontsize=7)
            ax.set_ylabel("Layer", fontsize=7)
            ax.set_xticks(range(0, 12, 2))
            ax.set_yticks(range(0, 12, 2))
            cb = fig.colorbar(im, ax=ax, ticks=range(len(ks)))
            cb.ax.set_yticklabels([str(k) for k in ks], fontsize=6)
            cb.set_label("switch length k", fontsize=7)
    fig.suptitle("Switch length: smallest k from which the head's role is switched (nl_k vs none) at every larger k; "
                 "gray = not switched at 31", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_types(results: dict[str, dict], ks: list[int], path: Path) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(6.2 * len(results), 4.4), squeeze=False, sharey=True)
    cmap = plt.get_cmap("tab10")
    for ax, (b, r) in zip(axes[0], results.items()):
        for i, (ty, v) in enumerate(r["types"].items()):
            ax.plot(ks, v["E"], marker="o", ms=3, lw=1, color=cmap(i % 10), label=f"{ty} ({', '.join(v['passages'])})")
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_xscale("log")
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks], fontsize=7)
        ax.set_xlabel("k")
        ax.set_ylabel("late pooled E(k), mean over rows")
        ax.set_title(f"{b} bases: E(k) per final-pair type (descriptive)", fontsize=10)
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def figures(run: Path, designs: dict[str, eng.Design], ext: dict[str, eng.Extraction], results: dict[str, dict],
            cfg: dict[str, Any], contrasts: dict[str, dict]) -> list[str]:
    ks, _, pks, _ = e32.check_layout(cfg["layout"])
    fig_dir = run / "figures"
    out = []
    ref = eng.make_contrast(next(iter(designs.values())), "ref", f"nl_{ks[-1]}", f"nl_{ks[-1]}")
    for b, e in ext.items():
        bos, role = {}, {}
        for v in MAP_VARIANTS:
            if v in e.variants:
                vi = e.index(v)
                bos[v] = e.att_row[:, vi, ..., 0].mean(axis=0)
                role[v] = eng.head_roles(eng.labels(e.att_row[:, vi], ref.codes, cfg["diffuse_threshold"])[0])
        rows = {"Mean BOS mass": bos, "What each head attends to most": role}
        name = f"attention_maps_{b}.png"
        plots.plot_head_maps(rows, {"Mean BOS mass": "mass", "What each head attends to most": "label"},
                             {k: "attention" for k in rows},
                             f"E3.2, {b} bases: per-head attention at the query (32), mean over {e.att_row.shape[0]} rows "
                             f"and 12 passages; roles use nl_{ks[-1]}'s masks", fig_dir / name, label_names=eng.LABELS)
        out.append(name)
    panels, kinds, cbar, outline = {}, {}, {}, {}
    for b, r in results.items():
        for m, label in (("BOS", "Δ BOS"), ("Span", "Δ span"), ("Phrase", "Δ phrase-position")):
            row = f"{b}: {label}"
            kinds[row], cbar[row], panels[row] = "delta", f"{label} (a − b)", {}
            for n in MAP_CONTRASTS:
                if (n, m) in r["heads"]:
                    col = f"{n}\n{contrasts[n]['a']} − {contrasts[n]['b']}"
                    panels[row][col] = r["heads"][(n, m)]["mean"]
                    outline[(row, col)] = r["heads"][(n, m)]["meaningful"]
    plots.plot_head_maps(panels, kinds, cbar, "Per-head Δ per contrast (boxed = meaningful: BH q ≤ 0.05, |Δ| ≥ 0.02)",
                         fig_dir / "contrast_maps.png", outline=outline)
    out.append("contrast_maps.png")
    mats = {}
    for b, r in results.items():
        for name, kind in ((f"E_{ks[-1]}", "argmax"), ("query_ctx", "argmax"), ("query_ctx", "mass"),
                           (f"placebo_{pks[-1]}", "argmax")):
            mats[f"{b}, {name} ({kind})"] = r["transitions"][(name, kind)]
    plots.plot_transitions(mats, eng.LABELS, "Role switches: source → destination (McNemar, BH, ≥ 15-point shift)",
                           fig_dir / "transitions.png")
    out.append("transitions.png")
    plot_sweep(results, ks, fig_dir / "sweep.png")
    out.append("sweep.png")
    for b, r in results.items():
        name = f"head_curves_{b}.png"
        plot_head_curves(r, ks, cfg["analysis"]["late_layers"],
                         f"E3.2, {b} bases: per-head Δ BOS (nl_k − none), late layers, mean over rows", fig_dir / name)
        out.append(name)
    plot_switch_length(results, ks, fig_dir / "switch_length.png")
    out.append("switch_length.png")
    plot_types(results, ks, fig_dir / "type_sweep.png")
    out.append("type_sweep.png")
    return out


# ---- report --------------------------------------------------------------------------------------------------

def fmt_ci(d: dict, lo: str = "lo", hi: str = "hi") -> str:
    return f"[{d[lo]:+.4f}, {d[hi]:+.4f}]"


def write_report(run: Path, cfg: dict, designs: dict, results: dict, checks: dict, preds: list[dict], timing: dict,
                 contrasts: dict, figs: list[str], info: dict) -> None:
    first = next(iter(designs.values()))
    a = cfg["analysis"]
    ks, _, _, _ = e32.check_layout(cfg["layout"])
    lo, hi = a["late_layers"]
    L = ["# Experiment 3, E3.2: span length, from one word to natural text", "",
         f"Run `{run.name}`. TL attention at the query (position 32); HF cross-check on {cfg['checks']['hf_rows']} rows "
         f"per base class. {cfg['n_bases']} rows per base class; {len(first.variants)} variants × {len(first.units)} "
         f"passages per row. k ∈ {ks}: the passage's last k tokens at 32 − k … 31. Late = layers {lo}–{hi}, the mean "
         "over their heads. Attention is averaged over a row's passages before masses and labels; masks come from each "
         "contrast's treated arm (placebos: the masks of matched at that k).", "",
         f"Extraction: {timing['sequences']} sequences in {timing['seconds']:.0f} s ({timing['seq_per_s']:.0f} seq/s, "
         "TL only).", "", "## Checks", "", "| check | result |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in checks.items()]
    all_pass = all(str(v).startswith("PASS") for v in checks.values())
    L += ["", f"**All checks pass: {'yes' if all_pass else 'NO'}.** The predictions count only if every check passes "
          "(exp3/PREDICTIONS_e32.md)."]
    L += ["", "## Pre-registered predictions (exp3/PREDICTIONS_e32.md)", "", "| id | verdict | detail |", "|---|---|---|"]
    L += [f"| {p['id']} | {p['verdict']} | {p['detail']} |" for p in preds]
    L += ["", "## Sweep: late pooled Δ BOS per k (95% CI over rows; per passage: mean [95% CI], passages > 0)", "",
          "| base | series | " + " | ".join(f"k={k}" for k in ks) + " |", "|---|---|" + "---|" * len(ks)]
    for b, r in results.items():
        for series, label in (("E", "nl − none (E)"), ("order", "nl − shuffled"), ("content", "nl − matched")):
            by_k = {s["k"]: s for s in r["sweep"] if s["series"] == series}
            cells = [f"{by_k[k]['mean']:+.4f} {fmt_ci(by_k[k])}" if k in by_k else "—" for k in ks]
            L.append(f"| {b} | {label} | " + " | ".join(cells) + " |")
            cells = [f"{by_k[k]['unit_mean']:+.4f} {fmt_ci(by_k[k], 'unit_lo', 'unit_hi')}, {by_k[k]['unit_n_pos']}/"
                     f"{by_k[k]['unit_units']}" if k in by_k else "—" for k in ks]
            L.append(f"| {b} | {label}, per passage | " + " | ".join(cells) + " |")
    L += ["", "## Saturation fit E(k) = E∞ · (1 − e^(−k/κ)) (no prediction)", "",
          "| base | E∞ [95% CI] | κ [95% CI] | RMSE | failed fits | resamples with κ on a bound | point κ on a bound |",
          "|---|---|---|---|---|---|---|"]
    for b, r in results.items():
        s = r["saturation"]
        L.append(f"| {b} | {s['e_inf']:+.4f} [{s['e_inf_ci'][0]:+.4f}, {s['e_inf_ci'][1]:+.4f}] | {s['kappa']:.3f} "
                 f"[{s['kappa_ci'][0]:.3f}, {s['kappa_ci'][1]:.3f}] | {s['rmse']:.5f} | {s['n_failed']}/{s['n_boot']} | "
                 f"{s['share_kappa_at_bound']:.1%} | {s['point_kappa_at_bound']} |")
    L += ["", "## E(k) per final-pair type (descriptive; mean over rows, then over the type's passages)", "",
          "| base | type | passages | " + " | ".join(f"k={k}" for k in ks) + " |", "|---|---|---|" + "---|" * len(ks)]
    for b, r in results.items():
        for ty, v in r["types"].items():
            L.append(f"| {b} | {ty} | {', '.join(v['passages'])} | " + " | ".join(f"{e:+.4f}" for e in v["E"]) + " |")
    L += ["", "## Switch length (smallest k switched in nl_k vs none at that k and every larger k)", "",
          "| base | labels | heads | count per switch length | late heads (switch length; role change at 31) |",
          "|---|---|---|---|---|"]
    for b, r in results.items():
        for kind, sl in r["switch_length"].items():
            ok = np.isfinite(sl)
            counts = ", ".join(f"{k}: {int((sl == k).sum())}" for k in ks if (sl == k).any()) or "none"
            late = [(int(layer), int(head)) for layer, head in zip(*np.nonzero(ok)) if lo <= layer <= hi]
            desc = ", ".join(f"{head_name(*lh)} ({int(sl[lh])}; {r['switch_dest'][kind].get(head_name(*lh), '')})"
                             for lh in late) or "none"
            L.append(f"| {b} | {kind} | {int(ok.sum())} | {counts} | {desc} |")
    L += ["", "## Passages", "", "| passage | genre | type | final pair | t | tokens before t | span punctuation | "
          "filler ids redrawn (" + ", ".join(info["redrawn"]) + ") |", "|---|---|---|---|---|---|---|---|"]
    for name, s in info["passages"].items():
        red = ", ".join(str(info["redrawn"][b][name]) for b in info["redrawn"])
        punct = " ".join(repr(x) for x in s["span_punctuation"].values())
        L.append(f"| {name} | {s['genre']} | {s['type']} | {s['final_pair']!r} | {s['t']!r} | {s['tokens_before_t']} | "
                 f"{punct} | {red} |")
    L += ["", "## Repeats inside the written span (nl; repeated ids / extra occurrences / shortest distance)", "",
          "| passage | " + " | ".join(f"k={k}" for k in ks) + " |", "|---|" + "---|" * len(ks)]
    rep = {(x["passage"], x["k"]): x for x in info["repeats"] if x["variant"] == "nl"}
    for name in info["passages"]:
        cells = []
        for k in ks:
            x = rep[(name, k)]
            cells.append(f"{x['repeated_ids']} / {x['extra_occurrences']} / {x['min_distance'] if x['min_distance'] else '—'}")
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    show = ("BOS", "Previous", "Span", "Filler", "Phrase")
    L += ["", f"## Late pooled per-row Δ (a − b), every contrast ({', '.join(show)}; all measures in pooled.csv)", "",
          "95% / 90% CIs bootstrap rows; verdict: positive / negative (95% CI excludes 0), equivalent (TOST, 90% CI inside "
          f"±{a['margin']}), else inconclusive. Per passage: bootstrap over the per-passage means, with a sign count.", "",
          "| base | contrast | measure | Δ | 95% CI | 90% CI | Wilcoxon p | verdict | per passage [95% CI] | passages > 0 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for b, r in results.items():
        units = {(u["contrast"], u["measure"]): u for u in r["units"]}
        for p in r["pooled"]:
            if p["measure"] not in show:
                continue
            u = units[(p["contrast"], p["measure"])]
            L.append(f"| {b} | {p['contrast']} | {p['measure']} | {p['mean']:+.4f} | {fmt_ci(p)} | "
                     f"{fmt_ci(p, 'lo90', 'hi90')} | {p['wilcoxon_p']:.2g} | {p['verdict']} | {u['unit_mean']:+.4f} "
                     f"{fmt_ci(u, 'unit_lo', 'unit_hi')} | {u['unit_n_pos']}/{u['unit_units']} |")
    L += ["", f"## Meaningful heads per contrast (BH q ≤ {a['meaningful']['q']} over 144 heads, |Δ| ≥ "
          f"{a['meaningful']['min_abs']}; + / −)", "",
          "| base | contrast | " + " | ".join(e32.MEASURES) + " |", "|---|---|" + "---|" * len(e32.MEASURES)]
    for b, r in results.items():
        for name in contrasts:
            cells = []
            for m in e32.MEASURES:
                h = r["heads"][(name, m)]
                cells.append(f"{int((h['meaningful'] & (h['mean'] > 0)).sum())} / {int((h['meaningful'] & (h['mean'] < 0)).sum())}")
            L.append(f"| {b} | {name} | " + " | ".join(cells) + " |")
    L += ["", f"## Role switches (McNemar, BH, ≥ {a['switch']['min_shift']:.0%} shift)", ""]
    for b, r in results.items():
        for n, plc in r["placebo_switching"].items():
            L.append(f"- {b}, {n} switching heads (either direction): argmax {len(plc['argmax'])} "
                     f"{[head_name(*x) for x in plc['argmax']]}, mass {len(plc['mass'])}")
        for (name, kind), sw in r["switches"].items():
            desc = ", ".join(f"{head_name(s['layer'], s['head'])} {s['from']}→{s['to']} ({s['shift']:+.2f})" for s in sw) or "none"
            L.append(f"- {b}, {name}, {kind} labels: {len(sw)} heads; {desc}")
    L += ["", "## Transition matrices (argmax labels; nonzero source → destination counts)", ""]
    for b, r in results.items():
        for name in contrasts:
            t = r["transitions"][(name, "argmax")]
            cells = ", ".join(f"{eng.LABELS[i]}→{eng.LABELS[j]} {t[i, j]}" for i, j in zip(*np.nonzero(t))) or "none"
            L.append(f"- {b}, {name}: {cells}")
    L += ["", "## TV distance between category profiles (per head; late mean, max)", "",
          "| base | contrast | late mean | max (head) |", "|---|---|---|---|"]
    for b, r in results.items():
        for name in r["tv"]:
            tv = r["tv"][name]
            i = np.unravel_index(tv.argmax(), tv.shape)
            L.append(f"| {b} | {name} | {tv[lo:hi + 1].mean():.4f} | {tv[i]:.4f} ({head_name(int(i[0]), int(i[1]))}) |")
    L += ["", "## Layer profile (mean Δ over rows and heads, per layer)", "",
          "| base | contrast | measure | " + " | ".join(f"L{i}" for i in range(12)) + " |", "|---|---|---|" + "---|" * 12]
    for b, r in results.items():
        for name in contrasts:
            for m in ("BOS", "Span"):
                prof = r["layer_profile"][(name, m)]
                L.append(f"| {b} | {name} | {m} | " + " | ".join(f"{x:+.3f}" for x in prof) + " |")
    L += ["", "## Figures", ""]
    L += [f"- `figures/{f}`" for f in figs]
    L.append("")
    (run / "exp3_e32_report.md").write_text("\n".join(L))


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Experiment 3, E3.2 (span length)")
    parser.add_argument("--config", default="configs/exp3_e32.yaml", type=Path)
    args = parser.parse_args(argv)
    cfg_path = resolve(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())
    contrasts = e32.contrast_specs(cfg)
    specs = cfg["predictions"]
    e32.validate_specs(specs, e32.variant_names(cfg["layout"]), contrasts)  # fail before any model work
    types = {name: p["type"] for name, p in cfg["passages"].items()}
    t0 = time.time()
    run = make_run_dir(resolve(cfg["paths"]["results_dir"]), tag="exp3e32")
    save_config_copy(cfg_path, run)
    shutil.copy2(PROJECT_ROOT / "exp3" / "PREDICTIONS_e32.md", run / "PREDICTIONS.md")
    save_versions(run, PROJECT_ROOT, code_dirs=("exp1", "exp2", "exp3"))
    (run / "figures").mkdir()
    log(f"run folder {run}", t0)

    set_determinism(cfg["seed"])
    bos = cfg["bos_token_id"]
    tl = load_model("transformer_lens", cfg["backends"]["transformer_lens"]["model_name"], cfg["device"], cfg["dtype"])
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg["figures"]["tokenizer"])  # TL's tokenizer prepends BOS
    designs, info, _ = build(cfg, tokenizer, t0)

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
    (run / "design_summary.json").write_text(json.dumps(info, indent=2, default=str))
    save_arrays(run, designs, ext, cfg)
    log("design and attention arrays saved", t0)

    t_chk = time.time()
    det = {b: eng.determinism_check(tl, d, ext[b], bos, chunk, atol, cfg["checks"]["determinism_rows"])
           for b, d in designs.items()}
    hf = load_model("huggingface", cfg["backends"]["huggingface"]["model_name"], cfg["device"], cfg["dtype"],
                    cfg["backends"]["huggingface"]["attn_implementation"])
    cross = {b: eng.hf_check(tl, hf, d, bos, cfg["checks"]["hf_rows"], chunk * len(d.units), atol)
             for b, d in designs.items()}
    timing["check_seconds"] = time.time() - t_chk
    log("determinism and cross-library checks done", t0)

    t_an = time.time()
    results = {}
    for b in designs:
        results[b] = e32.analyse(ext[b], designs[b], cfg, contrasts, specs, types)
        log(f"{b}: analysis done", t0)
    timing["analysis_seconds"] = time.time() - t_an
    preds = e32.check_predictions(specs, results)

    max_plc = cfg["analysis"]["max_placebo_switches"]
    checks = {
        "design (base outside written positions; filler holds no written id; no BOS; repeats only between positions "
        "holding one passage id, declared per passage, in every row)": "PASS (asserted per variant and base class)",
        "E3.2 layout (per-passage redraws only at collisions; nl / shuffled / matched / matched_b differ only on the "
        "span; shuffled keeps 31; matched keeps punctuation, one draw per word, nested; passage-rr only at 30–31; 2 × 2 "
        "cells differ only at 1–31 or 32; w from t's pool)": "PASS (asserted per base class)",
        "attention rows sum to 1": f"PASS (atol {atol:g}, asserted per batch; max |sum − 1| "
                                   f"{max(e.max_row_sum_dev for e in ext.values()):.1e})",
    }
    for b in designs:
        checks[f"determinism, {b} ({det[b]['rows']} rows)"] = (
            f"{'PASS' if det[b]['exact'] else 'FAIL'} (max |Δ| {det[b]['max_abs_delta']:.1e})")
        checks[f"TL vs HF, {b} ({cross[b]['rows']} rows, {cross[b]['sequences']} sequences)"] = (
            f"{'PASS' if cross[b]['max_abs_delta'] < cfg['checks']['hf_atol'] else 'FAIL'} "
            f"(max |Δ| {cross[b]['max_abs_delta']:.1e}, argmax-key agreement {cross[b]['argmax_agreement']:.4%})")
        for n, plc in results[b]["placebo_switching"].items():
            checks[f"{n} switching heads ≤ {max_plc}, {b}"] = (
                f"{'PASS' if len(plc['argmax']) <= max_plc else 'FAIL'} ({len(plc['argmax'])} argmax, "
                f"{len(plc['mass'])} mass)")

    save_tables(run, designs, results, cfg, info, types)
    (run / "checks.json").write_text(json.dumps(checks, indent=2))
    pred_out = [{**p, "stats": {b: r["predictions"][p["id"]]["stats"] for b, r in results.items()}} for p in preds]
    (run / "predictions.json").write_text(json.dumps(pred_out, indent=2, default=float))
    (run / "timing.json").write_text(json.dumps(timing, indent=2))
    figs = figures(run, designs, ext, results, cfg, contrasts)
    write_report(run, cfg, designs, results, checks, preds, timing, contrasts, figs, info)
    log(f"done: {run / 'exp3_e32_report.md'}", t0)
    for p in preds:
        print(f"{p['id']}: {p['verdict']}  {p['detail']}")
    return run


if __name__ == "__main__":
    main()
