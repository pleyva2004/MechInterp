"""Experiment 3, E3.4: repeated phrase, induction as the positive control (or its pre-registered fallback).

Run:  uv run --offline python -m exp3.run_e34 --config configs/exp3_e34.yaml

Writes results/<timestamp>_exp3e34/ (fallback: _exp3e34fb/). Contents:
- the config, the pre-registered predictions and the library versions;
- design and attention arrays, plus per-head, pooled and switch tables;
- checks.json, predictions.json and timing.json;
- figures and exp3_e34_report.md.
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

import numpy as np
import pandas as pd
import yaml

from exp1.backends import load_model, set_determinism
from exp1.runlog import make_run_dir, save_config_copy, save_versions
from exp1ext.features import token_features
from exp2.analysis import head_name
from exp2.design import cached_base_block, derive_seed, id_pool, matched_pools
from exp3 import e34
from exp3 import engine as eng
from exp3 import plots_engine as plots

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def phrase_ids(cfg: dict[str, Any], tokenizer) -> dict[str, list[int]]:
    out = {}
    for name, ph in cfg["phrases"].items():
        ids = tokenizer.encode(ph["text"])
        if ids != ph["ids"]:
            raise ValueError(f"phrase {name}: {ph['text']!r} tokenizes to {ids}, config says {ph['ids']}")
        out[name] = list(ph["ids"])
    return out


def build(cfg: dict[str, Any], tokenizer, fallback: bool, t0: float) -> tuple[dict[str, eng.Design], dict[str, Any]]:
    """Base rows per class (cached, hash-tagged, all phrase ids excluded), the E3.4 (or fallback) design, and every
    design check."""
    phrases = phrase_ids(cfg, tokenizer)
    all_ids = sorted({i for ids in phrases.values() for i in ids})
    seq_dir = resolve(cfg["paths"]["sequences_dir"])
    pools = None
    if not fallback:
        lo = min(r[0] for r in cfg["token_classes"].values())
        hi = max(r[1] for r in cfg["token_classes"].values())
        features = token_features(np.arange(lo, hi + 1), tokenizer)
        pools = {n: matched_pools(ids, features, cfg["match_features"], cfg["token_classes"]) for n, ids in phrases.items()}
    designs, info = {}, {"phrase_ids": all_ids, "design": {}, "base_pool_size": {}}
    if pools is not None:
        info["matched_pool_sizes"] = {n: [len(p) for p in pl] for n, pl in pools.items()}
    for base, id_range in cfg["bases"].items():
        rows = cached_base_block(f"exp3e34_{base}", id_range, all_ids, cfg["n_bases"], cfg["seq_len"], cfg["seed"],
                                 cfg["bos_token_id"], seq_dir)
        base_pool = id_pool(id_range, all_ids)
        seed = derive_seed(cfg["seed"], f"exp3e34/{base}")
        if fallback:
            d = e34.build_fallback(rows, base_pool, seed, cfg["fallback_layout"])
            info["design"][base] = eng.check_design(d, cfg["bos_token_id"])
            e34.check_fallback(d, base_pool, cfg["fallback_layout"])
        else:
            d = e34.build_e34(rows, phrases, cfg["other_nl"], pools, base_pool, seed, cfg["layout"])
            info["design"][base] = eng.check_design(d, cfg["bos_token_id"])
            e34.check_e34(d, phrases, cfg["other_nl"], pools, base_pool, cfg["layout"])
        info["base_pool_size"][base] = int(len(base_pool))
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
    head_rows, pooled, switch_rows, ind_rows = [], [], [], []
    for b, r in results.items():
        for (name, m), h in r["heads"].items():
            for layer in range(h["mean"].shape[0]):
                for head in range(h["mean"].shape[1]):
                    head_rows.append({"base": b, "contrast": name, "measure": m, "layer": layer, "head": head,
                                      **{k: float(h[k][layer, head]) for k in ("mean", "lo", "hi", "p", "q")},
                                      "meaningful": bool(h["meaningful"][layer, head]),
                                      "tv": float(r["tv"][name][layer, head]) if name in r["tv"] else np.nan})
        units = {(u["contrast"], u["measure"]): u for u in r["units"]}
        pooled += [{"base": b, **p, **{k: v for k, v in units[(p["contrast"], p["measure"])].items()
                                       if k not in ("contrast", "measure")}} for p in r["pooled"]]
        for (name, kind), sw in r["switches"].items():
            switch_rows += [{"base": b, "contrast": name, "labels": kind, **s} for s in sw]
        ind = r["induction"]
        for layer in range(ind["copy_mean"].shape[0]):
            for head in range(ind["copy_mean"].shape[1]):
                ind_rows.append({"base": b, "layer": layer, "head": head, "copy_mean": float(ind["copy_mean"][layer, head]),
                                 "induction": bool(ind["mask"][layer, head])})
    pd.DataFrame(head_rows).to_parquet(run / "heads.parquet", index=False)
    pd.DataFrame(pooled).to_csv(run / "pooled.csv", index=False)
    pd.DataFrame(switch_rows, columns=["base", "contrast", "labels", "layer", "head", "to", "from", "shift", "q"]).to_csv(
        run / "switches.csv", index=False)
    pd.DataFrame(ind_rows).to_csv(run / "induction_heads.csv", index=False)


def figures(run: Path, designs: dict[str, eng.Design], ext: dict[str, eng.Extraction], results: dict[str, dict],
            cfg: dict[str, Any], contrasts: dict[str, dict], specs: list[dict]) -> list[str]:
    out = []
    fig_dir = run / "figures"
    first = next(iter(results.values()))
    iv, copy_pos = first["induction"]["variant"], first["induction"]["copy_position"]
    ref = eng.make_contrast(next(iter(designs.values())), "ref", iv, iv)  # masks for the "attends to most" row
    for b, e in ext.items():
        bos, copy, role = {}, {}, {}
        for vi, v in enumerate(e.variants):
            bos[v] = e.att_row[:, vi, ..., 0].mean(axis=0)
            copy[v] = e.att_row[:, vi, ..., copy_pos].mean(axis=0)
            role[v] = eng.head_roles(eng.labels(e.att_row[:, vi], ref.codes, cfg["diffuse_threshold"])[0])
        rows = {"Mean BOS mass": bos, f"Copy-target mass (key {copy_pos})": copy, "What each head attends to most": role}
        kinds = {"Mean BOS mass": "mass", f"Copy-target mass (key {copy_pos})": "mass", "What each head attends to most": "label"}
        name = f"attention_maps_{b}.png"
        plots.plot_head_maps(rows, kinds, {k: "attention" for k in rows},
                             f"E3.4, {b} bases: per-head attention at the query (32), mean over {e.att_row.shape[0]} rows; "
                             f"boxed = induction heads; roles use {iv}'s masks", fig_dir / name,
                             outline={(f"Copy-target mass (key {copy_pos})", v): results[b]["induction"]["mask"] for v in e.variants},
                             label_names=eng.LABELS)
        out.append(name)
    panels, kinds, cbar, outline = {}, {}, {}, {}
    for b, r in results.items():
        for m, label in (("BOS", "Δ BOS"), ("Copy", "Δ copy-target")):
            row = f"{b}: {label}"
            kinds[row], cbar[row] = "delta", f"{label} (a − b)"
            panels[row] = {}
            for n, s in contrasts.items():
                if (n, m) in r["heads"]:  # restricted contrasts leave the other measures out
                    panels[row][f"{n}\n{s['a']} − {s['b']}"] = r["heads"][(n, m)]["mean"]
                    outline[(row, f"{n}\n{s['a']} − {s['b']}")] = r["heads"][(n, m)]["meaningful"]
    plots.plot_head_maps(panels, kinds, cbar, "Per-head Δ per contrast (boxed = meaningful: BH q ≤ 0.05, |Δ| ≥ 0.02)",
                         fig_dir / "contrast_maps.png", outline=outline)
    out.append("contrast_maps.png")
    main_contrast = next(s["contrast"] for s in specs if s["kind"] == "switch_count")
    mats = {}
    for b, r in results.items():
        for name, kind in ((main_contrast, "argmax"), (main_contrast, "mass"), ("placebo", "argmax")):
            mats[f"{b}, {name} ({kind})"] = r["transitions"][(name, kind)]
    plots.plot_transitions(mats, eng.LABELS, "Role switches: source → destination (McNemar, BH, ≥ 15-point shift)",
                           fig_dir / "transitions.png")
    out.append("transitions.png")
    return out


def fmt_ci(d: dict, lo: str = "lo", hi: str = "hi") -> str:
    return f"[{d[lo]:+.4f}, {d[hi]:+.4f}]"


def write_report(run: Path, cfg: dict, fallback: bool, designs: dict, results: dict, checks: dict, preds: list[dict],
                 timing: dict, contrasts: dict, figs: list[str]) -> None:
    first = next(iter(designs.values()))
    a = cfg["analysis"]
    layout = cfg["fallback_layout"] if fallback else cfg["layout"]
    title = "E3.4 fallback: repeated random block" if fallback else "E3.4: repeated phrase, induction as the positive control"
    L = [f"# Experiment 3, {title}", "",
         f"Run `{run.name}`. TL attention at the query (position 32); HF cross-check on {cfg['checks']['hf_rows']} rows "
         f"per base class. {cfg['n_bases']} rows per base class; {len(first.variants)} variants × {len(first.units)} "
         f"unit(s) per row ({', '.join(first.units)}). Layout: {layout}. Late = layers {a['late_layers'][0]}–"
         f"{a['late_layers'][1]}, the mean over their heads. Attention is averaged over a row's units before masses and "
         "labels; masks come from each contrast's treated arm (placebo: the masks of "
         f"{contrasts['placebo'].get('mask_from', contrasts['placebo']['a'])}).", "",
         f"Extraction: {timing['sequences']} sequences in {timing['seconds']:.0f} s ({timing['seq_per_s']:.0f} seq/s, TL only).",
         "", "## Checks", "", "| check | result |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in checks.items()]
    L += ["", "## Pre-registered predictions (exp3/PREDICTIONS_e34.md)", "", "| id | verdict | detail |", "|---|---|---|"]
    L += [f"| {p['id']} | {p['verdict']} | {p['detail']} |" for p in preds]
    ind0 = next(iter(results.values()))["induction"]
    L += ["", f"## Induction heads (mean copy-target mass ≥ {ind0['threshold']} at key {ind0['copy_position']} in "
          f"{ind0['variant']})", "",
          "| base | count | heads (mean copy-target mass) | " + " | ".join(cfg["ioi_heads"]) + " |",
          "|---|---|---|" + "---|" * len(cfg["ioi_heads"])]
    for b, r in results.items():
        ind = r["induction"]
        order = sorted(zip(*np.nonzero(ind["mask"])), key=lambda lh: -ind["copy_mean"][lh])
        names = ", ".join(f"{head_name(int(l_), int(h_))} {ind['copy_mean'][l_, h_]:.2f}" for l_, h_ in order) or "none"
        ioi = " | ".join(f"{'yes' if ind['mask'][lh] else 'no'} ({ind['copy_mean'][lh]:.2f})"
                         for lh in eng.parse_heads(cfg["ioi_heads"]))
        L.append(f"| {b} | {int(ind['mask'].sum())} | {names} | {ioi} |")
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
    L += ["", f"## Meaningful heads per contrast (BH q ≤ {a['meaningful']['q']} over 144 heads, |Δ| ≥ "
          f"{a['meaningful']['min_abs']}; + / −)", "",
          "| base | contrast | " + " | ".join(eng.MEASURES) + " |", "|---|---|" + "---|" * len(eng.MEASURES)]
    for b, r in results.items():
        for name in contrasts:
            cells = []
            for m in eng.MEASURES:
                h = r["heads"].get((name, m))
                cells.append("—" if h is None else
                             f"{int((h['meaningful'] & (h['mean'] > 0)).sum())} / {int((h['meaningful'] & (h['mean'] < 0)).sum())}")
            L.append(f"| {b} | {name} | " + " | ".join(cells) + " |")
    L += ["", f"## Role switches (McNemar, BH, ≥ {a['switch']['min_shift']:.0%} shift)", ""]
    for b, r in results.items():
        L.append(f"- {b}, placebo switching heads (either direction): argmax {len(r['placebo_switching']['argmax'])} "
                 f"{[head_name(*x) for x in r['placebo_switching']['argmax']]}, mass {len(r['placebo_switching']['mass'])}")
        for (name, kind), sw in r["switches"].items():
            desc = ", ".join(f"{head_name(s['layer'], s['head'])} {s['from']}→{s['to']} ({s['shift']:+.2f})" for s in sw) or "none"
            L.append(f"- {b}, {name}, {kind} labels: {len(sw)} heads; {desc}")
    L += ["", "## Transition matrices (argmax labels; nonzero source → destination counts)", ""]
    for b, r in results.items():
        for name in contrasts:
            if (name, "argmax") not in r["transitions"]:
                continue
            t = r["transitions"][(name, "argmax")]
            cells = ", ".join(f"{eng.LABELS[i]}→{eng.LABELS[j]} {t[i, j]}" for i, j in zip(*np.nonzero(t))) or "none"
            L.append(f"- {b}, {name}: {cells}")
    L += ["", "## TV distance between category profiles (per head; late mean, max)", "",
          "| base | contrast | late mean | max (head) |", "|---|---|---|---|"]
    lo, hi = a["late_layers"]
    for b, r in results.items():
        for name in r["tv"]:
            tv = r["tv"][name]
            i = np.unravel_index(tv.argmax(), tv.shape)
            L.append(f"| {b} | {name} | {tv[lo:hi + 1].mean():.4f} | {tv[i]:.4f} ({head_name(int(i[0]), int(i[1]))}) |")
    L += ["", "## Layer profile (mean Δ over rows and heads, per layer)", "",
          "| base | contrast | measure | " + " | ".join(f"L{i}" for i in range(12)) + " |", "|---|---|---|" + "---|" * 12]
    for b, r in results.items():
        for name in contrasts:
            for m in ("BOS", "Copy"):
                if (name, m) not in r["layer_profile"]:
                    continue
                prof = r["layer_profile"][(name, m)]
                L.append(f"| {b} | {name} | {m} | " + " | ".join(f"{x:+.3f}" for x in prof) + " |")
    L += ["", "## Figures", ""]
    L += [f"- `figures/{f}`" for f in figs]
    L.append("")
    (run / "exp3_e34_report.md").write_text("\n".join(L))


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Experiment 3, E3.4 (induction positive control)")
    parser.add_argument("--config", default="configs/exp3_e34.yaml", type=Path)
    args = parser.parse_args(argv)
    cfg_path = resolve(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())
    fallback = bool(cfg["fallback"])
    contrasts = cfg["fallback_contrasts"] if fallback else cfg["contrasts"]
    specs = cfg["fallback_predictions"] if fallback else cfg["predictions"]
    induction = cfg["fallback_induction"] if fallback else cfg["induction"]
    t0 = time.time()
    run = make_run_dir(resolve(cfg["paths"]["results_dir"]), tag="exp3e34fb" if fallback else "exp3e34")
    save_config_copy(cfg_path, run)
    shutil.copy2(PROJECT_ROOT / "exp3" / "PREDICTIONS_e34.md", run / "PREDICTIONS.md")
    save_versions(run, PROJECT_ROOT, code_dirs=("exp1", "exp2", "exp3"))
    (run / "figures").mkdir()
    log(f"run folder {run}{' (FALLBACK)' if fallback else ''}", t0)

    set_determinism(cfg["seed"])
    bos = cfg["bos_token_id"]
    tl = load_model("transformer_lens", cfg["backends"]["transformer_lens"]["model_name"], cfg["device"], cfg["dtype"])
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg["figures"]["tokenizer"])  # TL's tokenizer prepends BOS
    designs, info = build(cfg, tokenizer, fallback, t0)

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

    det = {b: eng.determinism_check(tl, d, ext[b], bos, chunk, atol, cfg["checks"]["determinism_rows"])
           for b, d in designs.items()}
    hf = load_model("huggingface", cfg["backends"]["huggingface"]["model_name"], cfg["device"], cfg["dtype"],
                    cfg["backends"]["huggingface"]["attn_implementation"])
    t_hf = time.time()
    cross = {b: eng.hf_check(tl, hf, d, bos, cfg["checks"]["hf_rows"], chunk * len(d.units), atol)
             for b, d in designs.items()}
    timing["hf_check_seconds"] = time.time() - t_hf
    log("determinism and cross-library checks done", t0)

    t_an = time.time()
    results = {b: e34.analyse(ext[b], designs[b], cfg, contrasts, induction, specs) for b in designs}
    timing["analysis_seconds"] = time.time() - t_an
    log("analysis done", t0)
    preds = e34.check_predictions(specs, results)

    max_plc = cfg["analysis"]["max_placebo_switches"]
    checks = {
        "design (base outside written positions; filler holds no written id; no BOS; repeats only where declared "
        "and present in every row)": "PASS (asserted per variant and base class)",
        ("fallback layout (controls differ only at 2–16; ids from the base pool)" if fallback else
         "E3.4 layout (controls differ only at 10–12, repeat-nl-1 only at 31; ids as in the table; other-nl shares no id)"):
            "PASS (asserted per base class)",
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
        ind = results[b]["induction"]
        ioi = {n: bool(ind["mask"][lh]) for n, lh in zip(cfg["ioi_heads"], eng.parse_heads(cfg["ioi_heads"]))}
        checks[f"{' and '.join(cfg['ioi_heads'])} among induction heads, {b}"] = (
            f"{'PASS' if all(ioi.values()) else 'FAIL'} ({int(ind['mask'].sum())} induction heads; "
            + ", ".join(f"{n} {'in' if v else 'out'} ({ind['copy_mean'][lh]:.2f})"
                        for (n, v), lh in zip(ioi.items(), eng.parse_heads(cfg["ioi_heads"]))) + ")")

    save_tables(run, designs, ext, results, cfg)
    (run / "checks.json").write_text(json.dumps(checks, indent=2))
    pred_out = [{**p, "stats": {b: r["predictions"][p["id"]]["stats"] for b, r in results.items()}} for p in preds]
    (run / "predictions.json").write_text(json.dumps(pred_out, indent=2, default=float))
    (run / "timing.json").write_text(json.dumps(timing, indent=2))
    (run / "design_summary.json").write_text(json.dumps(info, indent=2))
    figs = figures(run, designs, ext, results, cfg, contrasts, specs)
    write_report(run, cfg, fallback, designs, results, checks, preds, timing, contrasts, figs)
    log(f"done: {run / 'exp3_e34_report.md'}", t0)
    for p in preds:
        print(f"{p['id']}: {p['verdict']}  {p['detail']}")
    return run


if __name__ == "__main__":
    main()
