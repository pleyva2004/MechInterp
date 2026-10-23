"""Experiment 3, E3.5: punctuation, does a delimiter replace BOS as the resting place?

Run:  uv run --offline python -m exp3.run_e35 --config configs/exp3_e35.yaml

Writes results/<timestamp>_exp3e35/. Contents:
- the config, the pre-registered predictions and the library versions;
- design and attention arrays, plus per-head, pooled, switch and previous-token-head tables;
- checks.json, predictions.json and timing.json;
- figures and exp3_e35_report.md.
TransformerLens computes the attention. The random-context variants and the placebo are phrase-independent and are
computed once per row (exp3/e35.py). Hugging Face recomputes a subset of rows on identical ids as a cross-library
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
from exp2.analysis import head_name
from exp2.design import cached_base_block, derive_seed, id_pool
from exp3 import e35
from exp3 import engine as eng
from exp3 import plots_engine as plots

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DISPLAY = {"period": '"."', "comma": '","', "newline": '"\\n"', "x": '"x"', "seven": '"7"', "r": "r"}


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


def byte_ids(cfg: dict[str, Any], tokenizer) -> dict[str, int]:
    """Each byte's text must tokenize to exactly [id], decode back to the text, and the id must be 0-255."""
    out = {}
    for name, b in cfg["bytes"].items():
        ids = tokenizer.encode(b["text"])
        if ids != [b["id"]] or tokenizer.decode([b["id"]]) != b["text"] or not 0 <= b["id"] <= e35.MAX_BYTE_ID:
            raise ValueError(f"byte {name}: {b['text']!r} tokenizes to {ids}; config says [{b['id']}] (must be 0-255)")
        out[name] = int(b["id"])
    return out


def build(cfg: dict[str, Any], tokenizer, t0: float) -> tuple[dict[str, eng.Design], dict[str, Any], dict[str, int]]:
    """Base rows per class (cached, hash-tagged, all phrase ids excluded), the E3.5 design, and every design check."""
    phrases = phrase_ids(cfg, tokenizer)
    bytes_ = byte_ids(cfg, tokenizer)
    all_ids = sorted({i for ids in phrases.values() for i in ids})
    seq_dir = resolve(cfg["paths"]["sequences_dir"])
    designs, info = {}, {"phrase_ids": all_ids, "byte_ids": bytes_, "design": {}, "base_pool_size": {}}
    for base, id_range in cfg["bases"].items():
        if id_range[0] <= e35.MAX_BYTE_ID:
            raise ValueError(f"base class {base} {id_range} overlaps the byte ids 0-{e35.MAX_BYTE_ID}")
        rows = cached_base_block(f"exp3e35_{base}", id_range, all_ids, cfg["n_bases"], cfg["seq_len"], cfg["seed"],
                                 cfg["bos_token_id"], seq_dir)
        base_pool = id_pool(id_range, all_ids)
        seed = derive_seed(cfg["seed"], f"exp3e35/{base}")
        d = e35.build_e35(rows, phrases, bytes_, base_pool, seed, cfg["layout"])
        info["design"][base] = eng.check_design(d, cfg["bos_token_id"])
        e35.check_e35(d, phrases, bytes_, base_pool, cfg["layout"])
        info["base_pool_size"][base] = int(len(base_pool))
        designs[base] = d
        log(f"{base}: design built and checked ({len(d.variants)} variants x {len(d.units)} units x {d.rows} rows)", t0)
    return designs, info, bytes_


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
    head_rows, pooled, switch_rows, prev_rows = [], [], [], []
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
        pv = r["previous"]
        for v, roles in pv["roles"].items():
            for n, role in roles.items():
                prev_rows.append({"base": b, "variant": v, "head": n, "role": role,
                                  "previous_share": pv["previous_share"][v][n], "previous_mass": pv["previous_mass"][v][n]})
    pd.DataFrame(head_rows).to_parquet(run / "heads.parquet", index=False)
    pd.DataFrame(pooled).to_csv(run / "pooled.csv", index=False)
    pd.DataFrame(switch_rows, columns=["base", "contrast", "labels", "layer", "head", "to", "from", "shift", "q"]).to_csv(
        run / "switches.csv", index=False)
    pd.DataFrame(prev_rows).to_csv(run / "previous_heads.csv", index=False)


def figures(run: Path, designs: dict[str, eng.Design], ext: dict[str, eng.Extraction], results: dict[str, dict],
            cfg: dict[str, Any], bytes_: dict[str, int], specs: list[dict]) -> list[str]:
    out = []
    fig_dir = run / "figures"
    delim = int(cfg["layout"]["delimiter"])
    ds = e35.d_names(bytes_)
    for b, e in ext.items():
        d = designs[b]
        width = e.att_row.shape[-1]
        rows, kinds = {}, {}
        for ctx in e35.CONTEXTS:
            bos, dm, role = {}, {}, {}
            for x in ds:
                v = e35.variant(ctx, x)
                vi = e.index(v)
                bos[DISPLAY.get(x, x)] = e.att_row[:, vi, ..., 0].mean(axis=0)
                dm[DISPLAY.get(x, x)] = e.att_row[:, vi, ..., delim].mean(axis=0)
                codes = eng.category_codes(d.meta[v], d.key_len(v), width)
                role[DISPLAY.get(x, x)] = eng.head_roles(eng.labels(e.att_row[:, vi], codes, cfg["diffuse_threshold"])[0])
            rows[f"{ctx}: BOS mass"], kinds[f"{ctx}: BOS mass"] = bos, "mass"
            rows[f"{ctx}: key-{delim} mass"], kinds[f"{ctx}: key-{delim} mass"] = dm, "mass"
            rows[f"{ctx}: attends to most"], kinds[f"{ctx}: attends to most"] = role, "label"
        order = [f"{c}: {m}" for m in ("BOS mass", f"key-{delim} mass", "attends to most") for c in e35.CONTEXTS]
        name = f"attention_maps_{b}.png"
        plots.plot_head_maps({k: rows[k] for k in order}, kinds, {k: "attention" for k in order},
                             f"E3.5, {b} bases: per-head attention at the query (32), mean over {e.att_row.shape[0]} "
                             f"rows; columns = d at {delim}; roles use each variant's own masks", fig_dir / name,
                             label_names=eng.LABELS)
        out.append(name)
    panels, kinds, cbar, outline = {}, {}, {}, {}
    cols = [f"gain_{x}" for x in bytes_] + [f"x_{x}" for x in bytes_]
    for b, r in results.items():
        for m, label in (("BOS", "Δ BOS"), ("Delimiter", f"Δ key {delim}"), ("BOS+Delimiter", "Δ BOS + key")):
            row = f"{b}: {label}"
            kinds[row], cbar[row] = "delta", label
            panels[row] = {}
            for n in cols:
                if (n, m) in r["heads"]:
                    col = f"{n}\n{'phrase: d − r' if n.startswith('gain_') else 'phrase × d'}"
                    panels[row][col] = r["heads"][(n, m)]["mean"]
                    outline[(row, col)] = r["heads"][(n, m)]["meaningful"]
    plots.plot_head_maps(panels, kinds, cbar, "Per-head Δ: gain after the phrase and the phrase × d interaction "
                         "(boxed = meaningful: BH q ≤ 0.05, |Δ| ≥ 0.02)", fig_dir / "contrast_maps.png", outline=outline)
    out.append("contrast_maps.png")
    main = next(p["contrast"] for s in specs for p in ([s] + s.get("parts", [])) if p["kind"] == "switch_count")
    mats = {}
    for b, r in results.items():
        for name, kind in ((main, "argmax"), (main, "mass"), ("placebo", "argmax")):
            mats[f"{b}, {name} ({kind})"] = r["transitions"][(name, kind)]
    plots.plot_transitions(mats, eng.LABELS, "Role switches: source → destination (McNemar, BH, ≥ 15-point shift)",
                           fig_dir / "transitions.png")
    out.append("transitions.png")
    return out


def fmt_ci(d: dict, lo: str = "lo", hi: str = "hi") -> str:
    return f"[{d[lo]:+.4f}, {d[hi]:+.4f}]"


def write_report(run: Path, cfg: dict, designs: dict, results: dict, checks: dict, preds: list[dict], timing: dict,
                 contrasts: dict, interactions: dict, bytes_: dict[str, int], figs: list[str]) -> None:
    first = next(iter(designs.values()))
    a = cfg["analysis"]
    lo, hi = a["late_layers"]
    delim = int(cfg["layout"]["delimiter"])
    L = ["# Experiment 3, E3.5: punctuation, does a delimiter replace BOS as the resting place?", "",
         f"Run `{run.name}`. TL attention at the query (position 32); HF cross-check on {cfg['checks']['hf_rows']} rows "
         f"per base class. {cfg['n_bases']} rows per base class; {len(first.variants)} variants × {len(first.units)} "
         f"phrases per row. Layout: {cfg['layout']}. d ∈ "
         + ", ".join(f"{DISPLAY.get(n, n)} ({t})" for n, t in bytes_.items()) + ", r. "
         f"Late = layers {lo}–{hi}, the mean over their heads. Attention is averaged over a row's phrases before masses "
         f"and labels; Delimiter = key {delim} in every variant; masks come from each contrast's treated arm or its "
         "`mask_from` (rgain_d: phrase+d; placebo: phrase+period). The random variants and the placebo are one "
         "sequence per row, computed once.", "",
         f"Extraction: {timing['sequences']} sequences in {timing['seconds']:.0f} s ({timing['seq_per_s']:.0f} seq/s, "
         "TL only).", "", "## Checks", "", "| check | result |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in checks.items()]
    L += ["", "## Pre-registered predictions (exp3/PREDICTIONS_e35.md)", "", "| id | verdict | detail |", "|---|---|---|"]
    L += [f"| {p['id']} | {p['verdict']} | {p['detail']} |" for p in preds]

    L += ["", f"## The grid: late pooled per-row Δ (95% CI); verdict letter: + / − (CI excludes 0), = (TOST ±{a['margin']}), "
          "? (inconclusive)", ""]
    sym = {"positive": "+", "negative": "−", "equivalent": "=", "inconclusive": "?"}
    for m in ("Delimiter", "BOS", "BOS+Delimiter"):
        L += [f"### {m}{f' (key {delim})' if m == 'Delimiter' else ''}", "",
              "| base | d | gain after phrase (phrase+d − phrase+r) | gain on random (random+d − random+r) | "
              "interaction | context at d (phrase+d − random+d) |", "|---|---|---|---|---|---|"]
        for b, r in results.items():
            idx = {(p["contrast"], p["measure"]): p for p in r["pooled"]}
            for x in e35.d_names(bytes_):
                cells = []
                for n in (f"gain_{x}", f"rgain_{x}", f"x_{x}", f"ctx_{x}"):
                    p = idx.get((n, m))
                    cells.append("—" if p is None else f"{p['mean']:+.4f} {fmt_ci(p)} {sym[p['verdict']]}")
                L.append(f"| {b} | {DISPLAY.get(x, x)} | " + " | ".join(cells) + " |")
        L.append("")

    L += ["## Exp 1's previous-token heads (role per variant; must stay Previous)", ""]
    for b, r in results.items():
        pv = r["previous"]
        L.append(f"- {b}: {'PASS' if pv['ok'] else 'FAIL'}. Not Previous: "
                 + (", ".join(f"{o['head']} in {o['variant']} ({o['role']})" for o in pv["not_previous"]) or "none")
                 + ". Switches from Previous: "
                 + (", ".join(f"{head_name(s['layer'], s['head'])} in {s['contrast']} →{s['to']} ({s['shift']:+.2f})"
                              for s in pv["switches_from_previous"]) or "none") + ".")
        share = {n: min(pv["previous_share"][v][n] for v in pv["roles"]) for n in pv["heads"]}
        L.append("  Lowest Previous share over the variants: " + ", ".join(f"{n} {s:.2f}" for n, s in share.items()))

    keep = cfg.get("report_measures", list(e35.MEASURES))
    L += ["", "## Late pooled per-row Δ, every contrast and interaction (measures: " + ", ".join(keep) + "; pooled.csv "
          "has all)", "",
          "95% / 90% CIs bootstrap rows; verdict: positive / negative (95% CI excludes 0), equivalent (TOST, 90% CI inside "
          f"±{a['margin']}), else inconclusive. Per phrase: bootstrap over the per-unit means, with a sign count.", "",
          "| base | contrast | measure | Δ | 95% CI | 90% CI | Wilcoxon p | verdict | per unit [95% CI] | units > 0 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for b, r in results.items():
        units = {(u["contrast"], u["measure"]): u for u in r["units"]}
        for p in r["pooled"]:
            if p["measure"] not in keep:
                continue
            u = units[(p["contrast"], p["measure"])]
            L.append(f"| {b} | {p['contrast']} | {p['measure']} | {p['mean']:+.4f} | {fmt_ci(p)} | {fmt_ci(p, 'lo90', 'hi90')} | "
                     f"{p['wilcoxon_p']:.2g} | {p['verdict']} | {u['unit_mean']:+.4f} {fmt_ci(u, 'unit_lo', 'unit_hi')} | "
                     f"{u['unit_n_pos']}/{u['unit_units']} |")
    L += ["", "Contrast definitions: " + "; ".join(f"{n} = {w}" for n, w in next(iter(results.values()))["what"].items()),
          "", f"## Meaningful heads (BH q ≤ {a['meaningful']['q']} over 144 heads, |Δ| ≥ {a['meaningful']['min_abs']}; "
          "+ / −)", "", "| base | contrast | " + " | ".join(e35.MEASURES) + " |", "|---|---|" + "---|" * len(e35.MEASURES)]
    for b, r in results.items():
        for name in list(contrasts) + list(interactions):
            cells = []
            for m in e35.MEASURES:
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
        for name in [f"gain_{x}" for x in bytes_] + [f"x_{x}" for x in bytes_] + ["ctx_r", "placebo"]:
            for m in ("BOS", "Delimiter"):
                if (name, m) not in r["layer_profile"]:
                    continue
                prof = r["layer_profile"][(name, m)]
                L.append(f"| {b} | {name} | {m} | " + " | ".join(f"{x:+.3f}" for x in prof) + " |")
    L += ["", "## Figures", ""]
    L += [f"- `figures/{f}`" for f in figs]
    L.append("")
    (run / "exp3_e35_report.md").write_text("\n".join(L))


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Experiment 3, E3.5 (punctuation: delimiter vs BOS as resting place)")
    parser.add_argument("--config", default="configs/exp3_e35.yaml", type=Path)
    args = parser.parse_args(argv)
    cfg_path = resolve(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())
    contrasts, interactions, specs = cfg["contrasts"], cfg["interactions"], cfg["predictions"]
    t0 = time.time()
    run = make_run_dir(resolve(cfg["paths"]["results_dir"]), tag="exp3e35")
    save_config_copy(cfg_path, run)
    shutil.copy2(PROJECT_ROOT / "exp3" / "PREDICTIONS_e35.md", run / "PREDICTIONS.md")
    save_versions(run, PROJECT_ROOT, code_dirs=("exp1", "exp2", "exp3"))
    (run / "figures").mkdir()
    log(f"run folder {run}", t0)

    set_determinism(cfg["seed"])
    bos = cfg["bos_token_id"]
    tl = load_model("transformer_lens", cfg["backends"]["transformer_lens"]["model_name"], cfg["device"], cfg["dtype"])
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg["figures"]["tokenizer"])  # TL's tokenizer prepends BOS
    designs, info, bytes_ = build(cfg, tokenizer, t0)
    shared = e35.shared_variants(bytes_)

    chunks = {"per_unit": cfg["extraction"]["chunk_rows"], "shared": cfg["extraction"]["shared_chunk_rows"]}
    atol, late = cfg["checks"]["row_sum_atol"], cfg["analysis"]["late_layers"]
    ext, parts = {}, {}
    for b, d in designs.items():
        ext[b], parts[b] = e35.extract_split(tl, d, shared, bos, late, chunks, atol,
                                             log=lambda m, b=b: log(f"{b}: {m}", t0))
        log(f"{b}: extraction done ({ext[b].n_sequences} sequences, {ext[b].seconds:.0f} s, "
            f"{ext[b].n_sequences / ext[b].seconds:.0f} seq/s)", t0)
    n_seq = sum(e.n_sequences for e in ext.values())
    secs = sum(e.seconds for e in ext.values())
    timing = {"sequences": n_seq, "seconds": secs, "seq_per_s": n_seq / secs,
              "per_base": {b: {"sequences": e.n_sequences, "seconds": e.seconds} for b, e in ext.items()}}

    det = {b: {n: eng.determinism_check(tl, pd_, pe, bos, chunks[n], atol, cfg["checks"]["determinism_rows"])
               for n, (pd_, pe) in parts[b].items()} for b in designs}
    hf = load_model("huggingface", cfg["backends"]["huggingface"]["model_name"], cfg["device"], cfg["dtype"],
                    cfg["backends"]["huggingface"]["attn_implementation"])
    t_hf = time.time()
    cross = {b: {n: eng.hf_check(tl, hf, pd_, bos, cfg["checks"]["hf_rows"], chunks[n] * len(pd_.units), atol)
                 for n, (pd_, _) in parts[b].items()} for b in designs}
    timing["hf_check_seconds"] = time.time() - t_hf
    del parts
    log("determinism and cross-library checks done", t0)

    t_an = time.time()
    results = {b: e35.analyse(ext[b], designs[b], cfg, contrasts, interactions, specs) for b in designs}
    timing["analysis_seconds"] = time.time() - t_an
    log("analysis done", t0)
    preds = e35.check_predictions(specs, results)

    max_plc = cfg["analysis"]["max_placebo_switches"]
    checks = {
        "design (base outside written positions; filler holds no written id; no BOS; no repeats within a row)":
            "PASS (asserted per variant and base class)",
        "E3.5 layout (d arms differ from r arms only at 29; phrase arms from random arms and the placebo from none only "
        "at 26–28; none = the base row; random variants identical across phrases; the byte is the row's only id ≤ 255)":
            "PASS (asserted per base class)",
        "byte ids (gpt2 tokenizer)": "PASS (" + ", ".join(f"{json.dumps(cfg['bytes'][n]['text'])} → {t}"
                                                          for n, t in bytes_.items()) + ")",
        "attention rows sum to 1": f"PASS (atol {atol:g}, asserted per batch; max |sum − 1| "
                                   f"{max(e.max_row_sum_dev for e in ext.values()):.1e})",
    }
    for b in designs:
        for n in det[b]:
            checks[f"determinism, {b}, {n} variants ({det[b][n]['rows']} rows)"] = (
                f"{'PASS' if det[b][n]['exact'] else 'FAIL'} (max |Δ| {det[b][n]['max_abs_delta']:.1e})")
            c = cross[b][n]
            checks[f"TL vs HF, {b}, {n} variants ({c['rows']} rows, {c['sequences']} sequences)"] = (
                f"{'PASS' if c['max_abs_delta'] < cfg['checks']['hf_atol'] else 'FAIL'} "
                f"(max |Δ| {c['max_abs_delta']:.1e}, argmax-key agreement {c['argmax_agreement']:.4%})")
        n_plc = len(results[b]["placebo_switching"]["argmax"])
        checks[f"placebo switching heads ≤ {max_plc}, {b}"] = (
            f"{'PASS' if n_plc <= max_plc else 'FAIL'} ({n_plc} argmax, "
            f"{len(results[b]['placebo_switching']['mass'])} mass)")
        pv = results[b]["previous"]
        checks[f"Exp 1 previous-token heads keep Previous in all {len(designs[b].variants)} variants, {b}"] = (
            f"{'PASS' if pv['ok'] else 'FAIL (flags a confound)'} ({len(pv['not_previous'])} head × variant cells not "
            f"Previous; {len(pv['switches_from_previous'])} switches from Previous)")

    save_tables(run, designs, ext, results, cfg)
    (run / "checks.json").write_text(json.dumps(checks, indent=2))
    pred_out = [{**p, "stats": {b: r["predictions"][p["id"]]["stats"] for b, r in results.items()}} for p in preds]
    (run / "predictions.json").write_text(json.dumps(pred_out, indent=2, default=float))
    (run / "previous_heads.json").write_text(json.dumps({b: r["previous"] for b, r in results.items()}, indent=2,
                                                        default=float))
    (run / "timing.json").write_text(json.dumps(timing, indent=2))
    (run / "design_summary.json").write_text(json.dumps(info, indent=2))
    figs = figures(run, designs, ext, results, cfg, bytes_, specs)
    write_report(run, cfg, designs, results, checks, preds, timing, contrasts, interactions, bytes_, figs)
    log(f"done: {run / 'exp3_e35_report.md'}", t0)
    for p in preds:
        print(f"{p['id']}: {p['verdict']}  {p['detail']}")
    return run


if __name__ == "__main__":
    main()
