"""Experiment 3, E3.1: query inside vs outside the phrase.

Run:  uv run --offline python -m exp3.run_e31 --config configs/exp3_e31.yaml

Writes results/<timestamp>_exp3e31/. Contents:
- the config, the pre-registered predictions and the library versions;
- design and attention arrays, plus per-head, pooled, switch, logit-check and query-log-P tables;
- checks.json, predictions.json and timing.json;
- figures and exp3_e31_report.md.
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
from exp1.sequences import prepend_bos
from exp1ext.features import token_features
from exp2.analysis import head_name
from exp2.design import cached_base_block, derive_seed, id_pool, matched_pools
from exp3 import e31
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


def build(cfg: dict[str, Any], tokenizer, t0: float) -> tuple[dict[str, eng.Design], dict[str, Any], dict]:
    """Base rows per class (cached, hash-tagged, all phrase ids excluded), the E3.1 design, and every design check."""
    phrases = phrase_ids(cfg, tokenizer)
    all_ids = sorted({i for ids in phrases.values() for i in ids})
    seq_dir = resolve(cfg["paths"]["sequences_dir"])
    lo = min(r[0] for r in cfg["token_classes"].values())
    hi = max(r[1] for r in cfg["token_classes"].values())
    features = token_features(np.arange(lo, hi + 1), tokenizer)
    pools = {n: matched_pools(ids, features, cfg["match_features"], cfg["token_classes"]) for n, ids in phrases.items()}
    designs = {}
    info = {"phrase_ids": all_ids, "design": {}, "base_pool_size": {},
            "matched_pool_sizes": {n: [len(p) for p in pl] for n, pl in pools.items()},
            "oprime": {n: {"phrase": o, "id": phrases[o][2], "text": tokenizer.decode(phrases[n][:2] + [phrases[o][2]])}
                       for n, o in cfg["oprime"].items()},
            "oprime_is_derangement": sorted(cfg["oprime"].values()) == sorted(cfg["oprime"])}
    for base, id_range in cfg["bases"].items():
        rows = cached_base_block(f"exp3e31_{base}", id_range, all_ids, cfg["n_bases"], cfg["seq_len"], cfg["seed"],
                                 cfg["bos_token_id"], seq_dir)
        seed = derive_seed(cfg["seed"], f"exp3e31/{base}")
        d = e31.build_e31(rows, phrases, cfg["oprime"], pools, seed, cfg["layout"])
        info["design"][base] = eng.check_design(d, cfg["bos_token_id"])
        e31.check_e31(d, phrases, cfg["oprime"], pools, cfg["layout"])
        info["base_pool_size"][base] = int(len(id_pool(id_range, all_ids)))
        designs[base] = d
        log(f"{base}: design built and checked ({len(d.variants)} variants x {len(d.units)} units x {d.rows} rows)", t0)
    return designs, info, phrases


def query_logp(model, design: eng.Design, bos: int, n_rows: int, batch: int) -> dict[str, np.ndarray]:
    """log P of O, O', M and r at the query (32), from the logits at 31, under the PV, VP and rr contexts, for the
    first n_rows rows. One forward pass per context variant: PV+O's prefix 0..31 is that of every PV+x variant, VP+O's
    of VP+r, rr+O's of every rr+x variant."""
    import torch

    cand = e31.query_candidates(design, n_rows)  # [n, U, 4]
    flat_cand = torch.as_tensor(cand.reshape(-1, cand.shape[-1]))
    out = {}
    for ctx, var in e31.CONTEXTS.items():
        s = design.seqs[var][:n_rows]
        tokens = torch.as_tensor(prepend_bos(s.reshape(-1, s.shape[-1]), bos))
        pos = tokens.shape[1] - 2  # model position 31: its logits predict the token at the query
        parts = []
        with torch.inference_mode():
            for i in range(0, len(tokens), batch):
                lp = torch.log_softmax(model(tokens[i:i + batch], return_type="logits")[:, pos, :].double(), -1)
                parts.append(lp.gather(1, flat_cand[i:i + batch]).numpy())
        out[ctx] = np.concatenate(parts).reshape(cand.shape)
    return out


def save_tables(run: Path, designs: dict[str, eng.Design], ext: dict[str, eng.Extraction], results: dict[str, dict],
                manip: dict[str, dict], cfg: dict[str, Any]) -> None:
    np.savez_compressed(run / "design.npz", **{f"{b}/base": d.base[:, 0] for b, d in designs.items()},
                        **{f"{b}/{v}": s for b, d in designs.items() for v, s in d.seqs.items()})
    arrays = ("att_row", "att_late_unit", "att_unit") if cfg.get("outputs", {}).get("save_att_row", True) else \
        ("att_late_unit", "att_unit")
    np.savez_compressed(run / "attention.npz",
                        **{f"{b}/{k}": getattr(e, k) for b, e in ext.items() for k in arrays},
                        **{f"{b}/key_len": e.key_len for b, e in ext.items()},
                        variants=np.array(next(iter(ext.values())).variants), units=np.array(next(iter(ext.values())).units))
    np.savez_compressed(run / "labels.npz", **{f"{b}/{k}": v for b, r in results.items() for k, v in r["labels"].items()})
    head_rows, pooled, switch_rows, logit_rows = [], [], [], []
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
        logit_rows += [{"base": b, **x} for x in r["logit"]]
    pd.DataFrame(head_rows).to_parquet(run / "heads.parquet", index=False)
    pd.DataFrame(pooled).to_csv(run / "pooled.csv", index=False)
    pd.DataFrame(switch_rows, columns=["base", "contrast", "labels", "layer", "head", "to", "from", "shift", "q"]).to_csv(
        run / "switches.csv", index=False)
    pd.DataFrame(logit_rows).to_csv(run / "logit_check.csv", index=False)
    (run / "query_logp.json").write_text(json.dumps(manip, indent=2))


def figures(run: Path, designs: dict[str, eng.Design], ext: dict[str, eng.Extraction], results: dict[str, dict],
            cfg: dict[str, Any], contrasts: dict[str, dict], differences: dict[str, dict], specs: list[dict]) -> list[str]:
    out = []
    fig_dir = run / "figures"
    ref = eng.make_contrast(next(iter(designs.values())), "ref", "PV+O", "PV+O")  # PV+O's masks for every variant
    ctx = [int(p) for p in cfg["layout"]["context"]]
    for b, e in ext.items():
        bos, phrase, role = {}, {}, {}
        for vi, v in enumerate(e.variants):
            bos[v] = e.att_row[:, vi, ..., 0].mean(axis=0)
            phrase[v] = e.att_row[:, vi, ..., ctx[0]:ctx[1] + 1].sum(axis=-1).mean(axis=0)
            role[v] = eng.head_roles(eng.labels(e.att_row[:, vi], ref.codes, cfg["diffuse_threshold"])[0])
        rows = {"Mean BOS mass": bos, f"Mass on keys {ctx[0]}–{ctx[1]}": phrase, "What each head attends to most": role}
        kinds = {"Mean BOS mass": "mass", f"Mass on keys {ctx[0]}–{ctx[1]}": "mass", "What each head attends to most": "label"}
        name = f"attention_maps_{b}.png"
        plots.plot_head_maps(rows, kinds, {k: "attention" for k in rows},
                             f"E3.1, {b} bases: per-head attention at the query (32), mean over {e.att_row.shape[0]} "
                             "rows; roles use PV+O's masks (Span 30, Previous 31, Self 32)", fig_dir / name,
                             label_names=eng.LABELS)
        out.append(name)
    panels, kinds, cbar, outline = {}, {}, {}, {}
    cols = {n: f"{n}\n{s['a']} − {s['b']}" for n, s in contrasts.items()}
    cols.update({n: f"{n}\n{s['plus']} − {s['minus']}" for n, s in differences.items()})
    for b, r in results.items():
        for m, label in (("BOS", "Δ BOS"), ("Phrase", "Δ phrase-position")):
            row = f"{b}: {label}"
            kinds[row], cbar[row] = "delta", f"{label} (treated − control)"
            panels[row] = {}
            for n, col in cols.items():
                if (n, m) in r["heads"]:
                    panels[row][col] = r["heads"][(n, m)]["mean"]
                    outline[(row, col)] = r["heads"][(n, m)]["meaningful"]
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


def write_report(run: Path, cfg: dict, designs: dict, info: dict, results: dict, checks: dict, preds: list[dict],
                 timing: dict, manip: dict, contrasts: dict, differences: dict, figs: list[str]) -> None:
    first = next(iter(designs.values()))
    a = cfg["analysis"]
    L = ["# Experiment 3, E3.1: query inside vs outside the phrase", "",
         f"Run `{run.name}`. TL attention at the query (position 32); HF cross-check on {cfg['checks']['hf_rows']} rows "
         f"per base class. {cfg['n_bases']} rows per base class; {len(first.variants)} variants × {len(first.units)} "
         f"phrases per row ({', '.join(first.units)}). Layout: {cfg['layout']}. Late = layers {a['late_layers'][0]}–"
         f"{a['late_layers'][1]}, the mean over their heads. Attention is averaged over a row's phrases before masses "
         "and labels; masks come from each contrast's treated arm (differences and placebo: PV+O's masks, Span 30, "
         "Previous 31, Self 32, phrase-position = keys 30–31).", "",
         f"Extraction: {timing['sequences']} sequences in {timing['seconds']:.0f} s ({timing['seq_per_s']:.0f} seq/s, TL only).",
         "", "## Checks", "", "| check | result |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in checks.items()]
    L += ["", "## Pre-registered predictions (exp3/PREDICTIONS_e31.md)", "", "| id | verdict | detail |", "|---|---|---|"]
    L += [f"| {p['id']}{' (' + p['note'] + ')' if p['note'] else ''} | {p['verdict']} | {p['detail']} |" for p in preds]
    L += ["", "## O′ per phrase (fixed in the config before any run)", "",
          f"Derangement (every object once as O′): {info['oprime_is_derangement']}.", "",
          "| phrase | O′ (from) | PV+O′ text |", "|---|---|---|"]
    L += [f"| {n} | {o['id']} ({o['phrase']}) | {o['text']!r} |" for n, o in info["oprime"].items()]
    L += ["", f"## Query log P at 32 (first {next(iter(manip.values()))['rows']} rows; reported, not gating)", "",
          "Mean log P of each query id given the context's prefix (logits at 31). O beats x = share of row × phrase "
          "instances with log P(O) > log P(x) under PV. Lift = PV − rr, per query.", "",
          "| base | context | O | O′ | M | r |", "|---|---|---|---|---|---|"]
    for b, s in manip.items():
        for ctx, row in s["mean"].items():
            L.append(f"| {b} | {ctx} | " + " | ".join(f"{row[x]:.2f}" for x in e31.QUERIES) + " |")
    L += ["", "| base | O beats O′ (PV) | O beats M (PV) | O beats r (PV) | lift O | lift O′ | lift M | lift r |",
          "|---|---|---|---|---|---|---|---|"]
    for b, s in manip.items():
        L.append(f"| {b} | " + " | ".join(f"{s['pv_O_beats'][x]:.0%}" for x in e31.QUERIES[1:]) + " | "
                 + " | ".join(f"{s['lift_pv_minus_rr'][x]:+.2f}" for x in e31.QUERIES) + " |")
    L += ["", "Per phrase, PV context: mean log P(O) − log P(O′).", "",
          "| base | " + " | ".join(first.units) + " |", "|---|" + "---|" * len(first.units)]
    for b, s in manip.items():
        L.append(f"| {b} | " + " | ".join(f"{s['per_phrase_O_minus_Op'][u]:+.2f}" for u in first.units) + " |")
    L += ["", "## Late pooled per-row Δ, every contrast and measure", "",
          "Two-arm: a − b. Difference: (plus) − (minus), per row. 95% / 90% CIs bootstrap rows; verdict: positive / "
          f"negative (95% CI excludes 0), equivalent (TOST, 90% CI inside ±{a['margin']}), else inconclusive. Per "
          "phrase: bootstrap over the per-phrase means, with a sign count.", "",
          "| base | contrast | arms | measure | Δ | 95% CI | 90% CI | Wilcoxon p | verdict | per phrase [95% CI] | phrases > 0 |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for b, r in results.items():
        units = {(u["contrast"], u["measure"]): u for u in r["units"]}
        for p in r["pooled"]:
            u = units[(p["contrast"], p["measure"])]
            L.append(f"| {b} | {p['contrast']} | {p['arms']} | {p['measure']} | {p['mean']:+.4f} | {fmt_ci(p)} | "
                     f"{fmt_ci(p, 'lo90', 'hi90')} | {p['wilcoxon_p']:.2g} | {p['verdict']} | "
                     f"{u['unit_mean']:+.4f} {fmt_ci(u, 'unit_lo', 'unit_hi')} | {u['unit_n_pos']}/{u['unit_units']} |")
    L += ["", f"## logit(BOS) check on the differences (late mean per row; mass clipped to [{a['logit_eps']:g}, "
          f"1 − {a['logit_eps']:g}])", "",
          "| base | contrast | Δ logit | 95% CI | verdict | same sign as the mass Δ |", "|---|---|---|---|---|---|"]
    for b, r in results.items():
        mass = {p["contrast"]: p for p in r["pooled"] if p["measure"] == "BOS"}
        for x in r["logit"]:
            same = np.sign(x["mean"]) == np.sign(mass[x["contrast"]]["mean"])
            L.append(f"| {b} | {x['contrast']} | {x['mean']:+.4f} | {fmt_ci(x)} | {x['verdict']} | {'yes' if same else 'no'} |")
    measures = eng.MEASURES
    L += ["", f"## Meaningful heads per contrast (BH q ≤ {a['meaningful']['q']} over 144 heads, |Δ| ≥ "
          f"{a['meaningful']['min_abs']}; + / −; — = empty mask)", "",
          "| base | contrast | " + " | ".join(measures) + " |", "|---|---|" + "---|" * len(measures)]
    for b, r in results.items():
        for name in list(contrasts) + list(differences):
            cells = []
            for m in measures:
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
            L.append(f"- {b}, {name}, {kind} labels: {len(sw)} switches; {desc}")
    L += ["", "## Transition matrices (argmax labels; nonzero source → destination counts)", ""]
    for b, r in results.items():
        for name in contrasts:
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
        for name in list(contrasts) + list(differences):
            for m in ("BOS", "Phrase"):
                if (name, m) not in r["layer_profile"]:
                    continue
                prof = r["layer_profile"][(name, m)]
                L.append(f"| {b} | {name} | {m} | " + " | ".join(f"{x:+.3f}" for x in prof) + " |")
    L += ["", "## Figures", ""]
    L += [f"- `figures/{f}`" for f in figs]
    L.append("")
    (run / "exp3_e31_report.md").write_text("\n".join(L))


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Experiment 3, E3.1 (query inside vs outside the phrase)")
    parser.add_argument("--config", default="configs/exp3_e31.yaml", type=Path)
    args = parser.parse_args(argv)
    cfg_path = resolve(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())
    contrasts, differences, specs = cfg["contrasts"], cfg["differences"], cfg["predictions"]
    t0 = time.time()
    run = make_run_dir(resolve(cfg["paths"]["results_dir"]), tag="exp3e31")
    save_config_copy(cfg_path, run)
    shutil.copy2(PROJECT_ROOT / "exp3" / "PREDICTIONS_e31.md", run / "PREDICTIONS.md")
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

    det = {b: eng.determinism_check(tl, d, ext[b], bos, chunk, atol, cfg["checks"]["determinism_rows"])
           for b, d in designs.items()}
    t_m = time.time()
    n_m = cfg["checks"]["manipulation_rows"]
    manip = {b: e31.summarize_query_logp(query_logp(tl, d, bos, n_m, cfg["checks"]["manipulation_batch"]), d.units)
             for b, d in designs.items()}
    timing["query_logp_seconds"] = time.time() - t_m
    hf = load_model("huggingface", cfg["backends"]["huggingface"]["model_name"], cfg["device"], cfg["dtype"],
                    cfg["backends"]["huggingface"]["attn_implementation"])
    t_hf = time.time()
    cross = {b: eng.hf_check(tl, hf, d, bos, cfg["checks"]["hf_rows"], chunk * len(d.units), atol)
             for b, d in designs.items()}
    timing["hf_check_seconds"] = time.time() - t_hf
    log("determinism, query log P and cross-library checks done", t0)

    t_an = time.time()
    results = {b: e31.analyse(ext[b], designs[b], cfg, contrasts, differences, specs) for b in designs}
    timing["analysis_seconds"] = time.time() - t_an
    log("analysis done", t0)
    preds = e31.check_predictions(specs, results)

    max_plc = cfg["analysis"]["max_placebo_switches"]
    checks = {
        "design (base outside written positions, query included; filler holds no written id; no BOS; no repeated id "
        "in any row)": "PASS (asserted per variant and base class)",
        "E3.1 layout (rr+x vs PV+x and VP+x vs PV+x only at 30–31; PV+x / rr+x families only at 32; PVO+r vs rr+r only "
        "at 29–31; placebo vs rr+r only at 30–32; ids as in the table; O′ another phrase's object in O's matched pool; "
        "M and placebo ids in their pools)": "PASS (asserted per base class)",
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

    save_tables(run, designs, ext, results, manip, cfg)
    (run / "checks.json").write_text(json.dumps(checks, indent=2))
    pred_out = [{**p, "stats": {b: r["predictions"][p["id"]]["stats"] for b, r in results.items()}} for p in preds]
    (run / "predictions.json").write_text(json.dumps(pred_out, indent=2, default=float))
    (run / "timing.json").write_text(json.dumps(timing, indent=2))
    (run / "design_summary.json").write_text(json.dumps(info, indent=2))
    figs = figures(run, designs, ext, results, cfg, contrasts, differences, specs)
    write_report(run, cfg, designs, info, results, checks, preds, timing, manip, contrasts, differences, figs)
    log(f"done: {run / 'exp3_e31_report.md'}", t0)
    for p in preds:
        print(f"{p['id']}: {p['verdict']}  {p['detail']}")
    return run


if __name__ == "__main__":
    main()
