"""exp2 entrypoint.   Run:  uv run python -m exp2.run --config configs/exp2.yaml

Builds base rows (rare and common), writes each phrase and its controls into each placement, extracts the
final-query attention in TransformerLens and Hugging Face from the same id tensors, computes paired contrasts per
head, checks the pre-registered predictions, and writes everything to results/<YYYYMMDD-HHMMSS>_exp2/.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from transformers import AutoTokenizer

from exp1.backends import bos_token_id, check_rows_sum_to_one, get_final_attention, load_model, set_determinism
from exp1.compare import bh_fdr, label_shift, split_half
from exp1.metrics import LABELS, classify, compute_metrics
from exp1.plots import plot_diff_heatmaps
from exp1.runlog import make_run_dir, save_config_copy, save_versions
from exp1.sequences import prepend_bos, sequences_hash
from exp1.validation import check_determinism
from exp1ext.features import token_features
from exp2.analysis import (additivity_tables, cell_summary, phrase_level, check_prediction, contrast_table, pool_over_phrases,
                           surprisal_slopes, tv_table)
from exp2.design import (VARIANTS, build_custom, build_variants, cached_base_block, check_custom, check_variants, derive_seed,
                         matched_pools, positions_to_slots)
from exp2.plots import (plot_layer_profile, plot_meaningful_counts, plot_phrase_terms, plot_surprisal_variants,
                        plot_tracked_forest)
from exp2.report import write_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_NAMES = {"transformer_lens": "TL", "huggingface": "HF"}
PROB_METRICS = ("sink_mass", "slot_mass", "prev_mass")  # probability-valued: compared across backends at atol
HEATMAP_METRICS = ("sink_mass", "slot_mass")
NONE = "none"


def log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def build_design(cfg: dict[str, Any], tokenizer) -> tuple[dict[str, np.ndarray], dict[tuple, np.ndarray], dict[str, Any]]:
    """Returns (bases {class: [n, L]}, seqs {(base, phrase, placement, variant): [n, L]} for every non-`none`
    variant, info) after checking every invariant in exp2.design.check_variants."""
    seq_len, bos = cfg["seq_len"], cfg["bos_token_id"]
    all_ids: set[int] = set()
    for name, ph in cfg["phrases"].items():
        encoded = tokenizer.encode(ph["text"])
        if encoded != ph["ids"]:
            raise ValueError(f"phrase {name}: {ph['text']!r} tokenizes to {encoded}, config says {ph['ids']}")
        all_ids |= set(ph["ids"])
    for placement, positions in cfg["placements"].items():
        positions_to_slots(positions, seq_len)  # validates
        for name, ph in cfg["phrases"].items():
            if len(ph["ids"]) != len(positions):
                raise ValueError(f"phrase {name} has {len(ph['ids'])} ids but placement {placement} has {len(positions)} slots")

    lo = min(r[0] for r in cfg["token_classes"].values())
    hi = max(r[1] for r in cfg["token_classes"].values())
    features = token_features(np.arange(lo, hi + 1), tokenizer)
    pools = {name: matched_pools(ph["ids"], features, cfg["match_features"], cfg["token_classes"])
             for name, ph in cfg["phrases"].items()}

    data_dir = PROJECT_ROOT / cfg["paths"]["sequences_dir"]
    bases = {b: cached_base_block(f"base_{b}", rng, sorted(all_ids), cfg["n_bases"], seq_len, cfg["seed"], bos, data_dir)
             for b, rng in cfg["bases"].items()}
    seqs: dict[tuple, np.ndarray] = {}
    for b, base in bases.items():
        for name, ph in cfg["phrases"].items():
            for placement, positions in cfg["placements"].items():
                slots = positions_to_slots(positions, seq_len)
                variants = build_variants(base, slots, ph["ids"], ph["shuffled"], pools[name],
                                          derive_seed(cfg["seed"], f"{b}/{name}/{placement}/matched"))
                check_variants(variants, base, slots, ph["ids"], ph["shuffled"], pools[name], bos)
                for v in VARIANTS[1:]:
                    seqs[(b, name, placement, v)] = variants[v]
        for v, spec in cfg.get("custom_variants", {}).items():
            seqs[(b, spec["phrase"], spec["placement"], v)] = build_custom(base, spec["writes"], seq_len)
            check_custom(seqs[(b, spec["phrase"], spec["placement"], v)], base, spec["writes"], seq_len, bos)
    info = {"pool_sizes": {name: [len(p) for p in pl] for name, pl in pools.items()},
            "phrase_ids": sorted(all_ids)}
    return bases, seqs, info


def extract(model, tokens: np.ndarray, cfg: dict[str, Any], positions: list[int]) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, float, np.ndarray]:
    """tokens WITHOUT BOS. Returns (metrics {name: float64 [n, L, H]}, labels int8 [n, L, H],
    at float64 [n, L, H, len(positions)] weight on each slot position, row-sum deviation, raw attention)."""
    model_in = prepend_bos(tokens, cfg["bos_token_id"])
    attn = get_final_attention(model, model_in, cfg["batch_size"], cfg["query_position"])
    dev = check_rows_sum_to_one(attn, cfg["validation"]["row_sum_atol"])
    q = cfg["query_position"] % model_in.shape[1]
    metrics = {k: v.astype(np.float64) for k, v in compute_metrics(attn, q).items()}
    return metrics, classify(attn, cfg["diffuse_threshold"], q), attn[..., positions].astype(np.float64), dev, attn


def slot_nll(model, tokens: np.ndarray, positions: list[int], bos_id: int, batch_size: int) -> np.ndarray:
    """tokens WITHOUT BOS. -log p(token at each model position | everything before it), float64 [n, len(positions)].
    TransformerLens only: logits at input index p - 1 predict the token at p, and only those positions are unembedded."""
    import torch

    inputs = torch.as_tensor(prepend_bos(tokens, bos_id))
    pos = torch.as_tensor(positions)
    out = []
    with torch.inference_mode():
        for start in range(0, len(inputs), batch_size):
            batch = inputs[start : start + batch_size]
            _, cache = model.run_with_cache(batch, return_type=None, names_filter=lambda n: n == "ln_final.hook_normalized")
            logp = torch.log_softmax(model.unembed(cache["ln_final.hook_normalized"][:, pos - 1, :]).double(), dim=-1)
            out.append(-logp.gather(-1, batch[:, pos].unsqueeze(-1)).squeeze(-1).numpy())
    return np.concatenate(out)


def group_variants(cfg: dict[str, Any], phrase: str, placement: str) -> list[str]:
    return list(VARIANTS) + [v for v, spec in cfg.get("custom_variants", {}).items()
                             if spec["phrase"] == phrase and spec["placement"] == placement]


def additivity_metrics(cfg: dict[str, Any]) -> list[str]:
    ad = cfg.get("additivity")
    if not ad:
        return []
    return list(ad.get("metrics", [ad.get("metric", "sink_mass")]))


def assemble_cells(raw: dict[tuple, dict[str, Any]], cfg: dict[str, Any], positions: list[int]) -> tuple[dict, dict]:
    """Per (base, phrase, placement, variant): the metric dict incl. slot metrics for that placement, and labels.
    `none` rows are extracted once per base and reused for every phrase and placement."""
    eps = cfg["slot_logratio_eps"]
    keep = list(dict.fromkeys(cfg["metrics"] + additivity_metrics(cfg)))
    cells, labels = {}, {}
    for b in cfg["bases"]:
        for name in cfg["phrases"]:
            for placement, pos in cfg["placements"].items():
                cols = [positions.index(p) for p in pos]
                for v in group_variants(cfg, name, placement):
                    r = raw[(b, NONE)] if v == NONE else raw[(b, name, placement, v)]
                    m = dict(r["metrics"])
                    m["slot_mass"] = r["at"][..., cols].sum(axis=-1)
                    m["slot_logratio"] = np.log(np.maximum(m["slot_mass"], eps)) - np.log(np.maximum(m["sink_mass"], eps))
                    if "sink_logit" in keep:
                        p = np.clip(m["sink_mass"], cfg["sink_logit_eps"], 1 - cfg["sink_logit_eps"])
                        m["sink_logit"] = np.log(p) - np.log1p(-p)
                    key = (b, name, placement, v)
                    cells[key] = {k: m[k] for k in keep}
                    labels[key] = r["labels"]
    return cells, labels


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="exp2: a phrase written into random-token context, with controls")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text())
    t0 = time.time()

    run_dir = make_run_dir(PROJECT_ROOT / cfg["paths"]["results_dir"], tag="exp2")
    save_config_copy(args.config, run_dir)
    (run_dir / "PREDICTIONS.md").write_text((PROJECT_ROOT / cfg["predictions_file"]).read_text())
    save_versions(run_dir, PROJECT_ROOT, code_dirs=("exp1", "exp1ext", "exp2"))
    set_determinism(cfg["seed"])
    fig_dir = run_dir / "figures"
    fig_dir.mkdir()
    st = cfg["stats"]
    boot = dict(iters=st["bootstrap_iters"], seed=st["bootstrap_seed"], ci=st["bootstrap_ci"])
    log(f"run folder: {run_dir}", t0)

    tokenizer = AutoTokenizer.from_pretrained(cfg["figures"]["tokenizer"])
    bases, seqs, info = build_design(cfg, tokenizer)
    runs = {(b, NONE): arr for b, arr in bases.items()} | seqs
    np.savez_compressed(run_dir / "design.npz", **{"/".join(k): v for k, v in runs.items()})
    (run_dir / "sequences.json").write_text(json.dumps({"/".join(k): sequences_hash(v) for k, v in runs.items()}, indent=2) + "\n")
    log(f"design: {len(runs)} sequence sets x {cfg['n_bases']} rows; matched pool sizes {info['pool_sizes']}", t0)

    positions = sorted({p for pos in cfg["placements"].values() for p in pos})
    backends = [cfg["primary_backend"]] + [b for b in cfg["backends"] if b != cfg["primary_backend"]]
    raw: dict[str, dict[tuple, dict[str, Any]]] = {}
    checks: dict[str, Any] = {"row_sum_dev": {}, "determinism": {}}
    nll_raw: dict[tuple, np.ndarray] = {}  # [n, slots + query] per sequence set, TL
    for backend in backends:
        bcfg = cfg["backends"][backend]
        model = load_model(backend, bcfg["model_name"], cfg["device"], cfg["dtype"], bcfg.get("attn_implementation"))
        if bos_token_id(model) != cfg["bos_token_id"]:
            raise ValueError(f"{backend}: model BOS id differs from config")
        raw[backend] = {}
        dev = 0.0
        for i, (key, tokens) in enumerate(runs.items()):
            metrics, labs, at, d, attn = extract(model, tokens, cfg, positions)
            dev = max(dev, d)
            raw[backend][key] = {"metrics": metrics, "labels": labs, "at": at}
            if backend == "transformer_lens" and "surprisal" in cfg:
                sp = cfg["surprisal"]
                nll_raw[key] = slot_nll(model, tokens, sp["slot_positions"] + [cfg["seq_len"]], cfg["bos_token_id"], sp["batch_size"])
            if i == 0:
                v = cfg["validation"]
                checks["determinism"][backend] = check_determinism(model, prepend_bos(tokens, cfg["bos_token_id"]), attn,
                                                                   v["determinism_n"], cfg["batch_size"], cfg["query_position"])
        checks["row_sum_dev"][backend] = dev
        log(f"[{BACKEND_NAMES[backend]}] extracted {len(runs)} sets", t0)
        del model

    prim = cfg["primary_backend"]
    cells, labels = {}, {}
    for backend in backends:
        cells[backend], labels[backend] = assemble_cells(raw[backend], cfg, positions)

    # ---- analysis (primary backend) ----
    effects = contrast_table(cells[prim], cfg["contrasts"], cfg["metrics"], boot, cfg["min_effect"], st["fdr_q"])
    tv = tv_table(labels[prim], cfg["contrasts"])
    cells_df = cell_summary(cells[prim], labels[prim])
    tv_floor = {b: float(split_half(raw[prim][(b, NONE)]["labels"], st["bootstrap_seed"],
                                    lambda x, y: label_shift(x, y, len(LABELS)))["tv"].max()) for b in cfg["bases"]}
    extra: dict[str, pd.DataFrame] = {}
    if "additivity" in cfg:
        ad = cfg["additivity"]
        tables: dict[str, list[pd.DataFrame]] = {"additivity_heads": [], "additivity_depth": [], "order_fit": []}
        for metric in additivity_metrics(cfg):
            per_head, depth, order_fit = additivity_tables(cells[prim], ad["combos"], ad["differences"], cfg["depth_groups"],
                                                           metric, boot)
            per_head["q"] = bh_fdr(per_head["p"].to_numpy())  # over every interaction test of this metric
            for name, table in zip(tables, (per_head, depth, order_fit)):
                tables[name].append(table.assign(metric=metric))
        extra |= {name: pd.concat(ts, ignore_index=True) for name, ts in tables.items()}
    if "surprisal" in cfg:
        n_slots = len(cfg["surprisal"]["slot_positions"])
        nll = {}
        for key in cells[prim]:
            src = (key[0], NONE) if key[3] == NONE else key
            nll[key] = nll_raw[src][:, :n_slots].sum(axis=1)
        heads_s, depth_s, var_s = surprisal_slopes(cells[prim], nll, "sink_mass", cfg["depth_groups"], boot)
        var_s["mean_query_nll"] = [float(nll_raw[(r.base, NONE) if r.variant == NONE else (r.base, r.phrase, r.placement, r.variant)][:, n_slots].mean())
                                   for r in var_s.itertuples()]
        extra |= {"surprisal_heads": heads_s, "surprisal_depth": depth_s, "surprisal_variants": var_s}
    if len(cfg["phrases"]) > 1 and "additivity_depth" in extra:
        extra["phrase_pooled"] = phrase_level(extra["additivity_depth"], extra.get("surprisal_variants"), boot)
    predictions = [check_prediction(spec, effects, cells_df, extra) for spec in cfg["predictions"]]
    pooled = pool_over_phrases(effects) if len(cfg["phrases"]) > 1 else None
    log("analysis done", t0)

    # replication: the `none` rows against exp1's per-head mean sink_mass
    rep = {}
    for b in cfg["bases"]:
        spec = cfg["replication"][b]
        ref = np.load(PROJECT_ROOT / spec["run"] / f"{spec['condition']}_attn.npy")[..., 0].mean(axis=0)
        ours = raw[prim][(b, NONE)]["metrics"]["sink_mass"].mean(axis=0)
        rep[b] = float(np.corrcoef(ours.ravel(), ref.ravel())[0, 1])
    checks["replication"] = rep

    # cross-library: per-row metric agreement, label agreement, and agreement of `meaningful` decisions on sink_mass
    cross = {}
    for backend in backends[1:]:
        max_diff = {m: max(float(np.abs(cells[backend][k][m] - cells[prim][k][m]).max()) for k in cells[prim])
                    for m in cfg["metrics"] if m in PROB_METRICS}
        agree = np.mean([np.mean(labels[backend][k] == labels[prim][k]) for k in labels[prim]])
        other = contrast_table(cells[backend], cfg["contrasts"], ["sink_mass"], boot, cfg["min_effect"], st["fdr_q"])
        mine = effects[effects["metric"] == "sink_mass"].reset_index(drop=True)
        decisions = float((other["meaningful"].to_numpy() == mine["meaningful"].to_numpy()).mean())
        cross[backend] = {"max_abs_diff": max_diff, "label_agreement": float(agree), "meaningful_agreement": decisions,
                          "max_effect_size_diff": float(np.abs(other["effect_size"].to_numpy() - mine["effect_size"].to_numpy()).max())}
    checks["cross_backend"] = cross

    effects.to_parquet(run_dir / "effects.parquet", index=False)
    tv.to_parquet(run_dir / "label_tv.parquet", index=False)
    cells_df.to_parquet(run_dir / "cells.parquet", index=False)
    if pooled is not None:
        pooled.to_parquet(run_dir / "pooled_over_phrases.parquet", index=False)
    for name, table in extra.items():
        table.to_parquet(run_dir / f"{name}.parquet", index=False)
    (run_dir / "predictions.json").write_text(json.dumps(predictions, indent=2) + "\n")
    (run_dir / "checks.json").write_text(json.dumps(checks, indent=2, default=float) + "\n")

    # ---- figures ----
    heads = pd.read_csv(PROJECT_ROOT / cfg["tracked_heads_csv"])
    custom = set(cfg.get("custom_variants", {}))
    contrast_sets = {"contrasts": [c for c, (a, b) in cfg["contrasts"].items() if a not in custom and b not in custom],
                     "singles": [c for c, (a, b) in cfg["contrasts"].items() if a in custom or b in custom]}
    figures = []
    per_phrase = cfg["figures"].get("per_phrase", list(cfg["phrases"]))
    for b in cfg["bases"]:
        for name in per_phrase:
            for metric in HEATMAP_METRICS:
                for placement in cfg["placements"]:
                    sel = effects[(effects["base"] == b) & (effects["phrase"] == name) & (effects["placement"] == placement)
                                  & (effects["metric"] == metric)]
                    grid = lambda col, c: sel[sel["contrast"] == c].pivot(index="layer", columns="head", values=col).to_numpy()
                    for set_name, cs in contrast_sets.items():
                        cs = [c for c in cs if (sel["contrast"] == c).any()]
                        if not cs:
                            continue
                        path = fig_dir / f"{b}_{name}_{placement}_{metric}_{set_name}.png"
                        plot_diff_heatmaps({c: grid("mean", c) for c in cs},
                                           {c: grid("meaningful", c).astype(bool) for c in cs}, path,
                                           f"{b} bases, '{cfg['phrases'][name]['text']}' at {placement} {cfg['placements'][placement]}: Δ {metric}",
                                           f"Δ {metric}", sig_note=f"outlined = BH q <= {st['fdr_q']} and |Δ| >= {cfg['min_effect'][metric]}")
                        figures.append(path)
                path = fig_dir / f"{b}_{name}_{metric}_tracked_forest.png"
                plot_tracked_forest(effects, heads, b, name, metric, path)
                figures.append(path)
            path = fig_dir / f"{b}_{name}_sink_mass_layer_profile.png"
            plot_layer_profile(effects, b, name, "sink_mass", path)
            figures.append(path)
    if "additivity_heads" in extra:
        ah = extra["additivity_heads"]
        ah = ah[(ah["metric"] == "sink_mass") & ah["phrase"].isin(per_phrase)]
        for (b, name, placement), g in ah.groupby(["base", "phrase", "placement"]):
            panels, sig = {}, {}
            for combo, gc in g.groupby("combo", sort=False):
                for col, label in (("observed", f"{combo} − none"), ("additive", f"Σ parts ({combo})"), ("interaction", f"interaction ({combo})")):
                    panels[label] = gc.pivot(index="layer", columns="head", values=col).to_numpy()
                    sig[label] = (gc.pivot(index="layer", columns="head", values="q").to_numpy() <= st["fdr_q"]) if col == "interaction" \
                        else np.zeros_like(panels[label], dtype=bool)
            path = fig_dir / f"{b}_{name}_{placement}_additivity.png"
            plot_diff_heatmaps(panels, sig, path, f"{b} bases, '{cfg['phrases'][name]['text']}' at {placement}: observed vs additive Δ sink_mass",
                               "Δ sink_mass", sig_note=f"outlined = interaction BH q <= {st['fdr_q']}")
            figures.append(path)
        if len(cfg["phrases"]) > 1:
            for metric in additivity_metrics(cfg):
                path = fig_dir / f"phrase_terms_{metric}.png"
                plot_phrase_terms(extra["additivity_depth"], metric, "late", path, {n: ph["text"] for n, ph in cfg["phrases"].items()})
                figures.append(path)
    if "surprisal_variants" in extra:
        path = fig_dir / "surprisal_variants.png"
        sv = extra["surprisal_variants"]
        sd = extra["surprisal_depth"]
        plot_surprisal_variants(sv[sv["phrase"].isin(per_phrase)], sd[sd["phrase"].isin(per_phrase)], "sink_mass", path)
        figures.append(path)
    for metric in HEATMAP_METRICS:
        path = fig_dir / f"meaningful_counts_{metric}.png"
        plot_meaningful_counts(effects, metric, path)
        figures.append(path)
    log(f"{len(figures)} figures", t0)

    write_report(run_dir / "exp2_report.md", cfg, run_dir.name, {
        "checks": checks, "predictions": predictions, "effects": effects, "tv": tv, "tv_floor": tv_floor,
        "cells": cells_df, "heads": heads, "pooled": pooled, "info": info, "extra": extra,
        "figures": [p.relative_to(run_dir) for p in figures]})
    log(f"report: {run_dir / 'exp2_report.md'}", t0)
    return run_dir


if __name__ == "__main__":
    main()
