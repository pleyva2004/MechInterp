"""Experiment 3, class 2, R1-in: country (31) x capital or city AT the final query (32), plus anchor and capital-alone
cells.

Run:  uv run --offline python -m exp3.run_relations_in --config configs/exp3_relations_in.yaml

Same design, checks and analysis as exp3.run_relations (R1-out), with the predictions of
exp3/PREDICTIONS_relations_in.md. The query token is edited, but every grid contrast holds it fixed on average (each
capital sits once in a correct cell and 10 times in wrong ones).
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
import torch
import yaml
from scipy.stats import rankdata

from exp1.runlog import make_run_dir, save_config_copy, save_versions
from exp1.sequences import prepend_bos
from exp1ext.effects import bootstrap_mean_per_unit
from exp1ext.features import token_features
from exp2.design import cached_base_block, derive_seed, draw_matched, matched_pools
from exp1.backends import load_model, set_determinism
from exp3 import plots_relations as plots
from exp3 import relations as rel
from exp3.run_relations import PROJECT_ROOT, analyse, attention, bare_association, log, stimuli

EXTRA = ("anchor", "capital_alone")


def anchor_pools(cfg: dict[str, Any], tokenizer, stim: tuple, stim_ids: np.ndarray) -> list[np.ndarray]:
    """Per capital: ids of its class with the same match_features, minus every stimulus and demonstration id."""
    lo = min(r[0] for r in cfg["token_classes"].values())
    hi = max(r[1] for r in cfg["token_classes"].values())
    features = token_features(np.arange(lo, hi + 1), tokenizer)
    pools = matched_pools(stim[1], features, cfg["match_features"], cfg["token_classes"])
    return [p[~np.isin(p, stim_ids)] for p in pools]


def extra_writes(row_anchor: np.ndarray, stim: tuple, cp: int, pp: int) -> list[list[tuple[int, int]]]:
    """Anchor cells (country c at cp, its anchor word at pp), then capital-alone cells (capital k at pp)."""
    countries, capitals, _ = stim
    anchor = [[(cp, int(c)), (pp, int(a))] for c, a in zip(countries, row_anchor)]
    alone = [[(pp, int(k))] for k in capitals]
    return anchor + alone


def row_seqs(row: np.ndarray, row_anchor: np.ndarray, stim: tuple, cfg: dict[str, Any], stim_ids: np.ndarray) -> np.ndarray:
    countries, capitals, cities = stim
    cp, pp = cfg["positions"]["country"], cfg["positions"]["partner"]
    grid = rel.build_row(row, countries, capitals, cities, cp, pp)
    rel.check_row(grid, row, countries, capitals, cities, cp, pp, stim_ids, cfg["bos_token_id"])
    writes = extra_writes(row_anchor, stim, cp, pp)
    extra = rel.build_cells(row, writes)
    rel.check_cells(extra, row, writes, cfg["bos_token_id"])
    return np.concatenate([grid, extra])


def extract_class(model, bases: np.ndarray, anchors: np.ndarray, stim: tuple, cfg: dict[str, Any],
                  placebo: np.ndarray, stim_ids: np.ndarray, t0: float, base: str) -> dict[str, Any]:
    n, cp, pp = len(stim[0]), cfg["positions"]["country"], cfg["positions"]["partner"]
    nn, rows = n * n, len(bases)
    masses, labels, by_position = None, {}, {}
    for r, row in enumerate(bases):
        attn = attention(model, row_seqs(row, anchors[r], stim, cfg, stim_ids), cfg)
        m = rel.category_masses(attn, cp, pp).astype(np.float32)
        if masses is None:
            masses = np.empty((rows,) + m.shape, dtype=np.float32)
        masses[r] = m
        for g, grid in enumerate(rel.GRIDS):
            for arm, a in rel.arm_means(attn[g * nn:(g + 1) * nn], n, placebo).items():
                for kind, lab in zip(("argmax", "mass"), rel.arm_labels(a, cp, pp, cfg["diffuse_threshold"])):
                    labels.setdefault(f"{grid}/{arm}/{kind}", np.empty((rows,) + lab.shape, dtype=np.int8))[r] = lab
                if arm in ("correct", "wrong"):
                    by_position[f"{grid}/{arm}"] = by_position.get(f"{grid}/{arm}", 0) + a / rows
        by_position["none"] = by_position.get("none", 0) + attn[2 * nn] / rows
        by_position["anchor"] = by_position.get("anchor", 0) + attn[2 * nn + 1:2 * nn + 1 + n].mean(0) / rows
        by_position["capital_alone"] = by_position.get("capital_alone", 0) + attn[2 * nn + 1 + n:].mean(0) / rows
        if r % 25 == 0 or r == rows - 1:
            log(f"{base}: row {r + 1}/{rows}", t0)
    return {"masses": masses, "labels": labels, "by_position": by_position}


def manipulation(model, bases: np.ndarray, stim: tuple, cfg: dict[str, Any]) -> dict[str, Any]:
    """Next-token log-probs at the country (31), i.e. for the query position, over the capitals and the cities.
    Also returns each row's surprisal of each country's own capital (for the within-cell dose, Rin-7)."""
    countries, capitals, cities = stim
    n, cp = len(countries), cfg["positions"]["country"]
    seqs = np.repeat(bases, n, axis=0)
    seqs[:, cp - 1] = np.tile(countries, len(bases))
    x = torch.as_tensor(prepend_bos(seqs, cfg["bos_token_id"]))
    out = []
    with torch.inference_mode():
        for s in range(0, len(x), cfg["batch_size"]):
            out.append(torch.log_softmax(model(x[s:s + cfg["batch_size"]], return_type="logits")[:, cp, :].double(), -1))
    logp = torch.cat(out).numpy().reshape(len(bases), n, -1)
    res = {}
    for kind, cand in (("capital", capitals), ("city", cities)):
        lp = logp[:, :, cand]
        own = lp[:, np.arange(n), np.arange(n)]
        other = (lp.sum(-1) - own) / (n - 1)
        res[kind] = {"own_minus_other": (own - other).mean(0), "rank1": (lp.argmax(-1) == np.arange(n)).mean(0)}
    res["surprisal_own_capital"] = -logp[:, np.arange(n), capitals]  # [rows, n]
    return res


def cross_checks(tl, hf, bases, anchors, stim, cfg, masses, stim_ids) -> dict[str, Any]:
    cp, pp = cfg["positions"]["country"], cfg["positions"]["partner"]
    max_delta, agree, det = 0.0, [], 0.0
    for r in range(cfg["checks"]["hf_rows"]):
        seqs = row_seqs(bases[r], anchors[r], stim, cfg, stim_ids)
        a_tl, a_hf = attention(tl, seqs, cfg), attention(hf, seqs, cfg)
        max_delta = max(max_delta, float(np.abs(a_tl - a_hf).max()))
        agree.append(float((a_tl.argmax(-1) == a_hf.argmax(-1)).mean()))
        if r < cfg["checks"]["determinism_rows"]:
            det = max(det, float(np.abs(rel.category_masses(a_tl, cp, pp).astype(np.float32) - masses[r]).max()))
    return {"hf_max_abs_delta": max_delta, "hf_argmax_agreement": float(np.mean(agree)), "determinism_max_abs_delta": det}


def row_verdict(d: np.ndarray, margin: float, boot: dict[str, Any]) -> dict[str, Any]:
    """Three-way rule on a per-row difference with the row CI only (no pairing permutation applies)."""
    mean, lo, hi = (float(v) for v in bootstrap_mean_per_unit(d, boot["iters"], boot["seed"], 0.95))
    _, lo90, hi90 = (float(v) for v in bootstrap_mean_per_unit(d, boot["iters"], boot["seed"], 0.90))
    v = ("positive" if lo > 0 else "negative" if hi < 0 else "equivalent" if -margin < lo90 and hi90 < margin
         else "inconclusive")
    return {"mean": mean, "lo": lo, "hi": hi, "lo90": lo90, "hi90": hi90, "verdict": v}


def spearman_rows(x: np.ndarray, y: np.ndarray) -> float:
    """Mean over columns of the Spearman correlation between x[:, j] and y[:, j] (rows are observations)."""
    rx, ry = rankdata(x, axis=0), rankdata(y, axis=0)
    rx, ry = rx - rx.mean(0), ry - ry.mean(0)
    with np.errstate(invalid="ignore", divide="ignore"):  # a constant column (tiny bootstrap draws) gives NaN, skipped
        return float(np.nanmean((rx * ry).sum(0) / np.sqrt((rx ** 2).sum(0) * (ry ** 2).sum(0))))


def analyse_in(ext: dict[str, Any], n: int, cfg: dict[str, Any], perms: np.ndarray, placebo: np.ndarray,
               assoc: dict[str, np.ndarray], surprisal: np.ndarray) -> dict[str, Any]:
    a, boot = cfg["analysis"], cfg["analysis"]["bootstrap"]
    nn, layers = n * n, a["late_layers"]
    res = analyse({"masses": ext["masses"][:, :2 * nn + 1], "labels": ext["labels"]}, n, cfg, perms, placebo, assoc)
    grids = res["grids"]  # [rows, 2, n, n, 5]
    late = ext["masses"][:, :, layers[0]:layers[1] + 1].mean(axis=(2, 3))  # [rows, cells, 5]
    bos = rel.CATEGORIES.index("BOS")
    # Rin-4: the same relabelling applied to both grids -> permutation null for T_cap - T_city
    diff = grids[:, 0, :, :, bos] - grids[:, 1, :, :, bos]
    t = rel.pairing_effect(diff)
    res["capital_minus_city"] = rel.verdict(t, rel.permutation_p(diff.mean(0), float(t.mean()), perms), a["margin"], boot)
    # Rin-7: within each correct cell, across rows
    correct_bos = grids[:, 0, np.arange(n), np.arange(n), bos]  # [rows, n]
    rho = spearman_rows(surprisal, correct_bos)
    rng = np.random.default_rng(boot["seed"])
    draws = [spearman_rows(surprisal[idx], correct_bos[idx])
             for idx in (rng.integers(0, len(grids), len(grids)) for _ in range(boot["iters"]))]
    res["dose"] = {"rho": rho, "lo": float(np.nanpercentile(draws, 2.5)), "hi": float(np.nanpercentile(draws, 97.5))}
    # Rin-8 (descriptive): against the anchor (country at 31 + a same-form random word at 32)
    anchor = late[:, 2 * nn + 1:2 * nn + 1 + n, bos].mean(1)
    alone = late[:, 2 * nn + 1 + n:, bos].mean(1)
    diag = np.eye(n, dtype=bool)
    correct = grids[:, 0][:, diag][..., bos].mean(1)
    wrong = grids[:, 0][:, ~diag][..., bos].mean(1)
    res["anchor"] = {name: row_verdict(d, a["margin"], boot) for name, d in
                     (("correct − anchor", correct - anchor), ("wrong − anchor", wrong - anchor),
                      ("capital alone − anchor", alone - anchor))}
    return res


def check_predictions(specs: list[dict], results: dict[str, dict], cfg: dict[str, Any]) -> list[dict]:
    m = cfg["analysis"]["margin"]
    out = []
    for s in specs:
        parts = []
        for base, res in results.items():
            pooled = res["pooled"]
            row = lambda grid, metric: pooled[(pooled["grid"] == grid) & (pooled["metric"] == metric)].iloc[0]
            k = s["kind"]
            if k == "verdict":
                r_ = row(s["grid"], s["metric"])
                parts.append((r_["verdict"] == s["expect"], f"{base}: {r_['verdict']} {r_['mean']:+.4f} "
                              f"[{r_['lo']:+.4f}, {r_['hi']:+.4f}], perm p< {r_['p_less']:.4f}"))
            elif k == "country_absorbs":
                b, c = row("capital", "BOS"), row("capital", "country")
                q_, f_ = row("capital", "partner"), row("capital", "filler")
                ok = c["lo"] > 0 and c["mean"] >= s["min_share"] * abs(b["mean"]) and abs(q_["mean"]) < m and abs(f_["mean"]) < m
                parts.append((ok, f"{base}: country {c['mean']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}] vs |BOS| "
                              f"{abs(b['mean']):.4f}; query {q_['mean']:+.4f}, filler {f_['mean']:+.4f}"))
            elif k == "switches":
                sw = res["switches"][(s["grid"], "argmax")]
                hits = sum(1 for x in sw if x["from"] == s["from"] and x["to"] == s["to"])
                base_rate = res["extra"][f"placebo_switches/{s['grid']}/argmax"]
                parts.append((hits - base_rate >= s["min_beyond_placebo"],
                              f"{base}: {hits} {s['from']}→{s['to']} switches, placebo mean {base_rate:.2f}"))
            elif k == "capital_minus_city":
                r_ = res["capital_minus_city"]
                parts.append((r_["verdict"] == s["expect"], f"{base}: {r_['verdict']} {r_['mean']:+.4f} "
                              f"[{r_['lo']:+.4f}, {r_['hi']:+.4f}], perm p< {r_['p_less']:.4f}"))
            elif k == "assoc_slope":
                r_ = res["extra"][f"assoc_slope/{s['grid']}"]
                ok = r_["hi"] < 0 if s["sign"] == "negative" else r_["lo"] > 0
                parts.append((ok, f"{base}: {r_['slope']:+.5f} [{r_['lo']:+.5f}, {r_['hi']:+.5f}] per nat"))
            elif k == "item_signs":
                r_ = res["items"][s["grid"]]
                cnt = int((r_ < 0).sum() if s["sign"] == "negative" else (r_ > 0).sum())
                parts.append((cnt >= s["min_items"], f"{base}: {cnt}/{len(r_)}"))
            elif k == "dose":
                r_ = res["dose"]
                ok = r_["lo"] > 0 if s["sign"] == "positive" else r_["hi"] < 0
                parts.append((ok, f"{base}: rho {r_['rho']:+.3f} [{r_['lo']:+.3f}, {r_['hi']:+.3f}]"))
        ok = bool(parts) and all(p for p, _ in parts)
        out.append({"id": s["id"], "verdict": "PASS" if ok else "FAIL",
                    "detail": "; ".join(d for _, d in parts) or "no rows matched this check"})
    return out


def write_report(run: Path, cfg: dict, stim_text: dict, results: dict, checks: dict, preds: list[dict], manip: dict) -> None:
    n = len(stim_text["country"])
    L = ["# Experiment 3, class 2, R1-in: country → capital / city AT the final token", "",
         f"Run `{run.name}`. TL attention (HF cross-check on {cfg['checks']['hf_rows']} rows per base class). "
         f"{cfg['n_bases']} rows per base class; per row 2 × {n * n} grid cells, none, {n} anchor and {n} capital-alone "
         "cells. Layout: country at 31, capital / city / anchor word at 32 (the query).", "",
         "T = mean(correct cells) − mean(wrong cells): each capital (the query token) sits once in a correct cell and "
         f"{n - 1} times in wrong ones, so its own query effect cancels. Late = layers "
         f"{cfg['analysis']['late_layers'][0]}–{cfg['analysis']['late_layers'][1]}. Categories: 'country' = 31 "
         "(previous token), 'partner' = 32 (the query itself), 'self' is empty.", "",
         "## Checks", "", "| check | result |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in checks.items()]
    L += ["", "## Manipulation check (log p at the country for the query position, in context)", "",
          "| base | partner | own − mean(other 10) | countries > 0 | own ranked 1st |", "|---|---|---|---|---|"]
    for base, mm in manip.items():
        for kind in ("capital", "city"):
            x = mm[kind]
            L.append(f"| {base} | {kind} | {x['own_minus_other'].mean():+.2f} | {(x['own_minus_other'] > 0).sum()}/{n} | "
                     f"{x['rank1'].mean():.0%} |")
    L += ["", "## Pre-registered predictions", "", "| id | verdict | detail |", "|---|---|---|"]
    L += [f"| {p['id']} | {p['verdict']} | {p['detail']} |" for p in preds]
    L += ["", "## Pooled late-layer T", "",
          "| base | grid | metric | T | 95% CI | 90% CI | perm p (>) | perm p (<) | verdict |", "|---|---|---|---|---|---|---|---|---|"]
    for base, res in results.items():
        for _, r in res["pooled"].iterrows():
            L.append(f"| {base} | {r['grid']} | {r['metric']} | {r['mean']:+.4f} | [{r['lo']:+.4f}, {r['hi']:+.4f}] | "
                     f"[{r['lo90']:+.4f}, {r['hi90']:+.4f}] | {r['p_greater']:.4f} | {r['p_less']:.4f} | {r['verdict']} |")
    L += ["", "## Capital − city, dose, anchor (late BOS)", "", "| base | quantity | value | verdict |", "|---|---|---|---|"]
    for base, res in results.items():
        c = res["capital_minus_city"]
        L.append(f"| {base} | T capital − T city | {c['mean']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}], perm p< {c['p_less']:.4f} | {c['verdict']} |")
        d = res["dose"]
        L.append(f"| {base} | Spearman(surprisal of own capital, late BOS) within correct cells | {d['rho']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}] | |")
        for name, v in res["anchor"].items():
            L.append(f"| {base} | {name} | {v['mean']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}] | {v['verdict']} |")
        for grid in rel.GRIDS:
            pt = res["placebo_t"][grid]
            L.append(f"| {base} | placebo T ({grid}), 20 derangements | [{pt.min():+.4f}, {pt.max():+.4f}], "
                     f"{res['placebo_sig'][grid]} with p < 0.05; mean switches {res['extra'][f'placebo_switches/{grid}/argmax']:.2f} | |")
            s_ = res["extra"][f"assoc_slope/{grid}"]
            L.append(f"| {base} | slope of late BOS on log p({grid} given country), wrong cells | "
                     f"{s_['slope']:+.5f} [{s_['lo']:+.5f}, {s_['hi']:+.5f}] per nat | |")
    L += ["", "## Per country (late BOS, diagonal residual, × 10⁻³)", "",
          "| base | grid | " + " | ".join(c.strip() for c in stim_text["country"]) + " | < 0 |", "|---|---|" + "---|" * (n + 1)]
    for base, res in results.items():
        for grid in rel.GRIDS:
            r_ = res["items"][grid]
            L.append(f"| {base} | {grid} | " + " | ".join(f"{v * 1e3:+.1f}" for v in r_) + f" | {(r_ < 0).sum()}/{n} |")
    L += ["", "## Heads", "", "| base | grid | metric | meaningful | top 5 by |T| |", "|---|---|---|---|---|"]
    for base, res in results.items():
        for (grid, cat), h in res["heads"].items():
            order = np.argsort(-np.abs(h["mean"]).ravel())[:5]
            top = ", ".join(f"L{i // 12}H{i % 12} {h['mean'].ravel()[i]:+.3f}" for i in order)
            L.append(f"| {base} | {grid} | {cat} | {int(h['meaningful'].sum())} | {top} |")
    L += ["", "## Role switches, correct vs wrong", ""]
    for base, res in results.items():
        for (grid, kind), sw in res["switches"].items():
            desc = ", ".join(f"L{s['layer']}H{s['head']} {s['from']}→{s['to']} ({s['shift']:+.2f})" for s in sw) or "none"
            L.append(f"- {base}, {grid}, {kind} labels: {len(sw)} heads; {desc}")
    L += ["", "## Figures", "", "- `figures/head_T_BOS.png`, `figures/pooled_T_BOS.png`, `figures/residual_grids_BOS.png`", ""]
    (run / "exp3_relations_in_report.md").write_text("\n".join(L))


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Experiment 3, class 2, R1-in")
    parser.add_argument("--config", default="configs/exp3_relations_in.yaml", type=Path)
    args = parser.parse_args(argv)
    cfg_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    cfg = yaml.safe_load(cfg_path.read_text())
    t0 = time.time()
    run = make_run_dir(PROJECT_ROOT / cfg["paths"]["results_dir"], tag="exp3rel_in")
    save_config_copy(cfg_path, run)
    shutil.copy2(PROJECT_ROOT / "exp3" / "PREDICTIONS_relations_in.md", run / "PREDICTIONS.md")
    save_versions(run, PROJECT_ROOT, code_dirs=("exp1", "exp2", "exp3"))
    (run / "figures").mkdir()
    log(f"run folder {run}", t0)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg["figures"]["tokenizer"])  # display / features only
    set_determinism(cfg["seed"])
    tl = load_model("transformer_lens", cfg["backends"]["transformer_lens"]["model_name"], cfg["device"], cfg["dtype"])
    stim = stimuli(cfg, tokenizer)
    n = len(stim[0])
    stim_ids = np.concatenate([*stim, cfg["demonstration_ids"]])
    a = cfg["analysis"]
    perms = rel.random_permutations(n, a["n_permutations"], a["permutation_seed"])
    placebo = rel.random_derangements(n, a["n_placebo"], a["placebo_seed"])
    assoc = bare_association(tl, stim, cfg["bos_token_id"])
    pools = anchor_pools(cfg, tokenizer, stim, stim_ids)
    log(f"anchor pool sizes {[len(p) for p in pools]}", t0)

    bases, anchors, ext, manip = {}, {}, {}, {}
    query_slot = np.array([cfg["positions"]["partner"] - 1])
    for base, rng_ in cfg["bases"].items():
        bases[base] = cached_base_block(f"exp3rel_{base}", rng_, stim_ids, cfg["n_bases"], cfg["seq_len"], cfg["seed"],
                                        cfg["bos_token_id"], PROJECT_ROOT / cfg["paths"]["sequences_dir"])
        anchors[base] = np.concatenate([draw_matched(bases[base], query_slot, [pools[c]],
                                                     derive_seed(cfg["seed"], f"{base}/anchor/{c}")) for c in range(n)], axis=1)
        ext[base] = extract_class(tl, bases[base], anchors[base], stim, cfg, placebo, stim_ids, t0, base)
        manip[base] = manipulation(tl, bases[base], stim, cfg)
        log(f"{base}: extraction and manipulation check done", t0)

    hf = load_model("huggingface", cfg["backends"]["huggingface"]["model_name"], cfg["device"], cfg["dtype"],
                    cfg["backends"]["huggingface"]["attn_implementation"])
    cross = {b: cross_checks(tl, hf, bases[b], anchors[b], stim, cfg, ext[b]["masses"], stim_ids) for b in bases}
    log("cross-library and determinism checks done", t0)

    results = {b: analyse_in(ext[b], n, cfg, perms, placebo, assoc, manip[b]["surprisal_own_capital"]) for b in bases}
    log("analysis done", t0)

    checks = {"design (grid and extra cells; no repeats/BOS/stimulus ids in bases)": "PASS (asserted per row)",
              "attention rows sum to 1": f"PASS (atol {cfg['checks']['row_sum_atol']:g}, asserted per batch)"}
    for b, c in cross.items():
        checks[f"determinism, {b} ({cfg['checks']['determinism_rows']} rows)"] = (
            f"{'PASS' if c['determinism_max_abs_delta'] == 0 else 'FAIL'} (max |Δ| {c['determinism_max_abs_delta']:.1e})")
        checks[f"TL vs HF, {b} ({cfg['checks']['hf_rows']} rows)"] = (
            f"{'PASS' if c['hf_max_abs_delta'] < cfg['checks']['hf_atol'] else 'FAIL'} (max |Δ| {c['hf_max_abs_delta']:.1e}, "
            f"argmax-key agreement {c['hf_argmax_agreement']:.4%})")
        k = int((manip[b]["capital"]["own_minus_other"] > 0).sum())
        checks[f"manipulation, {b}"] = f"{'PASS' if k >= 9 else 'FAIL'} ({k}/{n} countries prefer their capital)"
    preds = check_predictions(cfg["predictions"], results, cfg)

    np.savez_compressed(run / "design.npz", countries=stim[0], capitals=stim[1], cities=stim[2], permutations=perms,
                        placebo=placebo, **{f"bases/{b}": v for b, v in bases.items()},
                        **{f"anchors/{b}": v for b, v in anchors.items()})
    np.savez_compressed(run / "late_grids.npz", **{b: r["grids"] for b, r in results.items()})
    np.savez_compressed(run / "cell_means.npz", **{b: e["masses"].mean(axis=0) for b, e in ext.items()})
    np.savez_compressed(run / "labels.npz", **{f"{b}/{k}": v for b, e in ext.items() for k, v in e["labels"].items()})
    np.savez_compressed(run / "attention_by_position.npz",
                        **{f"{b}/{k}": v for b, e in ext.items() for k, v in e["by_position"].items()})
    pd.concat([r["pooled"].assign(base=b) for b, r in results.items()]).to_csv(run / "pooled.csv", index=False)
    head_rows = [{"base": b, "grid": g, "metric": c, "layer": la, "head": he,
                  **{k: float(h[k][la, he]) for k in ("mean", "lo", "hi", "p", "q")}, "meaningful": bool(h["meaningful"][la, he])}
                 for b, r in results.items() for (g, c), h in r["heads"].items()
                 for la in range(h["mean"].shape[0]) for he in range(h["mean"].shape[1])]
    pd.DataFrame(head_rows).to_parquet(run / "heads.parquet", index=False)
    (run / "checks.json").write_text(json.dumps(checks, indent=2))
    (run / "predictions.json").write_text(json.dumps(preds, indent=2))

    stim_text = {k: [s[k] for s in cfg["stimuli"]] for k in ("country", "capital", "city")}
    plots.plot_head_maps({(b, g): r["heads"][(g, "BOS")] for b, r in results.items() for g in rel.GRIDS},
                         run / "figures" / "head_T_BOS.png", title="Correct vs wrong pairing, capital / city AT the final token")
    plots.plot_pooled({(b, r_["grid"]): r_ for b, r in results.items()
                       for _, r_ in r["pooled"][r["pooled"]["metric"] == "BOS"].iterrows()},
                      {(b, g): r["placebo_t"][g] for b, r in results.items() for g in rel.GRIDS}, a["margin"],
                      run / "figures" / "pooled_T_BOS.png")
    plots.plot_residual_grids({(b, g): r["extra"][f"grid_mean/{g}"] for b, r in results.items() for g in rel.GRIDS},
                              stim_text["country"], {"capital": stim_text["capital"], "city": stim_text["city"]},
                              run / "figures" / "residual_grids_BOS.png", positions=(31, 32))
    write_report(run, cfg, stim_text, results, checks, preds, manip)
    log(f"done: {run / 'exp3_relations_in_report.md'}", t0)
    for p in preds:
        print(f"{p['id']}: {p['verdict']}  {p['detail']}")
    return run


if __name__ == "__main__":
    main()
