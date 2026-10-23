"""Experiment 3, class 2, R1-out: country (30) x capital or city (31) grids before the row's own random final token.

Run:  uv run --offline python -m exp3.run_relations --config configs/exp3_relations.yaml

Writes results/<timestamp>_exp3rel/: config, pre-registered predictions, versions, saved arrays, per-head and pooled
tables, checks, predictions verdicts, figures and exp3_relations_report.md. Attention comes from TransformerLens;
Hugging Face recomputes a subset on identical ids as a cross-library check.
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

from exp1.backends import check_rows_sum_to_one, get_final_attention, load_model, set_determinism
from exp1.runlog import make_run_dir, save_config_copy, save_versions
from exp1.sequences import prepend_bos
from exp1ext.effects import bootstrap_mean_per_unit
from exp2.design import cached_base_block
from exp3 import plots_relations as plots
from exp3 import relations as rel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARMS_SAVED = ("correct", "wrong")


def log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def stimuli(cfg: dict[str, Any], tokenizer) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(countries, capitals, cities) id arrays; raises unless each text tokenizes to exactly its id."""
    out = {k: [] for k in ("country", "capital", "city")}
    for s in cfg["stimuli"]:
        for k in out:
            ids = tokenizer.encode(s[k])
            if ids != [s[f"{k}_id"]]:
                raise ValueError(f"{s[k]!r} tokenizes to {ids}, config says {s[f'{k}_id']}")
            out[k].append(s[f"{k}_id"])
    return tuple(np.asarray(out[k], dtype=np.int64) for k in ("country", "capital", "city"))


def attention(model, seqs: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    attn = get_final_attention(model, prepend_bos(seqs, cfg["bos_token_id"]), cfg["batch_size"], cfg["query_position"])
    check_rows_sum_to_one(attn, cfg["checks"]["row_sum_atol"])
    return attn


def extract_class(model, bases: np.ndarray, stim: tuple, cfg: dict[str, Any], placebo: np.ndarray, t0: float,
                  base: str) -> dict[str, Any]:
    """All cells of every base row through TL. Keeps per-cell category masses, per-arm labels and per-arm mean
    attention by key position (summed over rows)."""
    countries, capitals, cities = stim
    n, cp, pp = len(countries), cfg["positions"]["country"], cfg["positions"]["partner"]
    nn, stim_ids = n * n, np.concatenate([countries, capitals, cities, cfg["demonstration_ids"]])
    rows = len(bases)
    masses = None
    labels: dict[str, np.ndarray] = {}
    by_position: dict[str, np.ndarray] = {}
    for r, row in enumerate(bases):
        seqs = rel.build_row(row, countries, capitals, cities, cp, pp)
        rel.check_row(seqs, row, countries, capitals, cities, cp, pp, stim_ids, cfg["bos_token_id"])
        attn = attention(model, seqs, cfg)
        m = rel.category_masses(attn, cp, pp).astype(np.float32)
        if masses is None:
            masses = np.empty((rows,) + m.shape, dtype=np.float32)
        masses[r] = m
        for g, grid in enumerate(rel.GRIDS):
            for arm, a in rel.arm_means(attn[g * nn:(g + 1) * nn], n, placebo).items():
                for kind, lab in zip(("argmax", "mass"), rel.arm_labels(a, cp, pp, cfg["diffuse_threshold"])):
                    labels.setdefault(f"{grid}/{arm}/{kind}", np.empty((rows,) + lab.shape, dtype=np.int8))[r] = lab
                if arm in ARMS_SAVED:
                    by_position[f"{grid}/{arm}"] = by_position.get(f"{grid}/{arm}", 0) + a / rows
        by_position["none"] = by_position.get("none", 0) + attn[-1] / rows
        if r % 25 == 0 or r == rows - 1:
            log(f"{base}: row {r + 1}/{rows}", t0)
    return {"masses": masses, "labels": labels, "by_position": by_position}


def manipulation(model, bases: np.ndarray, stim: tuple, cfg: dict[str, Any]) -> dict[str, Any]:
    """Next-token log-probs at the country position (30) over the 11 capitals and the 11 cities, in context."""
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
        lp = logp[:, :, cand]  # [rows, country, candidate]
        own = lp[:, np.arange(n), np.arange(n)]
        other = (lp.sum(-1) - own) / (n - 1)
        res[kind] = {"own_minus_other": (own - other).mean(0), "rank1": (lp.argmax(-1) == np.arange(n)).mean(0)}
    return res


def bare_association(model, stim: tuple, bos_id: int) -> dict[str, np.ndarray]:
    """log p(partner | BOS country) for every (country, partner) pair: association with no context."""
    countries, capitals, cities = stim
    x = torch.tensor([[bos_id, int(c)] for c in countries])
    with torch.inference_mode():
        logp = torch.log_softmax(model(x, return_type="logits")[:, 1, :].double(), -1).numpy()
    return {"capital": logp[:, capitals], "city": logp[:, cities]}


def cross_checks(tl, hf, bases: np.ndarray, stim: tuple, cfg: dict[str, Any], masses: np.ndarray) -> dict[str, Any]:
    """TL vs HF on the first hf_rows rows (max |delta attention|, argmax-key agreement); the first determinism_rows
    rows' TL masses recomputed must equal the stored ones exactly."""
    countries, capitals, cities = stim
    cp, pp = cfg["positions"]["country"], cfg["positions"]["partner"]
    max_delta, agree, det = 0.0, [], 0.0
    for r in range(cfg["checks"]["hf_rows"]):
        seqs = rel.build_row(bases[r], countries, capitals, cities, cp, pp)
        a_tl, a_hf = attention(tl, seqs, cfg), attention(hf, seqs, cfg)
        max_delta = max(max_delta, float(np.abs(a_tl - a_hf).max()))
        agree.append(float((a_tl.argmax(-1) == a_hf.argmax(-1)).mean()))
        if r < cfg["checks"]["determinism_rows"]:
            det = max(det, float(np.abs(rel.category_masses(a_tl, cp, pp).astype(np.float32) - masses[r]).max()))
    return {"hf_max_abs_delta": max_delta, "hf_argmax_agreement": float(np.mean(agree)), "determinism_max_abs_delta": det}


def late_grids(masses: np.ndarray, n: int, layers: list[int]) -> np.ndarray:
    """masses [rows, 2 n^2 + 1, L, H, 5] -> [rows, 2 (grid), n, n, 5]: mean over the late layers' heads."""
    late = masses[:, : 2 * n * n, layers[0]:layers[1] + 1].mean(axis=(2, 3))
    return late.reshape(len(masses), 2, n, n, -1)


def analyse(ext: dict[str, Any], n: int, cfg: dict[str, Any], perms: np.ndarray, placebo: np.ndarray,
            assoc: dict[str, np.ndarray]) -> dict[str, Any]:
    a = cfg["analysis"]
    boot, layers = a["bootstrap"], a["late_layers"]
    masses, labels = ext["masses"], ext["labels"]
    grids = late_grids(masses, n, layers)
    none_late = masses[:, -1, layers[0]:layers[1] + 1].mean(axis=(1, 2))  # [rows, 5]
    pooled, items, placebo_t, placebo_sig, heads, switches, extra = [], {}, {}, {}, {}, {}, {}
    t_bos = {}
    for g, grid in enumerate(rel.GRIDS):
        for k, cat in enumerate(rel.CATEGORIES):
            t_rows = rel.pairing_effect(grids[:, g, :, :, k])
            gm = grids[:, g, :, :, k].mean(axis=0)
            res = rel.verdict(t_rows, rel.permutation_p(gm, float(t_rows.mean()), perms), a["margin"], boot)
            pooled.append({"grid": grid, "metric": cat, **res})
            if cat == "BOS":
                t_bos[grid] = t_rows
                items[grid] = rel.item_residuals(gm)
                ts = [rel.pairing_effect(grids[:, g, :, :, k], d) for d in placebo]
                placebo_t[grid] = np.array([t.mean() for t in ts])
                placebo_sig[grid] = int(sum(rel.permutation_p(gm, float(t.mean()), perms)["p_two_sided"] < 0.05 for t in ts))
                extra[f"grid_mean/{grid}"] = gm
                # association among wrong cells only (correct cells are always the most expected)
                off = ~np.eye(n, dtype=bool)
                x = assoc[grid][off]
                slopes = []
                rng = np.random.default_rng(boot["seed"])
                for _ in range(boot["iters"]):
                    idx = rng.integers(0, len(grids), len(grids))
                    slopes.append(np.polyfit(x, grids[idx, g, :, :, k].mean(axis=0)[off], 1)[0])
                extra[f"assoc_slope/{grid}"] = {"slope": float(np.polyfit(x, gm[off], 1)[0]),
                                                "lo": float(np.percentile(slopes, 2.5)), "hi": float(np.percentile(slopes, 97.5))}
                diag = np.eye(n, dtype=bool)
                for arm, sel in (("correct", diag), ("wrong", off)):
                    d = grids[:, g][:, sel][..., k].mean(axis=1) - none_late[:, k]
                    m_, lo_, hi_ = bootstrap_mean_per_unit(d, boot["iters"], boot["seed"], 0.95)
                    extra[f"total/{grid}/{arm}"] = {"mean": float(m_), "lo": float(lo_), "hi": float(hi_)}
        # per-head fact effect for every category
        block = masses[:, g * n * n:(g + 1) * n * n].reshape((len(masses), n, n) + masses.shape[2:])
        t_heads = rel.pairing_effect(np.moveaxis(block, (1, 2), (-2, -1)))  # [rows, L, H, 5]
        for k, cat in enumerate(rel.CATEGORIES):
            heads[(grid, cat)] = rel.head_effects(t_heads[..., k], boot, a["meaningful"]["q"], a["meaningful"]["min_abs"])
        for kind in ("argmax", "mass"):
            sw = a["switch"]
            switches[(grid, kind)] = rel.role_switches(labels[f"{grid}/correct/{kind}"], labels[f"{grid}/wrong/{kind}"],
                                                       sw["q"], sw["min_shift"])
            counts = [len(rel.role_switches(labels[f"{grid}/p{i}/{kind}"], labels[f"{grid}/p{i}_rest/{kind}"],
                                            sw["q"], sw["min_shift"])) for i in range(len(placebo))]
            extra[f"placebo_switches/{grid}/{kind}"] = float(np.mean(counts))
    d = t_bos["capital"] - t_bos["city"]
    m_, lo_, hi_ = bootstrap_mean_per_unit(d, boot["iters"], boot["seed"], 0.95)
    extra["capital_minus_city"] = {"mean": float(m_), "lo": float(lo_), "hi": float(hi_)}
    return {"pooled": pd.DataFrame(pooled), "items": items, "placebo_t": placebo_t, "placebo_sig": placebo_sig,
            "heads": heads, "switches": switches, "extra": extra, "grids": grids}


def check_predictions(specs: list[dict], results: dict[str, dict], cfg: dict[str, Any]) -> list[dict]:
    out = []
    for s in specs:
        parts = []
        for base, res in results.items():
            if s["kind"] == "verdict":
                row = res["pooled"][(res["pooled"]["grid"] == s["grid"]) & (res["pooled"]["metric"] == s["metric"])].iloc[0]
                parts.append((row["verdict"] == s["expect"], f"{base}: {row['verdict']} {row['mean']:+.4f} "
                              f"[{row['lo']:+.4f}, {row['hi']:+.4f}], perm p {row['p_greater']:.4f}"))
            elif s["kind"] == "item_signs":
                r_ = res["items"][s["grid"]]
                k = int((r_ > 0).sum() if s["sign"] == "positive" else (r_ < 0).sum())
                parts.append((k >= s["min_items"], f"{base}: {k}/{len(r_)}"))
            elif s["kind"] == "placebo":
                for grid in rel.GRIDS:
                    sig = res["placebo_sig"][grid]
                    sw = res["extra"][f"placebo_switches/{grid}/argmax"]
                    parts.append((sig <= s["max_significant"] and sw <= s["max_mean_switches"],
                                  f"{base}/{grid}: {sig}/{cfg['analysis']['n_placebo']} p<0.05, mean switches {sw:.2f}"))
        ok = bool(parts) and all(p for p, _ in parts)
        out.append({"id": s["id"], "verdict": "PASS" if ok else "FAIL", "detail": "; ".join(d for _, d in parts)
                    or "no rows matched this check"})
    return out


def write_report(run: Path, cfg: dict, stim_text: dict, results: dict, checks: dict, preds: list[dict],
                 manip: dict, assoc: dict) -> None:
    n = len(stim_text["country"])
    L = [f"# Experiment 3, class 2, R1-out: country → capital / city before a random final token", "",
         f"Run `{run.name}`. TL attention (HF cross-check on {cfg['checks']['hf_rows']} rows per base class). "
         f"{cfg['n_bases']} rows per base class; {n} countries, so 2 × {n * n} cells + none per row. "
         "Layout: country at 30, capital or city at 31, the row's own random token at 32 (the query).", "",
         "T = mean(correct cells) − mean(wrong cells); every token sits once in a correct cell and "
         f"{n - 1} times in wrong ones, so token identity cancels. Late = layers "
         f"{cfg['analysis']['late_layers'][0]}–{cfg['analysis']['late_layers'][1]}, mean over their heads.", "",
         "## Checks", "", "| check | result |", "|---|---|"]
    for k, v in checks.items():
        L.append(f"| {k} | {v} |")
    L += ["", "## Manipulation check (next-token log p at the country, in context)", "",
          "| base | partner | own − mean(other 10), mean over countries | countries > 0 | own ranked 1st (mean over countries) |",
          "|---|---|---|---|---|"]
    for base, m in manip.items():
        for kind in ("capital", "city"):
            x = m[kind]
            L.append(f"| {base} | {kind} | {x['own_minus_other'].mean():+.2f} | {(x['own_minus_other'] > 0).sum()}/{n} | "
                     f"{x['rank1'].mean():.0%} |")
    L += ["", "## Pre-registered predictions", "", "| id | verdict | detail |", "|---|---|---|"]
    for p in preds:
        L.append(f"| {p['id']} | {p['verdict']} | {p['detail']} |")
    L += ["", "## Pooled late-layer T (95% row CI; permutation p over pairings; three-way verdict)", "",
          "| base | grid | metric | T | 95% CI | 90% CI | perm p (>) | perm p (<) | Wilcoxon p | verdict |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for base, res in results.items():
        for _, r in res["pooled"].iterrows():
            L.append(f"| {base} | {r['grid']} | {r['metric']} | {r['mean']:+.4f} | [{r['lo']:+.4f}, {r['hi']:+.4f}] | "
                     f"[{r['lo90']:+.4f}, {r['hi90']:+.4f}] | {r['p_greater']:.4f} | {r['p_less']:.4f} | "
                     f"{r['wilcoxon_p']:.2g} | {r['verdict']} |")
    L += ["", "## Per country (late BOS, diagonal residual after double-centring)", "",
          "| base | grid | " + " | ".join(c.strip() for c in stim_text["country"]) + " | > 0 |",
          "|---|---|" + "---|" * (n + 1)]
    for base, res in results.items():
        for grid in rel.GRIDS:
            r_ = res["items"][grid]
            L.append(f"| {base} | {grid} | " + " | ".join(f"{v * 1e3:+.1f}" for v in r_) + f" | {(r_ > 0).sum()}/{n} |")
    L += ["", "(× 10⁻³)", "", "## Placebo, capital − city, total effect, association", "",
          "| base | quantity | value |", "|---|---|---|"]
    for base, res in results.items():
        e = res["extra"]
        for grid in rel.GRIDS:
            pt = res["placebo_t"][grid]
            L.append(f"| {base} | placebo T ({grid}), 20 derangements | range [{pt.min():+.4f}, {pt.max():+.4f}], "
                     f"{res['placebo_sig'][grid]} with p < 0.05; mean switching heads {e[f'placebo_switches/{grid}/argmax']:.2f} "
                     f"(argmax), {e[f'placebo_switches/{grid}/mass']:.2f} (mass) |")
        c = e["capital_minus_city"]
        L.append(f"| {base} | T capital − T city (late BOS) | {c['mean']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}] |")
        for grid in rel.GRIDS:
            for arm in ("correct", "wrong"):
                t = e[f"total/{grid}/{arm}"]
                L.append(f"| {base} | {grid} {arm} − none (late BOS) | {t['mean']:+.4f} [{t['lo']:+.4f}, {t['hi']:+.4f}] |")
            s = e[f"assoc_slope/{grid}"]
            L.append(f"| {base} | slope of late BOS on log p({grid} given country), wrong cells | "
                     f"{s['slope']:+.5f} [{s['lo']:+.5f}, {s['hi']:+.5f}] per nat |")
    L += ["", "## Heads", "", "| base | grid | metric | meaningful heads (q ≤ 0.05, |T| ≥ 0.02) | top 5 by |T| |",
          "|---|---|---|---|---|"]
    for base, res in results.items():
        for (grid, cat), h in res["heads"].items():
            order = np.argsort(-np.abs(h["mean"]).ravel())[:5]
            top = ", ".join(f"L{i // 12}H{i % 12} {h['mean'].ravel()[i]:+.3f}" for i in order)
            L.append(f"| {base} | {grid} | {cat} | {int(h['meaningful'].sum())} | {top} |")
    L += ["", "## Role switches, correct vs wrong (McNemar, BH, ≥ 15-point shift)", ""]
    for base, res in results.items():
        for (grid, kind), sw in res["switches"].items():
            desc = ", ".join(f"L{s['layer']}H{s['head']} {s['from']}→{s['to']} ({s['shift']:+.2f})" for s in sw) or "none"
            L.append(f"- {base}, {grid}, {kind} labels: {len(sw)} heads; {desc}")
    L += ["", "## Figures", "", "- `figures/head_T_BOS.png`: T per head, BOS mass.",
          "- `figures/pooled_T_BOS.png`: late-layer T with CIs, placebo derangements and the equivalence band.",
          "- `figures/residual_grids_BOS.png`: 11 × 11 residual grids of late BOS (correct pairs on the diagonal).", ""]
    (run / "exp3_relations_report.md").write_text("\n".join(L))


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Experiment 3, class 2, R1-out")
    parser.add_argument("--config", default="configs/exp3_relations.yaml", type=Path)
    args = parser.parse_args(argv)
    cfg_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    cfg = yaml.safe_load(cfg_path.read_text())
    t0 = time.time()
    run = make_run_dir(PROJECT_ROOT / cfg["paths"]["results_dir"], tag="exp3rel")
    save_config_copy(cfg_path, run)
    shutil.copy2(PROJECT_ROOT / "exp3" / "PREDICTIONS_relations.md", run / "PREDICTIONS.md")
    save_versions(run, PROJECT_ROOT, code_dirs=("exp1", "exp2", "exp3"))
    (run / "figures").mkdir()
    log(f"run folder {run}", t0)

    set_determinism(cfg["seed"])
    tl = load_model("transformer_lens", cfg["backends"]["transformer_lens"]["model_name"], cfg["device"], cfg["dtype"])
    from transformers import AutoTokenizer

    stim = stimuli(cfg, AutoTokenizer.from_pretrained(cfg["figures"]["tokenizer"]))  # TL's tokenizer prepends BOS
    n = len(stim[0])
    stim_ids = np.concatenate([*stim, cfg["demonstration_ids"]])
    a = cfg["analysis"]
    perms = rel.random_permutations(n, a["n_permutations"], a["permutation_seed"])
    placebo = rel.random_derangements(n, a["n_placebo"], a["placebo_seed"])
    assoc = bare_association(tl, stim, cfg["bos_token_id"])

    bases, ext, manip = {}, {}, {}
    for base, rng_ in cfg["bases"].items():
        bases[base] = cached_base_block(f"exp3rel_{base}", rng_, stim_ids, cfg["n_bases"], cfg["seq_len"], cfg["seed"],
                                        cfg["bos_token_id"], PROJECT_ROOT / cfg["paths"]["sequences_dir"])
        ext[base] = extract_class(tl, bases[base], stim, cfg, placebo, t0, base)
        manip[base] = manipulation(tl, bases[base], stim, cfg)
        log(f"{base}: extraction and manipulation check done", t0)

    hf = load_model("huggingface", cfg["backends"]["huggingface"]["model_name"], cfg["device"], cfg["dtype"],
                    cfg["backends"]["huggingface"]["attn_implementation"])
    cross = {base: cross_checks(tl, hf, bases[base], stim, cfg, ext[base]["masses"]) for base in bases}
    log("cross-library and determinism checks done", t0)

    results = {base: analyse(ext[base], n, cfg, perms, placebo, assoc) for base in bases}
    log("analysis done", t0)

    min_items = 9
    checks = {
        "design (ids at 30/31, base elsewhere, no repeats/BOS/stimulus ids in bases)": "PASS (asserted per row)",
        "attention rows sum to 1": f"PASS (atol {cfg['checks']['row_sum_atol']:g}, asserted per batch)",
    }
    for base, c in cross.items():
        checks[f"determinism, {base} ({cfg['checks']['determinism_rows']} rows)"] = (
            f"{'PASS' if c['determinism_max_abs_delta'] == 0 else 'FAIL'} (max |Δ| {c['determinism_max_abs_delta']:.1e})")
        checks[f"TL vs HF, {base} ({cfg['checks']['hf_rows']} rows)"] = (
            f"{'PASS' if c['hf_max_abs_delta'] < cfg['checks']['hf_atol'] else 'FAIL'} (max |Δ| {c['hf_max_abs_delta']:.1e}, "
            f"argmax-key agreement {c['hf_argmax_agreement']:.4%})")
        k = int((manip[base]["capital"]["own_minus_other"] > 0).sum())
        checks[f"manipulation, {base}"] = f"{'PASS' if k >= min_items else 'FAIL'} ({k}/{n} countries prefer their capital)"
    preds = check_predictions(cfg["predictions"], results, cfg)

    # saved arrays and tables
    np.savez_compressed(run / "design.npz", countries=stim[0], capitals=stim[1], cities=stim[2], permutations=perms,
                        placebo=placebo, **{f"bases/{b}": v for b, v in bases.items()})
    np.savez_compressed(run / "late_grids.npz", **{f"{b}": r["grids"] for b, r in results.items()})
    np.savez_compressed(run / "cell_means.npz", **{b: e["masses"][:, : 2 * n * n].mean(axis=0) for b, e in ext.items()},
                        **{f"none/{b}": e["masses"][:, -1] for b, e in ext.items()})
    np.savez_compressed(run / "labels.npz", **{f"{b}/{k}": v for b, e in ext.items() for k, v in e["labels"].items()})
    np.savez_compressed(run / "attention_by_position.npz",
                        **{f"{b}/{k}": v for b, e in ext.items() for k, v in e["by_position"].items()})
    pd.concat([r["pooled"].assign(base=b) for b, r in results.items()]).to_csv(run / "pooled.csv", index=False)
    head_rows = []
    for b, r in results.items():
        for (grid, cat), h in r["heads"].items():
            for layer in range(h["mean"].shape[0]):
                for head in range(h["mean"].shape[1]):
                    head_rows.append({"base": b, "grid": grid, "metric": cat, "layer": layer, "head": head,
                                      **{k: float(h[k][layer, head]) for k in ("mean", "lo", "hi", "p", "q")},
                                      "meaningful": bool(h["meaningful"][layer, head])})
    pd.DataFrame(head_rows).to_parquet(run / "heads.parquet", index=False)
    (run / "checks.json").write_text(json.dumps(checks, indent=2))
    (run / "predictions.json").write_text(json.dumps(preds, indent=2))
    (run / "manipulation.json").write_text(json.dumps(
        {b: {k: {kk: vv.tolist() for kk, vv in v.items()} for k, v in m.items()} for b, m in manip.items()}, indent=2))

    # figures
    stim_text = {k: [s[k] for s in cfg["stimuli"]] for k in ("country", "capital", "city")}
    plots.plot_head_maps({(b, g): r["heads"][(g, "BOS")] for b, r in results.items() for g in rel.GRIDS},
                         run / "figures" / "head_T_BOS.png")
    plots.plot_pooled({(b, r_["grid"]): r_ for b, r in results.items()
                       for _, r_ in r["pooled"][r["pooled"]["metric"] == "BOS"].iterrows()},
                      {(b, g): r["placebo_t"][g] for b, r in results.items() for g in rel.GRIDS}, a["margin"],
                      run / "figures" / "pooled_T_BOS.png")
    plots.plot_residual_grids({(b, g): r["extra"][f"grid_mean/{g}"] for b, r in results.items() for g in rel.GRIDS},
                              stim_text["country"], {"capital": stim_text["capital"], "city": stim_text["city"]},
                              run / "figures" / "residual_grids_BOS.png")
    write_report(run, cfg, stim_text, results, checks, preds, manip, assoc)
    log(f"done: {run / 'exp3_relations_report.md'}", t0)
    for p in preds:
        print(f"{p['id']}: {p['verdict']}  {p['detail']}")
    return run


if __name__ == "__main__":
    main()
