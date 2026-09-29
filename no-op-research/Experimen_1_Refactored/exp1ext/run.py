"""exp1ext entrypoint.   Run:  uv run python -m exp1ext.run --config configs/exp1ext.yaml

Stage 0: per-token views of the exp1 arrays (attention overlays, attention received per token id).
Stage 1: 2x2 factorial (context class x final-token class) and a dose-response, TL and HF.
Stage 2: per-token sweeps: final (query) token, and substitutions at earlier positions; TL, plus an HF subset.
Stage 3: token features vs per-token effects. Everything lands in results/<YYYYMMDD-HHMMSS>_exp1ext/.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from transformers import AutoTokenizer

from exp1.backends import bos_token_id, get_final_attention, load_model, set_determinism
from exp1.compare import label_shift, split_half
from exp1.metrics import LABELS, classify, compute_metrics
from exp1.runlog import make_run_dir, save_config_copy, save_versions
from exp1.sequences import prepend_bos, sequences_hash
from exp1.validation import check_determinism
from exp1ext.design import CELLS, build_dose_response, build_factorial, cached_block, check_design, position_sweep
from exp1ext.effects import (attention_received, bootstrap_mean_per_unit, decompose_2x2, per_token_effects,
                             split_reliability)
from exp1ext.extract import run_metrics
from exp1ext.features import fit_feature_model, token_features
from exp1ext.plots_cells import plot_decomposition, plot_dose_response, plot_interaction_grid
from exp1ext.plots_tokens import (plot_attention_overlay, plot_attention_received, plot_token_effects,
                                  plot_token_head_heatmap)
from exp1ext.report import write_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_NAMES = {"transformer_lens": "TL", "huggingface": "HF"}
EFFECT_METRICS = ("sink_mass", "entropy", "prev_mass")  # sink_mass is primary throughout
BYTE_TOKENS = 256  # GPT-2's byte-level BPE: ids 0-255 are the raw bytes


def head_name(head: tuple[int, int]) -> str:
    return f"L{head[0]}H{head[1]}"


def log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def token_class(ids: np.ndarray, rare_lo: int) -> np.ndarray:
    return np.where(ids < BYTE_TOKENS, "byte", np.where(ids < rare_lo, "common", "rare"))


def load_exp1(cfg: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """exp1 primary-backend attention and model input per condition, hash-checked against the exp1 sequence cache."""
    hs = cfg["head_selection"]
    attn, tokens = {}, {}
    for cond, run in hs["source_runs"].items():
        seq_dir = PROJECT_ROOT / hs["exp1_sequences_dir"]
        seqs = np.load(seq_dir / f"{cond}.npy")
        if sequences_hash(seqs) != json.loads((seq_dir / f"{cond}.json").read_text())["sha256"]:
            raise ValueError(f"exp1 sequences for {cond} do not match their recorded hash")
        tokens[cond] = prepend_bos(seqs, cfg["bos_token_id"])
        attn[cond] = np.load(PROJECT_ROOT / run / f"{cond}_attn.npy")
        if attn[cond].shape[0] != len(seqs):
            raise ValueError(f"{cond}: attention and sequences disagree in length")
    return attn, tokens


def select_heads(cfg: dict[str, Any], attn: dict[str, np.ndarray]) -> pd.DataFrame:
    """Tracked heads = label-mix TV >= min_tv between the reference and any compared exp1 condition, plus controls."""
    hs = cfg["head_selection"]
    thr, q = cfg["diffuse_threshold"], cfg["query_position"] % attn[hs["reference"]].shape[-1]
    labels = {c: classify(a, thr, q) for c, a in attn.items()}
    sink = {c: a[..., 0].mean(axis=0) for c, a in attn.items()}
    tv = {c: label_shift(labels[hs["reference"]], labels[c], len(LABELS))["tv"] for c in hs["compare_to"]}
    over = np.stack([tv[c] >= hs["min_tv"] for c in hs["compare_to"]])
    sink_change = np.mean([sink[c] - sink[hs["reference"]] for c in hs["compare_to"]], axis=0)
    rows = []
    def group(change: float) -> str:
        if abs(change) < hs["min_sink_change"]:
            return "label mix only"
        return "sink rises" if change > 0 else "sink falls"

    for layer, head in np.argwhere(over.any(axis=0)):
        rows.append({"layer": int(layer), "head": int(head), "group": group(sink_change[layer, head]),
                     "both": bool(over[:, layer, head].all()), "sink_change": float(sink_change[layer, head]),
                     **{f"tv_{c}": float(tv[c][layer, head]) for c in hs["compare_to"]}})
    for layer, head in hs["controls"]:
        rows.append({"layer": layer, "head": head, "group": "control", "both": False,
                     "sink_change": float(sink_change[layer, head]), **{f"tv_{c}": float(tv[c][layer, head]) for c in hs["compare_to"]}})
    order = {"sink rises": 0, "sink falls": 1, "label mix only": 2, "control": 3}
    df = pd.DataFrame(rows)
    df = df.sort_values(by=["group", "both", "layer", "head"], key=lambda s: s.map(order) if s.name == "group" else s,
                        ascending=[True, False, True, True]).reset_index(drop=True)
    df.insert(0, "name", [head_name((r.layer, r.head)) for r in df.itertuples()])
    return df


def stage0(cfg, exp1_attn, exp1_tokens, heads, tokenizer, fig_dir, t0) -> dict[str, Any]:
    s0 = cfg["stage0"]
    idx = list(zip(heads["layer"], heads["head"]))
    over_dir, recv_dir = fig_dir / "overlay", fig_dir / "received"
    over_dir.mkdir(parents=True)
    recv_dir.mkdir(parents=True)
    n_ex = s0["overlay_examples"]
    for name, (layer, head) in zip(heads["name"], idx):
        toks = np.concatenate([exp1_tokens[c][:n_ex] for c in s0["overlay_conditions"]])
        weights = np.concatenate([exp1_attn[c][:n_ex, layer, head, :] for c in s0["overlay_conditions"]])
        row_labels = [f"{c} #{i}" for c in s0["overlay_conditions"] for i in range(n_ex)]
        plot_attention_overlay(toks, weights, row_labels, tokenizer, over_dir / f"{name}.png",
                               f"{name}: attention from the final token (exp1 sequences)", cfg["bos_token_id"])
    received = []
    for cond in s0["received_conditions"]:
        key_len = exp1_tokens[cond].shape[1]
        rows = np.stack([exp1_attn[cond][:, l, h, :] for l, h in idx], axis=1)
        r = attention_received(exp1_tokens[cond], rows, np.arange(1, key_len - 1))
        texts = [tokenizer.decode([int(i)]) for i in r["ids"]]
        for j, name in enumerate(heads["name"]):
            plot_attention_received(r["ids"], texts, r["mean_attn"][:, j], r["count"], recv_dir / f"{name}_{cond}.png",
                                    f"{name}: attention received per key token, {cond}", "mean attention from the final token",
                                    s0["received_top_k"])
            received.append(pd.DataFrame({"condition": cond, "head": name, "token_id": r["ids"], "text": texts,
                                          "count": r["count"], "mean_attn": r["mean_attn"][:, j], "mean_share": r["mean_share"][:, j],
                                          "mean_logratio": r["mean_logratio"][:, j]}))
    log("stage 0 done", t0)
    return {"received": pd.concat(received, ignore_index=True)}


def stage1_inputs(cfg) -> tuple[dict[str, np.ndarray], dict[int, np.ndarray]]:
    r, c, sd = cfg["ranges"]["rare"], cfg["ranges"]["common"], PROJECT_ROOT / cfg["paths"]["sequences_dir"]
    n, L, bos, seed = cfg["factorial"]["n_per_cell"], cfg["seq_len"], cfg["bos_token_id"], cfg["seed"]
    cells = build_factorial(cached_block("factorial_rare", r, n, L, seed, bos, sd), cached_block("factorial_common", c, n, L, seed, bos, sd))
    nd = cfg["dose_response"]["n"]
    dose = build_dose_response(cached_block("dose_rare", r, nd, L, seed, bos, sd), cached_block("dose_replacements", c, nd, L - 1, seed, bos, sd),
                               cfg["dose_response"]["ks"], seed)
    for seqs in [*cells.values(), *dose.values()]:
        check_design(seqs, [r, c], bos)
    return cells, dose


def stage2_inputs(cfg) -> dict[str, Any]:
    r, c, sd = cfg["ranges"]["rare"], cfg["ranges"]["common"], PROJECT_ROOT / cfg["paths"]["sequences_dir"]
    K, L, bos, seed, sw = cfg["sweep"]["n_bases_per_class"], cfg["seq_len"], cfg["bos_token_id"], cfg["seed"], cfg["sweep"]
    bases = np.concatenate([cached_block("sweep_bases_rare", r, K, L, seed, bos, sd), cached_block("sweep_bases_common", c, K, L, seed, bos, sd)])
    common_tokens = np.arange(sw["common_tokens"][0], sw["common_tokens"][1] + 1)
    rare_tokens = np.sort(cached_block("sweep_rare_tokens", r, 1, sw["n_rare_tokens"], seed, bos, sd)[0])
    return {"bases": bases, "K": K, "common_tokens": common_tokens, "rare_tokens": rare_tokens,
            "tokens": np.concatenate([common_tokens, rare_tokens])}


def sweep_specs(cfg, s2) -> list[dict[str, Any]]:
    """One spec per (position, base group): which bases, which tokens, which array index gets replaced."""
    K, L = s2["K"], cfg["seq_len"]
    rare_b, common_b = np.arange(K), np.arange(K, 2 * K)
    specs = [{"position": L, "kind": "query", "group": "rare contexts", "bases": rare_b, "tokens": s2["tokens"]},
             {"position": L, "kind": "query", "group": "common contexts", "bases": common_b, "tokens": s2["tokens"]}]
    for p in cfg["sweep"]["substitution_positions"]:
        specs.append({"position": p, "kind": "substitution", "group": "rare contexts", "bases": rare_b, "tokens": s2["common_tokens"]})
        specs.append({"position": p, "kind": "substitution", "group": "common contexts", "bases": common_b, "tokens": s2["rare_tokens"]})
    return specs


def run_sweep(model, cfg, s2, spec, base_metrics, bases_subset=None) -> dict[str, Any]:
    """Metrics [n_bases, T, L, H] (NaN where the token would repeat) for one spec; Δ vs the base for substitutions."""
    v = cfg["validation"]
    bases_idx = spec["bases"] if bases_subset is None else spec["bases"][bases_subset]
    seqs, invalid = position_sweep(s2["bases"][bases_idx], spec["tokens"], spec["position"] - 1)
    key_pos = [spec["position"]] if spec["kind"] == "substitution" else None
    out = run_metrics(model, prepend_bos(seqs, cfg["bos_token_id"]), cfg["batch_size"], cfg["chunk_size"], cfg["query_position"],
                      cfg["diffuse_threshold"], v["row_sum_atol"], key_positions=key_pos)
    shape = (len(bases_idx), len(spec["tokens"]))
    mask = invalid[..., None, None]
    metrics = {m: np.where(mask, np.nan, x.reshape(*shape, *x.shape[1:])) for m, x in out["metrics"].items()}
    res = {"metrics": metrics, "invalid": invalid, "bases_idx": bases_idx, "row_sum_dev": out["row_sum_dev"]}
    if key_pos:
        res["at"] = np.where(mask, np.nan, out["at"][:, 0].reshape(*shape, *out["at"].shape[2:]))
    res["delta"] = {m: x - base_metrics[m][bases_idx][:, None] for m, x in metrics.items()}
    return res


def mean_log_prior(model, bases: np.ndarray, tokens: np.ndarray, bos_id: int, batch_size: int) -> np.ndarray:
    """The model's mean log p(token | base context) at the final position, averaged over bases: [len(tokens)].

    Logits at input index L-1 (the second-to-last random token) predict the final token at index L,
    i.e. exactly the slot the query sweep fills. TransformerLens only: the final normalized residual is
    unembedded at that one position, so full-vocabulary logits are never built for every position.
    """
    import torch

    inputs = torch.as_tensor(prepend_bos(bases, bos_id))
    total = torch.zeros(len(tokens), dtype=torch.float64)
    with torch.inference_mode():
        for start in range(0, len(inputs), batch_size):
            _, cache = model.run_with_cache(inputs[start : start + batch_size], return_type=None,
                                            names_filter=lambda name: name == "ln_final.hook_normalized")
            logits = model.unembed(cache["ln_final.hook_normalized"][:, -2:-1, :])[:, 0]
            total += torch.log_softmax(logits.double(), dim=-1)[:, torch.as_tensor(tokens)].sum(0)
    return (total / len(inputs)).numpy()


def heldout_top_tokens(values: np.ndarray, top_k: int, seed: int) -> dict[str, np.ndarray]:
    """Winner's-curse check. values [K, T, n_heads]: pick the top_k / bottom_k tokens on one random half of
    the bases and measure them on the other half. Returns per-head gap (top mean - bottom mean) in each half."""
    K = values.shape[0]
    perm = np.random.default_rng(seed).permutation(K)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        h1, h2 = np.nanmean(values[perm[: K // 2]], axis=0), np.nanmean(values[perm[K // 2 :]], axis=0)
    gap1, gap2 = np.full(values.shape[2], np.nan), np.full(values.shape[2], np.nan)
    for j in range(values.shape[2]):
        ok = np.isfinite(h1[:, j]) & np.isfinite(h2[:, j])
        if ok.sum() < 2 * top_k:
            continue
        order = np.argsort(h1[ok, j])
        bottom, top = order[:top_k], order[-top_k:]
        gap1[j] = h1[ok, j][top].mean() - h1[ok, j][bottom].mean()
        gap2[j] = h2[ok, j][top].mean() - h2[ok, j][bottom].mean()
    return {"selection_half": gap1, "heldout_half": gap2}


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="exp1ext: which tokens cause the rare-vs-common head changes")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text())
    t0 = time.time()

    run_dir = make_run_dir(PROJECT_ROOT / cfg["paths"]["results_dir"], tag="exp1ext")
    save_config_copy(args.config, run_dir)
    (run_dir / "PREDICTIONS.md").write_text((PROJECT_ROOT / "exp1ext" / "PREDICTIONS.md").read_text())
    save_versions(run_dir, PROJECT_ROOT, code_dirs=("exp1", "exp1ext"))
    set_determinism(cfg["seed"])
    fig_dir = run_dir / "figures"
    st = cfg["stats"]
    boot = dict(iters=st["bootstrap_iters"], seed=st["bootstrap_seed"], ci=st["bootstrap_ci"])
    log(f"run folder: {run_dir}", t0)

    tokenizer = AutoTokenizer.from_pretrained(cfg["figures"]["tokenizer"])
    exp1_attn, exp1_tokens = load_exp1(cfg)
    heads = select_heads(cfg, exp1_attn)
    heads.to_csv(run_dir / "tracked_heads.csv", index=False)
    idx = list(zip(heads["layer"], heads["head"]))
    li, hi_ = np.array([i[0] for i in idx]), np.array([i[1] for i in idx])
    names, groups = list(heads["name"]), list(heads["group"])
    log(f"tracked {len(heads)} heads ({int(heads['both'].sum())} in both comparisons)", t0)

    s0 = stage0(cfg, exp1_attn, exp1_tokens, heads, tokenizer, fig_dir, t0)
    s0["received"].to_parquet(run_dir / "stage0_attention_received.parquet", index=False)

    cells, dose = stage1_inputs(cfg)
    s2 = stage2_inputs(cfg)
    specs = sweep_specs(cfg, s2)
    (run_dir / "sequences.json").write_text(json.dumps(
        {**{f"factorial_{k}": sequences_hash(v) for k, v in cells.items()},
         **{f"dose_{k}": sequences_hash(v) for k, v in dose.items()}, "sweep_bases": sequences_hash(s2["bases"]),
         "sweep_tokens": sequences_hash(s2["tokens"][None])}, indent=2) + "\n")

    backends = [cfg["primary_backend"]] + [b for b in cfg["backends"] if b != cfg["primary_backend"]]
    s1: dict[str, dict[str, Any]] = {}
    sweeps: dict[str, list[dict[str, Any]]] = {}
    checks: dict[str, Any] = {"row_sum_dev": {}, "determinism": {}}
    n = cfg["factorial"]["n_per_cell"]
    for backend in backends:
        bcfg = cfg["backends"][backend]
        model = load_model(backend, bcfg["model_name"], cfg["device"], cfg["dtype"], bcfg.get("attn_implementation"))
        if bos_token_id(model) != cfg["bos_token_id"]:
            raise ValueError(f"{backend}: model BOS id differs from config")
        v = cfg["validation"]
        all_seqs = prepend_bos(np.concatenate([*[cells[c] for c in CELLS], *dose.values()]), cfg["bos_token_id"])
        out = run_metrics(model, all_seqs, cfg["batch_size"], cfg["chunk_size"], cfg["query_position"], cfg["diffuse_threshold"],
                          v["row_sum_atol"], keep_heads=idx)
        first = get_final_attention(model, all_seqs[: v["determinism_n"]], cfg["batch_size"], cfg["query_position"])
        det_equal, det_diff = check_determinism(model, all_seqs, first, v["determinism_n"], cfg["batch_size"], cfg["query_position"])
        checks["determinism"][backend] = (det_equal, det_diff)
        checks["row_sum_dev"][backend] = out["row_sum_dev"]
        split = {c: slice(i * n, (i + 1) * n) for i, c in enumerate(CELLS)}
        nd = cfg["dose_response"]["n"]
        dsplit = {k: slice(4 * n + i * nd, 4 * n + (i + 1) * nd) for i, k in enumerate(dose)}
        s1[backend] = {"cells": {c: {m: x[s] for m, x in out["metrics"].items()} for c, s in split.items()},
                       "labels": {c: out["labels"][s] for c, s in split.items()},
                       "dose": {k: {m: x[s] for m, x in out["metrics"].items()} for k, s in dsplit.items()},
                       "rows": out["rows"]}
        log(f"[{BACKEND_NAMES[backend]}] stage 1: {len(all_seqs)} sequences", t0)

        # Stage 2: bases first (the reference for substitution deltas), then every spec.
        base_out = run_metrics(model, prepend_bos(s2["bases"], cfg["bos_token_id"]), cfg["batch_size"], cfg["chunk_size"],
                               cfg["query_position"], cfg["diffuse_threshold"], v["row_sum_atol"])
        if backend == "transformer_lens":
            log_prior = mean_log_prior(model, s2["bases"], s2["tokens"], cfg["bos_token_id"], cfg["batch_size"])
        subset = None
        if backend != cfg["primary_backend"]:
            subset = np.arange(cfg["sweep"]["hf_bases_per_class"])
        sweeps[backend] = []
        for spec in specs:
            res = run_sweep(model, cfg, s2, spec, base_out["metrics"], subset)
            checks["row_sum_dev"][backend] = max(checks["row_sum_dev"][backend], res["row_sum_dev"])
            if backend == cfg["primary_backend"]:
                # Keep all-head per-token means; keep per-base values only for tracked heads to bound memory.
                res["all_mean"] = {m: np.nanmean(x, axis=0) for m, x in res["metrics"].items()} if spec["kind"] == "query" else \
                    {m: np.nanmean(x, axis=0) for m, x in res["delta"].items()}
            for key in ("metrics", "delta"):
                res[key] = {m: x[:, :, li, hi_] for m, x in res[key].items()}
            if "at" in res:
                res["at"] = res["at"][:, :, li, hi_]
            sweeps[backend].append(res)
            log(f"[{BACKEND_NAMES[backend]}] sweep p={spec['position']} {spec['kind']} {spec['group']}: "
                f"{res['invalid'].size} sequences ({int(res['invalid'].sum())} masked)", t0)
        del model

    prim, other = cfg["primary_backend"], [b for b in backends if b != cfg["primary_backend"]]

    # ---- Stage 1 analysis (primary backend) ----
    c1 = s1[prim]
    decomp = {m: decompose_2x2({c: c1["cells"][c][m] for c in CELLS}, **boot) for m in EFFECT_METRICS}
    tv_cells = {pair: label_shift(c1["labels"][pair[0]], c1["labels"][pair[1]], len(LABELS))["tv"]
                for pair in [("AA", "AC"), ("AA", "CA"), ("AA", "CC")]}
    tv_floor = max(float(split_half(c1["labels"][c], st["bootstrap_seed"], lambda a, b: label_shift(a, b, len(LABELS)))["tv"].max())
                   for c in ("AA", "CC"))
    rep = {}
    for cell, cond in (("AA", "A_rare"), ("CC", "C_common_clean")):
        rep[cell] = float(np.corrcoef(c1["cells"][cell]["sink_mass"].mean(0).ravel(), exp1_attn[cond][..., 0].mean(0).ravel())[0, 1])
    dose_curves = {m: {k: bootstrap_mean_per_unit(d[m][:, li, hi_], **boot) for k, d in c1["dose"].items()} for m in EFFECT_METRICS}

    decomp_rows = []
    for m, eff in decomp.items():
        for key, val in eff.items():
            for layer in range(12):
                for head in range(12):
                    decomp_rows.append({"metric": m, "effect": key, "layer": layer, "head": head,
                                        "est": val["est"][layer, head], "lo": val["lo"][layer, head], "hi": val["hi"][layer, head]})
    pd.DataFrame(decomp_rows).to_parquet(run_dir / "stage1_decomposition.parquet", index=False)
    pd.DataFrame([{"metric": m, "k": k, "head": names[j], "mean": c[0][j], "lo": c[1][j], "hi": c[2][j]}
                  for m, curves in dose_curves.items() for k, c in curves.items() for j in range(len(names))]
                 ).to_parquet(run_dir / "stage1_dose_response.parquet", index=False)
    np.savez_compressed(run_dir / "stage1_tracked_rows.npz", rows=c1["rows"], heads=np.array(names),
                        cells=np.array([c for c in CELLS for _ in range(n)] + [f"dose_k{k}" for k in dose for _ in range(cfg["dose_response"]["n"])]))

    sel = lambda a: a[li, hi_]
    for m, ylabel in (("sink_mass", "mean sink_mass"), ("entropy", "mean entropy (nats)"), ("prev_mass", "mean prev_mass")):
        plot_interaction_grid({c: tuple(sel(decomp[m][f"cell_{c}"][k]) for k in ("est", "lo", "hi")) for c in CELLS}, names, groups,
                              fig_dir / f"stage1_interaction_{m}.png", f"2×2: {m} by context and final-token class [TL]", ylabel)
        plot_decomposition({e: tuple(sel(decomp[m][e][k]) for k in ("est", "lo", "hi")) for e in ("total", "query", "context", "interaction")},
                           names, groups, fig_dir / f"stage1_decomposition_{m}.png",
                           f"Where the rare→common change in {m} comes from [TL]", f"Δ {m}")
        curves = {names[j]: tuple(np.array([dose_curves[m][k][i][j] for k in dose]) for i in range(3)) for j in range(len(names))}
        plot_dose_response(np.array(list(dose)), curves, groups, fig_dir / f"stage1_dose_{m}.png",
                           f"{m} as common tokens replace k of 31 rare context tokens (final token rare) [TL]", ylabel)
    log("stage 1 analysis done", t0)

    # ---- Stage 2 + 3 analysis (primary backend) ----
    feats = token_features(s2["tokens"], tokenizer)
    feats["log_prior"] = log_prior
    feats.to_parquet(run_dir / "stage3_token_features.parquet", index=False)
    classes = token_class(s2["tokens"], cfg["ranges"]["rare"][0])
    texts = list(feats["text"])
    tok_pos = {int(t): i for i, t in enumerate(s2["tokens"])}
    effect_rows, rel_rows, model_rows, heldout_rows = [], [], [], []
    panels: dict[tuple[int, str], dict[str, dict[str, np.ndarray]]] = {}
    all_means = {}
    for spec, res in zip(specs, sweeps[prim]):
        values = res["metrics"] if spec["kind"] == "query" else res["delta"]
        cols = np.array([tok_pos[int(t)] for t in spec["tokens"]])
        all_means[(spec["position"], spec["group"])] = res["all_mean"]
        for m in EFFECT_METRICS:
            e = per_token_effects(values[m], **boot)
            r = split_reliability(values[m], cfg["sweep"]["reliability_seed"])
            full = {k: np.full((len(s2["tokens"]), len(names)), np.nan) for k in ("mean", "lo", "hi")}
            for k in full:
                full[k][cols] = e[k]
            panels.setdefault((spec["position"], m), {})[spec["group"]] = full
            for j, name in enumerate(names):
                rel_rows.append({"position": spec["position"], "kind": spec["kind"], "group": spec["group"], "metric": m,
                                 "head": name, "reliability_r": float(r[j])})
                effect_rows.append(pd.DataFrame({"position": spec["position"], "kind": spec["kind"], "group": spec["group"],
                                                 "metric": m, "head": name, "token_id": spec["tokens"], "text": [texts[c] for c in cols],
                                                 "class": classes[cols], "mean": e["mean"][:, j], "lo": e["lo"][:, j],
                                                 "hi": e["hi"][:, j], "n": e["n"][:, j]}))
                if m == "sink_mass":
                    with warnings.catch_warnings():  # constant columns (e.g. is_byte among rare tokens) are dropped by design
                        warnings.simplefilter("ignore", UserWarning)
                        fit = fit_feature_model(e["mean"][:, j], feats.iloc[cols].reset_index(drop=True), cfg["features"],
                                                st["bootstrap_ci"])
                    model_rows.append(fit.assign(position=spec["position"], kind=spec["kind"], group=spec["group"], head=name))
            if m == "sink_mass":
                ho = heldout_top_tokens(values[m], cfg["sweep"]["top_k"], cfg["sweep"]["reliability_seed"])
                heldout_rows += [{"position": spec["position"], "kind": spec["kind"], "group": spec["group"], "head": name,
                                  "selection_gap": ho["selection_half"][j], "heldout_gap": ho["heldout_half"][j]}
                                 for j, name in enumerate(names)]
            if "at" in res and m == "sink_mass":
                e_at = per_token_effects(res["at"], **boot)
                for j, name in enumerate(names):
                    effect_rows.append(pd.DataFrame({"position": spec["position"], "kind": spec["kind"], "group": spec["group"],
                                                     "metric": "attn_on_substituted", "head": name, "token_id": spec["tokens"],
                                                     "text": [texts[c] for c in cols], "class": classes[cols], "mean": e_at["mean"][:, j],
                                                     "lo": e_at["lo"][:, j], "hi": e_at["hi"][:, j], "n": e_at["n"][:, j]}))
    effects = pd.concat(effect_rows, ignore_index=True)
    effects.to_parquet(run_dir / "stage2_token_effects.parquet", index=False)
    reliability = pd.DataFrame(rel_rows)
    reliability.to_parquet(run_dir / "stage2_reliability.parquet", index=False)
    feature_models = pd.concat(model_rows, ignore_index=True)
    feature_models.to_parquet(run_dir / "stage3_feature_models.parquet", index=False)
    heldout = pd.DataFrame(heldout_rows)
    heldout.to_parquet(run_dir / "stage2_heldout_top_tokens.parquet", index=False)
    np.savez_compressed(run_dir / "stage2_all_heads_token_means.npz", tokens=s2["tokens"],
                        **{f"p{p}_{g.split()[0]}_{m}": x for (p, g), means in all_means.items() for m, x in means.items()})

    tok_dir = fig_dir / "tokens"
    tok_dir.mkdir()
    for (p, m), by_group in panels.items():
        kind = "query" if p == cfg["seq_len"] else "substitution"
        for j, name in enumerate(names):
            pan = {f"{g}" + ("" if kind == "query" else (": common tokens in" if g.startswith("rare") else ": rare tokens in")):
                   (d["mean"][:, j], d["lo"][:, j], d["hi"][:, j]) for g, d in by_group.items()}
            ylabel = m if kind == "query" else f"Δ {m} vs unsubstituted"
            title = (f"{name}: {m} as a function of the final token" if kind == "query"
                     else f"{name}: Δ {m} when the token at position {p} is replaced")
            plot_token_effects(s2["tokens"], texts, list(classes), pan, tok_dir / f"{name}_p{p}_{m}.png", title, ylabel,
                               cfg["sweep"]["top_k"], zero_line=kind != "query")
    # Token x head heatmaps of centred sink_mass effects (tokens ranked by their largest effect on any tracked head).
    for p in [cfg["seq_len"], *cfg["sweep"]["substitution_positions"]]:
        g = "rare contexts"
        mean = panels[(p, "sink_mass")][g]["mean"]
        centred = mean - np.nanmean(mean, axis=0) if p == cfg["seq_len"] else mean
        ok = np.isfinite(centred).any(axis=1)
        top = np.argsort(-np.nan_to_num(np.abs(centred), nan=0).max(axis=1))[: cfg["figures"]["heatmap_top_tokens"]]
        top = top[ok[top]]
        plot_token_head_heatmap(centred[top], [f"{texts[i]!r} ({s2['tokens'][i]})" for i in top], names,
                                fig_dir / f"stage2_token_head_heatmap_p{p}.png",
                                f"Tokens with the largest sink_mass effects, position {p}, rare contexts [TL]",
                                "sink_mass − head mean over tokens" if p == cfg["seq_len"] else "Δ sink_mass vs unsubstituted")
    log("stage 2/3 analysis done", t0)

    # ---- Cross-library check on the HF subset ----
    cross = {}
    for b in other:
        diffs = []
        for spec, r_prim, r_other in zip(specs, sweeps[prim], sweeps[b]):
            rows = r_other["bases_idx"] - spec["bases"][0]
            for m in ("sink_mass", "prev_mass", "self_mass", "max_weight"):
                a, c = r_prim["metrics"][m][rows], r_other["metrics"][m]
                diffs.append(np.nanmax(np.abs(a - c)))
        s1_diff = max(float(np.abs(s1[prim]["cells"][c][m] - s1[b]["cells"][c][m]).max())
                      for c in CELLS for m in ("sink_mass", "prev_mass", "self_mass", "max_weight"))
        cross[b] = {"sweep_max_diff": float(max(diffs)), "stage1_max_diff": s1_diff,
                    "stage1_label_agreement": float(np.mean([(s1[prim]["labels"][c] == s1[b]["labels"][c]).mean() for c in CELLS]))}

    write_report(run_dir / "exp1ext_report.md", cfg, run_dir.name, {
        "heads": heads, "decomp": decomp, "tv_cells": tv_cells, "tv_floor": tv_floor, "replication": rep,
        "dose_curves": dose_curves, "ks": list(dose), "effects": effects, "reliability": reliability,
        "feature_models": feature_models, "heldout": heldout, "received": s0["received"], "checks": checks, "cross": cross,
    })
    log(f"wrote {run_dir / 'exp1ext_report.md'}", t0)
    return run_dir


if __name__ == "__main__":
    main()
