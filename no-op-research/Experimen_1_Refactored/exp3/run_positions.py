"""Experiment 3, class 2, position sweep: country at model position i, capital at j, for every 1 <= i < j <= 32.

Run:     uv run --offline python -m exp3.run_positions --config configs/exp3_positions.yaml
Resume:  uv run --offline python -m exp3.run_positions --config configs/exp3_positions.yaml --resume results/<run>

Per placement and base row: 11 correct cells (each country with its capital), two wrong derangements and a placebo
derangement (11 cells each), all from one random Latin square, so each capital sits once per arm and additive token
effects cancel in T = mean(correct) - mean(wrong). Each placement's per-row summaries are saved as they finish
(placements/<base>_<i>_<j>.npz), so an interrupted run can be resumed. See exp3/PREDICTIONS_positions.md.
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
import yaml
from matplotlib.colors import TwoSlopeNorm

from exp1.backends import load_model, set_determinism
from exp1.compare import bh_fdr
from exp1.runlog import make_run_dir, save_config_copy, save_versions
from exp1ext.effects import bootstrap_mean_per_unit
from exp2.analysis import paired_test
from exp2.design import cached_base_block
from exp3.run_relations import PROJECT_ROOT, attention, log, stimuli

CATS = ("BOS", "country", "capital", "previous", "self", "filler")
ARMS = ("correct", "d1", "d2", "d3")
BLUE, ORANGE = "#2a78d6", "#eb6834"


def latin_derangements(n: int, k: int, seed: int) -> np.ndarray:
    """[k, n] mutually disjoint derangements (rows s = 1..k of a randomly relabelled cyclic Latin square)."""
    sigma = np.random.default_rng(seed).permutation(n)
    inv = np.argsort(sigma)
    return np.stack([inv[(sigma + s) % n] for s in range(1, k + 1)])


def all_placements(seq_len: int) -> list[tuple[int, int]]:
    """Every (i, j) with 1 <= i < j <= seq_len; capital-at-query placements (j = seq_len) first, then by j descending."""
    return [(i, j) for j in range(seq_len, 1, -1) for i in range(j - 1, 0, -1)]


def build_placement(bases: np.ndarray, countries: np.ndarray, capitals: np.ndarray, i: int, j: int,
                    partners: np.ndarray) -> np.ndarray:
    """bases [R, seq_len] -> [R, A * n, seq_len]: cell a * n + c writes country c at position i and capital
    partners[a, c] at position j (model positions; array index = position - 1)."""
    a, n = partners.shape
    seqs = np.repeat(bases[:, None, :], a * n, axis=1)
    seqs[:, :, i - 1] = np.tile(countries, a)
    seqs[:, :, j - 1] = capitals[partners.ravel()]
    return seqs


def check_placement(seqs: np.ndarray, bases: np.ndarray, countries: np.ndarray, capitals: np.ndarray, i: int, j: int,
                    partners: np.ndarray, bos_id: int) -> None:
    a, n = partners.shape
    if not (seqs[:, :, i - 1] == np.tile(countries, a)).all() or not (seqs[:, :, j - 1] == capitals[partners.ravel()]).all():
        raise AssertionError(f"placement ({i}, {j}): wrong ids at i or j")
    keep = np.ones(seqs.shape[-1], dtype=bool)
    keep[[i - 1, j - 1]] = False
    if not (seqs[:, :, keep] == bases[:, None, keep]).all():
        raise AssertionError(f"placement ({i}, {j}): differs from the base outside i and j")
    s = np.sort(seqs.reshape(-1, seqs.shape[-1]), axis=1)
    if (s[:, 1:] == s[:, :-1]).any() or (seqs == bos_id).any():
        raise AssertionError(f"placement ({i}, {j}): a sequence repeats a token or contains BOS")


def masses(attn: np.ndarray, i: int, j: int) -> np.ndarray:
    """attn [..., K] at the final query (K - 1) -> [..., 6] masses on CATS. previous = K - 2 and self = K - 1 are
    counted only when they are neither i nor j (else 0, the mass sits under country / capital)."""
    q = attn.shape[-1] - 1
    named = {"BOS": 0, "country": i, "capital": j,
             "previous": None if q - 1 in (i, j) else q - 1, "self": None if q in (i, j) else q}
    out = np.zeros(attn.shape[:-1] + (len(CATS),), dtype=np.float64)
    for k, cat in enumerate(CATS[:-1]):
        if named[cat] is not None:
            out[..., k] = attn[..., named[cat]]
    out[..., -1] = 1.0 - out[..., :-1].sum(axis=-1)
    return out


def summarise_placement(attn: np.ndarray, rows: int, n: int, i: int, j: int, layers: list[int]) -> dict[str, np.ndarray]:
    """attn [rows * 4 * n, L, H, K] -> per-row late-pooled arm masses [rows, 4, 6], per-head T means and Wilcoxon p
    for the fact (correct - wrong) and placebo (d3 - wrong) contrasts [L, H, 6], and mean attention by key per arm
    [4, L, H, K] (over rows)."""
    shape = attn.shape[1:]
    arm = attn.reshape((rows, len(ARMS), n) + shape).mean(axis=2)  # [rows, 4, L, H, K]
    m = masses(arm, i, j)  # [rows, 4, L, H, 6]
    wrong = m[:, 1:3].mean(axis=1)
    out = {"late": m[:, :, layers[0]:layers[1] + 1].mean(axis=(2, 3)).astype(np.float32),
           "by_key": arm.mean(axis=0).astype(np.float32)}
    for name, t in (("fact", m[:, 0] - wrong), ("placebo", m[:, 3] - wrong)):
        out[f"head_mean_{name}"] = t.mean(axis=0).astype(np.float32)
        out[f"head_p_{name}"] = paired_test(t)["p"].astype(np.float32)
    return out


def run_sweep(model, bases: dict[str, np.ndarray], stim: tuple, cfg: dict[str, Any], partners: np.ndarray,
              placements: list[tuple[int, int]], out_dir: Path, t0: float) -> None:
    countries, capitals, _ = stim
    n, layers = len(countries), cfg["analysis"]["late_layers"]
    todo = [(b, i, j) for b in bases for i, j in placements if not (out_dir / f"{b}_{i}_{j}.npz").exists()]
    log(f"{len(todo)} placement x base jobs to run ({len(bases) * len(placements) - len(todo)} already done)", t0)
    start = time.time()
    for k, (b, i, j) in enumerate(todo):
        seqs = build_placement(bases[b], countries, capitals, i, j, partners)
        check_placement(seqs, bases[b], countries, capitals, i, j, partners, cfg["bos_token_id"])
        attn = attention(model, seqs.reshape(-1, seqs.shape[-1]), cfg)
        np.savez(out_dir / f"{b}_{i}_{j}.npz", **summarise_placement(attn, len(bases[b]), n, i, j, layers))
        if k % 20 == 0 or k == len(todo) - 1:
            rate = (time.time() - start) / (k + 1)
            log(f"{b} ({i}, {j}) done; {k + 1}/{len(todo)}; ETA {rate * (len(todo) - k - 1) / 60:.0f} min", t0)


def cross_checks(tl, hf, bases, stim, cfg, partners, out_dir) -> dict[str, Any]:
    countries, capitals, _ = stim
    n, layers, c = len(countries), cfg["analysis"]["late_layers"], cfg["checks"]
    res = {}
    for b, rows in bases.items():
        worst = 0.0
        for i, j in c["hf_placements"]:
            seqs = build_placement(rows[: c["hf_rows"]], countries, capitals, i, j, partners).reshape(-1, rows.shape[1])
            worst = max(worst, float(np.abs(attention(tl, seqs, cfg) - attention(hf, seqs, cfg)).max()))
        i, j = c["determinism_placement"]
        r = c["determinism_rows"]
        seqs = build_placement(rows[:r], countries, capitals, i, j, partners).reshape(-1, rows.shape[1])
        again = summarise_placement(attention(tl, seqs, cfg), r, n, i, j, layers)["late"]
        saved = np.load(out_dir / f"{b}_{i}_{j}.npz")["late"][:r]
        res[b] = {"hf_max_abs_delta": worst, "determinism_max_abs_delta": float(np.abs(again - saved).max())}
    return res


def aggregate(out_dir: Path, bases: list[str], placements: list[tuple[int, int]], cfg: dict[str, Any], seq_len: int) -> dict:
    """Stack the saved placements: per base, late T per row [R, P, 6] for the fact and placebo contrasts, with
    bootstrap CIs, Wilcoxon p and BH q over placements; per-head meaningful counts; mean attention by key."""
    a = cfg["analysis"]
    agg = {}
    for b in bases:
        files = [np.load(out_dir / f"{b}_{i}_{j}.npz") for i, j in placements]
        late = np.stack([f["late"] for f in files], axis=1).astype(np.float64)  # [R, P, 4, 6]
        wrong = late[:, :, 1:3].mean(axis=2)
        res = {"placements": placements, "late": late}
        for name, t in (("fact", late[:, :, 0] - wrong), ("placebo", late[:, :, 3] - wrong)):
            mean, lo, hi = bootstrap_mean_per_unit(t, a["bootstrap"]["iters"], a["bootstrap"]["seed"], 0.95)
            p = paired_test(t)["p"]  # [P, 6]
            q = np.stack([bh_fdr(p[:, k]) for k in range(len(CATS))], axis=1)
            res[name] = {"rows": t, "mean": mean, "lo": lo, "hi": hi, "p": p, "q": q,
                         "sig": (q <= a["q"]) & (np.abs(mean) >= a["margin"])}
        bos = CATS.index("BOS")
        heads = []
        for f in files:
            qh = bh_fdr(f["head_p_fact"][..., bos])
            heads.append(int(((qh <= a["q"]) & (np.abs(f["head_mean_fact"][..., bos]) >= a["head_min_abs"])).sum()))
        res["meaningful_heads"] = np.array(heads)
        res["by_key"] = {pl: f["by_key"] for pl, f in zip(placements, files)}
        agg[b] = res
    return agg


def index_of(placements: list[tuple[int, int]]) -> dict[tuple[int, int], int]:
    return {pl: k for k, pl in enumerate(placements)}


def check_predictions(specs: list[dict], agg: dict, cfg: dict) -> list[dict]:
    boot = cfg["analysis"]["bootstrap"]
    bos, cty, cap = (CATS.index(c) for c in ("BOS", "country", "capital"))
    q = cfg["seq_len"]
    out = []
    for s in specs:
        parts = []
        for b, res in agg.items():
            ix = index_of(res["placements"])
            f = res["fact"]
            if s["kind"] == "query_gap_negative":
                lo_i, hi_i = s["i_range"]
                cells = [(i, f["mean"][ix[(i, q)], bos], f["hi"][ix[(i, q)], bos]) for i in range(lo_i, hi_i + 1)]
                ok = all(h < 0 for _, _, h in cells)
                parts.append((ok, f"{b}: " + ", ".join(f"i={i} {m:+.4f} (hi {h:+.4f})" for i, m, h in cells)))
            elif s["kind"] == "query_decay":
                near = f["rows"][:, ix[(s["near"], q)], bos]
                far = np.mean([f["rows"][:, ix[(i, q)], bos] for i in range(1, s["far_max"] + 1)], axis=0)
                m, lo, hi = (float(v) for v in bootstrap_mean_per_unit(near - far, boot["iters"], boot["seed"], 0.95))
                parts.append((hi < 0, f"{b}: T(near) − mean T(far) {m:+.4f} [{lo:+.4f}, {hi:+.4f}]"))
            elif s["kind"] == "max_fraction_significant":
                sel = [k for k, (i, j) in enumerate(res["placements"]) if j <= s["j_max"]]
                sig = res[s["contrast"]]["sig"][sel, CATS.index(s["metric"])]
                frac = float(sig.mean())
                parts.append((frac <= s["max_fraction"], f"{b}: {int(sig.sum())}/{len(sel)} = {frac:.1%}"))
            elif s["kind"] == "within_pair_shift":
                cells = []
                for j in range(s["j_range"][0], s["j_range"][1] + 1):
                    d = f["rows"][:, ix[(j - 1, j)], cty] - f["rows"][:, ix[(j - 1, j)], cap]
                    m, lo, hi = (float(v) for v in bootstrap_mean_per_unit(d, boot["iters"], boot["seed"], 0.95))
                    cells.append((j, m, lo))
                parts.append((all(lo > 0 for _, _, lo in cells),
                              f"{b}: " + ", ".join(f"({j - 1},{j}) {m:+.4f} (lo {lo:+.4f})" for j, m, lo in cells)))
        ok = bool(parts) and all(p for p, _ in parts)
        out.append({"id": s["id"], "verdict": "PASS" if ok else "FAIL", "detail": "; ".join(d for _, d in parts)})
    return out


def triangle(values: np.ndarray, placements: list[tuple[int, int]], seq_len: int) -> np.ndarray:
    """[P] -> [seq_len, seq_len + 1] array indexed [i, j], NaN where no placement exists."""
    grid = np.full((seq_len, seq_len + 1), np.nan)
    for k, (i, j) in enumerate(placements):
        grid[i, j] = values[k]
    return grid


def plot_triangles(agg: dict, seq_len: int, path: Path, contrast: str = "fact", metrics=("BOS", "country", "capital")) -> None:
    bases = list(agg)
    fig, axes = plt.subplots(len(metrics), len(bases), figsize=(5.6 * len(bases), 4.9 * len(metrics)), squeeze=False)
    for r, metric in enumerate(metrics):
        k = CATS.index(metric)
        vmax = max(float(np.nanmax(np.abs(agg[b][contrast]["mean"][:, k]))) for b in bases) or 1.0
        norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
        for c, b in enumerate(bases):
            ax, res = axes[r, c], agg[b]
            g = triangle(res[contrast]["mean"][:, k], res["placements"], seq_len)
            im = ax.imshow(g[1:, 2:], origin="lower", cmap="RdBu_r", norm=norm, aspect="auto",
                           extent=(1.5, seq_len + 0.5, 0.5, seq_len - 0.5))
            sig = [(j, i) for (i, j), s in zip(res["placements"], res[contrast]["sig"][:, k]) if s]
            if sig:
                ax.scatter(*zip(*sig), s=3, color="black", linewidths=0)
            ax.set_title(f"{b} bases: T on {metric} mass ({contrast})", fontsize=10)
            ax.set_xlabel("capital position j (32 = final token)", fontsize=8)
            ax.set_ylabel("country position i", fontsize=8)
            ax.tick_params(labelsize=7)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="correct − wrong")
    fig.suptitle("Where a correct country→capital pairing changes final-token attention (layers 6–11; dots = significant)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_profiles(agg: dict, seq_len: int, path: Path) -> None:
    bos = CATS.index("BOS")
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.9))
    specs = [("adjacent pair (i = j − 1), by capital position j", lambda i, j: i == j - 1, lambda i, j: j),
             (f"capital at the final token (j = {seq_len}), by country position i", lambda i, j: j == seq_len, lambda i, j: i),
             (f"capital at {seq_len - 1}, by country position i", lambda i, j: j == seq_len - 1, lambda i, j: i)]
    for ax, (title, keep, xof) in zip(axes, specs):
        ax.axhline(0, color="0.6", lw=0.8)
        for b, color in zip(agg, (BLUE, ORANGE)):
            res = agg[b]
            ks = [k for k, (i, j) in enumerate(res["placements"]) if keep(i, j)]
            ks.sort(key=lambda k: xof(*res["placements"][k]))
            x = [xof(*res["placements"][k]) for k in ks]
            f = res["fact"]
            ax.plot(x, f["mean"][ks, bos], color=color, lw=2, label=f"{b} bases")
            ax.fill_between(x, f["lo"][ks, bos], f["hi"][ks, bos], color=color, alpha=0.2, lw=0)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel("late BOS T (correct − wrong)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_patterns(agg: dict, placements: list[list[int]], layers: list[int], path: Path) -> None:
    """Per selected placement and base: change in late-layer attention by key position, correct − wrong."""
    bases = list(agg)
    fig, axes = plt.subplots(len(placements), len(bases), figsize=(6.4 * len(bases), 1.9 * len(placements)),
                             squeeze=False, sharex=True)
    for r, (i, j) in enumerate(placements):
        for c, b in enumerate(bases):
            ax = axes[r, c]
            by_key = agg[b]["by_key"][(i, j)][:, layers[0]:layers[1] + 1].mean(axis=(1, 2))  # [4, K]
            d = by_key[0] - by_key[1:3].mean(axis=0)
            colors = ["0.55"] * len(d)
            colors[0], colors[i], colors[j] = "black", BLUE, ORANGE
            ax.bar(np.arange(len(d)), d, color=colors, width=0.8)
            ax.axhline(0, color="0.6", lw=0.6)
            ax.set_title(f"{b}: country at {i} (blue), capital at {j} (orange); BOS black", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.spines[["top", "right"]].set_visible(False)
            if c == 0:
                ax.set_ylabel("Δ attention", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("key position (final token at 32)", fontsize=8)
    fig.suptitle("Final-token attention, correct − wrong pairing, layers 6–11", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_head_counts(agg: dict, seq_len: int, path: Path) -> None:
    bases = list(agg)
    fig, axes = plt.subplots(1, len(bases), figsize=(5.6 * len(bases), 4.6), squeeze=False)
    vmax = max(int(agg[b]["meaningful_heads"].max()) for b in bases) or 1
    for c, b in enumerate(bases):
        g = triangle(agg[b]["meaningful_heads"].astype(float), agg[b]["placements"], seq_len)
        im = axes[0, c].imshow(g[1:, 2:], origin="lower", cmap="viridis", vmin=0, vmax=vmax, aspect="auto",
                               extent=(1.5, seq_len + 0.5, 0.5, seq_len - 0.5))
        axes[0, c].set_title(f"{b} bases: heads with a meaningful BOS change", fontsize=10)
        axes[0, c].set_xlabel("capital position j", fontsize=8)
        axes[0, c].set_ylabel("country position i", fontsize=8)
        fig.colorbar(im, ax=axes[0, c], fraction=0.046, pad=0.02, label="heads (of 144)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_report(run: Path, cfg: dict, agg: dict, checks: dict, preds: list[dict]) -> None:
    q = cfg["seq_len"]
    L = ["# Experiment 3, class 2: position sweep of the country → capital pair", "",
         f"Run `{run.name}`. Country at i, capital at j, every 1 ≤ i < j ≤ {q} ({len(next(iter(agg.values()))['placements'])} "
         f"placements); {cfg['n_rows']} rows per base class; per placement and row 11 correct, 22 wrong (2 derangements) "
         "and 11 placebo cells. T = mean(correct) − mean(wrong); late = layers "
         f"{cfg['analysis']['late_layers'][0]}–{cfg['analysis']['late_layers'][1]}. Significant = BH q ≤ 0.05 over "
         f"placements and |T| ≥ {cfg['analysis']['margin']}.", "", "## Checks", "", "| check | result |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in checks.items()]
    L += ["", "## Pre-registered predictions", "", "| id | verdict | detail |", "|---|---|---|"]
    L += [f"| {p['id']} | {p['verdict']} | {p['detail']} |" for p in preds]
    L += ["", "## Summary by region of the triangle (late T, mean over placements)", "",
          "| base | region | placements | BOS T | country T | capital T | significant on BOS | significant placebo |",
          "|---|---|---|---|---|---|---|---|"]
    bos, cty, cap = (CATS.index(c) for c in ("BOS", "country", "capital"))
    regions = {f"capital at the final token (j = {q})": lambda i, j: j == q,
               f"adjacent, j ≥ {q - 4}, j < {q}": lambda i, j: i == j - 1 and q - 4 <= j < q,
               f"adjacent, j < {q - 4}": lambda i, j: i == j - 1 and j < q - 4,
               "non-adjacent, j < 32": lambda i, j: i < j - 1 and j < q}
    for b, res in agg.items():
        for name, keep in regions.items():
            ks = [k for k, (i, j) in enumerate(res["placements"]) if keep(i, j)]
            f, p = res["fact"], res["placebo"]
            L.append(f"| {b} | {name} | {len(ks)} | {f['mean'][ks, bos].mean():+.4f} | {f['mean'][ks, cty].mean():+.4f} | "
                     f"{f['mean'][ks, cap].mean():+.4f} | {int(f['sig'][ks, bos].sum())} | {int(p['sig'][ks, bos].sum())} |")
    L += ["", "## Largest late BOS effects", "", "| base | (i, j) | T | 95% CI | q | meaningful heads |", "|---|---|---|---|---|---|"]
    for b, res in agg.items():
        f = res["fact"]
        for k in np.argsort(-np.abs(f["mean"][:, bos]))[:8]:
            i, j = res["placements"][k]
            L.append(f"| {b} | ({i}, {j}) | {f['mean'][k, bos]:+.4f} | [{f['lo'][k, bos]:+.4f}, {f['hi'][k, bos]:+.4f}] | "
                     f"{f['q'][k, bos]:.2g} | {res['meaningful_heads'][k]} |")
    L += ["", "## Figures", "",
          "- `figures/triangle_T.png`: T on BOS, country and capital mass for every placement (dots = significant).",
          "- `figures/triangle_placebo_BOS.png`: the same for the placebo contrast.",
          "- `figures/profiles_BOS.png`: late BOS T along the adjacent diagonal and with the capital at 32 or 31.",
          "- `figures/attention_patterns.png`: change in late attention by key position at selected placements.",
          "- `figures/meaningful_heads.png`: heads with a meaningful BOS change per placement.", ""]
    (run / "exp3_positions_report.md").write_text("\n".join(L))


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Experiment 3, class 2, position sweep")
    parser.add_argument("--config", default="configs/exp3_positions.yaml", type=Path)
    parser.add_argument("--resume", default=None, type=Path, help="an earlier run folder to finish")
    args = parser.parse_args(argv)
    cfg_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    cfg = yaml.safe_load(cfg_path.read_text())
    t0 = time.time()
    if args.resume:
        run = args.resume if args.resume.is_absolute() else PROJECT_ROOT / args.resume
    else:
        run = make_run_dir(PROJECT_ROOT / cfg["paths"]["results_dir"], tag="exp3pos")
        save_config_copy(cfg_path, run)
        shutil.copy2(PROJECT_ROOT / "exp3" / "PREDICTIONS_positions.md", run / "PREDICTIONS.md")
        save_versions(run, PROJECT_ROOT, code_dirs=("exp1", "exp2", "exp3"))
        (run / "figures").mkdir()
    out_dir = run / "placements"
    out_dir.mkdir(exist_ok=True)
    log(f"run folder {run}", t0)

    from transformers import AutoTokenizer

    set_determinism(cfg["seed"])
    tl = load_model("transformer_lens", cfg["backends"]["transformer_lens"]["model_name"], cfg["device"], cfg["dtype"])
    stim = stimuli(cfg, AutoTokenizer.from_pretrained(cfg["figures"]["tokenizer"]))
    n = len(stim[0])
    stim_ids = np.concatenate([*stim, cfg["demonstration_ids"]])
    partners = np.concatenate([np.arange(n)[None], latin_derangements(n, 3, cfg["latin_square_seed"])])  # [4, n]
    placements = (all_placements(cfg["seq_len"]) if cfg["placements"] == "all"
                  else [tuple(p) for p in cfg["placements"]])
    bases = {}
    for b, rng_ in cfg["bases"].items():
        block = cached_base_block(f"exp3rel_{b}", rng_, stim_ids, cfg["base_cache_n"], cfg["seq_len"], cfg["seed"],
                                  cfg["bos_token_id"], PROJECT_ROOT / cfg["paths"]["sequences_dir"])
        bases[b] = block[: cfg["n_rows"]]
        if np.isin(bases[b], stim_ids).any():
            raise AssertionError(f"{b}: a stimulus or demonstration id occurs in a base row")
    run_sweep(tl, bases, stim, cfg, partners, placements, out_dir, t0)

    hf = load_model("huggingface", cfg["backends"]["huggingface"]["model_name"], cfg["device"], cfg["dtype"],
                    cfg["backends"]["huggingface"]["attn_implementation"])
    cross = cross_checks(tl, hf, bases, stim, cfg, partners, out_dir)
    log("cross-library and determinism checks done", t0)
    agg = aggregate(out_dir, list(bases), placements, cfg, cfg["seq_len"])
    log("aggregation done", t0)

    checks = {"design (ids at i/j, base elsewhere, no repeats/BOS/stimulus ids)": "PASS (asserted per placement)",
              "attention rows sum to 1": f"PASS (atol {cfg['checks']['row_sum_atol']:g}, asserted per batch)"}
    for b, c in cross.items():
        checks[f"determinism, {b}"] = f"{'PASS' if c['determinism_max_abs_delta'] == 0 else 'FAIL'} (max |Δ| {c['determinism_max_abs_delta']:.1e})"
        checks[f"TL vs HF, {b}"] = (f"{'PASS' if c['hf_max_abs_delta'] < cfg['checks']['hf_atol'] else 'FAIL'} "
                                    f"(max |Δ| {c['hf_max_abs_delta']:.1e})")
    ix = index_of(placements)
    for b, res in agg.items():
        for pl in ((30, 31), (31, 32)):
            if pl in ix:
                checks[f"consistency {b} {pl} (report only)"] = f"late BOS T {res['fact']['mean'][ix[pl], 0]:+.4f}"
    preds = check_predictions(cfg["predictions"], agg, cfg) if cfg["placements"] == "all" else []

    np.savez_compressed(run / "sweep.npz", placements=np.array(placements), partners=partners,
                        **{f"{b}/late": r["late"] for b, r in agg.items()},
                        **{f"{b}/{c}/{k}": r[c][k] for b, r in agg.items() for c in ("fact", "placebo")
                           for k in ("mean", "lo", "hi", "p", "q")},
                        **{f"{b}/meaningful_heads": r["meaningful_heads"] for b, r in agg.items()})
    (run / "checks.json").write_text(json.dumps(checks, indent=2))
    (run / "predictions.json").write_text(json.dumps(preds, indent=2))
    if cfg["placements"] == "all":
        plot_triangles(agg, cfg["seq_len"], run / "figures" / "triangle_T.png")
        plot_triangles(agg, cfg["seq_len"], run / "figures" / "triangle_placebo_BOS.png", contrast="placebo", metrics=("BOS",))
        plot_profiles(agg, cfg["seq_len"], run / "figures" / "profiles_BOS.png")
        plot_head_counts(agg, cfg["seq_len"], run / "figures" / "meaningful_heads.png")
    pats = [p for p in cfg["figures"]["pattern_placements"] if tuple(p) in ix]
    if pats:
        plot_patterns(agg, pats, cfg["analysis"]["late_layers"], run / "figures" / "attention_patterns.png")
    write_report(run, cfg, agg, checks, preds)
    log(f"done: {run / 'exp3_positions_report.md'}", t0)
    for p in preds:
        print(f"{p['id']}: {p['verdict']}  {p['detail']}")
    return run


if __name__ == "__main__":
    main()
