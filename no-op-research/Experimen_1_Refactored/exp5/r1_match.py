"""Experiment 5, analysis R1: when does the stream at the final token T know whether the country at τ matches the
capital at T?

Run:  uv run --offline python -m exp5.r1_match --data results/20261025-170021_exp5res [--results-dir results] [--workers 8]
      uv run --offline python -m exp5.r1_match --data <same> --posthoc results/<run>_exp5R1   (diagnostic, after the run)

Reads seq_<b>.npz of an exp5.extract run: own = cond 0, wrong = cond 1, cell k = capital k at T in both, so the two
differ only in the country at τ. Per filler type b, τ ∈ seq_taus and stream state l = 0..12, each state is
normalised as LayerNorm does without its gain, and δ_l = x̂_l(own) − x̂_l(wrong) per (row, cell).
- R1-a: max |δ_0| (the built-in null).
- R1-b: S = ‖mean δ‖ / RMS(δ) against a per-δ sign-flip null, BH over layers 1–12 × τ.
- R1-c: the gap d_(1) − d_(2) between the two largest finite Rips H0 deaths of the symmetrised cloud {δ} ∪ {−δ}
  (PCA 50; 20 / 100 checked) against the re-centred sign-flip null, plus average- and single-linkage 2-cluster
  agreement with the sign labels.
- Baseline: a logistic probe own vs wrong on x̂_l, GroupKFold with rows as groups (and leave-one-cell-out).
Operational choices are fixed in exp5/PREDICTIONS_R1.md. Writes results/<timestamp>_exp5R1/: both prediction files,
versions.json, r1_stats.npz, r1_verdicts.json, figures and r1_report.md.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # write figures to files, never open a window
import matplotlib.pyplot as plt
import numpy as np
import yaml
from ripser import ripser
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from exp1.compare import bh_fdr
from exp1.runlog import make_run_dir, save_versions

# Defined here rather than imported from exp3.run_relations, which would import torch into every worker process.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "results" / "20261025-170021_exp5res"
FILLERS = ("common", "rare")
OWN, WRONG = 0, 1           # cond index in seq_<b>.npz (own, wrong, random)
LN_EPS = 1e-5               # GPT-2's LayerNorm epsilon
KS = (20, 50, 100)          # PCA dimensions for R1-c
K_MAIN = 50                 # the one that decides the verdict
TOP_DIMS = 3                # largest-magnitude dimensions dropped in the massive-activation check
PCA_LAYERS = (1, 4, 8, 12)  # layers in the 2-D view
ALPHA = 0.05
BLUE, ORANGE, GREEN, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#898781"
TAU_COLORS = (BLUE, ORANGE, GREEN)
_LIMITS = None


def log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def normalise(x: np.ndarray) -> np.ndarray:
    """LayerNorm without gain or bias over the last axis, in float64."""
    x = x.astype(np.float64)
    x = x - x.mean(-1, keepdims=True)
    return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + LN_EPS)


def perm_p(obs: np.ndarray, null: np.ndarray) -> np.ndarray:
    """(1 + #{null ≥ obs}) / (1 + B), null [..., B]."""
    return (1 + (null >= obs[..., None]).sum(-1)) / (1 + null.shape[-1])


def consistency(d: np.ndarray, signs: np.ndarray, row_signs: np.ndarray, n_rows: int,
                counts: np.ndarray) -> dict[str, Any]:
    """S = ‖mean δ‖ / RMS(δ) with RMS = sqrt(mean_i ‖δ_i‖²), so S ∈ [0, 1]. Nulls: per-δ flips signs [B, N] and
    per-row flips row_signs [B, R] (one sign for a row's δ's); counts [n_boot, R] are row-bootstrap multiplicities.
    RMS is flip-invariant, so only the numerator changes under either null."""
    n, dm = d.shape
    rms = float(np.sqrt((d ** 2).sum(1).mean()))
    if rms == 0:  # layer 0: own and wrong identical
        z = np.zeros(len(signs))
        return {"S": 0.0, "null": z, "row_null": np.zeros(len(row_signs)), "boot": np.zeros(len(counts))}
    rows = d.reshape(n_rows, -1, dm)
    row_sum, row_sq = rows.sum(1), (rows ** 2).sum((1, 2))
    boot = np.linalg.norm(counts @ row_sum, axis=1) / np.sqrt((counts @ row_sq) * n)
    return {"S": float(np.linalg.norm(d.mean(0)) / rms),
            "null": np.linalg.norm(signs @ d, axis=1) / n / rms,
            "row_null": np.linalg.norm(row_signs @ row_sum, axis=1) / n / rms,
            "boot": boot}


def sym_distances(y: np.ndarray) -> np.ndarray:
    """Euclidean distances of the symmetrised cloud [y; −y], built blockwise so that a pair and its mirror image get
    bitwise-identical distances. Rounding would otherwise break the exact mirror ties the gap statistic relies on."""
    g = y @ y.T
    g = (g + g.T) / 2
    sq = np.diag(g).copy()
    same = np.sqrt(np.clip(sq[:, None] + sq[None] - 2 * g, 0, None))
    cross = np.sqrt(np.clip(sq[:, None] + sq[None] + 2 * g, 0, None))
    np.fill_diagonal(same, 0.0)
    return np.block([[same, cross], [cross, same]])


def top_deaths(dist: np.ndarray) -> np.ndarray:
    """The three largest finite Rips H0 death times, largest first. These are the single-linkage merge heights,
    which scipy finds ~20x faster than ripser; every observed cloud is also run through ripser as a check."""
    return linkage(squareform(dist, checks=False), method="single")[-3:, 2][::-1].copy()


def two_cluster_agreement(cond: np.ndarray, method: str, n: int) -> tuple[float, int]:
    """Agreement of the linkage's 2-cluster cut with the sign labels (the first n points are +δ), up to relabelling,
    and the size of the smaller cluster."""
    lab = fcluster(linkage(cond, method=method), 2, criterion="maxclust")
    plus = np.r_[np.ones(n, bool), np.zeros(n, bool)]
    acc = float(((lab == 1) == plus).mean())
    return max(acc, 1 - acc), int(min((lab == 1).sum(), (lab != 1).sum()))


def _init_worker() -> None:
    global _LIMITS
    _LIMITS = threadpool_limits(1)  # one BLAS thread per process; the pool supplies the parallelism


def h0_task(y: np.ndarray, signs: np.ndarray) -> dict[str, Any]:
    """R1-c for one cloud y [N, k] (the δ's in PCA coordinates): the top deaths by scipy and ripser, the linkage
    views, and the gap of each re-centred sign-flip null draw (signs [B, N])."""
    n = len(y)
    dist = sym_distances(y)
    out: dict[str, Any] = {"top": top_deaths(dist)}
    dg = ripser(dist, maxdim=0, distance_matrix=True)["dgms"][0]
    out["top_ripser"] = np.sort(dg[np.isfinite(dg[:, 1]), 1])[::-1][:3]
    cond = squareform(dist, checks=False)
    for method in ("average", "single"):
        out[f"agree_{method}"], out[f"minsize_{method}"] = two_cluster_agreement(cond, method, n)
    # Flipping single δ's maps {δ} ∪ {−δ} onto itself, so the null instead keeps the observed scatter and replaces
    # its consistent shift by the shift of the sign-flipped δ's (PREDICTIONS_R1.md).
    centred = y - y.mean(0)
    null = np.empty(len(signs))
    for b, s in enumerate(signs.astype(np.float64)):
        top = top_deaths(sym_distances(centred + (s @ y) / n))
        null[b] = top[0] - top[1]
    out["null"] = null
    return out


def probe_task(x: np.ndarray, n_rows: int, n_cells: int, n_null: int, seed: list[int]) -> dict[str, Any]:
    """Logistic probe own (first half of x) vs wrong (second half); each half [N, 768] ordered (row, cell).
    Accuracy is pooled over out-of-fold predictions for GroupKFold with rows as groups (a pair never straddles train
    and test), for leave-one-cell-out, and for n_null draws that swap the two labels of each pair at random."""
    n = len(x) // 2
    y = np.r_[np.ones(n, int), np.zeros(n, int)]
    rows = np.tile(np.repeat(np.arange(n_rows), n_cells), 2)
    cells = np.tile(np.tile(np.arange(n_cells), n_rows), 2)
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
    by_rows = GroupKFold(n_splits=min(5, n_rows))

    def acc(labels: np.ndarray, groups: np.ndarray, cv: GroupKFold) -> float:
        return float((cross_val_predict(clf, x, labels, groups=groups, cv=cv) == labels).mean())

    out: dict[str, Any] = {"rows": acc(y, rows, by_rows), "cells": acc(y, cells, GroupKFold(n_splits=n_cells))}
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_null):
        swap = rng.random(n) < 0.5
        y_null = y.copy()
        y_null[:n][swap], y_null[n:][swap] = 0, 1
        null.append(acc(y_null, rows, by_rows))
    out["null"] = np.array(null)
    return out


def cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")


def plot_layers(st: dict[str, np.ndarray], taus: list[int], path: Path) -> None:
    """S by layer with its row-bootstrap CI and the sign-flip null's 95th percentile; probe accuracy with the
    label-swap null band."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    ls = np.arange(1, st["S"].shape[-1])
    for b, name in enumerate(FILLERS):
        ax = axes[b, 0]
        for t, tau in enumerate(taus):
            c = TAU_COLORS[t]
            ax.plot(ls, st["S"][b, t, 1:], color=c, marker="o", ms=3, label=f"τ = {tau} (d = {32 - tau})")
            ax.fill_between(ls, st["S_lo"][b, t, 1:], st["S_hi"][b, t, 1:], color=c, alpha=0.18, lw=0)
            ax.plot(ls, st["S_null95"][b, t, 1:], color=c, ls="--", lw=1)
        ax.set_yscale("log")
        ax.set_xticks(ls)
        ax.set_xlabel("stream state l (l = 0 is identically 0)")
        ax.set_ylabel("S = ‖mean δ‖ / RMS(δ)")
        ax.set_title(f"{name}: consistency (dashed: sign-flip null 95th pct)", fontsize=10)
        ax.legend(fontsize=8, frameon=False)
        ax = axes[b, 1]
        lo, hi = st["probe_band"][b, :, 0].min(), st["probe_band"][b, :, 1].max()
        ax.axhspan(lo, hi, color=GREY, alpha=0.25, lw=0, label="label-swap null 2.5–97.5%")
        la = np.arange(st["S"].shape[-1])
        for t, tau in enumerate(taus):
            c = TAU_COLORS[t]
            ax.plot(la, st["probe_rows"][b, t], color=c, marker="o", ms=3, label=f"τ = {tau}, rows CV")
            ax.plot(la, st["probe_cells"][b, t], color=c, ls=":", lw=1.3, label=f"τ = {tau}, leave-one-cell-out")
        ax.set_ylim(0.3, 1.02)
        ax.set_xticks(la)
        ax.set_xlabel("stream state l")
        ax.set_ylabel("held-out accuracy, own vs wrong")
        ax.set_title(f"{name}: linear probe (baseline)", fontsize=10)
        ax.legend(fontsize=7, frameon=False, ncol=2, loc="lower right")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_h0(st: dict[str, np.ndarray], taus: list[int], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    ki = KS.index(K_MAIN)
    ls = np.arange(1, st["gap"].shape[2])
    for b, name in enumerate(FILLERS):
        ax = axes[b, 0]
        for t, tau in enumerate(taus):
            c = TAU_COLORS[t]
            ax.plot(ls, st["gap"][b, t, 1:, ki], color=c, marker="o", ms=3, label=f"τ = {tau}")
            ax.plot(ls, st["gap_null95"][b, t, 1:, ki], color=c, ls="--", lw=1)
        ax.set_xticks(ls)
        ax.set_xlabel("stream state l")
        ax.set_ylabel("gap d(1) − d(2) (normalised units)")
        ax.set_title(f"{name}: Rips H0 2-component gap, PCA {K_MAIN} (dashed: null 95th pct)", fontsize=10)
        ax.legend(fontsize=8, frameon=False)
        ax = axes[b, 1]
        for t, tau in enumerate(taus):
            c = TAU_COLORS[t]
            ax.plot(ls, st["agree_average"][b, t, 1:], color=c, marker="o", ms=3, label=f"τ = {tau}, average")
            ax.plot(ls, st["agree_single"][b, t, 1:], color=c, ls=":", lw=1.3, label=f"τ = {tau}, single")
        ax.axhline(0.5, color=GREY, lw=0.8)
        ax.set_ylim(0.45, 1.02)
        ax.set_xticks(ls)
        ax.set_xlabel("stream state l")
        ax.set_ylabel("agreement with sign labels")
        ax.set_title(f"{name}: 2-cluster cut of ±δ (PCA {K_MAIN}) vs the ± labels", fontsize=10)
        ax.legend(fontsize=7, frameon=False, ncol=2)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pca(pca2: np.ndarray, mean2: np.ndarray, taus: list[int], name: str, path: Path) -> None:
    """pca2 [nτ, len(PCA_LAYERS), N, 2]: the δ's on the first two principal axes of {δ} ∪ {−δ}."""
    nt, nl = pca2.shape[:2]
    fig, axes = plt.subplots(nt, nl, figsize=(3.1 * nl, 3.0 * nt), constrained_layout=True)
    for t in range(nt):
        for j, l in enumerate(PCA_LAYERS):
            ax = axes[t, j]
            p = pca2[t, j]
            ax.scatter(p[:, 0], p[:, 1], s=3, color=BLUE, alpha=0.35, lw=0, label="+δ")
            ax.scatter(-p[:, 0], -p[:, 1], s=3, color=ORANGE, alpha=0.35, lw=0, label="−δ")
            ax.scatter(*mean2[t, j], marker="x", color="k", s=30, label="mean δ")
            ax.set_title(f"{name}, τ = {taus[t]}, l = {l}", fontsize=9)
            ax.tick_params(labelsize=7)
            if t == 0 and j == 0:
                ax.legend(fontsize=7, frameon=False, markerscale=3)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    ap.add_argument("--n-null", type=int, default=2000, help="sign-flip draws: R1-b, and R1-c at PCA 50")
    ap.add_argument("--n-null-check", type=int, default=500, help="R1-c draws at PCA 20 / 100")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-probe-null", type=int, default=10, help="label-swap draws per layer for the probe band")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--posthoc", type=Path, default=None,
                    help="an existing exp5R1 run folder: only append the post-hoc R1-c diagnostic to its report")
    args = ap.parse_args(argv)
    t0 = time.time()
    data = args.data if args.data.is_absolute() else PROJECT_ROOT / args.data
    if not (data / "DONE").exists():
        raise SystemExit(f"{data} is incomplete (no DONE file)")
    if args.posthoc is not None:
        run = args.posthoc if args.posthoc.is_absolute() else PROJECT_ROOT / args.posthoc
        posthoc(run, data, t0)
        return run
    cfg = yaml.safe_load((data / "config.yaml").read_text())
    taus = list(cfg["seq_taus"])
    stim = cfg["stimuli"]
    inv = np.load(data / "weights.npz")["inv"]
    run = make_run_dir(args.results_dir, tag="exp5R1")
    save_versions(run, PROJECT_ROOT, code_dirs=("exp1", "exp2", "exp3", "exp4", "exp5"))
    for name in ("PREDICTIONS.md", "PREDICTIONS_R1.md"):
        shutil.copy2(PROJECT_ROOT / "exp5" / name, run / name)
    log(f"run folder {run}; data {data}", t0)

    pool = ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker)
    h0_jobs: dict[tuple[int, int, int, int], Future] = {}
    probe_jobs: dict[tuple[int, int, int], Future] = {}
    per: dict[str, list] = {k: [] for k in ("S", "null", "row_null", "boot", "S_top", "null_top", "top_dims",
                                            "mean_vec", "half_vec", "rms", "pca2", "mean2", "max_d0", "max_d0_raw")}
    n_rows = n_cells = None
    for b, name in enumerate(FILLERS):
        with np.load(data / f"seq_{name}.npz") as z:
            resid = z["resid"][:, :2]  # [R, own/wrong, nτ, n, 13, 768] fp16
        n_rows, _, nt, n_cells, ns, dm = resid.shape
        n = n_rows * n_cells
        rng = np.random.default_rng([args.seed, b])
        # One sign per (row, cell, τ) δ and draw, shared by all layers (and by R1-b and R1-c).
        flips = (rng.integers(0, 2, size=(args.n_null, nt, n), dtype=np.int8) * 2 - 1).astype(np.int8)
        row_flips = rng.integers(0, 2, size=(args.n_null, nt, n_rows)) * 2.0 - 1
        counts = rng.multinomial(n_rows, np.full(n_rows, 1 / n_rows), size=args.n_boot).astype(np.float64)
        acc = {k: np.zeros((nt, ns)) for k in ("S", "S_top", "rms")}
        acc.update(null=np.zeros((nt, ns, args.n_null)), row_null=np.zeros((nt, ns, args.n_null)),
                   null_top=np.zeros((nt, ns, args.n_null)), boot=np.zeros((nt, ns, args.n_boot)),
                   top_dims=np.zeros((nt, ns, TOP_DIMS), int), mean_vec=np.zeros((nt, ns, dm)),
                   half_vec=np.zeros((nt, ns, 2, dm)), pca2=np.zeros((nt, len(PCA_LAYERS), n, 2)),
                   mean2=np.zeros((nt, len(PCA_LAYERS), 2)))
        max_d0 = max_d0_raw = 0.0
        for t in range(nt):
            signs = flips[:, t].astype(np.float64)
            for l in range(ns):
                xo = normalise(resid[:, OWN, t, :, l])
                xw = normalise(resid[:, WRONG, t, :, l])
                d = (xo - xw).reshape(n, dm)
                if l == 0:
                    max_d0 = max(max_d0, float(np.abs(d).max()))
                    raw = resid[:, OWN, t, :, 0].astype(np.float32) - resid[:, WRONG, t, :, 0].astype(np.float32)
                    max_d0_raw = max(max_d0_raw, float(np.abs(raw).max()))
                c = consistency(d, signs, row_flips[:, t], n_rows, counts)
                acc["S"][t, l], acc["null"][t, l], acc["row_null"][t, l], acc["boot"][t, l] = (
                    c["S"], c["null"], c["row_null"], c["boot"])
                acc["rms"][t, l] = np.sqrt((d ** 2).sum(1).mean())
                # Massive-activation check: drop the dimensions with the largest mean |x̂| at this (τ, l).
                top = np.argsort(np.abs(np.concatenate([xo, xw])).reshape(-1, dm).mean(0))[-TOP_DIMS:]
                d_top = d.copy()
                d_top[:, top] = 0
                c = consistency(d_top, signs, row_flips[:, t], n_rows, counts)
                acc["S_top"][t, l], acc["null_top"][t, l], acc["top_dims"][t, l] = c["S"], c["null"], top
                acc["mean_vec"][t, l] = d.mean(0)
                half = n_rows // 2
                rows = d.reshape(n_rows, n_cells, dm)
                acc["half_vec"][t, l] = rows[:half].mean((0, 1)), rows[half:].mean((0, 1))
                x = np.concatenate([xo.reshape(n, dm), xw.reshape(n, dm)]).astype(np.float32)
                probe_jobs[b, t, l] = pool.submit(probe_task, x, n_rows, n_cells,
                                                  args.n_probe_null if l > 0 else 0, [args.seed, 2, b, t, l])
                if l == 0:
                    continue
                # PCA of {δ} ∪ {−δ}: its mean is exactly 0, so the axes are the right singular vectors of δ.
                _, _, vt = np.linalg.svd(d, full_matrices=False)
                y = d @ vt[:max(KS)].T
                if l in PCA_LAYERS:
                    j = PCA_LAYERS.index(l)
                    acc["pca2"][t, j], acc["mean2"][t, j] = y[:, :2], y[:, :2].mean(0)
                for ki, k in enumerate(KS):
                    nb = args.n_null if k == K_MAIN else args.n_null_check
                    h0_jobs[b, t, l, ki] = pool.submit(h0_task, np.ascontiguousarray(y[:, :k]), flips[:nb, t])
            log(f"{name} τ = {taus[t]}: statistics done, {len(h0_jobs)} H0 and {len(probe_jobs)} probe jobs queued", t0)
        for k in acc:
            per[k].append(acc[k])
        per["max_d0"].append(max_d0)
        per["max_d0_raw"].append(max_d0_raw)
        del resid

    st: dict[str, np.ndarray] = {k: np.array(v) for k, v in per.items()}
    nb_, nt, ns = len(FILLERS), len(taus), st["S"].shape[-1]
    st["S_null95"] = np.percentile(st["null"], 95, axis=-1)
    st["S_p"] = perm_p(st["S"], st["null"])
    st["S_lo"], st["S_hi"] = np.percentile(st["boot"], [2.5, 97.5], axis=-1)
    st["S_rownull95"] = np.percentile(st["row_null"], 95, axis=-1)
    st["S_rowp"] = perm_p(st["S"], st["row_null"])
    st["S_top_null95"] = np.percentile(st["null_top"], 95, axis=-1)
    st["S_top_p"] = perm_p(st["S_top"], st["null_top"])
    st["mean_norm"] = np.linalg.norm(st["mean_vec"], axis=-1)

    def bh_pass(stat: np.ndarray, null95: np.ndarray, p: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per filler: BH over the masked cells; a cell passes if above the null's 95th pct and q < ALPHA."""
        q = np.ones_like(p)
        for b in range(nb_):
            q[b][mask[b]] = bh_fdr(p[b][mask[b]])
        return q, (stat > null95) & (q < ALPHA) & mask

    layer_mask = np.zeros((nb_, nt, ns), bool)
    layer_mask[:, :, 1:] = True
    st["S_q"], st["S_pass"] = bh_pass(st["S"], st["S_null95"], st["S_p"], layer_mask)
    st["S_rowq"], st["S_rowpass"] = bh_pass(st["S"], st["S_rownull95"], st["S_rowp"], layer_mask)
    st["S_top_q"], st["S_top_pass"] = bh_pass(st["S_top"], st["S_top_null95"], st["S_top_p"], layer_mask)
    first = np.full((nb_, nt), -1)
    for b in range(nb_):
        for t in range(nt):
            ok = np.flatnonzero(st["S_pass"][b, t])
            first[b, t] = ok[0] if len(ok) else -1
    st["first_layer"] = first
    tested = np.zeros((nb_, nt, ns), bool)
    for b in range(nb_):
        for t in range(nt):
            tested[b, t, max(first[b, t], 1):] = True
    st["h0_tested"] = tested
    log("R1-b done; waiting for H0 jobs", t0)

    nk = len(KS)
    st["top"], st["top_ripser"] = np.zeros((nb_, nt, ns, nk, 3)), np.zeros((nb_, nt, ns, nk, 3))
    st["gap_null95"], st["gap_p"] = np.zeros((nb_, nt, ns, nk)), np.ones((nb_, nt, ns, nk))
    for name in ("agree_average", "agree_single", "minsize_average", "minsize_single"):
        st[name] = np.full((nb_, nt, ns), np.nan)
    st["agree_average"][:, :, 0] = st["agree_single"][:, :, 0] = 0.5
    null_main = np.zeros((nb_, nt, ns, args.n_null))
    for i, ((b, t, l, ki), fut) in enumerate(h0_jobs.items()):
        r = fut.result()
        st["top"][b, t, l, ki], st["top_ripser"][b, t, l, ki] = r["top"], r["top_ripser"]
        gap = r["top"][0] - r["top"][1]
        st["gap_null95"][b, t, l, ki] = np.percentile(r["null"], 95)
        st["gap_p"][b, t, l, ki] = perm_p(np.array(gap), r["null"])
        if KS[ki] == K_MAIN:
            null_main[b, t, l] = r["null"]
            for name in ("agree_average", "agree_single", "minsize_average", "minsize_single"):
                st[name][b, t, l] = r[name]
        if (i + 1) % 36 == 0:
            log(f"H0 jobs {i + 1}/{len(h0_jobs)}", t0)
    st["gap"] = st["top"][..., 0] - st["top"][..., 1]
    st["gap_ratio"] = np.divide(st["top"][..., 0], st["top"][..., 1], out=np.full(st["gap"].shape, np.nan),
                                where=st["top"][..., 1] > 0)
    st["gap_null_main"] = null_main
    scale = np.maximum(st["top"][..., 0], 1e-12)
    st["ripser_rel_diff"] = np.abs(st["top_ripser"] - st["top"]).max(-1) / scale
    st["gap_q"], st["gap_pass"] = np.ones_like(st["gap_p"]), np.zeros_like(st["gap_p"], bool)
    for ki in range(nk):
        st["gap_q"][..., ki], st["gap_pass"][..., ki] = bh_pass(st["gap"][..., ki], st["gap_null95"][..., ki],
                                                                st["gap_p"][..., ki], tested)

    log("H0 done; waiting for probe jobs", t0)
    st["probe_rows"], st["probe_cells"] = np.zeros((nb_, nt, ns)), np.zeros((nb_, nt, ns))
    st["probe_null"] = np.full((nb_, nt, ns, args.n_probe_null), np.nan)
    for (b, t, l), fut in probe_jobs.items():
        r = fut.result()
        st["probe_rows"][b, t, l], st["probe_cells"][b, t, l] = r["rows"], r["cells"]
        if l > 0:
            st["probe_null"][b, t, l] = r["null"]
    pool.shutdown()
    st["probe_band"] = np.stack([np.nanpercentile(st["probe_null"][:, :, 1:].reshape(nb_, nt, -1), q, axis=-1)
                                 for q in (2.5, 97.5)], axis=-1)

    # Exploratory: is the mean δ direction shared across τ, across filler types, and between halves of the rows?
    mv = st["mean_vec"]
    st["cos_tau"] = np.array([[[[cos(mv[b, t1, l], mv[b, t2, l]) for t2 in range(nt)] for t1 in range(nt)]
                               for l in range(ns)] for b in range(nb_)])  # [b, l, τ1, τ2]
    st["cos_fill"] = np.array([[cos(mv[0, t, l], mv[1, t, l]) for l in range(ns)] for t in range(nt)])  # [τ, l]
    st["cos_half"] = np.array([[[cos(*st["half_vec"][b, t, l]) for l in range(ns)] for t in range(nt)]
                               for b in range(nb_)])  # [b, τ, l]

    plot_layers(st, taus, run / "r1_layers.png")
    plot_h0(st, taus, run / "r1_h0.png")
    for b, name in enumerate(FILLERS):
        plot_pca(st["pca2"][b], st["mean2"][b], taus, name, run / f"r1_pca_{name}.png")
    keep = {k: v for k, v in st.items() if k not in ("boot", "null_top", "pca2", "half_vec")}
    np.savez(run / "r1_stats.npz", taus=np.array(taus), fillers=np.array(FILLERS), ks=np.array(KS), **keep)
    verdicts = write_report(run, st, taus, stim, inv, data, args, n_rows, n_cells, time.time() - t0)
    (run / "r1_verdicts.json").write_text(json.dumps(verdicts, indent=2) + "\n")
    log(f"done: {run / 'r1_report.md'}", t0)
    return run


def write_report(run: Path, st: dict[str, np.ndarray], taus: list[int], stim: list[dict], inv: np.ndarray,
                 data: Path, args: argparse.Namespace, n_rows: int, n_cells: int, secs: float) -> dict[str, Any]:
    nb_, nt, ns = st["S"].shape
    ki = KS.index(K_MAIN)
    L = range(1, ns)
    r1a = all(m < 1e-5 for m in st["max_d0"])
    r1b_cells = [int(st["S_pass"][b, :, 1:].sum()) for b in range(nb_)]
    r1b = all(c == nt * (ns - 1) for c in r1b_cells)
    tested = st["h0_tested"]
    r1c_ok = [(int(st["gap_pass"][b, ..., ki][tested[b]].sum()), int(tested[b].sum())) for b in range(nb_)]
    r1c = all(a == m for a, m in r1c_ok)
    r1c_k = {k: all((st["gap_pass"][b, ..., i][tested[b]]).all() for b in range(nb_)) for i, k in enumerate(KS)}
    ratio = st["S"][:, :, 1:] / st["S_null95"][:, :, 1:]
    first = {f"{FILLERS[b]} τ={taus[t]}": int(st["first_layer"][b, t]) for b in range(nb_) for t in range(nt)}
    verdicts: dict[str, Any] = {
        "R1-a": {"verdict": "PASS" if r1a else "FAIL", "max_abs_delta0": [float(m) for m in st["max_d0"]]},
        "R1-b": {"verdict": "PASS" if r1b else "FAIL", "cells_passing": r1b_cells, "of": nt * (ns - 1),
                 "first_layer": first, "min_S_over_null95": [float(ratio[b].min()) for b in range(nb_)]},
        "R1-c": {"verdict": "PASS" if r1c else "FAIL", "tested_cells_beating_null": [list(x) for x in r1c_ok],
                 "verdict_by_pca_dim": {str(k): "PASS" if v else "FAIL" for k, v in r1c_k.items()}},
    }

    def f3(v: float) -> str:
        return f"{v:.3f}"

    def fp(p: float) -> str:
        return f"{p:.1e}" if p < 1e-3 else f"{p:.3f}"

    out = [f"# Experiment 5, R1: when does the stream at T know the match?", "",
           f"Run `{run.name}`, data `{data}` ({n_rows} rows per filler type × {n_cells} cells, τ ∈ {taus}), "
           f"{secs / 60:.1f} min. Pre-registration `PREDICTIONS.md`, operational addendum `PREDICTIONS_R1.md` (both "
           f"copied here). Sign-flip draws {args.n_null} (R1-c at PCA 20 / 100: {args.n_null_check}), row bootstrap "
           f"{args.n_boot}, probe label-swap draws {args.n_probe_null} per layer.", "",
           "## Verdicts", "", "| id | verdict | short |", "|---|---|---|"]
    out.append(f"| R1-a | {verdicts['R1-a']['verdict']} | max \\|δ_0\\| after normalisation = "
               f"{st['max_d0'][0]:.1e} / {st['max_d0'][1]:.1e} (common / rare); raw {st['max_d0_raw'][0]:.1e} / "
               f"{st['max_d0_raw'][1]:.1e}. |")
    out.append(f"| R1-b | {verdicts['R1-b']['verdict']} | S beats the per-δ sign-flip null (above its 95th pct, BH q "
               f"< 0.05) in {r1b_cells[0]}/{nt * (ns - 1)} and {r1b_cells[1]}/{nt * (ns - 1)} cells (layers 1–12 × "
               f"3 τ); S / null95 ≥ {ratio[0].min():.1f} / {ratio[1].min():.1f}. First passing layer: "
               + ", ".join(f"{k} → {v}" for k, v in first.items()) + ". |")
    out.append(f"| R1-c | {verdicts['R1-c']['verdict']} | PCA {K_MAIN}: gap beats the re-centred sign-flip null in "
               f"{r1c_ok[0][0]}/{r1c_ok[0][1]} (common) and {r1c_ok[1][0]}/{r1c_ok[1][1]} (rare) tested cells. "
               f"PCA 20 / 100: {verdicts['R1-c']['verdict_by_pca_dim']['20']} / "
               f"{verdicts['R1-c']['verdict_by_pca_dim']['100']}. |")
    pr = st["probe_rows"][:, :, 1:]
    out.append(f"| probe | baseline | rows-grouped CV accuracy over layers 1–12: {pr[0].min():.3f}–{pr[0].max():.3f} "
               f"(common), {pr[1].min():.3f}–{pr[1].max():.3f} (rare); layer 0 = "
               f"{st['probe_rows'][:, :, 0].min():.3f}–{st['probe_rows'][:, :, 0].max():.3f}. |")
    out += ["", "## R1-b: consistency S = ‖mean δ‖ / RMS(δ) by layer", "",
            "S [row-bootstrap 95% CI]; null95 = 95th percentile of the per-δ sign-flip null; ✓ = above null95 with "
            "BH q < 0.05 (BH over the 36 cells of a filler type). Pooled over 11 cells × rows within τ (N = "
            f"{n_rows * n_cells}).", ""]
    for b, name in enumerate(FILLERS):
        out += [f"**{name}**", "", "| l | " + " | ".join(f"τ = {t}: S [CI] | null95 | q" for t in taus) + " |",
                "|---|" + "---|---|---|" * nt]
        for l in range(ns):
            cells = []
            for t in range(nt):
                mark = " ✓" if st["S_pass"][b, t, l] else ""
                cells.append(f"{f3(st['S'][b, t, l])} [{f3(st['S_lo'][b, t, l])}, {f3(st['S_hi'][b, t, l])}]{mark} | "
                             f"{f3(st['S_null95'][b, t, l])} | {fp(st['S_q'][b, t, l]) if l else '—'}")
            out.append(f"| {l} | " + " | ".join(cells) + " |")
        out.append("")
    out += ["## R1-c: Rips H0 of the symmetrised cloud {δ} ∪ {−δ}", "",
            f"gap = d(1) − d(2), the two largest finite H0 death times (PCA {K_MAIN}, normalised units): the scale "
            "range over which exactly two components exist. null95 = 95th percentile of the re-centred sign-flip "
            "null (see the addendum); ✓ = above null95 with BH q < 0.05 over the tested cells (layers from the "
            "first R1-b layer of that τ). agree = agreement of the 2-cluster cut with the ± labels (average / "
            "single linkage), min = size of the smaller cluster (of "
            f"{2 * n_rows * n_cells}); min 0 means the top two merges tie, so no cut gives exactly two clusters and "
            "the agreement is 0.5. PCA 20 / 100: ✓ / · = beats / does not beat its own null.", ""]
    for b, name in enumerate(FILLERS):
        out += [f"**{name}**", "",
                "| τ | l | d(1) | d(2) | gap | null95 | q | PCA 20 / 100 | agree avg (min) | agree single (min) |",
                "|---|---|---|---|---|---|---|---|---|---|"]
        for t in range(nt):
            for l in L:
                g = st["gap_pass"][b, t, l]
                mark = " ✓" if g[ki] else ""
                chk = " / ".join("✓" if g[i] else "·" for i, k in enumerate(KS) if k != K_MAIN)
                tst = "" if st["h0_tested"][b, t, l] else " (not tested)"
                out.append(f"| {taus[t]} | {l} | {f3(st['top'][b, t, l, ki, 0])} | {f3(st['top'][b, t, l, ki, 1])} | "
                           f"{st['gap'][b, t, l, ki]:.4f}{mark}{tst} | {st['gap_null95'][b, t, l, ki]:.4f} | "
                           f"{fp(st['gap_q'][b, t, l, ki])} | {chk} | {st['agree_average'][b, t, l]:.3f} "
                           f"({int(st['minsize_average'][b, t, l])}) | {st['agree_single'][b, t, l]:.3f} "
                           f"({int(st['minsize_single'][b, t, l])}) |")
        out.append("")
    out += ["## Probe (non-topological baseline)", "",
            "Logistic regression (standardised features, L2, C = 1) on the normalised states, own = 1 vs wrong = 0, "
            "accuracy pooled over out-of-fold predictions. rows = GroupKFold(5) with rows as groups; LOCO = "
            "leave-one-cell-out (the held-out pair's capital never appears in training, its own country only in a "
            "wrong pair). Null band = 2.5–97.5 percentile of rows-CV accuracy after swapping each pair's labels at "
            "random, pooled over layers 1–12.", "",
            "| filler | τ | null band | " + " | ".join(f"l={l}" for l in range(ns)) + " |",
            "|---|---|---|" + "---|" * ns]
    for b, name in enumerate(FILLERS):
        for t in range(nt):
            band = f"{st['probe_band'][b, t, 0]:.3f}–{st['probe_band'][b, t, 1]:.3f}"
            out.append(f"| {name} | {taus[t]} | {band} | " + " | ".join(
                f"{st['probe_rows'][b, t, l]:.2f} / {st['probe_cells'][b, t, l]:.2f}" for l in range(ns)) + " |")
    out += ["", "(cells: rows-CV / LOCO accuracy)", "", "## Robustness", ""]
    rp = [int(st["S_rowpass"][b, :, 1:].sum()) for b in range(nb_)]
    tp = [int(st["S_top_pass"][b, :, 1:].sum()) for b in range(nb_)]
    dims = sorted({int(x) for x in st["top_dims"][:, :, 1:].ravel()})
    rel = float(st["ripser_rel_diff"][:, :, 1:].max())
    out += [f"- **Row-level sign-flip null** (one sign per row, the calibrated null when rows are the exchangeable "
            f"unit): S beats it in {rp[0]}/{nt * (ns - 1)} and {rp[1]}/{nt * (ns - 1)} cells.",
            f"- **Top-{TOP_DIMS} dimensions removed** (largest mean |x̂| per τ and layer; dims used: {dims}): S beats "
            f"its sign-flip null in {tp[0]}/{nt * (ns - 1)} and {tp[1]}/{nt * (ns - 1)} cells; S without them / S "
            f"with them, median {np.median(st['S_top'][:, :, 1:] / st['S'][:, :, 1:]):.3f}.",
            f"- **PCA dimension for R1-c**: verdict at PCA 20 / 50 / 100 = "
            + " / ".join(verdicts["R1-c"]["verdict_by_pca_dim"][str(k)] for k in KS) + "; beating cells per filler "
            + "; ".join(f"PCA {k}: " + " / ".join(f"{int(st['gap_pass'][b, ..., i][tested[b]].sum())}/"
                                                 f"{int(tested[b].sum())}" for b in range(nb_))
                        for i, k in enumerate(KS)) + ".",
            f"- **ripser vs scipy single linkage** (identical by definition): max relative difference of the top 3 "
            f"deaths over all observed clouds = {rel:.1e} (ripser computes in float32).",
            f"- **Average-linkage view** (not a criterion): agreement with the ± labels at PCA {K_MAIN}, layers 1–12, "
            + "; ".join(f"{FILLERS[b]} τ={taus[t]}: {np.nanmin(st['agree_average'][b, t, 1:]):.3f}–"
                        f"{np.nanmax(st['agree_average'][b, t, 1:]):.3f}" for b in range(nb_) for t in range(nt))
            + ". Single linkage: "
            + "; ".join(f"{FILLERS[b]} τ={taus[t]}: {np.nanmin(st['agree_single'][b, t, 1:]):.3f}–"
                        f"{np.nanmax(st['agree_single'][b, t, 1:]):.3f}" for b in range(nb_) for t in range(nt)) + ".",
            "", "## Exploratory (not pre-registered)", "",
            "‖mean δ‖ and RMS(δ) in normalised units (a normalised state has norm √768 ≈ 27.7). cos(τ, τ') = cosine "
            "between the mean δ directions of two τ; cos(fill) = common vs rare at the same τ; cos(half) = rows "
            "1–50 vs 51–100 at the same τ (the reliability ceiling for the other cosines).", "",
            "| filler | l | ‖mean δ‖ τ=" + " / ".join(map(str, taus)) + " | RMS(δ) | cos(31,16) | cos(31,1) | "
            "cos(16,1) | cos(fill) τ=" + " / ".join(map(str, taus)) + " | cos(half) |", "|---|---|---|---|---|---|---|---|---|"]
    ti = {tau: i for i, tau in enumerate(taus)}
    for b, name in enumerate(FILLERS):
        for l in L:
            c = st["cos_tau"][b, l]
            out.append(f"| {name} | {l} | " + " / ".join(f"{st['mean_norm'][b, t, l]:.2f}" for t in range(nt))
                       + " | " + " / ".join(f"{st['rms'][b, t, l]:.2f}" for t in range(nt))
                       + f" | {c[ti[31], ti[16]]:.2f} | {c[ti[31], ti[1]]:.2f} | {c[ti[16], ti[1]]:.2f} | "
                       + " / ".join(f"{st['cos_fill'][t, l]:.2f}" for t in range(nt)) + " | "
                       + " / ".join(f"{st['cos_half'][b, t, l]:.2f}" for t in range(nt)) + " |")
    pairs = ", ".join(f"{stim[k]['capital'].strip()}: {stim[k]['country'].strip()} vs "
                      f"{stim[int(inv[k])]['country'].strip()}" for k in range(len(stim)))
    out += ["", "## Caveats", "",
            f"- **The same 11 pairs and one derangement throughout** (cell k = capital: own country vs wrong country; "
            f"{pairs}). Pooling the 11 cells cancels each country's and each capital's main effect, so a consistent "
            "mean δ is a capital × country interaction shared across these pairs relative to this derangement; "
            "\"beyond the null\" means consistent across rows for these pairs, not general.",
            "- **The per-δ sign-flip null is conservative** for S: within a row the country main effects cancel "
            "across the 11 cells, so the real mean δ varies less than the flipped one. The row-level null above is "
            "the calibrated alternative.",
            "- **R1-c's null is an operational choice.** Flipping single δ's leaves {δ} ∪ {−δ} unchanged, so the "
            "literal sign-flip null equals the observed diagram in every draw (p = 1 everywhere). The re-centred "
            "null keeps the observed scatter and gives the cloud a sign-flip-sized shift; the projection basis is "
            "fitted once on the observed cloud.",
            "- **Rips H0 is single linkage** and chains (Exp 4). On a mirror-symmetric cloud every merge that does not "
            "join a set to its own mirror image happens twice at the same scale, so a positive gap means the last "
            "merge joins two mirror halves; whether those halves are the ± labels is what the linkage agreement "
            "shows.",
            "- **The probe sees 768 features and 1,760 training states per fold.** High accuracy is easy to reach "
            "with 11 fixed pairs; LOCO is the stricter test of one shared direction.",
            "- Statistics come from the fp16 arrays of exp5.extract (cast to float64 after loading).", ""]
    (run / "r1_report.md").write_text("\n".join(out))
    return verdicts


POSTHOC_HEADING = "## Post-hoc diagnostic of R1-c"


def posthoc(run: Path, data: Path, t0: float) -> None:
    """Diagnostic of R1-c, added after the main run (not pre-registered; it changes no verdict): are the ± halves
    separated at all, and what makes the Rips H0 gap vanish? Per (filler, τ, l ≥ 1), on the δ's and their PCA-50
    cloud: the share of δ's with a positive projection on the mean δ (all positive = the halves are linearly
    separable), the smallest distance between the halves vs the largest nearest-neighbour distance within one, and,
    where the gap is 0, the smallest group of the single-linkage 3-cluster cut (it and its mirror image merge last).
    Appends a labelled section to r1_report.md and writes r1_posthoc.npz."""
    import hashlib

    st = np.load(run / "r1_stats.npz")
    cfg = yaml.safe_load((data / "config.yaml").read_text())
    taus, capitals = list(cfg["seq_taus"]), [s["capital"].strip() for s in cfg["stimuli"]]
    ki = KS.index(K_MAIN)
    res: dict[str, list] = {k: [] for k in ("frac_pos", "min_proj_rel", "cross_min", "nn_max", "group_size")}
    table = []
    for b, name in enumerate(FILLERS):
        with np.load(data / f"seq_{name}.npz") as z:
            resid = z["resid"][:, :2]
        n_rows, _, nt, n_cells, ns, dm = resid.shape
        n = n_rows * n_cells
        arr = {k: np.full((nt, ns), np.nan) for k in res}
        for t in range(nt):
            for l in range(1, ns):
                d = (normalise(resid[:, OWN, t, :, l]) - normalise(resid[:, WRONG, t, :, l])).reshape(n, dm)
                proj = d @ (d.mean(0) / np.linalg.norm(d.mean(0)))
                arr["frac_pos"][t, l] = (proj > 0).mean()
                arr["min_proj_rel"][t, l] = proj.min() / np.median(np.linalg.norm(d, axis=1))
                _, _, vt = np.linalg.svd(d, full_matrices=False)
                dist = sym_distances(d @ vt[:K_MAIN].T)
                arr["cross_min"][t, l] = dist[:n, n:].min()
                arr["nn_max"][t, l] = (dist[:n, :n] + np.diag(np.full(n, np.inf))).min(1).max()
                if st["gap"][b, t, l, ki] > 0 or not st["h0_tested"][b, t, l]:
                    continue
                lab = fcluster(linkage(squareform(dist, checks=False), method="single"), 3, criterion="maxclust")
                g = np.flatnonzero(lab == 1 + int(np.argmin(np.bincount(lab)[1:])))
                arr["group_size"][t, l] = len(g)
                idx = g % n  # a point and its mirror image are the same δ; δ index = row * n_cells + cell
                cells = sorted({int(i % n_cells) for i in idx})
                rws = sorted({int(i // n_cells) for i in idx})
                table.append(f"| {name} | {taus[t]} | {l} | {len(g)} | {', '.join(capitals[c] for c in cells)} | "
                             f"{len(rws)}: {rws[:5]}{' …' if len(rws) > 5 else ''} | {arr['cross_min'][t, l]:.2f} | "
                             f"{arr['nn_max'][t, l]:.2f} |")
        for k in res:
            res[k].append(arr[k])
        del resid
        log(f"post hoc {name} done", t0)
    out = {k: np.array(v) for k, v in res.items()}
    np.savez(run / "r1_posthoc.npz", taus=np.array(taus), fillers=np.array(FILLERS), **out)

    tested = st["h0_tested"]
    zero = tested & (st["gap"][..., ki] == 0)
    fp, mp = out["frac_pos"][:, :, 1:], out["min_proj_rel"][:, :, 1:]
    avg_ok = [f"{int(((st['agree_average'][b] >= 0.99) & tested[b]).sum())}/{int(tested[b].sum())}" for b in range(2)]
    sgl_ok = [f"{int(((st['agree_single'][b] >= 0.99) & tested[b]).sum())}/{int(tested[b].sum())}" for b in range(2)]
    late = [int((zero[b] & (out["nn_max"][b] > out["cross_min"][b])).sum()) for b in range(2)]
    sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    lines = [f"{POSTHOC_HEADING} (added after the run; not pre-registered; changes no verdict)", "",
             f"Code: `exp5/r1_match.py --posthoc` (file sha256 {sha[:12]}…). The run's `versions.json` hash predates "
             "this mode; the main path is unchanged.", "",
             f"- **The ± halves are linearly separable in every cell.** Over all {fp.size} (filler, τ, l ≥ 1) cells, "
             f"the share of δ's with a positive projection on the mean δ direction is ≥ {fp.min():.4f}. The smallest "
             f"projection is {mp.min():.2f}–{mp.max():.2f} × the median ‖δ‖.",
             f"- **The average-linkage 2-cluster cut equals the ± labels (agreement ≥ 0.99)** at {avg_ok[0]} (common) "
             f"and {avg_ok[1]} (rare) tested cells; single linkage at {sgl_ok[0]} and {sgl_ok[1]}.",
             f"- **What makes the gap 0.** In {int(zero.sum())} tested cells, the cloud's last H0 merge is a set "
             "joining its own mirror image, and that happens twice at the same scale. The table lists the smallest "
             "group of the single-linkage 3-cluster cut. Groups of 100 are one cell across all rows (a cell-specific "
             "offset); groups of 1–8 are outlier δ's from one or two rows. In "
             f"{late[0]} + {late[1]} of these cells, some δ's nearest neighbour is farther away than the closest "
             "point of the opposite half.", "",
             "| filler | τ | l | group size | cells (capital) | rows: first ids | min ± distance | max 1-NN within |",
             "|---|---|---|---|---|---|---|---|", *table, ""]
    report = (run / "r1_report.md").read_text()
    report = report.split(POSTHOC_HEADING)[0].rstrip() + "\n\n"  # replace an earlier post-hoc section, if any
    (run / "r1_report.md").write_text(report + "\n".join(lines))
    log(f"post-hoc section appended to {run / 'r1_report.md'}", t0)


if __name__ == "__main__":
    main()
