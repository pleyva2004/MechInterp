"""Experiment 5, R4: how information flows across layers at the final token T (exp5/PREDICTIONS.md, R4-a / R4-b).

Run:  uv run --offline python -m exp5.r4_flow [--data results/20261025-170021_exp5res] [--boot 100] [--workers 5]
                                               [--results-dir DIR]

Points: per filler type and tau, every stream state x_l(T), l = 1..12, of the own and the wrong sequence of every
(row, cell). Each state is normalised as LayerNorm would without its gain, then capital-centred: minus the mean of its
(row, cell, tau, layer) own/wrong pair, so the pair becomes +delta/2 (own) and -delta/2 (wrong) and the capital's
identity is removed. PCA to 50 dimensions, pooled over layers (the cloud's mean is 0 by construction, so uncentred).
Mapper: lens = layer, windows of w consecutive layers with step s; average linkage inside each window, cut at
m x the window's median pairwise distance; nodes coloured by own share. R4-a / R4-b are scored on the main setting, a
20-setting grid and row-bootstrap resamples; the per-layer 2-cluster average-linkage split is the non-Mapper view.
Operational choices: exp5/PREDICTIONS_R4.md. Writes results/<timestamp>_exp5R4/: predictions copies, versions, the
report r4_report.md, figures, kmapper HTML of the main graphs (local only), r4_results.json and r4_arrays.npz.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # write figures to files, never open a window
import kmapper as km
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import pdist
from sklearn.base import BaseEstimator, ClusterMixin

from exp1.runlog import make_run_dir, save_versions
from exp3.run_relations import PROJECT_ROOT, log
from exp4.tda2_mapper import intervals, layer_cover, layout

DATA = "results/20261025-170021_exp5res"
FILLERS = ("common", "rare")
SEQ_TAUS = (31, 16, 1)  # order of the tau axis in seq_<b>.npz
TAUS = (16, 1)  # scored by R4-a / R4-b
LAYERS = np.arange(1, 13)  # stream states x_1 .. x_12 (x_0 is identical within a pair: the built-in null)
NL = len(LAYERS)
LN_EPS = 1e-5  # GPT-2's LayerNorm epsilon
PCA_K = 50
PCA_CHECK = (20, 100, 768)  # 768 = no reduction
COVERS = ((2, 1), (3, 1), (4, 1), (4, 2))  # (window w layers, step s); every cover tiles layers 1..12 exactly
MS = (0.8, 0.9, 1.0, 1.1, 1.2)
MAIN = (2, 1, 1.0)
SETTINGS = tuple((w, s, m) for w, s in COVERS for m in MS)
MAIN_COVER = tuple((*MAIN[:2], m) for m in MS)
MIN_NODE = 5
PURE_HI, PURE_LO = 0.8, 0.2
N_BOOT, BOOT_SEED = 100, 0
N_PAIR_NULL, NULL_SEED = 1000, 1
GLOBAL_SUB = 4000  # points subsampled for the global-scale rule's median distance (secondary)
OWN_C, WRONG_C = "#eb6834", "#2a78d6"


# ---------------------------------------------------------------- data

def normalise(x: np.ndarray) -> np.ndarray:
    """LayerNorm without its gain and bias, per state (last axis), in float64."""
    x = x.astype(np.float64)
    x = x - x.mean(axis=-1, keepdims=True)
    return x / np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + LN_EPS)


def load_half(data: Path, b: str) -> tuple[dict[int, np.ndarray], float]:
    """tau -> delta / 2 [R, n, 12, 768] float32 (half the normalised own - wrong difference at layers 1..12), and the
    max |own - wrong| of the normalised layer-0 states over all tau (the built-in null, 0 by construction)."""
    resid = np.load(data / f"seq_{b}.npz")["resid"]  # [R, 3 cond, 3 tau, n, 13, 768] fp16
    half, null0 = {}, 0.0
    for ti, tau in enumerate(SEQ_TAUS):
        own, wrong = normalise(resid[:, 0, ti]), normalise(resid[:, 1, ti])  # [R, n, 13, 768]
        null0 = max(null0, float(np.abs(own[:, :, 0] - wrong[:, :, 0]).max()))
        half[tau] = ((own[:, :, 1:] - wrong[:, :, 1:]) / 2).astype(np.float32)
    return half, null0


def cloud(half: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Points ordered (layer, condition, row, cell): +delta/2 for own, -delta/2 for wrong. X [24 R' n, 768], labels."""
    h = half[rows]  # [R', n, 12, 768]
    rp, n = h.shape[:2]
    per_layer = np.moveaxis(h, 2, 0).reshape(NL, rp * n, -1).astype(np.float64)
    X = np.concatenate([per_layer, -per_layer], axis=1).reshape(-1, h.shape[-1])
    pair = np.tile(np.arange(rp * n), 2 * NL)  # resampled (row, cell) index
    lab = {"layer": np.repeat(LAYERS, 2 * rp * n), "own": np.tile(np.repeat([True, False], rp * n), NL),
           "pair": pair, "cell": pair % n, "row": rows[pair // n]}
    return X, lab


def project(X: np.ndarray, own: np.ndarray, k: int) -> np.ndarray:
    """PCA to k dimensions. The cloud is symmetric (each pair is +-delta/2), so its mean is 0 and the second moment of
    the own half is the covariance."""
    if k >= X.shape[1]:
        return X
    xo = X[own]
    _, vec = np.linalg.eigh(xo.T @ xo)
    return X @ vec[:, ::-1][:, :k]


def rescale(X: np.ndarray, layer: np.ndarray) -> np.ndarray:
    """Sensitivity variant: each layer's points divided by the layer's RMS point norm, so that a window mixing two
    layers compares them on one scale."""
    out = X.copy()
    for l in LAYERS:
        sel = layer == l
        out[sel] /= np.sqrt((X[sel] ** 2).sum(axis=1).mean())
    return out


def retained(X: np.ndarray, Y: np.ndarray, layer: np.ndarray) -> np.ndarray:
    """Per layer, the share of the points' squared norm kept by the projection."""
    kept, tot = (Y ** 2).sum(axis=1), (X ** 2).sum(axis=1)
    return np.array([kept[layer == l].sum() / tot[layer == l].sum() for l in LAYERS])


# ---------------------------------------------------------------- Mapper

def cover_windows(w: int, s: int) -> list[tuple[int, ...]]:
    return [tuple(range(1 + j * s, 1 + j * s + w)) for j in range((NL - w) // s + 1)]


def trees(Y: np.ndarray, layer: np.ndarray, windows: set[tuple[int, ...]]) -> dict[tuple[int, ...], tuple]:
    """Per window: (members, average-linkage tree, median pairwise distance). One tree serves every m."""
    out = {}
    for win in sorted(windows):
        mem = np.flatnonzero(np.isin(layer, win))
        D = pdist(Y[mem])
        out[win] = (mem, linkage(D, "average"), float(np.median(D)))
    return out


def cut(tr: dict, windows: list[tuple[int, ...]], m: float, scale: float | None = None) -> dict[str, Any]:
    """Nodes of one Mapper graph: each window's tree cut at m x (its median distance, or a fixed global scale)."""
    nodes, cube = [], []
    for j, win in enumerate(windows):
        mem, Z, med = tr[win]
        lab = fcluster(Z, m * (med if scale is None else scale), "distance")
        order = np.argsort(lab, kind="stable")
        for part in np.split(mem[order], np.flatnonzero(np.diff(lab[order])) + 1):
            nodes.append(part)
            cube.append(j)
    flat_pts = np.concatenate(nodes)
    flat_node = np.repeat(np.arange(len(nodes)), [len(x) for x in nodes])
    return {"nodes": nodes, "cube": np.array(cube), "windows": windows, "flat_pts": flat_pts, "flat_node": flat_node,
            "size": np.bincount(flat_node, minlength=len(nodes))}


def score(g: dict, own: np.ndarray) -> dict[str, Any]:
    """R4-a on one graph: l* = first layer of the earliest window from which every window's nodes with >= 5 points
    are condition-pure (own share >= 0.8 or <= 0.2) through the last window; None if the last window is not clean."""
    size, cube = g["size"], g["cube"]
    share = np.bincount(g["flat_node"], weights=own[g["flat_pts"]].astype(float), minlength=len(size)) / size
    checked = size >= MIN_NODE
    pown, pwrong = checked & (share >= PURE_HI), checked & (share <= PURE_LO)
    mixed = checked & ~pown & ~pwrong
    nw = len(g["windows"])
    clean = np.array([not mixed[cube == j].any() for j in range(nw)])
    lstar = None
    for j in range(nw - 1, -1, -1):
        if not clean[j]:
            break
        lstar = g["windows"][j][0]
    wsum = lambda mask: np.bincount(cube[mask], weights=size[mask], minlength=nw)  # noqa: E731
    tot = wsum(np.ones(len(size), dtype=bool))
    return {"lstar": lstar, "clean": clean, "share": share, "checked": checked, "mixed": mixed,
            "pure_own_pts": wsum(pown) / tot, "pure_wrong_pts": wsum(pwrong) / tot, "mixed_pts": wsum(mixed) / tot,
            "small_pts": wsum(~checked) / tot, "n_nodes": np.bincount(cube, minlength=nw),
            "n_checked": np.bincount(cube[checked], minlength=nw)}


def grid(Y: np.ndarray, lab: dict, settings=SETTINGS, global_rule: bool = False) -> dict[tuple, dict]:
    """Every setting on one cloud. With global_rule, the main cover is also cut at m x one global scale (secondary)."""
    need = {win for w, s, _ in settings for win in cover_windows(w, s)}
    if global_rule:
        need |= set(cover_windows(*MAIN[:2]))
    tr = trees(Y, lab["layer"], need)
    out = {}
    for w, s, m in settings:
        g = cut(tr, cover_windows(w, s), m)
        out[w, s, m] = {"g": g, "r": score(g, lab["own"])}
    if global_rule:
        sub = np.random.default_rng(0).choice(len(Y), min(GLOBAL_SUB, len(Y)), replace=False)
        scale = float(np.median(pdist(Y[sub])))
        for m in MS:
            g = cut(tr, cover_windows(*MAIN[:2]), m, scale)
            out["global", m] = {"g": g, "r": score(g, lab["own"])}
    return out


def edges(g: dict, n_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Upper-triangle node pairs that share points, and the number of shared points."""
    M = csr_matrix((np.ones(len(g["flat_pts"])), (g["flat_node"], g["flat_pts"])), shape=(len(g["nodes"]), n_points))
    S = (M @ M.T).tocoo()
    keep = S.row < S.col
    return np.stack([S.row[keep], S.col[keep]], axis=1), S.data[keep].astype(int)


def r4(lstars: dict[tuple, int | None]) -> tuple[bool, bool]:
    """(R4-a, R4-b) from the l* of the four scored graphs (filler, tau)."""
    a = all(lstars[b, t] is not None for b in FILLERS for t in TAUS)
    return a, bool(a and max(lstars[b, t] for b in FILLERS for t in TAUS) <= 4)


class MedianAverage(BaseEstimator, ClusterMixin):
    """Average linkage cut at m x the median pairwise distance of the points it is given (the R4 scale rule), as a
    clusterer for kmapper."""

    def __init__(self, m: float = 1.0):
        self.m = m

    def fit(self, X, y=None):
        D = pdist(X)
        self.labels_ = fcluster(linkage(D, "average"), self.m * np.median(D), "distance") - 1
        return self


# ---------------------------------------------------------------- non-Mapper view and nulls

def split2(Y: np.ndarray, lab: dict) -> np.ndarray:
    """[12, 2] per layer: purity of the 2-cluster average-linkage split (sum over clusters of the majority count / N)
    and the minority cluster's share of the points."""
    out = np.zeros((NL, 2))
    for i, l in enumerate(LAYERS):
        sel = lab["layer"] == l
        own = lab["own"][sel]
        c = fcluster(linkage(pdist(Y[sel]), "average"), 2, "maxclust")
        sizes = np.array([(c == k).sum() for k in np.unique(c)])
        maj = sum(max((own[c == k]).sum(), (~own[c == k]).sum()) for k in np.unique(c))
        out[i] = maj / sel.sum(), (sizes.min() / sel.sum() if len(sizes) > 1 else 0.0)
    return out


def nulls(g: dict, lab: dict) -> dict[str, Any]:
    """The main graph's nodes with relabelled points: pair-level flips (own <-> wrong within random pairs, N_PAIR_NULL
    draws) and every cell-level flip pattern (all rows of a cell together; cell 0 fixed, so 1023 non-identity ones)."""
    rng = np.random.default_rng(NULL_SEED)
    n_pair, n_cell = lab["pair"].max() + 1, lab["cell"].max() + 1
    res: dict[str, Any] = {}
    for kind in ("pair", "cell"):
        if kind == "pair":
            pats = rng.integers(0, 2, (N_PAIR_NULL, n_pair)).astype(bool)
            key = lab["pair"]
        else:
            pats = ((np.arange(1, 2 ** (n_cell - 1))[:, None] >> np.arange(n_cell - 1)) & 1).astype(bool)
            pats = np.concatenate([np.zeros((len(pats), 1), dtype=bool), pats], axis=1)
            key = lab["cell"]
        ls, pure = [], []
        for f in pats:
            r = score(g, lab["own"] ^ f[key])
            ls.append(r["lstar"])
            pure.append(float(np.mean(r["pure_own_pts"] + r["pure_wrong_pts"])))
        res[kind] = {"n": len(pats), "lstar": ls, "share_defined": float(np.mean([x is not None for x in ls])),
                     "pure_pts_mean": float(np.mean(pure)), "pure_pts_q95": float(np.percentile(pure, 95))}
    return res


def describe_graph(g: dict, r: dict, lab: dict, n_points: int) -> dict[str, Any]:
    """Plain-words numbers for the main graph: per window node counts and point shares, cross-condition edges, and
    whether own and wrong nodes share a connected component after l*."""
    E, w = edges(g, n_points)
    share, checked = r["share"], r["checked"]
    pown, pwrong = checked & (share >= PURE_HI), checked & (share <= PURE_LO)
    a, b = E[:, 0], E[:, 1]
    cross = (pown[a] & pwrong[b]) | (pwrong[a] & pown[b])
    major_cross = checked[a] & checked[b] & ((share[a] > 0.5) != (share[b] > 0.5))
    cells = np.array([len(np.unique(lab["cell"][mem])) for mem in g["nodes"]])
    out: dict[str, Any] = {"n_nodes": len(g["nodes"]), "n_edges": len(E), "cross_pure_edges": int(cross.sum()),
                           "cross_majority_edges": int(major_cross.sum()),
                           "cross_majority_shared_pts": int(w[major_cross].sum())}
    pure_nodes = pown | pwrong
    out["cells_per_pure_node_median"] = float(np.median(cells[pure_nodes])) if pure_nodes.any() else None
    out["pure_nodes_single_cell_share"] = float(np.mean(cells[pure_nodes] == 1)) if pure_nodes.any() else None
    if r["lstar"] is not None:
        js = [j for j, win in enumerate(g["windows"]) if win[0] >= r["lstar"]]
        sel = np.flatnonzero(np.isin(g["cube"], js))
        sub = {int(x): i for i, x in enumerate(sel)}
        A = np.zeros((len(sel), len(sel)), dtype=bool)
        for x, y in E:
            if x in sub and y in sub:
                A[sub[x], sub[y]] = A[sub[y], sub[x]] = True
        nc, comp = connected_components(csr_matrix(A), directed=False)
        both = sum(1 for c in range(nc) if pown[sel[comp == c]].any() and pwrong[sel[comp == c]].any())
        out.update(post_components=int(nc), post_components_with_both=int(both))
    return out


# ---------------------------------------------------------------- bootstrap worker

_HALF: dict[str, dict[int, np.ndarray]] = {}


def _init(data: str) -> None:
    global _HALF
    _HALF = {b: load_half(Path(data), b)[0] for b in FILLERS}


def boot_task(task: tuple[int, str, np.ndarray]) -> tuple[int, str, dict]:
    """One resample of one filler type: l* for every setting at tau 16 and 1, and the per-layer split at every tau."""
    bi, b, rows = task
    out: dict[str, Any] = {}
    for tau in SEQ_TAUS:
        X, lab = cloud(_HALF[b][tau], rows)
        Y = project(X, lab["own"], PCA_K)
        out[f"split_{tau}"] = split2(Y, lab)
        if tau in TAUS:
            res = grid(Y, lab)
            out[f"lstar_{tau}"] = {k: v["r"]["lstar"] for k, v in res.items()}
            res = grid(project(rescale(X, lab["layer"]), lab["own"], PCA_K), lab, settings=MAIN_COVER)
            out[f"lstar_resc_{tau}"] = {k: v["r"]["lstar"] for k, v in res.items()}
    return bi, b, out


# ---------------------------------------------------------------- checks

def check_windows() -> bool:
    """My windows equal the layers inside kmapper's intervals for lens = layer - 1 (exp4.tda2_mapper's cover)."""
    for w, s in COVERS:
        iv = intervals(*layer_cover(w, s))
        kmw = [tuple(int(l) + 1 for l in range(NL) if a <= l <= b) for a, b in iv]
        if kmw != cover_windows(w, s):
            return False
    return True


def check_kmapper(Y: np.ndarray, lab: dict, g: dict) -> tuple[bool, str, dict]:
    n, p, lim = layer_cover(*MAIN[:2])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # limits inside [0, 11] on purpose (half-integer window edges)
        graph = km.KeplerMapper(verbose=0).map((lab["layer"] - 1.0)[:, None], Y, clusterer=MedianAverage(MAIN[2]),
                                               cover=km.Cover(n_cubes=n, perc_overlap=p, limits=[list(lim)]))
    ours = sorted(tuple(sorted(int(x) for x in mem)) for mem in g["nodes"])
    theirs = sorted(tuple(sorted(int(x) for x in mem)) for mem in graph["nodes"].values())
    n_links = sum(len(v) for v in graph["links"].values())
    E, _ = edges(g, len(Y))
    ok = ours == theirs and n_links == len(E)
    return ok, f"{len(theirs)} nodes / {n_links} edges (kmapper) vs {len(ours)} / {len(E)} (ours)", graph


# ---------------------------------------------------------------- figures

def lname(x: int | None) -> str:
    return "—" if x is None else str(x)


def plot_graphs(main: dict, keys: list[tuple[str, int]], path: Path, title: str) -> None:
    ncol = 2
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 5.6 * nrow), squeeze=False)
    norm = TwoSlopeNorm(0.5, 0.0, 1.0)
    for ax, key in zip(axes.flat, keys):
        g, r, lab = main[key]["g"], main[key]["r"], main[key]["lab"]
        E, wts = edges(g, len(lab["layer"]))
        layer_mean = np.array([lab["layer"][mem].mean() for mem in g["nodes"]])
        A = np.zeros((len(g["nodes"]),) * 2, dtype=bool)
        A[E[:, 0], E[:, 1]] = True
        pos = layout({"nodes": g["nodes"], "A": A | A.T}, layer_mean)
        xy = np.array([pos[i] for i in range(len(g["nodes"]))])
        for (a, b), k in zip(E, wts):
            ax.plot(*xy[[a, b]].T, color="0.7", lw=0.3 + 1.5 * np.sqrt(k / wts.max()), zorder=1)
        chk = r["checked"]
        sizes = 4 + 600 * np.sqrt(g["size"] / g["size"].max())
        ax.scatter(xy[~chk, 0], xy[~chk, 1], s=8, facecolors="none", edgecolors="0.55", linewidths=0.5, zorder=2)
        sc = ax.scatter(xy[chk, 0], xy[chk, 1], s=sizes[chk], c=r["share"][chk], cmap="RdBu_r", norm=norm,
                        edgecolors="0.2", linewidths=0.5, zorder=3)
        if r["lstar"] is not None:
            ax.axvline(r["lstar"] - 0.5, color="0.3", ls="--", lw=0.9)
        b, tau = key
        ax.set_title(f"{b} filler, τ = {tau} (d = {32 - tau}): {len(g['nodes'])} nodes, {len(E)} edges, "
                     f"l* = {lname(r['lstar'])}", fontsize=10)
        ax.set_xticks(LAYERS)
        ax.set_xlim(0.5, 12.5)
        ax.set_xlabel("node mean layer (the lens; state x_l before layer l, x_12 after layer 11)", fontsize=8)
        ax.set_yticks([])
        ax.spines[["top", "right", "left"]].set_visible(False)
    for ax in axes.flat[len(keys):]:
        ax.axis("off")
    cb = fig.colorbar(sc, ax=axes, fraction=0.02, pad=0.01)
    cb.set_label("own share of the node's points (red = own, blue = wrong)", fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_windows(main: dict, keys: list[tuple[str, int]], path: Path) -> None:
    fig, axes = plt.subplots(len(SEQ_TAUS), len(FILLERS), figsize=(12, 3.0 * len(SEQ_TAUS)), squeeze=False,
                             sharey=True)
    for key in keys:
        b, tau = key
        ax = axes[SEQ_TAUS.index(tau), FILLERS.index(b)]
        r, wins = main[key]["r"], main[key]["g"]["windows"]
        x = np.arange(len(wins))
        bottom = np.zeros(len(wins))
        for k, col, name in (("pure_own_pts", OWN_C, "pure own nodes"), ("pure_wrong_pts", WRONG_C, "pure wrong nodes"),
                             ("mixed_pts", "#8a8a8a", "mixed nodes (≥ 5 points)"), ("small_pts", "#dddddd",
                                                                                     "nodes < 5 points")):
            ax.bar(x, r[k], bottom=bottom, color=col, width=0.8, label=name)
            bottom += r[k]
        ax.set_xticks(x)
        ax.set_xticklabels([f"{w[0]}–{w[-1]}" for w in wins], fontsize=7)
        if r["lstar"] is not None:
            ax.axvline(x[[w[0] for w in wins].index(r["lstar"])] - 0.5, color="k", ls="--", lw=0.9)
        ax.set_title(f"{b}, τ = {tau}: l* = {lname(r['lstar'])}", fontsize=9)
        ax.set_ylim(0, 1)
    axes[0, 0].legend(fontsize=7, frameon=False, ncol=2, loc="lower left")
    for ax in axes[:, 0]:
        ax.set_ylabel("share of the window's points", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("window (layers), main setting w = 2, s = 1, m = 1", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_split(split_full: dict, split_boot: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, len(FILLERS), figsize=(12, 4), sharey=True)
    cols = {31: "#1baf7a", 16: OWN_C, 1: WRONG_C}
    for ax, b in zip(axes, FILLERS):
        for tau in SEQ_TAUS:
            full = split_full[b, tau][:, 0]
            ax.plot(LAYERS, full, color=cols[tau], marker="o", ms=3, lw=1.5, label=f"τ = {tau}")
            if (b, tau) in split_boot:
                lo, hi = np.percentile(split_boot[b, tau][:, :, 0], [2.5, 97.5], axis=0)
                ax.fill_between(LAYERS, lo, hi, color=cols[tau], alpha=0.2, lw=0)
        ax.axhline(0.5, color="0.6", lw=0.7)
        ax.set_xticks(LAYERS)
        ax.set_xlabel("layer (state x_l)", fontsize=8)
        ax.set_title(f"{b} filler: purity of the per-layer 2-cluster average-linkage split", fontsize=9)
    axes[0].set_ylabel("purity (0.5 = no condition signal, 1 = own | wrong)", fontsize=8)
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_stability(full_l: dict, boot_l: dict, n_boot: int, path: Path) -> None:
    """Left: l* per grid setting and scored graph on the full data, with the bootstrap share where R4-a holds.
    Right: bootstrap distribution of l* at the main setting."""
    keys = [(b, t) for t in TAUS for b in FILLERS]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 7.5), gridspec_kw={"width_ratios": [1.1, 1]})
    mat = np.array([[np.nan if full_l[s][k] is None else full_l[s][k] for k in keys] for s in SETTINGS])
    ax.imshow(mat, cmap="viridis_r", vmin=1, vmax=12, aspect="auto")
    for (i, j), v in np.ndenumerate(mat):
        ax.text(j, i, "—" if np.isnan(v) else f"{int(v)}", ha="center", va="center", fontsize=7,
                color="white" if not np.isnan(v) and v >= 7 else "black")
    for i, s in enumerate(SETTINGS):
        sa = np.mean([r4({k: boot_l[k][bi][s] for k in keys})[0] for bi in range(n_boot)])
        ax.text(len(keys) - 0.3, i, f"{sa:.2f}", ha="left", va="center", fontsize=7)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([f"{b}\nτ={t}" for b, t in keys], fontsize=8)
    ax.set_yticks(range(len(SETTINGS)))
    ax.set_yticklabels([f"w={w} s={s} m={m}" + (" (main)" if (w, s, m) == MAIN else "") for w, s, m in SETTINGS],
                       fontsize=7)
    ax.set_xlim(-0.5, len(keys) + 0.6)
    ax.set_title(f"l* on the full data (— = R4-a fails for that graph);\nright column: share of {n_boot} bootstrap "
                 "resamples where R4-a holds", fontsize=9)
    vals = list(range(1, 13)) + [None]
    width = 0.8 / len(keys)
    for j, k in enumerate(keys):
        ls = [boot_l[k][bi][MAIN] for bi in range(n_boot)]
        cnt = [sum(1 for x in ls if x == v) / n_boot for v in vals]
        ax2.bar(np.arange(len(vals)) + (j - (len(keys) - 1) / 2) * width, cnt, width=width, label=f"{k[0]}, τ={k[1]}")
    ax2.set_xticks(range(len(vals)))
    ax2.set_xticklabels([str(v) for v in vals[:-1]] + ["none"], fontsize=8)
    ax2.set_xlabel("l* (first layer of the earliest clean window); none = R4-a fails", fontsize=8)
    ax2.set_ylabel("share of bootstrap resamples", fontsize=8)
    ax2.set_title(f"Bootstrap distribution of l*, main setting (w=2, s=1, m=1), {n_boot} resamples", fontsize=9)
    ax2.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- report

def stable(outcome: bool, s_grid: float, s_main_boot: float) -> str:
    """The main-setting outcome is 'stable' if it recurs in >= 70% of grid settings and of main-setting resamples."""
    agree_g = s_grid if outcome else 1 - s_grid
    agree_b = s_main_boot if outcome else 1 - s_main_boot
    return "stable" if min(agree_g, agree_b) >= 0.7 else "unstable"


def dist_str(ls: list) -> str:
    vals = sorted({x for x in ls if x is not None})
    parts = [f"{v}: {sum(1 for x in ls if x == v)}" for v in vals]
    nn = sum(1 for x in ls if x is None)
    return ", ".join(parts + ([f"none: {nn}"] if nn else []))


def write_report(run: Path, ctx: dict) -> None:
    v, n_boot = ctx["verdicts"], ctx["n_boot"]
    keys = [(b, t) for t in TAUS for b in FILLERS]
    L = ["# Experiment 5, R4: own and wrong across layers (Mapper on the capital-centred stream at T)", "",
         f"Run `{run.name}`. Data `{ctx['data']}` ({ctx['R']} rows per filler type). Pre-registration: "
         "`PREDICTIONS.md` (R4-a, R4-b) and the operational addendum `PREDICTIONS_R4.md` (both copied here; the "
         "addendum was written before this script touched the full data). Code `exp5/r4_flow.py`.", "",
         "**Points.** Per filler type and τ: x_l(T), l = 1..12, of the own and wrong sequence of each (row, cell), "
         "LayerNorm-normalised without gain, then minus the mean of its own/wrong pair (each pair becomes ±δ/2). "
         f"{2 * ctx['R'] * 11} points per layer, {24 * ctx['R'] * 11} per cloud. PCA to {PCA_K} (pooled over layers, "
         "uncentred because the cloud's mean is 0). **Mapper:** lens = layer; windows of w layers with step s; "
         "average linkage per window cut at ε = m × the window's median pairwise distance. Main setting w = 2, s = 1, "
         "m = 1 (11 windows: layers 1–2, 2–3, …, 11–12). A node with ≥ 5 points is pure if its own share is ≥ 0.8 "
         "or ≤ 0.2. l* = the first layer of the earliest window from which every window is clean through the last "
         "window (layers 11–12).", "",
         "## Checks", "", "| check | result |", "|---|---|"]
    L += [f"| {k} | {x} |" for k, x in ctx["checks"].items()]
    L += ["", "## Verdicts", "",
          "R4-a holds in a graph set when all four scored graphs (common/rare × τ = 16/1) have an l*; R4-b when also "
          "max l* ≤ 4. Verdict = main setting on the full data. s_grid = share of the 20 grid settings (4 covers × "
          f"5 m) where it holds on the full data; s_boot = share of the {n_boot} resamples × 20 settings; main boot = "
          f"share of the {n_boot} resamples at the main setting. 'Stable' = the main outcome recurs in ≥ 70% of grid "
          "settings and of main-setting resamples.", "",
          "| id | main setting (full data) | s_grid | s_boot | main boot | verdict |", "|---|---|---|---|---|---|"]
    for h in ("R4-a", "R4-b"):
        x = v[h]
        L.append(f"| {h} | {'holds' if x['main'] else 'fails'} | {x['s_grid']:.2f} | {x['s_boot']:.2f} | "
                 f"{x['main_boot']:.2f} | **{'PASS' if x['main'] else 'FAIL'}** ({x['stability']}) |")
    L += ["", "**l\\* per scored graph (main setting, full data) and its bootstrap distribution (main setting):**", "",
          "| graph | l* (full) | bootstrap l* counts | grid l* over 20 settings |", "|---|---|---|---|"]
    for k in keys + [(b, 31) for b in FILLERS]:
        boot = dist_str(ctx["boot_l"][k]) if k in ctx["boot_l"] else "not bootstrapped (descriptive τ)"
        L.append(f"| {k[0]}, τ = {k[1]} | {lname(ctx['main'][k]['r']['lstar'])} | {boot} | "
                 f"{dist_str([ctx['full_l'][s][k] for s in SETTINGS])} |")
    L += ["", "## Main graphs: windows (main setting, full data)", "",
          "Per window: nodes (nodes with ≥ 5 points), and the share of the window's points in pure-own / pure-wrong / "
          "mixed nodes / nodes under 5 points.", ""]
    for k in keys + [(b, 31) for b in FILLERS]:
        r, wins = ctx["main"][k]["r"], ctx["main"][k]["g"]["windows"]
        L += [f"**{k[0]}, τ = {k[1]}** (l* = {lname(r['lstar'])})", "",
              "| window | nodes (≥5) | pure own | pure wrong | mixed | small |", "|---|---|---|---|---|---|"]
        for j, win in enumerate(wins):
            L.append(f"| {win[0]}–{win[-1]} | {r['n_nodes'][j]} ({r['n_checked'][j]}) | {r['pure_own_pts'][j]:.3f} | "
                     f"{r['pure_wrong_pts'][j]:.3f} | {r['mixed_pts'][j]:.3f} | {r['small_pts'][j]:.3f} |")
        d = ctx["desc"][k]
        L += ["", f"- {d['n_nodes']} nodes, {d['n_edges']} edges. Edges joining a pure-own and a pure-wrong node: "
              f"{d['cross_pure_edges']}; edges joining an own-majority and a wrong-majority node (both ≥ 5 points): "
              f"{d['cross_majority_edges']} ({d['cross_majority_shared_pts']} shared points)."]
        if "post_components" in d:
            L.append(f"- From l* on: {d['post_components']} connected components, {d['post_components_with_both']} "
                     "holding both a pure-own and a pure-wrong node.")
        if d["cells_per_pure_node_median"] is not None:
            L.append(f"- Pure nodes hold a median of {d['cells_per_pure_node_median']:.0f} of the 11 cells; "
                     f"{d['pure_nodes_single_cell_share']:.2f} of pure nodes hold a single cell.")
        L.append("")
    L += ["## Robustness: per-layer 2-cluster average-linkage split (no Mapper)", "",
          "Purity = Σ over the two clusters of the majority-condition count / N (0.5 = no condition signal). "
          f"Bootstrap 95% interval over {n_boot} resamples in brackets; 'min' = the smaller cluster's share of the "
          "layer's points (a 0.99/0.01 split has purity near 0.5 whatever the labels).", "",
          "| layer | " + " | ".join(f"{b} τ={t}" for b in FILLERS for t in SEQ_TAUS) + " |",
          "|---" * (1 + len(FILLERS) * len(SEQ_TAUS)) + "|"]
    for i, l in enumerate(LAYERS):
        cells = []
        for b in FILLERS:
            for t in SEQ_TAUS:
                p, mn = ctx["split_full"][b, t][i]
                lo, hi = np.percentile(ctx["split_boot"][b, t][:, i, 0], [2.5, 97.5])
                cells.append(f"{p:.2f} [{lo:.2f}, {hi:.2f}] min {mn:.2f}")
        L.append(f"| {l} | " + " | ".join(cells) + " |")
    L += ["", "## Sensitivity (full data, descriptive)", "",
          "l* per scored graph; R4-a / R4-b on the four.", "",
          "| variant | " + " | ".join(f"{b} τ={t}" for b, t in keys) + " | R4-a | R4-b |", "|---" * (len(keys) + 3) + "|"]
    for name, ls in ctx["sens"]:
        a, bb = r4(ls)
        L.append(f"| {name} | " + " | ".join(lname(ls[k]) for k in keys) + f" | {'holds' if a else 'fails'} | "
                 f"{'holds' if bb else 'fails'} |")
    rs = ctx["resc"]
    L += ["", "**Per-layer rescaled variant** (each layer's points divided by the layer's RMS point norm before PCA, "
          "so a window compares its two layers on one scale; pre-declared sensitivity, not the verdict):", "",
          f"- Main setting: R4-a {'holds' if rs['main'][0] else 'fails'}, R4-b {'holds' if rs['main'][1] else 'fails'}; "
          "l* " + ", ".join(f"{b} τ={t} {lname(rs['lstar_main'][b, t])}" for b, t in keys + [(b, 31) for b in FILLERS])
          + ".",
          f"- s_grid (20 settings, full data): R4-a {rs['s_grid'][0]:.2f}, R4-b {rs['s_grid'][1]:.2f}.",
          f"- Bootstrap ({n_boot} resamples, main cover): "
          + "; ".join(f"{m}: R4-a {x[0]:.2f}, R4-b {x[1]:.2f}" for m, x in rs["boot"].items()) + ".",
          "- Bootstrap l* at the main setting: " + "; ".join(f"{b} τ={t}: {dist_str(rs['lstar_boot_main'][b, t])}"
                                                           for b, t in keys) + ".",
          "- Mixed-point share per window, main setting: " + "; ".join(
              f"{b} τ={t}: " + " ".join(f"{x:.2f}" for x in rs["windows_main"][b, t]["mixed_pts"])
              for b, t in keys) + ".",
          "", "Grid l* (rescaled, full data):", "",
          "| setting | " + " | ".join(f"{b} τ={t}" for b, t in keys) + " | R4-a | R4-b |", "|---" * (len(keys) + 3) + "|"]
    for s in SETTINGS:
        ls = {k: ctx["resc_l"][s][k] for k in keys}
        a, bb = r4(ls)
        L.append(f"| w={s[0]} s={s[1]} m={s[2]} | " + " | ".join(lname(ls[k]) for k in keys)
                 + f" | {'holds' if a else 'fails'} | {'holds' if bb else 'fails'} |")
    L += ["", "PCA energy kept per layer (share of the points' squared norm; main k = 50):", "",
          "| graph | " + " | ".join(str(l) for l in LAYERS) + " |", "|---" * (NL + 1) + "|"]
    for k, e in ctx["energy"].items():
        L.append(f"| {k[0]} τ={k[1]} k={k[2]} | " + " | ".join(f"{x:.2f}" for x in e) + " |")
    L += ["", "## Label nulls on the main graphs (descriptive; geometry fixed, labels permuted)", "",
          "Pair flip: own ↔ wrong swapped within random (row, cell) pairs, consistently across layers "
          f"({N_PAIR_NULL} draws). Cell flip: own ↔ wrong swapped for all rows of a set of cells (all 1023 "
          "non-identity patterns with cell 0 fixed). If cell flips also leave nodes pure, purity reflects "
          "(cell, condition) groups rather than a direction shared by all capitals.", "",
          "| graph | real l* | real pure-point share | pair flip: share with an l* / mean pure share | cell flip: share "
          "with an l* / mean pure share |", "|---|---|---|---|---|"]
    for k, nl in ctx["nulls"].items():
        r = ctx["main"][k]["r"]
        L.append(f"| {k[0]}, τ = {k[1]} | {lname(r['lstar'])} | "
                 f"{np.mean(r['pure_own_pts'] + r['pure_wrong_pts']):.3f} | {nl['pair']['share_defined']:.3f} / "
                 f"{nl['pair']['pure_pts_mean']:.3f} | {nl['cell']['share_defined']:.3f} / "
                 f"{nl['cell']['pure_pts_mean']:.3f} |")
    L += ["", "## Caveats", "",
          "- The same 11 country–capital pairs throughout, and the derangement is fixed: a result means consistent "
          "across rows and these pairs, not general. The bootstrap resamples rows only.",
          "- δ = own − wrong changes two things at τ: the match and the country's identity (country k vs the "
          "country whose deranged capital is k). Purity alone cannot say which one separates the points; the "
          "cell-flip null and the cells-per-node numbers above address it.",
          "- Mapper depends on the cover and the clustering scale. The verdict is the main setting; the 20-setting "
          "grid, the global-scale rule, the PCA dimension and the bootstrap show how much it moves. Smaller m "
          "gives finer, more often pure nodes.",
          "- R4-a is strict: one mixed node of ≥ 5 points in any later window breaks it. The point shares per window "
          "show how close a failing graph is.",
          "- Windows overlap, so l* is defined on window starts; with w = 2 a clean window starting at l covers "
          "layers l and l + 1.",
          "- Layer l means the stream state x_l: before block l (after blocks 0..l−1); x_12 is after block 11, "
          "before ln_final.",
          "- The graph layout (spring layout, x = node mean layer) is for display only.", "",
          "## Files", "",
          "- `figures/mapper_main.png`: main graphs, τ = 16 and 1, both fillers (colour = own share, size = points).",
          "- `figures/mapper_tau31.png`: the same for τ = 31 (descriptive).",
          "- `figures/windows.png`: per-window point shares in pure-own / pure-wrong / mixed / small nodes.",
          "- `figures/split2.png`: per-layer 2-cluster purity with bootstrap bands.",
          "- `figures/stability.png`: l* over the grid and its bootstrap distribution.",
          "- `mapper_main_<filler>_tau<τ>.html`: kmapper graphs of the main setting (local only).",
          "- `r4_results.json`: verdicts, l* per setting/graph, window shares, nulls, descriptions.",
          "- `r4_arrays.npz`: bootstrap l* arrays (setting × resample, NaN = none) and 2-cluster purities.", ""]
    (run / "r4_report.md").write_text("\n".join(L))


# ---------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Exp 5 R4: Mapper of own vs wrong across layers")
    parser.add_argument("--data", default=DATA, type=Path, help="exp5 extraction folder (must contain DONE)")
    parser.add_argument("--boot", default=N_BOOT, type=int, help="bootstrap resamples (fewer for a smoke test)")
    parser.add_argument("--workers", default=5, type=int, help="processes for the bootstrap")
    parser.add_argument("--results-dir", default=None, type=Path, help="override the results folder (smoke tests)")
    args = parser.parse_args(argv)
    t0 = time.time()
    data = args.data if args.data.is_absolute() else PROJECT_ROOT / args.data
    if not (data / "DONE").exists():
        raise SystemExit(f"{data} has no DONE file: the extraction is not finished")
    run = make_run_dir(args.results_dir or PROJECT_ROOT / "results", tag="exp5R4")
    for name in ("PREDICTIONS.md", "PREDICTIONS_R4.md"):
        shutil.copy2(PROJECT_ROOT / "exp5" / name, run / name)
    save_versions(run, PROJECT_ROOT, code_dirs=("exp1", "exp2", "exp3", "exp4", "exp5"))
    (run / "figures").mkdir()
    log(f"run folder {run}", t0)

    half, null0 = {}, {}
    for b in FILLERS:
        half[b], null0[b] = load_half(data, b)
    R = next(iter(half["common"].values())).shape[0]
    log(f"loaded {R} rows per filler", t0)

    # Bootstrap in worker processes while the main process does the full-data analysis.
    rng = np.random.default_rng(BOOT_SEED)
    picks = [{b: rng.integers(0, R, R) for b in FILLERS} for _ in range(args.boot)]
    pool = ProcessPoolExecutor(max_workers=args.workers, initializer=_init, initargs=(str(data),))
    futures = [pool.submit(boot_task, (bi, b, picks[bi][b])) for bi in range(args.boot) for b in FILLERS]

    checks: dict[str, str] = {}
    checks["built-in null: normalised own vs wrong x_0(T) identical"] = (
        f"{'PASS' if max(null0.values()) < 1e-5 else 'FAIL'} (max |Δ| "
        + ", ".join(f"{b} {x:.1e}" for b, x in null0.items()) + ")")
    checks["cover windows = kmapper intervals (lens = layer − 1)"] = "PASS" if check_windows() else "FAIL"

    rows_all = np.arange(R)
    main_g: dict[tuple, dict] = {}
    full_l: dict[tuple, dict] = {s: {} for s in SETTINGS}
    split_full, energy, sens_rows = {}, {}, {}
    resc_l: dict[tuple, dict] = {s: {} for s in SETTINGS}
    resc_main: dict[tuple, dict] = {}
    for b in FILLERS:
        for tau in SEQ_TAUS:
            X, lab = cloud(half[b][tau], rows_all)
            sym = float(np.abs(X[lab["own"]] + X[~lab["own"]]).max())
            Y = project(X, lab["own"], PCA_K)
            energy[b, tau, PCA_K] = retained(X, Y, lab["layer"])
            split_full[b, tau] = split2(Y, lab)
            res = grid(Y, lab, global_rule=True)
            for s in SETTINGS:
                full_l[s][b, tau] = res[s]["r"]["lstar"]
            main_g[b, tau] = {"g": res[MAIN]["g"], "r": res[MAIN]["r"], "lab": lab, "Y": Y}
            for k in PCA_CHECK:
                Yk = project(X, lab["own"], k)
                if k < X.shape[1]:
                    energy[b, tau, k] = retained(X, Yk, lab["layer"])
                rk = grid(Yk, lab, settings=MAIN_COVER)
                for m in MS:
                    sens_rows.setdefault((f"PCA {k}", m), {})[b, tau] = rk[(*MAIN[:2], m)]["r"]["lstar"]
            rr = grid(project(rescale(X, lab["layer"]), lab["own"], PCA_K), lab)
            for s in SETTINGS:
                resc_l[s][b, tau] = rr[s]["r"]["lstar"]
            resc_main[b, tau] = rr[MAIN]["r"]
            for m in MS:
                sens_rows.setdefault(("global scale", m), {})[b, tau] = res["global", m]["r"]["lstar"]
            checks[f"cloud symmetric (own = −wrong), {b} τ={tau}"] = f"{'PASS' if sym == 0 else 'FAIL'} ({sym:.1e})"
            log(f"full data {b} tau={tau}: main l* = {lname(res[MAIN]['r']['lstar'])}", t0)

    km_graphs = {}
    for b in FILLERS:
        for tau in TAUS:
            k_ok, k_msg, km_graphs[b, tau] = check_kmapper(main_g[b, tau]["Y"], main_g[b, tau]["lab"],
                                                           main_g[b, tau]["g"])
            checks[f"hand-rolled Mapper = kmapper 2.1.0 (main setting), {b} τ={tau}"] = (
                f"{'PASS' if k_ok else 'FAIL'} ({k_msg})")
    log("kmapper checks done", t0)

    desc = {k: describe_graph(v["g"], v["r"], v["lab"], len(v["Y"])) for k, v in main_g.items()}
    null_res = {(b, t): nulls(main_g[b, t]["g"], main_g[b, t]["lab"]) for b in FILLERS for t in TAUS}
    log("nulls and descriptions done", t0)

    boot_l: dict[tuple, list] = {(b, t): [None] * args.boot for b in FILLERS for t in TAUS}
    boot_resc: dict[tuple, list] = {(b, t): [None] * args.boot for b in FILLERS for t in TAUS}
    split_boot = {(b, t): np.zeros((args.boot, NL, 2)) for b in FILLERS for t in SEQ_TAUS}
    for i, fut in enumerate(futures):
        bi, b, out = fut.result()
        for t in TAUS:
            boot_l[b, t][bi] = out[f"lstar_{t}"]
            boot_resc[b, t][bi] = out[f"lstar_resc_{t}"]
        for t in SEQ_TAUS:
            split_boot[b, t][bi] = out[f"split_{t}"]
        if (i + 1) % 20 == 0:
            log(f"bootstrap task {i + 1}/{len(futures)}", t0)
    pool.shutdown()

    keys = [(b, t) for t in TAUS for b in FILLERS]
    verdicts: dict[str, dict] = {}
    for hi, h in enumerate(("R4-a", "R4-b")):
        main_hold = r4({k: full_l[MAIN][k] for k in keys})[hi]
        s_grid = float(np.mean([r4({k: full_l[s][k] for k in keys})[hi] for s in SETTINGS]))
        per = np.array([[r4({k: boot_l[k][bi][s] for k in keys})[hi] for s in SETTINGS] for bi in range(args.boot)])
        main_boot = float(per[:, SETTINGS.index(MAIN)].mean())
        verdicts[h] = {"main": bool(main_hold), "s_grid": s_grid, "s_boot": float(per.mean()), "main_boot": main_boot,
                       "per_setting_boot": dict(zip([f"w{w}s{s}m{m}" for w, s, m in SETTINGS], per.mean(0).tolist())),
                       "stability": stable(main_hold, s_grid, main_boot)}
    boot_main_l = {k: [boot_l[k][bi][MAIN] for bi in range(args.boot)] for k in keys}
    resc = {"main": [bool(x) for x in r4({k: resc_l[MAIN][k] for k in keys})],
            "s_grid": [float(np.mean([r4({k: resc_l[s][k] for k in keys})[i] for s in SETTINGS])) for i in (0, 1)],
            "boot": {f"m={m}": [float(np.mean([r4({k: boot_resc[k][bi][(*MAIN[:2], m)] for k in keys})[i]
                                               for bi in range(args.boot)])) for i in (0, 1)] for m in MS},
            "lstar_main": {k: resc_l[MAIN][k] for k in resc_l[MAIN]},
            "lstar_boot_main": {k: [boot_resc[k][bi][MAIN] for bi in range(args.boot)] for k in keys},
            "windows_main": resc_main}

    sens = [(f"{name}, m = {m}" + (" (main m)" if m == MAIN[2] else ""), ls) for (name, m), ls in sens_rows.items()]
    main_keys_all = keys + [(b, 31) for b in FILLERS]
    plot_graphs(main_g, keys, run / "figures" / "mapper_main.png",
                "R4 Mapper graphs, main setting (lens = layer, windows of 2 layers, step 1; average linkage cut at the "
                "window's median distance). Node size ∝ √points; hollow = < 5 points; dashed line = l*")
    plot_graphs(main_g, [(b, 31) for b in FILLERS], run / "figures" / "mapper_tau31.png",
                "R4 Mapper graphs, τ = 31 (descriptive), main setting")
    plot_windows(main_g, main_keys_all, run / "figures" / "windows.png")
    plot_split(split_full, split_boot, run / "figures" / "split2.png")
    plot_stability(full_l, boot_l, args.boot, run / "figures" / "stability.png")
    for (b, tau), graph in km_graphs.items():
        lab = main_g[b, tau]["lab"]
        tips = np.array([f"row {r} cell {c} {'own' if o else 'wrong'} x_{l}"
                         for r, c, o, l in zip(lab["row"], lab["cell"], lab["own"], lab["layer"])])
        km.KeplerMapper(verbose=0).visualize(
            graph, path_html=str(run / f"mapper_main_{b}_tau{tau}.html"),
            title=f"Exp 5 R4: own vs wrong across layers, {b} filler, tau {tau} (w 2, s 1, m 1)",
            color_values=np.column_stack([lab["own"].astype(float), lab["layer"]]),
            color_function_name=["own share", "layer"], custom_tooltips=tips)
    log("figures and HTML written", t0)

    def jl(x):
        return None if x is None else int(x)

    results = {
        "verdicts": verdicts, "R": R, "n_boot": args.boot, "main_setting": dict(zip(("w", "s", "m"), MAIN)),
        "lstar_main_full": {f"{b}_tau{t}": jl(main_g[b, t]["r"]["lstar"]) for b, t in main_keys_all},
        "lstar_grid_full": {f"w{w}s{s}m{m}": {f"{b}_tau{t}": jl(full_l[(w, s, m)][b, t]) for b, t in main_keys_all}
                            for w, s, m in SETTINGS},
        "lstar_boot_main": {f"{b}_tau{t}": [jl(x) for x in v] for (b, t), v in boot_main_l.items()},
        "sensitivity": {name: {f"{b}_tau{t}": jl(x) for (b, t), x in ls.items()} for name, ls in sens},
        "windows_main": {f"{b}_tau{t}": {k: np.asarray(main_g[b, t]["r"][k]).tolist()
                                         for k in ("pure_own_pts", "pure_wrong_pts", "mixed_pts", "small_pts",
                                                   "n_nodes", "n_checked")} for b, t in main_keys_all},
        "describe_main": {f"{b}_tau{t}": d for (b, t), d in desc.items()},
        "nulls_main": {f"{b}_tau{t}": {kind: {k: v for k, v in x.items() if k != "lstar"} for kind, x in nl.items()}
                       for (b, t), nl in null_res.items()},
        "split2_full": {f"{b}_tau{t}": v.tolist() for (b, t), v in split_full.items()},
        "rescaled_variant": {
            "main": resc["main"], "s_grid": resc["s_grid"], "main_cover_boot": resc["boot"],
            "lstar_main_full": {f"{b}_tau{t}": jl(x) for (b, t), x in resc["lstar_main"].items()},
            "lstar_grid_full": {f"w{w}s{s}m{m}": {f"{b}_tau{t}": jl(resc_l[(w, s, m)][b, t]) for b, t in main_keys_all}
                                for w, s, m in SETTINGS},
            "lstar_boot_main": {f"{b}_tau{t}": [jl(x) for x in v] for (b, t), v in resc["lstar_boot_main"].items()},
            "mixed_pts_main": {f"{b}_tau{t}": r["mixed_pts"].tolist() for (b, t), r in resc_main.items()}},
        "checks": checks}
    (run / "r4_results.json").write_text(json.dumps(results, indent=1))
    np.savez(run / "r4_arrays.npz",
             settings=np.array([f"w{w}s{s}m{m}" for w, s, m in SETTINGS]),
             **{f"boot_lstar_{b}_tau{t}": np.array([[np.nan if boot_l[b, t][bi][s] is None else boot_l[b, t][bi][s]
                                                     for bi in range(args.boot)] for s in SETTINGS])
                for b in FILLERS for t in TAUS},
             **{f"split2_boot_{b}_tau{t}": v for (b, t), v in split_boot.items()},
             **{f"split2_full_{b}_tau{t}": v for (b, t), v in split_full.items()})
    write_report(run, {"verdicts": verdicts, "n_boot": args.boot, "data": str(args.data), "R": R, "checks": checks,
                       "main": main_g, "full_l": full_l, "boot_l": boot_main_l, "desc": desc, "nulls": null_res,
                       "split_full": split_full, "split_boot": split_boot, "sens": sens, "energy": energy,
                       "resc": resc, "resc_l": resc_l})
    log(f"done: {run / 'r4_report.md'}", t0)
    for k, x in checks.items():
        print(f"{k}: {x}")
    for h, x in verdicts.items():
        print(f"{h}: main {x['main']}  s_grid {x['s_grid']:.2f}  s_boot {x['s_boot']:.2f}  "
              f"main_boot {x['main_boot']:.2f}  ({x['stability']})")
    return run


if __name__ == "__main__":
    main()
