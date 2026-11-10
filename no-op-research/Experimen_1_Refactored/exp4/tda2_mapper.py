"""TDA 2: a Mapper graph of attention-head behaviour across depth, from the saved Exp 4 / 4b arrays (no model).

Run:  uv run --offline python -m exp4.tda2_mapper

One point per head: the country arm's D(d), d = 1..31, with its capital, a wrong capital and a random final token, for
both filler types (6 curves, 186 numbers), mapped to shape with shrinkage f = v / (|v| + lambda). Lens = layer index,
covered by uniform windows of w layers with step s; single-linkage clustering inside each window. The pre-registered
hypotheses (exp4/PREDICTIONS_tda2.md) are scored on the main setting, a 21-setting grid and 200 row-bootstrap
resamples. Writes results/<timestamp>_exp4tda2/: predictions copy, versions, the main graph (figure, kmapper HTML,
JSON), stability tables and tda2_report.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # write figures to files, never open a window
import kmapper as km
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.cluster import DBSCAN

from exp1.runlog import make_run_dir, save_versions
from exp3.run_relations import PROJECT_ROOT, log
from exp4.run_controlled import ARMS, LABELS, head_name, row_differences

CTRL_NPZ = "results/20261024-000548_exp4ctrl/controlled.npz"  # random final token (Exp 4)
CAP_NPZ = "results/20261024-150407_exp4cap/capital.npz"  # its capital / a wrong capital (Exp 4b)
T, NL, NH = 32, 12, 12
FILLERS = ("common", "rare")
CONDS = ("own", "wrong", "random")
MARGIN = 0.02
LAMBDA = MARGIN * np.sqrt(T - 1)  # a margin-sized effect sustained over one whole curve
KAPPA = 0.005
NEAR, FAR = (2, 4), (16, 31)
COVERS = ((2, 1), (3, 1), (4, 1), (5, 1), (4, 2), (6, 2), (6, 3))  # (window w layers, step s)
EPS_MULT = (0.75, 1.0, 1.5)
MAIN = (4, 2, 1.0)
KNN = 3
N_BOOT, BOOT_SEED = 200, 0
S_SAME = ((0, 2), (1, 11), (3, 0), (4, 10))
S_PREV = ((1, 0), (2, 2), (2, 3), (2, 5), (2, 8), (2, 9), (3, 2), (3, 6), (3, 7), (3, 8), (4, 11))
S_LATE = ((9, 8), (10, 3), (11, 7), (11, 2))
HYPS = ("M1a", "M1b", "M1c", "M2", "M3", "M4")
HYP_TEXT = {"M1a": "near-only and any-distance heads in one connected structure",
            "M1b": "no depth window splits local and global heads into separate nodes",
            "M1c": "local end in layers 2–3 and global end in 9–11",
            "M2": "same-country heads (L0H2, L1H11, L3H0, L4H10) form a distinct branch",
            "M3": "Exp 1 previous-token heads cluster together",
            "M4": "late any-distance heads (L9H8, L10H3, L11H7, L11H2) form their own branch"}
SET_COLORS = {"same": "#1baf7a", "prev": "#eb6834", "late": "#2a78d6"}
LAYER = np.repeat(np.arange(NL), NH).astype(float)
NAMES = np.array([head_name(l, h) for l in range(NL) for h in range(NH)])


def idx(heads: tuple[tuple[int, int], ...]) -> np.ndarray:
    return np.array([l * NH + h for l, h in heads])


# ---------------------------------------------------------------- data

def load(t0: float) -> tuple[dict, dict]:
    """Per-row country contrasts rows[(cond, b)] [R, 31, 144] and the saved stats saved[(cond, b)]."""
    zc, zk = np.load(PROJECT_ROOT / CTRL_NPZ), np.load(PROJECT_ROOT / CAP_NPZ)
    rows, saved = {}, {}
    for b in FILLERS:
        for cond in CONDS:
            z, pre = (zc, b) if cond == "random" else (zk, f"{cond}/{b}")
            ext = {"control": z[f"{pre}/control"], **{arm: {"masses": z[f"{pre}/{arm}_masses"]} for arm in ARMS}}
            x = row_differences(ext, T)["country"]
            rows[cond, b] = x.reshape(x.shape[0], T - 1, -1)
            saved[cond, b] = {"mean": z[f"{pre}/country/mean"].reshape(T - 1, -1),
                              "sig": z[f"{pre}/country/sig"].reshape(T - 1, -1),
                              "labels": z[f"{pre}/labels"].reshape(-1)}
        log(f"{b}: per-row contrasts rebuilt", t0)
    return rows, saved


def curves(rows: dict, pick: dict[str, np.ndarray] | None = None) -> dict:
    """D curves [31, 144] per (cond, b): the mean over rows, or over a bootstrap draw (same rows for every cond)."""
    return {k: (x if pick is None else x[pick[k[1]]]).mean(axis=0) for k, x in rows.items()}


def raw_features(cv: dict, per_filler: bool = False) -> np.ndarray:
    """[144, 186] (6 curves per head), or [288, 93] with one point per head and filler type."""
    blocks = [np.concatenate([cv[c, b].T for c in CONDS], axis=1) for b in FILLERS]
    return np.concatenate(blocks, axis=0) if per_filler else np.concatenate(blocks, axis=1)


def shape(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + LAMBDA)


def locality(v: np.ndarray) -> np.ndarray:
    """F = far / (near + far + kappa) from [..., n_curves * 31] curves; near/far = mean over curves of |mean D|."""
    c = v.reshape(v.shape[:-1] + (-1, T - 1))
    near = np.abs(c[..., NEAR[0] - 1:NEAR[1]].mean(-1)).mean(-1)
    far = np.abs(c[..., FAR[0] - 1:FAR[1]].mean(-1)).mean(-1)
    return far / (near + far + KAPPA)


def eps_base(dist: np.ndarray, resp: np.ndarray) -> float:
    """Median over responsive points of the distance to the 3rd-nearest other point (column 0 is the point itself)."""
    return float(np.median(np.sort(dist, axis=1)[resp, KNN]))


# ---------------------------------------------------------------- Mapper

def layer_cover(w: int, s: int) -> tuple[int, float, tuple[float, float]]:
    """kmapper (n_cubes, perc_overlap, limits) whose windows are exactly layers js .. js + w - 1."""
    n = (NL - w) // s + 1
    lo = (w - 1 - s) / 2
    return n, 1 - s / w, (lo, lo + n * s)


def intervals(n: int, p: float, limits: tuple[float, float]) -> np.ndarray:
    """[n, 2] closed intervals, kmapper 2.1.0's Cover formula."""
    lo, hi = limits
    rng = hi - lo
    r = rng / (2 * n * (1 - p))
    c = np.linspace(lo + rng / (2 * n), hi - rng / (2 * n), n)
    return np.stack([c - r, c + r], axis=1)


def mapper(dist: np.ndarray, lens: np.ndarray, iv: np.ndarray, eps: float) -> dict[str, Any]:
    """Single linkage cut at eps (= DBSCAN with min_samples 1) inside each interval; nodes joined if they share a point."""
    nodes, cube = [], []
    for j, (a, b) in enumerate(iv):
        members = np.flatnonzero((lens >= a) & (lens <= b))
        if len(members) == 0:
            continue
        if len(members) == 1:
            lab = np.ones(1, dtype=int)
        else:
            lab = fcluster(linkage(squareform(dist[np.ix_(members, members)], checks=False), "single"), eps, "distance")
        for k in np.unique(lab):
            nodes.append(members[lab == k])
            cube.append(j)
    M = np.zeros((len(nodes), dist.shape[0]), dtype=bool)
    for i, mem in enumerate(nodes):
        M[i, mem] = True
    A = (M.astype(np.int32) @ M.T.astype(np.int32)) > 0
    np.fill_diagonal(A, False)
    _, comp = connected_components(csr_matrix(A), directed=False)
    head_comp = np.empty(dist.shape[0], dtype=int)
    for i, mem in enumerate(nodes):
        head_comp[mem] = comp[i]  # all nodes holding a point share one component
    return {"nodes": nodes, "cube": np.array(cube), "M": M, "A": A, "comp": comp, "head_comp": head_comp}


# ---------------------------------------------------------------- hypotheses

def co_pur(g: dict, S: np.ndarray) -> tuple[float, float]:
    """co: share of same-window S pairs that share a node; pur: S share of the memberships of nodes holding S."""
    M, cube = g["M"], g["cube"]
    pairs = shared = 0
    for j in np.unique(cube):
        sub = M[cube == j][:, S]  # each point is in exactly one node of its window
        per_node = sub.sum(axis=1)
        k = int(per_node.sum())
        pairs += k * (k - 1) // 2
        shared += int((per_node * (per_node - 1) // 2).sum())
    hold = M[:, S].any(axis=1)
    return (shared / pairs if pairs else np.nan), float(M[hold][:, S].sum() / M[hold].sum())


def evaluate(g: dict, v: np.ndarray, layer: np.ndarray, sets: dict[str, np.ndarray]) -> dict[str, Any]:
    M, comp, cube, hc = g["M"], g["comp"], g["cube"], g["head_comp"]
    L, G = sets["L"], sets["G"]
    LG = np.union1d(L, G)
    comps, counts = np.unique(hc[LG], return_counts=True)
    main = comps[np.argmax(counts)]
    out: dict[str, Any] = {"main_share": counts.max() / len(LG)}
    out["M1a"] = bool(out["main_share"] >= 0.8 and (hc[L] == main).any() and (hc[G] == main).any())
    hasL, hasG = M[:, L].any(axis=1), M[:, G].any(axis=1)
    both = split = 0
    for j in np.unique(cube):
        s = cube == j
        if hasL[s].any() and hasG[s].any():
            both += 1
            split += int((hasL[s] & ~hasG[s]).any() and (hasG[s] & ~hasL[s]).any())
    out.update(split=split, both=both, M1b=split == 0)
    sel = np.flatnonzero((comp == main) & (hasL | hasG))
    out["M1c"], out["ends"] = False, (np.nan, np.nan)
    if len(sel) >= 3:
        Ms = M[sel].astype(float)
        F = locality(Ms @ v / Ms.sum(axis=1, keepdims=True))
        node_layer = Ms @ layer / Ms.sum(axis=1)
        order = np.argsort(F, kind="stable")
        k = len(sel) // 3
        ends = (float(node_layer[order[:k]].mean()), float(node_layer[order[-k:]].mean()))
        out["ends"] = ends
        out["M1c"] = bool(2 <= ends[0] < 4 and ends[1] >= 9)
    for key, hyp in (("same", "M2"), ("prev", "M3"), ("late", "M4")):
        co, pur = co_pur(g, sets[key])
        out[f"{key}_co"], out[f"{key}_pur"] = co, pur
        out[hyp] = bool(co >= 0.8 and pur >= 0.5)  # undefined co (nan) fails
    body = np.argmax(np.bincount(hc))
    out["same_in_body"] = float((hc[sets["same"]] == body).mean())
    out["body_share"] = float(np.bincount(hc).max() / len(hc))
    out["n_nodes"], out["n_edges"], out["n_comp"] = len(g["nodes"]), int(g["A"].sum() // 2), int(comp.max() + 1)
    return out


def head_sets(saved: dict, per_filler: bool = False) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Fixed head sets from the full-data Exp 4 labels, and the responsive flag (indices doubled for 288 points)."""
    lab = {b: saved["random", b]["labels"] for b in FILLERS}
    glob = np.logical_or.reduce([lab[b] == LABELS[0] for b in FILLERS])
    loc = np.logical_or.reduce([lab[b] == LABELS[1] for b in FILLERS]) & ~glob
    sets = {"L": np.flatnonzero(loc), "G": np.flatnonzero(glob), "same": idx(S_SAME), "prev": idx(S_PREV),
            "late": idx(S_LATE)}
    resp_b = {b: np.logical_or.reduce([saved[c, b]["sig"].any(axis=0) for c in CONDS]) for b in FILLERS}
    if not per_filler:
        return sets, resp_b["common"] | resp_b["rare"]
    n = NL * NH
    return {k: np.concatenate([s, s + n]) for k, s in sets.items()}, np.concatenate([resp_b[b] for b in FILLERS])


def grid_graphs(f: np.ndarray, v: np.ndarray, lens: np.ndarray | None, layer: np.ndarray, sets: dict,
                resp: np.ndarray) -> list[dict]:
    """All 21 settings on one point cloud. lens None = the depth lens (layer windows); else a continuous lens."""
    dist = squareform(pdist(f))
    e0 = eps_base(dist, resp)
    out = []
    for w, s in COVERS:
        n, p, lim = layer_cover(w, s)
        iv = intervals(n, p, lim) if lens is None else intervals(n, p, (float(lens.min()), float(lens.max())))
        for m in EPS_MULT:
            g = mapper(dist, layer if lens is None else lens, iv, m * e0)
            r = evaluate(g, v, layer, sets)
            r.update(w=w, s=s, n=n, p=p, m=m, eps=m * e0)
            if lens is not None:  # do all late heads fall in an end window?
                late = lens[sets["late"]]
                r["late_end"] = bool(all(iv[0, 0] <= x <= iv[0, 1] for x in late)
                                     or all(iv[-1, 0] <= x <= iv[-1, 1] for x in late))
            out.append(r)
    return out


SWEEP_M = np.round(np.arange(0.5, 2.501, 0.05), 2)


def eps_sweep(f: np.ndarray, v: np.ndarray, sets: dict, resp: np.ndarray) -> list[dict]:
    """Exploratory (not pre-registered): the main cover at a fine range of m, full data. The grid's m values sit on
    either side of the scale where the graph goes from mostly singletons to one blob per window."""
    dist = squareform(pdist(f))
    e0 = eps_base(dist, resp)
    iv = intervals(*layer_cover(*MAIN[:2]))
    out = []
    for m in SWEEP_M:
        r = evaluate(mapper(dist, LAYER, iv, m * e0), v, LAYER, sets)
        r["m"] = float(m)
        out.append(r)
    return out


def plot_sweep(sweep: list[dict], path: Path) -> None:
    m = [r["m"] for r in sweep]
    fig, axes = plt.subplots(3, 1, figsize=(8, 8.5), sharex=True, gridspec_kw={"height_ratios": [1, 1.2, 0.8]})
    ax = axes[0]
    ax.plot(m, [r["n_comp"] for r in sweep], color="0.2", lw=1.5, label="connected components")
    ax.set_ylabel("components", fontsize=8)
    ax2 = ax.twinx()
    ax2.plot(m, [r["body_share"] for r in sweep], color=SET_COLORS["late"], lw=1.5)
    ax2.set_ylabel("share of heads in the largest component", fontsize=8, color=SET_COLORS["late"])
    for k in EPS_MULT:
        ax.axvline(k, color="0.8", lw=0.8, ls="--")
    ax.set_title("Exploratory ε sweep, main cover (windows of 4 layers, step 2), full data; dashed = grid m values",
                 fontsize=9)
    ax = axes[1]
    for key, lab in (("same", "same-country"), ("prev", "previous-token"), ("late", "late any-distance")):
        ax.plot(m, [r[f"{key}_co"] for r in sweep], color=SET_COLORS[key], lw=1.5, label=f"{lab}: co")
        ax.plot(m, [r[f"{key}_pur"] for r in sweep], color=SET_COLORS[key], lw=1.2, ls=":", label=f"{lab}: pur")
    ax.axhline(0.8, color="0.6", lw=0.7)
    ax.axhline(0.5, color="0.6", lw=0.7, ls=":")
    ax.set_ylabel("co (solid, needs ≥ 0.8) / pur (dotted, ≥ 0.5)", fontsize=8)
    ax.legend(fontsize=6.5, frameon=False, ncol=3, loc="center right")
    ax = axes[2]
    ax.imshow(np.array([[r[h] for r in sweep] for h in HYPS], dtype=float), aspect="auto", cmap="Greens", vmin=0,
              vmax=1.3, extent=(m[0] - 0.025, m[-1] + 0.025, len(HYPS) - 0.5, -0.5))
    ax.set_yticks(range(len(HYPS)))
    ax.set_yticklabels(HYPS, fontsize=8)
    ax.set_xlabel("m (ε = m · ε0)", fontsize=8)
    ax.set_title("green = hypothesis holds", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def ranges(ms: list[float]) -> str:
    """[0.5, 0.55, 0.6, 1.0] -> '0.50–0.60, 1.00' (consecutive sweep steps merged)."""
    if not ms:
        return "never"
    out, start, prev = [], ms[0], ms[0]
    for x in ms[1:] + [None]:
        if x is not None and abs(x - prev - 0.05) < 1e-9:
            prev = x
            continue
        out.append(f"{start:.2f}" if start == prev else f"{start:.2f}–{prev:.2f}")
        if x is not None:
            start = prev = x
    return ", ".join(out)


# ---------------------------------------------------------------- checks

def check_curves(cv: dict, saved: dict) -> dict[str, float]:
    return {f"{c}/{b}": float(np.abs(cv[c, b] - saved[c, b]["mean"]).max()) for c, b in cv}


def check_kmapper(f: np.ndarray, g: dict, eps: float) -> tuple[bool, str, dict]:
    n, p, lim = layer_cover(*MAIN[:2])
    with warnings.catch_warnings():
        # the limits sit inside [0, 11] on purpose, so that window edges fall on half-integers
        warnings.simplefilter("ignore", UserWarning)
        graph = km.KeplerMapper(verbose=0).map(LAYER[:, None], f, clusterer=DBSCAN(eps=eps, min_samples=1),
                                               cover=km.Cover(n_cubes=n, perc_overlap=p, limits=[list(lim)]))
    ours = sorted(tuple(sorted(int(x) for x in mem)) for mem in g["nodes"])
    theirs = sorted(tuple(sorted(int(x) for x in mem)) for mem in graph["nodes"].values())
    n_links = sum(len(v) for v in graph["links"].values())
    ok = ours == theirs and n_links == int(g["A"].sum() // 2)
    return ok, f"{len(theirs)} nodes / {n_links} edges (kmapper) vs {len(ours)} / {int(g['A'].sum() // 2)} (ours)", graph


def check_windows() -> bool:
    layers = np.arange(NL)
    for w, s in COVERS:
        iv = intervals(*layer_cover(w, s))
        sets = [set(layers[(layers >= a) & (layers <= b)]) for a, b in iv]
        if any(len(x) != w for x in sets) or any(len(x & y) != w - s for x, y in zip(sets, sets[1:])):
            return False
        if set().union(*sets) != set(layers):
            return False
    return True


# ---------------------------------------------------------------- outputs

def node_attrs(g: dict, v: np.ndarray, per_head: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    Mf = g["M"].astype(float)
    cnt = Mf.sum(axis=1)
    out = {k: Mf @ x / cnt for k, x in per_head.items()}
    out["locality"] = locality(Mf @ v / cnt[:, None])  # F of the node's mean curves, as in M1c
    out["count"] = cnt
    return out


def layout(g: dict, layer_mean: np.ndarray) -> dict[int, tuple[float, float]]:
    """x = node mean layer (the lens); y from a spring layout, so branches at the same depth sit apart."""
    G = nx.Graph()
    G.add_nodes_from(range(len(g["nodes"])))
    G.add_edges_from(zip(*np.nonzero(np.triu(g["A"]))))
    rng = np.random.default_rng(0)
    init = {i: (layer_mean[i] / 11, rng.uniform()) for i in G}
    pos = nx.spring_layout(G, pos=init, seed=0, k=0.6 / np.sqrt(len(G)), iterations=300)
    return {i: (float(layer_mean[i]), float(pos[i][1])) for i in G}


def plot_graph(g: dict, attrs: dict, path: Path, title: str) -> None:
    pos = layout(g, attrs["layer"])
    named = {"same": idx(S_SAME), "prev": idx(S_PREV), "late": idx(S_LATE)}
    labels: dict[int, list[tuple[str, str]]] = {}
    for key, heads in named.items():
        for h in heads:
            cand = [i for i, mem in enumerate(g["nodes"]) if h in mem]
            i = min(cand, key=lambda c: (len(g["nodes"][c]), c))  # the most specific node holding the head
            labels.setdefault(i, []).append((NAMES[h], SET_COLORS[key]))
    panels = [("layer", "mean layer of members", "viridis", None),
              ("fact", "fact contrast: its capital − wrong capital (mean over d, fillers)", "RdBu_r", "div"),
              ("locality", "locality F = far / (near + far + κ) of the node's mean curves", "magma", None),
              ("size", "response size: mean over curves of max |D(d)|", "YlOrBr", None)]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12.5))
    edges = list(zip(*np.nonzero(np.triu(g["A"]))))
    xy = np.array([pos[i] for i in range(len(g["nodes"]))])
    sizes = 12 + 9 * attrs["count"]
    for ax, (key, name, cmap, kind) in zip(axes.flat, panels):
        for a, b in edges:
            ax.plot(*xy[[a, b]].T, color="0.75", lw=0.6, zorder=1)
        val = attrs[key]
        norm = TwoSlopeNorm(0.0, -np.abs(val).max(), np.abs(val).max()) if kind == "div" else None
        sc = ax.scatter(xy[:, 0], xy[:, 1], s=sizes, c=val, cmap=cmap, norm=norm, edgecolors="0.25", linewidths=0.4,
                        zorder=2)
        fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.01)
        for i, items in labels.items():
            for k, (txt, col) in enumerate(items):
                ax.annotate(txt, xy[i], xytext=(5, 4 + 7 * k), textcoords="offset points", fontsize=6.5, color=col,
                            fontweight="bold", zorder=3)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("node mean layer (the lens)", fontsize=8)
        ax.set_xticks(range(NL))
        ax.set_yticks([])
        ax.spines[["top", "right", "left"]].set_visible(False)
    handles = [plt.Line2D([], [], color=c, marker="s", ls="", label=lab) for lab, c in
               (("same-country heads", SET_COLORS["same"]), ("Exp 1 previous-token heads", SET_COLORS["prev"]),
                ("late any-distance heads", SET_COLORS["late"]))]
    fig.legend(handles=handles, fontsize=8, frameon=False, loc="lower center", ncol=3)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0.025, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_stability(grid_full: list[dict], boot_share: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 8))
    ax.imshow(boot_share, cmap="Greens", vmin=0, vmax=1, aspect="auto")
    for (r, c), x in np.ndenumerate(boot_share):
        full = grid_full[r][HYPS[c]]
        ax.text(c, r, f"{x:.2f}{'*' if full else ''}", ha="center", va="center", fontsize=7,
                color="white" if x > 0.6 else "black")
    ax.set_xticks(range(len(HYPS)))
    ax.set_xticklabels(HYPS)
    ax.set_yticks(range(len(grid_full)))
    ax.set_yticklabels([f"w={r['w']} s={r['s']} m={r['m']}" for r in grid_full], fontsize=7)
    ax.set_title("share of 200 bootstrap graphs where the hypothesis holds\n(* = holds on the full data)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def verdict(s_grid: float, s_boot: float) -> str:
    if min(s_grid, s_boot) >= 0.7:
        return "PASS"
    return "FAIL" if max(s_grid, s_boot) < 0.3 else "UNSTABLE"


def q(x: np.ndarray) -> str:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if not len(x):
        return "undefined"
    return f"{np.median(x):.2f} [{np.percentile(x, 2.5):.2f}, {np.percentile(x, 97.5):.2f}]"


def describe(g: dict, attrs: dict) -> list[str]:
    """Auto description of the main graph: components, nodes per window, and where the named heads sit."""
    hc = g["head_comp"]
    sizes = np.bincount(hc)
    L = [f"- {len(g['nodes'])} nodes, {int(g['A'].sum() // 2)} edges, {len(sizes)} connected components; heads per "
         f"component (largest first): {', '.join(str(x) for x in sorted(sizes, reverse=True))}."]
    for j in np.unique(g["cube"]):
        cnt = sorted((int(attrs["count"][i]) for i in np.flatnonzero(g["cube"] == j)), reverse=True)
        L.append(f"- window {j} (layers {MAIN[1] * j}–{MAIN[1] * j + MAIN[0] - 1}): {len(cnt)} nodes, sizes {cnt}")
    for name, heads in (("same-country", S_SAME), ("previous-token", S_PREV), ("late any-distance", S_LATE)):
        parts = []
        for h in idx(heads):
            mems = [g["nodes"][i] for i in range(len(g["nodes"])) if h in g["nodes"][i]]
            small = min(mems, key=len)
            others = [NAMES[x] for x in small if x != h]
            tail = ", ".join(others[:6]) + (f" +{len(others) - 6}" if len(others) > 6 else "")
            parts.append(f"{NAMES[h]} (comp {hc[h]}, smallest node {len(small)}: {tail or 'alone'})")
        L.append(f"- {name}: " + "; ".join(parts))
    return L


def write_report(run: Path, ctx: dict) -> None:
    v, main, grid, boot = ctx["verdicts"], ctx["main"], ctx["grid"], ctx["boot"]
    L = ["# TDA 2: Mapper graph of head behaviour across depth", "",
         f"Run `{run.name}`. Pre-registration: `PREDICTIONS_tda2.md` (copied from `exp4/PREDICTIONS_tda2.md`, written "
         "before any graph). Data: `controlled.npz` (Exp 4, random final token) and `capital.npz` (Exp 4b), no model.",
         "",
         "Point cloud: 144 heads × 186 features (country-arm D(d), d = 1..31, with its capital / a wrong capital / a "
         f"random final token, common and rare filler), f = v / (‖v‖ + {LAMBDA:.3f}). Depth lens = layer index, "
         "windows of w layers with step s; single-linkage clustering per window cut at ε = m · ε0 (ε0 = median 3-NN "
         "distance of responsive heads). Main setting w = 4, s = 2, m = 1. Grid: 7 covers × 3 m = 21 settings; "
         f"bootstrap {ctx['n_boot']} row resamples × 21 settings.", "",
         f"Head sets (fixed): 𝓛 (near-only) {len(ctx['sets']['L'])} heads, 𝓖 (any-distance) {len(ctx['sets']['G'])} "
         f"heads, responsive {int(ctx['resp'].sum())} of 144. ε0 (full data) = {ctx['eps0']:.3f}.", "",
         "## Checks", "", "| check | result |", "|---|---|"]
    L += [f"| {k} | {x} |" for k, x in ctx["checks"].items()]
    L += ["", "## Pre-registered verdicts", "",
          "s_grid = share of the 21 settings where it holds on the full data; s_boot = share of the "
          f"{ctx['n_boot']} × 21 bootstrap graphs; main boot = share of the {ctx['n_boot']} bootstrap graphs at the main "
          "setting. PASS: min(s_grid, s_boot) ≥ 0.70; FAIL: max < 0.30; otherwise UNSTABLE.", "",
          "| id | hypothesis | main setting | s_grid | s_boot | main boot | verdict |", "|---|---|---|---|---|---|---|"]
    for h in HYPS:
        L.append(f"| {h} | {HYP_TEXT[h]} | {'holds' if main[h] else 'fails'} | {v[h]['s_grid']:.2f} | "
                 f"{v[h]['s_boot']:.2f} | {v[h]['main_boot']:.2f} | **{v[h]['verdict']}** |")
    mb = ctx["main_boot_rows"]
    L += ["", "## Numbers behind the verdicts (main setting; bootstrap median [2.5%, 97.5%] over the main-setting "
          "resamples)", "",
          f"- M1a: main component holds {main['main_share']:.2f} of 𝓛 ∪ 𝓖 heads; bootstrap "
          f"{q([r['main_share'] for r in mb])}.",
          f"- M1b: split windows {main['split']} of {main['both']} windows holding both 𝓛 and 𝓖 heads; bootstrap "
          f"split count {q([r['split'] for r in mb])}.",
          f"- M1c: layer of the lowest-F third {main['ends'][0]:.2f}, of the highest-F third {main['ends'][1]:.2f} "
          f"(needs [2, 4) and ≥ 9); bootstrap {q([r['ends'][0] for r in mb])} and {q([r['ends'][1] for r in mb])}."]
    for key, hyp in (("same", "M2"), ("prev", "M3"), ("late", "M4")):
        L.append(f"- {hyp}: co {main[f'{key}_co']:.2f}, pur {main[f'{key}_pur']:.2f} (needs ≥ 0.8 and ≥ 0.5); "
                 f"bootstrap co {q([r[f'{key}_co'] for r in mb])}, pur {q([r[f'{key}_pur'] for r in mb])}.")
    L.append(f"- Same-country heads in the component with the most heads (flare rather than separate component): "
             f"{main['same_in_body']:.2f} of the four (main setting); share over the 21 settings "
             f"{np.mean([r['same_in_body'] for r in grid]):.2f}.")
    L += ["", "## Stability by setting (s_boot per setting; * = holds on the full data)", "",
          "| w | s | m | ε | nodes | edges | components | " + " | ".join(HYPS) + " |", "|---" * (7 + len(HYPS)) + "|"]
    for r, row in zip(grid, ctx["boot_share"]):
        L.append(f"| {r['w']} | {r['s']} | {r['m']} | {r['eps']:.3f} | {r['n_nodes']} | {r['n_edges']} | {r['n_comp']} | "
                 + " | ".join(f"{x:.2f}{'*' if r[h] else ''}" for h, x in zip(HYPS, row)) + " |")
    L += ["", "## Secondary analyses (descriptive, full data, share of the 21 settings where it holds)", "",
          "| analysis | " + " | ".join(HYPS) + " | other |", "|---" * (len(HYPS) + 2) + "|"]
    for name, rows_, depth_only in ctx["secondary"]:
        cells = [f"{np.mean([r[h] for r in rows_]):.2f}" if (h in ("M1a", "M2", "M3", "M4") or not depth_only)
                 else "n/a" for h in HYPS]
        other = (f"late heads all in an end window: {np.mean([r['late_end'] for r in rows_]):.2f}"
                 if "late_end" in rows_[0] else "")
        L.append(f"| {name} | " + " | ".join(cells) + f" | {other} |")
    sw, pc = ctx["sweep"], ctx["pc1"]
    L += ["", "## Exploratory, not pre-registered", "",
          "**ε sweep** (main cover, full data, m = 0.50..2.50 in steps of 0.05; `figures/eps_sweep.png`). The grid's "
          "m values straddle the scale at which the graph goes from mostly singletons to one blob per window, so this "
          "asks whether any scale in between gives the predicted structure.", "",
          "| m | components | largest component (share of heads) | same co / pur | prev co / pur | late co / pur |",
          "|---|---|---|---|---|---|"]
    for r in sw:
        if any(abs(r["m"] - x) < 1e-9 for x in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5)):
            L.append(f"| {r['m']:.2f} | {r['n_comp']} | {r['body_share']:.2f} | "
                     + " | ".join(f"{r[f'{k}_co']:.2f} / {r[f'{k}_pur']:.2f}" for k in ("same", "prev", "late")) + " |")
    L += [""] + [f"- {h} holds at m = {ranges([r['m'] for r in sw if r[h]])}." for h in HYPS]
    L += ["", f"**PC1 of the features** explains {pc['var']:.2f} of the variance. Spearman with layer "
          f"{pc['rho_layer']:+.2f}, with locality F {pc['rho_F']:+.2f}. The four late any-distance heads rank "
          f"{', '.join(pc['late_ranks'])} of 144 from the {pc['end']} end; the 8 heads at that end: {pc['top']}."]
    L += ["", "**Nearest heads in feature space** (full data, Euclidean on f; fact = mean own − wrong):"]
    L += [f"- {NAMES[h]}: " + ", ".join(f"{NAMES[j]} {d:.2f} (fact {ctx['fact'][j]:+.3f})" for j, d in nn)
          for h, nn in ctx["nn"]]
    L += ["", "## Main graph (auto description)", ""] + ctx["describe"]
    L += ["", "## Caveats", "",
          "- Mapper is sensitive to the cover and the clustering scale; the verdicts use the pre-declared grid and the "
          "row bootstrap, but a different feature transform or clustering rule could change the picture (see the "
          "secondary rows).",
          "- 144 points. For the 4-head sets a single head moving between nodes flips co and pur.",
          "- The bootstrap resamples rows (the stimulus noise); it does not resample heads, countries or the 11 fixed "
          "country–capital pairs.",
          "- The head sets 𝓛, 𝓖 and the named heads come from earlier analyses of the same data, so the graph tests "
          "how their response shapes are arranged, not which heads respond.",
          "- Locality F is ill-defined for heads with no response (κ pulls them toward 0); read the locality panel "
          "together with the response-size panel.",
          "- The depth lens is discrete (12 values); the windows are aligned to whole layers by construction.", "",
          "## Files", "",
          "- `figures/mapper_main.png`: main graph, 4 colourings, node size = member count, named heads labelled.",
          "- `figures/stability.png`: s_boot per setting and hypothesis.",
          "- `figures/eps_sweep.png`: exploratory ε sweep at the main cover.",
          "- `mapper_main.html`: kmapper interactive graph of the main setting (local only).",
          "- `main_graph.json`: nodes (window, members), edges and node attributes of the main graph.",
          "- `stability.csv`: one row per analysis and setting, with full-data holds, bootstrap shares and graph sizes.",
          "- `verdicts.json`.", ""]
    (run / "tda2_report.md").write_text("\n".join(L))


# ---------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="TDA 2: Mapper graph of head behaviour across depth")
    parser.add_argument("--boot", default=N_BOOT, type=int, help="bootstrap resamples (fewer for a smoke test)")
    parser.add_argument("--results-dir", default=None, type=Path, help="override the results folder (smoke tests)")
    args = parser.parse_args(argv)
    t0 = time.time()
    run = make_run_dir(args.results_dir or PROJECT_ROOT / "results", tag="exp4tda2")
    shutil.copy2(PROJECT_ROOT / "exp4" / "PREDICTIONS_tda2.md", run / "PREDICTIONS_tda2.md")
    save_versions(run, PROJECT_ROOT, code_dirs=("exp1", "exp2", "exp3", "exp4"))
    (run / "figures").mkdir()
    log(f"run folder {run}", t0)

    rows, saved = load(t0)
    sets, resp = head_sets(saved)
    cv = curves(rows)
    v = raw_features(cv)
    f = shape(v)
    checks: dict[str, str] = {}
    dev = check_curves(cv, saved)
    checks["recomputed D curves vs saved means (tol 1e-3)"] = (
        f"{'PASS' if max(dev.values()) < 1e-3 else 'FAIL'} (max |Δ| {max(dev.values()):.1e}; "
        + ", ".join(f"{k} {x:.1e}" for k, x in dev.items()) + ")")
    checks["depth windows hold w layers and share w − s"] = "PASS" if check_windows() else "FAIL"

    grid = grid_graphs(f, v, None, LAYER, sets, resp)
    mi = [(r["w"], r["s"], r["m"]) for r in grid].index(MAIN)
    main_r = grid[mi]
    dist = squareform(pdist(f))
    eps0 = eps_base(dist, resp)
    n, p, lim = layer_cover(*MAIN[:2])
    g_main = mapper(dist, LAYER, intervals(n, p, lim), MAIN[2] * eps0)
    ok, msg, km_graph = check_kmapper(f, g_main, MAIN[2] * eps0)
    checks["hand-rolled Mapper = kmapper 2.1.0 (main setting)"] = f"{'PASS' if ok else 'FAIL'} ({msg})"
    log("full-data grid done", t0)

    rng = np.random.default_rng(BOOT_SEED)
    R = next(iter(rows.values())).shape[0]
    boot = np.zeros((args.boot, len(grid), len(HYPS)), dtype=bool)
    main_boot_rows = []
    for bi in range(args.boot):
        pick = {b: rng.integers(0, R, R) for b in FILLERS}
        vb = raw_features(curves(rows, pick))
        res = grid_graphs(shape(vb), vb, None, LAYER, sets, resp)
        boot[bi] = [[r[h] for h in HYPS] for r in res]
        main_boot_rows.append(res[mi])
        if (bi + 1) % 25 == 0:
            log(f"bootstrap {bi + 1}/{args.boot}", t0)
    boot_share = boot.mean(axis=0)  # [settings, hyps]

    verdicts = {}
    for k, h in enumerate(HYPS):
        s_grid = float(np.mean([r[h] for r in grid]))
        s_boot = float(boot[:, :, k].mean())
        verdicts[h] = {"main": bool(main_r[h]), "s_grid": s_grid, "s_boot": s_boot,
                       "main_boot": float(boot[:, mi, k].mean()), "verdict": verdict(s_grid, s_boot)}

    # Secondary, descriptive: other lenses, two points per head, asinh features.
    loc_lens = locality(v)
    centred = f - f.mean(axis=0)
    pc1 = centred @ np.linalg.svd(centred, full_matrices=False)[2][0]
    sets288, resp288 = head_sets(saved, per_filler=True)
    v288 = raw_features(cv, per_filler=True)
    layer288 = np.concatenate([LAYER, LAYER])
    secondary = [("locality lens", grid_graphs(f, v, loc_lens, LAYER, sets, resp), True),
                 ("PC1 lens", grid_graphs(f, v, pc1, LAYER, sets, resp), True),
                 ("288 points (head × filler), depth lens", grid_graphs(shape(v288), v288, None, layer288, sets288,
                                                                        resp288), False),
                 ("asinh(D / 0.02) features, depth lens", grid_graphs(np.arcsinh(v / MARGIN), v, None, LAYER, sets,
                                                                      resp), False)]
    log("secondary analyses done", t0)
    sweep = eps_sweep(f, v, sets, resp)
    plot_sweep(sweep, run / "figures" / "eps_sweep.png")
    sv = np.linalg.svd(centred, compute_uv=False)
    rank_asc = np.argsort(np.argsort(pc1)) + 1
    high = rank_asc[sets["late"]].mean() > len(pc1) / 2
    ranks = len(pc1) + 1 - rank_asc if high else rank_asc
    pc_info = {"var": float(sv[0] ** 2 / (sv ** 2).sum()), "rho_layer": float(spearmanr(pc1, LAYER)[0]),
               "rho_F": float(spearmanr(pc1, loc_lens)[0]), "end": "high" if high else "low",
               "late_ranks": [f"{NAMES[h]} {int(ranks[h])}" for h in sets["late"]],
               "top": ", ".join(NAMES[np.argsort(ranks)[:8]])}
    log("exploratory sweep done", t0)

    fact = np.mean([(cv["own", b] - cv["wrong", b]).mean(axis=0) for b in FILLERS], axis=0)
    size = np.abs(v.reshape(len(v), -1, T - 1)).max(axis=2).mean(axis=1)
    per_head = {"layer": LAYER, "fact": fact, "size": size}
    attrs = node_attrs(g_main, v, per_head)
    plot_graph(g_main, attrs, run / "figures" / "mapper_main.png",
               f"Mapper graph of the 144 heads (depth lens, windows of 4 layers, step 2; ε = {eps0:.3f}); "
               "node size = member count; x = node mean layer")
    plot_stability(grid, boot_share, run / "figures" / "stability.png")
    km.KeplerMapper(verbose=0).visualize(
        km_graph, path_html=str(run / "mapper_main.html"), title="TDA 2: heads across depth (w 4, s 2, m 1)",
        color_values=np.column_stack([LAYER, fact, locality(v), size]),
        color_function_name=["layer", "fact (own − wrong)", "locality F (mean of members)", "response size"],
        custom_tooltips=NAMES)

    (run / "main_graph.json").write_text(json.dumps({
        "setting": {"w": MAIN[0], "s": MAIN[1], "m": MAIN[2], "eps": MAIN[2] * eps0},
        "nodes": [{"window": int(g_main["cube"][i]), "members": NAMES[mem].tolist(),
                   **{k: float(attrs[k][i]) for k in ("layer", "fact", "locality", "size")}}
                  for i, mem in enumerate(g_main["nodes"])],
        "edges": [[int(a), int(b)] for a, b in zip(*np.nonzero(np.triu(g_main["A"])))]}, indent=1))
    cols = ["analysis", "w", "s", "n", "p", "m", "eps", "n_nodes", "n_edges", "n_comp"]
    with open(run / "stability.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(cols + [f"{h}_full" for h in HYPS] + [f"{h}_boot" for h in HYPS])
        for name, rs, sh in [("main (depth lens, 144 points)", grid, boot_share)] + [(a, b, None) for a, b, _ in secondary]:
            for k, r in enumerate(rs):
                wr.writerow([name] + [r[c] if c != "eps" else f"{r[c]:.4f}" for c in cols[1:]]
                            + [int(r[h]) for h in HYPS] + ([f"{x:.3f}" for x in sh[k]] if sh is not None else [""] * 6))
    (run / "verdicts.json").write_text(json.dumps(verdicts, indent=2))
    write_report(run, {"verdicts": verdicts, "main": main_r, "grid": grid, "boot": boot, "boot_share": boot_share,
                       "main_boot_rows": main_boot_rows, "n_boot": args.boot, "sets": sets, "resp": resp,
                       "eps0": eps0, "checks": checks, "secondary": secondary, "describe": describe(g_main, attrs),
                       "sweep": sweep, "pc1": pc_info, "fact": fact,
                       "nn": [(h, [(j, dist[h, j]) for j in np.argsort(dist[h])[1:5]])
                              for h in np.concatenate([sets["same"], sets["late"]])]})
    log(f"done: {run / 'tda2_report.md'}", t0)
    for k, x in checks.items():
        print(f"{k}: {x}")
    for h, x in verdicts.items():
        print(f"{h}: {x['verdict']}  main {x['main']}  s_grid {x['s_grid']:.2f}  s_boot {x['s_boot']:.2f}  "
              f"main_boot {x['main_boot']:.2f}")
    return run


if __name__ == "__main__":
    main()
