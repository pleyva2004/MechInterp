"""TDA 1: persistent homology of attention-head behaviour profiles (GPT-2 small; Exp 4 and 4b saved results).

Run:  uv run --offline python -m exp4.tda1_heads

Each head is one point: its 8 D(d) curves (country arm with a random final token, its own capital and a wrong capital;
placebo-word arm with a random final token; common and rare filler), 248 features. Vietoris-Rips H0 and H1 (ripser) on
the 144 heads under three metrics: raw Euclidean (main), correlation distance and per-feature standardised Euclidean.
Stability is a bootstrap over the 200 rows (Fasy et al. bottleneck band, per-cluster Jaccard recurrence). The nulls are
row sign-flips (noise) and a moment-matched Gaussian (shape). See exp4/PREDICTIONS_tda1.md. No model is run. Writes
results/<timestamp>_exp4tda1/: predictions, versions, tda1.npz, tda1_summary.json, tda1_clusters.csv, figures and
tda1_report.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # write figures to files, never open a window
import gudhi
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from ripser import ripser
from scipy.cluster.hierarchy import cophenet, dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS
from sklearn.metrics import adjusted_rand_score

from exp1.runlog import make_run_dir, save_versions
from exp3.run_relations import PROJECT_ROOT
from exp4.run_controlled import ARMS, LABELS, head_name, row_differences

RANDOM_RUN = Path("results/20261024-000548_exp4ctrl/controlled.npz")
CAPITAL_RUN = Path("results/20261024-150407_exp4cap/capital.npz")
T = 32
FILLERS = ("common", "rare")
CURVES = (("random", "country"), ("own", "country"), ("wrong", "country"), ("random", "word"))
CURVE_NAMES = {("random", "country"): "country, random T", ("own", "country"): "country, own capital",
               ("wrong", "country"): "country, wrong capital", ("random", "word"): "word, random T"}
VARIANTS = ("raw", "corr", "std")
VARIANT_NAMES = {"raw": "raw Euclidean (main)", "corr": "correlation distance", "std": "per-feature standardised"}
SAME_COUNTRY = ((0, 2), (4, 10), (3, 0), (1, 11))
PREV_HEADS = ((1, 0), (2, 2), (2, 3), (2, 5), (2, 8), (2, 9), (3, 2), (3, 6), (3, 7), (3, 8), (4, 11))
LATE_HEADS = ((9, 8), (10, 3), (11, 7), (11, 2))
OTHER_FOCUS = ((11, 8), (4, 11))
EXPLORE_GROUPS = {"same-country (4)": SAME_COUNTRY, "same-country trio L0H2, L4H10, L3H0 (post hoc)": SAME_COUNTRY[:3],
                  "late any-distance (4)": LATE_HEADS, "Exp 1 previous-token without L4H11 (10)": PREV_HEADS[:-1]}
STABLE, DISSOLVED, CLUSTER_MAX = 0.75, 0.5, 8
SEEDS = {"boot": 0, "N1": 1, "N2": 2, "mds": 0}
BLUE, ORANGE, GREY, GREEN = "#2a78d6", "#eb6834", "#898781", "#1baf7a"
SHORT = dict(zip(LABELS, ("global", "local", "gated prev", "negative", "plain prev", "no effect")))
MARKERS = dict(zip(LABELS, ("*", "o", "s", "v", "D", "X")))
LABEL_COLORS = dict(zip(LABELS, ("#c0392b", "#e67e22", "#2a78d6", "#8e44ad", "#16a085", "#7f7f7f")))
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#8e44ad", "#c0392b", "#d4ac0d", "#16a085", "#e84393", "#5d6d7e", "#a0522d")


def hidx(layer: int, head: int) -> int:
    return layer * 12 + head


def hname(i: int) -> str:
    return head_name(*divmod(int(i), 12))


def load_rows(root: Path) -> tuple[dict, dict, dict, dict]:
    """Per filler type, the per-row contrasts of the 4 profile curves, each [R, 31 (distance), 144 (head)]; the
    max |rebuilt mean - saved mean| per run; and Exp 4's labels and prev_score, flattened to [144]."""
    z, c = np.load(root / RANDOM_RUN), np.load(root / CAPITAL_RUN)
    rows, check = {}, {"Exp 4": 0.0, "Exp 4b": 0.0}
    for b in FILLERS:
        src = {"random": (z, f"{b}/", "Exp 4"), "own": (c, f"own/{b}/", "Exp 4b"), "wrong": (c, f"wrong/{b}/", "Exp 4b")}
        rows[b] = {}
        for cond, (f, pre, run) in src.items():
            x = row_differences({"control": f[pre + "control"], **{arm: {"masses": f[f"{pre}{arm}_masses"]} for arm in ARMS}}, T)
            for arm in ARMS:
                if (cond, arm) in CURVES:
                    rows[b][(cond, arm)] = x[arm].reshape(x[arm].shape[0], T - 1, -1)
                    check[run] = max(check[run], float(np.abs(x[arm].mean(axis=0) - f[f"{pre}{arm}/mean"]).max()))
    labels = {b: z[f"{b}/labels"].reshape(-1) for b in FILLERS}
    prev = {b: z[f"{b}/prev_score"].reshape(-1) for b in FILLERS}
    return rows, check, labels, prev


def profiles(rows: dict, w: dict[str, np.ndarray]) -> np.ndarray:
    """w[b]: [B, R] row weights (bootstrap counts or signs). Returns [B, 144, 248]: per head, the weighted row mean of
    each curve, concatenated filler-major, then curve, then distance (the order in PREDICTIONS_tda1.md)."""
    blocks = []
    for b in FILLERS:
        B, R = w[b].shape
        for key in CURVES:
            x = rows[b][key]
            blocks.append((w[b] @ x.reshape(R, -1) / R).reshape(B, T - 1, -1).transpose(0, 2, 1))
    return np.concatenate(blocks, axis=2)


def distances(P: np.ndarray, variant: str, sd: np.ndarray | None = None) -> np.ndarray:
    """[144, 248] -> [144, 144] under the variant's metric. std divides each feature by sd (default: P's own per-feature
    SD across heads); nulls and resamples pass the main profiles' SD so that every cloud is measured in one metric."""
    if variant == "corr":
        d = 1.0 - np.corrcoef(P)
    else:
        if variant == "std":
            sd = P.std(axis=0) if sd is None else sd
            P = P / np.where(sd > 0, sd, 1.0)  # centring is irrelevant to Euclidean distances
        d = squareform(pdist(P))
    d = np.clip((d + d.T) / 2, 0.0, None)
    np.fill_diagonal(d, 0.0)
    return d


def persistence(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rips H0 (finite bars only; the one infinite bar is dropped) and H1 diagrams, [n, 2] (birth, death)."""
    dgm0, dgm1 = ripser(d, maxdim=1, distance_matrix=True)["dgms"]
    return dgm0[np.isfinite(dgm0[:, 1])], dgm1


def h1_stats(dgm1: np.ndarray) -> tuple[float, float]:
    """Largest death / birth ratio and largest persistence of an H1 diagram (1 and 0 when it is empty)."""
    if not len(dgm1):
        return 1.0, 0.0
    return float((dgm1[:, 1] / dgm1[:, 0]).max()), float((dgm1[:, 1] - dgm1[:, 0]).max())


def gaussian_null(P: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """[n, 144, 248]: each draw is 144 points from a Gaussian with P's mean and sample covariance (rank <= 143)."""
    mu = P.mean(axis=0)
    return mu + rng.standard_normal((n, P.shape[0], P.shape[0])) @ (P - mu) / np.sqrt(P.shape[0] - 1)


def components(part: np.ndarray) -> list[np.ndarray]:
    """Head indices of each flat cluster, largest first (ties: smallest head index first)."""
    return sorted((np.flatnonzero(part == k) for k in np.unique(part)), key=lambda g: (-len(g), g[0]))


def recurrence(comps: list[np.ndarray], boot_parts: np.ndarray) -> np.ndarray:
    """Per main component, the mean over resamples of its best Jaccard overlap with a resample component."""
    out = np.zeros(len(comps))
    for part in boot_parts:
        for i, g in enumerate(comps):
            best = 0.0
            for k in np.unique(part[g]):  # only components that meet g can overlap it
                other = np.flatnonzero(part == k)
                best = max(best, len(np.intersect1d(g, other)) / len(np.union1d(g, other)))
            out[i] += best
    return out / len(boot_parts)


def same_country_ok(part: np.ndarray) -> tuple[bool, list[int]]:
    """T1-b's configuration: the four same-country heads share a component of <= CLUSTER_MAX heads with no Exp 1
    previous-token head. Also returns that component (or the union of their components)."""
    ids = [hidx(*hd) for hd in SAME_COUNTRY]
    comp = sorted(set(np.flatnonzero(np.isin(part, part[ids])).tolist()))
    together = len(set(part[ids].tolist())) == 1
    prev_in = any(hidx(*hd) in comp for hd in PREV_HEADS)
    return together and len(comp) <= CLUSTER_MAX and not prev_in, comp


def group_scale(Z: np.ndarray, ids: list[int]) -> tuple[float, np.ndarray]:
    """Smallest single-linkage scale at which the heads share one component, and that component's members."""
    h = float(squareform(cophenet(Z))[np.ix_(ids, ids)].max())
    part = fcluster(Z, h, criterion="distance")
    return h, np.flatnonzero(part == part[ids[0]])


def explore_groups(r: dict) -> dict[str, dict]:
    """Exploratory (not pre-registered): per named group, the scale at which it connects, the component it is then
    part of, its gap to the rest, and how often a resample's component at its own connecting scale is exactly the group."""
    out = {}
    for name, heads in EXPLORE_GROUPS.items():
        ids = [hidx(*hd) for hd in heads]
        h, comp = group_scale(r["linkage"], ids)
        rest = np.setdiff1d(np.arange(len(r["part"])), comp)
        gap = float(r["d"][np.ix_(comp, rest)].min()) if len(rest) else float("inf")
        exact = np.mean([len(group_scale(L, ids)[1]) == len(ids) for L in r["boot_links"]])
        out[name] = {"scale": h, "size": len(comp), "gap": gap, "boot_exact": float(exact)}
    return out


def merged(labels: np.ndarray) -> np.ndarray:
    """Labels with 'plain previous-token' and 'no country effect' merged (the profile lacks prev_score)."""
    return np.where(labels == LABELS[5], LABELS[4], labels)


def analyse_variant(variant: str, P: np.ndarray, boot_P: np.ndarray, n1_P: np.ndarray, n2_P: np.ndarray,
                    labels: dict[str, np.ndarray], t0: float) -> dict[str, Any]:
    sd = P.std(axis=0) if variant == "std" else None
    d = distances(P, variant, sd)
    dgm0, dgm1 = persistence(d)
    Z = linkage(squareform(d, checks=False), method="single")
    r: dict[str, Any] = {"d": d, "dgm0": dgm0, "dgm1": dgm1, "linkage": Z,
                         "h0_vs_linkage": float(np.abs(np.sort(dgm0[:, 1]) - np.sort(Z[:, 2])).max())}
    for name, cloud in (("N1", n1_P), ("N2", n2_P)):
        stats = np.empty((len(cloud), 3))
        sorted0 = np.empty((len(cloud), len(dgm0)))
        for i, Q in enumerate(cloud):
            q0, q1 = persistence(distances(Q, variant, sd))
            sorted0[i] = np.sort(q0[:, 1])[::-1]
            stats[i] = (sorted0[i, 0], *h1_stats(q1))
        r[name] = {"max_death": stats[:, 0], "max_ratio": stats[:, 1], "max_pers": stats[:, 2], "sorted0": sorted0}
    bn = np.empty((len(boot_P), 2))
    boot_links = np.empty((len(boot_P),) + Z.shape)
    boot_sorted0 = np.empty((len(boot_P), len(dgm0)))
    boot_dgm1 = []
    for i, Q in enumerate(boot_P):
        db = distances(Q, variant, sd)
        q0, q1 = persistence(db)
        bn[i] = gudhi.bottleneck_distance(q0, dgm0), gudhi.bottleneck_distance(q1, dgm1)
        boot_links[i] = linkage(squareform(db, checks=False), method="single")
        boot_sorted0[i] = np.sort(q0[:, 1])[::-1]
        boot_dgm1.append(q1)
    c0, c1 = (float(np.percentile(bn[:, k], 95)) for k in range(2))
    q95 = {f"{n}_{s}": float(np.percentile(r[n][s], 95)) for n in ("N1", "N2") for s in ("max_death", "max_ratio", "max_pers")}
    eps = max(q95["N1_max_death"], 2 * c0)
    part = fcluster(Z, eps, criterion="distance")
    boot_parts = np.stack([fcluster(L, eps, criterion="distance") for L in boot_links])
    comps = components(part)
    coph = squareform(cophenet(Z))
    rest = lambda g: np.setdiff1d(np.arange(len(part)), g)
    pers1 = dgm1[:, 1] - dgm1[:, 0]
    ratio1 = dgm1[:, 1] / dgm1[:, 0]
    dw = np.sqrt(2 * d) if variant == "corr" else d  # Ward needs Euclidean input; sqrt(2(1 - r)) is Euclidean
    r.update({
        "bottleneck": bn, "c0": c0, "c1": c1, "q95": q95, "eps": eps, "part": part, "boot_parts": boot_parts,
        "boot_links": boot_links,
        "comps": comps, "recurrence": recurrence(comps, boot_parts),
        "gap": [float(d[np.ix_(g, rest(g))].min()) if len(g) < len(part) else float("inf") for g in comps],
        "internal": [float(coph[np.ix_(g, g)].max()) for g in comps],
        "boot_sorted0": boot_sorted0, "boot_dgm1": boot_dgm1[:20],
        "h0_beyond": {"N1": int((dgm0[:, 1] > q95["N1_max_death"]).sum()), "band": int((dgm0[:, 1] > 2 * c0).sum()),
                      "N2": int((dgm0[:, 1] > q95["N2_max_death"]).sum())},
        "h1_band": pers1 > 2 * c1, "h1_n1": ratio1 > q95["N1_max_ratio"], "h1_n2": ratio1 > q95["N2_max_ratio"],
        "ari": {}, "ari_boot": {}, "ari_merged": {}, "ari_k6_single": {}, "ari_k6_ward": {},
        "t1b_main": same_country_ok(part), "t1b_boot": float(np.mean([same_country_ok(p)[0] for p in boot_parts])),
    })
    if variant == "std":
        # Disclosed alternative reading of the pre-registration: each N1 cloud z-scored by its own SDs.
        self_max = np.array([persistence(distances(Q, variant))[0][:, 1].max() for Q in n1_P])
        eps_self = max(float(np.percentile(self_max, 95)), 2 * c0)
        comps_self = components(fcluster(Z, eps_self, criterion="distance"))
        r["self_null"] = {"N1_q95_max_death": float(np.percentile(self_max, 95)), "eps": eps_self,
                          "n_components": len(comps_self), "n_multi": sum(len(g) >= 2 for g in comps_self)}
    k6_single = fcluster(Z, 6, criterion="maxclust")
    k6_ward = fcluster(linkage(squareform(dw, checks=False), method="ward"), 6, criterion="maxclust")
    for b, lab in labels.items():
        r["ari"][b] = adjusted_rand_score(lab, part)
        r["ari_boot"][b] = np.array([adjusted_rand_score(lab, p) for p in boot_parts])
        r["ari_merged"][b] = adjusted_rand_score(merged(lab), part)
        r["ari_k6_single"][b] = adjusted_rand_score(lab, k6_single)
        r["ari_k6_ward"][b] = adjusted_rand_score(lab, k6_ward)
    Y = MDS(n_components=2, metric="precomputed", init="classical_mds", random_state=SEEDS["mds"],
            max_iter=1000).fit_transform(d)
    r["mds"] = Y
    r["explore"] = explore_groups(r)
    r["mds_stress"] = float(np.sqrt(((pdist(Y) - squareform(d, checks=False)) ** 2).sum() / (squareform(d, checks=False) ** 2).sum()))
    print(f"[{time.time() - t0:6.1f}s] {variant}: eps {eps:.4g} (N1 {q95['N1_max_death']:.4g}, 2c0 {2 * c0:.4g}), "
          f"{len(comps)} components, H1 persistent (band & N2) {int((r['h1_band'] & r['h1_n2']).sum())}", flush=True)
    return r


def check_predictions(r: dict, labels: dict) -> list[dict]:
    """Pre-registered criteria (PREDICTIONS_tda1.md), applied to one variant's results."""
    multi = [(g, q) for g, q in zip(r["comps"], r["recurrence"]) if len(g) >= 2]
    stable = [(g, q) for g, q in multi if q >= STABLE]
    n_single = sum(len(g) == 1 for g in r["comps"])
    desc = ", ".join(f"{len(g)} heads (rec {q:.2f})" for g, q in multi) or "none"
    t1a = (len(stable) >= 3, f"ε* {r['eps']:.4g}; {len(r['comps'])} components: {len(multi)} with ≥ 2 heads "
                             f"[{desc}], {n_single} isolated heads; stable (rec ≥ {STABLE}) clusters: {len(stable)}")
    ids = [hidx(*hd) for hd in SAME_COUNTRY]
    ok, comp = r["t1b_main"]
    sizes = ", ".join(f"{hname(i)} in a {int((r['part'] == r['part'][i]).sum())}-head component" for i in ids)
    prev_in = [hname(hidx(*hd)) for hd in PREV_HEADS if hidx(*hd) in comp]
    t1b = (ok and r["t1b_boot"] >= STABLE,
           f"{sizes}; together: {len(set(r['part'][ids].tolist())) == 1}; previous-token heads in their component(s): "
           f"{', '.join(prev_in) or 'none'}; configuration recurs in {r['t1b_boot']:.0%} of resamples")
    band = lambda a: "labels real" if a >= 0.5 else ("partly" if a >= 0.2 else "labels do not match persistent clusters")
    parts = [f"{b}: ARI {r['ari'][b]:+.3f} [{np.percentile(r['ari_boot'][b], 2.5):+.3f}, "
             f"{np.percentile(r['ari_boot'][b], 97.5):+.3f}] ({band(r['ari'][b])})" for b in labels]
    t1c = (all(r["ari"][b] >= 0.5 for b in labels), "; ".join(parts))
    dg = r["dgm1"]
    persistent = r["h1_band"] & r["h1_n2"]
    if len(dg):
        top = int(np.argmax(dg[:, 1] / dg[:, 0]))
        tp = (f"top ratio feature: birth {dg[top, 0]:.4g}, death {dg[top, 1]:.4g}, ratio {dg[top, 1] / dg[top, 0]:.3f} "
              f"(N2 q95 {r['q95']['N2_max_ratio']:.3f}, N1 q95 {r['q95']['N1_max_ratio']:.3f}), persistence "
              f"{dg[top, 1] - dg[top, 0]:.4g} (2·c1 {2 * r['c1']:.4g})")
    else:
        tp = "no H1 features"
    t1d = (not persistent.any(), f"{len(dg)} H1 features; beyond band {int(r['h1_band'].sum())}, beyond N2 ratio "
                                 f"{int(r['h1_n2'].sum())}, beyond N1 ratio {int(r['h1_n1'].sum())}; band & N2: "
                                 f"{int(persistent.sum())}; {tp}")
    return [{"id": k, "verdict": "PASS" if ok_ else "FAIL", "detail": det}
            for k, (ok_, det) in zip(("T1-a", "T1-b", "T1-c", "T1-d"), (t1a, t1b, t1c, t1d))]


def comp_colors(r: dict) -> tuple[np.ndarray, dict[int, str]]:
    """Per head: its component's colour (multi-head components from PALETTE by size; isolated heads white)."""
    colors = np.array(["#ffffff"] * len(r["part"]), dtype=object)
    legend = {}
    for k, g in enumerate(c for c in r["comps"] if len(c) >= 2):
        colors[g] = PALETTE[k % len(PALETTE)]
        legend[k] = PALETTE[k % len(PALETTE)]
    return colors, legend


def plot_persistence(out: dict, n_boot: int, n_null: int, path: Path) -> None:
    fig, axes = plt.subplots(len(out), 2, figsize=(12.5, 3.9 * len(out)), squeeze=False)
    for (ax0, ax1), (v, r) in zip(axes, out.items()):
        deaths = np.sort(r["dgm0"][:, 1])[::-1]
        rank = np.arange(1, len(deaths) + 1)
        lo, hi = np.percentile(r["boot_sorted0"], [2.5, 97.5], axis=0)
        ax0.fill_between(rank, lo, hi, color=BLUE, alpha=0.2, lw=0, label="bootstrap 95% range, by rank")
        ax0.plot(rank, deaths, "o", ms=2.6, color=BLUE, label="H0 bar deaths (main)")
        ax0.plot(rank, np.percentile(r["N1"]["sorted0"], 95, axis=0), color=GREY, lw=1, ls="--",
                 label="N1 sign-flip, 95th pct by rank")
        ax0.axhline(r["q95"]["N2_max_death"], color=GREEN, lw=0.9, ls="-.", label="N2 Gaussian, 95th pct of max")
        ax0.axhline(2 * r["c0"], color=ORANGE, lw=0.9, ls=":", label=f"bootstrap band 2·c0 = {2 * r['c0']:.3g}")
        ax0.axhline(r["eps"], color="k", lw=1.0, label=f"cut ε* = {r['eps']:.3g}")
        ax0.set_xscale("log")
        ax0.set_yscale("log")
        ax0.set_xlabel("bar rank (1 = longest finite bar)", fontsize=8)
        ax0.set_ylabel("death (all H0 births are 0)", fontsize=8)
        ax0.set_title(f"{VARIANT_NAMES[v]}: H0", fontsize=9)
        ax0.legend(fontsize=6, frameon=False, loc="lower left")
        dg = r["dgm1"]
        for q in r["boot_dgm1"]:
            ax1.plot(q[:, 0], q[:, 1], ".", color="0.8", ms=1.5, zorder=1)
        top = max(float(dg[:, 1].max()) if len(dg) else 1.0, max(float(q[:, 1].max()) for q in r["boot_dgm1"] if len(q)))
        bot = min(float(dg[:, 0].min()) if len(dg) else 0.0, min(float(q[:, 0].min()) for q in r["boot_dgm1"] if len(q)))
        x = np.linspace(0, top * 1.05, 50)
        ax1.plot(x, x, color="0.4", lw=0.8)
        ax1.plot(x, x + 2 * r["c1"], color=ORANGE, lw=0.9, ls=":", label=f"band: death = birth + 2·c1 ({2 * r['c1']:.3g})")
        ax1.plot(x, x * r["q95"]["N2_max_ratio"], color=GREEN, lw=0.9, ls="-.",
                 label=f"N2: death/birth = {r['q95']['N2_max_ratio']:.3f} (95th pct of max)")
        ax1.plot(x, x * r["q95"]["N1_max_ratio"], color=GREY, lw=0.9, ls="--",
                 label=f"N1: death/birth = {r['q95']['N1_max_ratio']:.3f}")
        ax1.plot(dg[:, 0], dg[:, 1], "o", ms=3, color=BLUE, zorder=3, label=f"H1 (main, {len(dg)} features)")
        hit = r["h1_band"] & r["h1_n2"]
        if hit.any():
            ax1.plot(dg[hit, 0], dg[hit, 1], "o", ms=7, mfc="none", mec="k", zorder=4, label="band & N2")
        ax1.plot([], [], ".", color="0.8", label="20 bootstrap diagrams")
        ax1.set_xlim(bot * 0.9, top * 1.05)
        ax1.set_ylim(bot * 0.9, top * 1.05)
        ax1.set_xlabel("birth", fontsize=8)
        ax1.set_ylabel("death", fontsize=8)
        ax1.set_title(f"{VARIANT_NAMES[v]}: H1", fontsize=9)
        ax1.legend(fontsize=6, frameon=False, loc="lower right")
        for ax in (ax0, ax1):
            ax.tick_params(labelsize=7)
            ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Vietoris–Rips persistence of the 144 head profiles (bootstrap over rows, {n_boot} resamples; nulls "
                 f"{n_null} draws each)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_dendrogram(r: dict, labels: np.ndarray, variant: str, path: Path) -> None:
    Z, eps, n = r["linkage"], r["eps"], len(r["part"])
    colors, _ = comp_colors(r)
    members = [[i] for i in range(n)]
    for a, b_, *_ in Z:
        members.append(members[int(a)] + members[int(b_)])
    fig, ax = plt.subplots(figsize=(22, 6.5))
    dendrogram(Z, labels=[hname(i) for i in range(n)], leaf_font_size=5.5, ax=ax,
               link_color_func=lambda k: colors[members[k][0]] if Z[k - n, 2] <= eps else GREY)
    names = {hname(i): i for i in range(n)}
    for t in ax.get_xticklabels():
        t.set_color(LABEL_COLORS[labels[names[t.get_text()]]])
    ax.axhline(eps, color="k", lw=0.9, ls="--")
    ax.set_yscale("log")
    ax.set_ylim(Z[:, 2].min() * 0.7, Z[:, 2].max() * 1.3)
    ax.text(0.995, eps, f" ε* = {eps:.3g}", transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=8)
    ax.set_ylabel("single-linkage merge height = H0 death (log)", fontsize=8)
    fig.legend(handles=[Patch(color=LABEL_COLORS[lab], label=SHORT[lab]) for lab in LABELS], fontsize=7, frameon=False,
               loc="lower center", ncol=len(LABELS), title="leaf text colour = Exp 4 label (common filler)",
               title_fontsize=7)
    ax.set_title(f"{VARIANT_NAMES[variant]}: single-linkage dendrogram of the 144 heads. Links below ε* are coloured "
                 f"by persistent component; grey links merge above the cut.", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_mds(r: dict, labels: np.ndarray, variant: str, path: Path) -> None:
    Y = r["mds"]
    colors, legend = comp_colors(r)
    focus = {hidx(*hd) for hd in SAME_COUNTRY + PREV_HEADS + LATE_HEADS + OTHER_FOCUS}
    giant = set(r["comps"][0].tolist()) if len(r["comps"][0]) >= 2 else set()
    annotate = focus | (set(range(len(Y))) - giant)
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    core = np.array(sorted(giant)) if giant else np.arange(len(Y))
    lo, hi = np.percentile(Y[core], [2, 98], axis=0)
    pad = (hi - lo) * 0.15
    for ax, zoom in zip(axes, (False, True)):
        for lab in LABELS:
            sel = labels == lab
            ax.scatter(Y[sel, 0], Y[sel, 1], marker=MARKERS[lab], s=70 if lab == LABELS[0] else 34,
                       c=list(colors[sel]), edgecolors="k", linewidths=0.5, zorder=3)
        if zoom:
            ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
            ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        for i in annotate:
            if x0 <= Y[i, 0] <= x1 and y0 <= Y[i, 1] <= y1:
                ax.annotate(hname(i), Y[i], fontsize=5.5, xytext=(3, 3), textcoords="offset points",
                            color="k" if i in focus else "0.35", zorder=4)
        ax.set_title(("zoom on the largest component" if zoom else "all heads") +
                     f" (Kruskal stress-1 {r['mds_stress']:.2f})", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    handles = [Line2D([], [], marker=MARKERS[lab], ls="", color="0.6", mec="k", mew=0.5, ms=7 if lab == LABELS[0] else 5,
                      label=f"{SHORT[lab]}") for lab in LABELS]
    multi = [g for g in r["comps"] if len(g) >= 2]
    handles += [Patch(fc=c, ec="k", lw=0.5, label=f"component {k + 1}: {len(multi[k])} heads") for k, c in legend.items()]
    handles.append(Patch(fc="#ffffff", ec="k", lw=0.5, label="isolated head (own component)"))
    fig.legend(handles=handles, fontsize=7, frameon=False, loc="lower center", ncol=min(len(handles), 6),
               title="marker = Exp 4 label (common filler); fill = persistent component at ε*", title_fontsize=7)
    fig.suptitle(f"{VARIANT_NAMES[variant]}: metric MDS of the 144 head profiles (focus heads labelled in black)", fontsize=10)
    fig.tight_layout(rect=(0, 0.1, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def head_list(g: np.ndarray, limit: int = 12) -> str:
    names = [hname(i) for i in g]
    return ", ".join(names) if len(names) <= limit else ", ".join(names[:limit]) + f", … (+{len(names) - limit})"


def label_mix(g: np.ndarray, lab: np.ndarray) -> str:
    vals, counts = np.unique(lab[g], return_counts=True)
    return ", ".join(f"{SHORT[v]} {c}" for v, c in sorted(zip(vals, counts), key=lambda t: -t[1]))


def write_csv(out: dict, labels: dict, path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "component", "size", "gap_to_rest", "internal_scale", "recurrence", "members",
                    "labels_common", "labels_rare"])
        for v, r in out.items():
            for k, (g, q, gap, inn) in enumerate(zip(r["comps"], r["recurrence"], r["gap"], r["internal"])):
                w.writerow([v, k + 1, len(g), f"{gap:.5g}", f"{inn:.5g}", f"{q:.3f}", " ".join(hname(i) for i in g),
                            "; ".join(SHORT[x] for x in labels["common"][g]), "; ".join(SHORT[x] for x in labels["rare"][g])])


def write_report(run: Path, out: dict, preds: dict, checks: dict, labels: dict, P: np.ndarray,
                 args: argparse.Namespace) -> None:
    raw = out["raw"]
    L = ["# TDA 1: persistent homology of attention-head behaviour profiles", "",
         f"Run `{run.name}`. Pre-registration: `PREDICTIONS.md` (copy of `exp4/PREDICTIONS_tda1.md`, written before any "
         f"topology was computed). Saved results only: `{RANDOM_RUN}` and `{CAPITAL_RUN}`.",
         "",
         "**Profile.** One point per head (144). Each point is 8 D(d) curves × 31 distances = 248 features, in attention "
         "units:",
         "- country arm with a random final token, its own capital, and a wrong capital;",
         "- placebo-word arm with a random final token;",
         "- each in common and rare filler.",
         "",
         "**Method.**",
         "- Vietoris–Rips H0 and H1 with ripser.",
         f"- Stability: {args.boot} row bootstraps (Fasy et al. bottleneck band c_k; Hennig Jaccard recurrence at a "
         f"fixed cut).",
         f"- Nulls, {args.null} draws each: N1, row sign-flip (noise); N2, moment-matched Gaussian (shape).",
         "- Cut: ε* = max(N1's 95th percentile of the largest H0 death, 2·c0).",
         "",
         "## Checks", "", "| check | result |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in checks.items()]
    s = out["std"]["self_null"]
    L += ["", "## Clarification of the pre-registration (decided after a 20-draw smoke test, before the full run)", "",
          "- **Ambiguity.** The pre-registration did not say whether a null or bootstrap cloud is z-scored by its own "
          "per-feature SDs or by the main profiles' SDs.",
          "- **Choice.** Here every cloud uses the main profiles' SDs, so all clouds are measured in one fixed metric. "
          "That is the reading that matches N1's stated purpose: separations that estimation noise produces under the "
          "main metric.",
          "- **Why not self-standardisation.** In a noise cloud, each feature's variance across heads sits in a few "
          "high-noise heads. Self-standardisation therefore isolates those heads far more than the real metric would.",
          f"- **The other reading, for transparency.** With self-standardised N1 clouds, std gives N1 q95 "
          f"{s['N1_q95_max_death']:.4g}, ε* {s['eps']:.4g}, {s['n_components']} component(s), of which {s['n_multi']} "
          f"have ≥ 2 heads.",
          "", "## Thresholds", "",
          "| variant | c0 (H0 band half-width) | c1 | N1 q95 max H0 death | N2 q95 max H0 death | ε* | N1 q95 max H1 ratio | "
          "N2 q95 max H1 ratio | H0 bars > N1 / > band / > N2 |", "|---|---|---|---|---|---|---|---|---|"]
    for v, r in out.items():
        q = r["q95"]
        L.append(f"| {v} | {r['c0']:.4g} | {r['c1']:.4g} | {q['N1_max_death']:.4g} | {q['N2_max_death']:.4g} | "
                 f"{r['eps']:.4g} | {q['N1_max_ratio']:.3f} | {q['N2_max_ratio']:.3f} | {r['h0_beyond']['N1']} / "
                 f"{r['h0_beyond']['band']} / {r['h0_beyond']['N2']} |")
    L += ["", "## Pre-registered predictions (raw metric = main analysis)", "", "| id | verdict | detail |", "|---|---|---|"]
    L += [f"| {p['id']} | {p['verdict']} | {p['detail']} |" for p in preds["raw"]]
    L += ["", "## Robustness: the same criteria under the other metrics", "",
          "| id | raw | corr | std | robust |", "|---|---|---|---|---|"]
    for i, p in enumerate(preds["raw"]):
        vs = [preds[v][i]["verdict"] for v in VARIANTS]
        L.append(f"| {p['id']} | " + " | ".join(vs) + f" | {'yes' if len(set(vs)) == 1 else 'no'} |")
    for v in ("corr", "std"):
        L += ["", f"{v} details:", ""]
        L += [f"- {p['id']} {p['verdict']}: {p['detail']}" for p in preds[v]]
    for v, r in out.items():
        L += ["", f"## Persistent components at ε*: {VARIANT_NAMES[v]}", "",
              "Gap to rest = smallest distance from a member to a non-member (the H0 death at which the component "
              "merges). Internal = largest single-linkage merge inside it. Recurrence = mean best Jaccard over "
              "resamples.", "",
              "| # | size | gap to rest | internal | recurrence | members | labels (common) | labels (rare) |",
              "|---|---|---|---|---|---|---|---|"]
        singles = []
        for k, (g, q, gap, inn) in enumerate(zip(r["comps"], r["recurrence"], r["gap"], r["internal"])):
            if len(g) == 1:
                singles.append(f"{hname(g[0])} ({gap:.3g}, rec {q:.2f})")
                continue
            L.append(f"| {k + 1} | {len(g)} | {gap:.4g} | {inn:.4g} | {q:.2f} | {head_list(g, 20 if k else 10)} | "
                     f"{label_mix(g, labels['common'])} | {label_mix(g, labels['rare'])} |")
        L += ["", f"Isolated heads ({len(singles)}; gap to nearest head, recurrence): " + ("; ".join(singles) or "none") + "."]
    L += ["", "## Focus heads: component at ε* (size)", "", "| head | group | " + " | ".join(VARIANTS) + " |", "|---|---|---|---|---|"]
    groups = ([(hd, "same-country") for hd in SAME_COUNTRY] + [(hd, "Exp 1 previous-token") for hd in PREV_HEADS] +
              [(hd, "late any-distance") for hd in LATE_HEADS] + [((11, 8), "local country-reading"), ((4, 11), "clean previous-token")])
    seen = set()
    for hd, grp in groups:
        if hd in seen:
            continue
        seen.add(hd)
        cells = []
        for v in VARIANTS:
            r = out[v]
            i = hidx(*hd)
            k = next(j for j, g in enumerate(r["comps"]) if i in g)
            cells.append(f"#{k + 1} ({len(r['comps'][k])})")
        L.append(f"| {head_name(*hd)} | {grp} | " + " | ".join(cells) + " |")
    L += ["", "## Agreement with the threshold labels (ARI)", "",
          "| variant | filler | ARI at ε* [95% boot] | labels 5+6 merged | single linkage, 6 clusters | Ward, 6 clusters |",
          "|---|---|---|---|---|---|"]
    for v, r in out.items():
        for b in FILLERS:
            L.append(f"| {v} | {b} | {r['ari'][b]:+.3f} [{np.percentile(r['ari_boot'][b], 2.5):+.3f}, "
                     f"{np.percentile(r['ari_boot'][b], 97.5):+.3f}] | {r['ari_merged'][b]:+.3f} | "
                     f"{r['ari_k6_single'][b]:+.3f} | {r['ari_k6_ward'][b]:+.3f} |")
    L += ["", f"Reference: ARI between the common and rare labels themselves = "
              f"{adjusted_rand_score(labels['common'], labels['rare']):+.3f}.", "",
          "Labels × raw components (common filler; components with ≥ 2 heads, then isolated heads):", ""]
    multi = [g for g in raw["comps"] if len(g) >= 2]
    iso = np.concatenate([g for g in raw["comps"] if len(g) == 1] or [np.array([], dtype=int)])
    L += ["| label | " + " | ".join(f"#{k + 1} ({len(g)})" for k, g in enumerate(multi)) + " | isolated |",
          "|---" * (len(multi) + 2) + "|"]
    for lab in LABELS:
        L.append(f"| {SHORT[lab]} | " + " | ".join(str(int((labels['common'][g] == lab).sum())) for g in multi) +
                 f" | {int((labels['common'][iso] == lab).sum())} |")
    L += ["", "## H1 features (raw), largest death/birth first", "",
          "| birth | death | ratio | persistence | > 2·c1 | > N2 ratio q95 | > N1 ratio q95 |", "|---|---|---|---|---|---|---|"]
    dg = raw["dgm1"]
    for i in np.argsort(-(dg[:, 1] / dg[:, 0]))[:8]:
        L.append(f"| {dg[i, 0]:.4g} | {dg[i, 1]:.4g} | {dg[i, 1] / dg[i, 0]:.3f} | {dg[i, 1] - dg[i, 0]:.4g} | "
                 f"{bool(raw['h1_band'][i])} | {bool(raw['h1_n2'][i])} | {bool(raw['h1_n1'][i])} |")
    curve_labels = [f"{b}: {CURVE_NAMES[c]}" for b in FILLERS for c in CURVES]
    L += ["", "## Exploratory (not pre-registered; raw metric)", "",
          "Isolated heads at ε*, by gap. Curve shares are each curve's fraction of the head's squared profile norm; the "
          "two largest are shown.", "",
          "| head | gap | nearest head | recurrence | profile norm | largest curve shares | Exp 4 label common / rare |",
          "|---|---|---|---|---|---|---|"]
    iso_rows = sorted(((g[0], gap, q) for g, gap, q in zip(raw["comps"], raw["gap"], raw["recurrence"]) if len(g) == 1),
                      key=lambda t: -t[1])
    for i, gap, q in iso_rows:
        dd = raw["d"][i].copy()
        dd[i] = np.inf
        sq = (P[i].reshape(len(curve_labels), T - 1) ** 2).sum(axis=1)
        top = np.argsort(-sq)[:2]
        L.append(f"| {hname(i)} | {gap:.3g} | {hname(int(np.argmin(dd)))} | {q:.2f} | {np.sqrt(sq.sum()):.3g} | " +
                 ", ".join(f"{curve_labels[k]} {sq[k] / sq.sum():.0%}" for k in top) +
                 f" | {SHORT[labels['common'][i]]} / {SHORT[labels['rare'][i]]} |")
    L += ["", "Named groups across scales. Scale = the smallest single-linkage height at which the group shares one "
              "component. Size = that component's size; it is a cluster of exactly the group only when size = group "
              "size. Gap = that component's distance to the rest. Bootstrap = the share of resamples in which the "
              "component at the resample's own connecting scale is exactly the group.", "",
          "| group | scale | component size | gap | gap / scale | bootstrap exact |", "|---|---|---|---|---|---|"]
    for name, e in raw["explore"].items():
        L.append(f"| {name} | {e['scale']:.3g} | {e['size']} | {e['gap']:.3g} | {e['gap'] / e['scale']:.2f} | "
                 f"{e['boot_exact']:.0%} |")
    L += ["", "## Caveats", "",
          "- **144 points is a small cloud.**",
          "  - Single linkage (H0) sees gaps but not density. Modes inside the large component, or a dense core with a "
          "thin bridge, are invisible to it.",
          "  - H1 has low power. A loop needs several heads spread around it, so 'no persistent H1' does not rule out a "
          "family sampled by only a few heads.",
          "  - Every quantity is conditional on these 144 heads. The bootstrap resamples rows (measurement noise), not "
          "heads.",
          "- **The metric decides what 'close' means.**",
          "  - raw is dominated by the few heads with large effects (L0H2's own-capital curve is the largest).",
          "  - corr and std amplify noise in near-zero heads and features.",
          "- **The fixed-ε* cut is the finest partition supported above noise.** Coarser groupings exist higher up the "
          "dendrogram. The six-cluster cuts in the ARI table are descriptive only.",
          "- **The labels are per filler type and use prev_score**, which the profile lacks (hence the merged-label ARI). "
          "The labels themselves agree across filler types only at the reference ARI above.",
          "- **The Fasy band assumes the row bootstrap approximates the sampling law of the profiles.** That is reasonable "
          "for 200 exchangeable rows. The capital arrays are float16 (rebuild error ≤ 3e-5, far below every threshold).",
          "",
          "## Files", "",
          "- `tda1.npz`: the main profiles [144, 248]; per variant, the distance matrix, diagrams, linkage, partition, "
          "MDS coordinates, null statistics and bootstrap bottleneck distances and ARIs.",
          "- `tda1_summary.json`: thresholds, verdicts and library versions.",
          "- `tda1_clusters.csv`: every component per variant, with its members and labels.",
          "- `figures/persistence.png`: H0 deaths by rank and the H1 diagram per variant, with the bootstrap band and "
          "both nulls.",
          "- `figures/dendrogram_{raw,corr,std}.png`: single linkage with the ε* cut.",
          "- `figures/mds_{raw,corr,std}.png`: 2-D metric MDS, coloured by component and marked by label.", ""]
    (run / "tda1_report.md").write_text("\n".join(L))


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="TDA 1: persistent homology of head behaviour profiles")
    parser.add_argument("--boot", default=500, type=int, help="bootstrap resamples over rows")
    parser.add_argument("--null", default=500, type=int, help="draws per null model")
    parser.add_argument("--results-dir", default=None, type=Path, help="override results/ (smoke tests)")
    args = parser.parse_args(argv)
    t0 = time.time()
    run = make_run_dir(args.results_dir or PROJECT_ROOT / "results", tag="exp4tda1")
    shutil.copy2(PROJECT_ROOT / "exp4" / "PREDICTIONS_tda1.md", run / "PREDICTIONS.md")
    save_versions(run, PROJECT_ROOT, code_dirs=("exp1", "exp2", "exp3", "exp4"))
    (run / "figures").mkdir()
    print(f"run folder {run}", flush=True)

    rows, rebuild, labels, prev = load_rows(PROJECT_ROOT)
    R = {b: rows[b][CURVES[0]].shape[0] for b in FILLERS}
    P = profiles(rows, {b: np.ones((1, R[b])) for b in FILLERS})[0]
    rng = {k: np.random.default_rng(s) for k, s in SEEDS.items()}
    boot_P = profiles(rows, {b: rng["boot"].multinomial(R[b], np.full(R[b], 1 / R[b]), size=args.boot).astype(float)
                             for b in FILLERS})
    n1_P = profiles(rows, {b: rng["N1"].choice([-1.0, 1.0], size=(args.null, R[b])) for b in FILLERS})
    n2_P = gaussian_null(P, args.null, rng["N2"])
    print(f"[{time.time() - t0:6.1f}s] profiles {P.shape}, bootstrap {boot_P.shape}, nulls {n1_P.shape} / {n2_P.shape}",
          flush=True)

    out = {v: analyse_variant(v, P, boot_P, n1_P, n2_P, labels, t0) for v in VARIANTS}
    preds = {v: check_predictions(r, labels) for v, r in out.items()}
    n1_mean = float(np.abs(n1_P.mean(axis=0)).max())
    checks = {"rebuilt per-row means = saved means, Exp 4": f"{'PASS' if rebuild['Exp 4'] == 0 else 'FAIL'} "
                                                            f"(max |Δ| {rebuild['Exp 4']:.1e})",
              "rebuilt per-row means = saved means, Exp 4b (float16)": f"{'PASS' if rebuild['Exp 4b'] <= 1e-4 else 'FAIL'} "
                                                                     f"(max |Δ| {rebuild['Exp 4b']:.1e})",
              **{f"ripser H0 deaths = single-linkage heights, {v}": f"{'PASS' if r['h0_vs_linkage'] <= 1e-5 * max(1.0, float(r['linkage'][:, 2].max())) else 'FAIL'} "
                                                                     f"(max |Δ| {r['h0_vs_linkage']:.1e})" for v, r in out.items()},
              "N1 mean profile ≈ 0": f"max |mean over draws| {n1_mean:.1e} vs max |profile| {float(np.abs(P).max()):.3f}"}

    np.savez_compressed(run / "tda1.npz", profiles=P, head_names=np.array([hname(i) for i in range(len(P))]),
                        **{f"labels_{b}": labels[b].astype(str) for b in FILLERS},
                        **{f"prev_score_{b}": prev[b] for b in FILLERS},
                        **{f"{v}/{k}": r[k] for v, r in out.items()
                           for k in ("d", "dgm0", "dgm1", "linkage", "part", "mds", "bottleneck", "boot_parts")},
                        **{f"{v}/{n}/{s}": r[n][s] for v, r in out.items() for n in ("N1", "N2")
                           for s in ("max_death", "max_ratio", "max_pers")},
                        **{f"{v}/ari_boot_{b}": r["ari_boot"][b] for v, r in out.items() for b in FILLERS})
    summary = {"profile": {"features": [f"{b}: {CURVE_NAMES[c]}, d 1–31" for b in FILLERS for c in CURVES],
                           "n_heads": int(P.shape[0]), "n_features": int(P.shape[1])},
               "boot": args.boot, "null_draws": args.null, "seeds": SEEDS,
               "libraries": {p: version(p) for p in ("ripser", "persim", "gudhi", "scikit-learn", "scipy", "numpy")},
               "checks": checks, "predictions": preds, "std_self_standardised_null": out["std"]["self_null"],
               "variants": {v: {"c0": r["c0"], "c1": r["c1"], "eps": r["eps"], "q95": r["q95"], "h0_beyond": r["h0_beyond"],
                                "n_components": len(r["comps"]),
                                "components": [{"members": [hname(i) for i in g], "recurrence": float(q), "gap": gap,
                                                "internal": inn}
                                               for g, q, gap, inn in zip(r["comps"], r["recurrence"], r["gap"], r["internal"])],
                                "ari": r["ari"], "ari_merged": r["ari_merged"], "ari_k6_single": r["ari_k6_single"],
                                "ari_k6_ward": r["ari_k6_ward"], "mds_stress": r["mds_stress"],
                                "explore_groups": r["explore"]}
                            for v, r in out.items()}}
    (run / "tda1_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    write_csv(out, labels, run / "tda1_clusters.csv")
    plot_persistence(out, args.boot, args.null, run / "figures" / "persistence.png")
    for v, r in out.items():
        plot_dendrogram(r, labels["common"], v, run / "figures" / f"dendrogram_{v}.png")
        plot_mds(r, labels["common"], v, run / "figures" / f"mds_{v}.png")
    write_report(run, out, preds, checks, labels, P, args)
    print(f"[{time.time() - t0:6.1f}s] done: {run / 'tda1_report.md'}", flush=True)
    for k, v in checks.items():
        print(f"{k}: {v}")
    for v in VARIANTS:
        for p in preds[v]:
            print(f"{v} {p['id']}: {p['verdict']}  {p['detail']}")
    return run


if __name__ == "__main__":
    main()
