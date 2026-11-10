"""Post-hoc checks for R4 (exploratory, written after the run; not pre-registered). Uses exp5.r4_flow's functions on
the same full data. Writes posthoc_checks.txt next to this file.

Run (from the project root):  uv run --offline python results/20261025-172418_exp5R4/posthoc_checks.py
"""
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from exp5.r4_flow import (FILLERS, LAYERS, MAIN, MS, PCA_K, SETTINGS, TAUS, cloud, cover_windows, cut, grid,
                          load_half, project, score, trees)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "results" / "20261025-170021_exp5res"
OUT = Path(__file__).with_name("posthoc_checks.txt")
lines: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    lines.append(s)


def mixed_windows(r: dict, windows: list) -> str:
    bad = [f"{w[0]}-{w[-1]}" for j, w in enumerate(windows) if not r["clean"][j]]
    return ",".join(bad) if bad else "none"


for b in FILLERS:
    half, _ = load_half(DATA, b)
    for tau in TAUS:
        X, lab = cloud(half[tau], np.arange(100))
        Y = project(X, lab["own"], PCA_K)
        say(f"=== {b}, tau = {tau}")
        # (a) main setting: where the mixed points of window 11-12 come from
        res = grid(Y, lab, settings=SETTINGS, global_rule=True)
        g, r = res[MAIN]["g"], res[MAIN]["r"]
        last = len(g["windows"]) - 1
        mix = [i for i in np.flatnonzero(r["mixed"]) if g["cube"][i] == last]
        if mix:
            pts = np.concatenate([g["nodes"][i] for i in mix])
            say(f"(a) main, window 11-12: {len(mix)} mixed nodes, {len(pts)} points; share from layer 12: "
                f"{np.mean(lab['layer'][pts] == 12):.3f}; own share of their layer-12 points "
                f"{lab['own'][pts][lab['layer'][pts] == 12].mean():.3f}, of their layer-11 points "
                f"{(lab['own'][pts][lab['layer'][pts] == 11].mean() if (lab['layer'][pts] == 11).any() else float('nan')):.3f}")
        # (b) single-layer windows (w = 1) at the same scale rule, layers 10, 11, 12
        tr1 = trees(Y, lab["layer"], {(10,), (11,), (12,)})
        for m in MS:
            parts = []
            for l in (10, 11, 12):
                g1 = cut(tr1, [(l,)], m)
                r1 = score(g1, lab["own"])
                parts.append(f"x_{l}: {r1['n_nodes'][0]} nodes, mixed {r1['mixed_pts'][0]:.3f}")
            say(f"(b) single-layer windows, m = {m}: " + "; ".join(parts))
        # (c) windows that break each failing grid setting, and the global-scale rule
        for s in SETTINGS:
            rr = res[s]["r"]
            if rr["lstar"] is None:
                say(f"(c) {s}: unclean windows {mixed_windows(rr, res[s]['g']['windows'])}")
        for m in MS:
            rr = res["global", m]["r"]
            say(f"(c) global scale m = {m}: unclean windows {mixed_windows(rr, res['global', m]['g']['windows'])}")
        # (d) PCA 768 (no reduction), main cover
        r768 = grid(X, lab, settings=tuple((*MAIN[:2], m) for m in MS))
        for m in MS:
            rr = r768[(*MAIN[:2], m)]["r"]
            say(f"(d) PCA 768 m = {m}: unclean windows {mixed_windows(rr, cover_windows(*MAIN[:2]))}; mixed share "
                f"per window {np.round(rr['mixed_pts'], 2).tolist()}")
        # (e) the late-layer 2-cluster split: what the minority cluster is
        for l in (8, 10, 12):
            sel = np.flatnonzero(lab["layer"] == l)
            c = fcluster(linkage(pdist(Y[sel]), "average"), 2, "maxclust")
            k = np.argmin([np.sum(c == x) for x in (1, 2)]) + 1
            mem = sel[c == k]
            cells, cnt = np.unique(lab["cell"][mem], return_counts=True)
            say(f"(e) x_{l}: minority cluster {len(mem)} points, own share {lab['own'][mem].mean():.2f}, cells "
                + ", ".join(f"{x}:{n}" for x, n in zip(cells, cnt)))
        # (f) leave-one-cell-out nearest-centroid accuracy: share of pairs with delta . mu_(-cell) > 0
        acc = []
        own = lab["own"]
        for l in LAYERS:
            d = X[(lab["layer"] == l) & own]  # +delta/2 per pair
            cell = lab["cell"][(lab["layer"] == l) & own]
            ok = [(d[cell == k] @ d[cell != k].mean(axis=0) > 0).mean() for k in range(11)]
            acc.append(np.mean(ok))
        say("(f) leave-one-cell-out centroid accuracy x_1..x_12: " + " ".join(f"{x:.3f}" for x in acc))
        say()

OUT.write_text("\n".join(lines) + "\n")
