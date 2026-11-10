"""TDA analysis 4 of 4: topology of the full attention graph (every query position), own vs wrong capital at 32.

Run:  uv run --offline python -m exp4.tda4_graph --config configs/exp4_tda4.yaml

Same rows, countries and capitals as exp4.run_capital. For each filler type, τ and condition, the full post-softmax
patterns [layer, head, q, k] are averaged over each row's 11 countries into one 33 × 33 graph per unit: the 12 layer
means over heads, 8 named heads and 2 pooled layer spans. The conditions are:
- control: the row's own token at τ, its capital at 32;
- own: country at τ, its capital at 32;
- wrong: country at τ, a deranged capital at 32;
- random: country at τ, the row's own final token at 32.

Undirected weight w(i, j) = A[max(i, j), min(i, j)] (causal mask) and filtration distance 1 − w. The flag complex is
built up to triangles with gudhi and H0 and H1 are computed, without (primary) and with the BOS node. Own vs wrong is
tested against a paired label-swap null. See exp4/PREDICTIONS_tda4.md.

Writes results/<timestamp>_exp4tda4/: configs, predictions, versions, graphs.npz, diagrams.npz, tda_arrays.npz,
tda4.json, checks.json, predictions.json, figures and tda4_report.md.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # write figures to files, never open a window
import gudhi
import gudhi.hera
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.patches import Arc
from ripser import ripser

from exp1.backends import _patterns_tl, check_rows_sum_to_one, load_model, set_determinism
from exp1.runlog import make_run_dir, save_config_copy, save_versions
from exp1.sequences import prepend_bos
from exp1ext.effects import bootstrap_mean_per_unit
from exp2.design import cached_base_block
from exp3.run_positions import latin_derangements
from exp3.run_relations import PROJECT_ROOT, log, stimuli
from exp4.run_controlled import check_variants, head_name, variants, with_final

N_LAYERS = 12
CONDS = ("control", "own", "wrong", "random")
VARIANTS = {"noBOS": False, "withBOS": True}  # name -> BOS node kept
PAIRS = (("own", "wrong"), ("own", "random"), ("own", "control"))
ROLES = ("merge", "cycle", "instant")
GREEN, GREY, BLUE, ORANGE = "#1baf7a", "#898781", "#2a78d6", "#eb6834"


def load_config(path: Path) -> dict:
    """The TDA config laid over its base config (stimuli, rows, seeds and model come from the base)."""
    tda = yaml.safe_load(path.read_text())
    base = yaml.safe_load((PROJECT_ROOT / tda["base_config"]).read_text())
    return base | tda


def unit_names(cfg: dict) -> list[str]:
    return ([f"L{l}" for l in range(N_LAYERS)] + [head_name(l, h) for l, h, _ in cfg["named_heads"]]
            + list(cfg["pooled_units"]))


def check_control(seqs: np.ndarray, bases: np.ndarray, final: np.ndarray, bos_id: int) -> None:
    if not (seqs[:, :, -1] == final).all() or not (seqs[:, :, :-1] == bases[:, None, :-1]).all():
        raise AssertionError("control: a sequence differs from its row outside 32, or has the wrong capital at 32")
    s = np.sort(seqs.reshape(-1, seqs.shape[-1]), axis=1)
    if (s[:, 1:] == s[:, :-1]).any() or (seqs == bos_id).any():
        raise AssertionError("control: a sequence repeats a token or contains BOS")


def row_graphs(model, seqs: np.ndarray, cfg: dict) -> tuple[np.ndarray, float]:
    """seqs [R, n, T] -> per-row graphs [R, U_raw, T + 1, T + 1] (float32) and the max |row sum − 1|. Per row, the mean
    over its n sequences of each layer's head-mean pattern A[q, k], then of each named head's pattern. Batches hold
    whole rows, so a full pattern [b, layer, head, q, k] is never kept beyond its batch."""
    R, n, T = seqs.shape
    li = torch.tensor([l for l, _, _ in cfg["named_heads"]])
    hi = torch.tensor([h for _, h, _ in cfg["named_heads"]])
    k = cfg["rows_per_batch"]
    out, dev = None, 0.0
    with torch.inference_mode():
        for s in range(0, R, k):
            tok = torch.as_tensor(prepend_bos(seqs[s:s + k].reshape(-1, T), cfg["bos_token_id"]), dtype=torch.long)
            p = _patterns_tl(model, tok)  # [b, layer, head, q, k]
            dev = max(dev, check_rows_sum_to_one(p.numpy(), cfg["checks"]["row_sum_atol"]))
            u = torch.cat([p.mean(dim=2), p[:, li, hi]], dim=1)  # [b, U_raw, q, k]
            u = u.reshape(-1, n, *u.shape[1:]).mean(dim=1).numpy()
            if out is None:
                out = np.empty((R,) + u.shape[1:], dtype=np.float32)
            out[s:s + k] = u
    return out, dev


def add_pooled(g: np.ndarray, cfg: dict) -> np.ndarray:
    """[..., U_raw, K, K] -> [..., U, K, K]: append the mean of the layer means over each pooled span (every layer has
    12 heads, so this is also the mean over the span's heads)."""
    pooled = [g[..., lo:hi + 1, :, :].mean(axis=-3, keepdims=True) for lo, hi in cfg["pooled_units"].values()]
    return np.concatenate([g, *pooled], axis=-3)


def dist(a: np.ndarray, bos: bool) -> np.ndarray:
    """Graph A[q, k] -> filtration distances 1 − w with w(i, j) = max(A[i, j], A[j, i]), which under the causal mask
    is A[max, min], so no weight is lost or counted twice. The diagonal (self-attention) is not an edge. The BOS node
    is dropped unless bos."""
    low = np.tril(np.asarray(a, dtype=np.float64), -1)
    d = 1.0 - np.maximum(low, low.T)
    np.fill_diagonal(d, 0.0)
    return d if bos else d[1:, 1:]


def _tree(d: np.ndarray) -> gudhi.SimplexTree:
    st = gudhi.SimplexTree.create_from_array(d)  # vertices at 0 (diagonal), every edge at its distance
    st.expansion(2)  # flag complex up to triangles: enough for H1 deaths
    st.compute_persistence(homology_coeff_field=11, persistence_dim_max=False)
    return st


def diagrams(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(H0 without the infinite bar, H1), each [n, 2] of (birth, death) in 1 − w units."""
    st = _tree(d)
    h0 = np.asarray(st.persistence_intervals_in_dimension(0)).reshape(-1, 2)
    h1 = np.asarray(st.persistence_intervals_in_dimension(1)).reshape(-1, 2)
    return h0[np.isfinite(h0[:, 1])], h1


def wdist(x: tuple, y: tuple) -> float:
    """1-Wasserstein (internal L-inf) distance of H0 plus that of H1."""
    return float(sum(gudhi.hera.wasserstein_distance(a, b, order=1, internal_p=np.inf) for a, b in zip(x, y)))


def bdist(x: tuple, y: tuple) -> float:
    return float(max(gudhi.hera.bottleneck_distance(a, b, delta=0) for a, b in zip(x, y)))


def persistent(b: float, dd: float, t: dict) -> bool:
    """A class survives ≥ persist_min in 1 − w units and down to ≤ persist_ratio of its birth weight."""
    return dd - b >= t["persist_min"] and (1.0 - dd) <= t["persist_ratio"] * (1.0 - b)


def rep_cycle(d: np.ndarray, u: int, v: int, b: float) -> list[int]:
    """Nodes u ... v of a hop-shortest path among edges with filtration < b; with the birth edge (v, u) it closes a
    representative of the class that edge creates (modulo older classes, the elder-rule sense of 'new')."""
    adj = d < b
    np.fill_diagonal(adj, False)
    prev: dict[int, int | None] = {v: None}
    frontier = [v]
    while frontier and u not in prev:
        nxt = []
        for x in frontier:
            for y in np.flatnonzero(adj[x]):
                if int(y) not in prev:
                    prev[int(y)] = x
                    nxt.append(int(y))
        frontier = nxt
    if u not in prev:
        raise AssertionError("an H1 birth edge whose endpoints were not yet connected")
    path = [u]
    while path[-1] != v:
        path.append(prev[path[-1]])
    return path


def summarize(d: np.ndarray, tau: int, T: int, bos: bool, t: dict) -> dict:
    """Diagrams, the role of E1 = {T, tau}, the number of persistent H1 classes and the most persistent matching-pair
    cycle (a persistent H1 class whose representative cycle uses the edge {T, tau}). Node ids are positions."""
    off = 0 if bos else 1
    st = _tree(d)
    pairs = [(len(bs) - 1, tuple(sorted(bs)), tuple(sorted(ds)), st.filtration(bs), st.filtration(ds) if ds else np.inf)
             for bs, ds in st.persistence_pairs()]
    h = [np.array([[b, dd] for k, _, _, b, dd in pairs if k == dim and np.isfinite(dd)]).reshape(-1, 2) for dim in (0, 1)]
    e1 = (tau - off, T - off)
    if any(k == 0 and ds == e1 for k, _, ds, _, _ in pairs):
        role = "merge"
    elif any(k == 1 and bs == e1 for k, bs, _, _, _ in pairs):
        role = "cycle"
    else:
        role = "instant"  # it closed a cycle that a triangle filled at once (32 and tau already shared a neighbour)
    n_pers, mpc = 0, None
    for k, bs, ds, b, dd in pairs:
        if k != 1 or not persistent(b, dd, t):
            continue
        n_pers += 1
        cyc = rep_cycle(d, bs[0], bs[1], b)
        edges = {frozenset((cyc[i], cyc[(i + 1) % len(cyc)])) for i in range(len(cyc))}
        if frozenset(e1) in edges and (mpc is None or dd - b > mpc["persistence"]):
            mpc = {"birth": b, "death": dd, "persistence": dd - b, "birth_edge": [p + off for p in bs],
                   "death_triangle": [p + off for p in ds], "cycle": [p + off for p in cyc]}
    return {"H0": h[0], "H1": h[1], "role": role, "n_persistent": n_pers, "mpc": mpc}


def tda_cell(job: dict) -> dict:
    """All persistence work for one (filler, tau): full-sample summaries, observed distances, swap nulls, the E1
    transplant, the split-half scale and the stability bootstrap. job["rows"][cond]: per-row graphs [R, U, K, K]."""
    rows, tau, t, T = job["rows"], job["tau"], job["tda"], job["seq_len"]
    R, U, K, _ = rows["own"].shape
    V = list(VARIANTS.items())
    rng = np.random.default_rng([t["seed"], job["filler_index"], tau])
    means = {c: x.mean(axis=0, dtype=np.float64) for c, x in rows.items()}
    full, rip = {}, 0.0
    for c in CONDS:
        for u in range(U):
            for v, bos in V:
                d = dist(means[c][u], bos)
                s = summarize(d, tau, T, bos, t)
                full[(c, u, v)] = s
                r = ripser(d, distance_matrix=True, maxdim=1)["dgms"]
                rip = max(rip, bdist((s["H0"], s["H1"]), (r[0][np.isfinite(r[0][:, 1])], r[1])))
    dg = {key: (s["H0"], s["H1"]) for key, s in full.items()}
    obs = np.array([[[[wdist(dg[(x, u, v)], dg[(y, u, v)]), bdist(dg[(x, u, v)], dg[(y, u, v)])] for v, _ in V]
                     for u in range(U)] for x, y in PAIRS])  # [P, U, V, 2]
    own_t = means["own"].copy()
    own_t[:, T, tau] = means["wrong"][:, T, tau]  # transplant: own with wrong's capital -> country weight
    transplant = np.array([[wdist(diagrams(dist(own_t[u], bos)), dg[("wrong", u, v)]) for v, bos in V] for u in range(U)])

    null = np.empty((len(PAIRS), t["n_swap"], U, len(V)))
    for pi, (x, y) in enumerate(PAIRS):
        diff = (rows[x].astype(np.float64) - rows[y]).reshape(R, -1)
        swap = rng.integers(0, 2, size=(t["n_swap"], R)).astype(np.float64)
        shift = (swap @ diff / R).reshape(t["n_swap"], U, K, K)  # rows with swap = 1 trade their x and y graphs
        for s in range(t["n_swap"]):
            xs, ys = means[x] - shift[s], means[y] + shift[s]
            for u in range(U):
                for vi, (_, bos) in enumerate(V):
                    null[pi, s, u, vi] = wdist(diagrams(dist(xs[u], bos)), diagrams(dist(ys[u], bos)))

    split = np.empty((t["n_split"], U, len(V)))
    for s in range(t["n_split"]):
        perm = rng.permutation(R)
        ga = rows["own"][perm[:R // 2]].mean(axis=0, dtype=np.float64)
        gb = rows["own"][perm[R // 2:]].mean(axis=0, dtype=np.float64)
        for u in range(U):
            for vi, (_, bos) in enumerate(V):
                split[s, u, vi] = wdist(diagrams(dist(ga[u], bos)), diagrams(dist(gb[u], bos)))

    boot_mpc = np.zeros((len(CONDS), U, len(V)))
    boot_role = np.zeros((len(CONDS), U, len(V), len(ROLES)))
    for _ in range(t["n_boot"]):
        idx = rng.integers(0, R, size=R)  # the same resample for every condition
        for ci, c in enumerate(CONDS):
            g = rows[c][idx].mean(axis=0, dtype=np.float64)
            for u in range(U):
                for vi, (_, bos) in enumerate(V):
                    s = summarize(dist(g[u], bos), tau, T, bos, t)
                    boot_mpc[ci, u, vi] += s["mpc"] is not None
                    boot_role[ci, u, vi, ROLES.index(s["role"])] += 1
    return {"full": full, "obs": obs, "transplant": transplant, "null": null, "split": split,
            "boot_mpc": boot_mpc / t["n_boot"], "boot_role": boot_role / t["n_boot"], "ripser_dev": rip,
            "means": {c: m.astype(np.float32) for c, m in means.items()}}


def edge_stats(rows: dict, tau: int, T: int, boot: dict) -> dict:
    """Per condition and unit, the mean weights of E1 = {T, tau}, E2 = {tau + 1, tau} and E3 = {T, T − 1}. Also
    bootstrap 95% CIs over rows for the differences: own − wrong (E1, E3), own − random (E1), own − control (E1, E2)."""
    e = {c: {"E1": x[:, :, T, tau], "E2": x[:, :, tau + 1, tau], "E3": x[:, :, T, T - 1]} for c, x in rows.items()}
    out: dict[str, Any] = {"mean": {c: {k: v.mean(axis=0, dtype=np.float64) for k, v in d.items()} for c, d in e.items()},
                           "diff": {}}
    for name, (x, y, k) in {"E1 own−wrong": ("own", "wrong", "E1"), "E3 own−wrong": ("own", "wrong", "E3"),
                            "E1 own−random": ("own", "random", "E1"), "E1 own−control": ("own", "control", "E1"),
                            "E2 own−control": ("own", "control", "E2")}.items():
        m, lo, hi = bootstrap_mean_per_unit(e[x][k].astype(np.float64) - e[y][k], boot["iters"], boot["seed"], 0.95)
        out["diff"][name] = {"mean": m, "lo": lo, "hi": hi}
    return out


def add_tests(z: dict, t: dict) -> None:
    """Swap-null p-values and 95th percentiles, in place."""
    n = z["null"].shape[1]
    z["p"] = (1 + (z["null"] >= z["obs"][:, None, :, :, 0]).sum(axis=1)) / (1 + n)  # [P, U, V]
    z["null95"] = np.quantile(z["null"], 1 - t["alpha"], axis=1)  # [P, U, V]


def mpc_stable(z: dict, c: str, u: int, t: dict) -> bool:
    """noBOS: an MPC in the full-sample graph and in ≥ boot_stable of the resamples."""
    return z["full"][(c, u, "noBOS")]["mpc"] is not None and z["boot_mpc"][CONDS.index(c), u, 0] >= t["boot_stable"]


def mpc_absent(z: dict, c: str, u: int, t: dict) -> bool:
    """noBOS: no MPC in the full-sample graph and in ≤ boot_absent of the resamples."""
    return z["full"][(c, u, "noBOS")]["mpc"] is None and z["boot_mpc"][CONDS.index(c), u, 0] <= t["boot_absent"]


def g2_cells(res: dict, fillers: list[str], taus: list[int], t: dict, a: str = "own", b: str = "wrong") -> list[tuple]:
    """(layer, tau) cells where, in every filler, condition a has a stable MPC and condition b has none."""
    return [(u, tau) for tau in taus for u in range(N_LAYERS)
            if all(mpc_stable(res[(f, tau)], a, u, t) and mpc_absent(res[(f, tau)], b, u, t) for f in fillers)]


def check_predictions(res: dict, fillers: list[str], taus: list[int], cfg: dict) -> list[dict]:
    t = cfg["tda"]
    vi = list(VARIANTS).index("noBOS")
    early = range(t["early_layers"][0], t["early_layers"][1] + 1)
    late = range(t["late_layers"][0], t["late_layers"][1] + 1)
    out = []

    g1 = [(b, l, res[(b, t["g1_tau"])]) for b in fillers for l in early]
    ok = all(z["p"][0, l, vi] <= t["alpha"] for _, l, z in g1)
    out.append({"id": "G1", "verdict": "PASS" if ok else "FAIL", "detail": "; ".join(
        f"{b} L{l}: D {z['obs'][0, l, vi, 0]:.4f} vs null95 {z['null95'][0, l, vi]:.4f} (p {z['p'][0, l, vi]:.3f})"
        for b, l, z in g1)})

    hits, reverse = g2_cells(res, fillers, taus, t), g2_cells(res, fillers, taus, t, "wrong", "own")
    det = []
    for u, tau in hits:
        parts = []
        for b in fillers:
            z = res[(b, tau)]
            m = z["full"][("own", u, "noBOS")]["mpc"]
            parts.append(f"{b} pers {m['persistence']:.4f}, boot {z['boot_mpc'][1, u, vi]:.2f}/"
                         f"{z['boot_mpc'][2, u, vi]:.2f}, {len(m['cycle'])}-cycle")
        det.append(f"L{u} τ {tau} ({'; '.join(parts)})")
    out.append({"id": "G2", "verdict": "PASS" if hits else "FAIL",
                "detail": f"{len(hits)} own-only cells: {', '.join(det) or 'none'}; reverse (wrong-only) cells: "
                          f"{len(reverse)} {', '.join(f'L{u} τ {tau}' for u, tau in reverse)}"})

    def transplant_cells(layers: range) -> list[tuple]:
        cells = []
        for b in fillers:
            for tau in t["far_taus"]:
                if tau not in taus:
                    continue
                z = res[(b, tau)]
                for l in layers:
                    if z["p"][0, l, vi] <= t["alpha"]:
                        cells.append((b, tau, l, z["transplant"][l, vi], z["null95"][0, l, vi], z["obs"][0, l, vi, 0]))
        return cells

    ce = transplant_cells(early)
    removed = [c for c in ce if c[3] <= c[4]]
    out.append({"id": "G3a", "verdict": "PASS" if ce and len(removed) == len(ce) else "FAIL",
                "detail": f"{len(removed)}/{len(ce)} eligible early cells fall within the null after the transplant; " +
                          ", ".join(f"{b} τ {tau} L{l}: D {o:.4f} → {tr:.4f} (null95 {n95:.4f})" for b, tau, l, tr, n95, o in ce)})
    cl = transplant_cells(late)
    kept = [c for c in cl if c[3] > c[4]]
    out.append({"id": "G3b", "verdict": "PASS" if cl and len(kept) >= len(cl) / 2 else "FAIL",
                "detail": f"{len(kept)}/{len(cl)} eligible late cells stay beyond the null after the transplant; " +
                          ", ".join(f"{b} τ {tau} L{l}: D {o:.4f} → {tr:.4f} (null95 {n95:.4f})" for b, tau, l, tr, n95, o in cl)})

    found = [f"{b} τ {tau} L{u}" for b in fillers for tau in taus for u in range(1, N_LAYERS)
             if res[(b, tau)]["full"][("own", u, "withBOS")]["mpc"] is not None
             or res[(b, tau)]["full"][("wrong", u, "withBOS")]["mpc"] is not None
             or res[(b, tau)]["full"][("random", u, "withBOS")]["mpc"] is not None
             or res[(b, tau)]["full"][("control", u, "withBOS")]["mpc"] is not None]
    out.append({"id": "B1", "verdict": "PASS" if not found else "FAIL",
                "detail": f"withBOS MPCs in layers 1–11 (any condition): {', '.join(found) or 'none'}"})
    return out


def plot_distances(res: dict, units: list[str], fillers: list[str], taus: list[int], cfg: dict, variant: str,
                   path: Path) -> None:
    vi = list(VARIANTS).index(variant)
    alpha = cfg["tda"]["alpha"]
    fig, axes = plt.subplots(len(taus), len(fillers), figsize=(6.6 * len(fillers), 1.75 * len(taus) + 1.2), sharex=True,
                             squeeze=False)
    x = np.arange(len(units))
    for r, tau in enumerate(taus):
        for c, b in enumerate(fillers):
            ax, z = axes[r, c], res[(b, tau)]
            lo, hi = np.maximum(np.quantile(z["null"][0, :, :, vi], [0.05, 0.95], axis=0), 1e-7)
            ax.fill_between(x, lo, hi, color="0.86", lw=0, step="mid", label="swap null, 5–95%")
            ax.plot(x, np.maximum(np.median(z["split"][:, :, vi], axis=0), 1e-7), color=GREY, ls="--", lw=0.9,
                    drawstyle="steps-mid", label="split-half median (own)")
            obs, sig = np.maximum(z["obs"][0, :, vi, 0], 1e-7), z["p"][0, :, vi] <= alpha
            ax.scatter(x[sig], obs[sig], color=GREEN, s=15, zorder=3, label=f"own vs wrong, p ≤ {alpha}")
            ax.scatter(x[~sig], obs[~sig], facecolor="white", edgecolor=GREEN, s=15, zorder=3, label=f"own vs wrong, p > {alpha}")
            for xv in (N_LAYERS - 0.5, N_LAYERS + len(cfg["named_heads"]) - 0.5):
                ax.axvline(xv, color="0.7", lw=0.6)
            ax.set_yscale("log")
            ax.set_title(f"{b} filler, τ = {tau}", fontsize=8, loc="left")
            ax.tick_params(labelsize=6)
            ax.spines[["top", "right"]].set_visible(False)
        axes[r, 0].set_ylabel("W1(H0) + W1(H1)", fontsize=7)
    for ax in axes[-1]:
        ax.set_xticks(x)
        ax.set_xticklabels(units, rotation=90, fontsize=6)
    axes[0, 0].legend(fontsize=6, frameon=False, ncol=2, loc="lower left")
    fig.suptitle(f"Own vs wrong capital: persistence-diagram distance per unit ({variant}); layer means | named heads | "
                 "pooled", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_mpc(res: dict, units: list[str], fillers: list[str], taus: list[int], path: Path) -> None:
    vi = list(VARIANTS).index("noBOS")
    fig, axes = plt.subplots(len(fillers), len(CONDS), figsize=(3.0 * len(CONDS) + 1, 5.6 * len(fillers)), squeeze=False)
    im = None
    for r, b in enumerate(fillers):
        for c, cond in enumerate(CONDS):
            ax = axes[r, c]
            grid = np.array([[res[(b, tau)]["boot_mpc"][CONDS.index(cond), u, vi] for tau in taus] for u in range(len(units))])
            im = ax.imshow(grid, aspect="auto", cmap="Greens", vmin=0, vmax=1)
            yy, xx = np.nonzero([[res[(b, tau)]["full"][(cond, u, "noBOS")]["mpc"] is not None for tau in taus]
                                 for u in range(len(units))])
            ax.scatter(xx, yy, s=7, color="black", linewidths=0)
            ax.set_xticks(range(len(taus)))
            ax.set_xticklabels(taus, fontsize=7)
            ax.set_yticks(range(len(units)))
            ax.set_yticklabels(units if c == 0 else [], fontsize=6)
            ax.set_xlabel("τ", fontsize=7)
            ax.set_title(f"{b}: {cond}", fontsize=8)
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02,
                 label="bootstrap frequency of a matching-pair cycle (dot = present in the full-sample graph)")
    fig.suptitle("Matching-pair cycles (persistent H1 through the edge {32, τ}), noBOS graphs", fontsize=10)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _arcs(ax, w: np.ndarray, thr: float, tau: int, T: int, cycle: list[int] | None, title: str) -> None:
    """Arc diagram of the noBOS graph w [T + 1, T + 1] (positions 1..T): every edge with weight ≥ thr."""
    low = np.tril(w, -1)
    cyc = set()
    if cycle:
        cyc = {frozenset((cycle[i], cycle[(i + 1) % len(cycle)])) for i in range(len(cycle))}
    top = 0.0
    for i in range(2, T + 1):
        for j in range(1, i):
            if low[i, j] < thr:
                continue
            span = i - j
            top = max(top, span / 2)
            e = frozenset((i, j))
            if e in cyc:
                kw = {"color": GREEN, "lw": 2.4, "zorder": 3}
            elif e == frozenset((T, tau)):
                kw = {"color": ORANGE, "lw": 2.0, "zorder": 3}
            else:
                kw = {"color": GREY, "lw": 0.4 + 3 * min(low[i, j] / 0.3, 1), "alpha": 0.45, "zorder": 1}
            ax.add_patch(Arc(((i + j) / 2, 0), span, span, theta1=0, theta2=180, **kw))
    if frozenset((T, tau)) not in cyc and low[T, tau] < thr:
        span = T - tau
        top = max(top, span / 2)
        ax.add_patch(Arc(((T + tau) / 2, 0), span, span, theta1=0, theta2=180, color=ORANGE, lw=1.2, ls=":", zorder=2))
    pos = np.arange(1, T + 1)
    ax.scatter(pos, np.zeros_like(pos), s=12, color="black", zorder=4)
    for p in (tau, T):
        ax.annotate("τ" if p == tau else "32", (p, 0), xytext=(0, -11), textcoords="offset points", ha="center",
                    fontsize=8, fontweight="bold")
    ax.set_xlim(0, T + 1)
    ax.set_ylim(-1.6, top + 1)
    ax.set_yticks([])
    ax.set_xticks([1, 8, 16, 24, 32])
    ax.tick_params(labelsize=7)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_title(title, fontsize=8)


def plot_graph(z: dict, b: str, u: int, unit: str, tau: int, T: int, path: Path) -> None:
    """The unit's noBOS graph for own and wrong at the weight where own's matching-pair cycle is born (else at own's
    E1 weight), and both conditions' H0 / H1 diagrams."""
    s = {c: z["full"][(c, u, "noBOS")] for c in ("own", "wrong")}
    g = {c: z["means"][c][u] for c in ("own", "wrong")}
    mpc = s["own"]["mpc"]
    thr = 1.0 - mpc["birth"] if mpc else float(g["own"][T, tau])
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1])
    for c, cond in enumerate(("own", "wrong")):
        m = s[cond]["mpc"] if cond == "own" else None
        _arcs(fig.add_subplot(gs[0, c]), g[cond], thr, tau, T, m["cycle"] if m else None,
              f"{cond} capital at 32: edges with w ≥ {thr:.4f}; E1 = {{32, τ}} weight {g[cond][T, tau]:.4f} "
              f"({s[cond]['role']})" + (f"\ngreen = matching-pair cycle {m['cycle']} (persistence {m['persistence']:.4f})"
                                        if m else "\norange dotted = E1, below the threshold" if g[cond][T, tau] < thr else ""))
    style = {"own": {"color": GREEN, "s": 16}, "wrong": {"facecolor": "none", "edgecolor": GREY, "s": 34, "lw": 1.0}}
    ax = fig.add_subplot(gs[1, 0])  # every H0 bar is born at 0, so show the deaths (MST edges) sorted
    for cond in ("own", "wrong"):
        h0 = np.sort(s[cond]["H0"][:, 1])
        ax.scatter(np.arange(1, len(h0) + 1), h0, label=f"{cond} ({len(h0)} bars)", **style[cond])
    ax.set_xlabel("H0 bar rank", fontsize=8)
    ax.set_ylabel("death = 1 − w of the merging edge", fontsize=8)
    ax.set_title("H0: deaths of the finite bars (births all 0)", fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    ax.tick_params(labelsize=7)
    ax = fig.add_subplot(gs[1, 1])
    for cond in ("own", "wrong"):
        h = s[cond]["H1"]
        ax.scatter(h[:, 0], h[:, 1], label=f"{cond} ({len(h)} points)", **style[cond])
    if mpc:
        ax.scatter([mpc["birth"]], [mpc["death"]], s=110, marker="*", color=ORANGE, zorder=4, label="matching-pair cycle (own)")
    lim = [min(float(np.min(s[c]["H1"][:, 0], initial=1)) for c in s) - 0.002,
           max(float(np.max(s[c]["H1"][:, 1], initial=0)) for c in s) + 0.002]
    ax.plot(lim, lim, color="0.6", lw=0.7)
    ax.set_xlabel("birth (1 − w)", fontsize=8)
    ax.set_ylabel("death (1 − w)", fontsize=8)
    ax.set_title("H1 diagram", fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    ax.tick_params(labelsize=7)
    fig.suptitle(f"{b} filler, {unit}, country at τ = {tau}: noBOS attention graph (mean over 11 countries × rows)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def saved_consistency(G: dict, taus: list[int], cfg: dict, R: int) -> float:
    """Max |Δ| between this run's query-32 weights on (τ, BOS, 31) and the saved per-row arrays of Exp 4b (own, wrong,
    control; float16) and Exp 4 (random), for the layer means and the named heads, at every τ."""
    zc = np.load(PROJECT_ROOT / cfg["capital_run"] / "capital.npz")
    zr = np.load(PROJECT_ROOT / cfg["random_run"] / "controlled.npz")
    T = cfg["seq_len"]
    li = [l for l, _, _ in cfg["named_heads"]]
    hi = [h for _, h, _ in cfg["named_heads"]]
    worst = 0.0
    for b in cfg["bases"]:
        for cond in ("own", "wrong", "random"):
            m = (zr[f"{b}/country_masses"] if cond == "random" else zc[f"{cond}/{b}/country_masses"])[:R].astype(np.float32)
            for ti, tau in enumerate(taus):
                mk = m[:, T - tau - 1]  # [R, L, H, 3]: attention from 32 to tau, BOS, 31
                ref = np.concatenate([mk.mean(axis=2), mk[:, li, hi]], axis=1)
                ours = G[(b, cond)][ti][:, :, T][:, :, [tau, 0, T - 1]]
                worst = max(worst, float(np.abs(ours - ref).max()))
        ctrl = zc[f"own/{b}/control"][:R].astype(np.float32)  # [R, L, H, K], mean over the row's 11 controls
        ref = np.concatenate([ctrl.mean(axis=2), ctrl[:, li, hi]], axis=1)
        worst = max(worst, float(np.abs(G[(b, "control")][:, :, T] - ref).max()))
    return worst


def fmt_ci(m: float, lo: float, hi: float) -> str:
    star = "*" if lo > 0 or hi < 0 else ""
    return f"{m:+.4f}{star}"


def write_report(run: Path, cfg: dict, units: list[str], fillers: list[str], taus: list[int], res: dict, edges: dict,
                 checks: dict, preds: list[dict], timing: dict) -> None:
    t, T = cfg["tda"], cfg["seq_len"]
    vi = {v: i for i, v in enumerate(VARIANTS)}
    L = ["# TDA analysis 4: topology of the full attention graph, own vs wrong capital", "",
         f"Run `{run.name}`. GPT-2 small, {cfg['n_rows']} rows per filler type ({', '.join(fillers)}), "
         f"τ ∈ {{{', '.join(map(str, taus))}}}. There are four conditions:",
         "- control: the row's own token at τ, its capital at 32;",
         "- own: country at τ, its capital at 32;",
         "- wrong: country at τ, a deranged capital at 32;",
         "- random: country at τ, the row's own token at 32.",
         "",
         "Each graph is the mean full attention pattern over 11 countries × rows, with 33 nodes "
         "(BOS + 32), for each of the 12 layer means, 8 named heads and 2 pooled spans. Undirected weight w(i, j) = "
         "A[max, min]; filtration distance 1 − w; flag complex to triangles; H0 and H1 (gudhi). Primary variant: BOS node "
         f"removed (noBOS). D = W1(H0) + W1(H1). Null: {t['n_swap']} paired own/wrong label swaps. Persistent: death − "
         f"birth ≥ {t['persist_min']} and death weight ≤ {t['persist_ratio']} × birth weight. MPC (matching-pair cycle) = "
         "a persistent H1 class whose representative cycle uses the edge {32, τ}. Pre-registration: `PREDICTIONS.md`.",
         "", f"Runtime: extraction {timing['extract'] / 60:.1f} min, TDA {timing['tda'] / 60:.1f} min, "
         f"total {timing['total'] / 60:.1f} min (CPU).", "", "## Checks", "", "| check | result |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in checks.items()]
    L += ["", "## Pre-registered predictions", "", "| id | verdict | detail |", "|---|---|---|"]
    L += [f"| {p['id']} | {p['verdict']} | {p['detail']} |" for p in preds]

    for variant in VARIANTS:
        L += ["", f"## Own vs wrong distance ({variant}): D / swap-null 95th percentile (* = p ≤ {t['alpha']})", ""]
        for b in fillers:
            L += [f"**{b}**", "", "| unit | " + " | ".join(f"τ {tau}" for tau in taus) + " |", "|---" * (len(taus) + 1) + "|"]
            for u, name in enumerate(units):
                cells = []
                for tau in taus:
                    z = res[(b, tau)]
                    o, n95, p = z["obs"][0, u, vi[variant], 0], z["null95"][0, u, vi[variant]], z["p"][0, u, vi[variant]]
                    cells.append(f"{o / n95:.2f}{'*' if p <= t['alpha'] else ''}")
                L.append(f"| {name} | " + " | ".join(cells) + " |")
            L.append("")

    L += ["## Absolute distances at τ = 31 and τ = 16 (noBOS, layer means)", "",
          "| filler | τ | unit | D own–wrong | null 95th | p | split-half median | bottleneck own–wrong | D own–random (p) | "
          "D own–control (p) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for b in fillers:
        for tau in [x for x in (31, 16) if x in taus]:
            z = res[(b, tau)]
            for u in range(N_LAYERS):
                L.append(f"| {b} | {tau} | {units[u]} | {z['obs'][0, u, 0, 0]:.4f} | {z['null95'][0, u, 0]:.4f} | "
                         f"{z['p'][0, u, 0]:.3f} | {np.median(z['split'][:, u, 0]):.4f} | {z['obs'][0, u, 0, 1]:.4f} | "
                         f"{z['obs'][1, u, 0, 0]:.4f} ({z['p'][1, u, 0]:.3f}) | {z['obs'][2, u, 0, 0]:.4f} ({z['p'][2, u, 0]:.3f}) |")

    L += ["", "## Matching-pair cycles (noBOS)", "",
          "Symbol per cell: O = own has a stable MPC (full sample, bootstrap ≥ "
          f"{t['boot_stable']}) and wrong has none (bootstrap ≤ {t['boot_absent']}); W = the reverse; B = both stable; "
          "o / w = present in the full-sample graph but not meeting the stability rule; . = neither.", ""]
    for b in fillers:
        L += [f"**{b}**", "", "| unit | " + " | ".join(f"τ {tau}" for tau in taus) + " |", "|---" * (len(taus) + 1) + "|"]
        for u, name in enumerate(units):
            cells = []
            for tau in taus:
                z = res[(b, tau)]
                f = {c: z["full"][(c, u, "noBOS")]["mpc"] is not None for c in ("own", "wrong")}
                bm = {c: z["boot_mpc"][CONDS.index(c), u, 0] for c in ("own", "wrong")}
                st = {c: f[c] and bm[c] >= t["boot_stable"] for c in f}
                ab = {c: (not f[c]) and bm[c] <= t["boot_absent"] for c in f}
                if st["own"] and st["wrong"]:
                    cells.append("B")
                elif st["own"] and ab["wrong"]:
                    cells.append("O")
                elif st["wrong"] and ab["own"]:
                    cells.append("W")
                else:
                    cells.append(("o" if f["own"] else "") + ("w" if f["wrong"] else "") or ".")
            L.append(f"| {name} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["Full-sample MPCs in the own condition (noBOS), with the same cell in the other conditions:", "",
          "| filler | unit | τ | own: persistence, birth w → death w, cycle (positions) | boot own | wrong | random | control |",
          "|---|---|---|---|---|---|---|---|"]
    for b in fillers:
        for tau in taus:
            z = res[(b, tau)]
            for u, name in enumerate(units):
                m = z["full"][("own", u, "noBOS")]["mpc"]
                if m is None:
                    continue
                other = []
                for c in ("wrong", "random", "control"):
                    mc = z["full"][(c, u, "noBOS")]["mpc"]
                    bf = z["boot_mpc"][CONDS.index(c), u, 0]
                    other.append(f"{mc['persistence']:.4f} (boot {bf:.2f})" if mc else f"none (boot {bf:.2f})")
                L.append(f"| {b} | {name} | {tau} | {m['persistence']:.4f}, {1 - m['birth']:.4f} → {1 - m['death']:.4f}, "
                         f"{m['cycle']} | {z['boot_mpc'][1, u, 0]:.2f} | " + " | ".join(other) + " |")

    L += ["", "## Role of E1 = {32, τ} in the noBOS filtration (own / wrong, full sample; M = merge, C = cycle, "
          "I = instant)", ""]
    ab = {"merge": "M", "cycle": "C", "instant": "I"}
    for b in fillers:
        L += [f"**{b}**", "", "| unit | " + " | ".join(f"τ {tau}" for tau in taus) + " |", "|---" * (len(taus) + 1) + "|"]
        for u, name in enumerate(units):
            L.append(f"| {name} | " + " | ".join(
                f"{ab[res[(b, tau)]['full'][('own', u, 'noBOS')]['role']]}/{ab[res[(b, tau)]['full'][('wrong', u, 'noBOS')]['role']]}"
                for tau in taus) + " |")
        L.append("")

    for name, label in (("E1 own−wrong", "E1 = {32, τ}, own − wrong"), ("E1 own−random", "E1, own − random"),
                        ("E3 own−wrong", "E3 = {32, 31}, own − wrong"), ("E2 own−control", "E2 = {τ+1, τ}, own − control")):
        L += [f"## Edge weight: {label} (mean over rows; * = 95% row CI excludes 0)", ""]
        for b in fillers:
            L += [f"**{b}**", "", "| unit | " + " | ".join(f"τ {tau}" for tau in taus) + " |", "|---" * (len(taus) + 1) + "|"]
            for u, un in enumerate(units):
                L.append(f"| {un} | " + " | ".join(
                    fmt_ci(edges[(b, tau)]["diff"][name]["mean"][u], edges[(b, tau)]["diff"][name]["lo"][u],
                           edges[(b, tau)]["diff"][name]["hi"][u]) for tau in taus) + " |")
            L.append("")
    L += ["## Edge weight E1 in each condition (layer means L0–L4 and named heads; mean over rows)", "",
          "| filler | τ | unit | own | wrong | random | control |", "|---|---|---|---|---|---|---|"]
    show = list(range(5)) + list(range(N_LAYERS, N_LAYERS + len(cfg["named_heads"])))
    for b in fillers:
        for tau in [x for x in (31, 24, 16) if x in taus]:
            for u in show:
                mm = edges[(b, tau)]["mean"]
                L.append(f"| {b} | {tau} | {units[u]} | " + " | ".join(f"{mm[c]['E1'][u]:.4f}" for c in ("own", "wrong", "random", "control")) + " |")

    L += ["", "## Persistent H1 classes per graph, own condition (layer means; noBOS / withBOS)", "",
          "| filler | τ | " + " | ".join(units[:N_LAYERS]) + " |", "|---|---" + "|---" * N_LAYERS + "|"]
    for b in fillers:
        for tau in taus:
            z = res[(b, tau)]
            L.append(f"| {b} | {tau} | " + " | ".join(
                f"{z['full'][('own', u, 'noBOS')]['n_persistent']}/{z['full'][('own', u, 'withBOS')]['n_persistent']}"
                for u in range(N_LAYERS)) + " |")

    L += ["", "## Caveats", "",
          "- The graphs are mean attention over 2,200 sequences per condition. Their topology is that of the average, not "
          "the typical topology of one sequence's attention.",
          "- Own, wrong and random differ only in node 32's 32 edges (checked: identical query rows 0..31). The "
          "topological contrasts therefore re-express how the capital's query row attaches to a fixed graph. They add "
          "whether a strengthened edge closes a loop or only tightens node 32's attachment.",
          "- The split-half scale compares two 100-row means that differ in all 528 edges (sampling noise everywhere). "
          "Own vs wrong differ only in node 32's 32 edges. Split-half is therefore far larger than the swap null and is "
          "not a usable null for own vs wrong; as pre-registered, it is context only.",
          "- The persistence thresholds are a pre-registered judgement call. A cycle's representative is the shortest path "
          "at birth, which is one valid choice; 'uses the edge {32, τ}' is a property of that representative.",
          "- noBOS keeps raw weights, so late-layer graphs live at small weights (most mass is on BOS). withBOS is "
          "dominated by the BOS cone, which fills cycles.",
          "- The swap null tests exchangeability of own and wrong row graphs given these 11 fixed pairs; countries are "
          "not resampled.", "", "## Files", "",
          "- `graphs.npz`: mean graphs [τ, unit, q, k] per filler and condition (control is τ-independent but repeated "
          "per τ), with `units` and `taus`.",
          "- `diagrams.npz`: H0 / H1 diagrams per filler, τ, condition, unit and variant (1 − w units).",
          "- `tda_arrays.npz`: observed distances [pair, unit, variant, (W1, bottleneck)], swap nulls [pair, swap, "
          "unit, variant], p, null95, split-half [split, unit, variant], transplant, bootstrap MPC frequencies "
          "[condition, unit, variant] and E1 role frequencies.",
          "- `tda4.json`: per-graph summaries (E1 role, persistent H1 count, MPC with its cycle) and edge weights with CIs.",
          "- `figures/distances_noBOS.png`, `figures/distances_withBOS.png`: own vs wrong D per unit and τ against the null.",
          "- `figures/mpc_stability.png`: bootstrap frequency of an MPC per unit × τ for the four conditions.",
          "- `figures/graph_*.png`: arc drawing of one unit's noBOS graph (own vs wrong) with its MPC, plus diagrams.", ""]
    (run / "tda4_report.md").write_text("\n".join(L))


def jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return np.round(x.astype(np.float64), 6).tolist()
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else round(float(x), 6)
    if isinstance(x, np.integer):
        return int(x)
    return x


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="TDA analysis 4, topology of the full attention graph")
    parser.add_argument("--config", default="configs/exp4_tda4.yaml", type=Path)
    parser.add_argument("--rows", default=None, type=int, help="use only the first rows (smoke test)")
    parser.add_argument("--taus", default=None, type=int, nargs="+", help="override taus (smoke test)")
    parser.add_argument("--iters", default=None, type=int, help="override n_swap, n_split and n_boot (smoke test)")
    parser.add_argument("--results-dir", default=None, type=Path, help="override the results dir (smoke tests)")
    args = parser.parse_args(argv)
    cfg_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    cfg = load_config(cfg_path)
    if args.rows:
        cfg["n_rows"] = args.rows
    if args.taus:
        cfg["taus"] = args.taus
    if args.iters:
        cfg["tda"] |= {"n_swap": args.iters, "n_split": args.iters, "n_boot": args.iters}
    t0 = time.time()
    run = make_run_dir(args.results_dir or PROJECT_ROOT / "results", tag="exp4tda4")
    save_config_copy(cfg_path, run)
    shutil.copy2(PROJECT_ROOT / cfg["base_config"], run / "base_config.yaml")
    shutil.copy2(PROJECT_ROOT / "exp4" / "PREDICTIONS_tda4.md", run / "PREDICTIONS.md")
    save_versions(run, PROJECT_ROOT, code_dirs=("exp1", "exp2", "exp3", "exp4"))
    (run / "figures").mkdir()
    log(f"run folder {run}", t0)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg["figures"]["tokenizer"])
    set_determinism(cfg["seed"])
    tl = load_model("transformer_lens", cfg["backends"]["transformer_lens"]["model_name"], cfg["device"], cfg["dtype"])
    if tl.cfg.n_layers != N_LAYERS:
        raise AssertionError("GPT-2 small expected")
    countries, capitals, cities = stimuli(cfg, tokenizer)
    n, T, bos, taus = len(countries), cfg["seq_len"], cfg["bos_token_id"], list(cfg["taus"])
    stim_ids = np.concatenate([countries, capitals, cities, cfg["demonstration_ids"]])
    wrong = latin_derangements(n, 1, cfg["wrong_capital_seed"])[0]
    if (wrong == np.arange(n)).any():
        raise AssertionError("the wrong-capital derangement has a fixed point")
    log("wrong capitals: " + ", ".join(f"{tokenizer.decode([int(c)]).strip()}→{tokenizer.decode([int(k)]).strip()}"
                                       for c, k in zip(countries, capitals[wrong])), t0)

    bases, G, fills, finals = {}, {}, {}, {}
    dev = 0.0
    for b, rng_ in cfg["bases"].items():
        block = cached_base_block(f"exp3rel_{b}", rng_, stim_ids, cfg["base_cache_n"], T, cfg["seed"], bos,
                                  PROJECT_ROOT / cfg["paths"]["sequences_dir"])
        bases[b] = block[: cfg["n_rows"]]
        if np.isin(bases[b], stim_ids).any():
            raise AssertionError(f"{b}: a stimulus or demonstration id occurs in a base row")
        R = len(bases[b])
        fills[b] = np.tile(countries, (R, 1))
        finals[b] = {"own": np.tile(capitals, (R, 1)), "wrong": np.tile(capitals[wrong], (R, 1)), "random": None}
        for cond in ("own", "wrong"):
            if not (np.sort(finals[b][cond], axis=1) == np.sort(capitals)).all():
                raise AssertionError(f"{cond}: each capital must appear exactly once per row")
        ctrl = with_final(bases[b], n, finals[b]["own"])
        check_control(ctrl, bases[b], finals[b]["own"], bos)
        G[(b, "control")], d = row_graphs(tl, ctrl, cfg)
        dev = max(dev, d)
        for cond in ("own", "wrong", "random"):
            out = []
            for tau in taus:
                seqs = variants(bases[b], fills[b], tau, finals[b][cond])
                check_variants(seqs, bases[b], fills[b], tau, bos, finals[b][cond])
                g, d = row_graphs(tl, seqs, cfg)
                dev = max(dev, d)
                out.append(g)
            G[(b, cond)] = np.stack(out)
            log(f"{b}/{cond}: graphs for {len(taus)} τ", t0)
    t_extract = time.time() - t0
    fillers = list(cfg["bases"])
    R = cfg["n_rows"]

    # Checks on the extracted graphs.
    upper = max(float(np.triu(x, 1).max()) for x in G.values())
    prefix = max(float(np.abs(G[(b, c)][..., :T, :] - G[(b, "own")][..., :T, :]).max()) for b in fillers
                 for c in ("wrong", "random"))
    b0, k0 = fillers[0], min(cfg["rows_per_batch"], R)
    again, _ = row_graphs(tl, variants(bases[b0][:k0], fills[b0][:k0], taus[0], finals[b0]["own"][:k0]), cfg)
    det = float(np.abs(again - G[(b0, "own")][0][:k0]).max())
    saved = saved_consistency(G, taus, cfg, R)
    log("graph checks done", t0)

    units = unit_names(cfg)
    boot = cfg["analysis"]["bootstrap"]
    jobs, edges = [], {}
    for fi, b in enumerate(fillers):
        ctrl = add_pooled(G[(b, "control")], cfg)
        for ti, tau in enumerate(taus):
            rows = {"control": ctrl, **{c: add_pooled(G[(b, c)][ti], cfg) for c in ("own", "wrong", "random")}}
            edges[(b, tau)] = edge_stats(rows, tau, T, boot)
            jobs.append({"filler": b, "filler_index": fi, "tau": tau, "rows": rows, "tda": cfg["tda"], "seq_len": T})
    t1 = time.time()
    res = {}
    with ProcessPoolExecutor(max_workers=cfg["tda"]["workers"]) as pool:
        futs = {pool.submit(tda_cell, job): (job["filler"], job["tau"]) for job in jobs}
        for f in as_completed(futs):
            res[futs[f]] = f.result()
            add_tests(res[futs[f]], cfg["tda"])
            log(f"TDA {futs[f][0]} τ {futs[f][1]} done", t0)
    del jobs
    t_tda = time.time() - t1

    c = cfg["checks"]
    rip = max(z["ripser_dev"] for z in res.values())
    checks = {
        "design (ids at τ and 32, row elsewhere, no repeats/BOS/stimulus ids; control = row + capital)": "PASS (asserted per τ and condition)",
        "wrong-capital derangement has no fixed point; each capital once per row": "PASS",
        "attention rows sum to 1 (every query, every head)": f"PASS (max abs(sum − 1) {dev:.1e}, atol {c['row_sum_atol']:g}, asserted per batch)",
        "causal: upper triangle of every mean graph is 0": f"{'PASS' if upper == 0 else 'FAIL'} (max {upper:.1e})",
        "prefix identity: wrong / random query rows 0..31 equal own's": f"{'PASS' if prefix <= c['prefix_atol'] else 'FAIL'} (max abs Δ {prefix:.1e})",
        f"determinism: first batch ({k0} rows, {b0}, own, τ {taus[0]}) recomputed": f"{'PASS' if det == 0 else 'FAIL'} (max abs Δ {det:.1e})",
        "query-32 weights on τ / BOS / 31 match Exp 4b (own, wrong, control) and Exp 4 (random)": f"{'PASS' if saved <= c['saved_atol'] else 'FAIL'} (max abs Δ {saved:.1e}, atol {c['saved_atol']:g})",
        "gudhi vs ripser diagrams (bottleneck, every full-sample graph)": f"{'PASS' if rip <= c['ripser_atol'] else 'FAIL'} (max {rip:.1e})",
    }
    preds = check_predictions(res, fillers, taus, cfg)
    timing = {"extract": t_extract, "tda": t_tda, "total": time.time() - t0}

    vn = list(VARIANTS)
    np.savez_compressed(run / "graphs.npz", units=np.array(units), taus=np.array(taus),
                        **{f"{b}/{cnd}": np.stack([res[(b, tau)]["means"][cnd] for tau in taus]) for b in fillers for cnd in CONDS})
    np.savez_compressed(run / "diagrams.npz",
                        **{f"{b}/t{tau}/{cnd}/{units[u]}/{v}/H{k}": res[(b, tau)]["full"][(cnd, u, v)][f"H{k}"]
                           for b in fillers for tau in taus for cnd in CONDS for u in range(len(units)) for v in vn for k in (0, 1)})
    np.savez_compressed(run / "tda_arrays.npz", units=np.array(units), taus=np.array(taus), pairs=np.array(["-".join(p) for p in PAIRS]),
                        conds=np.array(CONDS), variants=np.array(vn), roles=np.array(ROLES),
                        **{f"{b}/t{tau}/{k}": res[(b, tau)][k] for b in fillers for tau in taus
                           for k in ("obs", "null", "p", "null95", "split", "transplant", "boot_mpc", "boot_role")})
    summ = {f"{b}/t{tau}": {"graphs": {f"{cnd}/{units[u]}/{v}": {k: res[(b, tau)]["full"][(cnd, u, v)][k]
                                                                 for k in ("role", "n_persistent", "mpc")}
                                       for cnd in CONDS for u in range(len(units)) for v in vn},
                            "edges": edges[(b, tau)]} for b in fillers for tau in taus}
    (run / "tda4.json").write_text(json.dumps(jsonable({"units": units, "taus": taus, "timing_s": timing, "cells": summ})))
    (run / "checks.json").write_text(json.dumps(checks, indent=2))
    (run / "predictions.json").write_text(json.dumps(preds, indent=2))

    for v in vn:
        plot_distances(res, units, fillers, taus, cfg, v, run / "figures" / f"distances_{v}.png")
    plot_mpc(res, units, fillers, taus, run / "figures" / "mpc_stability.png")
    # Drawing: the G2 cell with the most persistent own MPC in the first filler, else the configured fallback.
    hits = g2_cells(res, fillers, taus, cfg["tda"])
    if hits:
        b = fillers[0]
        u, tau = max(hits, key=lambda h: res[(b, h[1])]["full"][("own", h[0], "noBOS")]["mpc"]["persistence"])
    else:
        dr = cfg["draw"]
        b, u, tau = dr["filler"], units.index(dr["unit"]), dr["tau"] if dr["tau"] in taus else taus[-1]
    plot_graph(res[(b, tau)], b, u, units[u], tau, T, run / "figures" / f"graph_{b}_{units[u]}_t{tau}.png")
    write_report(run, cfg, units, fillers, taus, res, edges, checks, preds, timing)
    log(f"done: {run / 'tda4_report.md'}", t0)
    for k, v in checks.items():
        print(f"{k}: {v}")
    for p in preds:
        print(f"{p['id']}: {p['verdict']}  {p['detail'][:400]}")
    return run


if __name__ == "__main__":
    main()
