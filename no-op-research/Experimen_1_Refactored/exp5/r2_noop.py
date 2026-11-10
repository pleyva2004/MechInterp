"""Experiment 5, R2: the no-op test. Do heads that rest on BOS write a near-constant vector?

Run:  uv run --offline python -m exp5.r2_noop [--data results/20261025-170021_exp5res] [--results-dir DIR] [--boot N]

Per head (l, i), at the final token T: write w = z W_O[l, i]; spread over a set S of sequences = RMS distance of the
writes from their mean / RMS norm of the stream x_l(T) over S; BOS share = mean attention from T to BOS over S.
- R2-a: Spearman(BOS share, spread) over the 144 heads, random final token (cond 2).
- R2-b: Spearman(Δ BOS share, Δ spread), own − wrong capital, layers 6–11.
- R2-c: median spread of resting (share ≥ 0.8) vs active (≤ 0.3) heads.
CIs come from a row bootstrap on exact per-row sufficient statistics. Exploratory: depth-confound checks, TwoNN, Rips H0
total persistence and participation ratio of each head's write cloud, write norms. Saved arrays only, no model. See
exp5/PREDICTIONS.md and exp5/PREDICTIONS_R2.md. Writes results/<timestamp>_exp5R2/: predictions, versions,
r2_stats.npz, r2_summary.json, figures and r2_report.md.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # write figures to files, never open a window
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from ripser import ripser
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata, spearmanr

from exp1.runlog import make_run_dir, save_versions
from exp3.run_relations import PROJECT_ROOT, log
from exp4.run_controlled import head_name

DATA = Path("results/20261025-170021_exp5res")
FILLERS = ("common", "rare")
CONDS = ("own", "wrong", "random")
L, H, DH, DM = 12, 12, 64, 768
LATE = 6  # R2-b uses layers LATE..11
NOTABLE = ((0, 2), (4, 11), (9, 8), (10, 3), (11, 2), (11, 7), (11, 8))
REST, ACTIVE = 0.8, 0.3  # R2-c thresholds on BOS share
BOOT, SEED, PCA_DIM, CHUNK = 2000, 0, 20, 250
INK, MUTED, ORANGE = "#2b2a27", "#898781", "#eb6834"
LAYER_COLORS = plt.cm.Blues(np.linspace(0.3, 1.0, L))


# ---------------------------------------------------------------- data

def frames(W_O: np.ndarray) -> np.ndarray:
    """[L, H, 64, 768] -> F [L, H, 64, 64] with z F isometric to z W_O: W_O = U Σ Vᵀ, F = U Σ (Vᵀ has orthonormal rows)."""
    U, S, _ = np.linalg.svd(W_O.astype(np.float64), full_matrices=False)
    return U * S[..., None, :]


def isometric(z16: np.ndarray, F: np.ndarray) -> np.ndarray:
    """z [R, m, L, H, 64] (fp16) -> y [R, m, L, H, 64] float64, the writes in the isometric frame."""
    R, m = z16.shape[:2]
    z = z16.astype(np.float64).reshape(R * m, L * H, DH).transpose(1, 0, 2)
    return np.matmul(z, F.reshape(L * H, DH, DH)).transpose(1, 0, 2).reshape(R, m, L, H, DH)


def quant_var(z16: np.ndarray, rownorm2: np.ndarray) -> np.ndarray:
    """Expected ‖δw‖² from rounding z to fp16 (uniform error, variance spacing² / 12 per element, independent),
    mapped through W_O: Σ_d var_d ‖W_O[l, i, d, :]‖². z [R, m, L, H, 64] -> [R, m, L, H]."""
    var = np.spacing(np.abs(z16)).astype(np.float64) ** 2 / 12
    return np.einsum("rmlhd,lhd->rmlh", var, rownorm2)


def row_sums(y: np.ndarray, bos: np.ndarray, xx: np.ndarray, qv: np.ndarray, centre: np.ndarray) -> dict[str, Any]:
    """Exact per-row sufficient statistics of one set: every resampled set's moments are count-weighted sums of these.
    y is centred on a fixed per-head vector first, so the sum-of-squares identity keeps its precision."""
    yc = y - centre
    return {"m": y.shape[1], "sy": yc.sum(1), "sq": np.einsum("rmlhd,rmlhd->rlh", yc, yc), "bos": bos.sum(1),
            "xx": xx.sum(1), "qv": qv.sum(1), "centre": centre}


def combine(a: dict, b: dict) -> dict:
    if not np.array_equal(a["centre"], b["centre"]):
        raise AssertionError("sets combined with different centres")
    return {"m": a["m"] + b["m"], "centre": a["centre"],
            **{k: a[k] + b[k] for k in ("sy", "sq", "bos", "xx", "qv")}}


def load_filler(data: Path, b: str, F: np.ndarray) -> tuple[dict[str, dict], dict[str, np.ndarray]]:
    """Per set (own, wrong, random, random + controls): row sums. Also the random set's isometric writes
    [R * m, L, H, 64] and fp16 z (for the topology and the 768-d check)."""
    s = np.load(data / f"seq_{b}.npz", mmap_mode="r")
    rownorm2 = (F ** 2).sum(-1)  # ‖W_O[l, i, d, :]‖² = ‖F[l, i, d, :]‖²
    R = s["z"].shape[0]
    raw: dict[str, tuple] = {}
    for c, name in enumerate(CONDS):
        z16 = np.asarray(s["z"][:, c]).reshape(R, -1, L, H, DH)
        bos = np.asarray(s["att"][:, c, ..., 0], dtype=np.float64).reshape(R, -1, L, H)
        x = np.asarray(s["resid"][:, c, ..., :L, :], dtype=np.float64).reshape(R, -1, L, DM)
        raw[name] = (z16, bos, (x ** 2).sum(-1))
    raw["rnd_control"] = (np.asarray(s["rnd_ctrl_z"])[:, None], np.asarray(s["rnd_ctrl_att_row"][..., 0], np.float64)[:, None],
                          (np.asarray(s["rnd_ctrl_resid"][:, :L], np.float64) ** 2).sum(-1)[:, None])
    y_rand = isometric(raw["random"][0], F)
    centre = y_rand.mean(axis=(0, 1))  # one centre per head for every set (combined sets must share it)
    sets = {}
    for name, (z16, bos, xx) in raw.items():
        y = y_rand if name == "random" else isometric(z16, F)
        sets[name] = row_sums(y, bos, xx, quant_var(z16, rownorm2), centre)
    sets["random+ctrl"] = combine(sets["random"], sets.pop("rnd_control"))
    keep = {"y": y_rand.reshape(-1, L, H, DH), "z16": raw["random"][0].reshape(-1, L, H, DH),
            "xx": raw["random"][2].reshape(-1, L)}
    return sets, keep


# ---------------------------------------------------------------- statistics

def moments(st: dict, c: np.ndarray) -> dict[str, np.ndarray]:
    """Row counts c [..., R] -> BOS share [..., L, H], write variance about the set's mean [..., L, H], stream mean
    square [..., L], fp16 rounding variance [..., L, H], mean write minus the centre [..., L, H, 64]."""
    N = c.sum(-1) * st["m"]
    td = lambda a: np.tensordot(c, a, axes=(-1, 0))
    md = td(st["sy"]) / N[..., None, None, None]
    var = np.maximum(td(st["sq"]) / N[..., None, None] - (md ** 2).sum(-1), 0.0)
    return {"bos": td(st["bos"]) / N[..., None, None], "var": var, "xx": td(st["xx"]) / N[..., None],
            "qv": td(st["qv"]) / N[..., None, None], "md": md}


def boot_moments(st: dict, counts: np.ndarray) -> dict[str, np.ndarray]:
    parts = [moments(st, counts[i:i + CHUNK]) for i in range(0, len(counts), CHUNK)]
    return {k: np.concatenate([p[k] for p in parts]) for k in ("bos", "var", "xx", "qv")}


def boot_counts(R: int, n: int, seed: int) -> np.ndarray:
    idx = np.random.default_rng(seed).integers(0, R, size=(n, R))
    counts = np.zeros((n, R))
    np.add.at(counts, (np.arange(n)[:, None], idx), 1)
    return counts


def spread(mo: dict, xx: np.ndarray | None = None, var: np.ndarray | None = None) -> np.ndarray:
    """RMS distance from the mean / RMS stream norm, [..., L, H]. xx: another per-layer mean square to divide by."""
    v = mo["var"] if var is None else var
    return np.sqrt(v) / np.sqrt(mo["xx"] if xx is None else xx)[..., None]


def rho(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Spearman ρ along the last axis (vectorised over leading axes)."""
    ra, rb = rankdata(a, axis=-1), rankdata(b, axis=-1)
    ra, rb = ra - ra.mean(-1, keepdims=True), rb - rb.mean(-1, keepdims=True)
    return (ra * rb).sum(-1) / np.sqrt((ra ** 2).sum(-1) * (rb ** 2).sum(-1))


def rho_within(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Mean over layers of the within-layer Spearman ρ; a, b [..., layers, heads]."""
    return rho(a, b).mean(-1)


def rho_partial(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Partial Spearman controlling for layer: Pearson ρ of the ranks after removing each layer's mean rank.
    a, b [..., layers, heads]."""
    sh = a.shape[:-2] + (-1,)
    ra, rb = rankdata(a.reshape(sh), axis=-1).reshape(a.shape), rankdata(b.reshape(sh), axis=-1).reshape(b.shape)
    ra, rb = (ra - ra.mean(-1, keepdims=True)).reshape(sh), (rb - rb.mean(-1, keepdims=True)).reshape(sh)
    return (ra * rb).sum(-1) / np.sqrt((ra ** 2).sum(-1) * (rb ** 2).sum(-1))


def summary(point: float, boot: np.ndarray) -> dict[str, float]:
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return {"rho": float(point), "lo": float(lo), "hi": float(hi)}


def fmt(s: dict[str, float], key: str = "rho") -> str:
    return f"{s[key]:+.3f} [{s['lo']:+.3f}, {s['hi']:+.3f}]"


def ratio_fmt(s: dict[str, float]) -> str:
    return f"{s['rho']:.2f} [{s['lo']:.2f}, {s['hi']:.2f}]"


def group_medians(share: np.ndarray, spr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """[..., L, H] -> median spread of resting and active heads, and their counts."""
    sh, sp = share.reshape(share.shape[:-2] + (-1,)), spr.reshape(spr.shape[:-2] + (-1,))
    rest, act = sh >= REST, sh <= ACTIVE
    with warnings.catch_warnings():  # an empty group in a resample gives NaN, which nanpercentile skips
        warnings.simplefilter("ignore", RuntimeWarning)
        med = lambda m: np.nanmedian(np.where(m, sp, np.nan), axis=-1)
        return med(rest), med(act), rest.sum(-1), act.sum(-1)


def posthoc(sh_p: np.ndarray, sp_p: np.ndarray, sh_b: np.ndarray, sp_b: np.ndarray) -> dict[str, dict]:
    """Post hoc, added after the first full run (20261025-172046_exp5R2) and not in the addendum. How much of R2-a is
    the mechanical end (share near 1) or layer 0, and is it more than dilution? A head's BOS term is the same vector in
    every sequence, so if the rest of its attention wrote as variably as any head's, spread would scale with
    1 − share; spread / (1 − share) removes that. Head subsets are fixed from the point estimates."""
    out: dict[str, dict] = {}
    layer = np.repeat(np.arange(L), H).reshape(L, H)
    for name, m in (("share ≤ 0.9", sh_p <= 0.9), ("share ≤ 0.8", sh_p <= 0.8), ("layers 1–11", layer >= 1)):
        out[name] = {**summary(rho(sh_p[m], sp_p[m]), rho(sh_b[:, m], sp_b[:, m])), "n": int(m.sum())}
    m = sh_p <= 0.95
    out["dilution: spread / (1 − share), share ≤ 0.95"] = {
        **summary(rho(sh_p[m], (sp_p / (1 - sh_p))[m]), rho(sh_b[:, m], (sp_b / (1 - sh_b))[:, m])), "n": int(m.sum())}
    dp, db = sp_p / np.maximum(1 - sh_p, 0.05), sp_b / np.maximum(1 - sh_b, 0.05)
    out["dilution, partial ρ controlling for layer (1 − share floored at 0.05)"] = {
        **summary(rho_partial(sh_p, dp), rho_partial(sh_b, db)), "n": L * H}
    out["n_share_gt_0.9"] = {"rho": float((sh_p > 0.9).sum()), "lo": np.nan, "hi": np.nan, "n": int((sh_p > 0.9).sum())}
    return out


def analyse(sets: dict[str, dict], boot: int) -> dict[str, Any]:
    R = sets["random"]["sy"].shape[0]
    counts = boot_counts(R, boot, SEED)
    ones = np.ones(R)
    P = {k: moments(st, ones) for k, st in sets.items()}
    B = {k: boot_moments(st, counts) for k, st in sets.items()}
    out: dict[str, Any] = {"R": R, "P": P}

    # R2-a (and its sensitivity with the random controls)
    for key, name in (("random", "R2-a"), ("random+ctrl", "R2-a + controls")):
        out[name] = summary(rho(P[key]["bos"].ravel(), spread(P[key]).ravel()),
                            rho(B[key]["bos"].reshape(boot, -1), spread(B[key]).reshape(boot, -1)))
    pr, br = P["random"], B["random"]
    out["spread_random"], out["share_random"] = spread(pr), pr["bos"]
    # exploratory depth-confound checks on the R2-a set
    out["unnormalised"] = summary(rho(pr["bos"].ravel(), pr["var"].ravel()),
                                  rho(br["bos"].reshape(boot, -1), br["var"].reshape(boot, -1)))
    out["within-layer"] = summary(rho_within(pr["bos"], spread(pr)), rho_within(br["bos"], spread(br)))
    out["partial (layer)"] = summary(rho_partial(pr["bos"], spread(pr)), rho_partial(br["bos"], spread(br)))
    cv_p, cv_b = np.maximum(pr["var"] - pr["qv"], 0), np.maximum(br["var"] - br["qv"], 0)
    out["fp16-corrected"] = summary(rho(pr["bos"].ravel(), spread(pr, var=cv_p).ravel()),
                                    rho(br["bos"].reshape(boot, -1), spread(br, var=cv_b).reshape(boot, -1)))
    # within-row part: each row's 33 sequences share context and final token and differ only at tau (country, distance)
    st = sets["random"]
    wss = np.maximum(st["sq"] - (st["sy"] ** 2).sum(-1) / st["m"], 0.0)  # [R, L, H], centring-invariant
    win_p = np.sqrt(wss.sum(0) / (R * st["m"])) / np.sqrt(pr["xx"])[:, None]
    win_b = np.sqrt(np.tensordot(counts, wss, axes=(-1, 0)) / (R * st["m"])) / np.sqrt(br["xx"])[..., None]
    out["within-row"] = summary(rho(pr["bos"].ravel(), win_p.ravel()), rho(br["bos"].reshape(boot, -1), win_b.reshape(boot, -1)))
    out["within_row_spread"] = win_p
    out["fp16_floor_ratio"] = np.sqrt(pr["qv"] / np.maximum(pr["var"], 1e-300))
    out["posthoc"] = posthoc(pr["bos"], spread(pr), br["bos"], spread(br))
    out["layer_rho"] = rho(pr["bos"], spread(pr))  # [L], point estimates only
    out["share_by_layer"] = pr["bos"].mean(-1)

    # R2-b: one denominator per layer for both conditions (own and wrong have the same sequence count per row)
    po, pw, bo, bw = P["own"], P["wrong"], B["own"], B["wrong"]
    pool_p, pool_b = (po["xx"] + pw["xx"]) / 2, (bo["xx"] + bw["xx"]) / 2
    d_share_p, d_share_b = (po["bos"] - pw["bos"])[LATE:], (bo["bos"] - bw["bos"])[:, LATE:]
    d_spr_p = (spread(po, pool_p) - spread(pw, pool_p))[LATE:]
    d_spr_b = (spread(bo, pool_b) - spread(bw, pool_b))[:, LATE:]
    out["R2-b"] = summary(rho(d_share_p.ravel(), d_spr_p.ravel()), rho(d_share_b.reshape(boot, -1), d_spr_b.reshape(boot, -1)))
    sep_p = (spread(po) - spread(pw))[LATE:]
    sep_b = (spread(bo) - spread(bw))[:, LATE:]
    out["R2-b separate denominators"] = summary(rho(d_share_p.ravel(), sep_p.ravel()),
                                                rho(d_share_b.reshape(boot, -1), sep_b.reshape(boot, -1)))
    out["R2-b within-layer"] = summary(rho_within(d_share_p, d_spr_p), rho_within(d_share_b, d_spr_b))
    out["R2-b partial (layer)"] = summary(rho_partial(d_share_p, d_spr_p), rho_partial(d_share_b, d_spr_b))
    out["d_share"], out["d_spread"] = d_share_p, d_spr_p
    out["d_share_mean"] = summary(d_share_p.mean(), d_share_b.mean(axis=(1, 2)))
    out["d_spread_mean"] = summary(d_spr_p.mean(), d_spr_b.mean(axis=(1, 2)))
    out["stream_own_vs_wrong_l0"] = float(abs(po["xx"][0] - pw["xx"][0]) / po["xx"][0])

    # R2-c, on the R2-a set and on the own / wrong sets
    out["R2-c"] = {}
    for key in ("random", "own", "wrong"):
        mr, ma, nr, na = group_medians(P[key]["bos"], spread(P[key]))
        bmr, bma, _, _ = group_medians(B[key]["bos"], spread(B[key]))
        ratio = summary(ma / mr, bma / bmr)
        out["R2-c"][key] = {"median_rest": float(mr), "median_active": float(ma), "n_rest": int(nr), "n_active": int(na),
                            "ratio_active_over_rest": ratio}
    return out


# ---------------------------------------------------------------- exploratory topology

def twonn(X: np.ndarray) -> float:
    """TwoNN intrinsic dimension (Facco et al. 2017), MLE d = N / Σ log(r2 / r1); X without duplicate points."""
    d, _ = cKDTree(X).query(X, k=3)
    mu = d[:, 2] / d[:, 1]
    return float(len(X) / np.log(mu).sum())


def mst_total(X: np.ndarray) -> float:
    """Rips H0 total persistence = sum of the minimum-spanning-tree edge lengths (Prim's on the dense matrix)."""
    D = squareform(pdist(X))
    n = len(X)
    done = np.zeros(n, dtype=bool)
    done[0] = True
    best = D[0].copy()
    total = 0.0
    for _ in range(n - 1):
        cand = np.where(done, np.inf, best)
        j = int(np.argmin(cand))
        total += cand[j]
        done[j] = True
        np.minimum(best, D[j], out=best)
    return total


def topology(keep: dict[str, np.ndarray], stream_rms: np.ndarray) -> dict[str, np.ndarray]:
    """Per head of the R2-a set: TwoNN, H0 total persistence, participation ratio, PCA-20 variance kept, unique points."""
    out = {k: np.zeros((L, H)) for k in ("twonn", "h0_total", "participation", "pca20_var", "n_unique")}
    for l in range(L):
        for h in range(H):
            _, idx = np.unique(keep["z16"][:, l, h], axis=0, return_index=True)
            Y = keep["y"][np.sort(idx), l, h] / stream_rms[l]
            Y = Y - Y.mean(0)
            _, S, Vt = np.linalg.svd(Y, full_matrices=False)
            lam = S ** 2
            X = Y @ Vt[:PCA_DIM].T
            out["participation"][l, h] = lam.sum() ** 2 / (lam ** 2).sum()
            out["pca20_var"][l, h] = lam[:PCA_DIM].sum() / lam.sum()
            out["n_unique"][l, h] = len(X)
            out["twonn"][l, h] = twonn(X)
            out["h0_total"][l, h] = mst_total(X)
    return out


# ---------------------------------------------------------------- checks

def checks(data: Path, b: str, F: np.ndarray, W_O: np.ndarray, res: dict, keep: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    s = np.load(data / f"seq_{b}.npz", mmap_mode="r")
    R = s["z"].shape[0]
    worst = 0.0
    for l, h in ((0, 2), (5, 1), (9, 8), (11, 8)):
        w = np.asarray(s["z"][:, 2, ..., l, h, :], np.float64).reshape(-1, DH) @ W_O[l, h].astype(np.float64)
        direct = np.sqrt(((w - w.mean(0)) ** 2).sum(-1).mean()) / np.sqrt(keep["xx"][:, l].mean())
        worst = max(worst, abs(direct - res["spread_random"][l, h]) / direct)
    out["spread from 768-d writes vs isometric sums (4 heads)"] = f"{'PASS' if worst < 1e-5 else 'FAIL'} (max rel |Δ| {worst:.1e})"
    share = res["share_random"]
    out["BOS share in [0, 1]"] = "PASS" if (share >= 0).all() and (share <= 1).all() else "FAIL"
    d = res["stream_own_vs_wrong_l0"]
    out["stream RMS at l = 0 identical, own vs wrong (built-in null)"] = f"{'PASS' if d < 1e-12 else 'FAIL'} (rel |Δ| {d:.1e})"
    X = keep["y"][:400, 9, 8]
    _, idx = np.unique(keep["z16"][:400, 9, 8], axis=0, return_index=True)
    X = X[np.sort(idx)]
    dg = ripser(X, maxdim=0)["dgms"][0]
    rip = float(dg[np.isfinite(dg[:, 1]), 1].sum())
    sp = float(minimum_spanning_tree(squareform(pdist(X))).sum())
    mine = mst_total(X)
    dev = max(abs(mine - rip), abs(mine - sp)) / mine
    out["H0 total persistence: Prim vs ripser vs scipy MST (L9H8, 400 points)"] = f"{'PASS' if dev < 1e-5 else 'FAIL'} (rel |Δ| {dev:.1e})"
    n_ex = int((res["topo"]["write_rms"] < 1e-6).sum())  # RMS ‖w‖ / RMS ‖x_l‖
    out["heads excluded for near-zero writes (RMS ‖w‖ < 1e-6 RMS ‖x_l‖)"] = f"{n_ex}"
    out["rows"] = f"{R}"
    return out


# ---------------------------------------------------------------- figures

def layer_legend(fig, ax, layers: range) -> None:
    cmap = ListedColormap(LAYER_COLORS[list(layers)])
    norm = BoundaryNorm(np.arange(layers.start - 0.5, layers.stop), cmap.N)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, ticks=list(layers), fraction=0.04, pad=0.02)
    cb.set_label("layer", color=INK)
    cb.outline.set_visible(False)


def label_heads(ax, x: np.ndarray, y: np.ndarray, heads: list[tuple[int, int]], l0: int = 0) -> None:
    for l, h in heads:
        if l < l0:
            continue
        xi, yi = x[l - l0, h], y[l - l0, h]
        ax.scatter([xi], [yi], s=46, facecolors="none", edgecolors=ORANGE, linewidths=1.2, zorder=4)
        ax.annotate(head_name(l, h), (xi, yi), xytext=(4, 3), textcoords="offset points", fontsize=7, color=INK, zorder=5)


def style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=8)
    ax.grid(alpha=0.25, linewidth=0.5)


def extra_heads(res: dict, n: int = 2) -> list[tuple[int, int]]:
    """The most extreme heads of the R2-a scatter not already in NOTABLE: highest spread among share ≥ 0.8, lowest
    spread among share ≤ 0.3."""
    sh, sp = res["share_random"], res["spread_random"]
    out = []
    for mask, sign in ((sh >= REST, -1), (sh <= ACTIVE, 1)):
        cand = [(sign * sp[l, h], (l, h)) for l, h in zip(*np.nonzero(mask)) if (l, h) not in NOTABLE]
        out += [hd for _, hd in sorted(cand)[:n]]
    return out


def plot_r2a(res: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    for ax, b in zip(axes, FILLERS):
        r = res[b]
        sh, sp = r["share_random"], r["spread_random"]
        for l in range(L):
            ax.scatter(sh[l], sp[l], s=22, color=LAYER_COLORS[l], edgecolors=INK, linewidths=0.3, zorder=3)
        label_heads(ax, sh, sp, list(NOTABLE) + extra_heads(r))
        ax.set_yscale("log")
        ax.set_xlim(-0.03, 1.03)
        ax.set_xlabel("BOS share (mean attention from T to BOS)", color=INK)
        ax.set_ylabel("write spread / RMS ‖x_l(T)‖", color=INK)
        ax.set_title(f"{b} filler: ρ = {fmt(r['R2-a'])}", fontsize=10, color=INK, loc="left")
        style(ax)
    layer_legend(fig, axes[-1], range(L))
    fig.suptitle("R2-a: BOS share vs write spread, 144 heads, random final token", color=INK, fontsize=11, x=0.01, ha="left")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_r2b(res: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    for ax, b in zip(axes, FILLERS):
        r = res[b]
        dx, dy = r["d_share"], r["d_spread"]
        ax.axhline(0, color=MUTED, linewidth=0.8)
        ax.axvline(0, color=MUTED, linewidth=0.8)
        for k in range(L - LATE):
            ax.scatter(dx[k], dy[k], s=24, color=LAYER_COLORS[LATE + k], edgecolors=INK, linewidths=0.3, zorder=3)
        big = np.argsort(-np.abs(dx).ravel())[:4]
        heads = list(NOTABLE) + [(LATE + i // H, i % H) for i in big if (LATE + i // H, i % H) not in NOTABLE]
        label_heads(ax, dx, dy, heads, l0=LATE)
        ax.set_xlabel("Δ BOS share (own − wrong capital)", color=INK)
        ax.set_ylabel("Δ write spread (own − wrong) / RMS ‖x_l(T)‖", color=INK)
        ax.set_title(f"{b} filler: ρ = {fmt(r['R2-b'])}", fontsize=10, color=INK, loc="left")
        style(ax)
    layer_legend(fig, axes[-1], range(LATE, L))
    fig.suptitle("R2-b: does waking up add variety? Layers 6–11 (72 heads)", color=INK, fontsize=11, x=0.01, ha="left")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_explore(res: dict, path: Path) -> None:
    cols = (("twonn", "TwoNN intrinsic dimension", False), ("participation", "PCA participation ratio", False),
            ("h0_total", "Rips H0 total persistence (PCA-20)", True), ("write_rms", "RMS ‖w‖ / RMS ‖x_l‖ (no folded constant)", True))
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), constrained_layout=True)
    for row, b in enumerate(FILLERS):
        r = res[b]
        for col, (key, name, logy) in enumerate(cols):
            ax = axes[row, col]
            sh, v = r["share_random"], r["topo"][key]
            for l in range(L):
                ax.scatter(sh[l], v[l], s=16, color=LAYER_COLORS[l], edgecolors=INK, linewidths=0.25, zorder=3)
            label_heads(ax, sh, v, list(NOTABLE))
            if logy:
                ax.set_yscale("log")
            ax.set_title(f"{b}: {name}\nρ vs BOS share {r['explore_rho'][key]:+.2f}", fontsize=9, color=INK, loc="left")
            ax.set_xlabel("BOS share", color=INK, fontsize=8)
            style(ax)
    layer_legend(fig, axes[:, -1], range(L))
    fig.suptitle("Exploratory: shape of each head's write cloud vs BOS share (random final token)", color=INK,
                 fontsize=11, x=0.01, ha="left")
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- report

def verdicts(res: dict) -> list[dict]:
    rows = []
    for hid, key in (("R2-a", "R2-a"), ("R2-b", "R2-b")):
        ok = all(res[b][key]["hi"] < 0 for b in FILLERS)
        rows.append({"id": hid, "verdict": "PASS" if ok else "FAIL",
                     "short": "; ".join(f"{b} ρ = {fmt(res[b][key])}" for b in FILLERS)})
    c = "; ".join(f"{b}: rest {res[b]['R2-c']['random']['median_rest']:.4f} (n {res[b]['R2-c']['random']['n_rest']}) vs "
                  f"active {res[b]['R2-c']['random']['median_active']:.4f} (n {res[b]['R2-c']['random']['n_active']})"
                  for b in FILLERS)
    rows.append({"id": "R2-c", "verdict": "descriptive", "short": c})
    return rows


def head_row(res: dict, b: str, l: int, h: int) -> str:
    r = res[b]
    P = r["P"]
    late = l >= LATE
    ds = f"{r['d_share'][l - LATE, h]:+.3f}" if late else "—"
    dp = f"{r['d_spread'][l - LATE, h]:+.4f}" if late else "—"
    return (f"| {head_name(l, h)} | {b} | {P['random']['bos'][l, h]:.3f} | {P['own']['bos'][l, h]:.3f} | "
            f"{P['wrong']['bos'][l, h]:.3f} | {r['spread_random'][l, h]:.4f} | {ds} | {dp} | {r['topo']['twonn'][l, h]:.1f} | "
            f"{r['topo']['write_rms'][l, h]:.3f} | {r['topo']['mean_norm'][l, h]:.3f} |")


def write_report(run: Path, res: dict, rows: list[dict], chk: dict, data: Path, boot: int, elapsed: float) -> None:
    R = res[FILLERS[0]]["R"]
    lines = ["# Experiment 5, R2: the no-op test", "",
             f"Run `{run.name}` on `{data.name}` ({R} rows per filler type), {boot} row-bootstrap resamples, "
             f"{elapsed:.0f} s. Hypotheses: `exp5/PREDICTIONS.md`; operational choices: `exp5/PREDICTIONS_R2.md` "
             "(written before this script read the full data). Code: `exp5/r2_noop.py`.", "",
             "Write of head (l, i) at T: w = z W_O[l, i] (processed weights). Spread = RMS distance of the writes from their "
             "mean / RMS norm of x_l(T), over a set of sequences. BOS share = mean attention from T to BOS.", "",
             "## Verdicts", "", "| id | verdict | short |", "|---|---|---|"]
    lines += [f"| {r['id']} | {r['verdict']} | {r['short']} |" for r in rows]
    lines += ["", "Pass rule: the row-bootstrap 95% CI of Spearman ρ is below 0 in both filler types.", "",
              "## R2-a: BOS share vs write spread (144 heads, random final token)", "",
              "| quantity | " + " | ".join(FILLERS) + " |", "|---|---|---|"]
    for key, name in (("R2-a", "**ρ (primary)**"), ("R2-a + controls", "ρ, S plus the random controls (sensitivity)"),
                      ("fp16-corrected", "ρ, spread minus expected fp16 rounding variance (check)"),
                      ("unnormalised", "ρ, unnormalised spread (exploratory)"),
                      ("within-row", "ρ, within-row spread: only the country / τ varies (exploratory)"),
                      ("within-layer", "mean within-layer ρ, 12 heads each (exploratory)"),
                      ("partial (layer)", "partial ρ controlling for layer (exploratory)")):
        lines.append(f"| {name} | " + " | ".join(fmt(res[b][key]) for b in FILLERS) + " |")
    lines += ["", "**Post hoc (added after the first full run, `20261025-172046_exp5R2`; not in the addendum).** How much of "
              "R2-a is the mechanical end (BOS share near 1 writes a near-constant vector by construction) or layer 0, and "
              "is the fall in spread more than dilution? The BOS term of a write is the same vector in every sequence, so "
              "if the rest of a head's attention wrote as variably as any other head's, spread would scale with "
              "1 − share; ρ of spread / (1 − share) tests what is left. Head subsets fixed from the point estimates.", "",
              "| post hoc | " + " | ".join(f"{b} ρ [CI] (n heads)" for b in FILLERS) + " |", "|---|---|---|"]
    for key in res[FILLERS[0]]["posthoc"]:
        if key.startswith("n_share"):
            continue
        lines.append(f"| {key} | " + " | ".join(f"{fmt(res[b]['posthoc'][key])} ({res[b]['posthoc'][key]['n']})"
                                               for b in FILLERS) + " |")
    lines.append("")
    lines.append("Heads with BOS share > 0.9: " + ", ".join(
        f"{b} {res[b]['posthoc']['n_share_gt_0.9']['n']} (" + ", ".join(
            f"{head_name(l, h)} {res[b]['share_random'][l, h]:.2f}" for l, h in zip(*np.nonzero(res[b]['share_random'] > 0.9))) + ")"
        for b in FILLERS) + ".")
    lines += ["", "Within-layer ρ (12 heads each, point estimates):", "",
              "| layer | " + " | ".join(str(l) for l in range(L)) + " |", "|---" * (L + 1) + "|"]
    for b in FILLERS:
        lines.append(f"| {b} ρ | " + " | ".join(f"{v:+.2f}" for v in res[b]["layer_rho"]) + " |")
        lines.append(f"| {b} mean BOS share | " + " | ".join(f"{v:.2f}" for v in res[b]["share_by_layer"]) + " |")
    lines += ["", "## R2-b: Δ BOS share vs Δ spread, own − wrong capital (layers 6–11, 72 heads)", "",
              "| quantity | " + " | ".join(FILLERS) + " |", "|---|---|---|"]
    for key, name in (("R2-b", "**ρ (primary; one stream denominator per layer)**"),
                      ("R2-b separate denominators", "ρ, each condition's own denominator (sensitivity)"),
                      ("R2-b within-layer", "mean within-layer ρ (exploratory)"),
                      ("R2-b partial (layer)", "partial ρ controlling for layer (exploratory)"),
                      ("d_share_mean", "mean Δ BOS share over the 72 heads"),
                      ("d_spread_mean", "mean Δ spread over the 72 heads")):
        lines.append(f"| {name} | " + " | ".join(fmt(res[b][key]) for b in FILLERS) + " |")
    neg = {b: int((res[b]["d_share"] < 0).sum()) for b in FILLERS}
    lines += ["", "Heads with Δ BOS share < 0: " + ", ".join(f"{b} {neg[b]}/72" for b in FILLERS) + ".", "",
              "## R2-c: median spread, resting (BOS share ≥ 0.8) vs active (≤ 0.3) heads", "",
              "| set | filler | resting: median (n) | active: median (n) | active / resting [95% CI] |", "|---|---|---|---|---|"]
    for key in ("random", "own", "wrong"):
        for b in FILLERS:
            c = res[b]["R2-c"][key]
            lines.append(f"| {key}{' (primary)' if key == 'random' else ''} | {b} | {c['median_rest']:.4f} ({c['n_rest']}) | "
                         f"{c['median_active']:.4f} ({c['n_active']}) | {ratio_fmt(c['ratio_active_over_rest'])} |")
    lines += ["", "## Notable heads", "",
              "Share = BOS share; spread on the random set; Δ = own − wrong (layers 6–11 only); TwoNN on the random set; "
              "RMS ‖w‖ and ‖mean w‖ divided by RMS ‖x_l‖ (both exclude the folded constant).", "",
              "| head | filler | share random | share own | share wrong | spread | Δ share | Δ spread | TwoNN | RMS ‖w‖ | ‖mean w‖ |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for l, h in NOTABLE:
        for b in FILLERS:
            lines.append(head_row(res, b, l, h))
    lines += ["", "Extremes of the R2-a scatter (highest-spread resting heads, lowest-spread active heads):", ""]
    for b in FILLERS:
        lines.append(f"- {b}: " + ", ".join(f"{head_name(l, h)} (share {res[b]['share_random'][l, h]:.2f}, spread "
                                            f"{res[b]['spread_random'][l, h]:.4f})" for l, h in extra_heads(res[b])))
    lines += ["", "Largest |Δ BOS share| in layers 6–11:", ""]
    for b in FILLERS:
        dx, dy = res[b]["d_share"], res[b]["d_spread"]
        top = np.argsort(-np.abs(dx).ravel())[:6]
        lines.append(f"- {b}: " + ", ".join(f"{head_name(LATE + i // H, i % H)} (Δ share {dx.ravel()[i]:+.3f}, Δ spread "
                                            f"{dy.ravel()[i]:+.4f})" for i in top))
    lines += ["", "## Exploratory: topology and norms of each head's write cloud (random set; not pre-registered tests)", "",
              f"Writes in the isometric frame / the layer's stream RMS norm, exact duplicates removed, PCA to {PCA_DIM} "
              "dimensions for TwoNN and H0. Spearman ρ vs BOS share across the 144 heads (point estimates), and the "
              "partial ρ controlling for layer.", "",
              "| metric | " + " | ".join(f"{b} ρ | {b} partial ρ" for b in FILLERS) + " |", "|---|---|---|---|---|"]
    for key, name in (("twonn", "TwoNN intrinsic dimension"), ("participation", "PCA participation ratio (64-d)"),
                      ("h0_total", "Rips H0 total persistence"), ("write_rms", "RMS ‖w‖ / RMS ‖x_l‖"),
                      ("mean_norm", "‖mean w‖ / RMS ‖x_l‖")):
        lines.append(f"| {name} | " + " | ".join(f"{res[b]['explore_rho'][key]:+.3f} | {res[b]['explore_partial'][key]:+.3f}"
                                                  for b in FILLERS) + " |")
    for b in FILLERS:
        t = res[b]["topo"]
        lines.append("")
        lines.append(f"- {b}: PCA-{PCA_DIM} keeps a median {np.median(t['pca20_var']):.3f} of variance (min "
                     f"{t['pca20_var'].min():.3f}); unique points per head median {np.median(t['n_unique']):.0f} of "
                     f"{len(res[b]['keep_n'])} (min {t['n_unique'].min():.0f}). TwoNN range {t['twonn'].min():.1f}–"
                     f"{t['twonn'].max():.1f}.")
    lines += ["", "## Checks", "", "| check | " + " | ".join(FILLERS) + " |", "|---|---|---|"]
    for k in chk[FILLERS[0]]:
        lines.append(f"| {k} | " + " | ".join(chk[b][k] for b in FILLERS) + " |")
    for b in FILLERS:
        fr = res[b]["fp16_floor_ratio"]
        near = [(head_name(l, h), fr[l, h]) for l, h in zip(*np.nonzero(fr > 0.5))]
        lines.append("")
        lines.append(f"- fp16 floor, {b}: expected rounding noise / measured spread has median {np.median(fr):.3f}, max "
                     f"{fr.max():.3f}; heads above 0.5: " + (", ".join(f"{n} ({v:.2f})" for n, v in near) or "none") + ".")
    lines += ["", "## Caveats", "",
              "- The same 11 country–capital pairs throughout: CIs are over rows, so beyond 0 means consistent, not general.",
              "- Writes are TransformerLens's processed writes: mean-centred, and without each head's folded constant "
              "(β_ln1 W_V + b_V) W_O (moved into b_O). A constant offset does not change spread, so R2-a–c are unaffected, "
              "but the norms above are of the non-constant part only.",
              "- BOS's stream is identical in every sequence, so a head with BOS share exactly 1 writes an exactly constant "
              "vector; the extreme of R2-a is partly mechanical.",
              "- Depth confound: BOS share and the stream norm both change with depth. The within-layer and partial ρ rows "
              "show how much of R2-a survives within a layer.",
              "- The topology metrics see within-row neighbours first (the 33 sequences of a row differ only at τ), and for "
              "nearly constant heads the cloud's fine structure is at fp16 resolution (see the floor check).",
              "- Three τ values only (31, 16, 1) for the per-sequence clouds.", "",
              "## Files", "", "- `r2_stats.npz`: per-head statistics per filler (keys `<filler>_<name>`).",
              "- `r2_summary.json`: verdicts and every ρ with its CI.",
              "- `fig_r2a_scatter.png`, `fig_r2b_delta.png`, `fig_r2_explore.png`."]
    (run / "r2_report.md").write_text("\n".join(lines) + "\n")


def save_arrays(run: Path, res: dict) -> None:
    arr: dict[str, np.ndarray] = {}
    for b in FILLERS:
        r = res[b]
        for key in CONDS:
            P = r["P"][key]
            arr[f"{b}_bos_{key}"] = P["bos"].astype(np.float32)
            arr[f"{b}_spread_{key}"] = spread(P).astype(np.float32)
            arr[f"{b}_stream_rms_{key}"] = np.sqrt(P["xx"]).astype(np.float32)
        arr[f"{b}_d_share_late"], arr[f"{b}_d_spread_late"] = r["d_share"].astype(np.float32), r["d_spread"].astype(np.float32)
        arr[f"{b}_fp16_floor_ratio"] = r["fp16_floor_ratio"].astype(np.float32)
        arr[f"{b}_within_row_spread_random"] = r["within_row_spread"].astype(np.float32)
        for k, v in r["topo"].items():
            arr[f"{b}_{k}"] = v.astype(np.float32)
    np.savez(run / "r2_stats.npz", **arr)


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Experiment 5, R2: the no-op test")
    parser.add_argument("--data", default=DATA, type=Path, help="exp5res run folder")
    parser.add_argument("--results-dir", default=None, type=Path, help="override the results folder (dev runs)")
    parser.add_argument("--boot", default=BOOT, type=int)
    args = parser.parse_args(argv)
    data = args.data if args.data.is_absolute() else PROJECT_ROOT / args.data
    if not (data / "DONE").exists():
        raise SystemExit(f"{data} has no DONE marker")
    t0 = time.time()
    run = make_run_dir(args.results_dir or PROJECT_ROOT / "results", tag="exp5R2")
    shutil.copy2(PROJECT_ROOT / "exp5" / "PREDICTIONS.md", run / "PREDICTIONS.md")
    shutil.copy2(PROJECT_ROOT / "exp5" / "PREDICTIONS_R2.md", run / "PREDICTIONS_R2.md")
    save_versions(run, PROJECT_ROOT, code_dirs=("exp1", "exp2", "exp3", "exp4", "exp5"))
    log(f"run folder {run}", t0)
    W_O = np.load(data / "weights.npz")["W_O"]
    F = frames(W_O)
    res: dict[str, Any] = {}
    chk: dict[str, dict] = {}
    for b in FILLERS:
        sets, keep = load_filler(data, b, F)
        log(f"{b}: loaded", t0)
        r = analyse(sets, args.boot)
        r["centre"] = sets["random"]["centre"]
        log(f"{b}: bootstrap done", t0)
        P = r["P"]["random"]
        stream_rms = np.sqrt(P["xx"])
        topo = topology(keep, stream_rms)
        mean_w = P["md"] + r["centre"]
        topo["mean_norm"] = np.linalg.norm(mean_w, axis=-1) / stream_rms[:, None]
        topo["write_rms"] = np.sqrt(P["var"] + (mean_w ** 2).sum(-1)) / stream_rms[:, None]
        r["topo"] = topo
        r["explore_rho"] = {k: float(rho(r["share_random"].ravel(), v.ravel())) for k, v in topo.items()}
        r["explore_partial"] = {k: float(rho_partial(r["share_random"], v)) for k, v in topo.items()}
        r["keep_n"] = keep["xx"][:, 0]
        log(f"{b}: topology done", t0)
        chk[b] = checks(data, b, F, W_O, r, keep)
        res[b] = r
    rows = verdicts(res)
    save_arrays(run, res)
    keys = ("R2-a", "R2-a + controls", "fp16-corrected", "unnormalised", "within-row", "within-layer", "partial (layer)", "R2-b",
            "R2-b separate denominators", "R2-b within-layer", "R2-b partial (layer)", "d_share_mean", "d_spread_mean")
    summ = {"verdicts": rows, "checks": chk,
            **{b: {**{k: res[b][k] for k in keys}, "R2-c": res[b]["R2-c"], "posthoc": res[b]["posthoc"],
                   "explore_rho": res[b]["explore_rho"],
                   "explore_partial": res[b]["explore_partial"]} for b in FILLERS}}
    (run / "r2_summary.json").write_text(json.dumps(summ, indent=2) + "\n")
    plot_r2a(res, run / "fig_r2a_scatter.png")
    plot_r2b(res, run / "fig_r2b_delta.png")
    plot_explore(res, run / "fig_r2_explore.png")
    write_report(run, res, rows, chk, data, args.boot, time.time() - t0)
    log(f"done: {run}", t0)
    for r in rows:
        print(f"{r['id']}: {r['verdict']} ({r['short']})")
    return run


if __name__ == "__main__":
    main()
