"""TDA analysis 3: 1-D persistence of every attention head's distance curve (Exp 4 and 4b, saved arrays only).

Run:  uv run --offline python -m exp4.tda3_curves [--boot 2000] [--seed 0]

Each curve D(d), d = 1..31 (country arm; random / own-capital / wrong-capital final token, plus the fact contrast own - wrong),
is reduced to its sublevel and superlevel H0 persistence on the 31-point path graph (peaks and troughs, endpoints
included). A bar is significant when it beats a row-bootstrap null of mean-zero curves, per head, condition and filler, and
a 0.02 margin; the significant bars give each curve a shape class. See exp4/PREDICTIONS_tda3.md (written before any
persistence was computed). Writes results/<timestamp>_exp4tda3/: versions, checks, predictions, tda3_features.csv,
tda3_shapes.csv, figures and tda3_report.md.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # write figures to files, never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exp1.runlog import make_run_dir, save_versions
from exp4.run_controlled import BLUE, GREY, ORANGE, PROJECT_ROOT, head_name, row_differences

RANDOM_RUN = "results/20261024-000548_exp4ctrl/controlled.npz"
CAPITAL_RUN = "results/20261024-150407_exp4cap/capital.npz"
FILLERS = ("common", "rare")
CONDS = ("random", "own", "wrong", "fact")
T, NL, NH = 32, 12, 12
QS = (0.95, 0.99)
CLASSES = ("null", "plateau", "near-peak decay", "mid-peak", "far rise", "U-shape", "multi-modal")
FEATURE_CLASSES = CLASSES[2:]
CLASS_COLORS = dict(zip(CLASSES, ("#dcdad4", "#8d8a80", BLUE, "#1baf7a", ORANGE, "#8e5bd6", "#d6a21f")))
COND_COLORS = {"random": GREY, "own": "#1baf7a", "wrong": ORANGE, "fact": BLUE}
PREV_HEADS = [(1, 0), (2, 2), (2, 3), (2, 5), (2, 8), (2, 9), (3, 2), (3, 6), (3, 7), (3, 8), (4, 11)]
NAMED_HEADS = [(9, 8), (11, 8), (3, 0), (0, 2)]
PRIMARY = {"q": 0.99, "margin": 0.02, "rel": 0.5, "level": 0.02, "near": 3, "far": 25}
VARIANTS = {"primary": PRIMARY, "q95": PRIMARY | {"q": 0.95}, "no margin": PRIMARY | {"margin": 0.0},
            "rel 0.3": PRIMARY | {"rel": 0.3}, "rel 0.7": PRIMARY | {"rel": 0.7}}


def log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


# ---- persistence on a path graph ---------------------------------------------------------------------------------

def sublevel_bars(f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """f [N, n] -> (pers, death) [N, n]: the sublevel H0 bar born at vertex i (0 / nan where none).

    Elder rule on a path graph, vectorised over the N curves: vertices enter in increasing value; the components are
    intervals, stored at their two endpoints. Values sit on vertices and an edge enters at the larger end value, so
    the curve's two endpoints are ordinary vertices with one neighbour. The essential bar (global minimum, first
    occurrence) gets the curve's range as persistence."""
    N, n = f.shape
    rows = np.arange(N)
    order = np.argsort(f, axis=1, kind="stable")
    t_of = np.full((N, n + 2), n + 1)  # time each (padded) vertex enters; the padding never does
    t_of[rows[:, None], order + 1] = np.arange(n)
    fp = np.zeros((N, n + 2))
    fp[:, 1:n + 1] = f
    added = np.zeros((N, n + 2), bool)
    other = np.zeros((N, n + 2), np.int64)  # opposite end of the interval that has this vertex as an end
    root = np.zeros((N, n + 2), np.int64)  # the interval's oldest vertex (its birth)
    pers = np.zeros((N, n))
    death = np.full((N, n), np.nan)
    for t in range(n):
        v = order[:, t] + 1
        has_l, has_r = added[rows, v - 1], added[rows, v + 1]
        bl, br = np.where(has_l, root[rows, v - 1], 0), np.where(has_r, root[rows, v + 1], 0)
        a, b = np.where(has_l, other[rows, v - 1], v), np.where(has_r, other[rows, v + 1], v)
        left_older = t_of[rows, bl] < t_of[rows, br]  # a missing side has t = n + 1, so the present side wins
        survivor = np.where(has_l | has_r, np.where(left_older, bl, br), v)
        merge = has_l & has_r
        if merge.any():
            m = rows[merge]
            dying = np.where(left_older, br, bl)[merge]
            pers[m, dying - 1] = fp[m, v[merge]] - fp[m, dying]
            death[m, dying - 1] = fp[m, v[merge]]
        added[rows, v] = True
        other[rows, a], other[rows, b] = b, a
        root[rows, a], root[rows, b] = survivor, survivor
    top = order[:, 0]
    pers[rows, top] = f.max(axis=1) - f.min(axis=1)
    death[rows, top] = f.max(axis=1)
    return pers, death


def raw_bars(f: np.ndarray) -> dict[str, np.ndarray]:
    """f [N, n] -> peaks (superlevel H0 = sublevel H0 of -f) and troughs, persistence and death by birth vertex."""
    t_pers, t_death = sublevel_bars(f)
    p_pers, p_death = sublevel_bars(-f)
    return {"t_pers": t_pers, "t_death": t_death, "p_pers": p_pers, "p_death": -p_death}


def null_stats(rows: np.ndarray, boot: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """rows [R, D, L, H] -> (range, largest non-essential bar), each [boot, L, H], over mean-zero bootstrap curves.

    A resample's mean minus the full-sample mean equals counts @ (rows - mean) / R, so the whole null is one matmul."""
    R, D = rows.shape[:2]
    xc = (rows - rows.mean(axis=0)).reshape(R, -1)
    counts = rng.multinomial(R, np.full(R, 1 / R), size=boot).astype(np.float64)
    curves = (counts @ xc / R).reshape(boot, D, NL * NH).transpose(0, 2, 1).reshape(-1, D)
    rng_out, non_out = np.empty(len(curves)), np.empty(len(curves))
    for lo in range(0, len(curves), 30000):  # chunked: the merge needs ~8 arrays of N x 33
        c = curves[lo:lo + 30000]
        b = raw_bars(c)
        idx = np.arange(len(c))
        b["p_pers"][idx, c.argmax(axis=1)] = 0.0
        b["t_pers"][idx, c.argmin(axis=1)] = 0.0
        rng_out[lo:lo + 30000] = c.max(axis=1) - c.min(axis=1)
        non_out[lo:lo + 30000] = np.maximum(b["p_pers"].max(axis=1), b["t_pers"].max(axis=1))
    return rng_out.reshape(boot, NL, NH), non_out.reshape(boot, NL, NH)


# ---- shape of one curve --------------------------------------------------------------------------------------------

def analyse_curve(f: np.ndarray, rb: dict[str, np.ndarray], th_ess: float, th_non: float,
                  prm: dict[str, float]) -> dict[str, Any]:
    """f [D], rb: that curve's rows of raw_bars -> {"shape": {...}, "bars": [...]} (rules in PREDICTIONS_tda3.md)."""
    imax, imin = int(f.argmax()), int(f.argmin())
    rng_ = float(f[imax] - f[imin])
    gate = rng_ > th_ess and rng_ >= prm["margin"]
    med = float(np.median(f))
    sign = 1 if f[imax] - med >= med - f[imin] else -1
    dom_type = "peak" if sign > 0 else "trough"
    bars = []
    for typ, pers, death, ess in (("peak", rb["p_pers"], rb["p_death"], imax), ("trough", rb["t_pers"], rb["t_death"], imin)):
        for i in np.flatnonzero(pers > 1e-12):
            essential = bool(i == ess)
            thr = th_ess if essential else th_non
            sig = bool(gate and pers[i] > thr and pers[i] >= prm["margin"])
            role = ("dominant" if typ == dom_type else "baseline") if essential else "secondary"
            bars.append({"type": typ, "d": int(i) + 1, "pers": float(pers[i]), "birth": float(f[i]), "death": float(death[i]),
                         "essential": essential, "thr": float(thr), "sig": sig, "role": role})
    feats = [b for b in bars if b["sig"] and b["role"] != "baseline"]
    same = [b for b in feats if b["type"] == dom_type]
    level = float(f.mean())
    rho = rng_ / max(float(np.abs(f).max()), 1e-12)
    d_star = (imax if sign > 0 else imin) + 1
    near = [b for b in same if b["d"] <= prm["near"]]
    far = [b for b in same if b["d"] >= prm["far"]]
    if not gate or rho < prm["rel"]:
        cls = "plateau" if abs(level) >= prm["level"] else "null"
    elif near and far:
        cls = "U-shape"
    elif len(same) >= 2:
        cls = "multi-modal"
    elif d_star <= prm["near"]:
        cls = "near-peak decay"
    elif d_star >= prm["far"]:
        cls = "far rise"
    else:
        cls = "mid-peak"
    shape = {"class": cls, "sign": sign, "gate": bool(gate), "range": rng_, "rho": rho, "mean": level, "d_star": d_star,
             "n_features": len(feats), "n_same_sign_peaks": len(same),
             "n_interior": sum(2 <= b["d"] <= T - 2 for b in feats), "th_ess": float(th_ess), "th_non": float(th_non)}
    return {"shape": shape, "bars": bars}


def analyse_all(curves: dict, th: dict, prm: dict[str, float]) -> dict:
    """curves[(cond, filler)] = {"f": [144, D], "raw": raw_bars of it} -> res[(cond, filler)][(layer, head)]."""
    out: dict = {}
    for key, c in curves.items():
        e, n = th[key]["ess"][prm["q"]], th[key]["non"][prm["q"]]
        out[key] = {}
        for l, h in np.ndindex(NL, NH):
            i = l * NH + h
            out[key][(l, h)] = {"f": c["f"][i], **analyse_curve(c["f"][i], {k: v[i] for k, v in c["raw"].items()},
                                                                 e[l, h], n[l, h], prm)}
    return out


# ---- data -----------------------------------------------------------------------------------------------------------

def load_groups() -> dict[tuple[str, str], dict[str, np.ndarray]]:
    """(condition, filler) -> observed saved mean curve [D, L, H] and per-row curves [R, D, L, H] (for the null only)."""
    zr, zc = np.load(PROJECT_ROOT / RANDOM_RUN), np.load(PROJECT_ROOT / CAPITAL_RUN)

    def per_row(z, prefix: str) -> np.ndarray:
        m = z[f"{prefix}country_masses"]
        ext = {"control": z[f"{prefix}control"], "country": {"masses": m}, "word": {"masses": m}}
        return row_differences(ext, T)["country"]

    out: dict = {}
    for b in FILLERS:
        out["random", b] = {"obs": zr[f"{b}/country/mean"], "rows": per_row(zr, f"{b}/")}
        for c in ("own", "wrong"):
            out[c, b] = {"obs": zc[f"{c}/{b}/country/mean"], "rows": per_row(zc, f"{c}/{b}/")}
        out["fact", b] = {"obs": zc[f"{b}/fact/mean"], "rows": out["own", b]["rows"] - out["wrong", b]["rows"]}
    return out


# ---- checks on the code and the null -------------------------------------------------------------------------------

def gudhi_pairs(f: np.ndarray) -> list[tuple[float, float]]:
    import gudhi

    st = gudhi.SimplexTree()
    for i, v in enumerate(f):
        st.insert([i], filtration=float(v))
    for i in range(len(f) - 1):
        st.insert([i, i + 1], filtration=float(max(f[i], f[i + 1])))
    return sorted((round(b, 10), round(d, 10)) for dim, (b, d) in st.persistence() if dim == 0 and np.isfinite(d))


def validate_algorithm(seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    out: dict[str, Any] = {}
    for name, f in (("continuous", rng.normal(size=(1000, 31))), ("ties", rng.integers(0, 5, size=(1000, 31)).astype(float))):
        pers, death = sublevel_bars(f)
        bad = 0
        for c in range(len(f)):
            mine = [i for i in np.flatnonzero(pers[c] > 1e-12) if i != f[c].argmin()]
            mine = sorted((round(f[c, i], 10), round(death[c, i], 10)) for i in mine)
            bad += mine != gudhi_pairs(f[c])
        ess_ok = bool(np.allclose(pers[np.arange(len(f)), f.argmin(axis=1)], f.max(axis=1) - f.min(axis=1)))
        out[name] = {"curves": len(f), "mismatching_curves": int(bad), "essential_is_range": ess_ok}
    # endpoints are extrema: [3, 1, 2] has peaks at d = 1 (essential, 2) and d = 3 (1), a trough at d = 2 (essential, 2)
    b = raw_bars(np.array([[3.0, 1.0, 2.0]]))
    out["endpoints"] = bool(np.allclose(b["p_pers"][0], [2, 0, 1]) and np.allclose(b["t_pers"][0], [0, 2, 0]))
    return out


def check_means(groups: dict) -> dict[str, float]:
    return {f"{c}/{b}": float(np.abs(g["rows"].mean(axis=0) - g["obs"]).max()) for (c, b), g in groups.items()}


# ---- rendering ------------------------------------------------------------------------------------------------------

def feature_rows(res: dict) -> pd.DataFrame:
    recs = []
    for (cond, b), heads in res.items():
        for (l, h), r in heads.items():
            sig = [x for x in r["bars"] if x["sig"]]
            ranked = sorted((x for x in sig if x["role"] != "baseline"), key=lambda x: -x["pers"])
            rank = {id(x): k + 1 for k, x in enumerate(ranked)}
            for x in sig:
                recs.append({"condition": cond, "filler": b, "layer": l, "head": h, "name": head_name(l, h),
                             "type": x["type"], "d": x["d"], "persistence": round(x["pers"], 6),
                             "birth_value": round(x["birth"], 6), "death_value": round(x["death"], 6),
                             "essential": x["essential"], "role": x["role"], "threshold": round(x["thr"], 6),
                             "rank": rank.get(id(x)), "shape": r["shape"]["class"]})
    return pd.DataFrame(recs)


def shape_rows(res: dict, variants: dict[str, dict]) -> pd.DataFrame:
    recs = []
    for (cond, b), heads in res["primary"].items():
        for (l, h), r in heads.items():
            s = r["shape"]
            feats = "; ".join(f"{x['type']}@{x['d']}:{x['pers']:.4f}"
                              for x in sorted((x for x in r["bars"] if x["sig"] and x["role"] != "baseline"),
                                              key=lambda x: -x["pers"]))
            rec = {"condition": cond, "filler": b, "layer": l, "head": h, "name": head_name(l, h), "shape": s["class"],
                   "sign": s["sign"], "gate": s["gate"], "range": round(s["range"], 6), "rho": round(s["rho"], 4),
                   "mean": round(s["mean"], 6), "d_star": s["d_star"], "n_features": s["n_features"],
                   "n_same_sign_peaks": s["n_same_sign_peaks"], "n_interior": s["n_interior"],
                   "th_ess": round(s["th_ess"], 6), "th_non": round(s["th_non"], 6), "features": feats}
            for v in variants:
                if v != "primary":
                    rec[f"shape [{v}]"] = res[v][(cond, b)][(l, h)]["shape"]["class"]
            recs.append(rec)
    return pd.DataFrame(recs)


def mark_features(ax, f: np.ndarray, bars: list[dict], color: str, label_dominant: bool) -> None:
    """Significant features on a curve: a triangle at each, and a thin segment from its birth to its death level."""
    d = np.arange(1, len(f) + 1)
    for x in bars:
        if not x["sig"] or x["role"] == "baseline":
            continue
        ax.plot([x["d"], x["d"]], [x["birth"], x["death"]], color=color, lw=0.8, alpha=0.6)
        ax.plot(x["d"], x["birth"], marker="^" if x["type"] == "peak" else "v", color=color, ms=5, mec="white", mew=0.5)
        if label_dominant and x["role"] == "dominant":
            ax.annotate(f"{x['pers']:.3f}", (x["d"], x["birth"]), textcoords="offset points", xytext=(3, 4), fontsize=6,
                        color=color)
    ax.plot(d, f, color=color, lw=1.3)


def plot_shape_counts(res: dict, path: Path) -> None:
    fig, axes = plt.subplots(len(FILLERS), len(CONDS), figsize=(3.4 * len(CONDS), 2.7 * len(FILLERS)), sharey=True,
                             squeeze=False)
    layers = np.arange(NL)
    for r, b in enumerate(FILLERS):
        for c, cond in enumerate(CONDS):
            ax = axes[r, c]
            shapes = [[res[cond, b][(l, h)]["shape"] for h in range(NH)] for l in range(NL)]
            bottom = np.zeros(NL)
            for cls in CLASSES:
                for sign, hatch in ((1, None), (-1, "////")):
                    # null and plateau have no meaningful sign, so they are always drawn solid
                    n = np.array([sum(s["class"] == cls and (s["sign"] if cls in FEATURE_CLASSES else 1) == sign
                                      for s in row) for row in shapes], float)
                    if n.any():
                        ax.bar(layers, n, bottom=bottom, color=CLASS_COLORS[cls], hatch=hatch, edgecolor="white", lw=0.4,
                               width=0.85)
                        bottom += n
            ax.set_title(f"{cond}, {b}", fontsize=8)
            ax.set_xticks(layers)
            ax.tick_params(labelsize=6)
            ax.spines[["top", "right"]].set_visible(False)
            if r == len(FILLERS) - 1:
                ax.set_xlabel("layer", fontsize=7)
            if c == 0:
                ax.set_ylabel("heads (of 12)", fontsize=7)
    handles = [plt.Rectangle((0, 0), 1, 1, color=CLASS_COLORS[k]) for k in CLASSES]
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="0.4", hatch="////"))
    fig.legend(handles, list(CLASSES) + ["hatched = negative sign (a dip)"], loc="lower center", ncol=8, fontsize=7,
               frameon=False)
    fig.suptitle("Shape class of each head's D(d) curve by layer (persistence taxonomy, 99th-percentile null)", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_examples(res: dict, heads: list[tuple[int, int]], conds: tuple[str, ...], path: Path, ncols: int, title: str,
                  label_dominant: bool) -> None:
    panels = [(l, h, b) for l, h in heads for b in FILLERS]
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 2.9 * nrows), squeeze=False)
    for ax, (l, h, b) in zip(axes.flat, panels):
        ax.axhline(0, color="0.7", lw=0.6)
        for cond in conds:
            r = res[cond, b][(l, h)]
            mark_features(ax, r["f"], r["bars"], COND_COLORS[cond], label_dominant)
            sign = "" if r["shape"]["class"] in CLASSES[:2] or r["shape"]["sign"] > 0 else " (dip)"
            ax.plot([], [], color=COND_COLORS[cond], lw=1.3, label=f"{cond}: {r['shape']['class']}{sign}")
        # legend above the axes: the curves can sit anywhere inside, so no in-axes position is safe
        ax.legend(fontsize=5.5, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), handlelength=1.2,
                  borderaxespad=0.2)
        ax.set_title(f"{head_name(l, h)}, {b}", fontsize=8, loc="right", pad=24)
        ax.tick_params(labelsize=6)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("distance T − τ", fontsize=7)
    fig.suptitle(title, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---- pre-registered checks --------------------------------------------------------------------------------------------

def named_checks(res: dict) -> dict[str, tuple[bool, str]]:
    out: dict[str, list[tuple[bool, str]]] = {k: [] for k in ("C1", "C2", "C3", "C4", "C5")}
    for b in FILLERS:
        r = res["random", b][(9, 8)]
        bar = next((x for x in r["bars"] if x["type"] == "peak" and x["d"] == T - 1), None)
        out["C1"].append((bool(bar and bar["sig"]),
                          f"{b}: class {r['shape']['class']}, f(31) {r['f'][-1]:+.4f}, f(30) {r['f'][-2]:+.4f}; " + (
                              f"peak at 31 pers {bar['pers']:.4f} vs θ {bar['thr']:.4f} "
                              f"({'essential' if bar['essential'] else 'non-essential'}), sig {bar['sig']}"
                              if bar else "no peak born at d = 31")))
        for cond in ("random", "own", "wrong"):
            s = res[cond, b][(11, 8)]["shape"]
            out["C2"].append((s["class"] == "near-peak decay" and s["sign"] > 0,
                              f"{cond}/{b}: {s['class']} (sign {s['sign']:+d}, d* {s['d_star']}, R {s['range']:.3f}, "
                              f"{s['n_same_sign_peaks']} peaks)"))
        s = res["own", b][(3, 0)]["shape"]
        out["C3"].append((s["sign"] > 0 and s["gate"] and s["d_star"] >= PRIMARY["far"],
                          f"{b}: sign {s['sign']:+d}, gate {s['gate']}, d* {s['d_star']}, R {s['range']:.3f} "
                          f"(θ {s['th_ess']:.3f}), class {s['class']}, f(30) {res['own', b][(3, 0)]['f'][-2]:+.3f}, "
                          f"f(31) {res['own', b][(3, 0)]['f'][-1]:+.3f}"))
        r = res["own", b][(0, 2)]
        s = r["shape"]
        out["C4"].append((s["class"] == "plateau",
                          f"{b}: {s['class']} (R {s['range']:.3f}, θ {s['th_ess']:.3f}, ρ {s['rho']:.2f}, "
                          f"mean {s['mean']:+.3f}, {s['n_interior']} significant bars at 2 ≤ d ≤ 30)"))
    viol, closest = [], (0.0, "")
    for cond in ("random", "own", "wrong"):
        for b in FILLERS:
            for hd in PREV_HEADS:
                viol += [f"{head_name(*hd)} {cond}/{b} {x['type']}@{x['d']}:{x['pers']:.3f}"
                         for x in res[cond, b][hd]["bars"] if x["sig"] and x["role"] != "baseline" and x["d"] >= 9]
                for x in res[cond, b][hd]["bars"]:  # the non-essential bar at d >= 9 that comes closest to its threshold
                    if x["d"] >= 9 and not x["essential"] and x["pers"] / x["thr"] > closest[0]:
                        closest = (x["pers"] / x["thr"], f"{head_name(*hd)} {cond}/{b} {x['type']}@{x['d']}: "
                                   f"{x['pers']:.4f} vs θ_non {x['thr']:.4f}")
    out["C5"].append((not viol, f"{len(viol)} bars at d ≥ 9 among 66 curves"
                      + (": " + "; ".join(viol[:10]) if viol else "")
                      + f"; closest non-essential bar to its null threshold: {closest[1]}"))
    return {k: (all(ok for ok, _ in v), " | ".join(d for _, d in v)) for k, v in out.items()}


def descriptive(res: dict) -> dict[str, tuple[bool, str]]:
    def count(cond: str, b: str, pred) -> int:
        return sum(pred(res[cond, b][hd]["shape"]) for hd in np.ndindex(NL, NH))

    e1 = [count("random", b, lambda s: s["class"] in CLASSES[:2]) for b in FILLERS]
    e2 = [(count("own", b, lambda s: s["n_features"] > 0), count("random", b, lambda s: s["n_features"] > 0)) for b in FILLERS]
    late = [(9, 8), (10, 3), (11, 7), (11, 2)]
    e3 = {b: [res["random", b][hd]["shape"]["class"] for hd in late] for b in FILLERS}
    return {"E1": (all(n > 72 for n in e1), f"null or plateau in the random condition: {e1[0]} / {e1[1]} of 144 (common / rare)"),
            "E2": (all(o > r for o, r in e2), "heads with ≥ 1 significant feature, own vs random: "
                   + ", ".join(f"{b} {o} vs {r}" for b, (o, r) in zip(FILLERS, e2))),
            "E3": (all(len(set(v)) > 1 for v in e3.values()),
                   "L9H8 / L10H3 / L11H7 / L11H2, random: " + "; ".join(f"{b}: {', '.join(v)}" for b, v in e3.items()))}


# ---- report -------------------------------------------------------------------------------------------------------------

def md_table(header: list[str], rows: list[list]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(map(str, r)) + " |" for r in rows]


def count_table(res: dict) -> list[list]:
    rows = []
    for cond in CONDS:
        for b in FILLERS:
            shapes = [res[cond, b][hd]["shape"] for hd in np.ndindex(NL, NH)]
            cells = []
            for k in CLASSES:
                n = sum(s["class"] == k for s in shapes)
                neg = sum(s["class"] == k and s["sign"] < 0 for s in shapes)
                cells.append(f"{n} ({neg})" if k in FEATURE_CLASSES else n)  # (n) = how many of them are dips
            rows.append([cond, b] + cells)
    return rows


def nondecay_table(res: dict) -> list[list]:
    """Heads whose class in some filler is neither null, plateau nor a near-peak decay."""
    rows = []
    for cond in CONDS:
        for hd in np.ndindex(NL, NH):
            s = [res[cond, b][hd]["shape"] for b in FILLERS]
            if any(x["class"] in CLASSES[3:] for x in s):
                cell = [f"{x['class']}{'' if x['class'] in CLASSES[:2] or x['sign'] > 0 else ' (dip)'}, d* {x['d_star']}"
                        for x in s]
                rows.append([cond, head_name(*hd), *cell])
    return rows


def ranking_tables(res: dict, top: int = 10) -> list[str]:
    L = []
    for cond in CONDS:
        rows = []
        for hd in np.ndindex(NL, NH):
            s = [res[cond, b][hd]["shape"] for b in FILLERS]
            if all(x["gate"] for x in s):
                rows.append((min(x["range"] for x in s), hd, s))
        rows.sort(key=lambda r: -r[0])
        L += ["", f"**{cond}** (heads whose curve passes the gate in both fillers: {len(rows)}; ranked by the smaller of the "
                  f"two ranges)", ""]
        L += md_table(["head", "R common", "R rare", "class common", "class rare", "d* c / r", "features c / r"],
                      [[head_name(*hd), f"{s[0]['range']:.3f}", f"{s[1]['range']:.3f}", s[0]["class"], s[1]["class"],
                        f"{s[0]['d_star']} / {s[1]['d_star']}", f"{s[0]['n_features']} / {s[1]['n_features']}"]
                       for _, hd, s in rows[:top]])
    return L


def write_report(run: Path, args: argparse.Namespace, checks: dict, calib: dict, res: dict, named: dict[str, dict],
                 desc: dict, thr_summary: dict) -> None:
    prim = res["primary"]
    L = ["# TDA analysis 3: 1-D persistence of each head's distance curve", "",
         f"Run `{run.name}`. {NL * NH} heads × {len(CONDS)} conditions × {len(FILLERS)} fillers = "
         f"{NL * NH * len(CONDS) * len(FILLERS)} curves of 31 points (d = 1..31). Saved arrays only: observed curves are the "
         f"saved means of `{RANDOM_RUN}` and `{CAPITAL_RUN}`; per-row curves feed the null. Predictions: "
         f"`exp4/PREDICTIONS_tda3.md` (written before any persistence was computed). Null: {args.boot} row-bootstrap mean-zero "
         f"curves per head × condition × filler, seed {args.seed}; bars significant at the 99th percentile of the null and "
         f"≥ 0.02.", "", "## Checks", ""]
    a = checks["algorithm"]
    L += md_table(["check", "result"], [
        ["bars vs gudhi SimplexTree, continuous curves", f"{a['continuous']['mismatching_curves']} of {a['continuous']['curves']} curves differ"],
        ["bars vs gudhi SimplexTree, curves with ties", f"{a['ties']['mismatching_curves']} of {a['ties']['curves']} curves differ"],
        ["essential bar = range", f"{a['continuous']['essential_is_range'] and a['ties']['essential_is_range']}"],
        ["endpoint extrema, hand-built [3, 1, 2]", f"{a['endpoints']}"],
        ["max |recomputed row mean − saved mean|", "; ".join(f"{k} {v:.1e}" for k, v in checks["mean_match"].items())],
        ["held-out null: share of fresh null curves with range > θ_ess (expect 5% / 1%)",
         f"{calib['range'][0.95]:.3f} / {calib['range'][0.99]:.3f}"],
        ["held-out null: share with a non-essential bar > θ_non (expect 5% / 1%)",
         f"{calib['non'][0.95]:.3f} / {calib['non'][0.99]:.3f}"]])
    L += ["", "## Verdicts on the pre-registered checks", ""]
    L += md_table(["id", "verdict", "numbers", "under the sensitivity settings"],
                  [[k, "PASS" if ok else "FAIL", d,
                    ", ".join(f"{v}: {'PASS' if named[v][k][0] else 'FAIL'}" for v in VARIANTS if v != "primary")
                    + (" (fragile)" if any(named[v][k][0] != ok for v in VARIANTS) else " (stable)")]
                   for k, (ok, d) in named["primary"].items()])
    L += ["", "Descriptive expectations (not pass/fail):", ""]
    L += md_table(["id", "held", "numbers"], [[k, "yes" if ok else "no", d] for k, (ok, d) in desc.items()])
    L += ["", "## Shape-class counts (primary settings; each row sums to 144)", "",
          "In the five feature classes, the number in brackets is how many of them have sign −, i.e. their extremum is a "
          "dip (\"far rise\" with sign − is attention that is lower than the control at the far end).", ""]
    L += md_table(["condition", "filler", *CLASSES], count_table(prim))
    L += ["", "### Heads with a mid-peak, far-rise, U-shape or multi-modal class in at least one filler", ""]
    L += md_table(["condition", "head", "common", "rare"], nondecay_table(prim))
    L += ["", "### Sensitivity of the counts (all conditions and fillers pooled, 1152 curves)", ""]
    L += md_table(["setting", *CLASSES],
                  [[v, *[sum(res[v][k][hd]["shape"]["class"] == c for k in res[v] for hd in np.ndindex(NL, NH))
                         for c in CLASSES]] for v in VARIANTS])
    L += ["", "Thresholds (median over the 144 heads, and the share of heads whose θ_ess is above the 0.02 margin, i.e. where "
               "the null rather than the margin sets the cut):", ""]
    L += md_table(["condition", "filler", "θ_ess q99", "θ_non q99", "θ_ess q95", "θ_non q95", "share θ_ess q99 > 0.02"],
                  [[c, b, *[f"{np.median(thr_summary[c, b][k][q]):.4f}" for q in (0.99, 0.95) for k in ("ess", "non")],
                    f"{(thr_summary[c, b]['ess'][0.99] > PRIMARY['margin']).mean():.2f}"]
                   for c in CONDS for b in FILLERS])
    L += ["", "## Ranking: curves with the largest significant range", "",
          "R is the range of the curve (the persistence of its dominant extremum); d* is the location of that extremum. "
          "Both fillers must pass the gate to be listed."]
    L += ranking_tables(prim)
    L += ["", "## Caveats", "",
          "- Each curve has 31 points, so features closer than a few points are barely resolvable and an interior peak needs "
          "support from both sides; \"interior\" means 2 ≤ d ≤ 30.",
          "- The null is a bootstrap of the mean curve with its shape removed. It assumes exchangeable rows and gives each "
          "head the noise level it actually has, which is highest where the effect is, so it is conservative for strong "
          "heads. It does not model the 11 fixed country/capital pairs: they are averaged inside each row, so "
          "pair-to-pair variation enters the row noise and is not independent sampling.",
          "- The essential bar has persistence equal to the range, so every curve with variation has an essential peak and "
          "an essential trough of equal size. The one on the side away from the median is treated as the baseline, not a "
          "feature. That choice is mine, stated before the run.",
          "- For most heads the 0.02 margin, not the bootstrap null, sets the cut (see the share column above). Without the "
          "margin the null class shrinks and many small end-point wiggles become U-shapes (sensitivity table), so \"noise-robust\" "
          "here means robust to sampling noise of the mean curve plus a size floor chosen by the repo's earlier convention, "
          "not by this null.",
          "- A class is about the mean curve's shape, not about attention magnitude: L0H2 with its own capital is a plateau "
          "because its slide from d = 1 to d = 31 is under half its height (ρ < 0.5), although that slide is many times its "
          "null threshold. The ρ rule was fixed before the run; the rel 0.3 / 0.7 rows show its influence.",
          "- At the 99th percentile roughly 1% of the 1152 curves (about 11) would pass the gate on noise alone; the "
          "held-out calibration above checks the rate on the null, not on real data, so any single borderline curve is "
          "uncertain.",
          "- The cut-offs 0.02, ρ = 0.5 and the zones 3 and 25 were fixed beforehand; the sensitivity rows show how much "
          "the counts move. The capital runs were stored as float16; the recomputed row means match the saved means "
          "to the precision shown in the checks, and only the null uses the per-row arrays."]
    (run / "tda3_report.md").write_text("\n".join(L) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    run = make_run_dir(PROJECT_ROOT / "results", tag="exp4tda3")
    save_versions(run, PROJECT_ROOT, code_dirs=("exp1", "exp2", "exp3", "exp4"))
    (run / "PREDICTIONS.md").write_text((PROJECT_ROOT / "exp4" / "PREDICTIONS_tda3.md").read_text())
    (run / "args.json").write_text(json.dumps(vars(args)) + "\n")
    log(f"run folder {run.name}", t0)

    algo = validate_algorithm(args.seed)
    log(f"algorithm vs gudhi: {algo}", t0)
    if algo["continuous"]["mismatching_curves"] or algo["ties"]["mismatching_curves"] or not algo["endpoints"]:
        raise AssertionError(f"persistence implementation disagrees with gudhi: {algo}")
    groups = load_groups()
    mean_match = check_means(groups)
    log(f"loaded {len(groups)} groups; max row-mean vs saved mean {max(mean_match.values()):.2e}", t0)

    curves: dict = {}
    th: dict = {}
    calib_rng: dict[str, dict[float, list[float]]] = {"range": {q: [] for q in QS}, "non": {q: [] for q in QS}}
    for k, (key, g) in enumerate(groups.items()):
        r_null, n_null = null_stats(g["rows"], args.boot, np.random.default_rng([args.seed, k]))
        th[key] = {"ess": {q: np.quantile(r_null, q, axis=0) for q in QS}, "non": {q: np.quantile(n_null, q, axis=0) for q in QS}}
        r2, n2 = null_stats(g["rows"], 500, np.random.default_rng([args.seed + 1000, k]))  # held-out null: calibration
        for q in QS:
            calib_rng["range"][q].append(float((r2 > th[key]["ess"][q]).mean()))
            calib_rng["non"][q].append(float((n2 > th[key]["non"][q]).mean()))
        f = g["obs"].reshape(T - 1, NL * NH).T
        curves[key] = {"f": f, "raw": raw_bars(f)}
        log(f"{key}: null done", t0)
    calib = {k: {q: float(np.mean(v)) for q, v in d.items()} for k, d in calib_rng.items()}

    res = {v: analyse_all(curves, th, prm) for v, prm in VARIANTS.items()}
    named = {v: named_checks(res[v]) for v in VARIANTS}
    desc = descriptive(res["primary"])
    checks = {"algorithm": algo, "mean_match": mean_match, "null_calibration": {k: {str(q): x for q, x in d.items()} for k, d in calib.items()}}
    (run / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    (run / "predictions.json").write_text(json.dumps(
        {"primary": {k: {"verdict": "PASS" if ok else "FAIL", "detail": d} for k, (ok, d) in named["primary"].items()},
         "sensitivity": {v: {k: ok for k, (ok, _) in named[v].items()} for v in VARIANTS if v != "primary"},
         "descriptive": {k: {"held": ok, "detail": d} for k, (ok, d) in desc.items()}}, indent=2, ensure_ascii=False) + "\n")

    feature_rows(res["primary"]).to_csv(run / "tda3_features.csv", index=False)
    shape_rows(res, VARIANTS).to_csv(run / "tda3_shapes.csv", index=False)
    fig_dir = run / "figures"
    fig_dir.mkdir()
    plot_shape_counts(res["primary"], fig_dir / "tda3_shape_counts.png")
    plot_examples(res["primary"], NAMED_HEADS, ("random", "own", "wrong"), fig_dir / "tda3_examples_named.png", 2,
                  "D(d) with significant peaks (▲) and troughs (▼); each segment runs from a feature's birth level to its\n"
                  "death level; the number is the persistence of the dominant feature", True)
    plot_examples(res["primary"], PREV_HEADS, ("random", "own", "wrong"), fig_dir / "tda3_examples_prevtoken.png", 6,
                  "Exp 1 previous-token heads: D(d) with significant features marked (▲ peak, ▼ trough)", False)
    write_report(run, args, checks, calib, res, named, desc, th)
    log(f"done: {run}", t0)
    for k, (ok, d) in named["primary"].items():
        print(f"{k}: {'PASS' if ok else 'FAIL'}  {d}")
    for k, (ok, d) in desc.items():
        print(f"{k}: {'held' if ok else 'not held'}  {d}")


if __name__ == "__main__":
    main()
