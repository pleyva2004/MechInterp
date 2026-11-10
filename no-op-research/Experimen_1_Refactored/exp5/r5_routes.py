"""Experiment 5, analysis R5: routes, not weights. Does the layer-0 matching-pair cycle that only the matching capital
closes (Exp 4 TDA 4, raw attention) survive when each attention edge is weighted by how much the head writes?

Run:  uv run --offline python -m exp5.r5_routes [--data results/20261025-170021_exp5res] [--n-boot 1000]

Pre-registration: exp5/PREDICTIONS.md (R5-a, R5-b); operational addendum: exp5/PREDICTIONS_R5.md. Reuses
exp4/tda4_graph.py (dist, summarize with its persistence and representative-cycle rules, _arcs) with the `tda`
thresholds of configs/exp4_tda4.yaml. Per-row graphs G_r [layer, 33, 33] (node 0 = BOS, removed before the filtration):
  raw  mean over heads of attention, as extracted (rows sum to 1);
  vw   sum over heads of attention x ||v_k W_O^h||, each query row divided by its sum (BOS and diagonal included), then
       averaged over rows (primary for R5-a);
  vwM  vw averaged over rows first, then row-normalised (ratio of means; sensitivity);
  vwU  vw un-normalised, divided by a per-(filler, layer) constant (mean row sum of the capital controls), clipped at 1.
Writes results/<timestamp>_exp5R5/: r5_report.md, r5.npz, r5.json, figures/, PREDICTIONS copies, versions.json.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write figures to files, never open a window
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Arc, Rectangle

import exp4.tda4_graph as tda4
from exp1.runlog import make_run_dir, save_versions
from exp3.run_relations import PROJECT_ROOT, log
from exp4.tda4_graph import BLUE, GREEN, GREY, ORANGE, _arcs, dist, jsonable, summarize

_rep_cycle = tda4.rep_cycle
TIE_FALLBACKS = [0]


def rep_cycle_ties(d: np.ndarray, u: int, v: int, b: float) -> list[int]:
    """tda4_graph.rep_cycle, tolerant of exact ties. The fp16 per-row graphs (and vwU's clipping) can tie an H1 birth
    edge with the edges that connect its endpoints, so no path exists strictly below b. Only then, allow edges with
    filtration ≤ b except the birth edge itself (the order among tied edges is arbitrary in gudhi too). Counted."""
    try:
        return _rep_cycle(d, u, v, b)
    except AssertionError:
        TIE_FALLBACKS[0] += 1
        d2 = d.copy()
        d2[u, v] = d2[v, u] = np.inf
        return _rep_cycle(d2, u, v, np.nextafter(b, np.inf))


tda4.rep_cycle = rep_cycle_ties  # summarize looks rep_cycle up in its module at call time (also in worker processes)

TAUS = (31, 30, 28, 24, 16, 8, 1)
T, L, K = 32, 12, 33
FILLERS = ("common", "rare")
CONDS = ("own", "wrong")  # bootstrapped
ALL_CONDS = ("own", "wrong", "random", "control")  # full sample
KINDS = ("raw", "vw", "vwM", "vwU")
KIND_LABEL = {"raw": "raw attention", "vw": "value-weighted, row-normalised", "vwM": "vw, normalised after mean",
              "vwU": "vw, un-normalised (/ c)"}
R5A_TAUS = (16, 8, 1)
DEFAULT_DATA = "results/20261025-170021_exp5res"
TDA4_RUN = "results/20261025-113614_exp4tda4"
CAP_RUN = "results/20261024-150407_exp4cap"


def row_norm(g: np.ndarray) -> np.ndarray:
    """Divide each query row (last axis = keys) by its sum."""
    s = g.sum(axis=-1, keepdims=True)
    return np.divide(g, s, out=np.zeros_like(g), where=s > 0)


def load_filler(data: Path, b: str) -> dict:
    """Per-row graphs as stored (fp16; converted to float64 slice by slice): raw / vw [cond] -> [R, 7, L, K, K]
    (control [R, L, K, K]), the vwU constant c [L] and the per-head layer-0 arrays (means over rows and cells)."""
    z = np.load(data / f"graphs_{b}.npz")
    out = {"raw": {}, "vw": {}}
    for kind in ("raw", "vw"):
        for c in ALL_CONDS:
            out[kind][c] = z[f"{kind}_{c}"]
    out["c"] = out["vw"]["control"][:, :, 1:, :].sum(axis=-1, dtype=np.float64).mean(axis=(0, 2))  # [L]
    out["head"] = {f"{kind}_{c}": z[f"head_{kind}_{c}"][:, 0].astype(np.float64) for kind in ("raw", "vw") for c in CONDS}
    return out


def kind_rows(raw: np.ndarray, vw: np.ndarray, c: np.ndarray) -> dict[str, np.ndarray]:
    """Stored per-row graphs [R, L, K, K] -> float64 graphs for raw, vw and vwU, plus the un-normalised vw sum that vwM
    averages first."""
    raw, vw = raw.astype(np.float64), vw.astype(np.float64)
    return {"raw": raw, "vw": row_norm(vw), "vwU": np.minimum(vw / c[:, None, None], 1.0), "vw_sum": vw}


def mean_graph(rows: dict[str, np.ndarray], kind: str, w: np.ndarray) -> np.ndarray:
    """w [B, R] row weights (each row sums to 1) -> mean graphs [B, L, K, K] of this kind."""
    x = rows["vw_sum" if kind == "vwM" else kind]
    g = (w @ x.reshape(x.shape[0], -1)).reshape(len(w), L, K, K)
    return row_norm(g) if kind == "vwM" else g


def boot_cell(job: dict) -> dict:
    """Bootstrap MPC persistence (NaN = no matching-pair cycle) [kind, cond, boot, layer] for one (filler, tau)."""
    tau, t, W = job["tau"], job["tda"], job["W"]
    ties0 = TIE_FALLBACKS[0]
    rows = {cond: kind_rows(job["raw"][cond], job["vw"][cond], job["c"]) for cond in CONDS}
    pers = np.full((len(KINDS), len(CONDS), len(W), L), np.nan, dtype=np.float32)
    for ki, kind in enumerate(KINDS):
        for ci, cond in enumerate(CONDS):
            g = mean_graph(rows[cond], kind, W)
            for s in range(len(W)):
                for l in range(L):
                    m = summarize(dist(g[s, l], False), tau, T, False, t)["mpc"]
                    if m is not None:
                        pers[ki, ci, s, l] = m["persistence"]
    return {"pers": pers, "ties": TIE_FALLBACKS[0] - ties0}


def loop_life(d: np.ndarray, tau: int) -> dict | None:
    """The longest-lived H1 class whose representative cycle uses E1 = {32, tau}, whatever the persistence rule says
    (descriptive: separates 'no loop through E1' from 'a loop that is filled too soon'). noBOS d; positions returned."""
    st = tda4._tree(d)
    e1, best = frozenset((tau - 1, T - 1)), None
    for bs, ds in st.persistence_pairs():
        if len(bs) != 2 or not ds:
            continue
        b, dd = st.filtration(bs), st.filtration(ds)
        cyc = rep_cycle_ties(d, bs[0], bs[1], b)
        if e1 in {frozenset((cyc[i], cyc[(i + 1) % len(cyc)])) for i in range(len(cyc))} and (
                best is None or dd - b > best["persistence"]):
            best = {"birth_w": 1 - b, "death_w": 1 - dd, "persistence": dd - b, "ratio": (1 - dd) / (1 - b),
                    "cycle": [p + 1 for p in cyc], "death_triangle": [p + 1 for p in ds]}
    return best


def boot_weights(R: int, n_boot: int, seed: int, fi: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.random.default_rng([seed, fi]).integers(0, R, size=(n_boot, R))
    W = np.stack([np.bincount(i, minlength=R) for i in idx]).astype(np.float64) / R
    return idx, W


def ci(boot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.quantile(boot, 0.025, axis=0), np.quantile(boot, 0.975, axis=0)


def boot_cat(fo: float, fw: float, t: dict) -> str:
    """Bootstrap-only category (addendum): O own-only, B both, W wrong-only, · neither, ~ unstable."""
    so, sw = fo >= t["boot_stable"], fw >= t["boot_stable"]
    ao, aw = fo <= t["boot_absent"], fw <= t["boot_absent"]
    if so and sw:
        return "B"
    if so and aw:
        return "O"
    if sw and ao:
        return "W"
    if ao and aw:
        return "·"
    return "~"


def tda4_symbol(fo: float, fw: float, full_o: bool, full_w: bool, t: dict) -> str:
    """TDA 4's report symbol: full-sample presence plus the bootstrap rule (exp4/tda4_graph.write_report)."""
    st = {"own": full_o and fo >= t["boot_stable"], "wrong": full_w and fw >= t["boot_stable"]}
    ab = {"own": (not full_o) and fo <= t["boot_absent"], "wrong": (not full_w) and fw <= t["boot_absent"]}
    if st["own"] and st["wrong"]:
        return "B"
    if st["own"] and ab["wrong"]:
        return "O"
    if st["wrong"] and ab["own"]:
        return "W"
    return ("o" if full_o else "") + ("w" if full_w else "") or "."


def tda4_grid(t: dict) -> dict:
    """TDA 4's final-run symbols per (filler, tau, layer), from its saved bootstrap frequencies and summaries."""
    run = PROJECT_ROOT / TDA4_RUN
    arr = np.load(run / "tda_arrays.npz")
    cells = json.loads((run / "tda4.json").read_text())["cells"]
    conds = list(arr["conds"])
    out = {}
    for b in FILLERS:
        for tau in TAUS:
            bm = arr[f"{b}/t{tau}/boot_mpc"]
            gr = cells[f"{b}/t{tau}"]["graphs"]
            for l in range(L):
                fo, fw = bm[conds.index("own"), l, 0], bm[conds.index("wrong"), l, 0]
                mo, mw = gr[f"own/L{l}/noBOS"]["mpc"], gr[f"wrong/L{l}/noBOS"]["mpc"]
                out[(b, tau, l)] = {"sym": tda4_symbol(fo, fw, mo is not None, mw is not None, t), "f_own": float(fo),
                                    "f_wrong": float(fw), "mpc_own": mo, "mpc_wrong": mw}
    return out


def data_identity(F: dict, R: int) -> float:
    """Max |Δ| between Exp 5's per-row raw query-32 weights on (tau, BOS, 31) and Exp 4b's saved arrays (country-paired
    design, first R rows, mean over heads)."""
    zc = np.load(PROJECT_ROOT / CAP_RUN / "capital.npz")
    worst = 0.0
    for b in FILLERS:
        for cond in CONDS:
            m = zc[f"{cond}/{b}/country_masses"][:R].astype(np.float64)  # [R, 31 (d - 1), L, H, 3]
            for ti, tau in enumerate(TAUS):
                ref = m[:, T - tau - 1].mean(axis=2)  # [R, L, 3]
                ours = F[b]["raw"][cond][:, ti, :, T][:, :, [tau, 0, T - 1]]
                worst = max(worst, float(np.abs(ours - ref).max()))
    return worst


def tda4_mean_delta(F: dict) -> float:
    """Max |Δ| between this run's raw mean graphs (R rows) and TDA 4's 200-row mean layer graphs (own, wrong)."""
    g = np.load(PROJECT_ROOT / TDA4_RUN / "graphs.npz")
    return max(float(np.abs(F[b]["raw"][c].mean(axis=0, dtype=np.float64) - g[f"{b}/{c}"][:, :L]).max())
               for b in FILLERS for c in CONDS)


def plot_freq(freq: np.ndarray, full: np.ndarray, b: str, n_boot: int, path: Path) -> None:
    """freq [tau, kind, cond, layer] bootstrap frequencies; full [tau, kind, cond(4), layer] full-sample MPC flags."""
    fig, axes = plt.subplots(2, len(KINDS), figsize=(3.1 * len(KINDS) + 1, 9.2), squeeze=False,
                             gridspec_kw={"hspace": 0.38})
    im = None
    j0 = TAUS.index(R5A_TAUS[0])
    for ci, cond in enumerate(CONDS):
        for ki, kind in enumerate(KINDS):
            ax = axes[ci, ki]
            grid = freq[:, ki, ci, :].T  # [layer, tau]
            im = ax.imshow(grid, aspect="auto", cmap="Greens", vmin=0, vmax=1)
            for l in range(L):
                for j in range(len(TAUS)):
                    v = grid[l, j]
                    if v >= 0.01:
                        ax.text(j, l, f"{100 * v:.0f}", ha="center", va="center", fontsize=6,
                                color="white" if v > 0.6 else "black")
            yy, xx = np.nonzero(full[:, ki, ci, :].T)
            ax.scatter(xx + 0.36, yy - 0.3, s=6, color="black", linewidths=0)
            ax.add_patch(Rectangle((j0 - 0.5, -0.5), len(R5A_TAUS), 1, fill=False, ec=ORANGE, lw=1.6))
            ax.set_xticks(range(len(TAUS)))
            ax.set_xticklabels(TAUS, fontsize=7)
            ax.set_yticks(range(L))
            ax.set_yticklabels([f"L{l}" for l in range(L)] if ki == 0 else [], fontsize=7)
            ax.set_xlabel("τ (country position)", fontsize=7)
            ax.set_title(f"{KIND_LABEL[kind]}\n{cond} capital at 32", fontsize=8)
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02,
                 label=f"share of {n_boot} row bootstraps with a matching-pair cycle (numbers in %; blank < 1%)")
    fig.suptitle(f"{b} filler: matching-pair cycles (persistent H1 through the edge {{32, τ}}), BOS removed. "
                 "Dot = present in the full-sample graph; orange box = the R5-a cells", fontsize=9)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_arcs(gm: dict, life: dict, t: dict, b: str, tau: int, path: Path) -> None:
    """Layer 0 at tau: raw vs vw (rows), own vs wrong (columns). Each kind is drawn at the birth weight of own's
    longest-lived loop through E1 (persistent or not; else at own's E1 weight); green = that loop, blue dashed = the
    death triangle's other edges (the chords that fill it, entering at the death weight)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.4), squeeze=False)
    for r, kind in enumerate(("raw", "vw")):
        lo_own = life[(kind, "own")]
        thr = lo_own["birth_w"] if lo_own else float(gm[(kind, "own")][T, tau])
        for c, cond in enumerate(CONDS):
            g, lf, ax = gm[(kind, cond)], life[(kind, cond)], axes[r, c]
            if lf:
                ok = lf["persistence"] >= t["persist_min"] and lf["ratio"] <= t["persist_ratio"]
                desc = (f"green = loop {lf['cycle']}: born at w {lf['birth_w']:.4f}\nfilled at w {lf['death_w']:.4f} by "
                        f"triangle {lf['death_triangle']}; death/birth {lf['ratio']:.2f} → "
                        f"{'persistent (an MPC)' if ok else 'NOT persistent (ratio > ' + str(t['persist_ratio']) + ')'}")
            else:
                desc = "no loop through E1"
            _arcs(ax, g, thr, tau, T, lf["cycle"] if lf else None,
                  f"{kind}, {cond} capital: edges with w ≥ {thr:.4f}; E1 weight {g[T, tau]:.4f}\n{desc}")
            if lf:
                cyc = {frozenset((lf["cycle"][i], lf["cycle"][(i + 1) % len(lf["cycle"])])) for i in range(len(lf["cycle"]))}
                tri = lf["death_triangle"]
                for i, j in ((tri[0], tri[1]), (tri[0], tri[2]), (tri[1], tri[2])):
                    if frozenset((i, j)) not in cyc:
                        ax.add_patch(Arc(((i + j) / 2, 0), abs(i - j), abs(i - j), theta1=0, theta2=180, color=BLUE,
                                         lw=1.8, ls="--", zorder=4))
    fig.suptitle(f"{b} filler, layer 0, country at τ = {tau}: mean graph over rows × 11 cells, BOS removed "
                 "(orange = E1; dotted = E1 below the threshold; blue dashed = chord(s) that fill the loop)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_e1(e1: dict, path: Path) -> None:
    """Left: E1 at tau = 16 per layer, raw vs vw, own vs wrong (95% row CI). Right: vw / raw ratio (own) per tau."""
    fig, axes = plt.subplots(len(FILLERS), 2, figsize=(12, 3.6 * len(FILLERS)), squeeze=False)
    x = np.arange(L)
    ti = TAUS.index(16)
    style = {("raw", "own"): (GREY, "-"), ("vw", "own"): (GREEN, "-"), ("raw", "wrong"): (GREY, ":"),
             ("vw", "wrong"): (GREEN, ":")}
    cmap = plt.get_cmap("viridis")
    for r, b in enumerate(FILLERS):
        ax = axes[r, 0]
        for (kind, cond), (col, ls) in style.items():
            d = e1[(b, kind, cond)]
            ax.errorbar(x, d["mean"][ti], yerr=[d["mean"][ti] - d["lo"][ti], d["hi"][ti] - d["mean"][ti]], color=col,
                        ls=ls, marker="o", ms=3, lw=1.2, capsize=2, label=f"{kind}, {cond}")
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([f"L{l}" for l in range(L)], fontsize=7)
        ax.set_ylabel("E1 = w(32, τ)", fontsize=8)
        ax.set_title(f"{b}: capital→country edge at τ = 16 (row-normalised weights)", fontsize=8, loc="left")
        ax.legend(fontsize=7, frameon=False, ncol=2)
        ax.spines[["top", "right"]].set_visible(False)
        ax = axes[r, 1]
        for j, tau in enumerate(TAUS):
            d = e1[(b, "ratio", "own")]
            ax.plot(x, d["mean"][j], color=cmap(j / (len(TAUS) - 1)), marker="o", ms=3, lw=1.1, label=f"τ {tau}")
        ax.axhline(1.0, color="0.6", lw=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([f"L{l}" for l in range(L)], fontsize=7)
        ax.set_ylabel("E1 vw / E1 raw (own)", fontsize=8)
        ax.set_title(f"{b}: how value weighting rescales the capital→country edge", fontsize=8, loc="left")
        ax.legend(fontsize=6, frameon=False, ncol=4)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def f2(x: float) -> str:
    return f"{x:.2f}"


def write_report(run: Path, ctx: dict) -> None:
    t, n_boot, R = ctx["tda"], ctx["n_boot"], ctx["R"]
    freq, full, summ, e1, ki = ctx["freq"], ctx["full"], ctx["summ"], ctx["e1"], {k: i for i, k in enumerate(KINDS)}
    san, verdicts = ctx["sanity"], ctx["verdicts"]
    Ln = ["# Exp 5, R5: routes, not weights (value-weighted attention graphs)", "",
          f"Run `{run.name}`. Data `{ctx['data']}` (Exp 5 extraction, GPT-2 small, {R} rows per filler, per-row graphs "
          "= means over the row's 11 capital-paired cells). Pre-registration `PREDICTIONS.md` (R5-a, R5-b); operational "
          "addendum `PREDICTIONS_R5.md`. Topology reused from `exp4/tda4_graph.py` (`dist`, `summarize`, `persistent`, "
          f"`rep_cycle`, `_arcs`) with TDA 4's thresholds (persist_min {t['persist_min']}, persist_ratio "
          f"{t['persist_ratio']}, stable ≥ {t['boot_stable']}, absent ≤ {t['boot_absent']}). BOS node removed. "
          f"{n_boot} row bootstraps per filler (same resamples for every τ, layer, kind and condition).", "",
          "Graph kinds: **raw** = mean over heads of attention; **vw** = Σ_h A_h[q, k] · ‖v_k W_O^h‖, each query row "
          "divided by its sum (BOS and diagonal included) per data row, then averaged (primary); **vwM** = vw averaged "
          "over rows, then row-normalised; **vwU** = vw un-normalised, divided by a per-(filler, layer) constant c (mean "
          "row sum of the capital controls), clipped at 1. MPC (matching-pair cycle) = a persistent H1 class whose "
          "representative cycle uses the edge {32, τ}.", "",
          f"Runtime: {ctx['runtime'] / 60:.1f} min (CPU){ctx['reused']}.", "", "## Checks", "", "| check | result |",
          "|---|---|"]
    Ln += [f"| {k} | {v} |" for k, v in ctx["checks"].items()]

    Ln += ["", "## Sanity check: TDA 4's layer-0 cycles with the raw graphs of these rows", "",
           f"- **Same graphs.** Exp 5 pairs cells by capital and TDA 4 by country, but each row's own set is the same 11 "
           f"sequences (country k, capital k) and its wrong set is the same 11 (country c, capital wrong[c]); the "
           f"per-row mean graphs are therefore the same objects. Check: Exp 5's per-row query-32 weights on (τ, BOS, 31) "
           f"match Exp 4b's saved country-paired arrays to max |Δ| {san['identity']:.1e} (fp16 storage) over 2 fillers × "
           f"2 conditions × 7 τ × 12 layers × {R} rows. So the pairing change does not matter here.",
           f"- Mean graphs over these {R} rows vs TDA 4's 200-row means: max |Δ| {san['mean_delta']:.4f}.",
           f"- **Layer 0, τ ∈ {{16, 8, 1}}, raw:** {san['l0_summary']}",
           f"- **Whole grid (12 layers × 7 τ × 2 fillers, TDA 4's symbol rule):** {san['agree']}/{san['n']} cells agree "
           "with TDA 4's final run." + (" Disagreements: " + "; ".join(san["disagree"]) + "." if san["disagree"] else ""),
           "", "| filler | τ | TDA 4 (200 rows): symbol, boot own/wrong, cycle | here (raw, " + f"{R} rows): symbol, boot "
           "own/wrong, cycle |", "|---|---|---|---|"]
    for b in FILLERS:
        for tau in R5A_TAUS:
            ref = ctx["tda4"][(b, tau, 0)]
            j = TAUS.index(tau)
            m = summ[(b, tau, "raw", "own")][0]["mpc"]
            Ln.append(f"| {b} | {tau} | {ref['sym']}, {f2(ref['f_own'])}/{f2(ref['f_wrong'])}, "
                      f"{ref['mpc_own']['cycle'] if ref['mpc_own'] else '—'} | {ctx['sym_raw'][(b, tau, 0)]}, "
                      f"{f2(freq[b][j, 0, 0, 0])}/{f2(freq[b][j, 0, 1, 0])}, {m['cycle'] if m else '—'} |")

    Ln += ["", "## R5-a: does the layer-0 loop survive value weighting?", "",
           f"Rule (pre-registered): layer 0, vw, f_own ≥ {t['boot_stable']} and f_wrong ≤ {t['boot_absent']} at τ 16, 8 "
           "and 1, both fillers.", "",
           f"**R5-a: {verdicts['vw']['verdict']}.** {verdicts['vw']['detail']}", "",
           "| filler | τ | f_own vw | f_wrong vw | full-sample vw own MPC: persistence, birth w → death w, cycle | full-sample vw "
           "wrong MPC | f_own raw | f_wrong raw |", "|---|---|---|---|---|---|---|---|"]
    for b in FILLERS:
        for tau in R5A_TAUS:
            j = TAUS.index(tau)
            mo, mw = summ[(b, tau, "vw", "own")][0]["mpc"], summ[(b, tau, "vw", "wrong")][0]["mpc"]
            Ln.append(f"| {b} | {tau} | {f2(freq[b][j, 1, 0, 0])} | {f2(freq[b][j, 1, 1, 0])} | " +
                      (f"{mo['persistence']:.4f}, {1 - mo['birth']:.4f} → {1 - mo['death']:.4f}, {mo['cycle']}" if mo else "none") +
                      f" | {'%.4f' % mw['persistence'] if mw else 'none'} | {f2(freq[b][j, 0, 0, 0])} | {f2(freq[b][j, 0, 1, 0])} |")
    Ln += ["", "Same rule, other variants (reported, not the verdict):", ""]
    for name, v in verdicts.items():
        if name != "vw":
            Ln.append(f"- {name}: {v['verdict']}. {v['detail']}")

    life, hub = ctx["life"], ctx["hub"]
    fl = lambda x: ("none" if x is None else f"{x['birth_w']:.4f} → {x['death_w']:.4f} ({x['ratio']:.2f}), "
                    f"filled by {x['death_triangle']}")
    Ln += ["", "### Loop or no loop? The longest-lived H1 class through E1, whatever the persistence rule (full sample)", "",
           f"Birth weight → death weight (death/birth ratio), and the triangle that fills it. The persistence rule needs "
           f"ratio ≤ {t['persist_ratio']} and death − birth ≥ {t['persist_min']} in 1 − w units. Descriptive.", "",
           "| filler | layer | τ | raw own | vw own | raw wrong | vw wrong |", "|---|---|---|---|---|---|---|"]
    for b in FILLERS:
        for l, taus_l in ((0, (24, 16, 8, 1)), (4, (28, 24))):
            for tau in taus_l:
                Ln.append(f"| {b} | L{l} | {tau} | " + " | ".join(fl(life[(b, tau, k, c, l)]) for c in CONDS
                                                                   for k in ("raw", "vw")) + " |")
    Ln += ["", "### Layer 0: how strongly later positions attach to the country (full sample, own)", "",
           "Mean weight from queries τ+1..31 to key τ (the country), to the same position in the capital control (the "
           "row's own token there), and to all other non-BOS keys of the same queries; and each one's vw / raw factor.",
           "", "| filler | τ | country raw → vw (×) | control token raw → vw (×) | other keys raw → vw (×) | country / other "
           "raw → vw |", "|---|---|---|---|---|---|"]
    for b in FILLERS:
        for tau in (24, 16, 8, 1):
            h = {(k, c): hub[(b, tau, k, c)] for k in ("raw", "vw") for c in ("own", "control", "other")}
            tri = lambda c: f"{h[('raw', c)]:.4f} → {h[('vw', c)]:.4f} (×{h[('vw', c)] / h[('raw', c)]:.2f})"
            Ln.append(f"| {b} | {tau} | {tri('own')} | {tri('control')} | {tri('other')} | "
                      f"{h[('raw', 'own')] / h[('raw', 'other')]:.2f} → {h[('vw', 'own')] / h[('vw', 'other')]:.2f} |")

    Ln += ["", "## R5-b (exploratory): which loops survive value weighting", "",
           "Bootstrap-only category per cell (addendum): O own-only (f_own ≥ 0.8, f_wrong ≤ 0.2), B both ≥ 0.8, W "
           "wrong-only, · both ≤ 0.2, ~ otherwise. Each cell shows raw → vw (vwM, vwU in brackets when they differ "
           "from vw).", ""]
    for b in FILLERS:
        Ln += [f"**{b}**", "", "| layer | " + " | ".join(f"τ {tau}" for tau in TAUS) + " |", "|---" * (len(TAUS) + 1) + "|"]
        for l in range(L):
            cells = []
            for j, tau in enumerate(TAUS):
                c = {k: boot_cat(freq[b][j, ki[k], 0, l], freq[b][j, ki[k], 1, l], t) for k in KINDS}
                extra = [f"{k} {c[k]}" for k in ("vwM", "vwU") if c[k] != c["vw"]]
                cells.append(f"{c['raw']} → {c['vw']}" + (f" ({', '.join(extra)})" if extra else ""))
            Ln.append(f"| L{l} | " + " | ".join(cells) + " |")
        Ln.append("")
    Ln += ["Raw loops (O or B), what value weighting does to them, and loops that appear only under vw:", ""]
    Ln += [f"- {x}" for x in ctx["survival"]] or ["- none"]
    Ln += ["", "Bootstrap frequencies f_own / f_wrong, layers 0–4:", ""]
    for b in FILLERS:
        Ln += [f"**{b}**", "", "| layer | kind | " + " | ".join(f"τ {tau}" for tau in TAUS) + " |",
               "|---|---" + "|---" * len(TAUS) + "|"]
        for l in range(5):
            for k in ("raw", "vw"):
                Ln.append(f"| L{l} | {k} | " + " | ".join(
                    f"{f2(freq[b][j, ki[k], 0, l])} / {f2(freq[b][j, ki[k], 1, l])}" for j in range(len(TAUS))) + " |")
        Ln.append("")

    Ln += ["## Capital→country edge E1 = w(32, τ): raw vs value-weighted (row-normalised)", "",
           "Mean over rows [95% row-bootstrap CI]. ratio = E1 vw / E1 raw (own; ratio of means). rank = E1's rank among "
           "the 496 non-BOS edges of the full-sample own graph (1 = heaviest), raw → vw. BOS share = weight of query 32 "
           "on BOS (own), raw → vw.", ""]
    for b in FILLERS:
        for tau in (16, 1, 31):
            j = TAUS.index(tau)
            Ln += [f"**{b}, τ = {tau}**", "", "| layer | E1 raw own | E1 vw own | ratio | E1 raw wrong | E1 vw wrong | "
                   "own − wrong raw | own − wrong vw | rank raw → vw | BOS share raw → vw |",
                   "|---|---|---|---|---|---|---|---|---|---|"]
            for l in range(L):
                g = lambda k, c: e1[(b, k, c)]
                cell = lambda d: f"{d['mean'][j, l]:.4f} [{d['lo'][j, l]:.4f}, {d['hi'][j, l]:.4f}]"
                Ln.append(f"| L{l} | {cell(g('raw', 'own'))} | {cell(g('vw', 'own'))} | {g('ratio', 'own')['mean'][j, l]:.2f} "
                          f"[{g('ratio', 'own')['lo'][j, l]:.2f}, {g('ratio', 'own')['hi'][j, l]:.2f}] | "
                          f"{g('raw', 'wrong')['mean'][j, l]:.4f} | {g('vw', 'wrong')['mean'][j, l]:.4f} | "
                          f"{cell(g('raw', 'diff'))} | {cell(g('vw', 'diff'))} | "
                          f"{ctx['rank'][(b, 'raw')][j, l]} → {ctx['rank'][(b, 'vw')][j, l]} | "
                          f"{ctx['bos'][(b, 'raw')][j, l]:.3f} → {ctx['bos'][(b, 'vw')][j, l]:.3f} |")
            Ln.append("")
    Ln += ["Median over τ of the own ratio vw / raw per layer: " + "; ".join(
        f"{b}: " + ", ".join(f"L{l} {np.median(e1[(b, 'ratio', 'own')]['mean'][:, l]):.2f}" for l in range(L)) for b in FILLERS), ""]

    Ln += ["## Layer 0: which heads carry E1 (own capital; means over rows and cells)", "",
           "Share of the layer-0 E1 weight per head, raw attention (A_h / Σ A) vs value-weighted (A_h ‖v W_O^h‖ / Σ). "
           "Top 4 heads.", "", "| filler | τ | raw | vw |", "|---|---|---|---|"]
    for b in FILLERS:
        for tau in R5A_TAUS:
            sr, sv = ctx["heads"][(b, tau)]
            top = lambda s: ", ".join(f"L0H{h} {s[h]:.2f}" for h in np.argsort(-s)[:4])
            Ln.append(f"| {b} | {tau} | {top(sr)} | {top(sv)} |")

    Ln += ["", "## Caveats", "",
           "- The same 11 country–capital pairs throughout; rows are resampled, pairs are not. A bootstrap frequency "
           "says the loop is consistent over these rows and pairs, not that it generalises to other pairs.",
           "- Topology of the mean graph (over rows × 11 cells), not of each sequence's attention.",
           "- Folded bias: TransformerLens folds each head's constant value term (β_ln1 W_V + b_V) W_O into b_O. "
           "The value norms ‖v_k W_O^h‖ exclude it, so vw measures the key-dependent part of each head's write. "
           "Attention rows sum to 1, so the folded part is added whatever the head attends to and carries no routing "
           "information, but ‖(v_k + b_V) W_O‖ ≠ ‖v_k W_O‖, so vw is not the norm of the full write. The writing "
           "weights are also mean-centred (TL default).",
           "- Norms ignore direction: Σ_k A_k ‖v_k W_O‖ ≥ ‖Σ_k A_k v_k W_O‖, and writes from different keys or heads "
           "can cancel. The vw graph is an upper bound on the norm actually moved along each route.",
           "- vw sums heads (raw averages them); row normalisation makes both have unit row sums, so the filtrations "
           "are on the same [0, 1] scale. vwU keeps each query's total write but its scale constant is a choice.",
           "- The persistence thresholds were calibrated on raw attention in TDA 4; vw weights on non-BOS keys are "
           "larger (the BOS share drops), so the absolute floor (0.002) is easier to clear; the ratio rule is "
           "scale-free.",
           "- Per-row graphs are stored in fp16; normalisation and filtrations are computed in float64.",
           "- Own and wrong graphs differ only in query row 32 (checked), so every own-vs-wrong topological difference "
           "is about how node 32 attaches to a fixed graph; under vw the normaliser of row 32 also depends on the "
           "capital's own value norm (the diagonal), which differs between own and wrong from layer 1 on.", "",
           "## Files", "",
           "- `r5.npz`: `freq` [filler, τ, kind, cond, layer] bootstrap MPC frequencies; `boot_pers` [filler, τ, kind, "
           "cond, boot, layer] MPC persistence (NaN = none); `full_mpc` [filler, τ, kind, cond(own, wrong, random, "
           "control), layer]; `e1_*` [filler, kind, τ, layer]; `rank_*`, `bos_*`; `heads_*`; `boot_idx`.",
           "- `r5.json`: verdicts, sanity check, checks, full-sample MPCs per cell, loop lifetimes through E1 (layers "
           "0–4), layer-0 hub weights, configuration.",
           "- `figures/freq_<filler>.png`, `figures/arcs_L0_t16_<filler>.png`, `figures/e1_ratio.png`.", ""]
    (run / "r5_report.md").write_text("\n".join(Ln))


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Exp 5, R5: value-weighted attention graphs")
    parser.add_argument("--data", default=DEFAULT_DATA, type=Path)
    parser.add_argument("--n-boot", default=1000, type=int)
    parser.add_argument("--workers", default=None, type=int)
    parser.add_argument("--results-dir", default=None, type=Path, help="override the results dir (dev runs)")
    parser.add_argument("--reuse-boot", default=None, type=Path,
                        help="previous R5 run folder whose bootstrap (same data, seed, n_boot) is reused")
    args = parser.parse_args(argv)
    data = args.data if args.data.is_absolute() else PROJECT_ROOT / args.data
    data_str = str(data.relative_to(PROJECT_ROOT) if data.is_relative_to(PROJECT_ROOT) else data)
    if not (data / "DONE").exists():
        raise SystemExit(f"{data} is not complete (no DONE)")
    t = yaml.safe_load((PROJECT_ROOT / "configs" / "exp4_tda4.yaml").read_text())["tda"]
    workers = args.workers or t["workers"]
    t0 = time.time()
    run = make_run_dir(args.results_dir or PROJECT_ROOT / "results", tag="exp5R5")
    shutil.copy2(PROJECT_ROOT / "exp5" / "PREDICTIONS.md", run / "PREDICTIONS.md")
    shutil.copy2(PROJECT_ROOT / "exp5" / "PREDICTIONS_R5.md", run / "PREDICTIONS_R5.md")
    save_versions(run, PROJECT_ROOT, code_dirs=("exp1", "exp2", "exp3", "exp4", "exp5"))
    (run / "figures").mkdir()
    log(f"run folder {run}", t0)

    F = {b: load_filler(data, b) for b in FILLERS}
    R = F[FILLERS[0]]["raw"]["own"].shape[0]
    log(f"loaded {R} rows per filler", t0)

    # Checks on the per-row graphs.
    upper = max(float(np.abs(np.triu(F[b][k][c], 1)).max()) for b in FILLERS for k in ("raw", "vw") for c in ALL_CONDS)
    # Own and wrong average the same 11 prefixes in a different cell order, so float32 sums can round to adjacent fp16
    # values: compare relative to the weight (one fp16 ulp is 2^-10 relative).
    prefix = {k: max(float((np.abs(F[b][k]["own"][..., :T, :].astype(np.float64) - F[b][k][c][..., :T, :])
                            / np.maximum(np.abs(F[b][k]["own"][..., :T, :].astype(np.float64)), 1e-30)).max())
                     for b in FILLERS for c in ("wrong", "random")) for k in ("raw", "vw")}
    rowsum = max(float(np.abs(F[b]["raw"][c].sum(axis=-1, dtype=np.float64) - 1).max()) for b in FILLERS for c in ALL_CONDS)
    lo_idx = np.tril_indices(K - 1, -1)  # the 496 non-BOS edges, as indices into positions 1..32
    clipped = {b: sum(int(((F[b]["vw"][c].astype(np.float64) / F[b]["c"][:, None, None])[..., 1:, 1:][..., lo_idx[0], lo_idx[1]]
                           > 1).sum()) for c in ALL_CONDS) for b in FILLERS}
    identity = data_identity(F, R)
    mean_delta = tda4_mean_delta(F)
    checks = {
        "causal: upper triangle of every per-row graph is 0 (raw, vw)": f"{'PASS' if upper == 0 else 'FAIL'} (max {upper:.1e})",
        "prefix identity: wrong / random query rows 0..31 equal own's (per row, up to one fp16 rounding step)":
            f"{'PASS' if max(prefix.values()) <= 2.0 ** -10 else 'FAIL'} (max relative |Δ| raw {prefix['raw']:.1e}, "
            f"vw {prefix['vw']:.1e}; tol 2^-10)",
        "raw rows sum to 1 (fp16 storage)": f"{'PASS' if rowsum < 2e-2 else 'FAIL'} (max |sum − 1| {rowsum:.1e})",
        "Exp 5 raw query-32 weights (τ, BOS, 31) = Exp 4b saved arrays (first rows; both fp16)":
            f"{'PASS' if identity <= 2e-3 else 'FAIL'} (max |Δ| {identity:.1e}, atol 2e-3)",
        "vwU: non-BOS off-diagonal per-row weights clipped at 1": ", ".join(f"{b} {v}" for b, v in clipped.items()),
        "vwU scale constants c (per layer)": "; ".join(f"{b}: " + ", ".join(f"{x:.0f}" for x in F[b]["c"]) for b in FILLERS),
    }

    # Full-sample summaries, E1 statistics and the bootstrap jobs.
    jobs, summ, gm_l0, e1, rank, bos, heads, Ws, idxs, life, hub = [], {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
    full = {b: np.zeros((len(TAUS), len(KINDS), len(ALL_CONDS), L), bool) for b in FILLERS}
    one = np.full((1, R), 1.0 / R)
    for fi, b in enumerate(FILLERS):
        idx, W = boot_weights(R, args.n_boot, t["seed"], fi)
        Ws[b], idxs[b] = W, idx
        c = F[b]["c"]
        per = {}  # per-row E1 and BOS share [R, 7, L] for raw / vw (only query row 32 is needed)
        for k in ("raw", "vw"):
            q = {cnd: F[b][k][cnd][:, :, :, T, :].astype(np.float64) for cnd in CONDS}  # [R, 7, L, K]
            if k == "vw":
                q = {cnd: row_norm(x) for cnd, x in q.items()}
            e_o = np.stack([q["own"][:, j, :, tau] for j, tau in enumerate(TAUS)], axis=1)
            e_w = np.stack([q["wrong"][:, j, :, tau] for j, tau in enumerate(TAUS)], axis=1)
            per[k] = {"own": e_o, "wrong": e_w, "diff": e_o - e_w, "bos": q["own"][..., 0]}
        for k in ("raw", "vw"):
            for cnd in ("own", "wrong", "diff"):
                x = per[k][cnd].reshape(R, -1)
                bm = (W @ x).reshape(-1, len(TAUS), L)
                lo, hi = ci(bm)
                e1[(b, k, cnd)] = {"mean": x.mean(axis=0).reshape(len(TAUS), L), "lo": lo, "hi": hi}
            bos[(b, k)] = per[k]["bos"].mean(axis=0)
        rb = (W @ per["vw"]["own"].reshape(R, -1)) / (W @ per["raw"]["own"].reshape(R, -1))
        lo, hi = ci(rb.reshape(-1, len(TAUS), L))
        e1[(b, "ratio", "own")] = {"mean": (per["vw"]["own"].mean(axis=0) / per["raw"]["own"].mean(axis=0)), "lo": lo, "hi": hi}
        for k in ("raw", "vw"):
            rank[(b, k)] = np.zeros((len(TAUS), L), int)
        for j, tau in enumerate(TAUS):
            rows = {cnd: kind_rows(F[b]["raw"][cnd][:, j], F[b]["vw"][cnd][:, j], c) for cnd in ("own", "wrong", "random")}
            rows["control"] = kind_rows(F[b]["raw"]["control"], F[b]["vw"]["control"], c)
            for k_i, kind in enumerate(KINDS):
                for c_i, cnd in enumerate(ALL_CONDS):
                    g = mean_graph(rows[cnd], kind, one)[0]
                    ss = [summarize(dist(g[l], False), tau, T, False, t) for l in range(L)]
                    summ[(b, tau, kind, cnd)] = ss
                    full[b][j, k_i, c_i] = [s["mpc"] is not None for s in ss]
                    if kind in ("raw", "vw") and cnd in CONDS:
                        for l in range(5):
                            life[(b, tau, kind, cnd, l)] = loop_life(dist(g[l], False), tau)
                    if kind in ("raw", "vw") and cnd in ("own", "control"):
                        g0 = g[0]  # layer 0: how strongly later queries attach to position tau
                        qs = np.arange(tau + 1, T)
                        hub[(b, tau, kind, cnd)] = float(g0[qs, tau].mean()) if len(qs) else np.nan
                        if cnd == "own":
                            oth = [g0[q, k] for q in qs for k in range(1, q) if k != tau]
                            hub[(b, tau, kind, "other")] = float(np.mean(oth)) if oth else np.nan
                    if kind in ("raw", "vw") and cnd == "own":
                        for l in range(L):
                            low = np.tril(g[l, 1:, 1:], -1)[lo_idx]
                            rank[(b, kind)][j, l] = int((low > g[l, T, tau]).sum()) + 1
                    if tau == 16 and cnd in CONDS:
                        gm_l0[(b, kind, cnd)] = g[0]
            if tau in R5A_TAUS:
                hs = [F[b]["head"][f"{k}_own"][j, :, T, tau] for k in ("raw", "vw")]
                heads[(b, tau)] = tuple(h / h.sum() for h in hs)
            jobs.append({"filler": b, "tau": tau, "W": W, "tda": t, "c": c,
                         "raw": {cnd: F[b]["raw"][cnd][:, j] for cnd in CONDS},
                         "vw": {cnd: F[b]["vw"][cnd][:, j] for cnd in CONDS}})
    ties_full = TIE_FALLBACKS[0]
    log(f"full-sample summaries done; {len(jobs)} bootstrap jobs, {args.n_boot} resamples, {workers} workers", t0)

    t1 = time.time()
    n_sum = len(FILLERS) * len(TAUS) * len(KINDS) * L
    if args.reuse_boot:
        # The bootstrap is deterministic given the data, seed and n_boot: reuse a previous run's MPC persistences
        # after checking that its resampled row indices and data are exactly this run's.
        prev = args.reuse_boot if args.reuse_boot.is_absolute() else PROJECT_ROOT / args.reuse_boot
        zp = np.load(prev / "r5.npz")
        pj = json.loads((prev / "r5.json").read_text())
        if not (np.array_equal(zp["boot_idx"], np.stack([idxs[b] for b in FILLERS])) and pj["n_boot"] == args.n_boot
                and pj["data"] == data_str and list(zp["kinds"]) == list(KINDS)):
            raise SystemExit(f"{prev}: bootstrap indices, n_boot, data or kinds differ; cannot reuse")
        boot_pers = {b: zp["boot_pers"][fi] for fi, b in enumerate(FILLERS)}
        old = pj["checks"][next(k for k in pj["checks"] if k.startswith("tie fallback"))]
        ties_boot = pj.get("ties_boot", int(re.search(r"bootstrap (\d+)", old).group(1)))
        reused = f" (bootstrap persistences reused from `{prev.name}`: identical row indices, data and n_boot)"
        log(f"bootstrap reused from {prev.name}", t0)
    else:
        pers, ties_boot, reused = {}, 0, ""
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(boot_cell, job): (job["filler"], job["tau"]) for job in jobs}
            for f in as_completed(futs):
                res = f.result()
                pers[futs[f]], ties_boot = res["pers"], ties_boot + res["ties"]
                log(f"bootstrap {futs[f][0]} τ {futs[f][1]} done", t0)
        boot_pers = {b: np.stack([pers[(b, tau)] for tau in TAUS]) for b in FILLERS}  # [tau, kind, cond, boot, L]
    checks["tie fallback in rep_cycle (H1 representatives whose birth edge ties the connecting edges)"] = (
        f"full sample {ties_full} (of {n_sum * len(ALL_CONDS)} graphs), bootstrap {ties_boot} "
        f"(of {n_sum * len(CONDS) * args.n_boot} graphs)" + reused)
    del jobs
    log(f"bootstrap: {(time.time() - t1) / 60:.1f} min", t0)
    freq = {b: np.isfinite(boot_pers[b]).mean(axis=3) for b in FILLERS}  # [tau, kind, cond, L]

    # Verdicts.
    verdicts = {}
    for kind in ("vw", "vwM", "vwU", "raw"):
        k_i = KINDS.index(kind)
        bad, parts = [], []
        for b in FILLERS:
            for tau in R5A_TAUS:
                j = TAUS.index(tau)
                fo, fw = freq[b][j, k_i, 0, 0], freq[b][j, k_i, 1, 0]
                ok = fo >= t["boot_stable"] and fw <= t["boot_absent"]
                parts.append(f"{b} τ {tau}: {fo:.3f}/{fw:.3f}")
                if not ok:
                    bad.append(f"{b} τ {tau}")
        strict = [f"{b} τ {tau}" for b in FILLERS for tau in R5A_TAUS
                  if tda4_symbol(freq[b][TAUS.index(tau), k_i, 0, 0], freq[b][TAUS.index(tau), k_i, 1, 0],
                                 full[b][TAUS.index(tau), k_i, 0, 0], full[b][TAUS.index(tau), k_i, 1, 0], t) == "O"]
        detail = f"f_own/f_wrong per cell: {'; '.join(parts)}."
        if bad:
            detail += f" Failing cells: {', '.join(bad)}."
        detail += f" TDA 4's stricter rule (full-sample MPC in own, none in wrong, too) holds in {len(strict)}/6 cells."
        verdicts[kind] = {"verdict": "PASS" if not bad else "FAIL", "detail": detail, "failing": bad}

    # Sanity check against TDA 4.
    ref = tda4_grid(t)
    sym_raw = {(b, tau, l): tda4_symbol(freq[b][j, 0, 0, l], freq[b][j, 0, 1, l], full[b][j, 0, 0, l], full[b][j, 0, 1, l], t)
               for b in FILLERS for j, tau in enumerate(TAUS) for l in range(L)}
    disagree = [f"{b} L{l} τ {tau}: TDA 4 {ref[(b, tau, l)]['sym']} ({f2(ref[(b, tau, l)]['f_own'])}/"
                f"{f2(ref[(b, tau, l)]['f_wrong'])}) vs here {s} ({f2(freq[b][TAUS.index(tau), 0, 0, l])}/"
                f"{f2(freq[b][TAUS.index(tau), 0, 1, l])})"
                for (b, tau, l), s in sym_raw.items() if s != ref[(b, tau, l)]["sym"]]
    l0 = [(b, tau) for b in FILLERS for tau in R5A_TAUS]
    l0_ok = all(sym_raw[(b, tau, 0)] == "O" for b, tau in l0)
    sanity = {"identity": identity, "mean_delta": mean_delta, "agree": len(sym_raw) - len(disagree), "n": len(sym_raw),
              "disagree": disagree, "l0_reproduced": l0_ok,
              "l0_summary": ("reproduced: own-only (O) in all 6 cells, as in TDA 4." if l0_ok else
                             "NOT fully reproduced: " + ", ".join(f"{b} τ {tau} {sym_raw[(b, tau, 0)]}" for b, tau in l0))}

    # Which raw loops survive value weighting.
    survival = []
    for b in FILLERS:
        for j, tau in enumerate(TAUS):
            for l in range(L):
                cr = boot_cat(freq[b][j, 0, 0, l], freq[b][j, 0, 1, l], t)
                cv = boot_cat(freq[b][j, 1, 0, l], freq[b][j, 1, 1, l], t)
                fr = f"{f2(freq[b][j, 0, 0, l])}/{f2(freq[b][j, 0, 1, l])}"
                fv = f"{f2(freq[b][j, 1, 0, l])}/{f2(freq[b][j, 1, 1, l])}"
                if cr in ("O", "B"):
                    what = "survives" if cv == cr else ("lost" if cv == "·" else f"changes to {cv}")
                    survival.append(f"{b} L{l} τ {tau}: {cr} → {cv} ({what}; f own/wrong raw {fr}, vw {fv})")
                elif cv in ("O", "B"):
                    survival.append(f"{b} L{l} τ {tau}: {cr} → {cv} (new under vw; f own/wrong raw {fr}, vw {fv})")
    runtime = time.time() - t0

    # Figures.
    for b in FILLERS:
        plot_freq(freq[b], full[b], b, args.n_boot, run / "figures" / f"freq_{b}.png")
        plot_arcs({(k, c): gm_l0[(b, k, c)] for k in ("raw", "vw") for c in CONDS},
                  {(k, c): life[(b, 16, k, c, 0)] for k in ("raw", "vw") for c in CONDS}, t, b, 16,
                  run / "figures" / f"arcs_L0_t16_{b}.png")
    plot_e1(e1, run / "figures" / "e1_ratio.png")

    # Arrays and JSON.
    np.savez_compressed(
        run / "r5.npz", fillers=np.array(FILLERS), taus=np.array(TAUS), kinds=np.array(KINDS), conds=np.array(CONDS),
        all_conds=np.array(ALL_CONDS),
        freq=np.stack([freq[b] for b in FILLERS]), boot_pers=np.stack([boot_pers[b] for b in FILLERS]),
        full_mpc=np.stack([full[b] for b in FILLERS]), boot_idx=np.stack([idxs[b] for b in FILLERS]),
        **{f"e1_{s}_{cnd}": np.stack([np.stack([e1[(b, k, cnd)][s] for k in ("raw", "vw")]) for b in FILLERS])
           for s in ("mean", "lo", "hi") for cnd in ("own", "wrong", "diff")},
        **{f"e1_ratio_{s}": np.stack([e1[(b, "ratio", "own")][s] for b in FILLERS]) for s in ("mean", "lo", "hi")},
        rank_e1=np.stack([np.stack([rank[(b, k)] for k in ("raw", "vw")]) for b in FILLERS]),
        bos_share_q32=np.stack([np.stack([bos[(b, k)] for k in ("raw", "vw")]) for b in FILLERS]),
        heads_l0_e1_share=np.stack([np.stack([np.stack(heads[(b, tau)]) for tau in R5A_TAUS]) for b in FILLERS]),
        vwU_c=np.stack([F[b]["c"] for b in FILLERS]))
    out = {"verdicts": verdicts, "sanity": sanity, "checks": checks, "n_rows": R, "n_boot": args.n_boot, "tda": t,
           "data": data_str, "survival": survival, "runtime_s": runtime, "ties_boot": ties_boot,
           "bootstrap_reused_from": str(args.reuse_boot) if args.reuse_boot else None,
           "loop_life_L0_4": {f"{b}/t{tau}/{k}/{c}/L{l}": v for (b, tau, k, c, l), v in life.items()},
           "hub_L0": {f"{b}/t{tau}/{k}/{c}": v for (b, tau, k, c), v in hub.items()},
           "full_sample_mpc": {f"{b}/t{tau}/{kind}/{cnd}/L{l}": summ[(b, tau, kind, cnd)][l]["mpc"]
                               for b in FILLERS for tau in TAUS for kind in KINDS for cnd in ALL_CONDS for l in range(L)}}
    (run / "r5.json").write_text(json.dumps(jsonable(out), indent=1))
    write_report(run, {"tda": t, "n_boot": args.n_boot, "R": R, "freq": freq, "full": full, "summ": summ, "e1": e1,
                       "sanity": sanity, "verdicts": verdicts, "checks": checks, "tda4": ref, "sym_raw": sym_raw,
                       "survival": survival, "rank": rank, "bos": bos, "heads": heads, "runtime": runtime,
                       "data": out["data"], "life": life, "hub": hub, "reused": reused})
    log(f"done: {run / 'r5_report.md'}", t0)
    for k, v in checks.items():
        print(f"{k}: {v}")
    print(f"sanity: {sanity['l0_summary']} grid {sanity['agree']}/{sanity['n']}")
    for k, v in verdicts.items():
        print(f"R5-a [{k}]: {v['verdict']}  {v['detail']}")
    return run


if __name__ == "__main__":
    main()
