"""Experiment 5, R3: does the stream at T carry less about how far back the country is, with depth?

Run:  uv run --offline python -m exp5.r3_distance --data results/<ts>_exp5res [--out DIR] [--boot 2000]

m_l(tau) = mean over rows of [x_l(T | country at tau) - x_l(T | control)], d = 32 - tau = 1..31, l = 1..12.
D_l = ||mean_{d 1-4} m_l - mean_{d 16-31} m_l|| / mean_d ||m_l(d)||. R3-a: Spearman(l, D_l) < 0; R3-b: mean D(l 9-12) -
mean D(l 1-4) < 0; both with a row-bootstrap 95% CI below 0 (exp5/PREDICTIONS.md, exp5/PREDICTIONS_R3.md).
Without --out, writes results/<timestamp>_exp5R3/: r3_report.md, r3_stats.npz, figures, versions.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ripser import ripser
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata
from sklearn.decomposition import PCA

from exp1.runlog import make_run_dir, save_versions
from exp3.run_relations import PROJECT_ROOT


def log(msg: str) -> None:
    print(msg, flush=True)


FILLERS = ("common", "rare")
CONDS = ("own", "wrong", "random")
LAYERS = np.arange(1, 13)
NEAR, FAR = slice(0, 4), slice(15, 31)  # distance index d - 1: d 1-4, d 16-31
EARLY, LATE = slice(0, 4), slice(8, 12)  # positions in LAYERS: l 1-4, l 9-12
COLORS = {"own": "#2a78d6", "wrong": "#eb6834", "random": "#898781"}


def rms(x: np.ndarray) -> np.ndarray:
    return np.sqrt((x ** 2).mean(-1, keepdims=True))


def shifts(d: dict, cond: int, per_state: bool) -> tuple[np.ndarray, np.ndarray]:
    """Per-row shifts [R, 31, 12, 768] (float32) and the per-layer control scale s_l [12] (norm of the control state)."""
    res = d["resid"][:, cond, :, 1:, :].astype(np.float32)  # [R, 31, 12, 768]
    ctrl = (d["ctrl_resid"] if cond < 2 else d["rnd_ctrl_resid"])[:, 1:, :].astype(np.float32)  # [R, 12, 768]
    scale = np.sqrt((ctrl ** 2).sum(-1).mean(0))
    if per_state:
        return res / rms(res) - (ctrl / rms(ctrl))[:, None], scale
    return res - ctrl[:, None], scale


def dist_stat(m: np.ndarray) -> np.ndarray:
    """m [..., 31, 768] -> D [...]."""
    num = np.linalg.norm(m[..., NEAR, :].mean(-2) - m[..., FAR, :].mean(-2), axis=-1)
    return num / np.linalg.norm(m, axis=-1).mean(-1)


def summaries(D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """D [..., 12] -> Spearman(l, D_l) and mean D(9-12) - mean D(1-4)."""
    r = rankdata(D, axis=-1)
    lr = rankdata(LAYERS)
    rc, lc = r - r.mean(-1, keepdims=True), lr - lr.mean()
    rho = (rc * lc).sum(-1) / np.sqrt((rc ** 2).sum(-1) * (lc ** 2).sum())
    return rho, D[..., LATE].mean(-1) - D[..., EARLY].mean(-1)


def bootstrap(delta: np.ndarray, boot: int, seed: int) -> np.ndarray:
    """delta [R, 31, 12, 768] -> D [boot, 12] over row resamples (counts @ delta, in layer chunks)."""
    R = len(delta)
    counts = np.random.default_rng(seed).multinomial(R, np.full(R, 1 / R), size=boot).astype(np.float32) / R
    out = np.empty((boot, len(LAYERS)))
    for li in range(len(LAYERS)):
        m = (counts @ delta[:, :, li, :].reshape(R, -1)).reshape(boot, 31, -1)
        out[:, li] = dist_stat(m)
    return out


def h0(m: np.ndarray) -> dict:
    """Rips H0 of the 31 points m [31, 768]: finite death times and the 2-cluster cut of the single-linkage tree."""
    deaths = np.sort(ripser(squareform(pdist(m)), maxdim=0, distance_matrix=True)["dgms"][0][:, 1])
    deaths = deaths[np.isfinite(deaths)][::-1]
    lab = fcluster(linkage(m, "single"), 2, "maxclust")
    d = np.arange(1, 32)
    a, b = d[lab == lab[0]], d[lab != lab[0]]  # cluster of d = 1, and the rest
    return {"death1": deaths[0], "death2": deaths[1], "ratio": deaths[0] / deaths[1], "n_with_d1": len(a),
            "n_other": len(b), "near_far_split": bool((lab[NEAR] == lab[0]).all() and (lab[FAR] != lab[0]).all()),
            "isolated_d": int(a[0]) if len(a) == 1 else (int(b[0]) if len(b) == 1 else 0)}


def descriptives(delta: np.ndarray, scale: np.ndarray) -> dict:
    m = delta.mean(0)  # [31, 12, 768]
    norm = np.linalg.norm(m, axis=-1) / scale  # [31, 12]
    path, e2e, pr, hh = [], [], [], []
    for li in range(len(LAYERS)):
        x = m[:, li, :]
        path.append(np.linalg.norm(np.diff(x, axis=0), axis=-1).sum())
        e2e.append(np.linalg.norm(x[-1] - x[0]))
        lam = PCA().fit(x).explained_variance_
        pr.append(lam.sum() ** 2 / (lam ** 2).sum())
        hh.append(h0(x))
    return {"norm": norm, "path": np.array(path), "e2e": np.array(e2e), "ratio": np.array(path) / np.array(e2e),
            "pr": np.array(pr), "h0": hh}


def ci(x: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    data = Path(args.data) if Path(args.data).is_absolute() else PROJECT_ROOT / args.data
    if args.out:
        run = Path(args.out)
        run.mkdir(parents=True, exist_ok=True)
    else:
        if not (data / "DONE").exists():
            raise SystemExit(f"{data} has no DONE file")
        run = make_run_dir(PROJECT_ROOT / "results", tag="exp5R3")
        shutil.copy2(PROJECT_ROOT / "exp5" / "PREDICTIONS_R3.md", run / "PREDICTIONS_R3.md")
        save_versions(run, PROJECT_ROOT, code_dirs=("exp1", "exp2", "exp3", "exp4", "exp5"))
    log(f"R3 -> {run}")

    res: dict = {}  # (filler, cond, norm) -> dict
    arrays: dict[str, np.ndarray] = {}
    for b in FILLERS:
        d = np.load(data / f"means_{b}.npz")
        d = {k: d[k] for k in d}
        R = len(d["resid"])
        for ci_, cond in enumerate(CONDS):
            for nm, per_state in (("ctrlscale", False), ("perstate", True)):
                delta, scale = shifts(d, ci_, per_state)
                Dpoint = dist_stat(delta.mean(0).transpose(1, 0, 2))  # [12]
                Dboot = bootstrap(delta, args.boot, seed=0)
                rho_b, diff_b = summaries(Dboot)
                rho, diff = (x[()] for x in summaries(Dpoint))
                r = {"D": Dpoint, "Dboot": Dboot, "rho": float(rho), "rho_ci": ci(rho_b), "diff": float(diff),
                     "diff_ci": ci(diff_b), "D_ci": np.percentile(Dboot, [2.5, 97.5], axis=0), "R": R}
                if nm == "ctrlscale":
                    r["desc"] = descriptives(delta, scale)
                    r["m_pca"] = delta.mean(0)  # [31, 12, 768] kept for the PCA figure
                res[b, cond, nm] = r
                log(f"{b} {cond} {nm}: rho={rho:+.3f} {r['rho_ci'][0]:+.3f}..{r['rho_ci'][1]:+.3f}  "
                    f"diff={diff:+.3f} {r['diff_ci'][0]:+.3f}..{r['diff_ci'][1]:+.3f}")
                tag = f"{b}_{cond}_{nm}"
                arrays[f"D_{tag}"], arrays[f"Dci_{tag}"] = Dpoint, r["D_ci"]
                arrays[f"rho_{tag}"] = np.array([rho, *r["rho_ci"]])
                arrays[f"diff_{tag}"] = np.array([diff, *r["diff_ci"]])
                if nm == "ctrlscale":
                    for k in ("norm", "path", "e2e", "ratio", "pr"):
                        arrays[f"{k}_{b}_{cond}"] = r["desc"][k]
                    arrays[f"h0ratio_{b}_{cond}"] = np.array([h["ratio"] for h in r["desc"]["h0"]])
    np.savez(run / "r3_stats.npz", **arrays)

    figures(run, res)
    report(run, res, args.boot)
    (run / "DONE").write_text("")


def figures(run: Path, res: dict) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey="row")
    for i, nm in enumerate(("ctrlscale", "perstate")):
        for j, b in enumerate(FILLERS + ("both",)):
            ax = axes[i, j]
            for cond in () if b == "both" else CONDS:
                r = res[b, cond, nm]
                ax.plot(LAYERS, r["D"], "-o", color=COLORS[cond], label=cond, ms=4)
                ax.fill_between(LAYERS, *r["D_ci"], color=COLORS[cond], alpha=0.2)
            if b == "both":
                for cond in CONDS:
                    for bb, ls in zip(FILLERS, ("-", "--")):
                        ax.plot(LAYERS, res[bb, cond, nm]["D"], ls, color=COLORS[cond], label=f"{cond} {bb}")
                ax.legend(fontsize=7, ncol=2)
            ax.set_title(f"{b} ({'one scale per layer = raw' if i == 0 else 'per-state RMS'})", fontsize=9)
            ax.set_xlabel("layer l (state before layer l; 12 = after layer 11)")
            ax.set_ylabel("D_l")
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
    fig.suptitle("Distance dependence D_l of the stream shift at T, with 95% row-bootstrap CI")
    fig.tight_layout()
    fig.savefig(run / "fig_D_by_layer.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for i, b in enumerate(FILLERS):
        for j, cond in enumerate(CONDS):
            norm = res[b, cond, "ctrlscale"]["desc"]["norm"].T  # [12, 31]
            im = axes[i, j].imshow(norm, aspect="auto", origin="lower", extent=[0.5, 31.5, 0.5, 12.5], cmap="viridis")
            axes[i, j].set_title(f"{b}, {cond}", fontsize=9)
            axes[i, j].set_xlabel("distance d = 32 - tau")
            axes[i, j].set_ylabel("layer l")
            fig.colorbar(im, ax=axes[i, j], label="||m_l|| / ||control||")
    fig.tight_layout()
    fig.savefig(run / "fig_norm_heatmap.png", dpi=140)
    plt.close(fig)

    show = (1, 4, 8, 12)
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for i, b in enumerate(FILLERS):
        m = res[b, "own", "ctrlscale"]["m_pca"]
        for j, l in enumerate(show):
            x = m[:, l - 1, :]
            z = PCA(2).fit(x)
            p = z.transform(x)
            ax = axes[i, j]
            ax.plot(p[:, 0], p[:, 1], "-", color="#898781", lw=0.6)
            sc = ax.scatter(p[:, 0], p[:, 1], c=np.arange(1, 32), cmap="viridis", s=25)
            ax.set_title(f"{b}, l={l}, PC1+2 = {z.explained_variance_ratio_.sum():.0%}", fontsize=9)
            if j == 3:
                fig.colorbar(sc, ax=ax, label="distance d")
    fig.suptitle("Own-capital shift m_l(d), 31 points in 2-D PCA (per layer)")
    fig.tight_layout()
    fig.savefig(run / "fig_trajectory_pca.png", dpi=140)
    plt.close(fig)


def verdict(r: dict, key: str) -> str:
    lo, hi = r[f"{key}_ci"]
    return "PASS" if r[key] < 0 and hi < 0 else "FAIL"


def report(run: Path, res: dict, boot: int) -> None:
    out = ["# R3: distance dependence of the stream at T (GPT-2 small)", "",
           f"Rows per filler: {res[FILLERS[0], 'own', 'ctrlscale']['R']}; bootstrap resamples: {boot} (rows), 95% percentile CI. "
           "Hypotheses and pass rules: `exp5/PREDICTIONS.md`; choices: `exp5/PREDICTIONS_R3.md`.", "",
           "## Verdicts (own capital, primary normalisation)", "",
           "| id | filler | statistic | CI | verdict |", "|---|---|---|---|---|"]
    final = {}
    for hid, key, desc in (("R3-a", "rho", "Spearman(l, D_l)"), ("R3-b", "diff", "mean D(9-12) - mean D(1-4)")):
        vs = []
        for b in FILLERS:
            r = res[b, "own", "ctrlscale"]
            v = verdict(r, key)
            vs.append(v)
            out.append(f"| {hid} | {b} | {desc} = {r[key]:+.3f} | [{r[key + '_ci'][0]:+.3f}, {r[key + '_ci'][1]:+.3f}] | {v} |")
        final[hid] = "PASS" if all(v == "PASS" for v in vs) else "FAIL"
    out += ["", f"**R3-a overall: {final['R3-a']}. R3-b overall: {final['R3-b']}** (must pass in both fillers).", "",
            "## All conditions and both normalisations (rho = Spearman(l, D_l); diff = late - early)", "",
            "Wrong-capital and random-final-token rows are descriptive. 'raw' = one scale per layer (D is scale-free, so "
            "equals raw centred states); 'per-state' = each state divided by its own RMS first.", "",
            "| filler | condition | norm | rho [CI] | diff [CI] | verdict a/b |", "|---|---|---|---|---|---|"]
    for b in FILLERS:
        for cond in CONDS:
            for nm, lab in (("ctrlscale", "raw"), ("perstate", "per-state")):
                r = res[b, cond, nm]
                out.append(f"| {b} | {cond} | {lab} | {r['rho']:+.3f} [{r['rho_ci'][0]:+.3f}, {r['rho_ci'][1]:+.3f}] | "
                           f"{r['diff']:+.3f} [{r['diff_ci'][0]:+.3f}, {r['diff_ci'][1]:+.3f}] | "
                           f"{verdict(r, 'rho')}/{verdict(r, 'diff')} |")
    out += ["", "## D_l by layer (point estimate, own capital, raw; CI in r3_stats.npz)", "",
            "| filler | " + " | ".join(f"l={l}" for l in LAYERS) + " |", "|---|" + "---|" * 12]
    for b in FILLERS:
        out.append(f"| {b} | " + " | ".join(f"{x:.2f}" for x in res[b, 'own', 'ctrlscale']['D']) + " |")
    out += ["", "## Trajectory descriptives (own capital; point estimates of the row-mean m_l(d))", ""]
    for b in FILLERS:
        for cond in CONDS:
            dsc = res[b, cond, "ctrlscale"]["desc"]
            out += [f"### {b}, {cond}", "",
                    "| l | ‖m‖/‖ctrl‖ d=1 | d=4 | d=16 | d=31 | path/end-to-end | PCA PR | H0 d1/d2 | H0 clusters (with d=1 / other) | near vs far split |",
                    "|---|---|---|---|---|---|---|---|---|---|"]
            for li, l in enumerate(LAYERS):
                n, h = dsc["norm"][:, li], dsc["h0"][li]
                out.append(f"| {l} | {n[0]:.3f} | {n[3]:.3f} | {n[15]:.3f} | {n[30]:.3f} | {dsc['ratio'][li]:.1f} | "
                           f"{dsc['pr'][li]:.2f} | {h['ratio']:.2f} | {h['n_with_d1']} / {h['n_other']} | "
                           f"{'yes' if h['near_far_split'] else 'no'} |")
            out.append("")
    out += ["## Caveats", "",
            "- The same 11 country-capital pairs in every row: a result means consistent across rows (row bootstrap), not general across pairs.",
            "- m is a mean over 100 rows x 11 cells; the 31 trajectory points carry that sampling noise, so H0 / path length of a noisy "
            "trajectory partly reflect noise (path/end-to-end is inflated by it).",
            "- D pools d 1-4 (4 points) against d 16-31 (16 points) and divides by the mean norm; it can change with depth "
            "through the norm denominator as well as the numerator.",
            "- The wrong-capital and random-final-token conditions are descriptive; no multiplicity correction across them.",
            "- Layer 12 is the state after layer 11 (pre-ln_final)."]
    (run / "r3_report.md").write_text("\n".join(out) + "\n")
    log(json.dumps(final))


if __name__ == "__main__":
    main()
