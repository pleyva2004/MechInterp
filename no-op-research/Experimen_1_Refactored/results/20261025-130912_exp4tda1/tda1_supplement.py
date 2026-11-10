"""TDA 1 supplement to the final run results/20261025-114921_exp4tda1 (no redesign; exp4/tda1_heads.py is unchanged).

Run from the project root:
    PYTHONPATH=. uv run --offline python results/20261025-130912_exp4tda1/tda1_supplement.py

Kept inside its results folder rather than exp4/ because exp1.runlog.save_versions hashes every exp1-4/*.py, so a new
file in exp4/ would change the code_sha256 recorded by every later run.

A. Independent cross-check of the main (raw) path: profiles rebuilt from the saved D(d) means, distances, ε* partition
   and ARIs recomputed without the run's helper code.
B. Pre-registered "also reported" items the run gave only partly: a 95% bootstrap interval for the merged-label ARI;
   which H0 bars beat N2 and which H1 features beat N1.
C. Sensitivity (not pre-registered, disclosed): Monte Carlo error of ε*, c1 and N2's ratio q95 (500 draws each), T1-a
   across that ε* range, std T1-d under the self-standardised reading of the clouds, and named groups under corr / std.
Bootstrap and null clouds are regenerated with tda1_heads's own functions and seeds; identity with the saved run is
asserted before use.
"""
from __future__ import annotations

import json

import gudhi
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import adjusted_rand_score

from exp1.runlog import save_versions
from exp3.run_relations import PROJECT_ROOT
from exp4 import tda1_heads as t

RUN = PROJECT_ROOT / "results/20261025-114921_exp4tda1"
OUT = PROJECT_ROOT / "results/20261025-130912_exp4tda1"
MC = 2000  # resamples of the 500 null / bootstrap statistics for Monte Carlo intervals of their 95th percentiles
rng_mc = np.random.default_rng(12345)


def q95_ci(x: np.ndarray) -> list[float]:
    idx = rng_mc.integers(0, len(x), size=(MC, len(x)))
    q = np.percentile(x[idx], 95, axis=1)
    return [float(np.percentile(q, 2.5)), float(np.percentile(q, 97.5))]


def names(ids) -> list[str]:
    return [t.hname(i) for i in ids]


def node_members(Z: np.ndarray, n: int) -> list[list[int]]:
    m = [[i] for i in range(n)]
    for a, b, *_ in Z:
        m.append(m[int(a)] + m[int(b)])
    return m


z = np.load(RUN / "tda1.npz")
S = json.loads((RUN / "tda1_summary.json").read_text())
P = z["profiles"]
n = len(P)
labels = {b: z[f"labels_{b}"] for b in t.FILLERS}
out: dict = {"source_run": RUN.name}
md = ["# TDA 1 supplement", "",
      f"Supplement to the final run `{RUN.name}` (reproduced bit-for-bit as `20261025-125719_exp4tda1`). Script: "
      "`tda1_supplement.py` in this folder. Sections B are pre-registered items the run reported only partly; "
      "sections C are sensitivity checks that were **not pre-registered**.", ""]

# ---------------------------------------------------------------- A. independent cross-check (raw)
f4, f4b = np.load(PROJECT_ROOT / t.RANDOM_RUN), np.load(PROJECT_ROOT / t.CAPITAL_RUN)
src = {"random": (f4, "{b}/"), "own": (f4b, "own/{b}/"), "wrong": (f4b, "wrong/{b}/")}
blocks = []
for b in t.FILLERS:
    for cond, arm in t.CURVES:
        f, pre = src[cond]
        blocks.append(f[pre.format(b=b) + f"{arm}/mean"].reshape(31, n).T)
P_means = np.concatenate(blocks, axis=1)
d_ind = squareform(pdist(P))
eps_raw = S["variants"]["raw"]["eps"]
part_ind = fcluster(linkage(pdist(P), method="single"), eps_raw, criterion="distance")
A = {"max|P(saved means) - P(run)|": float(np.abs(P_means - P).max()),
     "max|d_raw - run d_raw|": float(np.abs(d_ind - z["raw/d"]).max()),
     "ARI(independent ε* partition, run partition)": float(adjusted_rand_score(part_ind, z["raw/part"])),
     **{f"ARI vs labels, {b} (independent)": float(adjusted_rand_score(labels[b], part_ind)) for b in t.FILLERS}}
out["A_crosscheck"] = A
md += ["## A. Independent cross-check of the raw path", "", "| check | value |", "|---|---|"]
md += [f"| {k} | {v:.3g} |" for k, v in A.items()]
md += ["", "Run values for comparison: ARI common +0.123, rare +0.209. The profile difference is the float16 storage "
           "of Exp 4b's per-row arrays (pre-registered tolerance 1e-4).", ""]

# ---------------------------------------------------------------- B1. merged-label ARI with bootstrap interval
md += ["## B1. ARI with labels 5 + 6 merged, with 95% bootstrap interval (pre-registered: an interval for each ARI)",
       "", "| variant | filler | ARI (labels as is) [95%] | ARI (5+6 merged) [95%] |", "|---|---|---|---|"]
out["B1_ari"] = {}
for v in t.VARIANTS:
    bp, part = z[f"{v}/boot_parts"], z[f"{v}/part"]
    for b in t.FILLERS:
        lab, mlab = labels[b], t.merged(labels[b])
        a = adjusted_rand_score(lab, part)
        am = adjusted_rand_score(mlab, part)
        ab = np.array([adjusted_rand_score(lab, p) for p in bp])
        amb = np.array([adjusted_rand_score(mlab, p) for p in bp])
        ci, cim = np.percentile(ab, [2.5, 97.5]), np.percentile(amb, [2.5, 97.5])
        out["B1_ari"][f"{v}/{b}"] = {"ari": a, "ci": ci.tolist(), "ari_merged": am, "ci_merged": cim.tolist()}
        md.append(f"| {v} | {b} | {a:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] | {am:+.3f} [{cim[0]:+.3f}, {cim[1]:+.3f}] |")
md.append("")

# ---------------------------------------------------------------- B2. H0 against N2, H1 against N1
md += ["## B2. H0 bars beyond N2, H1 features beyond N1 (pre-registered 'under the other null')", ""]
out["B2"] = {}
for v in t.VARIANTS:
    Z = z[f"{v}/linkage"]
    mem = node_members(Z, n)
    thr0 = S["variants"][v]["q95"]["N2_max_death"]
    merges = []
    for k, (a, b, h, _) in enumerate(Z):
        if h > thr0:
            small, big = sorted((mem[int(a)], mem[int(b)]), key=len)
            merges.append({"height": float(h), "smaller": names(small) if len(small) <= 8 else f"{len(small)} heads",
                           "larger_size": len(big)})
    dg = z[f"{v}/dgm1"]
    ratio, pers = dg[:, 1] / dg[:, 0], dg[:, 1] - dg[:, 0]
    q1, c1 = S["variants"][v]["q95"]["N1_max_ratio"], S["variants"][v]["c1"]
    h1 = [{"birth": float(dg[i, 0]), "death": float(dg[i, 1]), "ratio": float(ratio[i]), "persistence": float(pers[i]),
           "beyond_band": bool(pers[i] > 2 * c1)} for i in np.flatnonzero(ratio > q1)]
    out["B2"][v] = {"N2_q95_max_H0_death": thr0, "H0_merges_beyond_N2": merges, "N1_q95_ratio": q1, "H1_beyond_N1": h1}
    md += [f"**{v}.** N2 q95 of the largest H0 death = {thr0:.4g}; {len(merges)} bar(s) beyond it:", ""]
    md += [f"- merge at {m['height']:.4g}: {m['smaller'] if isinstance(m['smaller'], str) else ', '.join(m['smaller'])}"
           f" joins a {m['larger_size']}-head group" for m in merges] or ["- none"]
    md += ["", f"H1 features with death/birth > N1 q95 ({q1:.3f}): " +
           ("; ".join(f"birth {h['birth']:.4g}, death {h['death']:.4g}, ratio {h['ratio']:.3f}, persistence "
                      f"{h['persistence']:.4g} ({'beyond' if h['beyond_band'] else 'inside'} the 2·c1 = {2 * c1:.4g} band)"
                      for h in h1) or "none") + ".", ""]

# ---------------------------------------------------------------- regenerate clouds (same functions and seeds)
rows, _, _, _ = t.load_rows(PROJECT_ROOT)
R = {b: rows[b][t.CURVES[0]].shape[0] for b in t.FILLERS}
P_rows = t.profiles(rows, {b: np.ones((1, R[b])) for b in t.FILLERS})[0]
assert np.array_equal(P_rows, P)
rb = np.random.default_rng(t.SEEDS["boot"])
boot_P = t.profiles(rows, {b: rb.multinomial(R[b], np.full(R[b], 1 / R[b]), size=500).astype(float) for b in t.FILLERS})
n2_P = t.gaussian_null(P, 500, np.random.default_rng(t.SEEDS["N2"]))
sd = {v: (P.std(axis=0) if v == "std" else None) for v in t.VARIANTS}
boot_links = {}
for v in ("raw", "std"):
    boot_links[v] = [linkage(squareform(t.distances(Q, v, sd[v]), checks=False), method="single") for Q in boot_P]
    bp = np.stack([fcluster(L, S["variants"][v]["eps"], criterion="distance") for L in boot_links[v]])
    assert np.array_equal(bp, z[f"{v}/boot_parts"]), f"bootstrap draws differ from the saved run ({v})"

# ---------------------------------------------------------------- C1. Monte Carlo error of thresholds; T1-a over ε*
md += ["## C1. Monte Carlo error of the thresholds (not pre-registered)", "",
       "95% intervals of each 95th percentile from resampling its 500 draws (or 500 bootstrap bottleneck distances).", "",
       "| variant | ε* = N1 q95 max H0 death [MC 95%] | 2·c0 | 2·c1 [MC 95%] | N2 q95 max H1 ratio [MC 95%] | top H1 ratio | top-ratio feature persistence |",
       "|---|---|---|---|---|---|---|"]
out["C1"] = {}
for v in t.VARIANTS:
    nd, bn, nr = z[f"{v}/N1/max_death"], z[f"{v}/bottleneck"], z[f"{v}/N2/max_ratio"]
    ci_e, ci_c1, ci_r = q95_ci(nd), q95_ci(bn[:, 1]), q95_ci(nr)
    dg = z[f"{v}/dgm1"]
    top = int(np.argmax(dg[:, 1] / dg[:, 0]))
    vv = S["variants"][v]
    out["C1"][v] = {"N1_q95_ci": ci_e, "c1_ci": ci_c1, "N2_ratio_q95_ci": ci_r}
    md.append(f"| {v} | {vv['q95']['N1_max_death']:.4g} [{ci_e[0]:.4g}, {ci_e[1]:.4g}] | {2 * vv['c0']:.4g} | "
              f"{2 * vv['c1']:.4g} [{2 * ci_c1[0]:.4g}, {2 * ci_c1[1]:.4g}] | {vv['q95']['N2_max_ratio']:.3f} "
              f"[{ci_r[0]:.3f}, {ci_r[1]:.3f}] | {dg[top, 1] / dg[top, 0]:.3f} | {dg[top, 1] - dg[top, 0]:.4g} |")
md += ["", "T1-a (≥ 3 components of ≥ 2 heads with recurrence ≥ 0.75) re-evaluated across the Monte Carlo interval of ε* "
           "(each point re-cuts the main and all 500 resample dendrograms):", "",
       "| variant | ε | components | multi-head components: size (recurrence) | stable | T1-a | ARI common / rare |",
       "|---|---|---|---|---|---|---|"]
out["C1"]["t1a_grid"] = {}
for v in ("raw", "std"):
    Z = z[f"{v}/linkage"]
    lo, hi = out["C1"][v]["N1_q95_ci"]
    grid = sorted(set(np.round(np.linspace(min(lo, S["variants"][v]["eps"]), hi, 7), 6).tolist() + [S["variants"][v]["eps"]]))
    res = []
    for e in grid:
        part = fcluster(Z, e, criterion="distance")
        comps = t.components(part)
        bp = np.stack([fcluster(L, e, criterion="distance") for L in boot_links[v]])
        rec = t.recurrence(comps, bp)
        multi = [(len(g), float(q)) for g, q in zip(comps, rec) if len(g) >= 2]
        stable = sum(q >= t.STABLE for _, q in multi)
        aris = [adjusted_rand_score(labels[b], part) for b in t.FILLERS]
        res.append({"eps": e, "n_components": len(comps), "multi": multi, "stable": stable, "ari": aris})
        md.append(f"| {v} | {e:.4g}{' (ε*)' if e == S['variants'][v]['eps'] else ''} | {len(comps)} | "
                  f"{', '.join(f'{s} ({q:.2f})' for s, q in multi)} | {stable} | {'PASS' if stable >= 3 else 'FAIL'} | "
                  f"{aris[0]:+.3f} / {aris[1]:+.3f} |")
    out["C1"]["t1a_grid"][v] = res
md += ["", "### C1b. Raw ε* under other N1 seeds (not pre-registered)", "",
       "500 sign-flip draws per seed; the run used seed 1. T1-a passes by the letter only for ε below 0.1814, the merge "
       "at which L0H7 joins L2H0 + L2H4.", "",
       "| N1 seed | ε* | multi-head components: members (recurrence) | stable | T1-a |", "|---|---|---|---|---|"]
out["C1"]["raw_seeds"] = []
Zr = z["raw/linkage"]
for s in (1, 11, 21, 31, 41, 51):
    r = np.random.default_rng(s)
    n1 = t.profiles(rows, {b: r.choice([-1.0, 1.0], size=(500, R[b])) for b in t.FILLERS})
    m = np.array([t.persistence(t.distances(Q, "raw"))[0][:, 1].max() for Q in n1])
    e = max(float(np.percentile(m, 95)), 2 * S["variants"]["raw"]["c0"])
    comps = t.components(fcluster(Zr, e, criterion="distance"))
    bp = np.stack([fcluster(L, e, criterion="distance") for L in boot_links["raw"]])
    rec = t.recurrence(comps, bp)
    multi = [(g, float(q)) for g, q in zip(comps, rec) if len(g) >= 2]
    stable = sum(q >= t.STABLE for _, q in multi)
    out["C1"]["raw_seeds"].append({"seed": s, "eps": e, "stable": stable,
                                   "multi": [(names(g) if len(g) <= 8 else len(g), q) for g, q in multi]})
    md.append(f"| {s} | {e:.4g} | " + "; ".join(f"{t.head_list(g) if len(g) <= 8 else f'{len(g)} heads'} ({q:.2f})"
                                                 for g, q in multi) + f" | {stable} | {'PASS' if stable >= 3 else 'FAIL'} |")
md.append("")

# ---------------------------------------------------------------- C2. std T1-d under self-standardised clouds
dg_std = z["std/dgm1"]
self_n2 = np.array([t.h1_stats(t.persistence(t.distances(Q, "std"))[1])[0] for Q in n2_P])
main_n2 = np.array([t.h1_stats(t.persistence(t.distances(Q, "std", sd["std"]))[1])[0] for Q in n2_P])
assert np.allclose(main_n2, z["std/N2/max_ratio"]), "N2 draws differ from the saved run"
self_bn1 = np.array([gudhi.bottleneck_distance(t.persistence(t.distances(Q, "std"))[1], dg_std) for Q in boot_P])
q_self, c1_self = float(np.percentile(self_n2, 95)), float(np.percentile(self_bn1, 95))
ratio, pers = dg_std[:, 1] / dg_std[:, 0], dg_std[:, 1] - dg_std[:, 0]
hit_self = (pers > 2 * c1_self) & (ratio > q_self)
out["C2_std_self"] = {"N2_ratio_q95_self": q_self, "N2_ratio_q95_self_ci": q95_ci(self_n2), "c1_self": c1_self,
                      "t1d_self": "PASS" if not hit_self.any() else "FAIL", "n_hits": int(hit_self.sum())}
md += ["## C2. std T1-d under the other reading of the pre-registration (not pre-registered)", "",
       "The run measured every std cloud with the main profiles' SDs (its disclosed clarification) and reported the "
       "self-standardised alternative only for N1's ε*. Here N2 and the bootstrap are self-standardised too.", "",
       f"- N2 q95 of the largest H1 ratio: {q_self:.3f} (MC 95% {q95_ci(self_n2)[0]:.3f}–{q95_ci(self_n2)[1]:.3f}); "
       f"main-SD reading {S['variants']['std']['q95']['N2_max_ratio']:.3f}.",
       f"- 2·c1: {2 * c1_self:.4g}; main-SD reading {2 * S['variants']['std']['c1']:.4g}.",
       f"- Top std H1 feature: ratio {ratio.max():.3f}, persistence {pers[np.argmax(ratio)]:.4g}.",
       f"- T1-d under std with this reading: **{out['C2_std_self']['t1d_self']}** ({int(hit_self.sum())} feature(s) "
       "beyond both).", ""]

# ---------------------------------------------------------------- C3. named groups under corr and std
md += ["## C3. Named groups under the robustness metrics (exploratory)", ""]
out["C3"] = {}
prev_ids = {t.hidx(*hd) for hd in t.PREV_HEADS}
for v in ("corr", "std"):
    for gname, heads in (("same-country (4)", t.SAME_COUNTRY), ("late any-distance (4)", t.LATE_HEADS)):
        ids = [t.hidx(*hd) for hd in heads]
        h, comp = t.group_scale(z[f"{v}/linkage"], ids)
        rest = np.setdiff1d(np.arange(n), comp)
        gap = float(z[f"{v}/d"][np.ix_(comp, rest)].min())
        entry = {"scale": h, "gap": gap, "size": len(comp), "members": names(comp) if len(comp) <= 15 else None,
                 "prev_heads_in": names(sorted(prev_ids & set(comp.tolist())))}
        out["C3"][f"{v}/{gname}"] = entry
        mem = (": " + ", ".join(entry["members"])) if entry["members"] else ""
        md.append(f"- **{v}, {gname}** connect at {h:.4g} in a {len(comp)}-head component (gap to rest {gap:.4g})"
                  f"{mem}. Exp 1 previous-token heads in it: {', '.join(entry['prev_heads_in']) or 'none'}.")
raw_giant = set(S["variants"]["raw"]["components"][0]["members"])
std_giant = set(S["variants"]["std"]["components"][0]["members"])
out["C3"]["raw_vs_std_giant"] = {"intersection": len(raw_giant & std_giant), "raw_only": sorted(raw_giant - std_giant),
                                 "std_only": sorted(std_giant - raw_giant)}
md += [f"- **Largest component, raw vs std** (110 heads each): {len(raw_giant & std_giant)} shared. raw only: "
       f"{', '.join(sorted(raw_giant - std_giant))}. std only: {', '.join(sorted(std_giant - raw_giant))}.",
       f"- **corr cut.** N1's smallest largest-H0-death over 500 draws is {z['corr/N1/max_death'].min():.4g}, above the "
       f"real data's largest H0 death {z['corr/dgm0'][:, 1].max():.4g}: under correlation distance, pure-noise profiles "
       "are less correlated with each other than any real pair of heads, so ε* is above every bar by construction and "
       "the corr T1-a/b/c verdicts are uninformative rather than independent confirmations.", ""]

(OUT / "tda1_supplement.json").write_text(json.dumps(out, indent=2, default=float))
(OUT / "tda1_supplement.md").write_text("\n".join(md))
save_versions(OUT, PROJECT_ROOT, code_dirs=("exp1", "exp2", "exp3", "exp4"))
print("\n".join(md))
