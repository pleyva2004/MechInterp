"""Stage 1: which position's tokens drive each head's rare-vs-common change, and is it rarity or surface form?

Fresh paired triples (r_i rare, m_i rare matched to the common pool's surface features, c_i common) are recombined
segment by segment (Q = position 32, P = 31, X = 1-30), so every contrast is paired over i:
  - 2x2x2 over (Q, P, X) in {r, c}: Yates main effects and interactions, with ccc - rrr = Q + P + X + QPX exactly;
  - surface form vs rarity: (mmm - rrr) + (ccc - mmm) = ccc - rrr, with m matched on (word start, length, alphabetic);
  - dose-response: 0-30 context positions switched r -> c (nested), with Q and P both rare or both common.
Checks V5-V8 go to validation_report.md; gate G1 decides whether Stage 2 also sweeps position 31.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from exp1.compare import bh_fdr, compare_conditions, label_shift
from exp1.metrics import LABELS, classify
from exp1.plots import plot_diff_heatmaps
from exp1.sequences import derive_seed, prepend_bos
from exp1.validation import md_table, verdict
from exp1extended.config import head_group, head_name, target_heads
from exp1extended.design import (
    check_design, dose_masks, dose_sequences, load_or_create_array, paired_triples, recombine,
)
from exp1extended.extract import run_final
from exp1extended.plots import plot_decomposition, plot_dose
from exp1extended.stats import paired_test, yates_effects_from_cells
from exp1extended.tokens import balance_table, token_features

METRICS = ("sink_logit", "prev_logit", "other_gap", "sink_mass", "prev_mass", "entropy")
EFFECTS = ("Q", "P", "X", "QP", "QX", "PX", "QPX")
DIAGONAL = ("Q", "P", "X", "QPX")  # ccc - rrr = Q + P + X + QPX; two-way terms cancel on the diagonal
MEMBER_POOL = {"r": "rare", "m": "rare", "c": "common"}


def _code(spec: dict[str, str]) -> str:
    return spec["Q"] + spec["P"] + spec["X"]


def _run(model, cfg, seqs: np.ndarray, heads) -> dict[str, np.ndarray]:
    return run_final(model, prepend_bos(seqs, cfg["bos_token_id"]), cfg["batch_size"], heads, cfg["diffuse_threshold"])


def build_design(cfg: dict[str, Any], root: Path, tokenizer, embed: np.ndarray):
    """Triples (cached, hash-verified), the configured cells, and the dose sequences."""
    data_dir = root / cfg["paths"]["sequences_dir"]
    triples = paired_triples(cfg, tokenizer, embed)
    meta = {"seed": cfg["seed"], "n_pairs": cfg["stage1"]["n_pairs"], "pools": cfg["pools"], "matching": cfg["matching"]}
    triples = {k: load_or_create_array(f"stage1_{k}", v, data_dir, dict(meta, member=k)) for k, v in triples.items()}

    cells = {}
    for spec in cfg["stage1"]["cells"]:
        seqs = recombine(triples, spec)
        check_design(seqs, {seg: tuple(cfg["pools"][MEMBER_POOL[m]]) for seg, m in spec.items()})
        cells[_code(spec)] = seqs

    dose = cfg["stage1"]["dose"]
    n = dose["n_pairs"]
    masks = dose_masks(n, dose["levels"], _seed(cfg, "stage1/dose_masks"))
    dose_seqs = {(arm, k): dose_sequences(triples["r"][:n], triples["c"][:n], masks[i], triples[arm][:n])
                 for arm in dose["qp_arms"] for i, k in enumerate(dose["levels"])}
    return triples, cells, masks, dose_seqs


def _seed(cfg: dict[str, Any], name: str) -> int:
    return derive_seed(cfg["seed"], f"exp1extended/{name}")


def design_checks(cfg, triples, cells, masks, dose_seqs, tokenizer) -> tuple[list[dict], pd.DataFrame]:
    """V5: design invariants and surface-feature balance of the matched pool."""
    checks = []
    dose = cfg["stage1"]["dose"]
    n = dose["n_pairs"]
    nested = all(np.all(masks[i] <= masks[i + 1]) for i in range(len(masks) - 1))
    exact = all(np.all(masks[i].sum(axis=1) == k) for i, k in enumerate(dose["levels"]))
    checks.append({"check": "V5 dose masks nested with exact counts", "detail": "", "result": verdict(nested and exact)})
    ends = []
    if "r" in dose["qp_arms"]:
        ends.append(np.array_equal(dose_seqs[("r", dose["levels"][0])], cells["rrr"][:n]))
    if "c" in dose["qp_arms"] and dose["levels"][-1] == 30:
        ends.append(np.array_equal(dose_seqs[("c", 30)], cells["ccc"][:n]))
    checks.append({"check": "V5 dose endpoints equal the rrr / ccc cells", "detail": "", "result": verdict(all(ends))})
    for arm_k, seqs in dose_seqs.items():
        check_design(seqs, {"Q": tuple(cfg["pools"][MEMBER_POOL[arm_k[0]]]), "P": tuple(cfg["pools"][MEMBER_POOL[arm_k[0]]])})
    checks.append({"check": "V5 no repeats; each segment within its pool (cells and dose)", "detail": "",
                   "result": verdict(True)})  # check_design raises otherwise

    features = {k: token_features(tokenizer, np.unique(v), cfg["matching"]["len_bins"]).set_index("id").loc[v.ravel()]
                for k, v in triples.items()}
    cols = ["word_start", "char_len", "len_bin", "is_alpha"]
    balance = pd.concat([
        balance_table(features["m"], features["c"], cols).assign(comparison="m vs c (matched)"),
        balance_table(features["r"], features["c"], cols).assign(comparison="r vs c (unmatched)"),
    ], ignore_index=True)
    matched = balance[(balance["comparison"] == "m vs c (matched)") & balance["feature"].isin(cfg["matching"]["features"])]
    worst = float(matched["smd"].abs().max())
    checks.append({"check": f"V5 matched pool balance |SMD| ≤ {cfg['validation']['max_smd']} on the matching features",
                   "detail": f"max |SMD| {worst:.3f}", "result": verdict(worst <= cfg["validation"]["max_smd"])})
    return checks, balance


def replication_checks(cfg, root: Path, runs: dict[str, dict], heads) -> list[dict]:
    """V6: fresh all-rare / all-common cells vs the saved Exp 1 A_rare / C_common_clean runs (independent samples),
    and V8: the known previous-token heads still look like previous-token heads."""
    s, v = cfg["stats"], cfg["validation"]
    checks = []
    rng = np.random.default_rng(s["bootstrap_seed"])
    for code, cond in (("rrr", "A_rare"), ("ccc", "C_common_clean")):
        saved = np.load(root / cfg["exp1_runs"][cond] / f"{cond}_attn.npy")
        fresh = runs[code]
        p = compare_conditions(fresh["sink_mass"], saved[..., 0])["p"]
        n_sig = int((bh_fdr(p) <= s["fdr_q"]).sum())
        saved_labels = classify(saved, cfg["diffuse_threshold"], saved.shape[-1] - 1)
        tv = label_shift(fresh["label"], saved_labels, len(LABELS))["tv"]
        pooled = np.concatenate([fresh["label"], saved_labels])
        null_max = []
        for _ in range(200):
            perm = rng.permutation(len(pooled))
            half = len(fresh["label"])
            null_max.append(label_shift(pooled[perm[:half]], pooled[perm[half:]], len(LABELS))["tv"].max())
        floor = float(np.percentile(null_max, 95))
        checks.append({"check": f"V6 fresh {code} vs saved {cond}: BH-significant heads on sink_mass",
                       "detail": f"{n_sig}/144", "result": verdict(n_sig == 0)})
        checks.append({"check": f"V6 fresh {code} vs saved {cond}: max label TV below the permutation floor",
                       "detail": f"max TV {tv.max():.3f} vs 95th pct of null max {floor:.3f}",
                       "result": verdict(tv.max() <= floor)})
        # V8 gates on the head keeping its role (mode label Previous). Exp 1's extra prev_mass >= 0.5 cutoff is
        # only reported: L2H2 already failed it in Exp 1's own A_rare run (0.478), so gating on it would re-flag
        # a known Exp 1 result rather than a problem with these sequences.
        for layer, head in v["known_prev_heads"]:
            mode = LABELS[int(np.bincount(fresh["label"][:, layer, head], minlength=len(LABELS)).argmax())]
            prev = float(fresh["prev_mass"][:, layer, head].mean())
            checks.append({"check": f"V8 known previous-token head L{layer}H{head} in {code}: mode label Previous",
                           "detail": f"mode {mode}; mean prev_mass {prev:.3f} (saved {cond}: "
                                     f"{float(saved[:, layer, head, -2].mean()):.3f}; Exp 1 cutoff "
                                     f"{v['prev_head_min_prev_mass']})",
                           "result": verdict(mode == "Previous")})
    return checks


def yates_table(cfg, runs, heads) -> tuple[pd.DataFrame, dict[str, dict[str, dict]]]:
    """Per metric, per effect: paired test over all 144 heads (BH-FDR over the 7 effects x 144 heads of a metric).
    Returns the target-head rows and the full per-metric results (for the layer x head figure)."""
    s = cfg["stats"]
    codes = [a + b + c for a in "rc" for b in "rc" for c in "rc"]
    rows, full = [], {}
    for metric in METRICS:
        effects = yates_effects_from_cells({code: runs[code][metric] for code in codes})
        effects["total (ccc − rrr)"] = runs["ccc"][metric] - runs["rrr"][metric]
        tests = {e: paired_test(effects[e], np.zeros_like(effects[e]), s["bootstrap_iters"], s["bootstrap_seed"], s["ci"])
                 for e in effects}
        q = bh_fdr(np.stack([tests[e]["p"] for e in EFFECTS]))
        for i, e in enumerate(EFFECTS):
            tests[e]["q"] = q[i]
        full[metric] = tests
        for layer, head in heads:
            for e, t in tests.items():
                rows.append({"metric": metric, "head": head_name((layer, head)), "effect": e,
                             "mean": t["mean_diff"][layer, head], "lo": t["lo"][layer, head], "hi": t["hi"][layer, head],
                             "q": t.get("q", np.full((12, 12), np.nan))[layer, head]})
    return pd.DataFrame(rows), full


def matched_table(cfg, runs, heads) -> pd.DataFrame:
    """Surface form vs rarity, all paired over i:
      surface form = mmm - rrr (same id range, surface features moved to the common pool's)
      rarity       = ccc - mmm (same surface features, id range moved to common)
      final token only, at matched form, in common context = ccc - (Q m, P c, X c)
      context only, at matched form, with a common final token = ccc - (Q c, P m, X m)"""
    s = cfg["stats"]
    contrasts = {
        "total (ccc − rrr)": ("ccc", "rrr"),
        "surface form (mmm − rrr)": ("mmm", "rrr"),
        "rarity at matched form (ccc − mmm)": ("ccc", "mmm"),
        "final-token rarity, matched (ccc − Qm·Pc·Xc)": ("ccc", "mcc"),
        "final-token rarity, unmatched (ccc − Qr·Pc·Xc)": ("ccc", "rcc"),
        "context rarity, matched (ccc − Qc·Pm·Xm)": ("ccc", "cmm"),
    }
    rows = []
    for metric in METRICS:
        for name, (a, b) in contrasts.items():
            if a not in runs or b not in runs:
                continue
            t = paired_test(runs[a][metric], runs[b][metric], s["bootstrap_iters"], s["bootstrap_seed"], s["ci"])
            for layer, head in heads:
                rows.append({"metric": metric, "head": head_name((layer, head)), "contrast": name,
                             "mean": t["mean_diff"][layer, head], "lo": t["lo"][layer, head], "hi": t["hi"][layer, head],
                             "p": t["p"][layer, head]})
    return pd.DataFrame(rows)


def dose_table(cfg, dose_runs, heads) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mean metric per (arm, level) with paired-bootstrap CIs, and the departure from linearity at each interior
    level: m(k) - [m(0) + k/30 (m(30) - m(0))], paired over i (0 = linear in the number of common context tokens)."""
    s, dose = cfg["stats"], cfg["stage1"]["dose"]
    levels, top = dose["levels"], dose["levels"][-1]
    curve_rows, lin_rows = [], []
    for metric in METRICS:
        for arm in dose["qp_arms"]:
            vals = {k: dose_runs[(arm, k)][metric] for k in levels}
            for k in levels:
                t = paired_test(vals[k], np.zeros_like(vals[k]), s["bootstrap_iters"], s["bootstrap_seed"], s["ci"])
                for layer, head in heads:
                    curve_rows.append({"metric": metric, "head": head_name((layer, head)), "arm": arm, "level": k,
                                       "mean": t["mean_diff"][layer, head], "lo": t["lo"][layer, head], "hi": t["hi"][layer, head]})
            for k in levels[1:-1]:
                linear = vals[levels[0]] + (k - levels[0]) / (top - levels[0]) * (vals[top] - vals[levels[0]])
                t = paired_test(vals[k], linear, s["bootstrap_iters"], s["bootstrap_seed"], s["ci"])
                for layer, head in heads:
                    lin_rows.append({"metric": metric, "head": head_name((layer, head)), "arm": arm, "level": k,
                                     "departure": t["mean_diff"][layer, head], "lo": t["lo"][layer, head],
                                     "hi": t["hi"][layer, head], "p": t["p"][layer, head]})
    return pd.DataFrame(curve_rows), pd.DataFrame(lin_rows)


def gate_g1(cfg, yates: pd.DataFrame, runs, heads) -> pd.DataFrame:
    """Per target head and primary metric: diagonal shares of ccc - rrr, reported only where the total's CI excludes
    0 and the probability-scale gap is large enough; G1 flags a head whose P share passes the gate."""
    s1, s = cfg["stage1"], cfg["stats"]
    groups = head_group(cfg)
    rows = []
    for head in heads:
        for metric in cfg["primary_metrics"][groups[head]]:
            sel = yates[(yates["head"] == head_name(head)) & (yates["metric"] == metric)].set_index("effect")
            total = sel.loc["total (ccc − rrr)"]
            prob = {"sink_logit": "sink_mass", "prev_logit": "prev_mass"}.get(metric)
            gap_prob = float(runs["ccc"][prob][:, head[0], head[1]].mean() - runs["rrr"][prob][:, head[0], head[1]].mean()) \
                if prob else np.nan
            usable = not (total["lo"] <= 0 <= total["hi"]) and (prob is None or abs(gap_prob) >= s["min_gap_for_share"])
            row = {"head": head_name(head), "metric": metric, "total": total["mean"],
                   "total CI": f"[{total['lo']:+.2f}, {total['hi']:+.2f}]", "Δ prob (ccc − rrr)": gap_prob}
            for e in DIAGONAL:
                row[f"{e} share"] = sel.loc[e, "mean"] / total["mean"] if usable else np.nan
            p_excl = not (sel.loc["P", "lo"] <= 0 <= sel.loc["P", "hi"])
            row["G1: sweep position 31"] = bool(usable and p_excl and row["P share"] >= s1["pos31_gate"])
            rows.append(row)
    return pd.DataFrame(rows)


def _save_runs(run_dir: Path, name: str, res: dict[str, np.ndarray]) -> None:
    np.savez_compressed(run_dir / f"stage1_{name}.npz", **{k: res[k] for k in (*METRICS, "label", "target_scores")})


def run_stage1(cfg: dict[str, Any], run_dir: Path, model, root: Path) -> pd.DataFrame:
    heads = target_heads(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg["figures"]["tokenizer"])
    embed = model.W_E.detach().cpu().numpy()
    triples, cells, masks, dose_seqs = build_design(cfg, root, tokenizer, embed)
    checks, balance = design_checks(cfg, triples, cells, masks, dose_seqs, tokenizer)

    runs = {}
    for code, seqs in cells.items():
        runs[code] = _run(model, cfg, seqs, heads)
        _save_runs(run_dir, f"cell_{code}", runs[code])
        print(f"cell {code}: {len(seqs)} sequences")
    dose_runs = {}
    for (arm, k), seqs in dose_seqs.items():
        dose_runs[(arm, k)] = _run(model, cfg, seqs, heads)
        _save_runs(run_dir, f"dose_{arm}_{k:02d}", dose_runs[(arm, k)])
    print(f"dose: {len(dose_seqs)} arms x levels")

    n_det = cfg["validation"]["determinism_n"]
    rerun = _run(model, cfg, cells["rrr"][:n_det], heads)
    checks.append({"check": f"V7 re-running the first {n_det} rrr sequences is bit-identical", "detail": "",
                   "result": verdict(np.array_equal(rerun["target_scores"], runs["rrr"]["target_scores"][:n_det]))})
    checks += replication_checks(cfg, root, runs, heads)
    checks_df = pd.DataFrame(checks)
    (run_dir / "validation_report.md").write_text("# exp1extended Stage 1: validation\n\n" + md_table(checks_df) + "\n")
    if (checks_df["result"] == "FAIL").any():
        print(checks_df.to_string(index=False))
        raise AssertionError(f"Stage 1 validation failed; see {run_dir / 'validation_report.md'}")

    yates, full = yates_table(cfg, runs, heads)
    matched = matched_table(cfg, runs, heads)
    curves, linearity = dose_table(cfg, dose_runs, heads)
    g1 = gate_g1(cfg, yates, runs, heads)
    tv = label_shift(runs["ccc"]["label"], runs["rrr"]["label"], len(LABELS))["tv"]
    for name, df in {"balance": balance, "yates": yates, "matched": matched, "dose_curves": curves,
                     "dose_linearity": linearity, "g1": g1}.items():
        df.to_parquet(run_dir / f"stage1_{name}.parquet", index=False)

    fig_dir = run_dir / "figures"
    diffs = {name: full["sink_logit"][e]["mean_diff"] for name, e in
             (("Q: final token (pos 32)", "Q"), ("P: previous token (pos 31)", "P"), ("X: context (pos 1-30)", "X"))}
    sig = {name: full["sink_logit"][e]["q"] <= cfg["stats"]["fdr_q"] for name, e in
           (("Q: final token (pos 32)", "Q"), ("P: previous token (pos 31)", "P"), ("X: context (pos 1-30)", "X"))}
    plot_diff_heatmaps(diffs, sig, fig_dir / "s1_main_effects.png",
                       "Main effect of switching each position from rare to common tokens on sink_logit [TL]",
                       "Δ sink_logit (common − rare)",
                       sig_note=f"paired over {cfg['stage1']['n_pairs']} recombined triples; outlined = BH-FDR "
                                f"significant (q ≤ {cfg['stats']['fdr_q']}) across 7 effects × 144 heads")
    groups = head_group(cfg)
    dec_rows = []
    for head in heads:
        metric = cfg["primary_metrics"][groups[head]][0]
        sel = yates[(yates["head"] == head_name(head)) & (yates["metric"] == metric)
                    & yates["effect"].isin([*DIAGONAL, "total (ccc − rrr)"])]
        sel = sel.assign(part=sel["effect"].replace({"total (ccc − rrr)": "total"}), comparison="rare → common (ccc − rrr)",
                         group=groups[head], head=f"{head_name(head)} {metric}")
        m = matched[(matched["head"] == head_name(head)) & (matched["metric"] == metric)
                    & matched["contrast"].isin(["total (ccc − rrr)", "surface form (mmm − rrr)",
                                                "rarity at matched form (ccc − mmm)"])]
        m = m.assign(part=m["contrast"].replace({"total (ccc − rrr)": "total", "surface form (mmm − rrr)": "surface form",
                                                 "rarity at matched form (ccc − mmm)": "rarity"}),
                     comparison="surface form vs rarity", group=groups[head], head=f"{head_name(head)} {metric}")
        dec_rows += [sel, m]
    dec = pd.concat(dec_rows, ignore_index=True)
    plot_decomposition(dec[dec["comparison"] == "rare → common (ccc − rrr)"], fig_dir / "s1_positions.png",
                       "Which position drives each head: ccc − rrr = Q + P + X + QPX (paired, TL)",
                       "change in primary metric (logit units)", parts=DIAGONAL,
                       note="dots = Yates effects with 95% paired-bootstrap CIs; outline = total. "
                            "Two-way interactions cancel on the ccc − rrr diagonal.")
    plot_decomposition(dec[dec["comparison"] == "surface form vs rarity"], fig_dir / "s1_surface_vs_rarity.png",
                       "Surface form vs rarity: (mmm − rrr) + (ccc − mmm) = ccc − rrr (paired, TL)",
                       "change in primary metric (logit units)", parts=("surface form", "rarity"),
                       note="m = rare-range tokens matched to the common pool on word start, length bin and alphabetic.")
    plot_dose(curves, heads, groups, cfg["primary_metrics"], fig_dir / "s1_dose.png",
              "Dose-response: context positions switched from rare to common (paired, TL)")

    _write_report(run_dir / "stage1_report.md", cfg, run_dir.name, balance, yates, matched, linearity, g1, tv, heads)
    return g1


def _fmt(mean: float, lo: float, hi: float) -> str:
    return f"{mean:+.2f} [{lo:+.2f}, {hi:+.2f}]"


def _write_report(path, cfg, run_id, balance, yates, matched, linearity, g1, tv, heads) -> None:
    groups = head_group(cfg)
    primary = {head_name(h): cfg["primary_metrics"][groups[h]] for h in heads}
    keep = lambda df: df[[m in primary[h] for h, m in zip(df["head"], df["metric"])]]
    s1 = cfg["stage1"]
    out = [
        "# exp1extended Stage 1: positions, surface form and rarity", "",
        f"- Run: `{run_id}`. {s1['n_pairs']} fresh paired triples (r rare 1000-39999, m rare matched to the common "
        "pool on word start / length bin / alphabetic, c common 256-999), recombined by segment: Q = position 32, "
        "P = 31, X = 1-30. Every contrast is paired over the triple index; CIs are 95% paired bootstrap, p-values "
        "Wilcoxon signed-rank. Checks V5-V8: `validation_report.md`.",
        "- Metrics in logit units from the pre-softmax scores (see the Stage 0 report). Effects are common − rare.", "",
        "## 1. Which position drives each head (gate G1)", "",
        "ccc − rrr = Q + P + X + QPX exactly (two-way interactions cancel on this diagonal). Shares are shown only "
        f"where the total's CI excludes 0 and the probability-scale gap is ≥ {cfg['stats']['min_gap_for_share']}.", "",
        "![positions](figures/s1_positions.png)", "", md_table(g1, ".2f"), "",
    ]
    y = keep(yates)
    y = y.assign(value=[_fmt(m, l, h) + ("*" if q <= cfg["stats"]["fdr_q"] else "") for m, l, h, q in
                        zip(y["mean"], y["lo"], y["hi"], y["q"].fillna(1.0))])
    wide = y.pivot_table(index=["head", "metric"], columns="effect", values="value", aggfunc="first").reset_index()
    out += ["All Yates effects for the target heads' primary metrics (* = BH-FDR significant):", "",
            md_table(wide[["head", "metric", *EFFECTS, "total (ccc − rrr)"]]), "",
            "Layer × head view for sink_logit, all heads: ![main effects](figures/s1_main_effects.png)", ""]
    m = keep(matched)
    m = m.assign(value=[_fmt(a, l, h) for a, l, h in zip(m["mean"], m["lo"], m["hi"])])
    wide = m.pivot_table(index=["head", "metric"], columns="contrast", values="value", aggfunc="first").reset_index()
    out += ["## 2. Surface form vs rarity", "", "![surface vs rarity](figures/s1_surface_vs_rarity.png)", "",
            md_table(wide), "", "Balance of the matched pool (standardized mean differences vs the common pool):", "",
            md_table(balance, ".3f"), ""]
    lin = keep(linearity)
    lin = lin.assign(value=[_fmt(d, l, h) for d, l, h in zip(lin["departure"], lin["lo"], lin["hi"])])
    wide = lin.pivot_table(index=["head", "metric", "arm"], columns="level", values="value", aggfunc="first").reset_index()
    out += ["## 3. Context dose-response", "", "![dose](figures/s1_dose.png)", "",
            "Departure from a straight line between 0 and 30 common context tokens, per level (0 = linear):", "",
            md_table(wide), ""]
    out += ["## 4. Label mix, fresh all-rare vs all-common (continuity with Exp 1)", "",
            md_table(pd.DataFrame({"head": [head_name(h) for h in heads], "label TV (ccc vs rrr)": [tv[h] for h in heads]}), ".2f"), ""]
    fires = g1[g1["G1: sweep position 31"]]["head"].tolist()
    out += ["## 5. Gate G1", "",
            (f"Fires for {', '.join(fires)}: add 31 to `stage2.positions` before Stage 2." if fires else
             "Does not fire: no head's previous-token share reaches the gate, so Stage 2 sweeps position 32 only."), ""]
    path.write_text("\n".join(out))
