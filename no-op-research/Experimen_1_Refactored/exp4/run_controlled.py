"""Experiment 4: controlled difference. Country (or a matched placebo word) at position tau vs the row's own token.

Run:  uv run --offline python -m exp4.run_controlled --config configs/exp4_controlled.yaml

Per random row and tau = 1..31, the variants differ from the row only at tau; the final token (32) is never edited.
For every head, the attention from the final token to tau is compared with the control row's attention to tau:
D(d) = E[a(variant) - a(control)] over rows, d = 32 - tau. The row is the unit (each row's 11 countries averaged).
See exp4/PREDICTIONS.md. Writes results/<timestamp>_exp4ctrl/: config, predictions, versions, arrays, checks,
verdicts, figures, viz_data.json (for exp4.build_viz) and exp4_controlled_report.md.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # write figures to files, never open a window
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.colors import TwoSlopeNorm

from exp1.backends import load_model, set_determinism
from exp1.compare import bh_fdr
from exp1.runlog import make_run_dir, save_config_copy, save_versions
from exp1ext.effects import bootstrap_mean_per_unit
from exp1ext.features import token_features
from exp2.analysis import paired_test
from exp2.design import cached_base_block, matched_pools
from exp3.run_relations import PROJECT_ROOT, attention, log, stimuli

ARMS = ("country", "word")
CONTRASTS = {"country": "country − control", "word": "word − control", "specific": "country − word"}
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#898781"
LABELS = ("global country-tracking", "local country-tracking", "content-gated previous-token", "negative",
          "plain previous-token", "no country effect")


def head_name(layer: int, head: int) -> str:
    return f"L{layer}H{head}"


def placebo_words(bases: np.ndarray, pools: list[np.ndarray], seed: int) -> np.ndarray:
    """[R, n]: per row and country, a uniform draw from that country's matched pool, never a token of the row
    (so the word can sit at any tau without repeating a token). Row r uses default_rng([seed, r])."""
    out = np.empty((len(bases), len(pools)), dtype=np.int64)
    for r, row in enumerate(bases):
        rng = np.random.default_rng([seed, r])
        for c, pool in enumerate(pools):
            out[r, c] = rng.choice(pool[~np.isin(pool, row)])
    return out


def with_final(bases: np.ndarray, n: int, final: np.ndarray | None) -> np.ndarray:
    """[R, n, T]: each row n times, with final[r, c] as the final token when given (the control of exp4.run_capital)."""
    seqs = np.repeat(bases[:, None, :], n, axis=1)
    if final is not None:
        seqs[:, :, -1] = final
    return seqs


def variants(bases: np.ndarray, fill: np.ndarray, tau: int, final: np.ndarray | None = None) -> np.ndarray:
    """bases [R, T], fill [R, n] -> [R, n, T]: row r with fill[r, c] at model position tau (array index tau - 1),
    and final[r, c] as the final token when given."""
    seqs = with_final(bases, fill.shape[1], final)
    seqs[:, :, tau - 1] = fill
    return seqs


def check_variants(seqs: np.ndarray, bases: np.ndarray, fill: np.ndarray, tau: int, bos_id: int,
                   final: np.ndarray | None = None) -> None:
    if not (seqs[:, :, tau - 1] == fill).all():
        raise AssertionError(f"tau {tau}: wrong ids at tau")
    keep = np.ones(seqs.shape[-1], dtype=bool)
    keep[tau - 1] = False
    if not (seqs[:, :, keep] == with_final(bases, fill.shape[1], final)[:, :, keep]).all():
        raise AssertionError(f"tau {tau}: a variant differs from its control row outside tau")
    s = np.sort(seqs.reshape(-1, seqs.shape[-1]), axis=1)
    if (s[:, 1:] == s[:, :-1]).any() or (seqs == bos_id).any():
        raise AssertionError(f"tau {tau}: a sequence repeats a token or contains BOS")


def extract(model, bases: np.ndarray, fills: dict[str, np.ndarray], cfg: dict[str, Any], t0: float, base: str,
            final: np.ndarray | None = None) -> dict:
    """Control attention [R, L, H, K]. Per arm, indexed by distance d - 1 (tau = T - d): per-row attention on tau, BOS
    and the previous token (31), each averaged over the row's n fills [R, T - 1, L, H, 3], and the mean attention row
    over rows and fills [T - 1, L, H, K]. final [R, n]: a final token per cell, in the control and every variant; the
    control is then the mean over the row's n controls (a row's D is a mean of per-cell differences, so this is exact)."""
    T = cfg["seq_len"]
    if final is None:
        control = attention(model, bases, cfg)
    else:
        ctrl = attention(model, with_final(bases, final.shape[1], final).reshape(-1, T), cfg)
        control = ctrl.reshape(final.shape + ctrl.shape[1:]).mean(axis=1)
    out: dict[str, Any] = {"control": control}
    for arm, fill in fills.items():
        R, n = fill.shape
        masses = np.empty((R, T - 1) + control.shape[1:3] + (3,), dtype=np.float32)
        by_key = np.empty((T - 1,) + control.shape[1:], dtype=np.float32)
        for tau in range(1, T):
            seqs = variants(bases, fill, tau, final)
            check_variants(seqs, bases, fill, tau, cfg["bos_token_id"], final)
            a = attention(model, seqs.reshape(-1, T), cfg).reshape((R, n) + control.shape[1:])  # [R, n, L, H, K]
            k = T - tau - 1
            masses[:, k] = np.stack([a[..., tau], a[..., 0], a[..., T - 1]], axis=-1).mean(axis=1)
            by_key[k] = a.mean(axis=(0, 1))
        out[arm] = {"masses": masses, "by_key": by_key}
        log(f"{base}: {arm} arm done", t0)
    return out


def row_differences(ext: dict, T: int) -> dict[str, np.ndarray]:
    """Per-row differences on the attention to tau, [R, T - 1 (distance), L, H], for the three contrasts."""
    taus = T - np.arange(1, T)  # tau at distance index k
    control = np.moveaxis(ext["control"][..., taus], -1, 1)  # [R, T - 1, L, H]
    x = {arm: ext[arm]["masses"][..., 0].astype(np.float64) - control for arm in ARMS}
    x["specific"] = x["country"] - x["word"]
    return x


def contrast_stats(x: np.ndarray, a: dict[str, Any]) -> dict[str, np.ndarray]:
    """x [R, D, L, H] -> mean, 95% CI, Wilcoxon p, BH q over all D x L x H cells, and the meaningful flag."""
    boot = a["bootstrap"]
    mean, lo, hi = bootstrap_mean_per_unit(x, boot["iters"], boot["seed"], 0.95)
    p = paired_test(x)["p"]
    q = bh_fdr(p)
    return {"mean": mean, "lo": lo, "hi": hi, "p": p, "q": q, "sig": (q <= a["q"]) & (np.abs(mean) >= a["margin"])}


def label_heads(st: dict[str, np.ndarray], prev: np.ndarray, a: dict[str, Any]) -> np.ndarray:
    """[L, H] label per head from the country contrast (PREDICTIONS.md, first matching rule wins)."""
    pos = st["sig"] & (st["mean"] > 0)  # [D, L, H]
    neg = st["sig"] & (st["mean"] < 0)
    far_pos = pos[1:].sum(axis=0)
    out = np.empty(prev.shape, dtype=object)
    for idx in np.ndindex(prev.shape):
        if far_pos[idx] >= a["global_min"]:
            out[idx] = LABELS[0]
        elif far_pos[idx] > 0:
            out[idx] = LABELS[1]
        elif pos[(0,) + idx]:
            out[idx] = LABELS[2]
        elif neg[(slice(None),) + idx].any():
            out[idx] = LABELS[3]
        elif prev[idx] >= a["prev_min"]:
            out[idx] = LABELS[4]
        else:
            out[idx] = LABELS[5]
    return out


def span_mean(x: np.ndarray, span: list[int]) -> np.ndarray:
    """Mean over distances span[0]..span[1] (inclusive) of x [R, D, ...] -> [R, ...]."""
    return x[:, span[0] - 1:span[1]].mean(axis=1)


def analyse(ext: dict, cfg: dict[str, Any]) -> dict[str, Any]:
    a, T = cfg["analysis"], cfg["seq_len"]
    boot = a["bootstrap"]
    x = row_differences(ext, T)
    res: dict[str, Any] = {"x": x, "stats": {c: contrast_stats(x[c], a) for c in CONTRASTS}}
    res["prev_score"] = ext["control"][..., T - 1].mean(axis=0)  # [L, H]
    res["labels"] = label_heads(res["stats"]["country"], res["prev_score"], a)
    layer_rows = x["country"].mean(axis=3)  # [R, D, L]
    res["layer"] = dict(zip(("mean", "lo", "hi"), bootstrap_mean_per_unit(layer_rows, boot["iters"], boot["seed"], 0.95)))
    li = span_mean(x["country"], a["near"]) - span_mean(x["country"], a["far"])  # [R, L, H]
    res["li_layer"] = dict(zip(("mean", "lo", "hi"), bootstrap_mean_per_unit(li.mean(axis=2), boot["iters"], boot["seed"], 0.95)))
    groups = {g: li[:, lo:hi + 1].mean(axis=(1, 2)) for g, (lo, hi) in a["layer_groups"].items()}
    g1, g2 = list(groups)
    res["li_group"] = {g: [float(v) for v in bootstrap_mean_per_unit(r, boot["iters"], boot["seed"], 0.95)]
                       for g, r in {**groups, f"{g1} − {g2}": groups[g1] - groups[g2]}.items()}
    return res


def check_predictions(res: dict[str, dict], cfg: dict[str, Any]) -> list[dict]:
    """res: per filler type, the output of analyse. Each prediction must hold in every filler type."""
    a = cfg["analysis"]
    prev_heads = [(l, h) for l, h, src in cfg["focus_heads"] if src == "Exp 1 previous-token"]
    l11 = (11, 8)
    parts: dict[str, list[tuple[bool, str]]] = {k: [] for k in ("C1", "C2", "C3", "C4", "C5", "C6", "H-loc")}
    for b, r in res.items():
        st, sp, prev, lab = r["stats"]["country"], r["stats"]["specific"], r["prev_score"], r["labels"]
        pos = st["sig"] & (st["mean"] > 0)
        strong = [hd for hd in prev_heads if prev[hd] >= a["prev_min"]]
        follow = [head_name(*hd) for hd in strong if pos[1:, hd[0], hd[1]].any()]
        parts["C1"].append((not follow, f"{b}: {len(strong)} heads with prev ≥ {a['prev_min']}; following the country "
                                        f"at d ≥ 2: {', '.join(follow) or 'none'}"))
        plain = [head_name(*hd) for hd in strong if lab[hd] == LABELS[4]]
        parts["C2"].append((len(plain) >= len(strong) / 2, f"{b}: plain {len(plain)}/{len(strong)} ({', '.join(plain)})"))
        d1, sig1 = st["mean"][0][l11], bool(st["sig"][0][l11])
        parts["C3"].append((prev[l11] < a["prev_min"] and d1 >= a["margin"] and sig1,
                            f"{b}: prev {prev[l11]:.3f}, D(1) {d1:+.4f} [{st['lo'][0][l11]:+.4f}, "
                            f"{st['hi'][0][l11]:+.4f}], q {st['q'][0][l11]:.2g}"))
        far = float(st["mean"][a["far"][0] - 1:a["far"][1], l11[0], l11[1]].mean())
        parts["C4"].append((d1 > 0 and far < d1 / 4, f"{b}: D(1) {d1:+.4f}, mean D(16–31) {far:+.4f}"))
        parts["C5"].append((sp["lo"][0][l11] > 0, f"{b}: D_country(1) − D_word(1) {sp['mean'][0][l11]:+.4f} "
                                                 f"[{sp['lo'][0][l11]:+.4f}, {sp['hi'][0][l11]:+.4f}]"))
        glob = [head_name(*idx) for idx in np.ndindex(lab.shape) if lab[idx] == LABELS[0]]
        parts["C6"].append((bool(glob), f"{b}: {', '.join(glob) or 'none'}"))
        g = r["li_group"]
        g1, g2 = list(a["layer_groups"])
        diff, late = g[f"{g1} − {g2}"], g[g2]
        parts["H-loc"].append((diff[1] > 0 and late[1] <= 0 <= late[2],
                               f"{b}: LI({g1}) {g[g1][0]:+.4f}, LI({g2}) {late[0]:+.4f} [{late[1]:+.4f}, {late[2]:+.4f}], "
                               f"difference {diff[0]:+.4f} [{diff[1]:+.4f}, {diff[2]:+.4f}]"))
    return [{"id": k, "verdict": "PASS" if all(ok for ok, _ in v) else "FAIL", "detail": "; ".join(d for _, d in v)}
            for k, v in parts.items()]


def cross_checks(tl, hf, bases: dict[str, np.ndarray], fills: dict[str, dict], ext: dict, cfg: dict) -> dict[str, Any]:
    c, T = cfg["checks"], cfg["seq_len"]
    out = {}
    for b, rows in bases.items():
        worst = 0.0
        for tau in c["hf_taus"]:
            seqs = variants(rows[: c["hf_rows"]], fills[b]["country"][: c["hf_rows"]], tau).reshape(-1, T)
            worst = max(worst, float(np.abs(attention(tl, seqs, cfg) - attention(hf, seqs, cfg)).max()))
        r = c["determinism_rows"]
        again = attention(tl, variants(rows[:r], fills[b]["country"][:r], T - 1).reshape(-1, T), cfg)
        again = again.reshape((r, -1) + again.shape[1:])[..., T - 1].mean(axis=1)
        det = float(np.abs(again - ext[b]["country"]["masses"][:r, 0, ..., 0]).max())
        out[b] = {"hf_max_abs_delta": worst, "determinism_max_abs_delta": det}
    return out


def plot_focus(res: dict, cfg: dict, path: Path) -> None:
    heads = cfg["focus_heads"]
    T, margin = cfg["seq_len"], cfg["analysis"]["margin"]
    d = np.arange(1, T)
    cols = 4
    fig, axes = plt.subplots(int(np.ceil(len(heads) / cols)), cols, figsize=(13, 2.5 * np.ceil(len(heads) / cols)),
                             sharex=True, squeeze=False)
    for ax, (l, h, src) in zip(axes.flat, heads):
        ax.axhspan(-margin, margin, color="0.93", lw=0)
        ax.axhline(0, color="0.6", lw=0.7)
        for (b, r), color in zip(res.items(), (BLUE, ORANGE)):
            st, sw = r["stats"]["country"], r["stats"]["word"]
            ax.fill_between(d, st["lo"][:, l, h], st["hi"][:, l, h], color=color, alpha=0.2, lw=0)
            ax.plot(d, st["mean"][:, l, h], color=color, lw=1.6, label=f"country, {b}")
            ax.plot(d, sw["mean"][:, l, h], color=color, lw=0.9, ls="--", label=f"word, {b}")
        labs = " / ".join(sorted({r["labels"][l, h] for r in res.values()}))
        prev = " / ".join(f"{r['prev_score'][l, h]:.2f}" for r in res.values())
        ax.set_title(f"{head_name(l, h)} ({src})\n{labs}; prev {prev}", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes.flat[len(heads):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("distance T − τ (1 = adjacent)", fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("Δ attention to τ", fontsize=7)
    axes.flat[0].legend(fontsize=6, frameon=False)
    fig.suptitle("Attention from the final token to τ: variant − control (band = 95% CI over rows; grey = ±margin)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_heads_by_distance(res: dict, cfg: dict, path: Path) -> None:
    T = cfg["seq_len"]
    fig, axes = plt.subplots(1, len(res), figsize=(5.2 * len(res), 13), squeeze=False)
    vmax = max(float(np.abs(r["stats"]["country"]["mean"]).max()) for r in res.values())
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    im = None
    for ax, (b, r) in zip(axes[0], res.items()):
        m = r["stats"]["country"]["mean"]  # [D, L, H]
        grid = m.reshape(T - 1, -1).T  # [L * H, D], row = layer * 12 + head
        im = ax.imshow(grid, aspect="auto", cmap="RdBu_r", norm=norm, origin="lower",
                       extent=(0.5, T - 0.5, -0.5, grid.shape[0] - 0.5))
        for k in range(1, m.shape[1]):
            ax.axhline(k * m.shape[2] - 0.5, color="0.75", lw=0.4)
        ax.set_yticks([k * m.shape[2] + m.shape[2] / 2 - 0.5 for k in range(m.shape[1])])
        ax.set_yticklabels([f"L{k}" for k in range(m.shape[1])], fontsize=7)
        ax.set_xlabel("distance T − τ", fontsize=8)
        ax.set_title(f"{b}: country − control, attention to τ\n(each band = one layer's heads 0–11, bottom to top)",
                     fontsize=9)
    fig.colorbar(im, ax=axes[0].tolist(), fraction=0.03, pad=0.02, label="Δ attention to τ")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_layers(res: dict, cfg: dict, path: Path) -> None:
    T = cfg["seq_len"]
    fig, axes = plt.subplots(2, len(res), figsize=(5.6 * len(res), 7.4), squeeze=False)
    vmax = max(float(np.abs(r["layer"]["mean"]).max()) for r in res.values())
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    for c, (b, r) in enumerate(res.items()):
        im = axes[0, c].imshow(r["layer"]["mean"].T, aspect="auto", origin="lower", cmap="RdBu_r", norm=norm,
                               extent=(0.5, T - 0.5, -0.5, r["layer"]["mean"].shape[1] - 0.5))
        axes[0, c].set_xlabel("distance T − τ", fontsize=8)
        axes[0, c].set_ylabel("layer", fontsize=8)
        axes[0, c].set_title(f"{b}: mean Δ attention to τ over each layer's heads", fontsize=9)
        fig.colorbar(im, ax=axes[0, c], fraction=0.046, pad=0.02)
        li = r["li_layer"]
        layers = np.arange(len(li["mean"]))
        axes[1, c].axhline(0, color="0.6", lw=0.7)
        axes[1, c].errorbar(layers, li["mean"], yerr=[li["mean"] - li["lo"], li["hi"] - li["mean"]], fmt="o",
                            color=BLUE, ms=4, capsize=2, lw=1)
        axes[1, c].set_xticks(layers)
        axes[1, c].set_xlabel("layer", fontsize=8)
        axes[1, c].set_ylabel("locality: mean Δ(d 2–4) − mean Δ(d 16–31)", fontsize=8)
        axes[1, c].spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def r4(x: np.ndarray) -> list:
    return np.round(np.asarray(x, dtype=np.float64), 4).tolist()


def viz_data(res: dict, ext: dict, bases: dict, fills: dict, cfg: dict, tokenizer) -> dict:
    """The interactive viewer's data (schema in exp4/build_viz.py's template); per-distance arrays indexed d - 1."""
    T = cfg["seq_len"]
    meta = {"seq_len": T, "query_position": T, "distances": list(range(1, T)), "fillers": list(res),
            "filler_labels": {b: f"{b} filler (ids {lo}–{hi})" for b, (lo, hi) in cfg["bases"].items()},
            "n_rows": cfg["n_rows"], "n_countries": len(cfg["stimuli"]), "margin": cfg["analysis"]["margin"],
            "focus_heads": [[l, h, src] for l, h, src in cfg["focus_heads"]],
            "example_tokens": {b: ["BOS"] + [tokenizer.decode([int(t)]) for t in bases[b][0]] for b in res},
            "country_example": cfg["stimuli"][0]["country"],
            "word_example": {b: tokenizer.decode([int(fills[b]["word"][0, 0])]) for b in res}}
    data = {}
    for b, r in res.items():
        data[b] = {"control": r4(ext[b]["control"].mean(axis=0)),
                   **{arm: r4(ext[b][arm]["by_key"]) for arm in ARMS},
                   "diff": {arm: {k: r4(np.moveaxis(r["stats"][arm][k], 0, -1)) for k in ("mean", "lo", "hi")}
                            | {"sig": np.moveaxis(r["stats"][arm]["sig"], 0, -1).astype(int).tolist()} for arm in ARMS},
                   "prev_score": r4(r["prev_score"]), "label": r["labels"].tolist(),
                   "layer_mean": {k: r4(r["layer"][k].T) for k in ("mean", "lo", "hi")}}
    return {"meta": meta, "data": data}


def write_report(run: Path, cfg: dict, res: dict, checks: dict, preds: list[dict]) -> None:
    a, T = cfg["analysis"], cfg["seq_len"]
    L = ["# Experiment 4: controlled difference, country vs original token at τ", "",
         f"Run `{run.name}`. {cfg['n_rows']} random rows per filler type; per row and τ = 1..{T - 1}, the country variant "
         f"(11 countries, averaged within row) and a matched placebo word vs the unchanged row. Measure: attention "
         f"from the final token ({T}) to τ, per head. d = {T} − τ. Meaningful = BH q ≤ {a['q']} over the "
         f"{T - 1} × 144 cells and |D| ≥ {a['margin']}.", "", "## Checks", "", "| check | result |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in checks.items()]
    L += ["", "## Pre-registered predictions", "", "| id | verdict | detail |", "|---|---|---|"]
    L += [f"| {p['id']} | {p['verdict']} | {p['detail']} |" for p in preds]
    L += ["", "## Focus heads", "",
          "| head | source | filler | prev_score | D(1) [95% CI] | mean D d 2–4 | mean D d 16–31 | positive d ≥ 2 | "
          "country − word at d 1 [95% CI] | label |", "|---|---|---|---|---|---|---|---|---|---|"]
    for l, h, src in cfg["focus_heads"]:
        for b, r in res.items():
            st, sp = r["stats"]["country"], r["stats"]["specific"]
            pos = int((st["sig"] & (st["mean"] > 0))[1:, l, h].sum())
            L.append(f"| {head_name(l, h)} | {src} | {b} | {r['prev_score'][l, h]:.3f} | {st['mean'][0, l, h]:+.4f} "
                     f"[{st['lo'][0, l, h]:+.4f}, {st['hi'][0, l, h]:+.4f}] | "
                     f"{st['mean'][a['near'][0] - 1:a['near'][1], l, h].mean():+.4f} | "
                     f"{st['mean'][a['far'][0] - 1:a['far'][1], l, h].mean():+.4f} | {pos}/{T - 2} | "
                     f"{sp['mean'][0, l, h]:+.4f} [{sp['lo'][0, l, h]:+.4f}, {sp['hi'][0, l, h]:+.4f}] | {r['labels'][l, h]} |")
    L += ["", "## Label counts (all 144 heads)", "", "| filler | " + " | ".join(LABELS) + " |", "|---" * (len(LABELS) + 1) + "|"]
    for b, r in res.items():
        L.append(f"| {b} | " + " | ".join(str(int((r["labels"] == lab).sum())) for lab in LABELS) + " |")
    L += ["", "## Strongest far-distance country effects (mean D over d 16–31)", "",
          "| filler | head | mean D d 16–31 | D(1) | positive d ≥ 2 | label |", "|---|---|---|---|---|---|"]
    for b, r in res.items():
        st = r["stats"]["country"]
        far = st["mean"][a["far"][0] - 1:a["far"][1]].mean(axis=0)
        for idx in np.argsort(-far, axis=None)[:8]:
            l, h = np.unravel_index(idx, far.shape)
            pos = int((st["sig"] & (st["mean"] > 0))[1:, l, h].sum())
            L.append(f"| {b} | {head_name(l, h)} | {far[l, h]:+.4f} | {st['mean'][0, l, h]:+.4f} | {pos}/{T - 2} | "
                     f"{r['labels'][l, h]} |")
    L += ["", "## Locality by layer (stretch): mean Δ(d 2–4) − mean Δ(d 16–31), mean over the layer's heads", "",
          "| filler | " + " | ".join(f"L{k}" for k in range(12)) + " |", "|---" * 13 + "|"]
    for b, r in res.items():
        li = r["li_layer"]
        L.append(f"| {b} | " + " | ".join(f"{m:+.4f}" for m in li["mean"]) + " |")
    for b, r in res.items():
        L.append("")
        L.append(f"{b}: " + "; ".join(f"LI({g}) {v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}]" for g, v in r["li_group"].items()))
    L += ["", "## Files", "",
          "- `controlled.npz`: per filler type, the stats of each contrast [distance, layer, head], prev_score, labels, "
          "the mean attention rows (control [L, H, K]; country / word [distance, L, H, K]), the per-row control "
          "attention [R, L, H, K] and per-row arm masses on τ / BOS / previous [R, distance, L, H, 3] "
          "(`row_differences` rebuilds the per-row contrasts), and the placebo words.",
          "- `figures/focus_heads.png`: D(d) for the focus heads, both filler types, country (solid) and word (dashed).",
          "- `figures/heads_by_distance.png`: D(d) for all 144 heads.",
          "- `figures/layers_by_distance.png`: mean D over each layer's heads, and locality by layer.",
          "- `viz_data.json`: input of `exp4.build_viz` (the interactive viewer).", ""]
    (run / "exp4_controlled_report.md").write_text("\n".join(L))


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Experiment 4, controlled difference")
    parser.add_argument("--config", default="configs/exp4_controlled.yaml", type=Path)
    parser.add_argument("--rows", default=None, type=int, help="use only the first rows (smoke test)")
    parser.add_argument("--results-dir", default=None, type=Path, help="override paths.results_dir (smoke tests)")
    args = parser.parse_args(argv)
    cfg_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    cfg = yaml.safe_load(cfg_path.read_text())
    if args.rows:
        cfg["n_rows"] = args.rows
    t0 = time.time()
    run = make_run_dir(args.results_dir or PROJECT_ROOT / cfg["paths"]["results_dir"], tag="exp4ctrl")
    save_config_copy(cfg_path, run)
    shutil.copy2(PROJECT_ROOT / "exp4" / "PREDICTIONS.md", run / "PREDICTIONS.md")
    save_versions(run, PROJECT_ROOT, code_dirs=("exp1", "exp2", "exp3", "exp4"))
    (run / "figures").mkdir()
    log(f"run folder {run}", t0)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg["figures"]["tokenizer"])
    set_determinism(cfg["seed"])
    tl = load_model("transformer_lens", cfg["backends"]["transformer_lens"]["model_name"], cfg["device"], cfg["dtype"])
    stim = stimuli(cfg, tokenizer)
    countries = stim[0]
    stim_ids = np.concatenate([*stim, cfg["demonstration_ids"]])
    lo = min(r[0] for r in cfg["token_classes"].values())
    hi = max(r[1] for r in cfg["token_classes"].values())
    pools = matched_pools(countries, token_features(np.arange(lo, hi + 1), tokenizer), cfg["match_features"],
                          cfg["token_classes"])
    pools = [p[~np.isin(p, stim_ids)] for p in pools]
    log(f"placebo pool sizes {[len(p) for p in pools]}", t0)

    bases, fills, ext, res = {}, {}, {}, {}
    for b, rng_ in cfg["bases"].items():
        block = cached_base_block(f"exp3rel_{b}", rng_, stim_ids, cfg["base_cache_n"], cfg["seq_len"], cfg["seed"],
                                  cfg["bos_token_id"], PROJECT_ROOT / cfg["paths"]["sequences_dir"])
        bases[b] = block[: cfg["n_rows"]]
        if np.isin(bases[b], stim_ids).any():
            raise AssertionError(f"{b}: a stimulus or demonstration id occurs in a base row")
        fills[b] = {"country": np.tile(countries, (len(bases[b]), 1)),
                    "word": placebo_words(bases[b], pools, cfg["word_seed"])}
        ext[b] = extract(tl, bases[b], fills[b], cfg, t0, b)
        res[b] = analyse(ext[b], cfg)
        log(f"{b}: analysis done", t0)

    hf = load_model("huggingface", cfg["backends"]["huggingface"]["model_name"], cfg["device"], cfg["dtype"],
                    cfg["backends"]["huggingface"]["attn_implementation"])
    cross = cross_checks(tl, hf, bases, fills, ext, cfg)
    log("cross-library and determinism checks done", t0)

    c = cfg["checks"]
    checks = {"design (ids at τ, row elsewhere, no repeats/BOS/stimulus ids)": "PASS (asserted per τ and arm)",
              "attention rows sum to 1": f"PASS (atol {c['row_sum_atol']:g}, asserted per batch)"}
    for b, x in cross.items():
        checks[f"determinism, {b}"] = f"{'PASS' if x['determinism_max_abs_delta'] == 0 else 'FAIL'} (max |Δ| {x['determinism_max_abs_delta']:.1e})"
        checks[f"TL vs HF, {b}"] = f"{'PASS' if x['hf_max_abs_delta'] < c['hf_atol'] else 'FAIL'} (max |Δ| {x['hf_max_abs_delta']:.1e})"
        ps = float(res[b]["prev_score"][tuple(c["prev_sanity_head"])])
        checks[f"{head_name(*c['prev_sanity_head'])} prev_score ≥ {c['prev_sanity_min']}, {b}"] = (
            f"{'PASS' if ps >= c['prev_sanity_min'] else 'FAIL'} ({ps:.3f})")
    preds = check_predictions(res, cfg)

    np.savez_compressed(run / "controlled.npz",
                        **{f"{b}/{con}/{k}": r["stats"][con][k] for b, r in res.items() for con in CONTRASTS
                           for k in ("mean", "lo", "hi", "p", "q", "sig")},
                        **{f"{b}/prev_score": r["prev_score"] for b, r in res.items()},
                        **{f"{b}/labels": r["labels"].astype(str) for b, r in res.items()},
                        **{f"{b}/control_by_key": ext[b]["control"].mean(axis=0) for b in res},
                        **{f"{b}/{arm}_by_key": ext[b][arm]["by_key"] for b in res for arm in ARMS},
                        **{f"{b}/{arm}_masses": ext[b][arm]["masses"] for b in res for arm in ARMS},
                        **{f"{b}/control": ext[b]["control"] for b in res},
                        **{f"{b}/words": fills[b]["word"] for b in res})
    (run / "checks.json").write_text(json.dumps(checks, indent=2))
    (run / "predictions.json").write_text(json.dumps(preds, indent=2))
    (run / "viz_data.json").write_text(json.dumps(viz_data(res, ext, bases, fills, cfg, tokenizer), separators=(",", ":")))
    plot_focus(res, cfg, run / "figures" / "focus_heads.png")
    plot_heads_by_distance(res, cfg, run / "figures" / "heads_by_distance.png")
    plot_layers(res, cfg, run / "figures" / "layers_by_distance.png")
    write_report(run, cfg, res, checks, preds)
    log(f"done: {run / 'exp4_controlled_report.md'}", t0)
    for k, v in checks.items():
        print(f"{k}: {v}")
    for p in preds:
        print(f"{p['id']}: {p['verdict']}  {p['detail']}")
    return run


if __name__ == "__main__":
    main()
