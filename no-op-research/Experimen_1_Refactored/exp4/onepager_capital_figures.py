"""Figures for the Experiment 4b one-pager (capital as the final token; saved arrays only, no model run).

exp4b_heads.png:  D(d) for four heads with its capital, a wrong capital and a random final token (Exp 4), one row per
                  filler type.
exp4b_fact.png:   fact contrast (its capital − wrong capital) on the attention to τ, heads x distance, per filler type,
                  and the late-layer BOS fact by distance.

Run:  uv run --offline python -m exp4.onepager_capital_figures --run results/<timestamp>_exp4cap [--out onepager]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write figures to files, never open a window
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.colors import TwoSlopeNorm

from exp3.attention_maps3 import AXIS_GREY, TEXT_GREY
from exp4.onepager_figures import style
from exp4.run_controlled import PROJECT_ROOT, head_name

GREEN, GREY, BLUE = "#1baf7a", "#898781", "#2a78d6"


def fig_heads(cfg: dict, z, zr, heads: list[tuple[int, int, str]], path: Path) -> None:
    T, margin = cfg["seq_len"], cfg["analysis"]["margin"]
    d = np.arange(1, T)
    fillers = list(cfg["bases"])
    fig, axes = plt.subplots(len(fillers), len(heads), figsize=(7.6, 1.5 * len(fillers) + 0.6), squeeze=False)
    for r, b in enumerate(fillers):
        for c, (l, h, title) in enumerate(heads):
            ax = axes[r, c]
            own = {k: z[f"own/{b}/country/{k}"][:, l, h] for k in ("mean", "lo", "hi")}
            wrong, rnd = z[f"wrong/{b}/country/mean"][:, l, h], zr[f"{b}/country/mean"][:, l, h]
            ax.axhspan(-margin, margin, color="0.94", lw=0)
            ax.axhline(0, color="0.7", lw=0.6)
            ax.plot(d, rnd, color=BLUE, lw=0.9, ls=(0, (2.5, 1.5)))
            ax.plot(d, wrong, color=GREY, lw=1.1)
            ax.fill_between(d, own["lo"], own["hi"], color=GREEN, alpha=0.22, lw=0)
            ax.plot(d, own["mean"], color=GREEN, lw=1.5)
            lo = min(own["lo"].min(), wrong.min(), rnd.min(), -margin)
            hi = max(own["hi"].max(), wrong.max(), rnd.max(), margin)
            pad = 0.08 * (hi - lo)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_xticks([1, 10, 20, 31])
            if r == 0:
                ax.set_title(f"{head_name(l, h)}: {title}", fontsize=6.5, pad=4)
            style(ax, "distance T − τ" if r == len(fillers) - 1 else None,
                  f"{b}: Δ attention to τ" if c == 0 else None, size=5.5)
    handles = [plt.Line2D([], [], color=GREEN, lw=1.5, label="T = its capital (band = 95% CI)"),
               plt.Line2D([], [], color=GREY, lw=1.1, label="T = a wrong capital"),
               plt.Line2D([], [], color=BLUE, lw=0.9, ls=(0, (2.5, 1.5)), label="T = a random token (Exp 4)"),
               plt.Rectangle((0, 0), 1, 1, color="0.94", label=f"±{margin} (≈ 0)")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=5.8, bbox_to_anchor=(0.5, 0),
               handlelength=1.6)
    fig.subplots_adjust(left=0.075, right=0.995, top=0.9, bottom=0.17, wspace=0.3, hspace=0.25)
    fig.savefig(path, dpi=250)
    plt.close(fig)
    print(f"saved {path}")


def fig_fact(cfg: dict, z, groups: list[list[tuple[int, int]]], path: Path) -> None:
    T = cfg["seq_len"]
    heads = [hd for g in groups for hd in g]
    fillers = list(cfg["bases"])
    grids = {b: np.stack([z[f"{b}/fact/mean"][:, l, h] for l, h in heads]) for b in fillers}
    sigs = {b: np.stack([z[f"{b}/fact/sig"][:, l, h] for l, h in heads]) for b in fillers}
    vmax = max(float(np.abs(g).max()) for g in grids.values())
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    fig = plt.figure(figsize=(7.6, 2.5))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 0.035, 0.3, 0.85], left=0.065, right=0.99, top=0.88, bottom=0.18,
                          wspace=0.08)
    im = None
    for k, b in enumerate(fillers):
        ax = fig.add_subplot(gs[0, k])
        im = ax.imshow(grids[b], aspect="auto", cmap="RdBu_r", norm=norm, extent=(0.5, T - 0.5, len(heads) - 0.5, -0.5))
        r, c = np.nonzero(sigs[b])
        ax.scatter(c + 1, r, s=1.2, color="black", linewidths=0)
        ax.set_yticks(range(len(heads)))
        ax.set_yticklabels([head_name(l, h) for l, h in heads] if k == 0 else [])
        ax.set_xticks([1, 10, 20, 31])
        ax.set_title(f"fact: its − wrong capital, {b}", fontsize=6.3, pad=4)
        style(ax, "distance T − τ", size=5.2)
        for spine in ax.spines.values():
            spine.set_visible(True)
        for edge in np.cumsum([len(g) for g in groups])[:-1]:
            ax.axhline(edge - 0.5, color="0.35", lw=0.6)
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[0, 2]))
    cb.ax.tick_params(labelsize=5, colors=AXIS_GREY, labelcolor=TEXT_GREY)
    cb.outline.set_edgecolor(AXIS_GREY)
    ax = fig.add_subplot(gs[0, 4])
    d = np.arange(1, T)
    ax.axhline(0, color="0.7", lw=0.6)
    for b, color in zip(fillers, (BLUE, "#eb6834")):
        rows = z[f"{b}/bos_fact_rows"]
        m = rows.mean(axis=0)
        se = rows.std(axis=0, ddof=1) / np.sqrt(len(rows))
        ax.fill_between(d, m - 1.96 * se, m + 1.96 * se, color=color, alpha=0.2, lw=0)
        ax.plot(d, m, color=color, lw=1.3, label=b)
    ax.set_xticks([1, 10, 20, 31])
    ax.set_title("late-layer BOS mass: its − wrong capital", fontsize=6.3, pad=4)
    ax.legend(fontsize=5.5, frameon=False)
    style(ax, "distance T − τ", size=5.2)
    fig.savefig(path, dpi=250)
    plt.close(fig)
    print(f"saved {path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Experiment 4b one-pager figures")
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", default=PROJECT_ROOT / "onepager", type=Path)
    args = parser.parse_args(argv)
    run = args.run if args.run.is_absolute() else PROJECT_ROOT / args.run
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    z = np.load(run / "capital.npz")
    zr = np.load(PROJECT_ROOT / cfg["random_run"] / "controlled.npz")
    args.out.mkdir(parents=True, exist_ok=True)
    fig_heads(cfg, z, zr, [(0, 2, "same-country association"), (11, 8, "advisor's head"), (3, 7, "previous-token"),
                           (9, 8, "late, any distance")], args.out / "exp4b_heads.png")
    prev = [(l, h) for l, h, src in cfg["focus_heads"] if src == "Exp 1 previous-token"]
    lookup = [(0, 2), (1, 11), (3, 0), (4, 10), (9, 8)]  # the largest positive fact contrasts in the run report
    fig_fact(cfg, z, [prev, [(11, 8)], lookup, [(10, 3), (11, 7), (11, 2)]], args.out / "exp4b_fact.png")


if __name__ == "__main__":
    main()
