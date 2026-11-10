"""Figures for the Experiment 4 one-pager (saved arrays only, no model run).

exp4_heads.png:  D(d) for four heads that show the four behaviours (country vs placebo word, both filler types).
exp4_grid.png:   D(d), heads x distance, per filler type: Exp 1's previous-token heads, L11H8, the global trackers;
                 dots = meaningful cells.
exp4_layers.png: mean D over each layer's heads, layer x distance, and the share of the near effect left at each
                 distance per layer group (exploratory).

Run:  uv run --offline python -m exp4.onepager_figures --run results/<timestamp>_exp4ctrl [--out onepager]
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

from exp1ext.effects import bootstrap_mean_per_unit
from exp3.attention_maps3 import AXIS_GREY, TEXT_GREY
from exp4.run_controlled import ARMS, PROJECT_ROOT, head_name, row_differences, span_mean

BLUE, ORANGE = "#2a78d6", "#eb6834"
FILLER_COLOR = {"common": BLUE, "rare": ORANGE}


def style(ax, xlabel: str | None = None, ylabel: str | None = None, size: float = 6.0) -> None:
    """Light-grey frame and index numbers, black axis titles (as in the other one-pager figures)."""
    for side, spine in ax.spines.items():
        spine.set_color(AXIS_GREY)
        spine.set_visible(side in ("left", "bottom"))
    ax.tick_params(colors=AXIS_GREY, labelcolor=TEXT_GREY, labelsize=size, length=2, pad=1.5)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=size + 0.5, color="black", labelpad=2)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=size + 0.5, color="black", labelpad=2)


def load(run: Path) -> tuple[dict, dict]:
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    z = np.load(run / "controlled.npz")
    return cfg, {k: z[k] for k in z.files}


def stat(z: dict, b: str, con: str, k: str) -> np.ndarray:
    return z[f"{b}/{con}/{k}"]  # [D, L, H]


def fig_heads(cfg: dict, z: dict, heads: list[tuple[int, int, str]], path: Path) -> None:
    T, margin = cfg["seq_len"], cfg["analysis"]["margin"]
    d = np.arange(1, T)
    fillers = list(cfg["bases"])
    fig, axes = plt.subplots(1, len(heads), figsize=(7.6, 2.05))
    for ax, (l, h, title) in zip(axes, heads):
        lo = min(float(stat(z, b, "country", "lo")[:, l, h].min()) for b in fillers)
        hi = max(float(stat(z, b, c, k)[:, l, h].max()) for b in fillers for c, k in (("country", "hi"), ("word", "mean")))
        pad = 0.08 * max(hi - lo, 2 * margin)
        ax.axhspan(-margin, margin, color="0.94", lw=0)
        ax.axhline(0, color="0.7", lw=0.6)
        for b in fillers:
            c = FILLER_COLOR[b]
            ax.fill_between(d, stat(z, b, "country", "lo")[:, l, h], stat(z, b, "country", "hi")[:, l, h], color=c,
                            alpha=0.18, lw=0)
            ax.plot(d, stat(z, b, "country", "mean")[:, l, h], color=c, lw=1.3)
            ax.plot(d, stat(z, b, "word", "mean")[:, l, h], color=c, lw=0.8, ls=(0, (2.5, 1.5)))
        prev = " / ".join(f"{z[f'{b}/prev_score'][l, h]:.2f}" for b in fillers)
        ax.set_title(f"{head_name(l, h)}: {title}\nprevious-token share {prev}", fontsize=6.3, pad=4)
        ax.set_ylim(min(lo, -margin) - pad, max(hi, margin) + pad)
        ax.set_xticks([1, 10, 20, 31])
        style(ax, "distance T − τ", "Δ attention to τ" if ax is axes[0] else None, size=5.5)
    handles = [plt.Line2D([], [], color=FILLER_COLOR[b], lw=1.3, label=f"country, {b} filler (band = 95% CI)")
               for b in fillers]
    handles += [plt.Line2D([], [], color="0.4", lw=0.8, ls=(0, (2.5, 1.5)), label="matched word (placebo)"),
                plt.Rectangle((0, 0), 1, 1, color="0.94", label=f"±{margin} (≈ 0)")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=5.8,
               bbox_to_anchor=(0.5, 0), handlelength=1.6)
    fig.subplots_adjust(left=0.065, right=0.995, top=0.8, bottom=0.27, wspace=0.28)
    fig.savefig(path, dpi=250)
    plt.close(fig)
    print(f"saved {path}")


def fig_grid(cfg: dict, z: dict, groups: list[list[tuple[int, int]]], path: Path) -> None:
    """Heads (rows, in groups separated by a line) x distance, per filler type."""
    T = cfg["seq_len"]
    heads = [hd for g in groups for hd in g]
    fillers = list(cfg["bases"])
    grids = {b: np.stack([stat(z, b, "country", "mean")[:, l, h] for l, h in heads]) for b in fillers}
    sigs = {b: np.stack([stat(z, b, "country", "sig")[:, l, h] for l, h in heads]) for b in fillers}
    vmax = max(float(np.abs(g).max()) for g in grids.values())
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    fig, axes = plt.subplots(1, len(fillers), figsize=(7.6, 2.45), sharey=True)
    im = None
    for ax, b in zip(axes, fillers):
        im = ax.imshow(grids[b], aspect="auto", cmap="RdBu_r", norm=norm, extent=(0.5, T - 0.5, len(heads) - 0.5, -0.5))
        r, c = np.nonzero(sigs[b])
        ax.scatter(c + 1, r, s=1.2, color="black", linewidths=0)
        ax.set_yticks(range(len(heads)))
        ax.set_yticklabels([head_name(l, h) for l, h in heads])
        ax.set_xticks([1, 5, 10, 15, 20, 25, 31])
        ax.set_title(f"{b} filler", fontsize=7, pad=4)
        style(ax, "distance T − τ (1 = adjacent)")
        for spine in ax.spines.values():
            spine.set_visible(True)
    for ax in axes:
        for edge in np.cumsum([len(g) for g in groups])[:-1]:
            ax.axhline(edge - 0.5, color="0.35", lw=0.6)
    cb = fig.colorbar(im, ax=list(axes), fraction=0.025, pad=0.015)
    cb.set_label("Δ attention to τ, country − control", fontsize=6, color="black")
    cb.ax.tick_params(labelsize=5.5, colors=AXIS_GREY, labelcolor=TEXT_GREY)
    cb.outline.set_edgecolor(AXIS_GREY)
    fig.subplots_adjust(left=0.07, right=0.88, top=0.9, bottom=0.18, wspace=0.06)
    fig.savefig(path, dpi=250)
    plt.close(fig)
    print(f"saved {path}")


def retention(x: np.ndarray, groups: dict[str, tuple[int, int]], near: list[int], iters: int,
              seed: int) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Share of a layer group's near effect left at each distance: mean D(d) / mean D(near) over the group's heads, with a
    pointwise 95% percentile CI from resampling rows (the same resample for numerator and denominator).
    x [R, D, L, H] -> per group (share, lo, hi), each [D]. Exploratory (not in PREDICTIONS.md)."""
    rng = np.random.default_rng(seed)
    draws = np.stack([rng.integers(0, len(x), len(x)) for _ in range(iters)])  # [B, R]

    def share(rows: np.ndarray) -> np.ndarray:  # rows [..., R, D]
        return rows.mean(axis=-2) / rows[..., near[0] - 1:near[1]].mean(axis=(-2, -1))[..., None]

    out = {}
    for g, (lo, hi) in groups.items():
        per_row = x[:, :, lo:hi + 1].mean(axis=(2, 3))  # [R, D]
        lo_, hi_ = np.percentile(share(per_row[draws]), [2.5, 97.5], axis=0)
        out[g] = (share(per_row), lo_, hi_)
    return out


def fig_layers(cfg: dict, z: dict, groups: dict[str, tuple[int, int]], path: Path) -> None:
    a, T = cfg["analysis"], cfg["seq_len"]
    boot = a["bootstrap"]
    fillers = list(cfg["bases"])
    layer, keep = {}, {}
    for b in fillers:
        ext = {"control": z[f"{b}/control"], **{arm: {"masses": z[f"{b}/{arm}_masses"]} for arm in ARMS}}
        x = row_differences(ext, T)["country"]  # [R, D, L, H]
        layer[b] = x.mean(axis=(0, 3))  # [D, L]
        keep[b] = retention(x, groups, a["near"], boot["iters"], boot["seed"])
        for w, (w0, w1) in {"mid": (5, 15), "far": a["far"]}.items():  # window means of the curve, for the log
            print(f"{b} share over d {w0}-{w1} ({w}): " + "; ".join(f"{g} {m[w0 - 1:w1].mean():.2f}" for g, (m, _, _) in keep[b].items()))
    vmax = max(float(np.abs(g).max()) for g in layer.values())
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    fig = plt.figure(figsize=(7.6, 2.5))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 0.04, 0.1, 1.7], left=0.06, right=0.99, top=0.8, bottom=0.185,
                          wspace=0.25)
    im = None
    for k, b in enumerate(fillers):
        ax = fig.add_subplot(gs[0, k])
        im = ax.imshow(layer[b].T, aspect="auto", origin="lower", cmap="RdBu_r", norm=norm,
                       extent=(0.5, T - 0.5, -0.5, layer[b].shape[1] - 0.5))
        ax.set_yticks(range(layer[b].shape[1]))
        ax.set_xticks([1, 5, 10, 15, 20, 25, 31])
        ax.set_title(f"mean Δ over heads, {b}", fontsize=6.5, pad=4)
        style(ax, "distance T − τ", "layer" if k == 0 else None, size=5.5)
        for spine in ax.spines.values():
            spine.set_visible(True)
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[0, 2]))
    cb.ax.tick_params(labelsize=5, colors=AXIS_GREY, labelcolor=TEXT_GREY)
    cb.outline.set_edgecolor(AXIS_GREY)
    near, d = a["near"], np.arange(2, T)  # d = 1 is left out
    group_color = dict(zip(groups, (plt.cm.viridis(v) for v in (0.1, 0.45, 0.75))))
    curve_axes = gs[0, 4].subgridspec(1, 2, wspace=0.12)
    axes = []
    for k, b in enumerate(fillers):
        ax = fig.add_subplot(curve_axes[0, k], sharey=axes[0] if axes else None)
        axes.append(ax)
        ax.axvspan(near[0] - 0.5, near[1] + 0.5, color="0.94", lw=0)
        ax.axhline(1, color="0.7", lw=0.6)
        for g, c in group_color.items():
            m, lo, hi = keep[b][g]
            ax.fill_between(d, lo[1:], hi[1:], color=c, alpha=0.18, lw=0)
            ax.plot(d, m[1:], color=c, lw=1.2, label=f"layers {g}")
        ax.set_xlim(2, T - 1)
        ax.set_xticks([2, 10, 20, 31])
        ax.set_title(f"{b} filler", fontsize=6.5, pad=4)
        style(ax, "distance T − τ", f"Δ(d) ÷ mean Δ, d {near[0]}–{near[1]}" if k == 0 else None, size=5.5)
        if k:
            ax.tick_params(labelleft=False)
    fig.text(0.5 * (axes[0].get_position().x0 + axes[1].get_position().x1), 0.985,
             f"share of the near effect left at each distance\n(÷ mean Δ over d {near[0]}–{near[1]}; exploratory; band = 95% CI)",
             ha="center", va="top", fontsize=6.3)
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Rectangle((0, 0), 1, 1, color="0.94"))
    labels.append(f"d {near[0]}–{near[1]} (= 1 on average)")
    axes[0].legend(handles, labels, fontsize=5.2, frameon=False, loc="upper right", handlelength=1.4)
    fig.savefig(path, dpi=250)
    plt.close(fig)
    print(f"saved {path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Experiment 4 one-pager figures")
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", default=PROJECT_ROOT / "onepager", type=Path)
    args = parser.parse_args(argv)
    run = args.run if args.run.is_absolute() else PROJECT_ROOT / args.run
    cfg, z = load(run)
    args.out.mkdir(parents=True, exist_ok=True)
    fig_heads(cfg, z, [(11, 8, "country, fading"), (4, 11, "pure previous-token"), (3, 6, "any nearby word"),
                       (9, 8, "country, any distance")], args.out / "exp4_heads.png")
    prev = [(l, h) for l, h, src in cfg["focus_heads"] if src == "Exp 1 previous-token"]
    fig_grid(cfg, z, [prev, [(11, 8)], [(9, 8), (10, 3), (11, 7), (11, 2)]], args.out / "exp4_grid.png")
    fig_layers(cfg, z, {"2-3": (2, 3), "5-8": (5, 8), "9-11": (9, 11)}, args.out / "exp4_layers.png")


if __name__ == "__main__":
    main()
