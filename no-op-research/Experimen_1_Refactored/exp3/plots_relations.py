"""Figures for Experiment 3, class 2 (R1-out). Red = more BOS attention in correct cells (attention-map convention)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write figures to files, never open a window
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

from exp3.relations import double_centre

BASES = ("rare", "common")
GRIDS = ("capital", "city")


def plot_head_maps(head_t: dict[tuple[str, str], dict[str, np.ndarray]], path: Path,
                   title: str = "Correct vs wrong pairing before a random final token") -> None:
    """head_t[(base, grid)] = head_effects(...) for BOS mass. 2 x 2 grid of [layer, head] maps of T (correct - wrong);
    meaningful heads are outlined."""
    vmax = max(float(np.abs(h["mean"]).max()) for h in head_t.values()) or 1.0
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 8.4), squeeze=False)
    im = None
    for i, base in enumerate(BASES):
        for j, grid in enumerate(GRIDS):
            ax, h = axes[i, j], head_t[(base, grid)]
            im = ax.imshow(h["mean"], origin="lower", cmap="RdBu_r", norm=norm)
            for layer, head in zip(*np.nonzero(h["meaningful"])):
                ax.add_patch(Rectangle((head - 0.5, layer - 0.5), 1, 1, fill=False, lw=1.4, ec="black"))
            ax.set_title(f"{base} bases, country + {grid}: {int(h['meaningful'].sum())} heads meaningful", fontsize=10)
            ax.set_xticks(range(0, 12, 2))
            ax.set_yticks(range(0, 12, 2))
            ax.set_xlabel("Head", fontsize=8)
            ax.set_ylabel("Layer", fontsize=8)
            ax.tick_params(labelsize=7)
    fig.subplots_adjust(left=0.07, right=0.86, top=0.9, bottom=0.07, hspace=0.3, wspace=0.25)
    box_top, box_bot = axes[0, -1].get_position(), axes[-1, -1].get_position()
    cax = fig.add_axes((box_top.x1 + 0.02, box_bot.y0, 0.015, box_top.y1 - box_bot.y0))
    fig.colorbar(im, cax=cax, label="T = BOS mass, correct − wrong")
    fig.suptitle(f"{title} (T per head; outlined = meaningful)", fontsize=11)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pooled(pooled: dict[tuple[str, str], dict], placebo: dict[tuple[str, str], np.ndarray], margin: float,
                path: Path) -> None:
    """Late-layer T on BOS mass with its 95% row CI per base x grid, the 20 placebo derangements as gray dots,
    and the equivalence band +-margin."""
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.axhspan(-margin, margin, color="0.92", zorder=0)
    ax.axhline(0, color="0.5", lw=0.8, zorder=1)
    labels = []
    for x, (base, grid) in enumerate((b, g) for b in BASES for g in GRIDS):
        r = pooled[(base, grid)]
        rng = np.random.default_rng(x)
        ax.scatter(x + 0.18 + rng.uniform(-0.05, 0.05, len(placebo[(base, grid)])), placebo[(base, grid)], s=10,
                   color="0.6", zorder=2, label="placebo derangements" if x == 0 else None)
        ax.errorbar(x, r["mean"], yerr=[[r["mean"] - r["lo"]], [r["hi"] - r["mean"]]], fmt="o", ms=8,
                    color="#2a78d6", capsize=4, lw=2, zorder=3, label="correct − wrong (95% CI)" if x == 0 else None)
        ax.annotate(r["verdict"], (x, r["hi"]), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
        labels.append(f"{base}\n{grid}")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("late-layer BOS mass, T", fontsize=9)
    ax.set_title(f"Layers 6–11 pooled; gray band = ±{margin} (equivalence)", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_residual_grids(grid_means: dict[tuple[str, str], np.ndarray], countries: list[str],
                        partners: dict[str, list[str]], path: Path, positions: tuple[int, int] = (30, 31)) -> None:
    """grid_means[(base, grid)] = [n, n] mean late BOS over rows. Shows the double-centred residuals (each token's own
    effect removed); a positive diagonal means correct pairings carry extra BOS attention."""
    res = {k: double_centre(v) for k, v in grid_means.items()}
    vmax = max(float(np.abs(r).max()) for r in res.values()) or 1.0
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 10.5), squeeze=False)
    im = None
    for i, base in enumerate(BASES):
        for j, grid in enumerate(GRIDS):
            ax, r = axes[i, j], res[(base, grid)]
            im = ax.imshow(r, cmap="RdBu_r", norm=norm)
            n = r.shape[0]
            for d in range(n):
                ax.add_patch(Rectangle((d - 0.5, d - 0.5), 1, 1, fill=False, lw=1.2, ec="black"))
            ax.set_xticks(range(n))
            ax.set_xticklabels([p.strip() for p in partners[grid]], rotation=60, ha="right", fontsize=7)
            ax.set_yticks(range(n))
            ax.set_yticklabels([c.strip() for c in countries], fontsize=7)
            ax.set_title(f"{base} bases, country ({positions[0]}) × {grid} ({positions[1]})", fontsize=10)
    fig.subplots_adjust(left=0.09, right=0.88, top=0.93, bottom=0.1, hspace=0.38, wspace=0.3)
    box_top, box_bot = axes[0, -1].get_position(), axes[-1, -1].get_position()
    cax = fig.add_axes((box_top.x1 + 0.02, box_bot.y0, 0.015, box_top.y1 - box_bot.y0))
    fig.colorbar(im, cax=cax, label="late BOS mass, residual after removing country and partner means")
    fig.suptitle("Which pairings move late-layer BOS attention (diagonal outlined = correct pairs)", fontsize=11)
    fig.savefig(path, dpi=150)
    plt.close(fig)
