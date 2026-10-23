"""Figures for Experiment 3, class 1: per-head maps in the style of exp2/attention_maps.py, and role transition
matrices. Red = more attention in the treated arm (attention-map convention)."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # write figures to files, never open a window
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from matplotlib.patches import Patch, Rectangle

# One colour per engine.LABELS entry: Diffuse, Sink, Previous, Self, Copy, Dup, Delimiter, Span, Filler
LABEL_COLORS = ("#f2f2f2", "tab:red", "tab:blue", "tab:purple", "tab:green", "tab:olive", "tab:brown", "#eda100",
                "#a8a8a8")


def plot_head_maps(panels: dict[str, dict[str, np.ndarray]], kinds: dict[str, str], colorbar_labels: dict[str, str],
                   suptitle: str, path: Path, outline: dict[tuple[str, str], np.ndarray] | None = None,
                   label_names: Sequence[str] | None = None) -> None:
    """One figure row per panels key, one column per inner key (the same columns in every row). Each map is
    [layer, head]. kinds[row] picks the colour scale:
    - 'mass': 0-1, viridis;
    - 'delta': diverging, symmetric within the row;
    - 'label': int codes into label_names, with a legend instead of a colorbar.
    outline[(row, col)] is a bool [layer, head]; those heads are boxed. Columns are the union over rows, in first-seen
    order; a row without a column shows that cell as 'not compared'."""
    rows = list(panels)
    cols = list(dict.fromkeys(c for row in rows for c in panels[row]))
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(2.9 * len(cols) + 1.6, 2.9 * len(rows) + 0.9), squeeze=False)
    ims = {}
    for i, row in enumerate(rows):
        if kinds[row] == "delta":
            vmax = max(float(np.abs(m).max()) for m in panels[row].values()) or 1.0
            style = {"cmap": "RdBu_r", "norm": TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)}
        elif kinds[row] == "label":
            style = {"cmap": ListedColormap(LABEL_COLORS[:len(label_names)]), "vmin": 0, "vmax": len(label_names) - 1}
        else:
            style = {"cmap": "viridis", "vmin": 0.0, "vmax": 1.0}
        for j, col in enumerate(cols):
            ax = axes[i, j]
            if col not in panels[row]:
                ax.set_title(col, fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.text(0.5, 0.5, "not compared", ha="center", va="center", color="gray", fontsize=8,
                        transform=ax.transAxes)
                continue
            ims[row] = ax.imshow(panels[row][col], origin="lower", **style)
            for layer, head in zip(*np.nonzero((outline or {}).get((row, col), np.zeros((0, 0), dtype=bool)))):
                ax.add_patch(Rectangle((head - 0.5, layer - 0.5), 1, 1, fill=False, lw=1.2, ec="black"))
            ax.set_title(col, fontsize=9)
            ax.set_xticks(range(0, 12, 2))
            ax.set_yticks(range(0, 12, 2))
            ax.set_xlabel("Head", fontsize=7)
            if j == 0:
                ax.set_ylabel("Layer", fontsize=7)
            ax.tick_params(labelsize=6)
    fig.subplots_adjust(left=0.08, right=0.88, top=0.9, bottom=0.07, hspace=0.42, wspace=0.22)
    right = axes[0, -1].get_position().x1 + 0.012
    for i, row in enumerate(rows):
        box = axes[i, 0].get_position()
        fig.text(0.012, (box.y0 + box.y1) / 2, row, rotation=90, va="center", ha="left", fontsize=9, fontweight="bold")
        box = axes[i, -1].get_position()
        if kinds[row] == "label":
            fig.legend(handles=[Patch(color=c, label=n) for c, n in zip(LABEL_COLORS, label_names)], loc="center left",
                       bbox_to_anchor=(right - 0.004, (box.y0 + box.y1) / 2), frameon=False, fontsize=7)
        else:
            fig.colorbar(ims[row], cax=fig.add_axes((right, box.y0, 0.012, box.height)), label=colorbar_labels[row])
    fig.suptitle(suptitle, fontsize=11)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_transitions(mats: dict[str, np.ndarray], labels: Sequence[str], suptitle: str, path: Path) -> None:
    """mats[title] = int [n_labels, n_labels] switching heads by (source row, destination column), annotated. Labels
    that occur in no matrix are dropped; all labels are kept when every matrix is empty."""
    used = np.zeros(len(labels), dtype=bool)
    for t in mats.values():
        used |= (t.sum(axis=0) > 0) | (t.sum(axis=1) > 0)
    keep = np.flatnonzero(used) if used.any() else np.arange(len(labels))
    names = [labels[i] for i in keep]
    n = len(mats)
    ncol = min(n, 3)
    nrow = -(-n // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol + 0.4, 3.4 * nrow + 0.7), squeeze=False)
    vmax = max(1, max(int(t.max()) for t in mats.values()))
    for ax, (title, t) in zip(axes.ravel(), mats.items()):
        sub = t[np.ix_(keep, keep)]
        ax.imshow(sub, cmap="Blues", vmin=0, vmax=vmax)
        for (r, c), v in np.ndenumerate(sub):
            if v:
                ax.text(c, r, str(v), ha="center", va="center", fontsize=8, color="white" if v > vmax / 2 else "black")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=50, ha="right", fontsize=7)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel("to", fontsize=8)
        ax.set_ylabel("from", fontsize=8)
        ax.set_title(f"{title}: {int(t.sum())} heads", fontsize=9)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
