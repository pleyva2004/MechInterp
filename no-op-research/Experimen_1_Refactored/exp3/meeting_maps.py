"""Compact Experiment 1-style attention strips for the meeting one-pager (saved arrays only, no model run).

Each strip: three "what each head attends to most" maps (reference + two conditions), then the change in BOS
attention for the two conditions against their reference.

Run:  uv run --offline python -m exp3.meeting_maps
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write figures to files, never open a window
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from matplotlib.patches import Patch

from exp3.attention_maps3 import AXIS_GREY, COLORS, RESULTS, TEXT_GREY, column, engine_means, most_attended, style_head_grid

OUT = Path(__file__).resolve().parents[1] / "onepager"


def strip(cols: list[dict], path: Path) -> None:
    """cols: reference first, then two conditions that carry a 'ref' for the change maps."""
    cats = [most_attended(c["attn"], c["cat"]) for c in cols]
    present = [k for k in COLORS if any((c == k).any() for c in cats)]
    code = {k: i for i, k in enumerate(present)}
    cmap = ListedColormap([COLORS[k] for k in present])
    diffs = [c["attn"][..., 0] - c["ref"][..., 0] for c in cols[1:]]
    vmax = max(float(np.abs(d).max()) for d in diffs)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    fig = plt.figure(figsize=(7.6, 2.2))
    gs = fig.add_gridspec(1, 7, width_ratios=[1, 1, 1, 0.16, 1, 1, 0.05], left=0.035, right=0.955, top=0.8,
                          bottom=0.2, wspace=0.26)
    axes = [fig.add_subplot(gs[0, k]) for k in (0, 1, 2, 4, 5)]
    for ax, c, ca in zip(axes[:3], cols, cats):
        ax.imshow(np.vectorize(code.get)(ca), origin="lower", cmap=cmap, vmin=-0.5, vmax=len(present) - 0.5)
        ax.set_title(c["title"], fontsize=6.5, pad=7)
    im = None
    for ax, c, d in zip(axes[3:], cols[1:], diffs):
        im = ax.imshow(d, origin="lower", cmap="RdBu_r", norm=norm)
        ax.set_title(c["title"], fontsize=6.5, pad=7)
    for k, ax in enumerate(axes):  # every panel numbered, so a head's layer and index can be read off directly
        style_head_grid(ax, xsize=4.0, ysize=4.6, ylabel=k in (0, 3))
    fig.text(0.035, 0.925, "What each head attends to most", fontsize=7, fontweight="bold")
    fig.text(axes[3].get_position().x0, 0.925, "Change in BOS attention (red = more resting)", fontsize=7,
             fontweight="bold")
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[0, 6]))
    cb.ax.tick_params(labelsize=5, colors=AXIS_GREY, labelcolor=TEXT_GREY)
    cb.outline.set_edgecolor(AXIS_GREY)
    fig.legend(handles=[Patch(facecolor=COLORS[k], edgecolor="0.6", label=k) for k in present], loc="lower left",
               bbox_to_anchor=(0.04, 0.0), ncol=len(present), frameon=False, fontsize=6, handlelength=1.0,
               columnspacing=1.0)
    fig.savefig(path, dpi=250)
    plt.close(fig)
    print(f"saved {path}")


def main() -> None:
    m = engine_means("20261015-223621_exp3e32", "common")
    text = lambda k: {p: ("Previous token" if p == 31 else "Phrase / text") for p in range(32 - k, 32)}
    strip([column("random tokens (reference)", m["none"]),
           column("8 natural tokens", m["nl_8"], m["none"], text(8)),
           column("31 natural tokens", m["nl_31"], m["none"], text(31))], OUT / "meeting_map_1.png")

    def placement(i: int, j: int) -> np.ndarray:
        return np.load(RESULTS / "20261015-210238_exp3pos" / "placements" / f"common_{i}_{j}.npz")["by_key"]

    near, far = placement(31, 32), placement(1, 32)
    strip([column("wrong capital (reference)", near[1:3].mean(axis=0), keys={31: "Country", 32: "Capital / city"}),
           column("correct, country adjacent", near[0], near[1:3].mean(axis=0), {31: "Country", 32: "Capital / city"}),
           column("correct, country at pos. 1", far[0], far[1:3].mean(axis=0), {1: "Country", 32: "Capital / city"})],
          OUT / "meeting_map_2.png")

    z = np.load(RESULTS / "20261015-190957_exp3rel" / "attention_by_position.npz")
    keys = {30: "Country", 31: "Capital / city"}
    strip([column("no pair (reference)", z["common/none"]),
           column("correct pair", z["common/capital/correct"], z["common/none"], keys),
           column("wrong pair", z["common/capital/wrong"], z["common/none"], keys)], OUT / "meeting_map_3.png")


if __name__ == "__main__":
    main()
