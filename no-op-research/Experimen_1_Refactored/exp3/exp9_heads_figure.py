"""Experiment 9 (R1-in, common filler): where each head's attention goes, wrong capital -> correct capital -> change.

Rows: attention on BOS, on the country (position 31) and on the capital itself (the final token, 32). Columns: wrong
capital, correct capital, correct - wrong. The heads that move most are boxed in every panel. Saved arrays only.

Run:  uv run --offline python -m exp3.exp9_heads_figure
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # write figures to files, never open a window
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

from exp3.attention_maps3 import AXIS_GREY, RESULTS, TEXT_GREY, style_head_grid
from exp3.meeting_maps import OUT

HEADS = ((5, 8), (5, 5), (4, 10), (10, 10))  # (layer, head), the largest BOS drops
ROWS = (("on BOS (resting)", 0), ("on the country (France, 31)", 31), ("on the capital itself (Paris, 32)", 32))


def main() -> None:
    z = np.load(RESULTS / "20261015-203150_exp3rel_in" / "attention_by_position.npz")
    wrong, correct = z["common/capital/wrong"], z["common/capital/correct"]
    diffs = [correct[..., k] - wrong[..., k] for _, k in ROWS]
    vmax = max(float(np.abs(d).max()) for d in diffs)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    fig, axes = plt.subplots(3, 3, figsize=(7.4, 7.0))
    fig.subplots_adjust(left=0.16, right=0.86, top=0.9, bottom=0.06, hspace=0.32, wspace=0.2)
    im_seq = im_div = None
    for r, ((label, k), d) in enumerate(zip(ROWS, diffs)):
        for c, (title, grid) in enumerate((("wrong capital", wrong[..., k]), ("correct capital", correct[..., k]),
                                           ("change: correct − wrong", d))):
            ax = axes[r, c]
            if c < 2:
                im_seq = ax.imshow(grid, origin="lower", cmap="viridis", vmin=0, vmax=1)
            else:
                im_div = ax.imshow(grid, origin="lower", cmap="RdBu_r", norm=norm)
            for layer, head in HEADS:
                ax.add_patch(Rectangle((head - 0.5, layer - 0.5), 1, 1, fill=False, lw=1.6,
                                       ec="white" if c < 2 else "black"))
            if r == 0:
                ax.set_title(title, fontsize=9, fontweight="bold", pad=9)
            style_head_grid(ax, xsize=5.5, ysize=6)
        box = axes[r, 0].get_position()
        fig.text(0.015, (box.y0 + box.y1) / 2, "Attention\n" + label, rotation=90, va="center", ha="left",
                 fontsize=8, fontweight="bold")
    label_at = {(5, 5): (0.3, 7.6), (5, 8): (6.2, 8.3), (4, 10): (5.6, 1.6), (10, 10): (4.8, 11.0)}  # text (x, y)
    for layer, head in HEADS:  # name the boxed heads once, on the change column of the BOS row
        axes[0, 2].annotate(f"L{layer}H{head}", (head, layer), xytext=label_at[(layer, head)], fontsize=6.5,
                            fontweight="bold", arrowprops=dict(arrowstyle="-", lw=0.6, color="black"))
    right = axes[0, 2].get_position().x1 + 0.015
    top, bottom = axes[0, 0].get_position(), axes[2, 0].get_position()
    for im, box, label in ((im_seq, top, "attention (wrong / correct)"), (im_div, bottom, "change (red = more, blue = less)")):
        cb = fig.colorbar(im, cax=fig.add_axes((right, box.y0, 0.015, box.height)))
        cb.set_label(label, color="black", fontsize=7)
        cb.ax.tick_params(labelsize=6, colors=AXIS_GREY, labelcolor=TEXT_GREY)
        cb.outline.set_edgecolor(AXIS_GREY)
    fig.suptitle("Experiment 9: where each head looks, wrong → correct capital (common filler, 200 rows)\n"
                 "boxed: L5H8, L5H5, L4H10, L10H10, the largest drops in resting", fontsize=9)
    path = OUT / "exp9_heads.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
