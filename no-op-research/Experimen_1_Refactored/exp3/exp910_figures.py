"""Figures for the Experiments 9-10 one-pager (saved arrays only, no model run).

exp9_strip.png: R1-in, common filler. What each head attends to most (wrong capital, correct capital, correct city)
and the change in BOS attention, correct - wrong, for capitals and for cities.
exp10_fig.png: position sweep, common filler, capital as the final token, country at positions 1, 16 and 31. What
each head attends to most with correct pairs, above the change in BOS attention (correct - wrong).

Run:  uv run --offline python -m exp3.exp910_figures
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # write figures to files, never open a window
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from matplotlib.patches import Patch

from exp3.attention_maps3 import AXIS_GREY, COLORS, RESULTS, TEXT_GREY, column, most_attended, style_head_grid
from exp3.meeting_maps import OUT, strip


def exp9() -> None:
    z = np.load(RESULTS / "20261015-203150_exp3rel_in" / "attention_by_position.npz")
    keys = {31: "Country", 32: "Capital / city"}
    strip([column("wrong capital (reference)", z["common/capital/wrong"], keys=keys),
           column("correct capital", z["common/capital/correct"], z["common/capital/wrong"], keys),
           column("correct city", z["common/city/correct"], z["common/city/wrong"], keys)], OUT / "exp9_strip.png")


CHIP = {  # token chips as in the one-pagers: BOS, random filler, country, capital (final token), elided gap
    "bos": dict(color="white", bbox=dict(boxstyle="square,pad=0.18", fc="#52514E", ec="#52514E", lw=0.4)),
    "fill": dict(color="#898781", bbox=dict(boxstyle="square,pad=0.18", fc="white", ec="#C3C2B7", lw=0.4)),
    "country": dict(color="black", bbox=dict(boxstyle="square,pad=0.18", fc="#DCEAFB", ec="#2A78D6", lw=0.5)),
    "capital": dict(color="black", fontweight="bold", bbox=dict(boxstyle="square,pad=0.18", fc="white", ec="#EB6834", lw=0.6)),
    "gap": dict(color="#898781"),
}


def chip_row(fig, x0: float, y: float, tokens: list[tuple[str, str]], size: float = 5.6) -> None:
    """Draw a token sequence as chips starting at figure coordinates (x0, y)."""
    renderer = fig.canvas.get_renderer()
    gap_px = 0.25 * size * fig.dpi / 72  # even space between chips
    x = x0
    for tok, kind in tokens:
        t = fig.text(x, y, tok, fontsize=size, family="monospace", va="center", ha="left", **CHIP[kind])
        t.draw(renderer)  # lays out the box so its true right edge can be measured
        box = t.get_bbox_patch()
        right = (box.get_window_extent(renderer) if box is not None else t.get_window_extent(renderer)).x1
        x = (right + gap_px) / fig.bbox.width


def sequences(i: int, country: str) -> list[tuple[str, str]]:
    """The first common-filler row with the country at position i and Paris as the final token (abbreviated)."""
    if i == 1:
        body = [(country, "country"), ("sc", "fill"), ("sa", "fill"), ("…", "gap"), ("dis", "fill")]
    elif i == 16:
        body = [("she", "fill"), ("…", "gap"), ("ors", "fill"), (country, "country"), ("ign", "fill"), ("…", "gap"),
                ("dis", "fill")]
    else:
        body = [("she", "fill"), ("sc", "fill"), ("…", "gap"), ("age", "fill"), (country, "country")]
    return [("BOS", "bos")] + body + [("Paris", "capital")]


def exp10() -> None:
    """Per country position (1, 16, 31): what each head attends to most with correct pairs (top) and the change in BOS
    attention, correct - wrong (bottom), in matching columns."""
    cases = []
    for i in (1, 16, 31):
        f = np.load(RESULTS / "20261015-210238_exp3pos" / "placements" / f"common_{i}_32.npz")["by_key"]
        cats = most_attended(column("", f[0], keys={i: "Country", 32: "Capital / city"})["attn"],
                             column("", f[0], keys={i: "Country", 32: "Capital / city"})["cat"])
        cases.append((i, cats, f[0, ..., 0] - f[1:3, ..., 0].mean(axis=0)))
    present = [k for k in COLORS if any((c == k).any() for _, c, _ in cases)]
    code = {k: n for n, k in enumerate(present)}
    cmap = ListedColormap([COLORS[k] for k in present])
    vmax = max(float(np.abs(d).max()) for _, _, d in cases)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    fig = plt.figure(figsize=(7.6, 5.2))
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 0.05, 0.3], left=0.07, right=0.97, top=0.8, bottom=0.07,
                          wspace=0.22, hspace=0.42)
    im = None
    for k, (i, cats, d) in enumerate(cases):
        title = f"country at {i}" + (" (adjacent)" if i == 31 else "")
        top = fig.add_subplot(gs[0, k])
        x0 = top.get_position().x0  # the sequences that produce this column, aligned above it
        chip_row(fig, x0, 0.965, sequences(i, "France"))
        chip_row(fig, x0, 0.925, sequences(i, "Japan"))
        top.imshow(np.vectorize(code.get)(cats), origin="lower", cmap=cmap, vmin=-0.5, vmax=len(present) - 0.5)
        top.set_title(title, fontsize=7, pad=7)
        style_head_grid(top, xsize=5.0, ysize=5.5, ylabel=k == 0)
        bottom = fig.add_subplot(gs[1, k])
        im = bottom.imshow(d, origin="lower", cmap="RdBu_r", norm=norm)
        bottom.set_title(title, fontsize=7, pad=7)
        style_head_grid(bottom, xsize=5.0, ysize=5.5, ylabel=k == 0)
    legend_ax = fig.add_subplot(gs[0, 3:])
    legend_ax.axis("off")
    legend_ax.legend(handles=[Patch(facecolor=COLORS[k], edgecolor="0.6", label=k) for k in present], loc="center left",
                     frameon=False, fontsize=6, handlelength=1.0)
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[1, 3]))
    cb.ax.tick_params(labelsize=5.5, colors=AXIS_GREY, labelcolor=TEXT_GREY)
    cb.outline.set_edgecolor(AXIS_GREY)
    fig.text(0.005, 0.965, "correct", fontsize=6.5, fontweight="bold", color="#52514E", va="center")
    fig.text(0.005, 0.925, "wrong", fontsize=6.5, fontweight="bold", color="#52514E", va="center")
    fig.text(0.07, 0.875, "What each head attends to most (correct pairs)", fontsize=7.5, fontweight="bold")
    fig.text(0.07, gs[1, 0].get_position(fig).y1 + 0.055, "Change in BOS attention, correct − wrong (blue = less resting)",
             fontsize=7.5, fontweight="bold")
    fig.savefig(OUT / "exp10_fig.png", dpi=250)
    plt.close(fig)
    print(f"saved {OUT / 'exp10_fig.png'}")


if __name__ == "__main__":
    exp9()
    exp10()
