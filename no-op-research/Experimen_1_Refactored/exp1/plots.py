"""Layer x head heatmaps for the per-head aggregate table (see exp1.aggregate).

Two chart families: a categorical heatmap of mode_label, and a sequential heatmap
of any numeric column (mean_sink_mass, mean_entropy, consistency, ...). Both share
axis conventions (layer on y with layer 0 at the bottom, head on x, integer ticks)
and a private per-axes drawing helper so Phase 2 can place several of these on one
figure as side-by-side panels.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # write figures to files, never open a window

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize, TwoSlopeNorm
from matplotlib.patches import Patch, Rectangle

# Fixed label -> hex color, validated with the dataviz skill's categorical checks
# (node scripts/validate_palette.js ... --pairs all): all 10 pairs clear the CVD
# and normal-vision floors, since any two labels can sit adjacent in the grid (like
# a choropleth, not a linear sequence). Order follows the palette's own slot order
# (1, 3, 4, 5, 7) applied to LABELS in its documented order, so the mapping is
# reproducible rather than eyeballed. Fixed at module scope so a label keeps its
# color across every call, including Phase 2's three-condition panels.
_LABEL_COLORS: dict[str, str] = {
    "Diffuse": "#2a78d6",  # blue
    "Sink": "#1baf7a",  # aqua
    "Previous": "#eda100",  # yellow
    "Self": "#e87ba4",  # magenta
    "Other": "#4a3aa7",  # violet
}

_SEQUENTIAL_CMAP = "viridis"  # perceptually uniform, colorblind-safe single ramp

# blue<->red diverging pair with a neutral gray midpoint (dataviz skill palette.md);
# negative -> red, positive -> blue, symmetric around 0 via TwoSlopeNorm at call time.
_DIVERGING_CMAP = LinearSegmentedColormap.from_list("diverging_blue_red", ["#e34948", "#f0efec", "#2a78d6"])

# Unsigned-magnitude ramp for plot_shift_heatmaps: one hue light->dark (dataviz skill palette.md
# sequential blue, steps 250 / 450 / 700), starting from the diverging map's neutral midpoint so
# "no change" looks the same as in plot_diff_heatmaps and recedes toward the surface. Not
# viridis: there the near-zero cells are the darkest, which pulls the eye away from big changes.
_MAGNITUDE_CMAP = LinearSegmentedColormap.from_list("magnitude_blue", ["#f0efec", "#86b6ef", "#2a78d6", "#0d366b"])

# Condition palette for plot_label_bars: a different encoding dimension (which run,
# not which mode_label), so drawn from the categorical palette's remaining slots to
# stay distinct in role from _LABEL_COLORS. The one 3-way subset that fully clears
# --pairs all with the widest margin (validate_palette.js: normal-vision floor 33.1,
# CVD 7.2 - WARN band, mitigated by the legend + axis category labels + bar-value
# annotations already on this chart) is (green, violet, red); violet overlaps one
# label color ("Other") but the two charts are never shown as one legend. Assigned
# by position in the caller's dict order, not by condition name. Extend this list if
# a caller ever passes more than 3 conditions.
_CONDITION_COLORS: tuple[str, ...] = ("#008300", "#4a3aa7", "#e34948")  # green, violet, red


def heads_to_grid(heads: pd.DataFrame, column: str) -> np.ndarray:
    """Pivot heads to a (n_layers, n_heads) array, row i = sorted layer i, col j = sorted head j.

    Raises ValueError if any (layer, head) pair is missing or appears more than once.
    """
    dup_mask = heads.duplicated(subset=["layer", "head"], keep=False)
    if dup_mask.any():
        dupes = heads.loc[dup_mask, ["layer", "head"]].drop_duplicates().to_records(index=False)
        raise ValueError(f"duplicate (layer, head) rows: {list(dupes)}")

    layers = np.sort(heads["layer"].unique())
    head_ids = np.sort(heads["head"].unique())
    grid = heads.pivot(index="layer", columns="head", values=column).reindex(index=layers, columns=head_ids)

    missing = np.argwhere(grid.isna().to_numpy())
    if missing.size:
        cells = [(int(layers[i]), int(head_ids[j])) for i, j in missing]
        raise ValueError(f"missing (layer, head) cells: {cells}")

    return grid.to_numpy()


def _grid_axes(heads: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """The sorted layer and head id values that index a heads_to_grid array."""
    return np.sort(heads["layer"].unique()), np.sort(heads["head"].unique())


def _relative_luminance(rgb: tuple[float, float, float]) -> float:
    """WCAG relative luminance (sRGB, gamma-expanded), used to pick legible text ink."""

    def lin(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (lin(c) for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _readable_text_color(rgb: tuple[float, float, float]) -> str:
    """Black or white, whichever has higher WCAG contrast against this background."""
    return "black" if _relative_luminance(rgb) > 0.179 else "white"


def _set_layer_head_axes(ax: Axes, layers: np.ndarray, head_ids: np.ndarray) -> None:
    """Integer layer/head ticks; imshow(origin="lower") already puts layer 0 at the bottom."""
    ax.set_xticks(np.arange(len(head_ids)))
    ax.set_xticklabels([str(int(h)) for h in head_ids])
    ax.set_yticks(np.arange(len(layers)))
    ax.set_yticklabels([str(int(l)) for l in layers])
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")


def _draw_mode_label(
    ax: Axes,
    label_grid: np.ndarray,
    consistency_grid: np.ndarray,
    labels: Sequence[str],
    layers: np.ndarray,
    head_ids: np.ndarray,
) -> dict[str, str]:
    """Draw the categorical mode_label heatmap on ax. Returns the label -> color map used."""
    color_map = {label: _LABEL_COLORS[label] for label in labels}
    code_of = {label: i for i, label in enumerate(labels)}
    codes = np.vectorize(code_of.get)(label_grid).astype(float)

    cmap = ListedColormap([color_map[label] for label in labels])
    ax.imshow(codes, cmap=cmap, vmin=-0.5, vmax=len(labels) - 0.5, origin="lower", aspect="auto")
    _set_layer_head_axes(ax, layers, head_ids)

    for (li, hi), consistency in np.ndenumerate(consistency_grid):
        bg = mcolors.to_rgb(color_map[label_grid[li, hi]])
        text = f"{consistency:.2f}".lstrip("0") or "0"
        ax.text(hi, li, text, ha="center", va="center", fontsize=7, color=_readable_text_color(bg))

    return color_map


def plot_mode_label_heatmap(heads: pd.DataFrame, labels: Sequence[str], path: str | Path, title: str) -> None:
    """Categorical layer x head heatmap of mode_label, annotated with each head's consistency."""
    label_grid = heads_to_grid(heads, "mode_label")
    consistency_grid = heads_to_grid(heads, "consistency")
    layers, head_ids = _grid_axes(heads)

    fig, ax = plt.subplots(figsize=(0.55 * len(head_ids) + 2.6, 0.45 * len(layers) + 1.2))
    color_map = _draw_mode_label(ax, label_grid, consistency_grid, labels, layers, head_ids)
    ax.set_title(title)

    handles = [Patch(facecolor=color_map[label], label=label) for label in labels]
    ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, title="Mode")

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_mode_label_panels(
    heads_by_condition: dict[str, pd.DataFrame],
    labels: Sequence[str],
    path: str | Path,
    title: str,
) -> None:
    """One mode_label panel per condition, side by side, sharing one legend and colors."""
    conditions = list(heads_by_condition)
    n = len(conditions)

    grids = {c: heads_to_grid(heads_by_condition[c], "mode_label") for c in conditions}
    consistencies = {c: heads_to_grid(heads_by_condition[c], "consistency") for c in conditions}
    axes_by_cond = {c: _grid_axes(heads_by_condition[c]) for c in conditions}
    max_heads = max(len(head_ids) for _, head_ids in axes_by_cond.values())

    fig, axes = plt.subplots(1, n, figsize=(n * (0.55 * max_heads + 1.6) + 1.2, 0.45 * 12 + 1.4), sharey=False)
    axes = np.atleast_1d(axes)

    color_map = {label: _LABEL_COLORS[label] for label in labels}
    for i, (ax, cond) in enumerate(zip(axes, conditions)):
        layers, head_ids = axes_by_cond[cond]
        _draw_mode_label(ax, grids[cond], consistencies[cond], labels, layers, head_ids)
        ax.set_title(cond)
        if i > 0:
            ax.set_ylabel("")  # y axis already labelled on the leftmost panel

    handles = [Patch(facecolor=color_map[label], label=label) for label in labels]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, title="Mode")
    fig.suptitle(title)

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_diff_heatmaps(
    diffs: dict[str, np.ndarray],
    significant: dict[str, np.ndarray],
    path: str | Path,
    title: str,
    cbar_label: str,
    sig_note: str = "",
) -> None:
    """Side-by-side diverging diff heatmaps, one shared symmetric scale and colorbar.

    Significant cells (per `significant`) get a bold black outline rather than a
    recolored fill or a text-color change, so the mark stays legible regardless of
    where the diff sits on the diverging scale.
    """
    comparisons = list(diffs)
    n_layers, n_heads = diffs[comparisons[0]].shape
    layers = np.arange(n_layers)
    head_ids = np.arange(n_heads)

    vmax = max(float(np.nanmax(np.abs(diffs[c]))) for c in comparisons)
    vmax = vmax if vmax > 1e-12 else 1.0
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)

    fig, axes = plt.subplots(
        1, len(comparisons), figsize=(len(comparisons) * (0.55 * n_heads + 1.4) + 1.0, 0.45 * n_layers + 1.6), sharey=False
    )
    axes = np.atleast_1d(axes)

    im = None
    for ax, cmp_name in zip(axes, comparisons):
        grid = diffs[cmp_name]
        sig = significant[cmp_name]
        im = ax.imshow(grid, cmap=_DIVERGING_CMAP, norm=norm, origin="lower", aspect="auto")
        _set_layer_head_axes(ax, layers, head_ids)
        ax.set_title(cmp_name)

        for (li, hi), value in np.ndenumerate(grid):
            bg = _DIVERGING_CMAP(norm(value))[:3]
            ax.text(hi, li, f"{value:.2f}", ha="center", va="center", fontsize=7, color=_readable_text_color(bg))
            if sig[li, hi]:
                ax.add_patch(
                    Rectangle((hi - 0.5, li - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=1.8, zorder=3)
                )

    fig.colorbar(im, ax=axes.tolist(), label=cbar_label, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    note = "outlined = significant" if not sig_note else sig_note
    fig.text(0.5, 0.0, note, ha="center", va="top", fontsize=8)

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_shift_heatmaps(
    values: dict[str, np.ndarray],
    highlight: dict[str, np.ndarray],
    annotations: dict[str, np.ndarray],
    path: str | Path,
    title: str,
    cbar_label: str,
    note: str = "",
) -> None:
    """Side-by-side sequential heatmaps of an unsigned per-head change, one shared scale from 0.

    Highlighted cells get the same bold outline as plot_diff_heatmaps plus a second text line
    from `annotations` (a string per cell), so the biggest changes say what changed in place.
    """
    comparisons = list(values)
    n_layers, n_heads = values[comparisons[0]].shape
    layers = np.arange(n_layers)
    head_ids = np.arange(n_heads)

    vmax = max(float(np.nanmax(values[c])) for c in comparisons)
    norm = Normalize(vmin=0.0, vmax=vmax if vmax > 1e-12 else 1.0)

    # Wider, taller cells than plot_diff_heatmaps: highlighted cells carry a second text line.
    fig, axes = plt.subplots(
        1, len(comparisons), figsize=(len(comparisons) * (0.7 * n_heads + 1.4) + 1.0, 0.5 * n_layers + 1.6), sharey=False
    )
    axes = np.atleast_1d(axes)

    im = None
    for ax, cmp_name in zip(axes, comparisons):
        grid = values[cmp_name]
        im = ax.imshow(grid, cmap=_MAGNITUDE_CMAP, norm=norm, origin="lower", aspect="auto")
        _set_layer_head_axes(ax, layers, head_ids)
        ax.set_title(cmp_name)

        for (li, hi), value in np.ndenumerate(grid):
            ink = _readable_text_color(_MAGNITUDE_CMAP(norm(value))[:3])
            if highlight[cmp_name][li, hi]:
                ax.text(hi, li, f"{value:.2f}\n{annotations[cmp_name][li, hi]}", ha="center", va="center",
                        fontsize=6.5, color=ink, linespacing=1.3)
                ax.add_patch(
                    Rectangle((hi - 0.5, li - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=1.8, zorder=3)
                )
            else:
                ax.text(hi, li, f"{value:.2f}", ha="center", va="center", fontsize=7, color=ink)

    fig.colorbar(im, ax=axes.tolist(), label=cbar_label, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    if note:
        fig.text(0.5, 0.0, note, ha="center", va="top", fontsize=8)

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_label_bars(
    pct: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    labels: Sequence[str],
    path: str | Path,
    title: str,
) -> None:
    """Grouped bar chart, one group per label, one bar per condition, with 95% CI whiskers."""
    conditions = list(pct)
    n_cond = len(conditions)
    n_labels = len(labels)
    x = np.arange(n_labels)
    width = 0.8 / n_cond

    fig, ax = plt.subplots(figsize=(max(6.0, 1.3 * n_labels + 1.5), 4.5))

    max_hi = max(float(np.nanmax(pct[c][2])) for c in conditions)
    for i, cond in enumerate(conditions):
        point, lo, hi = (np.asarray(a, dtype=float) for a in pct[cond])
        yerr = np.vstack([point - lo, hi - point])
        offset = (i - (n_cond - 1) / 2) * width
        color = _CONDITION_COLORS[i % len(_CONDITION_COLORS)]
        bars = ax.bar(
            x + offset,
            point,
            width=width * 0.9,
            color=color,
            label=cond,
            yerr=yerr,
            capsize=3,
            error_kw={"linewidth": 1, "ecolor": "black"},
        )
        for rect, value, top in zip(bars, point, hi):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                top + max_hi * 0.02,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("% of heads (mode label)")
    ax.set_title(title)
    ax.legend(frameon=False, title="Condition")

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _draw_scalar(
    ax: Axes,
    grid: np.ndarray,
    layers: np.ndarray,
    head_ids: np.ndarray,
    vmin: float | None,
    vmax: float | None,
):
    """Draw a sequential scalar heatmap on ax. Returns the imshow mappable (for a colorbar)."""
    norm = Normalize(vmin=np.nanmin(grid) if vmin is None else vmin, vmax=np.nanmax(grid) if vmax is None else vmax)
    cmap = plt.get_cmap(_SEQUENTIAL_CMAP)
    im = ax.imshow(grid, cmap=cmap, norm=norm, origin="lower", aspect="auto")
    _set_layer_head_axes(ax, layers, head_ids)

    for (li, hi), value in np.ndenumerate(grid):
        bg = cmap(norm(value))[:3]
        ax.text(hi, li, f"{value:.2f}", ha="center", va="center", fontsize=7, color=_readable_text_color(bg))

    return im


def plot_scalar_heatmap(
    heads: pd.DataFrame,
    column: str,
    path: str | Path,
    title: str,
    cbar_label: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Sequential layer x head heatmap of one numeric column, annotated to 2 decimals."""
    grid = heads_to_grid(heads, column)
    layers, head_ids = _grid_axes(heads)

    fig, ax = plt.subplots(figsize=(0.55 * len(head_ids) + 2.2, 0.45 * len(layers) + 1.2))
    im = _draw_scalar(ax, grid, layers, head_ids, vmin, vmax)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=cbar_label)

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
