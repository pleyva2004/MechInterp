"""Condition-level figures for the exp1ext 2x2 factorial and dose-response.

Three small-multiple/decomposition figures per head-level metric (mean sink_mass,
entropy, prev_mass, ...): the AA/AC/CA/CC interaction grid, the total/query/context/
interaction effect decomposition, and the k-token dose-response curve. Heads are
grouped by their group label ("sink rises" / "sink falls" / "control", or any other
caller-supplied label) throughout, and reuses exp1.plots' text-contrast helper and
condition palette so these figures read as part of the same family as exp1's.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # write figures to files, never open a window

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from exp1.plots import _CONDITION_COLORS, _readable_text_color

# ---- palettes ----------------------------------------------------------------
# Group tint: dataviz skill categorical slots 1-3 (blue/orange/aqua) -- the only
# 3-slot subset documented in palette.md as passing --pairs all in both modes
# (worst pair CVD dE 9.2 light, normal-vision dE 24.0), required here because small
# multiples let any two panels sit side by side, not just neighbors:
#   node scripts/validate_palette.js "#2a78d6,#eb6834,#1baf7a" --mode light --pairs all
#   -> ALL CHECKS PASS (aqua WARNs sub-3:1 contrast vs the white surface; mitigated
#      by always pairing the tint with the legible head-name text it sits behind,
#      via _readable_text_color, plus the group legend naming each color).
_GROUP_COLORS: dict[str, str] = {
    "sink rises": "#2a78d6",  # blue
    "sink falls": "#eb6834",  # orange
    "control": "#1baf7a",     # aqua
}
_FALLBACK_GROUP_COLOR = "#898781"  # muted ink; unknown group names, not a 4th categorical slot

# Context-line pair for plot_interaction_grid: two more slots (green/violet), chosen
# disjoint from _GROUP_COLORS so a panel's group tint and its two lines never share a
# hue within the same panel.
#   node scripts/validate_palette.js "#008300,#4a3aa7" --mode light
#   -> ALL CHECKS PASS (worst adjacent CVD dE 27.9, normal-vision dE 34.1)
_CONTEXT_LINE_COLORS: dict[str, str] = {"rare": "#008300", "common": "#4a3aa7"}

# Decomposition components reuse exp1.plots' validated _CONDITION_COLORS (green,
# violet, red) so effect-type colors stay consistent with exp1's own condition bars.
_COMPONENT_COLORS: dict[str, str] = {
    "query": _CONDITION_COLORS[0],
    "context": _CONDITION_COLORS[1],
    "interaction": _CONDITION_COLORS[2],
}
_TOTAL_COLOR = "black"

_MAX_COLS = 6


def _group_color(group: str) -> str:
    return _GROUP_COLORS.get(group, _FALLBACK_GROUP_COLOR)


def _order_by_group(groups: Sequence[str]) -> np.ndarray:
    """Panel order: group blocks in first-seen order, original relative order within a group."""
    first_seen: dict[str, int] = {}
    for g in groups:
        first_seen.setdefault(g, len(first_seen))
    return np.argsort([first_seen[g] for g in groups], kind="stable")


def _grid_dims(n: int) -> tuple[int, int]:
    ncols = min(_MAX_COLS, n)
    nrows = math.ceil(n / ncols)
    return nrows, ncols


def _shared_range(lo_arrays: Sequence[np.ndarray], hi_arrays: Sequence[np.ndarray]) -> tuple[float, float]:
    y_lo = float(min(np.nanmin(a) for a in lo_arrays))
    y_hi = float(max(np.nanmax(a) for a in hi_arrays))
    pad = 0.05 * (y_hi - y_lo) if y_hi > y_lo else 0.05
    return y_lo - pad, y_hi + pad


def _titled_panel(ax: plt.Axes, head: str, group: str) -> None:
    """Head-name title as a chip tinted by group; ink picked for contrast against the chip."""
    color = _group_color(group)
    ax.set_title(
        head, fontsize=8.5, fontweight="bold", pad=6,
        color=_readable_text_color(mcolors.to_rgb(color)),
        bbox=dict(facecolor=color, edgecolor="none", boxstyle="round,pad=0.28"),
    )


def _group_legend_handles(groups: Sequence[str]) -> list[Patch]:
    return [Patch(facecolor=_group_color(g), label=g) for g in dict.fromkeys(groups)]


def _hide_unused_axes(axes: np.ndarray, n_used: int, nrows: int, ncols: int) -> None:
    for panel_i in range(n_used, nrows * ncols):
        axes[panel_i // ncols][panel_i % ncols].axis("off")


def _declutter_grid(axes: np.ndarray, n_used: int, nrows: int, ncols: int) -> None:
    """Keep x tick labels only on each column's bottom-most used panel, y tick labels only
    in column 0 -- avoids redundant labels and frees vertical room for the title chips
    (a plain sharex/sharey would mislabel a column whose last row is a hidden trailing panel).
    """
    last_row_of_col: dict[int, int] = {}
    for panel_i in range(n_used):
        r, c = panel_i // ncols, panel_i % ncols
        last_row_of_col[c] = max(last_row_of_col.get(c, r), r)
    for panel_i in range(n_used):
        r, c = panel_i // ncols, panel_i % ncols
        ax = axes[r][c]
        if r != last_row_of_col[c]:
            ax.tick_params(labelbottom=False)
        if c != 0:
            ax.tick_params(labelleft=False)


def plot_interaction_grid(
    cells: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    heads: Sequence[str],
    groups: Sequence[str],
    path: str | Path,
    title: str,
    ylabel: str,
) -> None:
    """Small multiples, one panel per head, of the 2x2 AA/AC/CA/CC cell means.

    Each panel plots the metric against final-token class (rare, common) as two lines:
    context rare (AA->AC) and context common (CA->CC). Parallel lines mean additive
    query/context effects; a bowed pair means interaction. Panels share one y range so
    effect sizes are comparable across heads, and panels are grouped by `groups`.
    """
    required = {"AA", "AC", "CA", "CC"}
    if set(cells) != required:
        raise ValueError(f"cells must have exactly keys {required}, got {set(cells)}")
    n = len(heads)
    if n == 0:
        raise ValueError("heads must be non-empty")
    if len(groups) != n:
        raise ValueError(f"groups must have length {n} (len(heads)), got {len(groups)}")
    for key, triplet in cells.items():
        if len(triplet) != 3:
            raise ValueError(f"cells[{key!r}] must be a (mean, lo, hi) triplet, got {len(triplet)} arrays")
        for arr in triplet:
            if len(arr) != n:
                raise ValueError(f"cells[{key!r}] arrays must have length {n} (len(heads)), got {len(arr)}")

    mean = {k: np.asarray(v[0], dtype=float) for k, v in cells.items()}
    lo = {k: np.asarray(v[1], dtype=float) for k, v in cells.items()}
    hi = {k: np.asarray(v[2], dtype=float) for k, v in cells.items()}

    order = _order_by_group(groups)
    nrows, ncols = _grid_dims(n)
    y_range = _shared_range(list(lo.values()), list(hi.values()))

    fig, axes = plt.subplots(nrows, ncols, figsize=(2.15 * ncols + 0.6, 2.15 * nrows + 1.3), squeeze=False, sharey=True)
    fig.subplots_adjust(hspace=0.65, wspace=0.3)
    x = np.array([0, 1])

    for panel_i, head_i in enumerate(order):
        ax = axes[panel_i // ncols][panel_i % ncols]
        for context_label, (lo_key, hi_key) in (("rare", ("AA", "AC")), ("common", ("CA", "CC"))):
            y = np.array([mean[lo_key][head_i], mean[hi_key][head_i]])
            y_bot = np.array([lo[lo_key][head_i], lo[hi_key][head_i]])
            y_top = np.array([hi[lo_key][head_i], hi[hi_key][head_i]])
            ax.errorbar(
                x, y, yerr=np.vstack([y - y_bot, y_top - y]),
                marker="o", markersize=5, linewidth=1.8, capsize=3,
                color=_CONTEXT_LINE_COLORS[context_label],
            )
        ax.set_xticks(x)
        ax.set_xticklabels(["rare", "common"], fontsize=8)
        ax.set_xlim(-0.4, 1.4)
        ax.set_ylim(*y_range)
        ax.tick_params(axis="y", labelsize=7)
        _titled_panel(ax, heads[head_i], groups[head_i])

    _declutter_grid(axes, n, nrows, ncols)
    _hide_unused_axes(axes, n, nrows, ncols)
    for row in axes:
        row[0].set_ylabel(ylabel, fontsize=8)
    fig.text(0.5, 0.0 if nrows == 1 else 0.02, "final-token class", ha="center", va="top", fontsize=8)

    context_handles = [
        Line2D([0], [0], color=c, marker="o", linewidth=1.8, label=f"context: {label}")
        for label, c in _CONTEXT_LINE_COLORS.items()
    ]
    context_legend = fig.legend(
        handles=context_handles, loc="upper right", bbox_to_anchor=(1.0, 1.0),
        frameon=False, fontsize=8, title="Line",
    )
    fig.add_artist(context_legend)
    fig.legend(
        handles=_group_legend_handles(groups), loc="upper left", bbox_to_anchor=(0.0, 1.0),
        frameon=False, fontsize=8, title="Group",
    )
    fig.suptitle(title)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_decomposition(
    effects: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    heads: Sequence[str],
    groups: Sequence[str],
    path: str | Path,
    title: str,
    ylabel: str,
) -> None:
    """Horizontal grouped bars (query / context / interaction) plus a total marker,
    one row per head, sorted into group blocks with separators and block labels.
    """
    required = {"total", "query", "context", "interaction"}
    if set(effects) != required:
        raise ValueError(f"effects must have exactly keys {required}, got {set(effects)}")
    n = len(heads)
    if n == 0:
        raise ValueError("heads must be non-empty")
    if len(groups) != n:
        raise ValueError(f"groups must have length {n} (len(heads)), got {len(groups)}")
    for key, triplet in effects.items():
        if len(triplet) != 3:
            raise ValueError(f"effects[{key!r}] must be an (est, lo, hi) triplet, got {len(triplet)} arrays")
        for arr in triplet:
            if len(arr) != n:
                raise ValueError(f"effects[{key!r}] arrays must have length {n} (len(heads)), got {len(arr)}")

    est = {k: np.asarray(v[0], dtype=float) for k, v in effects.items()}
    lo = {k: np.asarray(v[1], dtype=float) for k, v in effects.items()}
    hi = {k: np.asarray(v[2], dtype=float) for k, v in effects.items()}

    order = _order_by_group(groups)
    components = ("query", "context", "interaction")
    bar_h = 0.8 / len(components)
    y_positions = np.arange(n)

    fig, ax = plt.subplots(figsize=(7.8, max(3.0, 0.42 * n + 1.4)))

    for c_i, comp in enumerate(components):
        offset = (c_i - (len(components) - 1) / 2) * bar_h
        vals, lo_v, hi_v = est[comp][order], lo[comp][order], hi[comp][order]
        ax.barh(
            y_positions + offset, vals, height=bar_h * 0.9, color=_COMPONENT_COLORS[comp],
            xerr=np.vstack([vals - lo_v, hi_v - vals]), capsize=2,
            error_kw={"linewidth": 1, "ecolor": "black"}, label=comp,
        )

    total_vals, total_lo, total_hi = est["total"][order], lo["total"][order], hi["total"][order]
    ax.errorbar(
        total_vals, y_positions, xerr=np.vstack([total_vals - total_lo, total_hi - total_vals]),
        fmt="D", color=_TOTAL_COLOR, markersize=6, capsize=2, linewidth=1.2, label="total",
    )

    ax.axvline(0.0, color="#898781", linewidth=1.0, zorder=0)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([heads[i] for i in order], fontsize=8)
    ax.invert_yaxis()  # first (group-ordered) head at the top
    ax.set_xlabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="lower right", frameon=False, fontsize=8)

    # group separators + labels: neutral ink, never the group's own color as text.
    ordered_groups = [groups[i] for i in order]
    for b in range(1, n):
        if ordered_groups[b] != ordered_groups[b - 1]:
            ax.axhline(b - 0.5, color="#c3c2b7", linewidth=0.8, zorder=0)

    block_start = 0
    for i in range(1, n + 1):
        if i == n or ordered_groups[i] != ordered_groups[block_start]:
            mid = (block_start + i - 1) / 2
            ax.text(
                1.01, mid, ordered_groups[block_start], transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=8, color="#52514e",
            )
            block_start = i

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_dose_response(
    ks: np.ndarray,
    curves: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    groups: Sequence[str],
    path: str | Path,
    title: str,
    ylabel: str,
) -> None:
    """Small multiples of mean +/- CI band vs k, one panel per head in `curves` order.

    x uses evenly spaced categorical positions labelled with the actual k values: ks
    (0, 1, 2, 4, 8, 16, 31, ...) are not evenly spaced and k=0 breaks a log scale, so a
    categorical axis keeps every k legible regardless of its numeric spacing.
    """
    ks = np.asarray(ks)
    n_k = len(ks)
    if n_k == 0:
        raise ValueError("ks must be non-empty")
    heads = list(curves)
    n = len(heads)
    if n == 0:
        raise ValueError("curves must be non-empty")
    if len(groups) != n:
        raise ValueError(f"groups must have length {n} (len(curves)), got {len(groups)}")
    for head, triplet in curves.items():
        if len(triplet) != 3:
            raise ValueError(f"curves[{head!r}] must be a (mean, lo, hi) triplet, got {len(triplet)} arrays")
        for arr in triplet:
            if len(arr) != n_k:
                raise ValueError(f"curves[{head!r}] arrays must have length {n_k} (len(ks)), got {len(arr)}")

    order = _order_by_group(groups)
    nrows, ncols = _grid_dims(n)
    lo_arrays = [np.asarray(curves[h][1], dtype=float) for h in heads]
    hi_arrays = [np.asarray(curves[h][2], dtype=float) for h in heads]
    y_range = _shared_range(lo_arrays, hi_arrays)
    x = np.arange(n_k)

    fig, axes = plt.subplots(nrows, ncols, figsize=(2.15 * ncols + 0.6, 2.15 * nrows + 1.3), squeeze=False, sharey=True)
    fig.subplots_adjust(hspace=0.65, wspace=0.3)

    for panel_i, head_i in enumerate(order):
        head = heads[head_i]
        ax = axes[panel_i // ncols][panel_i % ncols]
        color = _group_color(groups[head_i])
        mean, lo, hi = (np.asarray(a, dtype=float) for a in curves[head])

        ax.plot(x, mean, color=color, linewidth=1.8, marker="o", markersize=4)
        ax.fill_between(x, lo, hi, color=color, alpha=0.2, linewidth=0)
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(k)) for k in ks], fontsize=7)
        ax.set_ylim(*y_range)
        _titled_panel(ax, head, groups[head_i])

    _declutter_grid(axes, n, nrows, ncols)
    _hide_unused_axes(axes, n, nrows, ncols)
    for row in axes:
        row[0].set_ylabel(ylabel, fontsize=8)
    fig.text(0.5, 0.0 if nrows == 1 else 0.02, "k (common context tokens)", ha="center", va="top", fontsize=8)

    fig.legend(
        handles=_group_legend_handles(groups), loc="upper left", bbox_to_anchor=(0.0, 1.0),
        frameon=False, fontsize=8, title="Group",
    )
    fig.suptitle(title)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
