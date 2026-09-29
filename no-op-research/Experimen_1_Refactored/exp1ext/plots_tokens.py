"""Token-level figures for the exp1ext token sweep: per-token attention overlays, effect
scatters/bars across the byte/common/rare id space, a token x head effect heatmap, and a
per-head "who gets attended to" bar chart.

Palette and ink-contrast conventions are shared with exp1.plots rather than re-derived: the
diverging colormap and the black/white text-ink rule come from there, and the three token
classes below reuse the categorical palette's first three slots, which are the ones validated
for all-pairs adjacency (dataviz skill, palette.md) -- required here because scatter points of
any two classes can end up side by side.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # write figures to files, never open a window

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch

from exp1.plots import _DIVERGING_CMAP, _readable_text_color
from exp1.show_sequences import plot_sequences

# byte / common / rare, in the same order the config's ranges walk the id space; colors are
# categorical palette slots 1-3 (blue, orange, aqua), the subset that passes --pairs all.
_CLASS_ORDER: tuple[str, ...] = ("byte", "common", "rare")
_CLASS_COLORS: dict[str, str] = {"byte": "#2a78d6", "common": "#eb6834", "rare": "#1baf7a"}
_BAND_TITLES: dict[str, str] = {"byte": "byte 0\u2013255", "common": "common 256\u2013999", "rare": "rare sample"}


def _classify_id(token_id: int) -> str:
    """byte (0-255), common (256-999), or rare (>=1000) -- the three bands exp1ext sweeps."""
    if token_id < 256:
        return "byte"
    if token_id < 1000:
        return "common"
    return "rare"


def _display_text(text: str) -> str:
    """Repr-safe label for a decoded token: blank/whitespace-only text becomes a visible
    escape (the same fallback idea as exp1.show_sequences._display_piece's byte-symbol case),
    and '$' is escaped so matplotlib never switches into mathtext."""
    if text == "":
        display = "\u2205"  # empty set: nothing decoded
    elif text.isspace():
        display = text.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r").replace(" ", "\u2423")
    else:
        display = text
    return display.replace("$", r"\$")


def plot_attention_overlay(
    tokens: np.ndarray,
    weights: np.ndarray,
    row_labels: Sequence[str],
    tokenizer,
    path: str | Path,
    title: str,
    bos_id: int,
) -> None:
    """One box per token, shaded by how much attention the final query puts on it.

    tokens: int64 [n, key_len]; weights: float [n, key_len] in [0, 1], same shape. Thin wrapper
    over exp1.show_sequences.plot_sequences -- see that function for the box/shading mechanics.
    """
    plot_sequences(
        tokens,
        tokenizer,
        path,
        title,
        n_examples=tokens.shape[0],
        bos_id=bos_id,
        shade=weights,
        shade_label="attention from the final token",
        row_labels=row_labels,
    )


def plot_token_effects(
    token_ids: np.ndarray,
    texts: Sequence[str],
    classes: Sequence[str],
    panels: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    path: str | Path,
    title: str,
    ylabel: str,
    top_k: int,
    zero_line: bool = False,
) -> None:
    """One row per panel: a wide scatter of mean-vs-token-id (left) and the top/bottom-k tokens
    as horizontal bars (right). NaN entries in a panel's `mean` are unmeasured and are skipped.

    token_ids, texts, classes: [T] each ("byte"/"common"/"rare" per _classify_id). panels maps a
    panel title to (mean, lo, hi), each [T], e.g. a 95% CI on the per-token mean of some head
    metric or its change. x is rank position within ids sorted (class, id) -- not the raw id --
    so the huge id gap between "common" and "rare" doesn't squash the small-id bands.
    """
    token_ids = np.asarray(token_ids)
    texts = np.asarray(texts, dtype=object)
    classes = np.asarray(classes, dtype=object)
    n_tokens = len(token_ids)
    if not (len(texts) == n_tokens and len(classes) == n_tokens):
        raise ValueError("token_ids, texts, and classes must all have the same length")
    if not panels:
        raise ValueError("panels must not be empty")
    for name, arrays in panels.items():
        if len(arrays) != 3 or not all(len(a) == n_tokens for a in arrays):
            raise ValueError(f"panel {name!r} must be a (mean, lo, hi) triple, each of length {n_tokens}")
    bad_classes = set(np.unique(classes)) - set(_CLASS_ORDER)
    if bad_classes:
        raise ValueError(f"classes must be one of {_CLASS_ORDER}, got {sorted(bad_classes)}")

    # Rank position within ids sorted by (class, id): a stable, gap-free x axis.
    class_rank = np.array([_CLASS_ORDER.index(c) for c in classes])
    order = np.lexsort((token_ids, class_rank))
    rank_of = np.empty(n_tokens, dtype=int)
    rank_of[order] = np.arange(n_tokens)
    x_all = rank_of

    counts = {c: int(np.sum(classes == c)) for c in _CLASS_ORDER}
    bounds, start = {}, 0
    for c in _CLASS_ORDER:
        n = counts[c]
        bounds[c] = (start, start + n - 1) if n else (start, start - 1)  # lo > hi marks an empty band
        start += n
    point_colors = np.array([_CLASS_COLORS[c] for c in classes])

    # Pre-compute each panel's top/bottom-k split before sizing the figure: bar count, not the
    # scatter, is what determines how tall a row needs to be for its labels to stay legible.
    bar_gap = 1  # blank rows separating the "highest" and "lowest" bar groups
    panel_rows = {}
    for name in panels:
        mean, lo, hi = (np.asarray(a, dtype=float) for a in panels[name])
        valid = ~np.isnan(mean)
        valid_idx = np.flatnonzero(valid)
        order_desc = valid_idx[np.argsort(-mean[valid_idx])]
        k = min(top_k, len(order_desc) // 2)
        panel_rows[name] = (mean, lo, hi, valid, order_desc[:k], order_desc[len(order_desc) - k:], k)

    label_pitch, row_chrome = 0.18, 1.1  # inches per bar row / fixed room for title + x-axis
    row_heights = [max(2.6, label_pitch * (2 * k + bar_gap) + row_chrome) for *_, k in panel_rows.values()]

    panel_names = list(panels)
    fig, axes = plt.subplots(
        len(panel_names),
        2,
        figsize=(12.0, sum(row_heights) * 1.08),  # +8%: room the shared hspace below takes from each row
        gridspec_kw={"width_ratios": [3.2, 1.0], "height_ratios": row_heights, "hspace": 0.5, "wspace": 0.35},
        squeeze=False,
    )

    for row, name in enumerate(panel_names):
        ax_l, ax_r = axes[row]
        mean, lo, hi, valid, top_idx, bot_idx, k = panel_rows[name]

        for c in _CLASS_ORDER:
            lo_x, hi_x = bounds[c]
            if lo_x > hi_x:
                continue
            ax_l.axvspan(lo_x - 0.5, hi_x + 0.5, color=_CLASS_COLORS[c], alpha=0.08, zorder=0)
            ax_l.text((lo_x + hi_x) / 2, 0.96, _BAND_TITLES[c], transform=ax_l.get_xaxis_transform(),
                       ha="center", va="top", fontsize=7.5, color="dimgray")

        if zero_line:
            ax_l.axhline(0.0, color="dimgray", linewidth=1.0, linestyle="--", zorder=1)

        ax_l.errorbar(x_all[valid], mean[valid], yerr=np.vstack([mean[valid] - lo[valid], hi[valid] - mean[valid]]),
                       fmt="none", ecolor="0.65", elinewidth=0.6, alpha=0.6, zorder=1)
        ax_l.scatter(x_all[valid], mean[valid], c=point_colors[valid], s=9, zorder=2, linewidths=0)

        for c in _CLASS_ORDER:  # running median per band, over measured points only
            lo_x, hi_x = bounds[c]
            band_mask = valid & (x_all >= lo_x) & (x_all <= hi_x)
            if band_mask.sum() < 2:
                continue
            bx, bv = x_all[band_mask], mean[band_mask]
            band_order = np.argsort(bx)
            bx, bv = bx[band_order], bv[band_order]
            window = max(5, len(bv) // 20)
            med = pd.Series(bv).rolling(window, center=True, min_periods=1).median().to_numpy()
            ax_l.plot(bx, med, color=_CLASS_COLORS[c], linewidth=1.6, zorder=3)

        ax_l.set_xlim(-1, n_tokens)
        ax_l.set_xticks([])
        ax_l.set_title(name, loc="left", fontsize=10)

        # Right: top_k highest and top_k lowest tokens by mean, read as one descending list
        # (highest at the very top, a blank row, then the lowest at the very bottom).
        all_idx = np.concatenate([top_idx, bot_idx])
        all_y = np.concatenate([np.arange(k)[::-1] + (k + bar_gap), np.arange(k)[::-1]])
        if k:
            bar_colors = [_CLASS_COLORS[classes[i]] for i in all_idx]
            xerr = np.vstack([mean[all_idx] - lo[all_idx], hi[all_idx] - mean[all_idx]])
            ax_r.barh(all_y, mean[all_idx], xerr=xerr, color=bar_colors, error_kw={"linewidth": 0.7, "ecolor": "0.3"})
            ax_r.set_yticks(all_y)
            ax_r.set_yticklabels([f"{_display_text(texts[i])} ({token_ids[i]})" for i in all_idx], fontsize=7)
            ax_r.set_ylim(-0.8, 2 * k + bar_gap - 0.2)
        if zero_line:
            ax_r.axvline(0.0, color="dimgray", linewidth=0.8, linestyle="--", zorder=0)
        ax_r.set_title(f"top/bottom {k}", fontsize=8)

    handles = [Patch(facecolor=_CLASS_COLORS[c], label=c) for c in _CLASS_ORDER]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.995, 0.995), frameon=False, ncol=3, fontsize=8)
    fig.supylabel(ylabel)
    fig.suptitle(title, fontsize=13)

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_token_head_heatmap(
    matrix: np.ndarray,
    token_labels: Sequence[str],
    head_labels: Sequence[str],
    path: str | Path,
    title: str,
    cbar_label: str,
) -> None:
    """Rows = tokens, columns = heads; a signed effect, diverging around 0. NaN cells (a
    token/head combination that wasn't measured) render gray rather than a color."""
    matrix = np.asarray(matrix, dtype=float)
    n_tokens, n_heads = matrix.shape
    if len(token_labels) != n_tokens or len(head_labels) != n_heads:
        raise ValueError(f"matrix is {matrix.shape} but got {len(token_labels)} token labels, {len(head_labels)} head labels")

    vmax = np.nanmax(np.abs(matrix)) if np.any(~np.isnan(matrix)) else 0.0
    vmax = vmax if vmax > 1e-12 else 1.0
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    cmap = _DIVERGING_CMAP.with_extremes(bad="0.75")

    fig, ax = plt.subplots(figsize=(0.55 * n_heads + 2.6, 0.32 * n_tokens + 1.4))
    im = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(np.arange(n_heads))
    ax.set_xticklabels(head_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(n_tokens))
    ax.set_yticklabels(token_labels)

    if n_tokens * n_heads <= 600:
        for (i, j), value in np.ndenumerate(matrix):
            if np.isnan(value):
                continue
            bg = cmap(norm(value))[:3]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6.5, color=_readable_text_color(bg))

    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=cbar_label)

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_attention_received(
    ids: np.ndarray,
    texts: Sequence[str],
    values: np.ndarray,
    counts: np.ndarray,
    path: str | Path,
    title: str,
    xlabel: str,
    top_k: int,
) -> None:
    """One head's "who gets attended to" chart: the top_k ids by value, as horizontal bars
    colored by token class and annotated with the number of sequences each id appeared in."""
    ids = np.asarray(ids)
    texts = np.asarray(texts, dtype=object)
    values = np.asarray(values, dtype=float)
    counts = np.asarray(counts)
    if not (len(ids) == len(texts) == len(values) == len(counts)):
        raise ValueError("ids, texts, values, and counts must all have the same length")

    valid_idx = np.flatnonzero(~np.isnan(values))
    order = valid_idx[np.argsort(-values[valid_idx])]
    k = min(top_k, len(order))
    top_idx = order[:k]

    colors = [_CLASS_COLORS[_classify_id(int(i))] for i in ids[top_idx]]
    labels = [f"{_display_text(texts[i])} ({ids[i]})" for i in top_idx]
    y = np.arange(k)[::-1]  # highest value at the top

    fig, ax = plt.subplots(figsize=(6.5, 0.34 * k + 1.2))
    ax.barh(y, values[top_idx], color=colors)
    span = float(np.max(values[top_idx]) - min(0.0, np.min(values[top_idx]))) if k else 1.0
    for yi, value, n in zip(y, values[top_idx], counts[top_idx]):
        ax.text(value + span * 0.015, yi, f"n={int(n)}", va="center", fontsize=7, color="dimgray")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_title(title)

    present = [c for c in _CLASS_ORDER if c in {_classify_id(int(i)) for i in ids[top_idx]}]
    if present:
        handles = [Patch(facecolor=_CLASS_COLORS[c], label=c) for c in present]
        ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=7)

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
