"""Figures for Experiment 2 (phrase written into near/far slots), from the long `effects` table.

Three views of the paired per-(layer, head) differences: a forest plot of the tracked
heads (mean + bootstrap CI per contrast and placement), grouped bars counting heads with a
meaningful effect per contrast (placebo drawn as the grey, hatched noise floor), and a
per-layer profile of mean |difference| showing where in depth the phrase acts. Styling and
helpers follow exp1.plots / exp1ext.plots_cells.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # write figures to files, never open a window

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from exp1.plots import _CONDITION_COLORS
from exp1ext.plots_cells import _group_color

# Contrast colors: total in black (it is the sum of the other three), then the dataviz
# categorical slots blue/orange/aqua for its components; placebo is the muted-ink noise null.
# Marker shapes differ too, so identity never rests on color alone.
_CONTRAST_COLORS: dict[str, str] = {
    "total": "black",
    "order": "#2a78d6",
    "identity": "#eb6834",
    "form": "#1baf7a",
    "placebo": "#898781",
}
_CONTRAST_MARKERS: dict[str, str] = {"total": "D", "order": "o", "identity": "s", "form": "^", "placebo": "v"}
_FALLBACK_CONTRAST_COLOR = "#52514e"

# Placement colors for the count bars (green / violet from the validated condition palette).
_PLACEMENT_COLORS: dict[str, str] = {"near": _CONDITION_COLORS[0], "far": _CONDITION_COLORS[1]}
_PLACEMENT_ORDER = ("near", "far")
_GROUP_ORDER = ("sink rises", "sink falls", "label mix only", "control")

_ZERO_COLOR = "#898781"
_SEPARATOR_COLOR = "#c3c2b7"
_LABEL_INK = "#52514e"
_N_HEADS = 144


def _contrast_color(contrast: str) -> str:
    return _CONTRAST_COLORS.get(contrast, _FALLBACK_CONTRAST_COLOR)


def _present_placements(df: pd.DataFrame) -> list[str]:
    """Placements in the data, near/far first then any others alphabetically."""
    present = set(df["placement"].unique())
    known = [p for p in _PLACEMENT_ORDER if p in present]
    return known + sorted(present - set(known))


def _ordered_heads(heads: pd.DataFrame, present_names: set[str]) -> pd.DataFrame:
    """Tracked heads that exist in effects, ordered by group (known order first) then layer, head."""
    h = heads[heads["name"].isin(present_names)].copy()
    rank = {g: i for i, g in enumerate(_GROUP_ORDER)}
    h["_g"] = h["group"].map(lambda g: rank.get(g, len(rank)))
    return h.sort_values(["_g", "group", "layer", "head"], kind="stable").reset_index(drop=True)


def _subset(effects: pd.DataFrame, **conds: str) -> pd.DataFrame:
    mask = np.ones(len(effects), dtype=bool)
    for col, val in conds.items():
        mask &= (effects[col] == val).to_numpy()
    return effects[mask]


def _empty_figure(path: str | Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(5.0, 2.0))
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=9, color=_LABEL_INK)
    ax.set_title(title)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_tracked_forest(
    effects: pd.DataFrame,
    heads: pd.DataFrame,
    base: str,
    phrase: str,
    metric: str,
    path: str | Path,
    contrasts: Sequence[str] = ("total", "order", "identity", "form"),
) -> None:
    """Forest plot of tracked heads: one panel per placement, shared x, CI whisker per contrast.

    Rows are tracked heads in group blocks (separators plus block labels on the right edge);
    contrasts get a small vertical dodge. Filled marker = meaningful, hollow = not. Tracked
    heads absent from `effects` are skipped, as are placements with no data.
    """
    title = f"{base} / {phrase!r} / {metric}: paired difference, tracked heads"
    sub = _subset(effects, base=base, phrase=phrase, metric=metric)
    sub = sub[sub["contrast"].isin(contrasts)]
    hd = _ordered_heads(heads, set(sub["name"].unique()))
    placements = _present_placements(sub)
    if sub.empty or hd.empty or not placements:
        _empty_figure(path, title, "no matching rows")
        return

    n, n_c = len(hd), len(contrasts)
    y_of = {name: i for i, name in enumerate(hd["name"])}
    dodge = 0.8 / n_c
    x_lo = float(sub["lo"].min())
    x_hi = float(sub["hi"].max())
    pad = 0.05 * (x_hi - x_lo) if x_hi > x_lo else 0.05

    fig, axes = plt.subplots(
        1, len(placements), figsize=(4.6 * len(placements) + 1.6, max(3.5, 0.5 * n + 1.8)),
        sharex=True, sharey=True, squeeze=False,
    )
    axes = axes[0]

    for ax, placement in zip(axes, placements):
        panel = _subset(sub, placement=placement)
        for c_i, contrast in enumerate(contrasts):
            d = _subset(panel, contrast=contrast)
            d = d[d["name"].isin(y_of)]
            if d.empty:
                continue
            y = np.array([y_of[nm] for nm in d["name"]]) + (c_i - (n_c - 1) / 2) * dodge
            mean, lo, hi = (d[c].to_numpy(dtype=float) for c in ("mean", "lo", "hi"))
            color = _contrast_color(contrast)
            ax.hlines(y, lo, hi, color=color, linewidth=1.1, zorder=2)
            meaningful = d["meaningful"].to_numpy(dtype=bool)
            for mask, filled in ((meaningful, True), (~meaningful, False)):
                if mask.any():
                    ax.scatter(
                        mean[mask], y[mask], s=22, marker=_CONTRAST_MARKERS.get(contrast, "o"),
                        facecolor=color if filled else "white", edgecolor=color, linewidth=1.0, zorder=3,
                    )
        ax.axvline(0.0, color=_ZERO_COLOR, linewidth=1.0, zorder=0)
        ax.set_xlim(x_lo - pad, x_hi + pad)
        ax.set_title(placement)
        ax.set_xlabel(f"mean difference in {metric}")
        ax.set_yticks(np.arange(n))
        ax.set_yticklabels(hd["name"], fontsize=8)
        ax.set_ylim(n - 0.5, -0.5)  # first head at the top
        ax.grid(axis="x", color="#e8e7e1", linewidth=0.6, zorder=0)

        groups = list(hd["group"])
        for b in range(1, n):
            if groups[b] != groups[b - 1]:
                ax.axhline(b - 0.5, color=_SEPARATOR_COLOR, linewidth=0.8, zorder=0)

    # group block labels once, on the right edge of the last panel
    groups = list(hd["group"])
    start = 0
    for i in range(1, n + 1):
        if i == n or groups[i] != groups[start]:
            axes[-1].text(
                1.01, (start + i - 1) / 2, groups[start], transform=axes[-1].get_yaxis_transform(),
                ha="left", va="center", fontsize=8, color=_LABEL_INK,
            )
            start = i

    handles = [
        Line2D([0], [0], color=_contrast_color(c), marker=_CONTRAST_MARKERS.get(c, "o"), markersize=5,
               linewidth=1.1, label=c)
        for c in contrasts
    ]
    handles += [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="#52514e", markeredgecolor="#52514e",
               markersize=5, label="meaningful (filled)"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="white", markeredgecolor="#52514e",
               markersize=5, label="not meaningful (hollow)"),
    ]
    fig_h = fig.get_figheight()
    fig.subplots_adjust(top=1 - 0.9 / fig_h, bottom=0.8 / fig_h, wspace=0.08)
    fig.suptitle(title, y=1 - 0.1 / fig_h)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=len(handles),
               frameon=False, fontsize=8)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_meaningful_counts(effects: pd.DataFrame, metric: str, path: str | Path) -> None:
    """Grouped bars: heads (of 144) with meaningful=True per contrast, one bar per placement.

    One panel per (phrase row, base column). Placebo is hatched grey (the noise floor): its
    bars show what a pure-noise contrast yields under the same thresholds.
    """
    title = f"{metric}: heads with a meaningful effect"
    sub = _subset(effects, metric=metric)
    if sub.empty:
        _empty_figure(path, title, "no matching rows")
        return

    bases = sorted(sub["base"].unique())
    phrases = sorted(sub["phrase"].unique())
    placements = _present_placements(sub)
    contrast_order = [c for c in _CONTRAST_COLORS if c in set(sub["contrast"])]
    contrast_order += sorted(set(sub["contrast"]) - set(contrast_order))
    x = np.arange(len(contrast_order))
    width = 0.8 / len(placements)

    counts = (
        sub[sub["meaningful"].astype(bool)]
        .groupby(["base", "phrase", "placement", "contrast"])["name"].nunique()
    )
    max_count = max(float(counts.max()) if len(counts) else 0.0, 1.0)

    fig, axes = plt.subplots(
        len(phrases), len(bases), figsize=(max(4.5, 0.95 * len(contrast_order) + 1.5) * len(bases),
                                           3.6 * len(phrases) + 0.6),
        sharey=True, squeeze=False,
    )
    for r, phrase in enumerate(phrases):
        for c, base in enumerate(bases):
            ax = axes[r][c]
            for p_i, placement in enumerate(placements):
                vals = np.array([counts.get((base, phrase, placement, k), 0) for k in contrast_order], dtype=float)
                offset = (p_i - (len(placements) - 1) / 2) * width
                bars = ax.bar(x + offset, vals, width=width * 0.9,
                              color=_PLACEMENT_COLORS.get(placement, _FALLBACK_CONTRAST_COLOR))
                for rect, k, v in zip(bars, contrast_order, vals):
                    if k == "placebo":
                        rect.set_facecolor("#c3c2b7")
                        rect.set_edgecolor(_PLACEMENT_COLORS.get(placement, _FALLBACK_CONTRAST_COLOR))
                        rect.set_hatch("///")
                    ax.text(rect.get_x() + rect.get_width() / 2, v + max_count * 0.015, f"{int(v)}",
                            ha="center", va="bottom", fontsize=7)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{k}\n(noise floor)" if k == "placebo" else k for k in contrast_order], fontsize=8)
            ax.set_ylim(0, max_count * 1.15)
            ax.set_title(f"{base} / {phrase!r}" if len(phrases) > 1 else str(base))
            if c == 0:
                ax.set_ylabel(f"heads meaningful (of {_N_HEADS})")

    handles = [Patch(facecolor=_PLACEMENT_COLORS.get(p, _FALLBACK_CONTRAST_COLOR), label=p) for p in placements]
    handles.append(Patch(facecolor="#c3c2b7", edgecolor=_LABEL_INK, hatch="///", label="placebo = noise floor"))
    fig.suptitle(title)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=len(handles),
               frameon=False, fontsize=8, title="Placement")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_layer_profile(
    effects: pd.DataFrame,
    base: str,
    phrase: str,
    metric: str,
    path: str | Path,
    contrasts: Sequence[str] = ("total", "order", "identity", "form"),
) -> None:
    """Mean over each layer's heads of |mean difference|, one line per contrast, panel per placement.

    Panels share y so near and far magnitudes compare directly; layers with no rows are gaps.
    """
    title = f"{base} / {phrase!r} / {metric}: mean |difference| by layer"
    sub = _subset(effects, base=base, phrase=phrase, metric=metric)
    sub = sub[sub["contrast"].isin(contrasts)]
    placements = _present_placements(sub)
    if sub.empty or not placements:
        _empty_figure(path, title, "no matching rows")
        return

    sub = sub.assign(_abs=sub["mean"].abs())
    layers = np.arange(int(max(11, sub["layer"].max())) + 1)

    fig, axes = plt.subplots(1, len(placements), figsize=(4.6 * len(placements) + 0.6, 3.8),
                             sharey=True, squeeze=False)
    for ax, placement in zip(axes[0], placements):
        panel = _subset(sub, placement=placement)
        for contrast in contrasts:
            d = _subset(panel, contrast=contrast)
            if d.empty:
                continue
            prof = d.groupby("layer")["_abs"].mean().reindex(layers)
            ax.plot(layers, prof.to_numpy(), color=_contrast_color(contrast), linewidth=1.8,
                    marker=_CONTRAST_MARKERS.get(contrast, "o"), markersize=4.5, label=contrast)
        ax.set_xticks(layers)
        ax.set_xlabel("Layer")
        ax.set_title(placement)
        ax.grid(axis="y", color="#e8e7e1", linewidth=0.6, zorder=0)
    axes[0][0].set_ylabel(f"mean |difference| in {metric}")
    axes[0][0].set_ylim(bottom=0)
    axes[0][-1].legend(frameon=False, fontsize=8, title="Contrast", loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.suptitle(title, y=1.04)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_surprisal_variants(variants: pd.DataFrame, depth_slopes: pd.DataFrame, metric: str, path: str | Path) -> None:
    """One panel per base class: each variant's mean slot surprisal (x, nats) against its mean `metric` over the
    late-layer heads (filled) and early-layer heads (hollow), labelled by variant. The within-row slope and CI for
    each depth group (from exp2.analysis.surprisal_slopes) go in the panel legend. Means are descriptive; the
    slopes are the test."""
    bases = list(dict.fromkeys(variants["base"]))
    fig, axes = plt.subplots(1, len(bases), figsize=(6.2 * len(bases), 4.8), squeeze=False)
    styles = {"late": dict(color="black", facecolor="black"), "early": dict(color=_LABEL_INK, facecolor="white")}
    for ax, base in zip(axes[0], bases):
        v = variants[variants["base"] == base]
        handles = []
        for depth, style in styles.items():
            col = f"mean_{metric}_{depth}"
            if col not in v:
                continue
            ax.scatter(v["mean_slot_nll"], v[col], s=36, edgecolors=style["color"], facecolors=style["facecolor"], zorder=3)
            r = depth_slopes[(depth_slopes["base"] == base) & (depth_slopes["depth"] == depth)]
            label = depth if r.empty else f"{depth}: slope {r['slope'].iloc[0]:+.4f} [{r['lo'].iloc[0]:+.4f}, {r['hi'].iloc[0]:+.4f}]/nat"
            handles.append(Line2D([], [], marker="o", linestyle="", markeredgecolor=style["color"],
                                  markerfacecolor=style["facecolor"], label=label))
            if depth == "late":
                for row in v.itertuples():
                    ax.annotate(row.variant, (row.mean_slot_nll, getattr(row, col)), textcoords="offset points",
                                xytext=(4, 3), fontsize=7, color=_LABEL_INK)
        ax.set_title(f"{base} bases")
        ax.set_xlabel("mean slot surprisal (nats, summed over the slots)")
        ax.set_ylabel(f"mean {metric} over the depth group's heads")
        ax.legend(handles=handles, loc="best", fontsize=8, frameon=False)
    fig.suptitle(f"{metric} vs how predictable the slot tokens are, per variant")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_phrase_terms(depth: pd.DataFrame, metric: str, depth_group: str, path: str | Path,
                      texts: dict[str, str] | None = None, terms: Sequence[str] = ("nl", "shuffled", "nl−shuffled")) -> None:
    """Forest plot, phrases as rows: the depth-group mean interaction of each term (from
    exp2.analysis.additivity_tables) with its bootstrap CI over rows, one panel per base class. Rows are sorted by
    the last term's mean in the first panel; filled = CI excludes 0."""
    d = depth[(depth["metric"] == metric) & (depth["depth"] == depth_group) & depth["term"].isin(terms)]
    bases = list(dict.fromkeys(d["base"]))
    order = (d[(d["base"] == bases[0]) & (d["term"] == terms[-1])].sort_values("mean")["phrase"].tolist())
    colors = {"nl": "black", "shuffled": "#eb6834", "nl−shuffled": "#2a78d6"}
    markers = {"nl": "D", "shuffled": "s", "nl−shuffled": "o"}
    fig, axes = plt.subplots(1, len(bases), figsize=(5.4 * len(bases), 0.42 * len(order) + 1.8), sharey=True, squeeze=False)
    for ax, base in zip(axes[0], bases):
        ax.axvline(0, color=_ZERO_COLOR, linewidth=1, zorder=1)
        for k, term in enumerate(terms):
            g = d[(d["base"] == base) & (d["term"] == term)].set_index("phrase").reindex(order)
            y = np.arange(len(order)) + (k - (len(terms) - 1) / 2) * 0.22
            excl = ((g["lo"] > 0) | (g["hi"] < 0)).to_numpy()
            c = colors.get(term, _FALLBACK_CONTRAST_COLOR)
            ax.errorbar(g["mean"], y, xerr=[g["mean"] - g["lo"], g["hi"] - g["mean"]], fmt="none", ecolor=c, elinewidth=1, zorder=2)
            ax.scatter(g["mean"][excl], y[excl], marker=markers.get(term, "o"), color=c, s=26, zorder=3, label=term)
            ax.scatter(g["mean"][~excl], y[~excl], marker=markers.get(term, "o"), facecolors="white", edgecolors=c, s=26, zorder=3)
        ax.set_yticks(np.arange(len(order)))
        ax.set_yticklabels([(texts or {}).get(ph, ph).strip() for ph in order])
        ax.set_title(f"{base} bases")
        ax.set_xlabel(f"interaction, mean over {depth_group}-layer heads ({metric})")
    axes[0][0].legend(loc="lower right", fontsize=8, frameon=False)
    fig.suptitle(f"What each phrase adds beyond its single tokens ({depth_group} layers); filled = CI excludes 0")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
