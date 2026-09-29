"""Figures for exp1extended. Layer x head panels reuse exp1.plots; the per-head charts here follow the same
dataviz-skill palette (references/palette.md), with categorical hues assigned in slot order."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write figures to files, never open a window

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Categorical slots 1-4 of the dataviz palette (blue, orange, aqua, yellow), assigned to a chart's parts in the
# order given: the default slot order passes the adjacent-pair CVD and normal-vision checks, and a chart's parts
# always sit next to each other within a head's row.
_SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
_INK, _MUTED = "#0b0b0b", "#52514e"


def plot_decomposition(
    table: pd.DataFrame,
    path: str | Path,
    title: str,
    xlabel: str,
    parts: tuple[str, ...] = ("query", "key", "interaction"),
    note: str = "",
    total: bool = True,
) -> None:
    """One panel per comparison: for each head (rows), the parts of the change in its primary metric as dots
    with CI whiskers, plus (if `total`) the total as a black bar outline.

    table columns: comparison, head, group, part (each of `parts`, plus "total" if `total`), mean, lo, hi.
    """
    comparisons = list(dict.fromkeys(table["comparison"]))
    heads = list(dict.fromkeys(table["head"]))
    groups = table.drop_duplicates("head").set_index("head")["group"]
    if len(parts) > len(_SERIES_COLORS):
        raise ValueError(f"at most {len(_SERIES_COLORS)} parts, got {parts}")
    colors = dict(zip(parts, _SERIES_COLORS))
    offsets = dict(zip(parts, np.linspace(-0.27, 0.27, len(parts)) if len(parts) > 1 else [0.0]))

    fig, axes = plt.subplots(1, len(comparisons), figsize=(5.2 * len(comparisons) + 1.5, 0.42 * len(heads) + 1.8),
                             sharey=True)
    axes = np.atleast_1d(axes)
    y = np.arange(len(heads))[::-1]
    for ax, comparison in zip(axes, comparisons):
        rows = table[table["comparison"] == comparison].set_index(["head", "part"])
        if total:
            totals = np.array([rows.loc[(h, "total"), "mean"] for h in heads])
            ax.barh(y, totals, height=0.75, fill=False, edgecolor=_INK, linewidth=1.0, label="total")
        for part in parts:
            mean = np.array([rows.loc[(h, part), "mean"] for h in heads])
            lo = np.array([rows.loc[(h, part), "lo"] for h in heads])
            hi = np.array([rows.loc[(h, part), "hi"] for h in heads])
            # A percentile-bootstrap CI need not contain the point estimate (skewed statistics), so whisker
            # lengths are clipped at 0 rather than passed negative; the tables keep the raw bounds.
            xerr = np.clip(np.vstack([mean - lo, hi - mean]), 0.0, None)
            ax.errorbar(mean, y + offsets[part], xerr=xerr, fmt="o", markersize=5,
                        color=colors[part], ecolor=colors[part], elinewidth=1.2, capsize=0,
                        label=part)
        ax.axvline(0.0, color=_MUTED, linewidth=0.8)
        ax.set_title(comparison)
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", color="#e4e3df", linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([f"{h}  ({groups[h].replace('_', ' ')})" for h in heads])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.suptitle(title)
    if note:
        fig.text(0.5, 0.0, note, ha="center", va="top", fontsize=8, color=_MUTED)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_dose(
    curves: pd.DataFrame,
    heads: list[tuple[int, int]],
    groups: dict[tuple[int, int], str],
    primary_metrics: dict[str, list[str]],
    path: str | Path,
    title: str,
) -> None:
    """Small multiples, one per target head: its first primary metric vs the number of context positions switched
    from rare to common, one line per Q/P arm with a CI band.

    curves columns: metric, head (e.g. "L7H7"), arm (r | c), level, mean, lo, hi.
    """
    arms = list(dict.fromkeys(curves["arm"]))
    arm_label = {"r": "final + previous rare", "c": "final + previous common"}
    colors = dict(zip(arms, _SERIES_COLORS))
    n_cols = 5
    n_rows = int(np.ceil(len(heads) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 2.4 * n_rows + 0.8), sharex=True, squeeze=False)
    for ax, head in zip(axes.ravel(), heads):
        metric = primary_metrics[groups[head]][0]
        name = f"L{head[0]}H{head[1]}"
        for arm in arms:
            c = curves[(curves["head"] == name) & (curves["metric"] == metric) & (curves["arm"] == arm)].sort_values("level")
            ax.fill_between(c["level"], c["lo"], c["hi"], color=colors[arm], alpha=0.18, linewidth=0)
            ax.plot(c["level"], c["mean"], color=colors[arm], linewidth=2, marker="o", markersize=3,
                    label=arm_label.get(arm, arm))
        ax.set_title(f"{name} {metric}", fontsize=9)
        ax.grid(color="#e4e3df", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(heads):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("common context tokens (of 30)", fontsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=len(arms), frameon=False)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_token_profiles(
    effects: pd.DataFrame,
    confirm: pd.DataFrame,
    features: pd.DataFrame,
    heads: list[tuple[int, int]],
    metrics: list[str],
    path: str | Path,
    title: str,
    n_labels: int = 3,
) -> None:
    """The per-token graph. One row per head (its first primary metric): left, byte tokens 0-255 by id; right, ids
    256-39999 on a log axis with a running median. Points are coloured by whether the token starts a word; 0 is a
    typical rare token. The n_labels highest and lowest *confirmed* tokens are labelled with their text.

    effects columns: metric, head, id, group, effect (one position, pooled backgrounds).
    confirm columns: metric, head, side, id, token, confirmed.   features: id, word_start, ...
    """
    metric = metrics[0]
    ws = features.set_index("id")["word_start"]
    series = {"word start": _SERIES_COLORS[0], "continuation": _SERIES_COLORS[1]}
    fig, axes = plt.subplots(len(heads), 2, figsize=(13, 2.3 * len(heads) + 0.8), sharey="row",
                             gridspec_kw={"width_ratios": [1, 4]}, squeeze=False)
    for (ax_b, ax_r), head in zip(axes, heads):
        name = f"L{head[0]}H{head[1]}"
        e = effects[(effects["head"] == name) & (effects["metric"] == metric)]
        start = ws.loc[e["id"]].to_numpy(dtype=bool)
        for ax, sel in ((ax_b, e["id"].to_numpy() < 256), (ax_r, e["id"].to_numpy() >= 256)):
            for label, mask in (("word start", start), ("continuation", ~start)):
                m = sel & mask
                ax.scatter(e["id"].to_numpy()[m], e["effect"].to_numpy()[m], s=7, color=series[label], alpha=0.55,
                           linewidths=0, label=label)
            ax.axhline(0.0, color=_MUTED, linewidth=0.8)
            ax.grid(color="#e4e3df", linewidth=0.6)
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=7)
        rest = e[e["id"] >= 256].sort_values("id")
        ax_r.plot(rest["id"], rest["effect"].rolling(51, center=True, min_periods=15).median(), color=_INK,
                  linewidth=1.5, label="running median (51 tokens)")
        ax_r.set_xscale("log")
        ax_r.axvline(1000, color=_MUTED, linewidth=0.8, linestyle="--")
        ax_b.set_ylabel(f"{name}\n{metric}", fontsize=8)
        ax_b.set_xlim(-5, 260)
        c = confirm[(confirm["head"] == name) & (confirm["metric"] == metric) & (confirm["side"] != "")]
        c = c.sort_values("confirmed")
        for _, row in pd.concat([c.head(n_labels), c.tail(n_labels)]).iterrows():
            ax = ax_b if row["id"] < 256 else ax_r
            y_disc = e.loc[e["id"] == row["id"], "effect"]
            if len(y_disc):
                ax.annotate(repr(row["token"])[1:-1], (row["id"], y_disc.iloc[0]), fontsize=6.5, color=_INK,
                            xytext=(3, 2), textcoords="offset points")
    axes[0, 0].set_title("byte tokens (0-255)", fontsize=9)
    axes[0, 1].set_title("tokens 256-39999 (log scale; dashed = 1000, the common / rare boundary)", fontsize=9)
    axes[-1, 0].set_xlabel("token id", fontsize=8)
    axes[-1, 1].set_xlabel("token id", fontsize=8)
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_rarity_wordstart(table: pd.DataFrame, path: str | Path, title: str) -> None:
    """Per head: common − rare_matched within word-start and continuation tokens, and their difference.
    table columns: head, metric, contrast, mean, lo, hi."""
    parts = tuple(dict.fromkeys(table["contrast"]))
    t = table.assign(part=table["contrast"], comparison="common − rare_matched, same surface form",
                     group="", head=table["head"] + " " + table["metric"])
    plot_decomposition(t, path, title, "difference in mean token effect (logit units)", parts=parts, total=False,
                       note="two-way (token × background) bootstrap 95% CIs")


def plot_feature_forest(table: pd.DataFrame, path: str | Path, title: str) -> None:
    """One panel per feature: each head's regression coefficient with a CI.
    table columns: head, metric, feature, coef, lo, hi."""
    t = table.rename(columns={"coef": "mean"}).assign(part="coefficient", comparison=table["feature"], group="",
                                                     head=table["head"] + " " + table["metric"])
    plot_decomposition(t, path, title, "coefficient (logit units)", parts=("coefficient",), total=False,
                       note="binary features: effect of having the feature; char_len, log_id, embed_norm: per SD. "
                            "Two-way bootstrap 95% CIs.")


def plot_token_heatmap(confirm: pd.DataFrame, path: str | Path, title: str, per_head: int = 2) -> None:
    """Tokens (the per_head highest and lowest confirmed tokens of each head) x heads, cell = confirmed effect of
    the token on that head's primary metric (every confirmed token is measured for every head).
    confirm columns: head, side, id, token, confirmed (primary metric rows only)."""
    from exp1.plots import _DIVERGING_CMAP  # the same blue <-> red diverging map as the Exp 1 diff figures
    from matplotlib.colors import TwoSlopeNorm

    picked = confirm[confirm["side"] != ""].sort_values("confirmed")
    chosen = pd.concat([picked.groupby("head").head(per_head), picked.groupby("head").tail(per_head)])
    ids = list(dict.fromkeys(chosen.sort_values("confirmed")["id"]))
    heads = list(dict.fromkeys(confirm["head"]))
    matrix = confirm.pivot_table(index="id", columns="head", values="confirmed", aggfunc="first").reindex(
        index=ids, columns=heads)
    text = confirm.drop_duplicates("id").set_index("id")["token"]
    vmax = float(np.nanmax(np.abs(matrix.to_numpy()))) or 1.0
    fig, ax = plt.subplots(figsize=(0.55 * len(heads) + 3.0, 0.22 * len(ids) + 1.8))
    im = ax.imshow(matrix.to_numpy(), cmap=_DIVERGING_CMAP, norm=TwoSlopeNorm(0.0, -vmax, vmax), aspect="auto")
    ax.set_xticks(np.arange(len(heads)))
    ax.set_xticklabels(heads, rotation=90, fontsize=7)
    ax.set_yticks(np.arange(len(ids)))
    ax.set_yticklabels([f"{repr(text[i])[1:-1]}  ({i})" for i in ids], fontsize=6.5)
    fig.colorbar(im, ax=ax, label="confirmed token effect (logit units; 0 = typical rare token)", fraction=0.04)
    ax.set_title(title, fontsize=10)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
