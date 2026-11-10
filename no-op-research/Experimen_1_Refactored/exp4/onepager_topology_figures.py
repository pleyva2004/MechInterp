"""Figures for the topology one-pager (saved arrays only, no model run). Reuses the analyses' own code, so the
figures draw exactly what was tested.

topo_families.png: head-behaviour space (exp4.tda2_mapper's shape-normalised features, 2-D PCA), coloured by the
                   head families found by hand, then by layer.
topo_shapes.png:   four distance curves with the significant persistence features of exp4.tda3_curves marked.
topo_loops.png:    matching-pair cycles (exp4.tda4_graph) by layer and country position, its capital vs a wrong
                   capital, and the layer-0 attention graph that closes one.

Run:  uv run --offline python -m exp4.onepager_topology_figures [--out onepager]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write figures to files, never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from exp3.attention_maps3 import AXIS_GREY, TEXT_GREY
from exp4 import tda2_mapper as m2
from exp4 import tda4_graph as g4
from exp4.onepager_figures import style
from exp4.run_controlled import PROJECT_ROOT

TDA3_RUN = "results/20261025-102102_exp4tda3"
TDA4_RUN = "results/20261025-113614_exp4tda4"
T = 32
GREEN, ORANGE, BLUE, PURPLE, GREY = "#1baf7a", "#eb6834", "#2a78d6", "#9467bd", "#898781"
FAMILIES = {  # the families found by hand (exp4/LOG.md), recovered by lens-free clustering
    "same-country": (GREEN, ["L0H2", "L1H11", "L3H0", "L4H10"]),
    "previous-token, look away": (ORANGE, ["L2H2", "L2H5", "L2H8", "L3H2", "L3H7", "L3H8", "L4H11"]),
    "previous-token, other": ("#f2b48f", ["L2H3", "L2H9", "L1H0", "L3H6"]),
    "late, any distance": (BLUE, ["L9H8", "L10H3", "L11H7", "L11H2"]),
    "advisor's L11H8": (PURPLE, ["L11H8"]),
}


def fig_families(path: Path) -> None:
    rows, _ = m2.load(time.time())
    v = m2.raw_features(m2.curves(rows))  # [144, 186]: D curves for its / wrong capital / random token, both fillers
    f = m2.shape(v)
    x = f - f.mean(axis=0)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    xy = u[:, :2] * s[:2]
    share = s[:2] ** 2 / (s ** 2).sum()
    resp = np.linalg.norm(v, axis=1)
    size = 6 + 60 * resp / resp.max()
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.35))
    ax = axes[0]
    ax.scatter(*xy.T, s=size, color="0.82", lw=0)
    named = {}
    for fam, (color, heads) in FAMILIES.items():
        ix = [int(np.flatnonzero(m2.NAMES == h)[0]) for h in heads]
        ax.scatter(*xy[ix].T, s=size[ix] + 6, color=color, lw=0.4, edgecolor="white", label=fam, zorder=3)
        named |= {h: i for h, i in zip(heads, ix)}
    offsets = {"L4H10": (-4, 5), "L0H2": (5, -1), "L1H11": (-18, -6)}  # the three sit on top of each other
    for h in ("L0H2", "L3H0", "L4H10", "L1H11", "L3H7", "L4H11", "L2H9", "L9H8", "L10H3", "L11H8"):
        ax.annotate(h, xy[named[h]], xytext=offsets.get(h, (3, 2)), textcoords="offset points", fontsize=4.8, color="0.25")
    ax.set_title("coloured by the families found by hand", fontsize=6.5, pad=4)
    ax.legend(fontsize=4.8, frameon=False, loc="best", handletextpad=0.2, borderaxespad=0.2)
    ax = axes[1]
    sc = ax.scatter(*xy.T, s=size, c=m2.LAYER, cmap="viridis", lw=0)
    ax.set_title("coloured by layer", fontsize=6.5, pad=4)
    cb = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.02, ticks=[0, 3, 6, 9, 11])
    cb.ax.tick_params(labelsize=5, colors=AXIS_GREY, labelcolor=TEXT_GREY)
    cb.outline.set_edgecolor(AXIS_GREY)
    for ax in axes:
        style(ax, f"PC1 ({share[0]:.0%})", f"PC2 ({share[1]:.0%})", size=5.2)
    fig.subplots_adjust(left=0.06, right=0.97, top=0.9, bottom=0.17, wspace=0.22)
    fig.savefig(path, dpi=250)
    plt.close(fig)
    print(f"saved {path}")


def fig_shapes(path: Path) -> None:
    zc = np.load(PROJECT_ROOT / m2.CTRL_NPZ)
    zk = np.load(PROJECT_ROOT / m2.CAP_NPZ)
    feats = pd.read_csv(PROJECT_ROOT / TDA3_RUN / "tda3_features.csv")
    shapes = pd.read_csv(PROJECT_ROOT / TDA3_RUN / "tda3_shapes.csv")
    panels = [("L11H8", "own", "decays from the adjacent spot"), ("L3H0", "own", "rises with distance"),
              ("L0H2", "own", "high everywhere, slow slope"), ("L9H8", "random", "flat, spikes at both ends")]
    d = np.arange(1, T)
    fig, axes = plt.subplots(1, len(panels), figsize=(7.6, 1.85))
    for ax, (name, cond, title) in zip(axes, panels):
        l, h = int(name[1:name.index("H")]), int(name[name.index("H") + 1:])
        pre = ("common" if cond == "random" else f"{cond}/common")
        z = zc if cond == "random" else zk
        mean, lo, hi = (z[f"{pre}/country/{k}"][:, l, h] for k in ("mean", "lo", "hi"))
        color = BLUE if cond == "random" else GREEN
        ax.axhline(0, color="0.7", lw=0.6)
        ax.fill_between(d, lo, hi, color=color, alpha=0.2, lw=0)
        ax.plot(d, mean, color=color, lw=1.3)
        sel = feats[(feats.name == name) & (feats.condition == cond) & (feats.filler == "common")
                    & feats.role.isin(["dominant", "secondary"])]
        for _, r in sel.iterrows():
            ax.scatter([r.d], [mean[int(r.d) - 1]], marker="^" if r.type == "peak" else "v", s=22, color="black", zorder=4)
        shp = shapes[(shapes.name == name) & (shapes.condition == cond) & (shapes.filler == "common")]["shape"].iloc[0]
        final = "its capital" if cond == "own" else "random token"
        ax.set_title(f"{name}: {title}\nT = {final}; class: {shp}", fontsize=6, pad=3)
        ax.set_xticks([1, 10, 20, 31])
        style(ax, "distance T − τ", "Δ attention to τ" if ax is axes[0] else None, size=5.2)
    fig.subplots_adjust(left=0.07, right=0.995, top=0.8, bottom=0.2, wspace=0.32)
    fig.savefig(path, dpi=250)
    plt.close(fig)
    print(f"saved {path}")


def fig_loops(path: Path) -> None:
    z = np.load(PROJECT_ROOT / TDA4_RUN / "tda_arrays.npz", allow_pickle=True)
    gr = np.load(PROJECT_ROOT / TDA4_RUN / "graphs.npz")
    t = yaml.safe_load((PROJECT_ROOT / "configs/exp4_tda4.yaml").read_text())["tda"]
    taus = [int(x) for x in z["taus"]]
    conds = list(z["conds"])
    fig = plt.figure(figsize=(7.6, 2.55))
    gs = fig.add_gridspec(2, 7, width_ratios=[1, 1, 1, 1, 0.06, 0.55, 4.0], left=0.05, right=0.995, top=0.86,
                          bottom=0.14, wspace=0.12, hspace=0.6)
    im = None
    for k, (b, cond) in enumerate([("common", "own"), ("common", "wrong"), ("rare", "own"), ("rare", "wrong")]):
        ax = fig.add_subplot(gs[:, k])
        freq = np.stack([z[f"{b}/t{tau}/boot_mpc"][conds.index(cond), :12, 0] for tau in taus], axis=1)  # [layer, τ]
        im = ax.imshow(freq, aspect="auto", cmap="Greens", vmin=0, vmax=1, origin="lower")
        ax.set_xticks(range(len(taus)))
        ax.set_xticklabels(taus, fontsize=4.2)
        ax.set_yticks(range(12))
        ax.set_yticklabels(range(12) if k == 0 else [], fontsize=4.6)
        ax.set_title(f"{'its' if cond == 'own' else 'wrong'} capital\n{b}", fontsize=5.6, pad=3)
        style(ax, "τ (country position)", "layer" if k == 0 else None, size=4.6)
        for spine in ax.spines.values():
            spine.set_visible(True)
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[:, 4]))
    cb.ax.tick_params(labelsize=4.6, colors=AXIS_GREY, labelcolor=TEXT_GREY)
    cb.outline.set_edgecolor(AXIS_GREY)
    cb.set_label("share of bootstraps with a loop", fontsize=5, color="black")
    tau, u = 16, 0  # layer 0, country 16 back: a loop with its capital only
    ti = taus.index(tau)
    s = {c: g4.summarize(g4.dist(gr[f"common/{c}"][ti, u], bos=False), tau, T, False, t) for c in ("own", "wrong")}
    thr = 1.0 - s["own"]["mpc"]["birth"]
    for r, cond in enumerate(("own", "wrong")):
        ax = fig.add_subplot(gs[r, 6])
        mpc = s[cond]["mpc"]
        g4._arcs(ax, gr[f"common/{cond}"][ti, u], thr, tau, T, mpc["cycle"] if mpc else None, "")
        label = ("its capital: loop via capital→country (green)" if cond == "own" else
                 "wrong capital: that edge too weak (dotted), no loop")
        ax.set_title(f"layer 0, τ = {tau} · {label}", fontsize=5.4, pad=2)
        ax.tick_params(labelsize=4.6, colors=AXIS_GREY, labelcolor=TEXT_GREY)
    fig.savefig(path, dpi=250)
    plt.close(fig)
    print(f"saved {path}")
    for c in ("own", "wrong"):
        print(c, "mpc:", s[c]["mpc"] and s[c]["mpc"]["cycle"], "role:", s[c]["role"])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Topology one-pager figures")
    parser.add_argument("--out", default=PROJECT_ROOT / "onepager", type=Path)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    fig_families(args.out / "topo_families.png")
    fig_shapes(args.out / "topo_shapes.png")
    fig_loops(args.out / "topo_loops.png")


if __name__ == "__main__":
    main()
