"""Experiment 5 figure for the advisor meeting (saved arrays only, no model run).

meet_exp5.png, three panels:
  A  consistency of the own − wrong direction at T by layer (S = ‖mean δ‖ / RMS(δ), LayerNorm-style normalised states),
     against the exact cell-level sign-flip null (all 2^11 patterns; the 11 cells are the independent units).
  B  where the layer-1 signal comes from: the projection of each part of δ_1 = x_1(own) − x_1(wrong) onto mean δ_1
     (layer-0 heads via z @ W_O, and MLP0 as the rest; x_0 is identical so δ_1 is exactly their sum; shares sum to 1).
  C  activation patching: fraction F(l) of the late-BOS effect carried by T's stream (exp5/patch.py results).

Run:  uv run --offline python -m exp5.meeting_figures [--out onepager]
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write figures to files, never open a window
import matplotlib.pyplot as plt
import numpy as np

from exp3.attention_maps3 import AXIS_GREY, TEXT_GREY
from exp4.onepager_figures import style
from exp4.run_controlled import PROJECT_ROOT

EXTRACT = PROJECT_ROOT / "results/20261025-170021_exp5res"
PATCH = PROJECT_ROOT / "results/20261025-174409_exp5P"
FILLERS = ("common", "rare")
COLOR = {"common": "#2a78d6", "rare": "#eb6834"}
TAUS = (31, 16, 1)


def normalised(x: np.ndarray) -> np.ndarray:
    """LayerNorm without gain: centre (TL states already are) and divide by the RMS over the model dimension."""
    x = x - x.mean(axis=-1, keepdims=True)
    return x / np.sqrt((x ** 2).mean(axis=-1, keepdims=True))


def consistency(b: str) -> tuple[np.ndarray, np.ndarray]:
    """S [3 τ, 12 layers] and the cell-level null's 95th percentile [3, 12]."""
    z = np.load(EXTRACT / f"seq_{b}.npz", mmap_mode="r")
    signs = np.array(list(itertools.product((-1.0, 1.0), repeat=11)))  # every flip of the 11 cells
    S, null = np.zeros((3, 12)), np.zeros((3, 12))
    for j in range(3):
        own = z["resid"][:, 0, j].astype(np.float32)  # [R, 11, 13, 768]
        wrong = z["resid"][:, 1, j].astype(np.float32)
        for l in range(1, 13):
            d = normalised(own[:, :, l]) - normalised(wrong[:, :, l])  # [R, 11, 768]
            rms = np.sqrt((d ** 2).sum(-1).mean())
            cell_sums = d.sum(axis=0)  # [11, 768]
            n = d.shape[0] * d.shape[1]
            S[j, l - 1] = np.linalg.norm(cell_sums.sum(0) / n) / rms
            null[j, l - 1] = np.quantile(np.linalg.norm(signs @ cell_sums / n, axis=1) / rms, 0.95)
    return S, null


def layer1_shares(b: str) -> dict[str, float]:
    """Shares of mean δ_1 (projection onto it) from L0H2, the other layer-0 heads and MLP0, pooled over τ."""
    z = np.load(EXTRACT / f"seq_{b}.npz", mmap_mode="r")
    W_O = np.load(EXTRACT / "weights.npz")["W_O"][0].astype(np.float32)  # [12 heads, 64, 768]
    resid = z["resid"][:, :2, :, :, 1].astype(np.float32)  # x_1 at T for own, wrong: [R, 2, 3, 11, 768]
    zz = z["z"][:, :2, :, :, 0].astype(np.float32)  # layer-0 z at T: [R, 2, 3, 11, 12, 64]
    delta = (resid[:, 0] - resid[:, 1]).reshape(-1, resid.shape[-1]).mean(0)
    heads = np.einsum("rtchd,hdm->hm", zz[:, 0] - zz[:, 1], W_O) / np.prod(zz.shape[0:1] + zz.shape[2:4])
    mlp = delta - heads.sum(0)  # b_O cancels in own − wrong
    proj = lambda v: float(v @ delta / (delta @ delta))
    return {"L0H2": proj(heads[2]), "other layer-0 heads": proj(heads.sum(0) - heads[2]), "MLP0": proj(mlp)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Experiment 5 meeting figure")
    parser.add_argument("--out", default=PROJECT_ROOT / "onepager", type=Path)
    args = parser.parse_args(argv)
    p = json.loads((PATCH / "p_results.json").read_text())
    fig = plt.figure(figsize=(7.6, 2.35))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 0.75, 1.15], left=0.07, right=0.99, top=0.8, bottom=0.2, wspace=0.42)
    ax = fig.add_subplot(gs[0, 0])
    layers = np.arange(1, 13)
    nulls = []
    for b in FILLERS:
        S, null = consistency(b)
        nulls.append(null)
        for j, (tau, ls) in enumerate(zip(TAUS, ("-", "--", ":"))):
            ax.plot(layers, S[j], color=COLOR[b], lw=1.2, ls=ls, label=f"{b}, d = {32 - tau}")
        print(b, "S l=1:", np.round(S[:, 0], 3), "l=12:", np.round(S[:, 11], 3), "null95 range", null.min().round(3), null.max().round(3))
    ax.fill_between(layers, 0, np.max(nulls, axis=(0, 1)), color="0.88", lw=0, label="chance (95%, per-pair null)")
    ax.set_ylim(0, 1)
    ax.set_xticks([1, 4, 8, 12])
    ax.set_title("A · the stream at T knows the match\nfrom layer 1 (all 11 pairs agree)", fontsize=6.3, pad=4)
    ax.legend(fontsize=4.5, frameon=False, ncol=2, loc="lower left", handlelength=1.6)  # over the light null band
    style(ax, "layer", "consistency of own − wrong", size=5.4)
    ax = fig.add_subplot(gs[0, 1])
    parts = ("L0H2", "other layer-0 heads", "MLP0")
    colors = {"L0H2": "#1baf7a", "other layer-0 heads": "#9fd8c2", "MLP0": "#898781"}
    short = {"L0H2": "L0H2", "other layer-0 heads": "other L0\nheads", "MLP0": "MLP0"}
    for i, b in enumerate(FILLERS):
        sh = layer1_shares(b)
        print(b, "layer-1 shares", {k: round(v, 3) for k, v in sh.items()})
        bottom = 0.0
        for part in parts:
            ax.bar(i, sh[part], bottom=bottom, color=colors[part], width=0.66)
            ax.text(i, bottom + sh[part] / 2, f"{short[part]}\n{sh[part]:.0%}", ha="center", va="center", fontsize=4.4,
                    color="white" if part != "other layer-0 heads" else "#2b5b4b", linespacing=1.0)
            bottom += sh[part]
    ax.set_xticks([0, 1])
    ax.set_xticklabels(FILLERS)
    ax.set_ylim(0, 1.02)
    ax.set_title("B · where the layer-1\nsignal comes from", fontsize=6.3, pad=4)
    style(ax, None, "share of mean δ at layer 1", size=5.4)
    ax = fig.add_subplot(gs[0, 2])
    ax.axvspan(6.5, 11.5, color="0.95", lw=0)
    ax.text(9, 0.06, "fewer measured\nlayers remain", fontsize=4.6, color=TEXT_GREY, ha="center")
    ax.axhline(1, color="0.7", lw=0.6)
    ls_ = np.arange(1, 12)
    for b in FILLERS:
        for tau, ls in ((16, "-"), (1, ":")):
            fwd = p["results"][b][str(tau)]["Pa"]["fwd"]
            ax.plot(ls_, fwd["F"], color=COLOR[b], lw=1.2, ls=ls, label=f"{b}, d = {32 - tau}")
            if "lo" in fwd:
                ax.fill_between(ls_, fwd["lo"], fwd["hi"], color=COLOR[b], alpha=0.15, lw=0)
    ax.set_xticks([1, 3, 5, 7, 9, 11])
    ax.set_ylim(0, 1.25)
    ax.set_title("C · patching T's stream: share of the\nresting effect it carries, by layer", fontsize=6.3, pad=4)
    ax.legend(fontsize=4.5, frameon=False, loc="upper left", handlelength=1.6)
    style(ax, "layer patched (wrong → own)", "fraction transferred F(l)", size=5.4)
    path = args.out / "meet_exp5.png"
    fig.savefig(path, dpi=250)
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
