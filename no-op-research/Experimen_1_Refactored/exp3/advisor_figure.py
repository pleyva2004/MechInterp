"""Three-panel summary figure for the advisor one-pager, from saved Experiment 3 results (no model is run).

A: late BOS mass vs amount of natural text before a random final token (E3.2).
B: completion releases it: capital at the final token, by country position (position sweep).
C: truth alone does not: R1-out, correct - wrong pairing vs any country + capital pair - no pair.

Run:  uv run --offline python -m exp3.advisor_figure
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write figures to files, never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exp1ext.effects import bootstrap_mean_per_unit
from exp3.relations import pairing_effect

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
COLOR = {"rare": "#2a78d6", "common": "#eb6834"}
LABEL = {"rare": "rare filler", "common": "common filler"}


def panel_text(ax) -> None:
    p = pd.read_csv(RES / "20261015-223621_exp3e32" / "pooled.csv")
    p = p[p["measure"] == "BOS"]
    for base in ("rare", "common"):
        q = p[p["base"] == base]
        e = q[q["contrast"].str.match(r"^E_\d+$")].copy()
        e["k"] = e["contrast"].str[2:].astype(int)
        e = e.sort_values("k")
        ax.plot(e["k"], e["mean"], color=COLOR[base], lw=2, marker="o", ms=3, label=f"natural text, {LABEL[base]}")
        ax.fill_between(e["k"], e["lo"], e["hi"], color=COLOR[base], alpha=0.18, lw=0)
        o = q[q["contrast"].str.match(r"^order_\d+$")].copy()
        o["k"] = o["contrast"].str[6:].astype(int)
        s = e.set_index("k")["mean"].sub(o.set_index("k")["mean"]).dropna()  # scrambled - none = E - order
        ax.plot(s.index, s.values, color=COLOR[base], lw=1.2, ls="--")
    ax.plot([], [], color="0.4", lw=1.2, ls="--", label="same words, scrambled")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xlabel("natural-text tokens before the final token", fontsize=8)
    ax.set_ylabel("Δ late attention on BOS\n(vs random tokens)", fontsize=8)
    ax.set_title("① More coherent text → more resting", fontsize=9, loc="left", fontweight="bold")
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")


def panel_completion(ax) -> None:
    z = np.load(RES / "20261015-210238_exp3pos" / "sweep.npz")
    pl = [tuple(x) for x in z["placements"]]
    ks = sorted((i, k) for k, (i, j) in enumerate(pl) if j == 32)
    x = [i for i, _ in ks]
    idx = [k for _, k in ks]
    for base in ("rare", "common"):
        m, lo, hi = (z[f"{base}/fact/{s}"][idx, 0] for s in ("mean", "lo", "hi"))
        ax.plot(x, m, color=COLOR[base], lw=2, label=LABEL[base])
        ax.fill_between(x, lo, hi, color=COLOR[base], alpha=0.18, lw=0)
    for k, base in enumerate(("rare", "common")):  # placebo: one wrong pairing minus the other wrong pairings
        pm, plo, phi = (z[f"{base}/placebo/{s}"][idx, 0] for s in ("mean", "lo", "hi"))
        ax.plot(x, pm, color="0.55", lw=1, label="placebo (wrong − wrong)" if k == 0 else None)
        ax.fill_between(x, plo, phi, color="0.8", alpha=0.5, lw=0)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xlabel("position of the country (capital = final token, 32)", fontsize=8)
    ax.set_ylabel("Δ late attention on BOS\n(correct − wrong capital)", fontsize=8)
    ax.set_title("② Completion releases it, from any distance", fontsize=9, loc="left", fontweight="bold")
    ax.legend(fontsize=6.5, frameon=False, loc="lower right")


def panel_truth(ax) -> None:
    run = RES / "20261015-190957_exp3rel"
    grids = np.load(run / "late_grids.npz")
    means = np.load(run / "cell_means.npz")
    xs, labels = [], []
    for b_i, base in enumerate(("rare", "common")):
        g = grids[base][:, 0, :, :, 0]  # capital grid, late BOS, [rows, n, n]
        none = means[f"none/{base}"][:, 6:12, :, 0].mean(axis=(1, 2))
        for k, (name, d) in enumerate((("any pair\nvs none", g.mean(axis=(1, 2)) - none),
                                       ("correct −\nwrong", pairing_effect(g)))):
            m, lo, hi = (float(v) for v in bootstrap_mean_per_unit(d, 2000, 0, 0.95))
            x = b_i * 2.6 + k
            ax.bar(x, m, width=0.8, color=COLOR[base], alpha=1.0 if k == 0 else 0.45)
            ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], color="black", capsize=3, lw=1)
            ax.annotate(f"{m:+.3f}", (x + 0.42, max(hi, 0) + 0.001), ha="left", fontsize=7)
            xs.append(x)
            labels.append(name)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=7)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_ylabel("Δ late attention on BOS\n(country + capital, random final token)", fontsize=8)
    ax.set_title("③ Truth alone does not", fontsize=9, loc="left", fontweight="bold")
    ax.text(0.22, 0.97, "rare filler", color=COLOR["rare"], fontsize=7.5, transform=ax.transAxes, ha="center")
    ax.text(0.62, 0.97, "common filler", color=COLOR["common"], fontsize=7.5, transform=ax.transAxes, ha="center")


def main() -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.0))
    panel_text(axes[0])
    panel_completion(axes[1])
    panel_truth(axes[2])
    for ax in axes:
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(w_pad=2.0)
    out = ROOT / "onepager" / "advisor_fig.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print(f"saved {out}")
    for k, draw in enumerate((panel_text, panel_completion, panel_truth), start=1):  # one panel per file
        fig, ax = plt.subplots(figsize=(4.4, 3.0))
        draw(ax)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        path = ROOT / "onepager" / f"advisor_panel_{k}.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        print(f"saved {path}")
    return out


if __name__ == "__main__":
    main()
