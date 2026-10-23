"""Experiment 1-style per-head attention maps for every Experiment 3 run, built from the runs' saved arrays (no model
is run).

Each figure has one column per variant (or placement) and three rows:
  1. mean attention the final token pays to BOS, per head (layer x head);
  2. what each head attends to most: the category of the key with the highest mean attention (Diffuse if that
     weight is < 0.2);
  3. the change in BOS attention against the column's reference arm, named in the column title.

Run:  uv run --offline python -m exp3.attention_maps3
Figures go to <run>/figures/exp1style_maps_<base>.png.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write figures to files, never open a window
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
BASES = ("rare", "common")
COLORS = {"Diffuse": "white", "BOS": "tab:red", "Previous token": "tab:blue", "Self": "tab:purple",
          "Country": "#eb6834", "Capital / city": "#1baf7a", "Phrase / text": "#eda100", "Copy target": "tab:green",
          "Earlier copy (dup)": "olive", "Punctuation / byte": "saddlebrown", "Other token": "lightgray"}
DIFFUSE = 0.2
AXIS_GREY, TEXT_GREY = "#C8C8C8", "#8C8C8C"  # recessive frames, ticks and numbers


def style_head_grid(ax, xsize: float = 5.5, ysize: float = 5.5, labels: bool = True, ylabel: bool = True) -> None:
    """Number every layer (y) and head (x) 0-11, in light grey, on a layer x head heatmap."""
    ax.set_xticks(range(12))
    ax.set_yticks(range(12))
    for spine in ax.spines.values():
        spine.set_color(AXIS_GREY)
    ax.tick_params(colors=AXIS_GREY, labelcolor=TEXT_GREY, length=1.5, pad=1)
    ax.tick_params(axis="x", labelsize=xsize)
    ax.tick_params(axis="y", labelsize=ysize)
    if labels:  # axis titles in black; only the index numbers stay grey
        ax.set_xlabel("head", fontsize=ysize + 1, color="black", labelpad=1)
        if ylabel:
            ax.set_ylabel("layer", fontsize=ysize + 1, color="black", labelpad=1)


def column(title: str, attn: np.ndarray, ref: np.ndarray | None = None, keys: dict[int, str] | None = None,
           key_len: int = 33) -> dict:
    """attn, ref: [L, H, K] mean attention at the final query. keys: model position -> category, overriding the
    defaults (0 BOS, key_len - 2 previous token, key_len - 1 self)."""
    cat = {0: "BOS", key_len - 2: "Previous token", key_len - 1: "Self"}
    cat.update(keys or {})
    return {"title": title, "attn": attn[..., :key_len], "ref": None if ref is None else ref[..., :key_len], "cat": cat}


def most_attended(attn: np.ndarray, cat: dict[int, str]) -> np.ndarray:
    """[L, H, K] -> [L, H] category names."""
    top = attn.argmax(axis=-1)
    names = np.vectorize(lambda k: cat.get(int(k), "Other token"))(top)
    return np.where(attn.max(axis=-1) < DIFFUSE, "Diffuse", names)


def plot(columns: list[dict], suptitle: str, path: Path) -> None:
    n = len(columns)
    fig, axes = plt.subplots(3, n, figsize=(3.3 * n + 2.2, 10.2), squeeze=False)
    cats = [most_attended(c["attn"], c["cat"]) for c in columns]
    present = [k for k in COLORS if any((c == k).any() for c in cats)]
    code = {k: i for i, k in enumerate(present)}
    cmap = ListedColormap([COLORS[k] for k in present])
    diffs = [c["attn"][..., 0] - c["ref"][..., 0] for c in columns if c["ref"] is not None]
    vmax = max([float(np.abs(d).max()) for d in diffs] + [1e-6])
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    im_bos = im_diff = None
    for j, (c, ca) in enumerate(zip(columns, cats)):
        im_bos = axes[0, j].imshow(c["attn"][..., 0], origin="lower", vmin=0, vmax=1, cmap="viridis")
        axes[0, j].set_title(c["title"], fontsize=9, pad=8)
        axes[1, j].imshow(np.vectorize(code.get)(ca), origin="lower", cmap=cmap, vmin=-0.5, vmax=len(present) - 0.5)
        if c["ref"] is None:
            axes[2, j].axis("off")
            axes[2, j].text(0.5, 0.5, "reference", ha="center", va="center", color="gray", transform=axes[2, j].transAxes)
        else:
            im_diff = axes[2, j].imshow(c["attn"][..., 0] - c["ref"][..., 0], origin="lower", cmap="RdBu_r", norm=norm)
        for i in range(3):
            ax = axes[i, j]
            if ax.axison:
                style_head_grid(ax, xsize=6, ysize=6.5)
    fig.subplots_adjust(left=0.07, right=0.84, top=0.9, bottom=0.05, hspace=0.38, wspace=0.28)
    for i, label in enumerate(("Mean attention on BOS", "What each head attends to most",
                               "Change in BOS attention vs reference")):
        box = axes[i, 0].get_position()
        fig.text(0.012, (box.y0 + box.y1) / 2, label, rotation=90, va="center", ha="left", fontsize=10, fontweight="bold")
    right = axes[0, -1].get_position().x1 + 0.012
    box = axes[0, -1].get_position()
    fig.colorbar(im_bos, cax=fig.add_axes((right, box.y0, 0.01, box.height)), label="attention on BOS")
    if im_diff is not None:
        box = axes[2, -1].get_position()
        fig.colorbar(im_diff, cax=fig.add_axes((right, box.y0, 0.01, box.height)), label="Δ attention on BOS")
    box = axes[1, -1].get_position()
    fig.legend(handles=[Patch(facecolor=COLORS[k], edgecolor="0.6", label=k) for k in present], loc="center left",
               bbox_to_anchor=(right - 0.004, (box.y0 + box.y1) / 2), frameon=False, fontsize=8)
    fig.suptitle(suptitle, fontsize=11)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def engine_means(run: str, base: str) -> dict[str, np.ndarray]:
    """Variant name -> mean over rows of the unit-averaged attention [L, H, K], for runs built on exp3/engine.py."""
    import json

    z = np.load(RESULTS / run / "attention.npz")
    names = list(json.loads((RESULTS / run / "design_summary.json").read_text())["design"][base])
    att = z[f"{base}/att_row"]
    return {v: att[:, i].mean(axis=0) for i, v in enumerate(names)}


def phrase(*positions: int, prev: int = 31) -> dict[int, str]:
    return {p: ("Previous token" if p == prev else "Phrase / text") for p in positions}


def r1_out(base: str) -> tuple[list[dict], str]:
    z = np.load(RESULTS / "20261015-190957_exp3rel" / "attention_by_position.npz")
    none = z[f"{base}/none"]
    keys = {30: "Country", 31: "Capital / city"}
    cols = [column("no pair (row's random tokens)", none)]
    for grid in ("capital", "city"):
        for arm in ("correct", "wrong"):
            cols.append(column(f"{arm} {grid} pairs\n(Δ vs no pair)", z[f"{base}/{grid}/{arm}"], none, keys))
    return cols, (f"R1-out, {base} bases: country at 30, capital or city at 31, random final token at 32; mean over "
                  "200 rows and the arm's cells")


def r1_in(base: str) -> tuple[list[dict], str]:
    z = np.load(RESULTS / "20261015-203150_exp3rel_in" / "attention_by_position.npz")
    alone = z[f"{base}/capital_alone"]
    keys = {31: "Country", 32: "Capital / city"}
    cols = [column("capital alone at 32\n(random token at 31)", alone, keys={32: "Capital / city"}),
            column("correct country + capital\n(Δ vs capital alone)", z[f"{base}/capital/correct"], alone, keys),
            column("wrong country + capital\n(Δ vs capital alone)", z[f"{base}/capital/wrong"], alone, keys),
            column("correct country + city\n(Δ vs capital alone)", z[f"{base}/city/correct"], alone, keys),
            column("country + random word\n(anchor; Δ vs capital alone)", z[f"{base}/anchor"], alone, {31: "Country"})]
    return cols, f"R1-in, {base} bases: country at 31, capital / city AT the final token (32); mean over 200 rows"


def sweep(base: str) -> tuple[list[dict], str]:
    cols = []
    for i, j in ((1, 32), (16, 32), (31, 32), (16, 31), (30, 31)):
        f = np.load(RESULTS / "20261015-210238_exp3pos" / "placements" / f"{base}_{i}_{j}.npz")["by_key"]
        cols.append(column(f"country {i}, capital {j}\n(correct; Δ vs wrong pairs)", f[0], f[1:3].mean(axis=0),
                           {i: "Country", j: "Capital / city"}))
    return cols, f"Position sweep, {base} bases: correct country→capital pairs at five placements; mean over 50 rows"


def e31(base: str) -> tuple[list[dict], str]:
    m = engine_means("20261015-221059_exp3e31", base)
    cols = [column("rr+r: no phrase", m["rr+r"]),
            column("PVO+r: phrase 29–31, random query\n(Δ vs rr+r)", m["PVO+r"], m["rr+r"], phrase(29, 30, 31)),
            column("PV+r: phrase 30–31, random query\n(Δ vs rr+r)", m["PV+r"], m["rr+r"], phrase(30, 31)),
            column("PV+O: query completes the phrase\n(Δ vs rr+O)", m["PV+O"], m["rr+O"], phrase(30, 31)),
            column("PV+M: query a same-form word\n(Δ vs rr+M)", m["PV+M"], m["rr+M"], phrase(30, 31))]
    return cols, f"E3.1, {base} bases: query inside vs outside the phrase; mean over 500 rows and 12 phrases"


def e32(base: str) -> tuple[list[dict], str]:
    m = engine_means("20261015-223621_exp3e32", base)
    cols = [column("none: random tokens", m["none"])]
    for k in (1, 8, 16, 31):
        cols.append(column(f"last {k} passage token{'s' if k > 1 else ''}\n(Δ vs none)", m[f"nl_{k}"], m["none"],
                           phrase(*range(32 - k, 32))))
    return cols, f"E3.2, {base} bases: natural text of growing length before a random final token; 200 rows, 12 passages"


def e33(base: str) -> tuple[list[dict], str]:
    m = engine_means("20261015-214541_exp3e33", base)
    cols = [column("none: random tokens", m["none"])]
    for g in (0, 1, 3, 15):
        pos = tuple(range(29 - g, 32 - g))
        cols.append(column(f"phrase at {pos[0]}–{pos[-1]} (gap {g})\n(Δ vs scrambled order)", m[f"nl_g{g}"],
                           m[f"shuffled_g{g}"], phrase(*pos)))
    return cols, f"E3.3, {base} bases: the phrase moved away from the final token; 200 rows, 12 phrases"


def e34(base: str) -> tuple[list[dict], str]:
    m = engine_means("20261015-211838_exp3e34", base)
    rep = {10: "Phrase / text", 11: "Earlier copy (dup)", 12: "Copy target"}
    cols = [column("norepeat-nl: A at 31, B at 32", m["norepeat-nl"]),
            column("repeat-nl: A B C at 10–12\n(Δ vs norepeat-nl)", m["repeat-nl"], m["norepeat-nl"], rep),
            column("repeat-nl-1: one-token match\n(Δ vs norepeat-nl)", m["repeat-nl-1"], m["norepeat-nl"], rep),
            column("repeat-rand: random X Y Z\n(Δ vs norepeat-rand)", m["repeat-rand"], m["norepeat-rand"], rep),
            column("repeat-matched: same-form words\n(Δ vs norepeat-nl)", m["repeat-matched"], m["norepeat-nl"], rep)]
    return cols, f"E3.4, {base} bases: repeated phrase (copy target 12, earlier copy 11); 500 rows, 12 phrases"


def e35(base: str) -> tuple[list[dict], str]:
    m = engine_means("20261015-220224_exp3e35", base)
    ph = {26: "Phrase / text", 27: "Phrase / text", 28: "Phrase / text"}
    pun = {29: "Punctuation / byte"}
    cols = [column("phrase at 26–28, random 29", m["phrase+r"], keys=ph),
            column('phrase + "." at 29\n(Δ vs phrase + random)', m["phrase+period"], m["phrase+r"], {**ph, **pun}),
            column('random + "." at 29\n(Δ vs random + random)', m["random+period"], m["random+r"], pun),
            column('phrase + "," at 29\n(Δ vs phrase + random)', m["phrase+comma"], m["phrase+r"], {**ph, **pun}),
            column('phrase + "x" (byte control)\n(Δ vs phrase + random)', m["phrase+x"], m["phrase+r"], {**ph, **pun})]
    return cols, f"E3.5, {base} bases: punctuation at 29, random tokens at 30–31; 300 rows, 12 phrases"


RUNS = {"20261015-190957_exp3rel": r1_out, "20261015-203150_exp3rel_in": r1_in, "20261015-210238_exp3pos": sweep,
        "20261015-221059_exp3e31": e31, "20261015-223621_exp3e32": e32, "20261015-214541_exp3e33": e33,
        "20261015-211838_exp3e34": e34, "20261015-220224_exp3e35": e35}


def main() -> list[Path]:
    paths = []
    for run, build in RUNS.items():
        for base in BASES:
            cols, title = build(base)
            path = RESULTS / run / "figures" / f"exp1style_maps_{base}.png"
            plot(cols, title, path)
            paths.append(path)
            print(f"saved {path.relative_to(PROJECT_ROOT)}")
    return paths


if __name__ == "__main__":
    main()
