"""Exp1-style attention maps for an exp2 run.

Per variant (none / phrase / scrambled / same-form random words): the mean attention the final token pays to BOS,
what each head attends to most, and the change in BOS attention against `none`.

Run:  uv run python -m exp2.attention_maps --run results/20261003-151225_exp2 [--phrase sushi]

Attention is re-extracted with TransformerLens from the run's own design.npz, so the inputs are exactly the run's
rows; the figures go to <run>/figures/attention_maps/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write figures to files, never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from matplotlib.patches import Patch

from exp1.backends import get_final_attention, load_model, set_determinism
from exp1.sequences import prepend_bos

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("none", "nl", "shuffled", "matched")
CATEGORIES = ("first token (BOS)", "previous token", "other phrase slot", "other token")
COLORS = ("tab:red", "tab:blue", "#eda100", "lightgray")  # exp1's red / blue / gray, plus one for the phrase slots


def categorize(attn: np.ndarray, slots: list[int]) -> np.ndarray:
    """attn: [n, layer, head, key]. Per row, the key the final token attends to most, as a CATEGORIES code:
    0 BOS, 1 the previous token, 2 another slot position of the placement, 3 anything else."""
    key_len = attn.shape[-1]
    top = attn.argmax(-1)
    cat = np.full(top.shape, 3, dtype=np.int8)
    cat[np.isin(top, [p for p in slots if p != key_len - 2])] = 2
    cat[top == key_len - 2] = 1
    cat[top == 0] = 0
    return cat


def head_category(cat: np.ndarray) -> np.ndarray:
    """[n, layer, head] codes -> [layer, head]: each head's most common target (ties go to the lower code)."""
    counts = np.stack([(cat == c).sum(axis=0) for c in range(len(CATEGORIES))])
    return counts.argmax(axis=0)


def variant_titles(cfg: dict, phrase: str) -> dict[str, str]:
    ph = cfg["phrases"][phrase]
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg["figures"]["tokenizer"])  # display only
    return {"none": "no phrase (random tokens)", "nl": repr(ph["text"]),
            "shuffled": f"{tok.decode(ph['shuffled'])!r} (scrambled)", "matched": "same-form random words"}


def plot_maps(bos: dict[str, np.ndarray], cats: dict[str, np.ndarray], titles: dict[str, str], suptitle: str,
              path: Path) -> None:
    """Rows: mean BOS attention, what each head attends to most, change in BOS attention vs `none`."""
    variants = list(bos)
    fig, axes = plt.subplots(3, len(variants), figsize=(3.3 * len(variants) + 1.2, 10.2), squeeze=False)
    diffs = {v: bos[v] - bos["none"] for v in variants if v != "none"}
    vmax = max(float(np.abs(d).max()) for d in diffs.values()) or 1.0
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    im_bos = im_diff = None
    for j, v in enumerate(variants):
        im_bos = axes[0, j].imshow(bos[v], origin="lower", vmin=0, vmax=1, cmap="viridis")
        axes[0, j].set_title(titles[v], fontsize=10)
        axes[1, j].imshow(cats[v], origin="lower", cmap=ListedColormap(COLORS), vmin=0, vmax=len(COLORS) - 1)
        if v == "none":
            axes[2, j].axis("off")
            axes[2, j].text(0.5, 0.5, "reference", ha="center", va="center", color="gray", transform=axes[2, j].transAxes)
        else:
            im_diff = axes[2, j].imshow(diffs[v], origin="lower", cmap="RdBu_r", norm=norm)
        for i in range(3):
            if axes[i, j].axison:
                axes[i, j].set_xticks(range(0, 12, 2))
                axes[i, j].set_yticks(range(0, 12, 2))
                axes[i, j].set_xlabel("Head", fontsize=8)
                if j == 0 or (i == 2 and j == 1):
                    axes[i, j].set_ylabel("Layer", fontsize=8)
                axes[i, j].tick_params(labelsize=7)
    # Fix the grid first, then hang the colorbars / legend off each row's right edge so they line up with it.
    fig.subplots_adjust(left=0.07, right=0.86, top=0.93, bottom=0.05, hspace=0.32, wspace=0.18)
    for i, label in enumerate(("Mean attention on the first token (BOS)", "What each head attends to most",
                               "Change in BOS attention vs no phrase")):
        box = axes[i, 0].get_position()
        fig.text(0.012, (box.y0 + box.y1) / 2, label, rotation=90, va="center", ha="left", fontsize=10, fontweight="bold")
    right = axes[0, -1].get_position().x1 + 0.012
    for row, im, label in ((0, im_bos, "attention on BOS"), (2, im_diff, "Δ attention on BOS")):
        box = axes[row, -1].get_position()
        fig.colorbar(im, cax=fig.add_axes((right, box.y0, 0.012, box.height)), label=label)
    box = axes[1, -1].get_position()
    fig.legend(handles=[Patch(color=c, label=l) for c, l in zip(COLORS, CATEGORIES)], loc="center left",
               bbox_to_anchor=(right - 0.004, (box.y0 + box.y1) / 2), ncol=1, frameon=False, fontsize=9)
    fig.suptitle(suptitle, fontsize=12)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> list[Path]:
    parser = argparse.ArgumentParser(description="exp1-style attention maps for an exp2 run")
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--phrase", default=None, help="phrase key (default: the first in the run's config)")
    parser.add_argument("--example-head", default="L7H0", help="head whose per-row attention is saved for exp2.token_effects")
    args = parser.parse_args(argv)
    ex_layer, ex_head = (int(x) for x in args.example_head[1:].split("H"))
    run = args.run if args.run.is_absolute() else PROJECT_ROOT / args.run
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    phrase = args.phrase or next(iter(cfg["phrases"]))
    design = np.load(run / "design.npz")
    set_determinism(cfg["seed"])
    bcfg = cfg["backends"]["transformer_lens"]
    model = load_model("transformer_lens", bcfg["model_name"], cfg["device"], cfg["dtype"])
    titles = variant_titles(cfg, phrase)
    reference = pd.read_parquet(run / "cells.parquet")
    out_dir = run / "figures" / "attention_maps"
    out_dir.mkdir(parents=True, exist_ok=True)

    paths, summary = [], []
    by_position: dict[str, np.ndarray] = {}  # mean attention over rows, [layer, head, key], per base/placement/variant
    example: dict[str, np.ndarray] = {}      # the example head's per-row attention, [n, key]
    for base in cfg["bases"]:
        for placement, slots in cfg["placements"].items():
            bos, cats = {}, {}
            for v in VARIANTS:
                key = f"{base}/none" if v == "none" else f"{base}/{phrase}/{placement}/{v}"
                attn = get_final_attention(model, prepend_bos(design[key], cfg["bos_token_id"]), cfg["batch_size"],
                                           cfg["query_position"])
                by_position[f"{base}/{placement}/{v}"] = attn.mean(axis=0)
                example[f"{base}/{placement}/{v}"] = attn[:, ex_layer, ex_head, :]
                bos[v] = attn[..., 0].mean(axis=0)
                cats[v] = head_category(categorize(attn, slots))
                ref = reference[(reference["base"] == base) & (reference["phrase"] == phrase)
                                & (reference["placement"] == placement) & (reference["variant"] == v)]
                ref_grid = ref.pivot(index="layer", columns="head", values="mean_sink_mass").to_numpy()
                if not np.allclose(ref_grid, bos[v], atol=1e-5):  # same rows as the run, so the means must match
                    raise AssertionError(f"{key}: re-extracted BOS attention differs from the run's cells.parquet")
                summary.append({"base": base, "placement": placement, "variant": v,
                                **{c: int((cats[v] == k).sum()) for k, c in enumerate(CATEGORIES)}})
            path = out_dir / f"{base}_{phrase}_{placement}.png"
            plot_maps(bos, cats, titles, f"Exp 2, {base} bases: phrase at positions {slots} ({placement}); "
                                         f"final token at {cfg['seq_len']}; mean over {len(design[f'{base}/none'])} rows (TL)", path)
            paths.append(path)
            print(f"saved {path}")
    table = pd.DataFrame(summary)
    table.to_csv(out_dir / f"head_categories_{phrase}.csv", index=False)
    np.savez_compressed(out_dir / f"attention_by_position_{phrase}.npz", **by_position)
    np.savez_compressed(out_dir / f"example_head_{args.example_head}_{phrase}.npz", **example)
    print(table.to_string(index=False))
    return paths


if __name__ == "__main__":
    main()
