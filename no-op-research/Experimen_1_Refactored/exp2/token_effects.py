"""How the phrase's tokens change the final token's attention: three figures.

1. attention_shift: where attention moves. For the late heads (layers 6-11), the change in attention the final token
   pays to each key position (BOS, the rest of the context, each phrase slot, itself), variant minus no phrase.
2. example_row: one real row, token by token, for one head, each box shaded by the attention it receives. The row is
   the one whose BOS increase (phrase minus no phrase) is closest to the median over all rows, not a hand pick.
3. buildup: more than the sum of its words. Late-layer BOS attention change for each word alone, each word pair and
   the full phrase, each with a tick at what its words predict if their effects simply added (Test A data).

Run (after exp2.attention_maps has saved its arrays for the iteration-1 run):
  uv run python -m exp2.token_effects --maps-run results/20261003-151225_exp2 --testA-run results/20261004-152303_exp2
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
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LATE = list(range(6, 12))
VARIANT_LABELS = {"none": "no phrase", "nl": "phrase", "shuffled": "scrambled", "matched": "random words"}
# Diverging blue <-> red with a gray midpoint, and a one-hue blue ramp (dataviz reference palette)
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", ["#2a78d6", "#f0efec", "#e34948"])
SEQUENTIAL = LinearSegmentedColormap.from_list("blue_ramp", ["#f4f8fd", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])
KIND_COLORS = {"word": "#1baf7a", "pair": "#eb6834", "phrase": "#2a78d6"}
INK, INK2, MUTED, AXIS = "#0b0b0b", "#52514e", "#898781", "#c3c2b7"


def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=8)


def position_bins(key_len: int, slots: list[int], slot_tokens: list[str]) -> list[tuple[str, list[int]]]:
    """Key positions grouped for display: BOS, the rest of the context, each slot, the previous token if it is not a
    slot, and the final token itself."""
    q = key_len - 1
    rest = [p for p in range(1, q) if p not in slots and (p != q - 1 or q - 1 in slots)]
    bins = [("BOS", [0]), (f"rest of\ncontext ({len(rest)})", rest)]
    if q - 1 not in slots:
        bins.append((f"previous\n({q - 1})", [q - 1]))
    bins += [(f"{p}\n{t!r}", [p]) for p, t in zip(slots, slot_tokens)]
    bins.append((f"itself\n({q})", [q]))
    return bins


def attention_shift(by_position: dict[str, np.ndarray], base: str, placement: str, slots: list[int],
                    slot_tokens: list[str], path: Path) -> pd.DataFrame:
    """Figure 1. Returns the late-head mean change per bin and variant (also drawn as the top strip)."""
    variants = ["nl", "shuffled", "matched"]
    none = by_position[f"{base}/{placement}/none"]
    bins = position_bins(none.shape[-1], slots, slot_tokens)
    shift = {}
    for v in variants:
        d = by_position[f"{base}/{placement}/{v}"] - none                       # [layer, head, key]
        shift[v] = np.stack([d[LATE][:, :, idx].sum(axis=-1) for _, idx in bins], axis=-1).reshape(len(LATE) * 12, -1)
    vmax = max(float(np.abs(s).max()) for s in shift.values())
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    fig, axes = plt.subplots(2, len(variants), figsize=(4.2 * len(variants) + 1.0, 10.5),
                             gridspec_kw={"height_ratios": [1, 13], "hspace": 0.04}, squeeze=False)
    rows = []
    for j, v in enumerate(variants):
        top, ax = axes[0, j], axes[1, j]
        mean = shift[v].mean(axis=0)
        top.imshow(mean[None, :], cmap=DIVERGING, norm=norm, aspect="auto")
        for k, m in enumerate(mean):
            top.text(k, 0, f"{m:+.3f}", ha="center", va="center", fontsize=8, color=INK)
            rows.append({"base": base, "placement": placement, "variant": v, "bin": bins[k][0].replace("\n", " "),
                         "mean_change_late_heads": float(m)})
        top.set_xticks([])
        top.set_yticks([0])
        top.set_yticklabels(["mean"], fontsize=8)
        top.set_title(f"{VARIANT_LABELS[v]} $-$ no phrase", fontsize=11)
        im = ax.imshow(shift[v], cmap=DIVERGING, norm=norm, aspect="auto")
        ax.set_xticks(range(len(bins)))
        ax.set_xticklabels([b for b, _ in bins], fontsize=8)
        ax.set_yticks([i * 12 + 5.5 for i in range(len(LATE))])
        ax.set_yticklabels([f"L{layer}" for layer in LATE] if j == 0 else [], fontsize=8)
        for i in range(1, len(LATE)):
            ax.axhline(i * 12 - 0.5, color="white", linewidth=1.2)
        ax.set_xlabel("key position the final token attends to", fontsize=9)
        if j == 0:
            ax.set_ylabel("late heads (12 per layer, head 0 at the top of each band)", fontsize=9)
    fig.colorbar(im, ax=axes[1, :].tolist(), fraction=0.02, pad=0.015, label="change in attention from the final token")
    fig.suptitle(f"Where the final token's attention moves when the slots change ({base} bases, positions {slots}; "
                 f"late heads, mean over rows)", fontsize=11)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(rows)


def example_row(example: dict[str, np.ndarray], design: np.lib.npyio.NpzFile, tokenizer, base: str, placement: str,
                phrase: str, head: str, first_shown: int, path: Path) -> dict[str, float]:
    """Figure 2. Picks the row whose BOS increase (phrase minus no phrase) for `head` is closest to the median."""
    variants = ["none", "nl", "shuffled", "matched"]
    att = {v: example[f"{base}/{placement}/{v}"] for v in variants}
    delta = att["nl"][:, 0] - att["none"][:, 0]
    row = int(np.argmin(np.abs(delta - np.median(delta))))
    key_len = att["none"].shape[1]
    shown = list(range(first_shown, key_len))                  # model positions shown one by one
    cols = [("BOS", [0])] + [(f"positions\n1–{first_shown - 1}", list(range(1, first_shown)))] + [(None, [p]) for p in shown]
    vmax = max(float(att[v][row].max()) for v in variants)
    fig, ax = plt.subplots(figsize=(1.18 * len(cols) + 1.6, 0.78 * len(variants) + 1.4))
    for i, v in enumerate(variants):
        y = len(variants) - 1 - i
        key = f"{base}/none" if v == "none" else f"{base}/{phrase}/{placement}/{v}"
        ids = design[key][row]                                  # 32 ids without BOS; model position p = index p - 1
        ax.text(-0.65, y, VARIANT_LABELS[v], ha="right", va="center", fontsize=9, color=INK)
        for c, (label, idx) in enumerate(cols):
            value = float(att[v][row, idx].sum())
            text = label if label is not None else repr(tokenizer.decode([int(ids[idx[0] - 1])]))
            face = SEQUENTIAL(min(value / vmax, 1.0))
            ax.add_patch(Rectangle((c - 0.46, y - 0.36), 0.92, 0.72, facecolor=face, edgecolor=AXIS, linewidth=0.6))
            dark = value / vmax > 0.55
            size = 6.5 if label else (7.5 if len(text) <= 9 else 6.3 if len(text) <= 12 else 5.4)
            ax.text(c, y + 0.13, text, ha="center", va="center", fontsize=size,
                    family="sans-serif" if label else "monospace", color="white" if dark else INK)
            ax.text(c, y - 0.17, f"{value:.2f}", ha="center", va="center", fontsize=8, fontweight="bold",
                    color="white" if dark else INK)
    q = key_len - 1
    ax.text(len(cols) - 1, len(variants) - 0.35, "final token\n(query)", ha="center", va="bottom", fontsize=7, color=INK2)
    ax.set_xlim(-2.2, len(cols) - 0.4)
    ax.set_ylim(-0.6, len(variants) + 0.2)
    ax.axis("off")
    ax.set_title(f"Head {head}: attention from the final token (position {q}) to each token, one {base}-base row\n"
                 f"(row {row}: its BOS increase {delta[row]:+.2f} is the median over {len(delta)} rows; "
                 f"darker = more attention)", fontsize=9.5)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"row": row, "delta_bos": float(delta[row]), "median_delta_bos": float(np.median(delta))}


def buildup(test_a_run: Path, path: Path, iters: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Figure 3, from Test A's cells.parquet: late-layer mean BOS attention change vs no phrase, pooled over phrases."""
    cfg = yaml.safe_load((test_a_run / "config.yaml").read_text())
    cells = pd.read_parquet(test_a_run / "cells.parquet")
    late = cells[cells["layer"].isin(LATE)].groupby(["base", "phrase", "variant"])["mean_sink_mass"].mean()
    roles = [("P", "word"), ("V", "word"), ("O", "word"), ("PV", "pair"), ("VO", "pair"), ("PO", "pair"), ("PVO", "phrase")]
    parts = {"PV": ["P", "V"], "VO": ["V", "O"], "PO": ["P", "O"], "PVO": ["P", "V", "O"]}
    labels = {"P": "I\npronoun", "V": "like\nverb", "O": "sushi\nobject", "PV": "I like\npron.+verb",
              "VO": "like sushi\nverb+obj.", "PO": "I … sushi\npron.…obj.", "PVO": "I like sushi\nall three"}
    rows = []
    for base in cfg["bases"]:
        for name, ph in cfg["phrases"].items():
            w1, w2, w3 = ph["text"].split()
            var = {"P": f"{name}:{w1}@29", "V": f"{name}:{w2}@30", "O": f"{name}:{w3}@31", "PV": f"{name}:{w1}@29+{w2}@30",
                   "VO": f"{name}:{w2}@30+{w3}@31", "PO": f"{name}:{w1}@29+{w3}@31", "PVO": "nl"}
            none = late[(base, name, "none")]
            d = {r: late[(base, name, var[r])] - none for r, _ in roles}
            for r, _ in roles:
                rows.append({"base": base, "phrase": name, "role": r, "delta": d[r],
                             "additive": sum(d[p] for p in parts[r]) if r in parts else np.nan})
    df = pd.DataFrame(rows)
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, len(cfg["bases"]), figsize=(7.6 * len(cfg["bases"]), 4.6), sharey=True, squeeze=False)
    summary = []
    for ax, base in zip(axes[0], cfg["bases"]):
        g = df[df["base"] == base]
        for x, (r, kind) in enumerate(roles):
            vals = g[g["role"] == r]["delta"].to_numpy() * 100
            boot = vals[rng.integers(0, len(vals), size=(iters, len(vals)))].mean(axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            mean = vals.mean()
            ax.bar(x, mean, width=0.62, color=KIND_COLORS[kind], zorder=2)
            ax.errorbar(x, mean, yerr=[[mean - lo], [hi - mean]], color=INK2, linewidth=0.9, capsize=3, zorder=3)
            add = g[g["role"] == r]["additive"].to_numpy() * 100
            entry = {"base": base, "role": r, "mean_delta_x100": mean, "lo": lo, "hi": hi}
            if r in parts:
                ax.plot([x - 0.36, x + 0.36], [add.mean()] * 2, color=INK, linewidth=2.2, zorder=4)
                entry["additive_x100"] = add.mean()
            summary.append(entry)
        full = [s for s in summary if s["base"] == base and s["role"] == "PVO"][0]
        top, bottom = full["mean_delta_x100"], full["additive_x100"]
        ax.annotate("", xy=(6.45, top), xytext=(6.45, bottom),
                    arrowprops=dict(arrowstyle="<->", color=INK, linewidth=0.9, shrinkA=0, shrinkB=0))
        ax.text(6.55, (top + bottom) / 2, f"+{top - bottom:.1f}\nbeyond the\nsum of its\nwords", fontsize=8, color=INK,
                va="center", ha="left")
        ax.axhline(0, color=AXIS, linewidth=0.8, zorder=1)
        ax.set_xticks(range(len(roles)))
        ax.set_xticklabels([labels[r] for r, _ in roles], fontsize=7.5)
        ax.set_title(f"{base} bases", fontsize=10)
        ax.set_xlim(-0.6, 7.6)
        ax.grid(axis="y", color="#e1e0d9", linewidth=0.5, zorder=0)
        _style(ax)
    axes[0][0].set_ylabel("change in late-layer BOS attention\nvs no phrase (×10⁻², mean over 12 phrases)", fontsize=9)
    handles = [Patch(color=KIND_COLORS["word"], label="one word alone"), Patch(color=KIND_COLORS["pair"], label="two words"),
               Patch(color=KIND_COLORS["phrase"], label="full phrase"),
               Line2D([], [], color=INK, linewidth=2.2, label="sum of its words alone (if effects simply added)")]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("More than the sum of its words: each word, each pair, and the full phrase "
                 "(labels show ' I like sushi'; bars pool 12 phrases, 95% CI over phrases)", fontsize=10, y=1.08)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(summary)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="figures: how the phrase's tokens change the final token's attention")
    parser.add_argument("--maps-run", required=True, type=Path, help="exp2 run already processed by exp2.attention_maps")
    parser.add_argument("--testA-run", required=True, type=Path)
    parser.add_argument("--phrase", default="sushi")
    parser.add_argument("--example-head", default="L7H0")
    args = parser.parse_args(argv)
    run = args.maps_run if args.maps_run.is_absolute() else PROJECT_ROOT / args.maps_run
    test_a = args.testA_run if args.testA_run.is_absolute() else PROJECT_ROOT / args.testA_run
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    tokenizer = AutoTokenizer.from_pretrained(cfg["figures"]["tokenizer"])  # display only
    maps = run / "figures" / "attention_maps"
    by_position = dict(np.load(maps / f"attention_by_position_{args.phrase}.npz"))
    example = dict(np.load(maps / f"example_head_{args.example_head}_{args.phrase}.npz"))
    design = np.load(run / "design.npz")
    out = run / "figures" / "token_effects"
    out.mkdir(parents=True, exist_ok=True)
    slot_tokens = [tokenizer.decode([i]) for i in cfg["phrases"][args.phrase]["ids"]]

    shifts = []
    for base in cfg["bases"]:
        for placement, slots in cfg["placements"].items():
            p = out / f"attention_shift_{base}_{placement}.png"
            shifts.append(attention_shift(by_position, base, placement, slots, slot_tokens, p))
            print(f"saved {p}")
        if "near" in cfg["placements"]:
            p = out / f"example_row_{base}_{args.example_head}.png"
            info = example_row(example, design, tokenizer, base, "near", args.phrase, args.example_head,
                               cfg["placements"]["near"][0] - 4, p)
            print(f"saved {p}  {info}")
    pd.concat(shifts).to_csv(out / "attention_shift_late_heads.csv", index=False)
    p = test_a / "figures" / "buildup_sink_mass.png"
    summary = buildup(test_a, p)
    summary.to_csv(test_a / "figures" / "buildup_sink_mass.csv", index=False)
    print(f"saved {p}")
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
