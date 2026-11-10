"""Figures for the Experiments 9-10 one-pager.

exp9_strip.png: R1-in, common filler. What each head attends to most with a country followed by its capital, and by
one of its cities.
exp10_fig.png: position sweep, common filler, capital as the final token, country at positions 1, 16 and 31. What
each head attends to most, per country position.
Each map has the sequence that produces it drawn above it.

The runs' saved attention arrays were removed in the Oct 23 commit, so the five maps are recomputed once (GPT-2 small,
the runs' own configs, cached base rows and attention code; about a minute on CPU) and cached in onepager/.

Run:  uv run --offline python -m exp3.exp910_figures
"""
from __future__ import annotations

import hashlib

import matplotlib

matplotlib.use("Agg")  # write figures to files, never open a window
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from exp3.attention_maps3 import COLORS, RESULTS, column, most_attended, style_head_grid
from exp3.meeting_maps import OUT

CACHE = OUT / "exp910_maps.npz"
RUNS = {"exp9": "20261015-203150_exp3rel_in", "exp10": "20261015-210238_exp3pos"}


def correct_maps() -> dict[str, np.ndarray]:
    """Mean attention by key position at the final query [L, H, K], each country followed by its own capital (or
    city) at 32, common filler. 'exp9/capital', 'exp9/city': country at 31 over R1-in's 200 rows. 'exp10/{i}': country
    at i over the sweep's 50 rows (its correct arm)."""
    if CACHE.exists():
        return dict(np.load(CACHE))
    from exp1.backends import load_model, set_determinism
    from exp2.design import cached_base_block, id_pool
    from exp3.run_relations import attention

    cfgs = {k: yaml.safe_load((RESULTS / run / "config.yaml").read_text()) for k, run in RUNS.items()}
    cfg = cfgs["exp9"]
    stim = {k: np.array([s[f"{k}_id"] for s in cfg["stimuli"]]) for k in ("country", "capital", "city")}
    stim_ids = np.concatenate([*stim.values(), cfg["demonstration_ids"]])
    pool = id_pool(cfg["bases"]["common"], stim_ids)
    tag = hashlib.sha256(np.ascontiguousarray(pool, dtype="<i8").tobytes()).hexdigest()[:10]
    if not (RESULTS.parent / cfg["paths"]["sequences_dir"] / f"exp3rel_common_{tag}.npy").exists():
        raise FileNotFoundError(f"cached base block exp3rel_common_{tag} is missing; it would be redrawn, not reused")
    bases = cached_base_block("exp3rel_common", cfg["bases"]["common"], stim_ids, cfg["n_bases"], cfg["seq_len"],
                              cfg["seed"], cfg["bos_token_id"], RESULTS.parent / cfg["paths"]["sequences_dir"])
    set_determinism(cfg["seed"])
    model = load_model("transformer_lens", cfg["backends"]["transformer_lens"]["model_name"], cfg["device"], cfg["dtype"])

    def mean_attention(rows: np.ndarray, i: int, partners: np.ndarray, run_cfg: dict) -> np.ndarray:
        seqs = np.repeat(rows, len(partners), axis=0)  # array index = model position - 1
        seqs[:, i - 1] = np.tile(stim["country"], len(rows))
        seqs[:, 31] = np.tile(partners, len(rows))
        return attention(model, seqs, run_cfg).mean(axis=0)

    out = {f"exp9/{k}": mean_attention(bases, 31, stim[k], cfg) for k in ("capital", "city")}
    rows = bases[: cfgs["exp10"]["n_rows"]]
    out |= {f"exp10/{i}": mean_attention(rows, i, stim["capital"], cfgs["exp10"]) for i in (1, 16, 31)}
    np.savez_compressed(CACHE, **out)
    print(f"saved {CACHE}")
    return out

CHIP = {  # token chips as in the one-pagers: BOS, random filler, country, capital (final token), elided gap
    "bos": dict(color="white", bbox=dict(boxstyle="square,pad=0.18", fc="#52514E", ec="#52514E", lw=0.4)),
    "fill": dict(color="#898781", bbox=dict(boxstyle="square,pad=0.18", fc="white", ec="#C3C2B7", lw=0.4)),
    "country": dict(color="black", bbox=dict(boxstyle="square,pad=0.18", fc="#DCEAFB", ec="#2A78D6", lw=0.5)),
    "capital": dict(color="black", fontweight="bold", bbox=dict(boxstyle="square,pad=0.18", fc="white", ec="#EB6834", lw=0.6)),
    "gap": dict(color="#898781"),
}


def chip_row(fig, x0: float, y: float, tokens: list[tuple[str, str]], size: float = 5.6) -> None:
    """Draw a token sequence as chips starting at figure coordinates (x0, y)."""
    renderer = fig.canvas.get_renderer()
    gap_px = 0.25 * size * fig.dpi / 72  # even space between chips
    x = x0
    for tok, kind in tokens:
        t = fig.text(x, y, tok, fontsize=size, family="monospace", va="center", ha="left", **CHIP[kind])
        t.draw(renderer)  # lays out the box so its true right edge can be measured
        box = t.get_bbox_patch()
        right = (box.get_window_extent(renderer) if box is not None else t.get_window_extent(renderer)).x1
        x = (right + gap_px) / fig.bbox.width


def sequences(i: int, country: str, final: str = "Paris") -> list[tuple[str, str]]:
    """The first common-filler row with the country at position i and `final` as the final token (abbreviated)."""
    if i == 1:
        body = [(country, "country"), ("sc", "fill"), ("sa", "fill"), ("…", "gap"), ("dis", "fill")]
    elif i == 16:
        body = [("she", "fill"), ("…", "gap"), ("ors", "fill"), (country, "country"), ("ign", "fill"), ("…", "gap"),
                ("dis", "fill")]
    else:
        body = [("she", "fill"), ("sc", "fill"), ("…", "gap"), ("age", "fill"), (country, "country")]
    return [("BOS", "bos")] + body + [(final, "capital")]


def head_maps(panels: list[tuple[str, np.ndarray, list[tuple[str, str]]]], path) -> None:
    """One row of 'what each head attends to most' maps in three columns, the legend in the next free column.
    panels: (title, categories [L, H], the sequence drawn above the map)."""
    present = [k for k in COLORS if any((c == k).any() for _, c, _ in panels)]
    code = {k: n for n, k in enumerate(present)}
    cmap = ListedColormap([COLORS[k] for k in present])
    fig = plt.figure(figsize=(7.6, 2.9))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.5], left=0.065, right=0.985, top=0.755, bottom=0.1,
                          wspace=0.2)
    for k, (title, cats, tokens) in enumerate(panels):
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(np.vectorize(code.get)(cats), origin="lower", cmap=cmap, vmin=-0.5, vmax=len(present) - 0.5)
        ax.set_title(title, fontsize=7, pad=7)
        style_head_grid(ax, xsize=5.0, ysize=5.5, ylabel=k == 0)
        ax.apply_aspect()  # the square map's true left edge, so the sequence lines up with it
        chip_row(fig, ax.get_position().x0, 0.87, tokens)
    legend_ax = fig.add_subplot(gs[0, len(panels):])
    legend_ax.axis("off")
    legend_ax.legend(handles=[Patch(facecolor=COLORS[k], edgecolor="0.6", label=k) for k in present], loc="center left",
                     frameon=False, fontsize=6, handlelength=1.0)
    fig.text(0.065, 0.955, "What each head attends to most", fontsize=7.5, fontweight="bold")
    fig.savefig(path, dpi=250)
    plt.close(fig)
    print(f"saved {path}")


def exp9(maps: dict[str, np.ndarray]) -> None:
    keys = {31: "Country", 32: "Capital / city"}
    panels = []
    for title, name, final in (("country + its capital", "capital", "Paris"),
                               ("country + one of its cities", "city", "Lyon")):
        c = column("", maps[f"exp9/{name}"], keys=keys)
        panels.append((title, most_attended(c["attn"], c["cat"]), sequences(31, "France", final)))
    head_maps(panels, OUT / "exp9_strip.png")


def exp10(maps: dict[str, np.ndarray]) -> None:
    panels = []
    for i in (1, 16, 31):
        c = column("", maps[f"exp10/{i}"], keys={i: "Country", 32: "Capital / city"})
        panels.append((f"country at {i}" + (" (adjacent)" if i == 31 else ""), most_attended(c["attn"], c["cat"]),
                       sequences(i, "France")))
    head_maps(panels, OUT / "exp10_fig.png")


if __name__ == "__main__":
    maps = correct_maps()
    exp9(maps)
    exp10(maps)
