"""Experiment 4b: controlled difference with the capital as the final token.

Run:  uv run --offline python -m exp4.run_capital --config configs/exp4_capital.yaml

As exp4.run_controlled, but the final token (32) is a capital in every version of a pair: the cell country's own
capital ("own") or the capital a fixed derangement assigns it ("wrong"). The pair still differs only at tau.
Fact contrast = D_own - D_wrong per row (the controls are the same 11 sequences in both conditions and cancel).
See exp4/PREDICTIONS_capital.md. Writes results/<timestamp>_exp4cap/: config, predictions, versions, arrays, checks,
verdicts, figures, viz_data.json (all three final-token conditions, for exp4.build_viz) and exp4_capital_report.md.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # write figures to files, never open a window
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.colors import TwoSlopeNorm

from exp1.backends import load_model, set_determinism
from exp1.runlog import make_run_dir, save_config_copy, save_versions
from exp1ext.effects import bootstrap_mean_per_unit
from exp1ext.features import token_features
from exp2.design import cached_base_block, matched_pools
from exp3.run_positions import latin_derangements
from exp3.run_relations import PROJECT_ROOT, attention, log, stimuli
from exp4.run_controlled import (ARMS, LABELS, analyse, contrast_stats, extract, head_name, placebo_words, row_differences,
                                 span_mean, variants)

CONDS = ("own", "wrong")
COND_LABELS = {"own": "its capital", "wrong": "a wrong capital", "random": "a random token"}
GREEN, GREY, BLUE = "#1baf7a", "#898781", "#2a78d6"


def late_bos(ext: dict, layers: list[int]) -> np.ndarray:
    """Per-row late-layer mean BOS mass with the country at tau, [R, D]."""
    return ext["country"]["masses"][:, :, layers[0]:layers[1] + 1, :, 1].astype(np.float64).mean(axis=(2, 3))


def random_rows(cfg: dict, rows: int) -> dict[str, dict[str, np.ndarray]]:
    """Exp 4's per-row contrasts (random final token) for the same first `rows` rows and the same countries."""
    z = np.load(PROJECT_ROOT / cfg["random_run"] / "controlled.npz")
    out = {}
    for b in cfg["bases"]:
        ext = {"control": z[f"{b}/control"][:rows], **{arm: {"masses": z[f"{b}/{arm}_masses"][:rows]} for arm in ARMS}}
        out[b] = row_differences(ext, cfg["seq_len"])
    return out


def analyse_fact(res: dict, ext: dict, x_random: dict, cfg: dict) -> dict:
    """Per filler type: the fact contrast (own - wrong) and own - random on the attention to tau, and the late BOS fact."""
    a = cfg["analysis"]
    boot = a["bootstrap"]
    out = {"fact": contrast_stats(res["own"]["x"]["country"] - res["wrong"]["x"]["country"], a),
           "vs_random": contrast_stats(res["own"]["x"]["country"] - x_random["country"], a)}
    bos = late_bos(ext["own"], a["late_layers"]) - late_bos(ext["wrong"], a["late_layers"])  # [R, D]
    out["bos"] = dict(zip(("mean", "lo", "hi"), bootstrap_mean_per_unit(bos, boot["iters"], boot["seed"], 0.95)))
    out["bos_far"] = [float(v) for v in bootstrap_mean_per_unit(span_mean(bos, a["far"]), boot["iters"], boot["seed"], 0.95)]
    out["bos_rows"] = bos
    return out


def check_predictions(res: dict, fact: dict, cfg: dict) -> list[dict]:
    """res[(cond, b)]: exp4.run_controlled.analyse output; fact[b]: analyse_fact output. Each must hold in every filler."""
    a = cfg["analysis"]
    prev_heads = [(l, h) for l, h, src in cfg["focus_heads"] if src == "Exp 1 previous-token"]
    l11, l4 = (11, 8), (4, 11)
    parts: dict[str, list[tuple[bool, str]]] = {k: [] for k in ("F1", "F2", "F3", "F4", "F5", "F6")}
    for b in cfg["bases"]:
        f, st = fact[b], fact[b]["fact"]
        bos = f["bos"]
        parts["F1"].append((bos["hi"][0] < 0, f"{b}: {bos['mean'][0]:+.4f} [{bos['lo'][0]:+.4f}, {bos['hi'][0]:+.4f}]"))
        m, lo, hi = f["bos_far"]
        parts["F2"].append((hi < 0, f"{b}: {m:+.4f} [{lo:+.4f}, {hi:+.4f}]"))
        parts["F3"].append((bool(st["sig"][0][l11]) and st["mean"][0][l11] > 0,
                            f"{b}: {st['mean'][0][l11]:+.4f} [{st['lo'][0][l11]:+.4f}, {st['hi'][0][l11]:+.4f}], "
                            f"q {st['q'][0][l11]:.2g}"))
        pos = st["sig"] & (st["mean"] > 0)
        counts = {head_name(l, h): int(pos[1:, l, h].sum()) for l, h in a["global_heads"]}
        parts["F4"].append((sum(c >= a["global_min"] for c in counts.values()) >= 2,
                            f"{b}: " + ", ".join(f"{k} {v}/30" for k, v in counts.items())))
        sigs = {"own": res[("own", b)]["stats"]["country"]["sig"], "wrong": res[("wrong", b)]["stats"]["country"]["sig"],
                "fact": st["sig"]}
        hit = [k for k, s in sigs.items() if s[1:, l4[0], l4[1]].any()]
        parts["F5"].append((not hit, f"{b}: meaningful at d ≥ 2 in: {', '.join(hit) or 'none'}"))
        so = sigs["own"]
        far = [f"{head_name(l, h)} (d {int(np.nonzero(so[:, l, h])[0].max()) + 1})" for l, h in prev_heads
               if so[a["local_max_d"]:, l, h].any()]
        parts["F6"].append((not far, f"{b}: beyond d = {a['local_max_d']}: {', '.join(far) or 'none'}"))
    return [{"id": k, "verdict": "PASS" if all(ok for ok, _ in v) else "FAIL", "detail": "; ".join(d for _, d in v)}
            for k, v in parts.items()]


def cross_checks(tl, hf, bases: dict, fills: dict, finals: dict, ext: dict, cfg: dict) -> dict:
    c, T = cfg["checks"], cfg["seq_len"]
    out = {}
    for b, rows in bases.items():
        k = c["hf_rows"]
        worst = 0.0
        for tau in c["hf_taus"]:
            seqs = variants(rows[:k], fills[b]["country"][:k], tau, finals[("own", b)][:k]).reshape(-1, T)
            worst = max(worst, float(np.abs(attention(tl, seqs, cfg) - attention(hf, seqs, cfg)).max()))
        r = c["determinism_rows"]
        again = attention(tl, variants(rows[:r], fills[b]["country"][:r], T - 1, finals[("own", b)][:r]).reshape(-1, T), cfg)
        again = again.reshape((r, -1) + again.shape[1:])[..., T - 1].mean(axis=1)
        det = float(np.abs(again - ext[("own", b)]["country"]["masses"][:r, 0, ..., 0]).max())
        same = float(np.abs(ext[("own", b)]["control"] - ext[("wrong", b)]["control"]).max())
        out[b] = {"hf_max_abs_delta": worst, "determinism_max_abs_delta": det, "control_own_vs_wrong": same}
    return out


def plot_conditions(res: dict, z_random: Any, cfg: dict, b: str, path: Path) -> None:
    """Focus heads: D(d) with the own capital (band = 95% CI), a wrong capital and a random final token (Exp 4)."""
    heads, T, margin = cfg["focus_heads"], cfg["seq_len"], cfg["analysis"]["margin"]
    d = np.arange(1, T)
    cols = 4
    fig, axes = plt.subplots(int(np.ceil(len(heads) / cols)), cols, figsize=(13, 2.5 * np.ceil(len(heads) / cols)),
                             sharex=True, squeeze=False)
    own, wrong = res[("own", b)]["stats"]["country"], res[("wrong", b)]["stats"]["country"]
    rnd = z_random[f"{b}/country/mean"]
    for ax, (l, h, src) in zip(axes.flat, heads):
        ax.axhspan(-margin, margin, color="0.93", lw=0)
        ax.axhline(0, color="0.6", lw=0.7)
        ax.plot(d, rnd[:, l, h], color=BLUE, lw=1.0, ls="--", label="random final token (Exp 4)")
        ax.plot(d, wrong["mean"][:, l, h], color=GREY, lw=1.3, label="wrong capital")
        ax.fill_between(d, own["lo"][:, l, h], own["hi"][:, l, h], color=GREEN, alpha=0.2, lw=0)
        ax.plot(d, own["mean"][:, l, h], color=GREEN, lw=1.7, label="its capital")
        ax.set_title(f"{head_name(l, h)} ({src})\n{res[('own', b)]['labels'][l, h]}", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes.flat[len(heads):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("distance T − τ (1 = adjacent)", fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("Δ attention to τ", fontsize=7)
    axes.flat[0].legend(fontsize=6, frameon=False)
    fig.suptitle(f"{b} filler: attention from the final token to the country, country − original token, by final token",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_fact(fact: dict, cfg: dict, path: Path) -> None:
    """Fact contrast (own − wrong capital) for the focus heads x distance, and the late BOS fact by distance."""
    T = cfg["seq_len"]
    heads = [(l, h) for l, h, _ in cfg["focus_heads"]]
    bases = list(fact)
    grids = {b: np.stack([fact[b]["fact"]["mean"][:, l, h] for l, h in heads]) for b in bases}
    vmax = max(float(np.abs(g).max()) for g in grids.values())
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    fig, axes = plt.subplots(2, len(bases), figsize=(6 * len(bases), 8.2), squeeze=False,
                             gridspec_kw={"height_ratios": [2.2, 1]})
    for c, b in enumerate(bases):
        ax = axes[0, c]
        im = ax.imshow(grids[b], aspect="auto", cmap="RdBu_r", norm=norm, extent=(0.5, T - 0.5, len(heads) - 0.5, -0.5))
        sig = np.stack([fact[b]["fact"]["sig"][:, l, h] for l, h in heads])
        r, k = np.nonzero(sig)
        ax.scatter(k + 1, r, s=3, color="black", linewidths=0)
        ax.set_yticks(range(len(heads)))
        ax.set_yticklabels([head_name(l, h) for l, h in heads], fontsize=7)
        ax.set_title(f"{b}: fact contrast on attention to τ (its capital − wrong capital)", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        ax2 = axes[1, c]
        bos, d = fact[b]["bos"], np.arange(1, T)
        ax2.axhline(0, color="0.6", lw=0.7)
        ax2.fill_between(d, bos["lo"], bos["hi"], color=GREEN, alpha=0.2, lw=0)
        ax2.plot(d, bos["mean"], color=GREEN, lw=1.6)
        ax2.set_xlabel("distance T − τ", fontsize=8)
        ax2.set_ylabel("late BOS mass, its − wrong capital", fontsize=8)
        ax2.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_layers(res: dict, cfg: dict, path: Path) -> None:
    T, bases = cfg["seq_len"], list(cfg["bases"])
    fig, axes = plt.subplots(1, len(bases), figsize=(5.6 * len(bases), 3.8), squeeze=False)
    vmax = max(float(np.abs(res[("own", b)]["layer"]["mean"]).max()) for b in bases)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    for ax, b in zip(axes[0], bases):
        m = res[("own", b)]["layer"]["mean"]
        im = ax.imshow(m.T, aspect="auto", origin="lower", cmap="RdBu_r", norm=norm, extent=(0.5, T - 0.5, -0.5, m.shape[1] - 0.5))
        ax.set_xlabel("distance T − τ", fontsize=8)
        ax.set_ylabel("layer", fontsize=8)
        ax.set_title(f"{b}: mean Δ attention to τ over each layer's heads, its capital at T", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def r3(x: Any) -> list:
    return np.round(np.asarray(x, dtype=np.float64), 3).tolist()


def viz_data(res: dict, ext: dict, bases: dict, fills: dict, finals_c: dict, cfg: dict, tokenizer) -> dict:
    """The viewers' data for all three final-token conditions. Dataset key '<filler>/<condition>'; the random condition
    is Exp 4's (copied from its viz_data.json). Per-distance arrays are indexed d - 1, as in exp4.run_controlled."""
    old = json.loads((PROJECT_ROOT / cfg["random_run"] / "viz_data.json").read_text())
    om = old["meta"]
    keys = [f"{b}/{cond}" for b in bases for cond in (*CONDS, "random")]
    meta = {k: om[k] for k in ("seq_len", "query_position", "distances", "n_rows", "n_countries", "margin", "country_example")}
    meta |= {"fillers": keys, "filler_types": list(bases), "conditions": [*CONDS, "random"], "condition_labels": COND_LABELS,
             "focus_heads": [[l, h, src] for l, h, src in cfg["focus_heads"]],
             "filler_labels": {}, "example_tokens": {}, "word_example": {}, "final_example": {}}
    data = {}
    for b in bases:
        tokens = om["example_tokens"][b]
        for cond in (*CONDS, "random"):
            k = f"{b}/{cond}"
            final = tokens[-1] if cond == "random" else tokenizer.decode([int(finals_c[cond][0])])
            meta["filler_labels"][k] = f"{om['filler_labels'][b]}; final token = {COND_LABELS[cond]}"
            meta["example_tokens"][k] = tokens[:-1] + [final]
            meta["word_example"][k] = om["word_example"][b]
            meta["final_example"][k] = final
            if cond == "random":
                od = old["data"][b]
                data[k] = {"control": r3(od["control"]), "country": r3(od["country"]), "word": r3(od["word"]),
                           "diff": {arm: {s: r3(od["diff"][arm][s]) for s in ("mean", "lo", "hi")} | {"sig": od["diff"][arm]["sig"]}
                                    for arm in ARMS},
                           "prev_score": r3(od["prev_score"]), "label": od["label"],
                           "layer_mean": {s: r3(od["layer_mean"][s]) for s in ("mean", "lo", "hi")}}
                continue
            r, e = res[(cond, b)], ext[(cond, b)]
            data[k] = {"control": r3(e["control"].mean(axis=0)), **{arm: r3(e[arm]["by_key"]) for arm in ARMS},
                       "diff": {arm: {s: r3(np.moveaxis(r["stats"][arm][s], 0, -1)) for s in ("mean", "lo", "hi")}
                                | {"sig": np.moveaxis(r["stats"][arm]["sig"], 0, -1).astype(int).tolist()} for arm in ARMS},
                       "prev_score": r3(r["prev_score"]), "label": r["labels"].tolist(),
                       "layer_mean": {s: r3(r["layer"][s].T) for s in ("mean", "lo", "hi")}}
    return {"meta": meta, "data": data}


def write_report(run: Path, cfg: dict, res: dict, fact: dict, z_random: Any, checks: dict, preds: list[dict]) -> None:
    a, T = cfg["analysis"], cfg["seq_len"]
    L = ["# Experiment 4b: controlled difference with the capital as the final token", "",
         f"Run `{run.name}`. {cfg['n_rows']} rows per filler type; per row, τ = 1..{T - 1} and 11 countries, with each "
         f"country's own capital (own) or a deranged capital (wrong) at {T} in both versions of the pair. D = attention "
         f"from the final token to τ, country − original token. Fact contrast = D_own − D_wrong. Meaningful = BH q ≤ "
         f"{a['q']} over the {T - 1} × 144 cells and |D| ≥ {a['margin']}. Random-final-token numbers are Exp 4's "
         f"(`{cfg['random_run']}`), same rows and countries.", "", "## Checks", "", "| check | result |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in checks.items()]
    L += ["", "## Pre-registered predictions", "", "| id | verdict | detail |", "|---|---|---|"]
    L += [f"| {p['id']} | {p['verdict']} | {p['detail']} |" for p in preds]
    L += ["", "## Focus heads", "",
          "| head | filler | prev (own) | D_own(1) | D_wrong(1) | D_random(1) | fact(1) [95% CI] | fact d 2–4 | fact d 16–31 | "
          "positive fact d ≥ 2 | D_own d 16–31 | label (own) |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for l, h, _ in cfg["focus_heads"]:
        for b in cfg["bases"]:
            o, w, f = res[("own", b)]["stats"]["country"], res[("wrong", b)]["stats"]["country"], fact[b]["fact"]
            rnd = z_random[f"{b}/country/mean"]
            pos = int((f["sig"] & (f["mean"] > 0))[1:, l, h].sum())
            nr, fr = a["near"], a["far"]
            L.append(f"| {head_name(l, h)} | {b} | {res[('own', b)]['prev_score'][l, h]:.3f} | {o['mean'][0, l, h]:+.4f} | "
                     f"{w['mean'][0, l, h]:+.4f} | {rnd[0, l, h]:+.4f} | {f['mean'][0, l, h]:+.4f} [{f['lo'][0, l, h]:+.4f}, "
                     f"{f['hi'][0, l, h]:+.4f}] | {f['mean'][nr[0] - 1:nr[1], l, h].mean():+.4f} | "
                     f"{f['mean'][fr[0] - 1:fr[1], l, h].mean():+.4f} | {pos}/{T - 2} | "
                     f"{o['mean'][fr[0] - 1:fr[1], l, h].mean():+.4f} | {res[('own', b)]['labels'][l, h]} |")
    L += ["", "## Label counts with its capital at T (all 144 heads)", "", "| filler | " + " | ".join(LABELS) + " |",
          "|---" * (len(LABELS) + 1) + "|"]
    for b in cfg["bases"]:
        lab = res[("own", b)]["labels"]
        L.append(f"| {b} | " + " | ".join(str(int((lab == x).sum())) for x in LABELS) + " |")
    L += ["", "## Strongest fact contrasts (mean over d 2–31)", "", "| filler | head | mean fact d 2–31 | fact(1) | "
          "positive fact d ≥ 2 |", "|---|---|---|---|---|"]
    for b in cfg["bases"]:
        f = fact[b]["fact"]
        mean = f["mean"][1:].mean(axis=0)
        for idx in np.argsort(-np.abs(mean), axis=None)[:10]:
            l, h = np.unravel_index(idx, mean.shape)
            pos = int((f["sig"] & (f["mean"] > 0))[1:, l, h].sum())
            L.append(f"| {b} | {head_name(l, h)} | {mean[l, h]:+.4f} | {f['mean'][0, l, h]:+.4f} | {pos}/{T - 2} |")
    L += ["", "## Late BOS fact (layers {}–{}): its capital − wrong capital, country at τ".format(*a["late_layers"]), "",
          "| filler | d = 1 | d = 2 | d = 4 | d = 8 | d = 16 | d = 31 | mean d 16–31 |", "|---|---|---|---|---|---|---|---|"]
    for b in cfg["bases"]:
        bos = fact[b]["bos"]
        cells = " | ".join(f"{bos['mean'][d - 1]:+.4f} [{bos['lo'][d - 1]:+.4f}, {bos['hi'][d - 1]:+.4f}]" for d in (1, 2, 4, 8, 16, 31))
        m, lo, hi = fact[b]["bos_far"]
        L.append(f"| {b} | {cells} | {m:+.4f} [{lo:+.4f}, {hi:+.4f}] |")
    L += ["", "## Files", "",
          "- `capital.npz`: per condition and filler type, the stats of each contrast [distance, layer, head], prev_score, "
          "labels, mean attention rows, per-row arm masses (float16) and control (float16, mean over the row's 11 "
          "controls); per filler type the fact and own − random stats and the per-row late BOS fact.",
          "- `figures/focus_<filler>.png`: D(d) for the focus heads with its capital, a wrong capital and a random final token.",
          "- `figures/fact.png`: fact contrast for the focus heads, and the late BOS fact by distance.",
          "- `figures/layers_own.png`: mean D over each layer's heads with its capital at T.",
          "- `viz_data.json`: input of `exp4.build_viz`, with all three final-token conditions.", ""]
    (run / "exp4_capital_report.md").write_text("\n".join(L))


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Experiment 4b, capital as the final token")
    parser.add_argument("--config", default="configs/exp4_capital.yaml", type=Path)
    parser.add_argument("--rows", default=None, type=int, help="use only the first rows (smoke test)")
    parser.add_argument("--results-dir", default=None, type=Path, help="override paths.results_dir (smoke tests)")
    args = parser.parse_args(argv)
    cfg_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    cfg = yaml.safe_load(cfg_path.read_text())
    if args.rows:
        cfg["n_rows"] = args.rows
    t0 = time.time()
    run = make_run_dir(args.results_dir or PROJECT_ROOT / cfg["paths"]["results_dir"], tag="exp4cap")
    save_config_copy(cfg_path, run)
    shutil.copy2(PROJECT_ROOT / "exp4" / "PREDICTIONS_capital.md", run / "PREDICTIONS.md")
    save_versions(run, PROJECT_ROOT, code_dirs=("exp1", "exp2", "exp3", "exp4"))
    (run / "figures").mkdir()
    log(f"run folder {run}", t0)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg["figures"]["tokenizer"])
    set_determinism(cfg["seed"])
    tl = load_model("transformer_lens", cfg["backends"]["transformer_lens"]["model_name"], cfg["device"], cfg["dtype"])
    countries, capitals, cities = stimuli(cfg, tokenizer)
    n = len(countries)
    stim_ids = np.concatenate([countries, capitals, cities, cfg["demonstration_ids"]])
    lo = min(r[0] for r in cfg["token_classes"].values())
    hi = max(r[1] for r in cfg["token_classes"].values())
    pools = matched_pools(countries, token_features(np.arange(lo, hi + 1), tokenizer), cfg["match_features"],
                          cfg["token_classes"])
    pools = [p[~np.isin(p, stim_ids)] for p in pools]
    wrong = latin_derangements(n, 1, cfg["wrong_capital_seed"])[0]
    if (wrong == np.arange(n)).any():
        raise AssertionError("the wrong-capital derangement has a fixed point")
    finals_c = {"own": capitals, "wrong": capitals[wrong]}  # final token per country
    log("wrong capitals: " + ", ".join(f"{tokenizer.decode([int(c)]).strip()}→{tokenizer.decode([int(k)]).strip()}"
                                       for c, k in zip(countries, finals_c["wrong"])), t0)

    bases, fills, finals, ext, res, fact = {}, {}, {}, {}, {}, {}
    x_random = random_rows(cfg, cfg["n_rows"])
    for b, rng_ in cfg["bases"].items():
        block = cached_base_block(f"exp3rel_{b}", rng_, stim_ids, cfg["base_cache_n"], cfg["seq_len"], cfg["seed"],
                                  cfg["bos_token_id"], PROJECT_ROOT / cfg["paths"]["sequences_dir"])
        bases[b] = block[: cfg["n_rows"]]
        if np.isin(bases[b], stim_ids).any():
            raise AssertionError(f"{b}: a stimulus or demonstration id occurs in a base row")
        R = len(bases[b])
        fills[b] = {"country": np.tile(countries, (R, 1)), "word": placebo_words(bases[b], pools, cfg["word_seed"])}
        for cond in CONDS:
            finals[(cond, b)] = np.tile(finals_c[cond], (R, 1))
            ext[(cond, b)] = extract(tl, bases[b], fills[b], cfg, t0, f"{b}/{cond}", final=finals[(cond, b)])
            res[(cond, b)] = analyse(ext[(cond, b)], cfg)
        fact[b] = analyse_fact({c: res[(c, b)] for c in CONDS}, {c: ext[(c, b)] for c in CONDS}, x_random[b], cfg)
        log(f"{b}: analysis done", t0)

    hf = load_model("huggingface", cfg["backends"]["huggingface"]["model_name"], cfg["device"], cfg["dtype"],
                    cfg["backends"]["huggingface"]["attn_implementation"])
    cross = cross_checks(tl, hf, bases, fills, finals, ext, cfg)
    log("cross-library and determinism checks done", t0)

    c = cfg["checks"]
    checks = {"design (ids at τ and 32, control row elsewhere, no repeats/BOS/stimulus ids)": "PASS (asserted per τ, arm, condition)",
              "attention rows sum to 1": f"PASS (atol {c['row_sum_atol']:g}, asserted per batch)",
              "wrong-capital derangement has no fixed point": "PASS"}
    for b, x in cross.items():
        checks[f"determinism, {b}"] = f"{'PASS' if x['determinism_max_abs_delta'] == 0 else 'FAIL'} (max |Δ| {x['determinism_max_abs_delta']:.1e})"
        checks[f"TL vs HF, {b}"] = f"{'PASS' if x['hf_max_abs_delta'] < c['hf_atol'] else 'FAIL'} (max |Δ| {x['hf_max_abs_delta']:.1e})"
        checks[f"controls identical own vs wrong, {b}"] = f"{'PASS' if x['control_own_vs_wrong'] < 1e-6 else 'FAIL'} (max |Δ| {x['control_own_vs_wrong']:.1e})"
    preds = check_predictions(res, fact, cfg)

    z_random = np.load(PROJECT_ROOT / cfg["random_run"] / "controlled.npz")
    np.savez_compressed(run / "capital.npz",
                        **{f"{cond}/{b}/{con}/{k}": res[(cond, b)]["stats"][con][k] for cond in CONDS for b in bases
                           for con in res[(cond, b)]["stats"] for k in ("mean", "lo", "hi", "p", "q", "sig")},
                        **{f"{cond}/{b}/prev_score": res[(cond, b)]["prev_score"] for cond in CONDS for b in bases},
                        **{f"{cond}/{b}/labels": res[(cond, b)]["labels"].astype(str) for cond in CONDS for b in bases},
                        **{f"{cond}/{b}/{arm}_by_key": ext[(cond, b)][arm]["by_key"] for cond in CONDS for b in bases for arm in ARMS},
                        **{f"{cond}/{b}/{arm}_masses": ext[(cond, b)][arm]["masses"].astype(np.float16) for cond in CONDS
                           for b in bases for arm in ARMS},
                        **{f"{cond}/{b}/control": ext[(cond, b)]["control"].astype(np.float16) for cond in CONDS for b in bases},
                        **{f"{b}/{con}/{k}": fact[b][con][k] for b in bases for con in ("fact", "vs_random")
                           for k in ("mean", "lo", "hi", "p", "q", "sig")},
                        **{f"{b}/bos_fact_rows": fact[b]["bos_rows"] for b in bases},
                        wrong=wrong)
    (run / "checks.json").write_text(json.dumps(checks, indent=2))
    (run / "predictions.json").write_text(json.dumps(preds, indent=2))
    (run / "viz_data.json").write_text(json.dumps(viz_data(res, ext, bases, fills, finals_c, cfg, tokenizer),
                                                  separators=(",", ":")))
    for b in bases:
        plot_conditions(res, z_random, cfg, b, run / "figures" / f"focus_{b}.png")
    plot_fact(fact, cfg, run / "figures" / "fact.png")
    plot_layers(res, cfg, run / "figures" / "layers_own.png")
    write_report(run, cfg, res, fact, z_random, checks, preds)
    log(f"done: {run / 'exp4_capital_report.md'}", t0)
    for k, v in checks.items():
        print(f"{k}: {v}")
    for p in preds:
        print(f"{p['id']}: {p['verdict']}  {p['detail']}")
    return run


if __name__ == "__main__":
    main()
