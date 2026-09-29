"""Stage 2: per-token sweep — how each head's metric changes with the token at the final position.

~2000 tokens (all of 0-999, plus rare-range tokens: uniform, surface-matched to 256-999, and a 1000-2999 transition
band) are placed at the swept position of K rare and K common fixed backgrounds. A token that already occurs in a
background is marked missing for it (never duplicated). Per background, values are taken relative to that
background's mean over the rare_uniform tokens, so a token effect of 0 means "behaves like a typical rare token";
token effects come from an additive token + background fit that handles the missing cells.

Statistics avoid per-token p-value hunting: one omnibus within-background permutation test per head (BH-FDR across
heads), background split-half reliability of each head's token profile, and the top / bottom tokens per head
re-measured on fresh backgrounds (the re-measured values are the ones to report). Token features (word start, length,
alphabetic, capitalised, digit, log id, embedding norm) are related to the effects by a regression with a two-way
(token x background) bootstrap; rarity x word start is compared at matched surface form.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from exp1.compare import bh_fdr
from exp1.sequences import derive_seed, prepend_bos
from exp1.validation import md_table, verdict
from exp1extended import measures
from exp1extended.config import head_group, head_name, target_heads
from exp1extended.design import build_sweep, load_or_create_array, sample_pool, sweep_tokens
from exp1extended.extract import run_final
from exp1extended.plots import plot_feature_forest, plot_rarity_wordstart, plot_token_heatmap, plot_token_profiles
from exp1extended.stats import split_half_reliability, twoway_effects, within_background_permutation_p
from exp1extended.tokens import display_text, token_features

METRIC_FNS = {"sink_logit": measures.sink_logit, "prev_logit": measures.prev_logit, "other_gap": measures.other_gap}
FEATURES = ("word_start", "is_alpha", "is_cap", "is_digit", "char_len", "log_id", "embed_norm")
CONTINUOUS = ("char_len", "log_id", "embed_norm")  # scaled per SD; binary features stay 0/1


def _seed(cfg: dict[str, Any], name: str) -> int:
    return derive_seed(cfg["seed"], f"exp1extended/{name}")


def _backgrounds(cfg, root: Path, prefix: str, n_per_type: int) -> tuple[np.ndarray, np.ndarray]:
    """n_per_type rare then n_per_type common backgrounds (cached), and each one's type."""
    data_dir = root / cfg["paths"]["sequences_dir"]
    out = []
    for kind in ("rare", "common"):
        name = f"{prefix}_bg_{kind}"
        seqs = sample_pool(tuple(cfg["pools"][kind]), n_per_type, cfg["seq_len"], _seed(cfg, f"stage2/{name}"))
        out.append(load_or_create_array(f"stage2_{name}", seqs, data_dir,
                                        {"seed": cfg["seed"], "pool": cfg["pools"][kind], "n": n_per_type}))
    return np.concatenate(out), np.array(["rare"] * n_per_type + ["common"] * n_per_type)


def _grid(res: dict[str, np.ndarray], tok_idx, bg_idx, n_tok: int, n_bg: int) -> dict[str, np.ndarray]:
    """Target-head primary metrics arranged [tokens, backgrounds, heads]; missing cells are 0 (masked downstream)."""
    out = {}
    for metric, fn in METRIC_FNS.items():
        values = fn(res["target_scores"])  # [n_rows, heads]
        grid = np.zeros((n_tok, n_bg, values.shape[1]))
        grid[tok_idx, bg_idx] = values
        out[metric] = grid
    return out


def token_effects(values: np.ndarray, missing: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """values [T, B, H]; baseline bool [T] marks the rare_uniform tokens. Each background is taken relative to its
    mean over the observed baseline tokens, then an additive token + background fit; returns [T, H] token effects
    with the baseline tokens' mean effect at 0."""
    observed = ~missing
    base_mask = observed & baseline[:, None]
    base = (values * base_mask[..., None]).sum(0) / base_mask.sum(0)[:, None]  # [B, H]
    eff, _, _ = twoway_effects(values - base[None], missing)
    return eff - eff[baseline].mean(0)


def _design_matrix(feats: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    cols = []
    for f in FEATURES:
        x = feats[f].to_numpy(dtype=float)
        if f in CONTINUOUS:
            x = (x - x.mean()) / x.std()
        cols.append(x)
    return np.column_stack([np.ones(len(feats)), *cols]), ["intercept", *FEATURES]


def bootstrap_features(values, missing, groups: np.ndarray, feats: pd.DataFrame, iters: int, seed: int, ci: float):
    """Two-way bootstrap (tokens and backgrounds resampled independently) of (a) the feature regression of token
    effects on ids >= 256 and (b) the rarity x word-start 2x2 at matched surface form (common vs rare_matched).
    Returns point estimates and percentile CIs as long tables."""
    keep = groups != "byte"
    v, m, g, f = values[keep], missing[keep], groups[keep], feats[keep].reset_index(drop=True)
    X, names = _design_matrix(f)
    ws = f["word_start"].to_numpy(dtype=bool)

    def stats(vv, mm, gg, XX, wws):
        eff = token_effects(vv, mm, gg == "rare_uniform")
        beta = np.linalg.lstsq(XX, eff, rcond=None)[0]  # [features + 1, H]
        cells = {}
        for label, w in (("word-start", True), ("continuation", False)):
            common = eff[(gg == "common") & (wws == w)].mean(0)
            rare = eff[(gg == "rare_matched") & (wws == w)].mean(0)
            cells[f"common − rare_matched, {label} tokens"] = common - rare
        cells["interaction (continuation − word-start)"] = (cells["common − rare_matched, continuation tokens"]
                                                            - cells["common − rare_matched, word-start tokens"])
        return beta, cells

    point_beta, point_cells = stats(v, m, g, X, ws)
    rng = np.random.default_rng(seed)
    boot_beta, boot_cells = [], {k: [] for k in point_cells}
    for _ in range(iters):
        ti = rng.integers(0, len(v), len(v))
        bi = rng.integers(0, v.shape[1], v.shape[1])
        b, c = stats(v[ti][:, bi], m[ti][:, bi], g[ti], X[ti], ws[ti])
        boot_beta.append(b)
        for k in c:
            boot_cells[k].append(c[k])
    tail = (1.0 - ci) / 2.0 * 100.0
    lo_b, hi_b = np.percentile(boot_beta, tail, axis=0), np.percentile(boot_beta, 100 - tail, axis=0)
    cells_ci = {k: (np.percentile(boot_cells[k], tail, axis=0), np.percentile(boot_cells[k], 100 - tail, axis=0))
                for k in point_cells}
    return names, point_beta, lo_b, hi_b, point_cells, cells_ci


def run_sweep(cfg, root, model, heads, backgrounds, bg_type, token_ids, column: int):
    seqs, tok_idx, bg_idx, missing = build_sweep(backgrounds, column, token_ids)
    res = run_final(model, prepend_bos(seqs, cfg["bos_token_id"]), cfg["batch_size"], heads, cfg["diffuse_threshold"])
    return res, tok_idx, bg_idx, missing, seqs


def run_stage2(cfg: dict[str, Any], run_dir: Path, model, root: Path) -> pd.DataFrame:
    s2, s = cfg["stage2"], cfg["stats"]
    heads = target_heads(cfg)
    groups_of = head_group(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg["figures"]["tokenizer"])
    embed = model.W_E.detach().cpu().numpy()
    data_dir = root / cfg["paths"]["sequences_dir"]

    tokens = sweep_tokens(cfg, tokenizer, embed)
    ids = load_or_create_array("stage2_tokens", tokens["id"].to_numpy(dtype=np.int64), data_dir,
                               {"seed": cfg["seed"], "tokens": s2["tokens"], "pools": cfg["pools"]})
    groups = tokens["group"].to_numpy()
    feats = token_features(tokenizer, ids, cfg["matching"]["len_bins"], embed)
    baseline = groups == "rare_uniform"
    backgrounds, bg_type = _backgrounds(cfg, root, "sweep", s2["backgrounds"]["rare"])

    checks, effect_rows, summary_rows, feature_rows, cell_rows, confirm_rows = [], [], [], [], [], []
    for position in s2["positions"]:
        column = position - 1
        res, tok_idx, bg_idx, missing, seqs = run_sweep(cfg, root, model, heads, backgrounds, bg_type, ids, column)
        np.savez_compressed(run_dir / f"stage2_pos{position}.npz", tok_idx=tok_idx, bg_idx=bg_idx, missing=missing,
                            target_scores=res["target_scores"], **{k: res[k] for k in ("sink_logit", "prev_logit", "label")})
        print(f"position {position}: {len(seqs)} sequences, {int(missing.sum())} missing (token, background) cells")

        # V9: a background swept with its own original token at this position reproduces its own run.
        own = _run_plain(cfg, model, heads, backgrounds)
        where = {t: i for i, t in enumerate(ids)}
        row_of = {(t, b): r for r, (t, b) in enumerate(zip(tok_idx, bg_idx))}
        v = cfg["validation"]
        pairs = [(res["target_scores"][row_of[(where[backgrounds[b, column]], b)]], own["target_scores"][b])
                 for b in range(len(backgrounds)) if backgrounds[b, column] in where]
        diffs = [float(np.abs(a - b).max()) for a, b in pairs]
        # Same sequence in a different batch: allow float32 rounding that scales with |score| (see check_qk).
        ok = all(np.allclose(a, b, rtol=v["qk_rtol"], atol=v["qk_atol"]) for a, b in pairs)
        checks.append({"check": f"V9 pos {position}: background swept with its own token reproduces its run",
                       "detail": f"{len(diffs)} backgrounds" + (f", max |diff| {max(diffs):.2e}" if diffs else
                                                                 " (no background's own token is in the sweep set)"),
                       "result": verdict(ok) if diffs else "SKIPPED"})
        n_det = cfg["validation"]["determinism_n"]
        rerun = _run_plain(cfg, model, heads, seqs[:n_det])
        checks.append({"check": f"V7 pos {position}: re-running the first {n_det} sweep rows is bit-identical",
                       "detail": "", "result": verdict(np.array_equal(rerun["target_scores"], res["target_scores"][:n_det]))})
        if any(c["result"] == "FAIL" for c in checks):
            (run_dir / "validation_report.md").write_text(
                "# exp1extended Stage 2: validation\n\n" + md_table(pd.DataFrame(checks)) + "\n")
            raise AssertionError(f"Stage 2 validation failed; see {run_dir / 'validation_report.md'}")

        grid = _grid(res, tok_idx, bg_idx, len(ids), len(backgrounds))
        for metric, values in grid.items():
            pooled = token_effects(values, missing, baseline)
            by_type = {kind: token_effects(values[:, bg_type == kind], missing[:, bg_type == kind], baseline)
                       for kind in ("rare", "common")}
            observed = ~missing
            naive = ((values * observed[..., None]).sum(1) / observed.sum(1)[:, None])
            naive = naive - naive[baseline].mean(0)
            rel = split_half_reliability(values, missing, _seed(cfg, f"stage2/split/{position}/{metric}"))
            perm_p = within_background_permutation_p(values, missing, s["n_permutations"],
                                                     _seed(cfg, f"stage2/perm/{position}/{metric}"))
            perm_q = bh_fdr(perm_p)
            for h, head in enumerate(heads):
                summary_rows.append({"position": position, "metric": metric, "head": head_name(head),
                                     "group": groups_of[head],
                                     "primary": metric in cfg["primary_metrics"][groups_of[head]],
                                     "reliability": rel[h], "omnibus p": perm_p[h], "omnibus q": perm_q[h],
                                     "two-way vs naive r": np.corrcoef(pooled[:, h], naive[:, h])[0, 1],
                                     "SD of token effects": pooled[:, h].std()})
                for kind, eff in (("pooled", pooled), *by_type.items()):
                    effect_rows.append(pd.DataFrame({"position": position, "metric": metric, "head": head_name(head),
                                                     "backgrounds": kind, "id": ids, "group": groups, "effect": eff[:, h]}))

            names, beta, lo, hi, cells, cells_ci = bootstrap_features(
                values, missing, groups, feats, s2["feature_bootstrap_iters"],
                _seed(cfg, f"stage2/features/{position}/{metric}"), s["ci"])
            for h, head in enumerate(heads):
                for i, name in enumerate(names[1:], start=1):
                    feature_rows.append({"position": position, "metric": metric, "head": head_name(head),
                                         "feature": name, "coef": beta[i, h], "lo": lo[i, h], "hi": hi[i, h]})
                for name, val in cells.items():
                    cell_rows.append({"position": position, "metric": metric, "head": head_name(head), "contrast": name,
                                      "mean": val[h], "lo": cells_ci[name][0][h], "hi": cells_ci[name][1][h]})

        confirm_rows += confirm_extremes(cfg, root, model, heads, ids, groups, grid, missing, baseline, column, position,
                                         tokenizer)

    effects = pd.concat(effect_rows, ignore_index=True)
    summary, features, cells_df = pd.DataFrame(summary_rows), pd.DataFrame(feature_rows), pd.DataFrame(cell_rows)
    confirm = pd.DataFrame(confirm_rows)
    checks += cross_stage_check(cfg, root, heads, effects)
    checks_df = pd.DataFrame(checks)
    (run_dir / "validation_report.md").write_text("# exp1extended Stage 2: validation\n\n" + md_table(checks_df) + "\n")
    feats.to_parquet(run_dir / "stage2_token_features.parquet", index=False)
    for name, df in {"effects": effects, "summary": summary, "features": features, "rarity_wordstart": cells_df,
                     "confirm": confirm}.items():
        df.to_parquet(run_dir / f"stage2_{name}.parquet", index=False)

    fig_dir = run_dir / "figures"
    for position in s2["positions"]:
        for group, group_heads in cfg["target_heads"].items():
            hs = [tuple(h) for h in group_heads]
            plot_token_profiles(effects[(effects["position"] == position) & (effects["backgrounds"] == "pooled")],
                                confirm[confirm["position"] == position], feats, hs, cfg["primary_metrics"][group],
                                fig_dir / f"s2_token_profile_{group}_pos{position}.png",
                                f"Per-token effect at position {position}, {group.replace('_', ' ')} heads "
                                "(0 = typical rare token; TL)")
        primary = [(head_name(h), cfg["primary_metrics"][groups_of[h]][0]) for h in heads]
        sel = cells_df[(cells_df["position"] == position)
                       & np.array([(hh, mm) in primary for hh, mm in zip(cells_df["head"], cells_df["metric"])])]
        plot_rarity_wordstart(sel, fig_dir / f"s2_rarity_x_wordstart_pos{position}.png",
                              f"Rarity at matched surface form, split by word start (position {position}, TL)")
        fsel = features[(features["position"] == position)
                        & np.array([(hh, mm) in primary for hh, mm in zip(features["head"], features["metric"])])]
        plot_feature_forest(fsel, fig_dir / f"s2_feature_effects_pos{position}.png",
                            f"What token features predict the effect at position {position} (ids ≥ 256; TL)")
        csel = confirm[(confirm["position"] == position) & confirm["primary"]]
        plot_token_heatmap(csel, fig_dir / f"s2_top_tokens_heatmap_pos{position}.png",
                           f"Confirmed extreme tokens × heads at position {position} (fresh backgrounds; TL)")
    _write_report(run_dir / "stage2_report.md", cfg, run_dir.name, summary, features, cells_df, confirm, checks_df,
                  len(ids), len(backgrounds))
    return summary


def _run_plain(cfg, model, heads, seqs):
    return run_final(model, prepend_bos(seqs, cfg["bos_token_id"]), cfg["batch_size"], heads, cfg["diffuse_threshold"])


def confirm_extremes(cfg, root, model, heads, ids, groups, grid, missing, baseline, column, position, tokenizer) -> list[dict]:
    """Re-measure each head's top / bottom tokens on its primary metrics (pooled discovery effects) on fresh
    backgrounds with a rare_uniform baseline subsample, so reported extremes are not selected and estimated on the
    same data. Every confirmed token is measured for every head, so rows cover the full token x head matrix;
    `side` marks the head / metric that picked the token (empty otherwise)."""
    c = cfg["stage2"]["confirm"]
    groups_of = head_group(cfg)
    primary = {h: cfg["primary_metrics"][groups_of[head]] for h, head in enumerate(heads)}
    discovery = {metric: token_effects(values, missing, baseline) for metric, values in grid.items()}
    picked: dict[tuple[int, str, int], str] = {}
    for h, metrics in primary.items():
        for metric in metrics:
            order = np.argsort(discovery[metric][:, h])
            for side, idx in (("bottom", order[: c["top_n"]]), ("top", order[::-1][: c["top_n"]])):
                for i in idx:
                    picked[(int(i), metric, h)] = side
    rng = np.random.default_rng(_seed(cfg, "stage2/confirm_baseline"))
    base_idx = rng.choice(np.flatnonzero(baseline), size=c["baseline_tokens"], replace=False)
    conf_idx = np.array(sorted({i for i, _, _ in picked} | set(base_idx.tolist())))
    backgrounds, _ = _backgrounds(cfg, root, "confirm", c["backgrounds"])
    seqs, tok_idx, bg_idx, conf_missing = build_sweep(backgrounds, column, ids[conf_idx])
    res = _run_plain(cfg, model, heads, seqs)
    print(f"confirmation, position {position}: {len(conf_idx)} tokens, {len(seqs)} sequences")
    conf_grid = _grid(res, tok_idx, bg_idx, len(conf_idx), len(backgrounds))
    conf_base = np.isin(conf_idx, base_idx)
    observed = ~conf_missing
    base_obs = observed & conf_base[:, None]
    rows = []
    for metric, values in conf_grid.items():
        base = (values * base_obs[..., None]).sum(0) / base_obs.sum(0)[:, None]
        rel = values - base[None]
        for h, head in enumerate(heads):
            if metric not in primary[h]:
                continue
            for j, i in enumerate(conf_idx):
                if conf_base[j] and (int(i), metric, h) not in picked:
                    continue  # baseline-only tokens are not reported
                y = rel[j, observed[j], h]
                se = y.std(ddof=1) / np.sqrt(len(y))
                rows.append({"position": position, "metric": metric, "head": head_name(head),
                             "primary": metric == primary[h][0], "side": picked.get((int(i), metric, h), ""),
                             "id": int(ids[i]), "token": display_text(tokenizer, int(ids[i])), "group": groups[i],
                             "discovery": discovery[metric][i, h], "confirmed": y.mean(), "lo": y.mean() - 1.96 * se,
                             "hi": y.mean() + 1.96 * se, "backgrounds": len(y)})
    return rows


def cross_stage_check(cfg, root, heads, effects: pd.DataFrame) -> list[dict]:
    """Stage 1's final-token effect in a rare context (cell crr − rrr, paired) vs Stage 2's mean effect of the common
    tokens in rare backgrounds (relative to rare_uniform tokens). Informational: CIs should overlap."""
    run = cfg["stage2"].get("stage1_run")
    if not run or 32 not in cfg["stage2"]["positions"]:
        return [{"check": "cross-stage (Stage 1 vs Stage 2 final-token effect)", "detail": "stage2.stage1_run not set",
                 "result": "SKIPPED"}]
    rows = []
    groups = head_group(cfg)
    crr = np.load(root / run / "stage1_cell_crr.npz")["target_scores"]
    rrr = np.load(root / run / "stage1_cell_rrr.npz")["target_scores"]
    for h, head in enumerate(heads):
        metric = cfg["primary_metrics"][groups[head]][0]
        d = METRIC_FNS[metric](crr[:, h]) - METRIC_FNS[metric](rrr[:, h])
        s1 = (d.mean(), 1.96 * d.std(ddof=1) / np.sqrt(len(d)))
        e = effects[(effects["position"] == 32) & (effects["metric"] == metric) & (effects["head"] == head_name(head))
                    & (effects["backgrounds"] == "rare") & (effects["group"] == "common")]["effect"]
        s2 = (e.mean(), 1.96 * e.std(ddof=1) / np.sqrt(len(e)))
        overlap = abs(s1[0] - s2[0]) <= s1[1] + s2[1]
        rows.append({"check": f"cross-stage {head_name(head)} {metric}",
                     "detail": f"Stage 1 {s1[0]:+.2f} ± {s1[1]:.2f} vs Stage 2 {s2[0]:+.2f} ± {s2[1]:.2f}",
                     "result": "PASS" if overlap else "CHECK"})
    return rows


def _write_report(path, cfg, run_id, summary, features, cells, confirm, checks, n_tokens, n_backgrounds) -> None:
    s2 = cfg["stage2"]
    prim = summary[summary["primary"]]
    low = prim[prim["reliability"] < s2["reliability_min"]]
    out = [
        "# exp1extended Stage 2: per-token sweep", "",
        f"- Run: `{run_id}`. {n_tokens} tokens × {n_backgrounds} backgrounds ({n_backgrounds // 2} rare, "
        f"{n_backgrounds // 2} common 256-999) at position(s) {s2['positions']}. Effects are relative to each "
        "background's mean over the rare_uniform tokens (0 = a typical rare token), from an additive token + background "
        "fit; a token already in a background is missing there, never duplicated. Checks: `validation_report.md`.",
        "- Metrics in logit units (see the Stage 0 report).", "",
        "## 1. Is there a per-token profile at all? (omnibus test and reliability)", "",
        f"Omnibus: within-background permutation test of the spread of token effects ({cfg['stats']['n_permutations']} "
        "permutations), BH-FDR across heads. Reliability: background split-half, Spearman-Brown corrected; the target "
        f"is ≥ {s2['reliability_min']}.", "",
        md_table(prim.drop(columns=["primary"]), ".3f"), "",
        (f"Below the reliability target: {', '.join(low['head'] + ' ' + low['metric'])}. Treat their per-token "
         "profiles as noisy; add backgrounds before interpreting them." if len(low) else
         "Every primary profile meets the reliability target."), "",
    ]
    for position in s2["positions"]:
        out += [f"## 2. Token profiles, position {position}", ""]
        for group in cfg["target_heads"]:
            out.append(f"![{group}](figures/s2_token_profile_{group}_pos{position}.png)")
        out += ["", f"Confirmed extremes (primary metrics; top/bottom {s2['confirm']['top_n']} per head re-measured on "
                f"{2 * s2['confirm']['backgrounds']} fresh backgrounds; `discovery` is the selection-biased estimate):", ""]
        c = confirm[(confirm["position"] == position) & (confirm["side"] != "")]
        c = c.sort_values(["head", "metric", "confirmed"], ascending=[True, True, False])
        shown = pd.concat([c.groupby(["head", "metric"]).head(5), c.groupby(["head", "metric"]).tail(5)]).drop_duplicates()
        shown = shown.sort_values(["head", "metric", "confirmed"], ascending=[True, True, False])
        out += [md_table(shown[["head", "metric", "token", "id", "group", "discovery", "confirmed", "lo", "hi"]], ".2f"), "",
                f"![top tokens](figures/s2_top_tokens_heatmap_pos{position}.png)", ""]
        out += [f"## 3. Rarity × word start at matched surface form, position {position}", "",
                "Mean token effect of common (256-999) minus rare_matched tokens, within word-start and continuation "
                "tokens; two-way bootstrap CIs.", "",
                f"![rarity x word start](figures/s2_rarity_x_wordstart_pos{position}.png)", ""]
        cc = cells[cells["position"] == position]
        cc = cc.assign(value=[f"{m:+.2f} [{l:+.2f}, {h:+.2f}]" for m, l, h in zip(cc["mean"], cc["lo"], cc["hi"])])
        out += [md_table(cc.pivot_table(index=["head", "metric"], columns="contrast", values="value",
                                        aggfunc="first").reset_index()), ""]
        out += [f"## 4. Token features, position {position}", "",
                "Regression of token effects (ids ≥ 256) on features; binary features are 0/1, char_len / log_id / "
                "embed_norm per SD. Two-way bootstrap CIs. log_id and char_len are correlated, so read them jointly.", "",
                f"![features](figures/s2_feature_effects_pos{position}.png)", ""]
    out += ["## 5. Checks", "", md_table(checks), ""]
    path.write_text("\n".join(out))
