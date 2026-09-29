"""Stage 0: route each target head using the saved Exp 1 sequences (no new sequences are sampled).

Re-runs A_rare, B_common and C_common_clean through TransformerLens with extra hooks, checks the attention is
bit-identical to the saved Exp 1 arrays (V1) and that scores, q and k are consistent (V2-V4), then:
  1. logit split: sink_logit = s0 - LSE(rest), where s0 = q.k0/8 depends only on the query because BOS's key
     k0 is the same in every sequence; the change splits exactly into a query-only term and a competing-keys term;
  2. offline q/k swap: scores recomputed with one condition's final-query vector against the other condition's
     keys (pairs i = sequence i of each condition), giving query / key / interaction parts of the change;
  3. attractor tables: which key tokens raise their score relative to BOS, per token in B/C (each token is a key
     ~30-40 times there) and per feature stratum in A (too sparse per token);
  4. ICC of each head's primary metric by the token at positions 32, 31 and 16 (16 = control);
  5. the A - C gap post-stratified on whether the final token starts a word.
Gate G0 labels each head query_token / query_context / key_side / mixed / no_gap, which sets its Stage 2 route.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from exp1.sequences import load_or_create_sequences, prepend_bos
from exp1.validation import md_table, verdict
from exp1extended import measures
from exp1extended.config import head_group, head_name, target_heads
from exp1extended.extract import check_bitmatch, check_bos_key, check_qk, check_softmax, run_final
from exp1extended.plots import plot_decomposition
from exp1extended.stats import icc_oneway, paired_test
from exp1extended.tokens import display_text, token_features

METRIC_FNS = {"sink_logit": measures.sink_logit, "prev_logit": measures.prev_logit, "other_gap": measures.other_gap}
PARTS = ("query", "key", "interaction", "total")
POSITIONS = {"final (32)": 31, "previous (31)": 30, "control (16)": 15}  # seqs columns (seqs has no BOS)


def load_exp1_sequences(cfg: dict[str, Any], root: Path, condition: str) -> np.ndarray:
    """An Exp 1 condition's sequences through exp1's own cache check (config + SHA-256 must match)."""
    e = cfg["_exp1"]
    lo, hi = e["conditions"][condition]["id_range"]
    return load_or_create_sequences(condition, [lo, hi], e["n_sequences"], e["seq_len"], e["seed"],
                                    e["sample_without_replacement"], e["bos_token_id"], root / e["paths"]["sequences_dir"])


def _conditions(cfg: dict[str, Any]) -> list[str]:
    return list(dict.fromkeys(c for pair in cfg["stage0"]["comparisons"] for c in pair))


def extract_conditions(cfg: dict[str, Any], root: Path, model, heads: list[tuple[int, int]]) -> tuple[dict, list[dict]]:
    """Per condition: run_final outputs (full pattern/scores dropped after checking) plus the sequences."""
    v, d_head = cfg["validation"], model.cfg.d_head
    runs, checks = {}, []
    for c in _conditions(cfg):
        seqs = load_exp1_sequences(cfg, root, c)
        res = run_final(model, prepend_bos(seqs, cfg["bos_token_id"]), cfg["batch_size"], heads,
                        cfg["diffuse_threshold"], keep_full=True, keep_qk=True)
        saved = np.load(root / cfg["exp1_runs"][c] / f"{c}_attn.npy")[: len(seqs)]
        for name, (ok, diff) in {
            "V1 pattern bit-identical to saved Exp 1 attention": check_bitmatch(res["pattern"], saved),
            "V2 softmax(scores) = pattern, rows sum to 1": check_softmax(res["scores"], res["pattern"], v["row_sum_atol"]),
            "V3 q.k/8 reproduces target-head scores": check_qk(res["q"], res["k"], res["target_scores"], d_head, v["qk_atol"],
                                                                v["qk_rtol"]),
            "V4 BOS key identical across sequences": check_bos_key(res["k"], v["bos_key_atol"]),
        }.items():
            checks.append({"check": name, "condition": c, "n": len(seqs), "max |diff|": diff, "result": verdict(ok)})
        del res["pattern"], res["scores"]
        res["seqs"] = seqs
        runs[c] = res
    all_k0 = np.concatenate([runs[c]["k"][:, :, :1, :] for c in runs])
    ok, diff = check_bos_key(all_k0, v["bos_key_atol"])
    checks.append({"check": "V4 BOS key identical across conditions", "condition": "all", "n": len(all_k0),
                   "max |diff|": diff, "result": verdict(ok)})
    return runs, checks


def _diff_ci(xa: np.ndarray, xb: np.ndarray, iters: int, seed: int, ci: float) -> tuple[np.ndarray, ...]:
    """mean(xb) - mean(xa) over axis 0 for two independent samples, with a percentile CI from resampling each."""
    rng = np.random.default_rng(seed)
    boot = np.empty((iters,) + xa.shape[1:])
    for i in range(iters):
        boot[i] = xb[rng.integers(0, len(xb), len(xb))].mean(0) - xa[rng.integers(0, len(xa), len(xa))].mean(0)
    tail = (1.0 - ci) / 2.0 * 100.0
    return xb.mean(0) - xa.mean(0), np.percentile(boot, tail, axis=0), np.percentile(boot, 100.0 - tail, axis=0)


def logit_split_table(cfg, runs, heads) -> pd.DataFrame:
    s = cfg["stats"]
    rows = []
    for a, b in cfg["stage0"]["comparisons"]:
        split = {c: measures.logit_split(runs[c]["target_scores"]) for c in (a, b)}
        terms = {
            "Δ sink_logit": (split[a][0] - split[a][1], split[b][0] - split[b][1]),
            "Δ s0 (query only)": (split[a][0], split[b][0]),
            "−Δ LSE rest (competing keys)": (-split[a][1], -split[b][1]),
        }
        for term, (xa, xb) in terms.items():
            mean, lo, hi = _diff_ci(xa, xb, s["bootstrap_iters"], s["bootstrap_seed"], s["ci"])
            for t, head in enumerate(heads):
                rows.append({"comparison": f"{a} → {b}", "head": head_name(head), "term": term,
                             "mean": mean[t], "lo": lo[t], "hi": hi[t]})
    return pd.DataFrame(rows)


def qk_swap_table(cfg, runs, heads, d_head: int) -> pd.DataFrame:
    """Query / key / interaction parts of each metric's change, first -> second condition, per head.

    With m(x, y) = metric of scores from condition x's final query and condition y's keys:
      query = mean over key states of the query switch = ½[(m(b,a) - m(a,a)) + (m(b,b) - m(a,b))]
      key   = ½[(m(a,b) - m(a,a)) + (m(b,b) - m(b,a))]
      interaction = m(b,b) - m(b,a) - m(a,b) + m(a,a);  query + key = m(b,b) - m(a,a) = total exactly.
    "Key" covers every key, including positions 31 and 32 (the final token's own key).
    """
    s = cfg["stats"]
    groups = head_group(cfg)
    rows = []
    for a, b in cfg["stage0"]["comparisons"]:
        n = min(len(runs[a]["q"]), len(runs[b]["q"]))
        q = {c: runs[c]["q"][:n] for c in (a, b)}
        k = {c: runs[c]["k"][:n] for c in (a, b)}
        scores = {(x, y): measures.qk_scores(q[x], k[y], d_head) for x in (a, b) for y in (a, b)}
        for metric, fn in METRIC_FNS.items():
            m = {xy: fn(sc) for xy, sc in scores.items()}
            aa, ab, ba, bb = m[(a, a)], m[(a, b)], m[(b, a)], m[(b, b)]
            parts = {
                "query": paired_test(0.5 * (ba + bb), 0.5 * (aa + ab), s["bootstrap_iters"], s["bootstrap_seed"], s["ci"]),
                "key": paired_test(0.5 * (ab + bb), 0.5 * (aa + ba), s["bootstrap_iters"], s["bootstrap_seed"], s["ci"]),
                "interaction": paired_test(bb - ba - ab + aa, np.zeros_like(aa), s["bootstrap_iters"], s["bootstrap_seed"], s["ci"]),
                "total": paired_test(bb, aa, s["bootstrap_iters"], s["bootstrap_seed"], s["ci"]),
            }
            for t, head in enumerate(heads):
                for part, res in parts.items():
                    rows.append({"comparison": f"{a} → {b}", "head": head_name(head), "group": groups[head],
                                 "metric": metric, "part": part, "mean": res["mean_diff"][t], "lo": res["lo"][t],
                                 "hi": res["hi"][t], "p": res["p"][t]})
    return pd.DataFrame(rows)


def icc_table(cfg, runs, heads) -> pd.DataFrame:
    rows = []
    for c in _conditions(cfg):
        if c == "A_rare":
            continue  # 39,000-token pool: almost no token repeats at a position, so ICC is not estimable
        for t, head in enumerate(heads):
            metric = cfg["primary_metrics"][head_group(cfg)[head]][0]
            y = METRIC_FNS[metric](runs[c]["target_scores"])[:, t]
            row = {"condition": c, "head": head_name(head), "metric": metric}
            row.update({pos: _icc_or_nan(y, runs[c]["seqs"][:, col]) for pos, col in POSITIONS.items()})
            rows.append(row)
    return pd.DataFrame(rows)


def _icc_or_nan(y: np.ndarray, groups: np.ndarray) -> float:
    """ICC, or NaN when fewer than 2 tokens repeat at that position (not estimable)."""
    try:
        return icc_oneway(y, groups)
    except ValueError:
        return float("nan")


def wordstart_table(cfg, runs, heads, tokenizer) -> tuple[pd.DataFrame, float, float]:
    """A_rare - C_common_clean gap in each head's sink_mass and sink_logit, overall, within final-token word-start
    and continuation, and with A re-weighted to C's word-start share."""
    a, c = "A_rare", "C_common_clean"
    ws = {}
    for cond in (a, c):
        final = runs[cond]["seqs"][:, -1]
        feats = token_features(tokenizer, np.unique(final), cfg["matching"]["len_bins"]).set_index("id")
        ws[cond] = feats.loc[final, "word_start"].to_numpy(dtype=bool)
    share_c = ws[c].mean()
    rows = []
    for t, head in enumerate(heads):
        layer, h = head
        values = {
            "sink_mass": {cond: runs[cond]["sink_mass"][:, layer, h] for cond in (a, c)},
            "sink_logit": {cond: measures.sink_logit(runs[cond]["target_scores"][:, t]) for cond in (a, c)},
        }
        for metric, v in values.items():
            gap_ws = v[a][ws[a]].mean() - v[c][ws[c]].mean()
            gap_cont = v[a][~ws[a]].mean() - v[c][~ws[c]].mean()
            reweighted = share_c * v[a][ws[a]].mean() + (1 - share_c) * v[a][~ws[a]].mean() - v[c].mean()
            rows.append({"head": head_name(head), "metric": metric, "gap A−C": v[a].mean() - v[c].mean(),
                         "gap, word-start final": gap_ws, "gap, continuation final": gap_cont,
                         "gap, A re-weighted to C's word-start share": reweighted})
    return pd.DataFrame(rows), float(ws[a].mean()), float(share_c)


def attractor_tables(cfg, runs, heads, tokenizer, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per key token (keys 1-30): mean score relative to BOS, after removing each position's mean (tokens sit at
    random positions, so this only strips positional variance). Returns (top tokens in B/C, per-head split-half
    reliability of the per-token means, A means by feature stratum)."""
    top_n = cfg["stage0"]["attractor_top_n"]
    top_rows, rel_rows = [], []
    rng = np.random.default_rng(seed)
    for c in _conditions(cfg):
        if c == "A_rare":
            continue
        ids = runs[c]["seqs"][:, :30].ravel()
        att = measures.attractor_scores(runs[c]["target_scores"])  # [n, T, 30]
        att = att - att.mean(axis=0, keepdims=True)
        half = rng.permutation(len(att)) < len(att) // 2
        size = int(ids.max()) + 1
        for t, head in enumerate(heads):
            vals = att[:, t, :].ravel()
            cnt = np.bincount(ids, minlength=size)
            mean = np.bincount(ids, vals, minlength=size) / np.maximum(cnt, 1)
            var = np.bincount(ids, vals ** 2, minlength=size) / np.maximum(cnt, 1) - mean ** 2
            se = np.sqrt(np.maximum(var, 0) / np.maximum(cnt - 1, 1))
            halves = []
            for mask in (half, ~half):
                hid, hv = runs[c]["seqs"][mask, :30].ravel(), att[mask, t, :].ravel()
                hc = np.bincount(hid, minlength=size)
                halves.append((np.bincount(hid, hv, minlength=size) / np.maximum(hc, 1), hc))
            both = (halves[0][1] >= 2) & (halves[1][1] >= 2)
            r = np.corrcoef(halves[0][0][both], halves[1][0][both])[0, 1]
            rel_rows.append({"condition": c, "head": head_name(head), "tokens": int(both.sum()),
                             "split-half r": r, "Spearman-Brown": 2 * r / (1 + r)})
            eligible = np.flatnonzero(cnt >= 5)
            for rank, tok in enumerate(eligible[np.argsort(-mean[eligible])][:top_n], start=1):
                top_rows.append({"condition": c, "head": head_name(head), "rank": rank, "id": int(tok),
                                 "token": display_text(tokenizer, int(tok)), "n": int(cnt[tok]),
                                 "mean score vs BOS (centred)": mean[tok], "se": se[tok]})

    a = runs["A_rare"]
    ids = a["seqs"][:, :30].ravel()
    feats = token_features(tokenizer, np.unique(ids), cfg["matching"]["len_bins"]).set_index("id").loc[ids]
    att = measures.attractor_scores(a["target_scores"])
    att = att - att.mean(axis=0, keepdims=True)
    strata = feats[["word_start", "len_bin", "is_alpha"]].reset_index(drop=True)
    stratum_rows = []
    for t, head in enumerate(heads):
        df = strata.assign(score=att[:, t, :].ravel())
        g = df.groupby(["word_start", "len_bin", "is_alpha"])["score"].agg(["mean", "count"]).reset_index()
        stratum_rows.append(g.assign(head=head_name(head)))
    return pd.DataFrame(top_rows), pd.DataFrame(rel_rows), pd.concat(stratum_rows, ignore_index=True)


def route_heads(cfg, swap: pd.DataFrame, icc: pd.DataFrame, heads) -> pd.DataFrame:
    """Gate G0. Uses each head's first primary metric, on A_rare → C_common_clean (A_rare → B_common for the
    byte-specific group, whose shift appears only with byte tokens)."""
    s0 = cfg["stage0"]
    groups = head_group(cfg)
    rows = []
    for head in heads:
        group = groups[head]
        metric = cfg["primary_metrics"][group][0]
        other = "B_common" if group == "byte_specific" else "C_common_clean"
        comparison = f"A_rare → {other}"
        sel = swap[(swap["comparison"] == comparison) & (swap["head"] == head_name(head)) & (swap["metric"] == metric)]
        part = sel.set_index("part")
        total = part.loc["total"]
        q_share = part.loc["query", "mean"] / total["mean"]
        k_share = part.loc["key", "mean"] / total["mean"]
        icc_row = icc[(icc["condition"] == other) & (icc["head"] == head_name(head))]
        icc_final = float(icc_row["final (32)"].iloc[0])
        if total["lo"] <= 0 <= total["hi"]:
            route = "no_gap"
        elif q_share >= s0["route_share"]:
            route = "query_token" if icc_final >= s0["route_icc"] else "query_context"
        elif k_share >= s0["route_share"]:
            route = "key_side"
        else:
            route = "mixed"
        rows.append({"head": head_name(head), "group": group, "metric": metric, "comparison": comparison,
                     "total": total["mean"], "total lo": total["lo"], "total hi": total["hi"],
                     "query share": q_share, "key share": k_share, f"ICC final ({other})": icc_final, "route": route})
    return pd.DataFrame(rows)


def run_stage0(cfg: dict[str, Any], run_dir: Path, model, root: Path) -> pd.DataFrame:
    heads = target_heads(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg["figures"]["tokenizer"])
    runs, checks = extract_conditions(cfg, root, model, heads)
    checks_df = pd.DataFrame(checks)
    (run_dir / "validation_report.md").write_text(
        "# exp1extended Stage 0: validation\n\n" + md_table(checks_df, ".2e") + "\n")
    if (checks_df["result"] == "FAIL").any():
        raise AssertionError(f"Stage 0 validation failed; see {run_dir / 'validation_report.md'}")
    for c, res in runs.items():
        np.save(run_dir / f"{c}_target_scores.npy", res["target_scores"])

    split = logit_split_table(cfg, runs, heads)
    swap = qk_swap_table(cfg, runs, heads, model.cfg.d_head)
    icc = icc_table(cfg, runs, heads)
    ws, ws_share_a, ws_share_c = wordstart_table(cfg, runs, heads, tokenizer)
    top, rel, strata = attractor_tables(cfg, runs, heads, tokenizer, cfg["stats"]["bootstrap_seed"])
    routes = route_heads(cfg, swap, icc, heads)
    for name, df in {"logit_split": split, "qk_swap": swap, "icc": icc, "wordstart": ws, "attractors": top,
                     "attractor_reliability": rel, "attractor_strata_A": strata, "routes": routes}.items():
        df.to_parquet(run_dir / f"stage0_{name}.parquet", index=False)

    groups = head_group(cfg)
    fig_rows = []
    for head in heads:
        metric = cfg["primary_metrics"][groups[head]][0]
        sel = swap[(swap["head"] == head_name(head)) & (swap["metric"] == metric)]
        fig_rows.append(sel.assign(head=f"{head_name(head)} {metric}"))
    plot_decomposition(
        pd.concat(fig_rows), run_dir / "figures" / "stage0_qk_swap.png",
        "Where each head's change comes from: final-query vector vs keys (offline q/k swap, TL)",
        "change in primary metric (logit units), first → second condition",
        parts=("query", "key", "interaction"),
        note="dots = query / key / interaction parts with 95% bootstrap CIs over 1000 index-paired sequences; "
             "outline = total change. query + key = total exactly; 'key' includes the final token's own key.",
    )
    _write_report(run_dir / "stage0_report.md", cfg, run_dir.name, split, swap, icc, ws, ws_share_a, ws_share_c,
                  top, rel, strata, routes)
    return routes


def _fmt_ci(mean: float, lo: float, hi: float) -> str:
    return f"{mean:+.2f} [{lo:+.2f}, {hi:+.2f}]"


def _write_report(path, cfg, run_id, split, swap, icc, ws, ws_share_a, ws_share_c, top, rel, strata, routes) -> None:
    s0 = cfg["stage0"]
    out = [
        "# exp1extended Stage 0: routing the target heads", "",
        f"- Run: `{run_id}`. Sequences: the saved Exp 1 A_rare / B_common / C_common_clean (1000 each), re-run in "
        "TransformerLens with extra hooks. Checks V1-V4: `validation_report.md` (all PASS, or the run would have stopped).",
        "- Metrics are in logit units, computed from the pre-softmax scores: sink_logit = log(p_BOS / (1 − p_BOS)), "
        "prev_logit likewise for position 31, other_gap = max score over keys 1-30 minus the BOS score.",
        "- Differences are second condition minus first (A_rare → C_common_clean = C − A). CIs are 95% bootstrap.", "",
        "## 1. Routes (gate G0)", "",
        f"Rule, on each head's first primary metric: no_gap if the total change's CI includes 0; else query side if "
        f"the query part is ≥ {s0['route_share']:.0%} of the total — `query_token` when the final token's ICC is ≥ "
        f"{s0['route_icc']} (its identity matters, so a per-token sweep at position 32 is informative), else "
        f"`query_context` (the query vector changes, but not because of the final token itself); `key_side` if the key "
        f"part is ≥ {s0['route_share']:.0%}; otherwise `mixed`.", "",
        md_table(routes, ".2f"), "",
        "## 2. Offline q/k swap", "",
        "Scores recomputed with one condition's final-query vector against the other condition's keys (sequence i of "
        "each condition paired by index). query + key = total exactly; interaction is shown separately.", "",
        "![q/k swap](figures/stage0_qk_swap.png)", "",
    ]
    wide = swap.assign(value=[_fmt_ci(m, l, h) for m, l, h in zip(swap["mean"], swap["lo"], swap["hi"])])
    groups = head_group(cfg)
    primary = {head_name(h): cfg["primary_metrics"][g] for h, g in groups.items()}
    wide = wide[[m in primary[h] for h, m in zip(wide["head"], wide["metric"])]]
    wide = wide.pivot_table(index=["comparison", "head", "metric"], columns="part", values="value", aggfunc="first")
    out += [md_table(wide.reset_index()[["comparison", "head", "metric", *PARTS]]), ""]

    s = split[split["term"].isin(["Δ sink_logit", "Δ s0 (query only)", "−Δ LSE rest (competing keys)"])]
    s = s.assign(value=[_fmt_ci(m, l, h) for m, l, h in zip(s["mean"], s["lo"], s["hi"])])
    s = s.pivot_table(index=["comparison", "head"], columns="term", values="value", aggfunc="first").reset_index()
    out += ["## 3. Logit split of the sink change", "",
            "Δ sink_logit = Δ s0 + (−Δ LSE of the other keys' scores), exactly. s0 = q·k_BOS/8 moves only through the "
            "query (BOS's key is identical in every sequence, check V4).", "",
            md_table(s[["comparison", "head", "Δ sink_logit", "Δ s0 (query only)", "−Δ LSE rest (competing keys)"]]), ""]

    out += ["## 4. Which position's token identity matters (ICC)", "",
            "One-way ICC(1) of each head's primary metric, grouping sequences that share the token at that position "
            "(tokens seen ≥ 2 times; ~280 groups). Position 16 is a control and should be ≈ 0.", "",
            md_table(icc, ".2f"), ""]

    out += ["## 5. Word-start vs rarity", "",
            f"Share of final tokens that start a word: A_rare {ws_share_a:.2f}, C_common_clean {ws_share_c:.2f}. "
            "Gap = A − C (positive: more sink on rare tokens).", "",
            md_table(ws, ".3f"), ""]

    out += ["## 6. Attractor tokens (keys 1-30)", "",
            "Mean score relative to BOS when the token is a key, after removing each position's mean; tokens seen "
            "≥ 5 times. Split-half reliability (Spearman-Brown) says whether per-token means are stable.", "",
            md_table(rel, ".2f"), ""]
    show = routes[routes["route"].isin(["key_side", "mixed"]) | (routes["group"] == "byte_specific")]["head"].tolist()
    show += [h for h in ("L2H0", "L2H4") if h not in show]
    out += [f"Top {s0['attractor_top_n']} tokens for heads routed key_side / mixed, the byte-specific heads, "
            "and L2H0 / L2H4:", "", md_table(top[top["head"].isin(show)], ".3f"), ""]
    out += ["A_rare by feature stratum (word_start, len_bin, is_alpha), same score, for the same heads "
            "(top 3 strata by mean, ≥ 50 occurrences):", ""]
    st = strata[(strata["head"].isin(show)) & (strata["count"] >= 50)]
    st = st.sort_values(["head", "mean"], ascending=[True, False]).groupby("head").head(3)
    out += [md_table(st[["head", "word_start", "len_bin", "is_alpha", "count", "mean"]], ".3f"), ""]
    path.write_text("\n".join(out))
