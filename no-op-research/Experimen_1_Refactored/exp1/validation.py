"""Phase 1 validation checks and the validation_report.md writer."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from exp1.backends import get_final_attention
from exp1.metrics import LABELS, classify

BACKEND_NAMES = {"transformer_lens": "TL", "huggingface": "HF"}


@dataclass
class BackendRun:
    """Everything one backend produced for one condition."""

    backend: str
    attn: np.ndarray  # [n_seq, n_layers, n_heads, key_len]
    heads: pd.DataFrame
    row_sum_dev: float
    det_equal: bool
    det_max_diff: float


def check_determinism(
    model: torch.nn.Module, tokens: np.ndarray, attn: np.ndarray, n: int, batch_size: int, query_position: int
) -> tuple[bool, float]:
    """Re-extract the first n sequences and compare them to the main run bit for bit."""
    rerun = get_final_attention(model, tokens[:n], batch_size, query_position)
    return bool(np.array_equal(rerun, attn[:n])), float(np.abs(rerun - attn[:n]).max())


def verdict(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def md_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    def fmt(v: Any) -> str:
        return format(v, floatfmt) if isinstance(v, (float, np.floating)) else str(v)

    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def label_percentages(heads: pd.DataFrame, by_layer: bool = False) -> pd.DataFrame | pd.Series:
    """% of heads whose mode_label is each label, overall or within each layer."""
    if not by_layer:
        return heads["mode_label"].value_counts(normalize=True).reindex(LABELS, fill_value=0) * 100
    counts = pd.crosstab(heads["layer"], heads["mode_label"], normalize="index") * 100
    return counts.reindex(columns=list(LABELS), fill_value=0)


def sink_by_layer(heads: pd.DataFrame) -> pd.Series:
    return heads.groupby("layer")["mean_sink_mass"].mean()


def _known_heads_section(runs: dict[str, BackendRun], known: list[list[int]], min_prev: float) -> tuple[str, list]:
    rows, checks = [], []
    for layer, head in known:
        for backend, run in runs.items():
            h = run.heads[(run.heads["layer"] == layer) & (run.heads["head"] == head)].iloc[0]
            ok = h["mode_label"] == "Previous" and h["mean_prev_mass"] >= min_prev
            rows.append({
                "head": f"L{layer}H{head}", "backend": BACKEND_NAMES[backend], "mode_label": h["mode_label"],
                "consistency": h["consistency"], "frac_previous": h["frac_previous"],
                "prev_mass mean": h["mean_prev_mass"], "prev_mass std": h["std_prev_mass"],
                "sink_mass mean": h["mean_sink_mass"], "result": verdict(ok),
            })
            detail = f"mode={h['mode_label']}, prev_mass={h['mean_prev_mass']:.3f}, consistency={h['consistency']:.3f}"
            checks.append((f"3. Known prev-token head L{layer}H{head}", BACKEND_NAMES[backend], verdict(ok), detail))
    return md_table(pd.DataFrame(rows)), checks


def write_validation_report(
    path: Path, cfg: dict[str, Any], run_id: str, runs: dict[str, dict[str, BackendRun]],
    figures: dict[str, dict[str, list[str]]], sequence_figures: dict[str, str], query_index: int,
) -> None:
    v = cfg["validation"]
    primary = cfg["primary_backend"]
    key_len = cfg["seq_len"] + int(cfg["prepend_bos"])
    out = [
        "# Experiment 1 validation report",
        "",
        f"- Run: `{run_id}`  |  model: `{cfg['model']}`  |  backends: "
        + ", ".join(f"{BACKEND_NAMES[b]} (`{c['model_name']}`)" for b, c in cfg["backends"].items())
        + f"  |  primary: {BACKEND_NAMES[primary]}",
        f"- {cfg['n_sequences']} sequences x {cfg['seq_len']} random tokens"
        + (f" + BOS ({cfg['bos_token_id']}) at position 0" if cfg["prepend_bos"] else "")
        + f"; key_len = {key_len}, query index = {query_index}; sampled "
        + ("without" if cfg["sample_without_replacement"] else "with") + " replacement within each sequence.",
        f"- Labels: Diffuse if max_weight < {cfg['diffuse_threshold']}, else Sink (argmax 0) / Previous "
        f"(argmax {query_index - 1}) / Self (argmax {query_index}) / Other.",
        "",
    ]

    for condition, by_backend in runs.items():
        lo, hi = cfg["conditions"][condition]["id_range"]
        prim = by_backend[primary]
        checks: list[tuple[str, str, str, str]] = []
        for backend, run in by_backend.items():
            name = BACKEND_NAMES[backend]
            checks.append(("1. Attention rows sum to 1", name, verdict(run.row_sum_dev <= v["row_sum_atol"]),
                           f"max abs(sum - 1) = {run.row_sum_dev:.2e} (atol {v['row_sum_atol']:g})"))
        for backend, run in by_backend.items():
            checks.append((f"2. Determinism (first {v['determinism_n']} re-extracted)", BACKEND_NAMES[backend],
                           verdict(run.det_equal),
                           "bit-identical" if run.det_equal else f"max abs diff {run.det_max_diff:.2e}"))
        known_table, known_checks = _known_heads_section(by_backend, v["known_prev_heads"], v["prev_head_min_prev_mass"])
        checks += known_checks

        pct = pd.DataFrame({BACKEND_NAMES[b]: label_percentages(r.heads) for b, r in by_backend.items()})
        mode_counts = ", ".join(f"{lab} {p:.1f}%" for lab, p in pct[BACKEND_NAMES[primary]].items())
        checks.append(("4. Head types by mode_label", BACKEND_NAMES[primary], "INFO", mode_counts))

        sink = pd.DataFrame({BACKEND_NAMES[b]: sink_by_layer(r.heads) for b, r in by_backend.items()})
        prim_sink = sink[BACKEND_NAMES[primary]]
        half = len(prim_sink) // 2
        early, late = prim_sink.iloc[:half].mean(), prim_sink.iloc[half:].mean()
        checks.append(("5. Sink mass higher in later layers", BACKEND_NAMES[primary], verdict(late > early),
                       f"layers {half}-{len(prim_sink) - 1}: {late:.3f} vs layers 0-{half - 1}: {early:.3f}; "
                       f"peak L{int(prim_sink.idxmax())} ({prim_sink.max():.3f}), last layer "
                       f"L{len(prim_sink) - 1} ({prim_sink.iloc[-1]:.3f}); "
                       f"{'monotonic' if prim_sink.is_monotonic_increasing else 'not monotonic'} across layers"))

        cross = None
        if len(by_backend) > 1:
            (b1, r1), (b2, r2) = list(by_backend.items())[:2]
            max_diff = float(np.abs(r1.attn - r2.attn).max())
            thr = cfg["diffuse_threshold"]
            seq_agree = float((classify(r1.attn, thr, query_index) == classify(r2.attn, thr, query_index)).mean())
            mode_agree = int((r1.heads["mode_label"].values == r2.heads["mode_label"].values).sum())
            n_heads = len(r1.heads)
            cross = (max_diff, seq_agree, mode_agree, n_heads)
            checks.append((f"6. Cross-library agreement", f"{BACKEND_NAMES[b1]} vs {BACKEND_NAMES[b2]}",
                           verdict(max_diff <= v["cross_backend_atol"]),
                           f"max abs attn diff {max_diff:.2e} (atol {v['cross_backend_atol']:g}); "
                           f"per-sequence labels agree {100 * seq_agree:.3f}%; mode_label agrees {mode_agree}/{n_heads} heads"))

        out += [f"## Condition `{condition}` (ids {lo}-{hi})", "", "### Checks", "",
                md_table(pd.DataFrame(checks, columns=["check", "backend", "result", "detail"])), ""]

        out += ["### 3. Known previous-token heads", "",
                f"Expected: mode_label = Previous with mean prev_mass >= {v['prev_head_min_prev_mass']}.", "",
                known_table, ""]

        out += ["### 4. Head types (% of heads by mode_label)", "", "Overall:", "",
                md_table(pct.rename_axis("mode_label").reset_index(), ".1f"), "",
                f"Per layer ({BACKEND_NAMES[primary]}, % of the {len(prim.heads) // len(prim_sink)} heads in each layer):", "",
                md_table(label_percentages(prim.heads, by_layer=True).reset_index(), ".1f"), ""]

        non_sink = prim.heads[prim.heads["mode_label"] != "Sink"]
        lines = []
        for label in LABELS:
            sel = non_sink[non_sink["mode_label"] == label]
            if len(sel):
                lines.append(f"- **{label}** ({len(sel)}): " + ", ".join(
                    f"L{r.layer}H{r.head} ({r.consistency:.2f})" for r in sel.itertuples()))
        out += [f"Heads whose mode_label is not Sink ({BACKEND_NAMES[primary]}; consistency in parentheses):", ""]
        out += lines or ["- none"]
        out += [""]

        out += ["### 5. Mean sink_mass per layer", "",
                "Mean over the layer's heads of each head's mean sink_mass (attention on BOS).", "",
                md_table(sink.rename_axis("layer").reset_index(), ".3f"), ""]

        if cross is not None:
            max_diff, seq_agree, mode_agree, n_heads = cross
            out += ["### 6. Cross-library agreement (TransformerLens vs Hugging Face)", "",
                    f"- Same token-ID tensors fed to both; max abs attention difference: {max_diff:.2e}",
                    f"- Per-(sequence, layer, head) labels agreeing: {100 * seq_agree:.3f}%",
                    f"- Per-head mode_label agreeing: {mode_agree}/{n_heads}", ""]

        out += ["### Figures", "", f"- Example input sequences: [{Path(sequence_figures[condition]).stem}]({sequence_figures[condition]})"]
        for backend, files in figures[condition].items():
            out += [f"- {BACKEND_NAMES[backend]}: " + ", ".join(f"[{Path(f).stem}]({f})" for f in files)]
        out += [""]

    path.write_text("\n".join(out))
