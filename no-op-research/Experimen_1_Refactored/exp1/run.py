"""Experiment 1 entrypoint.   Run:  uv run python -m exp1.run --config configs/exp1.yaml

For each condition in run_conditions and each configured backend (TransformerLens, Hugging Face):
sample or reload the sequences, extract final-query attention, aggregate per head, save
everything to results/<YYYYMMDD-HHMMSS>_exp1/, and write validation_report.md.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from exp1.aggregate import aggregate_heads
from exp1.backends import bos_token_id, check_rows_sum_to_one, get_final_attention, load_model, set_determinism
from exp1.config import load_config
from exp1.metrics import LABELS
from exp1.phase2 import run_phase2
from exp1.plots import plot_mode_label_heatmap, plot_scalar_heatmap
from exp1.runlog import make_run_dir, save_config_copy, save_versions
from exp1.sequences import load_or_create_sequences, prepend_bos, sequences_hash
from exp1.show_sequences import plot_sequences
from exp1.validation import BACKEND_NAMES, BackendRun, check_determinism, write_validation_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_figures(heads: pd.DataFrame, fig_dir: Path, stem: str, label: str, key_len: int) -> list[str]:
    """The three Phase 1 heatmaps; returns paths relative to the run folder."""
    files = [f"{stem}_mode_label.png", f"{stem}_sink_mass.png", f"{stem}_entropy.png"]
    plot_mode_label_heatmap(heads, LABELS, fig_dir / files[0], f"{label}: mode label (cell = consistency)")
    plot_scalar_heatmap(heads, "mean_sink_mass", fig_dir / files[1], f"{label}: mean sink_mass (attention on position 0)",
                        "mean sink_mass", vmin=0.0, vmax=1.0)
    plot_scalar_heatmap(heads, "mean_entropy", fig_dir / files[2], f"{label}: mean entropy (max ln {key_len} = {math.log(key_len):.2f})",
                        "mean entropy (nats)", vmin=0.0, vmax=math.log(key_len))
    return [f"{fig_dir.name}/{f}" for f in files]


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Experiment 1: random-token attention baseline")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    paths = {name: PROJECT_ROOT / p for name, p in cfg["paths"].items()}
    run_dir = make_run_dir(paths["results_dir"])
    save_config_copy(args.config, run_dir)
    save_versions(run_dir, PROJECT_ROOT)
    set_determinism(cfg["seed"])
    print(f"run folder: {run_dir}")

    key_len = cfg["seq_len"] + int(cfg["prepend_bos"])
    query_index = cfg["query_position"] % key_len
    v = cfg["validation"]

    # The tokenizer only turns ids into display text for the sequence figures.
    tokenizer = AutoTokenizer.from_pretrained(cfg["figures"]["tokenizer"])
    fig_dir = run_dir / "figures"
    fig_dir.mkdir()
    tokens, sequence_figures, seq_hashes = {}, {}, {}
    for condition in cfg["run_conditions"]:
        lo, hi = cfg["conditions"][condition]["id_range"]
        seqs = load_or_create_sequences(
            condition, [lo, hi], cfg["n_sequences"], cfg["seq_len"], cfg["seed"],
            cfg["sample_without_replacement"], cfg["bos_token_id"], paths["sequences_dir"],
        )
        tokens[condition] = prepend_bos(seqs, cfg["bos_token_id"]) if cfg["prepend_bos"] else seqs
        seq_hashes[condition] = sequences_hash(seqs)
        sequence_figures[condition] = f"{fig_dir.name}/{condition}_sequences.png"
        repeats = "no repeats" if cfg["sample_without_replacement"] else "repeats allowed"
        plot_sequences(
            tokens[condition], tokenizer, run_dir / sequence_figures[condition],
            f"{condition}: {'BOS + ' if cfg['prepend_bos'] else ''}{cfg['seq_len']} random ids from [{lo}, {hi}] "
            f"({repeats} within a sequence)",
            cfg["figures"]["n_example_sequences"], cfg["bos_token_id"] if cfg["prepend_bos"] else None,
        )

    # Ties this run's outputs to the exact sequence arrays (same hash as data/sequences/{condition}.json).
    (run_dir / "sequences.json").write_text(json.dumps(seq_hashes, indent=2) + "\n")

    backends = [cfg["primary_backend"]] + [b for b in cfg["backends"] if b != cfg["primary_backend"]]
    runs: dict[str, dict[str, BackendRun]] = {c: {} for c in cfg["run_conditions"]}
    figures: dict[str, dict[str, list[str]]] = {c: {} for c in cfg["run_conditions"]}
    for backend in backends:
        bcfg = cfg["backends"][backend]
        model = load_model(backend, bcfg["model_name"], cfg["device"], cfg["dtype"], bcfg.get("attn_implementation"))
        if bos_token_id(model) != cfg["bos_token_id"]:
            raise ValueError(f"{backend}: model BOS id {bos_token_id(model)} != config bos_token_id {cfg['bos_token_id']}")

        for condition, toks in tokens.items():
            attn = get_final_attention(model, toks, cfg["batch_size"], cfg["query_position"])
            row_sum_dev = check_rows_sum_to_one(attn, v["row_sum_atol"])
            np.save(run_dir / f"{condition}_attn{bcfg['suffix']}.npy", attn)

            heads = aggregate_heads(attn, cfg["diffuse_threshold"], query_index,
                                    id_cols={"model": cfg["model"], "backend": backend, "condition": condition})
            heads.to_parquet(run_dir / f"{condition}_heads{bcfg['suffix']}.parquet", index=False)

            det_equal, det_max_diff = check_determinism(
                model, toks, attn, v["determinism_n"], cfg["batch_size"], cfg["query_position"]
            )
            runs[condition][backend] = BackendRun(backend, attn, heads, row_sum_dev, det_equal, det_max_diff)
            figures[condition][backend] = make_figures(
                heads, fig_dir, f"{condition}{bcfg['suffix']}",
                f"{condition} [{BACKEND_NAMES[backend]}, n={len(toks)}]", key_len,
            )
            print(f"[{BACKEND_NAMES[backend]}] {condition}: attn {attn.shape}, max |row sum - 1| = {row_sum_dev:.1e}, "
                  f"determinism {'bit-identical' if det_equal else f'max diff {det_max_diff:.1e}'}")
        del model

    write_validation_report(
        run_dir / "validation_report.md", cfg, run_dir.name, runs, figures, sequence_figures, query_index
    )
    print(f"wrote {run_dir / 'validation_report.md'}")

    if cfg["comparison"]["enabled"]:
        attn = {b: {c: runs[c][b].attn for c in cfg["run_conditions"]} for b in backends}
        run_phase2(cfg, run_dir, attn, PROJECT_ROOT)
        print(f"wrote {run_dir / 'phase2_report.md'}")
    return run_dir


if __name__ == "__main__":
    main()
