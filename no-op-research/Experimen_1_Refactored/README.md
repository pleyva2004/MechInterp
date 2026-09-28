# Experiment 1 (refactored): random-token attention baseline, 3-condition design

GPT-2 small only. For each random-token sequence we take the attention row of the final
query position, compute per-head metrics (sink / previous / self mass, max weight, entropy),
label each (sequence, head), and aggregate per head. Every run executes in both
TransformerLens (`HookedTransformer`) and Hugging Face (`attn_implementation="eager"`) on the
same token-ID tensors, and the validation report checks that the two agree.

Supersedes the earlier `../experiment_1_refactored/` (N=100, 3 labels, no BOS, sampling with
replacement over the full vocab). Its folder and results are left untouched.

| Condition | ID range (inclusive) | Role |
|---|---|---|
| A_rare | 1000–39999 | Primary baseline |
| B_common | 0–999 | Professor's reference (includes byte tokens 0–255) |
| C_common_clean | 256–999 | Reference without byte tokens |

## Run

```
uv sync
uv run python -m exp1.run --config configs/exp1.yaml
uv run pytest
```

`configs/exp1.yaml` is the only place parameters live; `run_conditions` picks which conditions
run. Phase 1 ran `[A_rare]` (`results/20260927-235708_exp1`). The config is now set for Phase 2:
`[B_common, C_common_clean]` with `comparison.enabled: true`, reusing A_rare's attention from the
Phase 1 folder via `comparison.reference_runs` (the run refuses if that folder's config differs).
Phase 2 adds `comparison.parquet`, `split_half.parquet`, `phase2_report.md` and `figures/phase2_*.png`.

## Layout

| Path | Contents |
|---|---|
| `exp1/sequences.py` | `sample_sequences` (without replacement within a row), per-condition seeds, hashed cache in `data/sequences/`, `prepend_bos` |
| `exp1/backends.py` | `load_model`, `get_final_attention` for TL and HF, row-sum check |
| `exp1/metrics.py` | `compute_metrics`, `classify`, `LABELS` |
| `exp1/aggregate.py` | `aggregate_heads` → one row per (layer, head) |
| `exp1/plots.py` | layer × head heatmaps |
| `exp1/show_sequences.py` | example-sequence figure (port of the old `show_prompts.py`), BOS shown at position 0 |
| `exp1/validation.py` | Phase 1 checks and `validation_report.md` |
| `exp1/compare.py` | `compare_conditions` (Mann-Whitney U + rank-biserial), `bh_fdr`, `split_half`, bootstrap CIs, `label_shift` |
| `exp1/phase2.py` | Phase 2 analysis, figures, `phase2_report.md` |
| `exp1/runlog.py` | timestamped run folder, config copy, `versions.json` |
| `exp1/run.py` | entrypoint |

Each run writes `results/<YYYYMMDD-HHMMSS>_exp1/`: `config.yaml`, `versions.json`,
`{condition}_attn.npy` / `{condition}_heads.parquet` (TransformerLens, the primary backend),
the same with an `_hf` suffix (Hugging Face), `figures/` (`{condition}_sequences.png` plus
mode-label / sink-mass / entropy heatmaps per backend), and `validation_report.md`.

## Notes

- TransformerLens is pinned to `<4`: 4.x removed `HookedTransformer`.
- BOS (50256) is prepended to the ID array by hand; nothing is ever decoded to text and
  re-tokenized, and no library default BOS behavior is relied on.
- Sequence caches are verified against the config and a SHA-256 on every load; a mismatch
  raises instead of regenerating. Delete the files in `data/sequences/` explicitly to resample.
