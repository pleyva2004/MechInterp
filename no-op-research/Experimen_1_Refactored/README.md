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

## exp1ext: which tokens cause the rare-vs-common head changes

`uv run python -m exp1ext.run --config configs/exp1ext.yaml` → `results/<YYYYMMDD-HHMMSS>_exp1ext/exp1ext_report.md`.
Predictions were fixed before the first run in `exp1ext/PREDICTIONS.md`.

- **Stage 0** (exp1 arrays, no model runs): exp1 sequences with each token shaded by the attention the final token
  puts on it; attention received per key token id, as p_k and log(p_k / p_sink).
- **Stage 1**: 2×2 of context class × final-token class (rare 1000–39999 / common 256–999), rows paired across cells,
  so each head's rare→common change splits into a final-token (query) part, a context part and an interaction;
  plus a dose-response over k common context tokens.
- **Stage 2**: per-token sweeps over 100 base sequences per class: every id 0–999 and 2000 rare ids in the final slot
  (under the causal mask this changes only the query), and substitutions at positions 31 and 16.
- **Stage 3**: per-token effects regressed on token features, including the model's own log-prior for the token.
- Tracked heads come from a rule (label-mix TV ≥ 0.2 in exp1), not a hand list; TL runs everything, HF re-runs
  stage 1 and a subset of stage 2 as the cross-library check.

| Path | Contents |
|---|---|
| `exp1ext/design.py` | builds the 2×2 cells, dose-response and position sweeps from hash-cached blocks in `data/sequences/exp1ext/` |
| `exp1ext/extract.py` | chunked extraction: per-head metrics for all heads, full rows only for tracked heads |
| `exp1ext/effects.py` | `decompose_2x2`, `per_token_effects`, `split_reliability`, `attention_received` |
| `exp1ext/features.py` | `token_features`, `fit_feature_model` |
| `exp1ext/plots_cells.py`, `exp1ext/plots_tokens.py` | 2×2 / decomposition / dose-response figures; per-token graphs, heatmaps, overlays |
| `exp1ext/report.py`, `exp1ext/run.py` | report writer, entrypoint |
