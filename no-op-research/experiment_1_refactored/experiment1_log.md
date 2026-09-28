# Experiment 1 — Random Token Baseline: log

Goal, per the experimental design doc: characterize attention behavior on
random-token input and establish a quantitative no-op baseline — for each
head, classify what the final token attends to (first / previous / other),
across every layer x head in GPT-2 small.

## Where the code lives

All in `no-op-research/`, one uv project (`pyproject.toml` / `uv.lock`):
torch, transformers, transformer-lens, numpy, matplotlib.

| File | Role |
|---|---|
| `common.py` | Shared logic both backend scripts call. |
| `exp1_hf.py` | Experiment 1 on plain Hugging Face. |
| `exp1_tl.py` | Experiment 1 on TransformerLens 4.x. |
| `compare.py` | Checks the two backends agree. |
| `show_prompts.py` | Renders example prompts, to see what "32 random tokens" actually looks like. |

Two other visuals (a full attention-pattern atlas, a concentration heatmap)
were built and then removed at the user's request — see **Dead ends** below.
`show_prompts.py` is the last graph still in the repo.

### `common.py` — shared pieces

- `N = 100`, `SEQ_LEN = 32`, `SEED = 0` (line 17-19): the experiment's fixed
  parameters. Same seed in both backend scripts, so `make_tokens` returns
  identical input either way.
- `make_tokens(vocab_size)` (line 25): builds the `[N, SEQ_LEN]` random token
  tensor from a seeded `torch.Generator`. Raw ids, no BOS token added.
- `classify(attn)` (line 35): input `[N, layer, head, key]` = attention the
  *final* position pays to every key. Returns each head's argmax target per
  sequence, bucketed into 0=first token, 1=previous token, 2=other.
- `analyze(attn, name)` (line 49): runs `classify`, prints the summary
  percentages and the layer x head F/P/O grid, saves the raw attention to
  `results/exp1_<name>.npz`, and calls `plot`.
- `plot(mean_first, head_class, path)` (line 66): the two-panel figure —
  mean attention on the first token, and each head's majority target — saved
  to `results/exp1_<name>.png`.

### `exp1_hf.py` / `exp1_tl.py` — the two backends

These differ only in how they pull attention out of the model; everything
else is `common.analyze`.

- HF: `AutoModelForCausalLM.from_pretrained("gpt2", attn_implementation="eager")`,
  then `model(tokens, output_attentions=True)`. `attn_implementation="eager"`
  is required — the default fast kernel doesn't return attention weights.
- TL: `TransformerBridge.boot_transformers("gpt2", device="cpu")`, then
  `model.run_with_cache(tokens, names_filter=lambda n: n.endswith("hook_pattern"))`.
  TL 4.x removed `HookedTransformer`; `TransformerBridge` wraps the actual HF
  model and returns raw HF weights by default (no LayerNorm folding / weight
  centering unless `enable_compatibility_mode()` is called) — this is why the
  two backends agree so closely below.

Both slice out the final query position (`attn[..., -1, :]`) before handing
`[N, layer, head, key]` to `common.analyze`.

### `compare.py`

Loads both `.npz` files, reports the largest per-element attention
difference and the fraction of per-sequence head classifications that
agree between backends.

### `show_prompts.py`

Decodes 4 of the actual seeded sequences (same `make_tokens` call, so
identical to what the experiment runs on) back to text with the GPT-2
tokenizer, one box per token. `text_width_in` (line 43) measures each
token's real rendered width via matplotlib's renderer so every box is sized
to fit its own text — no fixed-width guessing, no clipping or overlap
regardless of token length. Wraps a sequence onto a new line when 32 boxes
don't fit one row. Saves `results/exp1_prompts.png`.

## How to run

```
uv run python exp1_hf.py       # -> results/exp1_hf.npz, exp1_hf.png
uv run python exp1_tl.py       # -> results/exp1_tl.npz, exp1_tl.png
uv run python compare.py       # numerical agreement between the two
uv run python show_prompts.py  # -> results/exp1_prompts.png
```

## Results (N=100, one seed=0, GPT-2 small)

- 76.4% of heads primarily attend to the first token
- 6.9% primarily attend to the previous token
- 16.7% primarily attend elsewhere
- Layer 0 is almost entirely "other"; layers 5-10 are almost entirely "first
  token."

**Backend agreement:** largest attention difference between HF and TL is
5.9e-6 (float32 noise); all 172,800 per-sequence head classifications
(100 sequences x 12 layers x 12 heads x 2 backends) match exactly.

**Caveats:** single seed, N=100, no error bars — a smoke test / working
baseline, not a result to quote yet. Multiple seeds would be the natural
next step before treating these percentages as stable.

## Dead ends (built, then removed — kept here so they aren't silently redone)

1. **Full attention-pattern atlas** (`show_attention_grid.py`,
   `results/exp1_attention_atlas.png`, removed): a 12x12 grid of small
   query x key heatmaps, one per head, averaged over the N sequences.
   Rejected as not helpful — most of each cell is the causal-mask triangle,
   and averaging over all 32 query rows dilutes the one row (the final
   position) that the experiment actually measures. The previous-token
   heads' diagonal was visible; the first-token sink pattern was not.
2. **Attention concentration heatmap** (`show_concentration.py`,
   `results/exp1_concentration.png`, removed): a single 12x12 heatmap of
   `attn.max(axis=-1).mean(axis=0)` (mean top-attention weight per head,
   regardless of which key it lands on), with the F/P/O letter overlaid per
   cell. Built to fix a real gap in the original `plot()` panel: for heads
   classified P or O, "mean attention on the first token" is close to zero
   *by construction* (they don't attend to the first token), so that panel
   can't show how concentrated those heads actually are on their own
   target. Verified numerically: F heads' `mean_first` (0.583) ~=
   `concentration` (0.594), but P heads' `mean_first` (0.091) vs
   `concentration` (0.488), O heads' 0.044 vs 0.331 — the gap is real, only
   matters off the F diagonal. Removed at the user's request; worth
   reconsidering if Experiment 2/3's Δattention analysis needs to compare
   magnitude across heads with different targets.

## Open items for Experiment 2

- Per the design doc: fix the random tokens, insert " I like sushi" (or
  similar) at a fixed position, and diff attention against this baseline.
  `common.make_tokens` and the HF/TL extraction code in `exp1_hf.py` /
  `exp1_tl.py` should be reusable as-is for the perturbed run.
- Decide whether Δattention is measured as a change in the classified
  target, a change in mass on the injected tokens, or (per the dead end
  above) a change in concentration — the current classification alone
  won't distinguish "shifted target" from "same target, different
  strength."
