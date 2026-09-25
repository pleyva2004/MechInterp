#%% [markdown]
## Path patching walkthrough (IOI paper, Appendix B / Algorithm 1)
#
# CPU-friendly entry point for studying `path_patching()` in
# `easy_transformer/ioi_utils.py`. The repo's own `experiments.py` asserts a
# CUDA GPU, so this file reproduces the path patching parts on CPU instead.
#
# Run cell-by-cell in VS Code / Cursor ("Run Cell" above each `#%%`), or all at
# once with `uv run python path_patching_walkthrough.py`.
#
# Notation (paper -> code):
#   x_orig  -> D_orig = ioi_dataset   (p_IOI: "When Mary and John went..., John gave a drink to")
#   x_new   -> D_new  = abc_dataset   (p_ABC: same template, three unrelated names A, B, C)
#   h       -> sender_heads           [(layer, head)], or (layer, None) for an MLP
#   R       -> receiver_hooks         [(hook_name, head_idx)], e.g. a head's hook_q, or the final resid
#   metric  -> logit_diff             mean over prompts of logit(IO) - logit(S) at the END token

#%%
import random
from functools import partial

import numpy as np
import torch
from tqdm import tqdm

from easy_transformer.EasyTransformer import EasyTransformer
from easy_transformer.ioi_dataset import IOIDataset
from easy_transformer.ioi_utils import path_patching, logit_diff, show_pp
from easy_transformer.ioi_circuit_extraction import CIRCUIT

torch.set_grad_enabled(False)
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

DEVICE = "cpu"
N = 50  # paper uses N > 200; smaller keeps CPU sweeps to a few minutes

#%% [markdown]
## 1. Model
# `use_attn_result` exposes each head's output separately at `blocks.L.attn.hook_result`
# (shape [batch, pos, head, d_model]) so a single head can be a *sender*.
# `use_headwise_qkv_input` gives every head its own copy of the residual stream as
# q/k/v input, so a single head's `hook_q`/`hook_k`/`hook_v` can be a *receiver*.

model = EasyTransformer.from_pretrained("gpt2", device=DEVICE)
model.set_use_attn_result(True)
model.set_use_headwise_qkv_input(True)

#%% [markdown]
## 2. x_orig and x_new
# p_ABC keeps the template (grammar, positions) but replaces IO, S1 and S2 with
# three random names, destroying the information needed to solve IOI.

ioi_dataset = IOIDataset(
    prompt_type="mixed", N=N, tokenizer=model.tokenizer, prepend_bos=False
)
abc_dataset = (
    ioi_dataset.gen_flipped_prompts(("IO", "RAND"))
    .gen_flipped_prompts(("S", "RAND"))
    .gen_flipped_prompts(("S1", "RAND"))
)

for i in range(3):
    print("x_orig:", ioi_dataset.sentences[i])
    print("x_new: ", abc_dataset.sentences[i])
    print()

# word_idx maps a role to its token position in each prompt; path patching only
# patches at the positions you ask for (e.g. "end", "S2").
print({k: v[:5].tolist() for k, v in ioi_dataset.word_idx.items()})

#%% [markdown]
## 3. Baseline
# Paper reports ~3.56 over 100k examples; expect roughly that here.

model.reset_hooks()
baseline = logit_diff(model, ioi_dataset).item()
print(f"Baseline logit diff on p_IOI: {baseline:.3f}")

#%% [markdown]
## 4. One path patching call, step by step
# Sender h = 9.9 (a Name Mover), receiver R = final residual stream at END,
# i.e. the direct path 9.9 -> logits (Figure 3a).
#
# Inside `path_patching` (ioi_utils.py):
#   Pass A: run x_new, cache the sender's output (hook_result).        [Alg. 1 line 1]
#   Pass B: run x_orig, cache everything.                              [Alg. 1 line 2]
#   Pass C: run x_orig with every head's q/k/v frozen to pass B values  [Alg. 1 lines 3-10]
#           (so every head's output is its x_orig output), except the
#           sender whose output at `positions` is overwritten with its
#           x_new value. MLPs and LayerNorms are recomputed, so the
#           sender's change reaches R only via residual + MLP paths.
#           The receiver's recomputed value is cached here.
#   Pass D: returned as hooks on the model: the next forward pass on    [Alg. 1 lines 12-20]
#           x_orig patches R to its pass-C value and recomputes
#           everything downstream. `logit_diff(model, ...)` below *is* pass D.

model.reset_hooks()
model = path_patching(
    model=model,
    D_new=abc_dataset,
    D_orig=ioi_dataset,
    sender_heads=[(9, 9)],
    receiver_hooks=[(f"blocks.{model.cfg.n_layers - 1}.hook_resid_post", None)],
    positions=["end"],
    freeze_mlps=False,
    have_internal_interactions=False,
)
patched = logit_diff(model, ioi_dataset).item()
model.reset_hooks()
print(f"9.9 -> logits: {baseline:.3f} -> {patched:.3f} ({100 * (patched - baseline) / baseline:+.1f}%)")

#%% [markdown]
## 5. Sweep every head as sender (reproduces Figure 3b)
# Receiver fixed; each of the 144 heads (and 12 MLPs) is patched in turn.
# Expect big drops (blue in the paper) at 9.9, 9.6, 10.0 (Name Movers) and big
# increases (red) at 10.7, 11.10 (Negative Name Movers).


def sweep_senders(receiver_hooks, position):
    """% change in logit diff when patching (sender -> receiver_hooks) for every sender."""
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    head_results = torch.zeros(n_layers, n_heads)
    mlp_results = torch.zeros(n_layers)
    for layer in tqdm(range(n_layers)):
        for head in [None] + list(range(n_heads)):
            model.reset_hooks()
            path_patching(
                model=model,
                D_new=abc_dataset,
                D_orig=ioi_dataset,
                sender_heads=[(layer, head)],
                receiver_hooks=receiver_hooks,
                positions=[position],
                freeze_mlps=False,
                have_internal_interactions=False,
            )
            delta = 100 * (logit_diff(model, ioi_dataset).item() - baseline) / baseline
            if head is None:
                mlp_results[layer] = delta
            else:
                head_results[layer, head] = delta
    model.reset_hooks()
    return head_results, mlp_results


heads_to_logits, mlps_to_logits = sweep_senders(
    receiver_hooks=[(f"blocks.{model.cfg.n_layers - 1}.hook_resid_post", None)],
    position="end",
)
show_pp(
    heads_to_logits,
    title="Direct effect on logit difference (sender -> logits @ END)",
    bartitle="% change in logit diff",
    xlabel="Head",
    ylabel="Layer",
)

#%% [markdown]
## 6. Sweep senders -> Name Mover queries (reproduces Figure 4b)
# Now R = the query inputs of the Name Movers at END. Heads that matter here
# change *where the Name Movers attend* rather than writing to the logits
# directly. Expect the S-Inhibition heads 7.3, 7.9, 8.6, 8.10 to stand out.

name_movers = [(9, 9), (9, 6), (10, 0)]
heads_to_nm_q, _ = sweep_senders(
    receiver_hooks=[(f"blocks.{l}.attn.hook_q", h) for l, h in name_movers],
    position="end",
)
show_pp(
    heads_to_nm_q,
    title="Effect of sender -> Name Mover queries @ END",
    bartitle="% change in logit diff",
    xlabel="Head",
    ylabel="Layer",
)

#%% [markdown]
## Next steps
# - Section 3.3: receivers = S-Inhibition heads' values at S2:
#   `[(f"blocks.{l}.attn.hook_v", h) for l, h in CIRCUIT["s2 inhibition"]]`, position="S2"
#   (this is the second `plot_path_patching` call in experiments.py).
# - Set `freeze_mlps=True` to also cut paths through MLPs (Appendix B, footnote on MLPs).
# - The same algorithm in modern TransformerLens: ARENA's IOI chapter (1.4.1) implements
#   path patching from scratch and is worth doing after reading this code.
print({k: v for k, v in CIRCUIT.items()})
