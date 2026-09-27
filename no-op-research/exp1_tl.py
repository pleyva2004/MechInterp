"""Experiment 1 with TransformerLens 4.x.   Run:  uv run python exp1_tl.py"""
import torch
from transformer_lens import TransformerBridge

from common import analyze, make_tokens

model = TransformerBridge.boot_transformers("gpt2", device="cpu")
tokens = make_tokens(model.cfg.d_vocab)  # [N, 32]

with torch.no_grad():
    # names_filter caches only the attention patterns instead of every activation.
    _, cache = model.run_with_cache(tokens, names_filter=lambda name: name.endswith("hook_pattern"))

# One [batch, head, query, key] pattern per layer, looked up by hook name.
attn = torch.stack([cache[f"blocks.{layer}.attn.hook_pattern"] for layer in range(model.cfg.n_layers)])
attn = attn[:, :, :, -1, :]  # keep only the final position as the query -> [layer, batch, head, key]
analyze(attn.permute(1, 0, 2, 3).numpy(), "tl")  # -> [batch, layer, head, key]
