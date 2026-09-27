"""Experiment 1 with plain Hugging Face.   Run:  uv run python exp1_hf.py"""
import torch
from transformers import AutoModelForCausalLM

from common import analyze, make_tokens

# "eager" attention is what makes HF return attention weights (the default fast kernel does not).
model = AutoModelForCausalLM.from_pretrained("gpt2", attn_implementation="eager").eval()
tokens = make_tokens(model.config.vocab_size)  # [N, 32]

with torch.no_grad():
    out = model(tokens, output_attentions=True)

# out.attentions is a tuple with one [batch, head, query, key] tensor per layer.
attn = torch.stack(out.attentions)  # [layer, batch, head, query, key]
attn = attn[:, :, :, -1, :]  # keep only the final position as the query -> [layer, batch, head, key]
analyze(attn.permute(1, 0, 2, 3).numpy(), "hf")  # -> [batch, layer, head, key]
