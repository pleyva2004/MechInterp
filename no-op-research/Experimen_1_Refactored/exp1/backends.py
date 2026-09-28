"""Load GPT-2 in TransformerLens or Hugging Face and pull out final-query attention rows.

Both backends receive the same int64 token-ID tensor (BOS already prepended by the caller),
so any difference between their patterns is numerical, never a difference in input.
"""
from __future__ import annotations

import numpy as np
import torch

BACKENDS = ("transformer_lens", "huggingface")


def set_determinism(seed: int) -> None:
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def load_model(
    backend: str, model_name: str, device: str, dtype: str, attn_implementation: str | None = None
) -> torch.nn.Module:
    torch_dtype = getattr(torch, dtype)
    if backend == "transformer_lens":
        from transformer_lens import HookedTransformer

        # Default weight processing (fold_ln, center_writing_weights, ...) leaves attention
        # patterns unchanged, which the HF comparison checks numerically.
        model = HookedTransformer.from_pretrained(model_name, device=device, dtype=torch_dtype)
        # With left padding TL builds an attention mask from the pad id, which for GPT-2 is also the
        # BOS id we prepend, so position 0 would be masked out (see _patterns_tl).
        if model.tokenizer.padding_side != "right":
            raise ValueError(f"TL tokenizer padding_side is {model.tokenizer.padding_side!r}; BOS would be masked")
    elif backend == "huggingface":
        from transformers import AutoModel

        # "eager" is the only HF attention implementation that returns the attention weights.
        model = AutoModel.from_pretrained(model_name, attn_implementation=attn_implementation, dtype=torch_dtype)
        model = model.to(device)
    else:
        raise ValueError(f"unknown backend {backend!r}; expected one of {BACKENDS}")
    return model.eval()


def model_shape(model: torch.nn.Module) -> tuple[int, int]:
    """(n_layers, n_heads) for either backend."""
    if hasattr(model, "cfg"):
        return model.cfg.n_layers, model.cfg.n_heads
    return model.config.n_layer, model.config.n_head


def bos_token_id(model: torch.nn.Module) -> int:
    if hasattr(model, "cfg"):
        return model.tokenizer.bos_token_id
    return model.config.bos_token_id


def _patterns_tl(model: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
    # Token tensors go straight into the forward pass, so TL never adds a BOS of its own. No
    # attention_mask is passed: with GPT-2's right padding side TL builds none, which matters
    # because our BOS id doubles as GPT-2's pad id and must not be masked out.
    _, cache = model.run_with_cache(
        batch, names_filter=lambda name: name.endswith("hook_pattern"), return_type=None
    )
    return torch.stack([cache["pattern", layer] for layer in range(model.cfg.n_layers)], dim=1)


def _patterns_hf(model: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
    out = model(input_ids=batch, output_attentions=True)
    return torch.stack(out.attentions, dim=1)


def get_final_attention(
    model: torch.nn.Module, tokens: np.ndarray | torch.Tensor, batch_size: int, query_position: int = -1
) -> np.ndarray:
    """tokens: int64 [n_seq, key_len], fed to the model as-is (never decoded and re-tokenized).

    Returns float32 [n_seq, n_layers, n_heads, key_len]: the post-softmax attention that the
    query at `query_position` pays to every key, computed in batches of `batch_size`.
    """
    tokens = torch.as_tensor(tokens, dtype=torch.long)
    if tokens.ndim != 2:
        raise ValueError(f"tokens must be [n_seq, key_len], got shape {tuple(tokens.shape)}")
    device = next(model.parameters()).device
    extract = _patterns_tl if hasattr(model, "cfg") else _patterns_hf

    rows = []
    with torch.inference_mode():
        for start in range(0, len(tokens), batch_size):
            patterns = extract(model, tokens[start : start + batch_size].to(device))  # [b, layer, head, q, k]
            rows.append(patterns[:, :, :, query_position, :].float().cpu().numpy())
    return np.concatenate(rows)


def check_rows_sum_to_one(attn: np.ndarray, atol: float) -> float:
    """Raise if any attention row is not a probability distribution; return the max |sum - 1|."""
    max_dev = float(np.abs(attn.sum(axis=-1, dtype=np.float64) - 1.0).max())
    if not max_dev <= atol:
        raise AssertionError(f"attention rows do not sum to 1: max |sum - 1| = {max_dev:.3e} > atol {atol:g}")
    if (attn < 0).any():
        raise AssertionError("attention contains negative weights; not a post-softmax pattern")
    return max_dev
