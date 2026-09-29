"""Extract final-query attention scores/pattern (and, for target heads, Q/K vectors) from GPT-2
small via TransformerLens, plus the validation checks that confirm the extraction is correct.

Follows exp1.backends._patterns_tl's conventions exactly: a raw int64 token-ID tensor (BOS
already prepended by the caller), no attention mask, torch.inference_mode, batched on the model's
device. Rather than exp1.backends.get_final_attention's model.run_with_cache (which caches the
*full* [batch, head, q, k] tensor for every requested hook), we attach read-only forward hooks
that slice out only the final query row (and, for hook_q/hook_k, only the target layers) before
storing anything -- this is a strictly larger set of hooks than backends uses, but since every
hook here only reads its tensor and returns None (leaving it untouched), the forward pass, and
therefore hook_pattern itself, is numerically identical to get_final_attention's -- this is
validation check V1.
"""
from __future__ import annotations

import numpy as np
import torch

from exp1extended.measures import qk_scores, summarize


def run_final(
    model: torch.nn.Module,
    tokens: np.ndarray,
    batch_size: int,
    target_heads: list[tuple[int, int]],
    diffuse_threshold: float,
    keep_full: bool = False,
    keep_qk: bool = False,
) -> dict[str, np.ndarray]:
    """tokens: int64 [n, key_len], BOS already prepended. Returns per-batch-concatenated arrays:

    - always: measures.summarize's keys (sink_logit, prev_logit, other_gap, entropy, sink_mass,
      prev_mass, label, other_argmax), each [n, n_layers, n_heads], plus "target_scores" float32
      [n, T, key_len] (final-query scores of `target_heads`, T = len(target_heads), given order).
    - keep_full: "pattern" and "scores" float32 [n, n_layers, n_heads, key_len] (final-query rows,
      all heads).
    - keep_qk: "q" float32 [n, T, d_head] (final query position) and "k" float32
      [n, T, key_len, d_head] (every key position), for the target heads only.

    Memory is bounded per batch: hooks slice to the final query row (and, for q/k, to the target
    layers) before anything is stored, so we never hold a full [n, L, H, key_len, key_len] tensor.
    """
    tokens_t = torch.as_tensor(tokens, dtype=torch.long)
    if tokens_t.ndim != 2:
        raise ValueError(f"tokens must be [n, key_len], got shape {tuple(tokens_t.shape)}")
    n, key_len = tokens_t.shape
    n_layers, n_heads, d_head = model.cfg.n_layers, model.cfg.n_heads, model.cfg.d_head
    target_layers = sorted({layer for layer, _ in target_heads})
    device = next(model.parameters()).device

    summary_batches: list[dict[str, np.ndarray]] = []
    target_scores_batches: list[np.ndarray] = []
    pattern_batches: list[np.ndarray] = []
    scores_batches: list[np.ndarray] = []
    q_batches: list[np.ndarray] = []
    k_batches: list[np.ndarray] = []

    with torch.inference_mode():
        for start in range(0, n, batch_size):
            batch = tokens_t[start : start + batch_size].to(device)
            b = batch.shape[0]

            scores_full = torch.empty((b, n_layers, n_heads, key_len), dtype=torch.float32)
            pattern_full = torch.empty((b, n_layers, n_heads, key_len), dtype=torch.float32)
            q_store: dict[int, torch.Tensor] = {}
            k_store: dict[int, torch.Tensor] = {}

            def make_scores_hook(layer: int):
                def hook(tensor: torch.Tensor, hook) -> None:  # noqa: ANN001 - TL hook signature
                    # [batch, head, q_pos, k_pos] -> final query row only.
                    scores_full[:, layer, :, :] = tensor[:, :, -1, :].detach().float().cpu()

                return hook

            def make_pattern_hook(layer: int):
                def hook(tensor: torch.Tensor, hook) -> None:  # noqa: ANN001
                    pattern_full[:, layer, :, :] = tensor[:, :, -1, :].detach().float().cpu()

                return hook

            def make_q_hook(layer: int):
                def hook(tensor: torch.Tensor, hook) -> None:  # noqa: ANN001
                    # [batch, pos, head, d_head] -> final position only.
                    q_store[layer] = tensor[:, -1, :, :].detach().float().cpu()

                return hook

            def make_k_hook(layer: int):
                def hook(tensor: torch.Tensor, hook) -> None:  # noqa: ANN001
                    k_store[layer] = tensor.detach().float().cpu()

                return hook

            fwd_hooks = []
            for layer in range(n_layers):
                fwd_hooks.append((f"blocks.{layer}.attn.hook_attn_scores", make_scores_hook(layer)))
                fwd_hooks.append((f"blocks.{layer}.attn.hook_pattern", make_pattern_hook(layer)))
            if keep_qk:
                for layer in target_layers:
                    fwd_hooks.append((f"blocks.{layer}.attn.hook_q", make_q_hook(layer)))
                    fwd_hooks.append((f"blocks.{layer}.attn.hook_k", make_k_hook(layer)))

            model.run_with_hooks(batch, fwd_hooks=fwd_hooks, return_type=None)

            scores_np = scores_full.numpy()
            pattern_np = pattern_full.numpy()
            summary_batches.append(summarize(scores_np, pattern_np, diffuse_threshold))

            target_scores = np.stack(
                [scores_np[:, layer, head, :] for layer, head in target_heads], axis=1
            )  # [b, T, key_len]
            target_scores_batches.append(target_scores)

            if keep_full:
                pattern_batches.append(pattern_np)
                scores_batches.append(scores_np)

            if keep_qk:
                q_batch = np.stack(
                    [q_store[layer][:, head, :].numpy() for layer, head in target_heads], axis=1
                )  # [b, T, d_head]
                k_batch = np.stack(
                    [k_store[layer][:, :, head, :].numpy() for layer, head in target_heads], axis=1
                )  # [b, T, key_len, d_head]
                q_batches.append(q_batch)
                k_batches.append(k_batch)

    out: dict[str, np.ndarray] = {
        key: np.concatenate([batch[key] for batch in summary_batches]) for key in summary_batches[0]
    }
    out["target_scores"] = np.concatenate(target_scores_batches)
    if keep_full:
        out["pattern"] = np.concatenate(pattern_batches)
        out["scores"] = np.concatenate(scores_batches)
    if keep_qk:
        out["q"] = np.concatenate(q_batches)
        out["k"] = np.concatenate(k_batches)
    return out


def check_bitmatch(pattern: np.ndarray, saved_attn: np.ndarray) -> tuple[bool, float]:
    """V1: pattern (freshly extracted) is bit-identical to saved_attn (Exp 1's saved array)."""
    if pattern.shape != saved_attn.shape:
        raise ValueError(f"pattern {pattern.shape} and saved_attn {saved_attn.shape} must match")
    ok = bool(np.array_equal(pattern, saved_attn))
    max_abs_diff = float(np.abs(pattern.astype(np.float64) - saved_attn.astype(np.float64)).max())
    return ok, max_abs_diff


def check_softmax(scores: np.ndarray, pattern: np.ndarray, atol: float) -> tuple[bool, float]:
    """V2: softmax(scores) matches pattern, and pattern rows sum to 1, both within atol."""
    if scores.shape != pattern.shape:
        raise ValueError(f"scores {scores.shape} and pattern {pattern.shape} must match")
    s64 = scores.astype(np.float64)
    expected = np.exp(s64 - s64.max(axis=-1, keepdims=True))
    expected /= expected.sum(axis=-1, keepdims=True)
    pattern64 = pattern.astype(np.float64)
    softmax_diff = float(np.abs(expected - pattern64).max())
    row_sum_diff = float(np.abs(pattern64.sum(axis=-1) - 1.0).max())
    max_abs_diff = max(softmax_diff, row_sum_diff)
    return max_abs_diff <= atol, max_abs_diff


def check_qk(
    q: np.ndarray, k: np.ndarray, target_scores: np.ndarray, d_head: int, atol: float, rtol: float = 0.0
) -> tuple[bool, float]:
    """V3: qk_scores(q, k, d_head) matches target_scores within atol + rtol * |score|. The relative term is
    there because TL computes scores in float32, whose rounding grows with the score (~2e-6 of |score| for
    GPT-2 small); a wrong head, position or scale would be off by O(1), far outside it."""
    computed = qk_scores(q, k, d_head)
    if computed.shape != target_scores.shape:
        raise ValueError(f"qk_scores shape {computed.shape} != target_scores shape {target_scores.shape}")
    ref = target_scores.astype(np.float64)
    diff = np.abs(computed.astype(np.float64) - ref)
    return bool(np.all(diff <= atol + rtol * np.abs(ref))), float(diff.max())


def check_bos_key(k: np.ndarray, atol: float) -> tuple[bool, float]:
    """V4: k[:, :, 0, :] (the BOS key vector) is identical across sequences.

    Position 0 always holds BOS under causal attention, so its residual stream -- and therefore
    its key vector at every layer -- depends only on the BOS token/position embedding, never on
    the rest of the sequence.
    """
    bos_key = k[:, :, 0, :].astype(np.float64)  # [n, T, d_head]
    max_abs_diff = float(np.abs(bos_key - bos_key[:1]).max())
    return max_abs_diff <= atol, max_abs_diff
