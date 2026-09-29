"""Smoke tests for exp1ext.plots_tokens: every figure function renders a non-empty PNG on
realistic synthetic inputs, and rejects shape-mismatched or otherwise malformed inputs before
touching matplotlib."""

import numpy as np
import pytest

from exp1ext.plots_tokens import (
    plot_attention_overlay,
    plot_attention_received,
    plot_token_effects,
    plot_token_head_heatmap,
)


class _StubTokenizer:
    """Deterministic decode/convert_ids_to_tokens, standing in for GPT-2's tokenizer so these
    tests don't depend on downloading or loading a real one."""

    def decode(self, ids):
        (tid,) = ids
        return f"t{tid}"

    def convert_ids_to_tokens(self, tid):
        return f"<t{tid}>"


def _rng():
    return np.random.default_rng(0)


# --- plot_attention_overlay ---------------------------------------------------------------


def test_plot_attention_overlay_smoke(tmp_path):
    rng = _rng()
    n, key_len = 4, 33
    tokens = np.empty((n, key_len), dtype=np.int64)
    tokens[:, 0] = 50256
    tokens[:, 1:] = rng.integers(0, 40000, size=(n, key_len - 1))
    weights = rng.dirichlet(np.ones(key_len), size=n)  # each row sums to 1, in [0, 1]
    row_labels = [f"A_rare #{i}" for i in range(n)]

    path = tmp_path / "overlay.png"
    plot_attention_overlay(tokens, weights, row_labels, _StubTokenizer(), path, "overlay smoke", bos_id=50256)

    assert path.exists() and path.stat().st_size > 0


# --- plot_token_effects --------------------------------------------------------------------


def _token_effects_inputs(n_byte=30, n_common=40, n_rare=60, seed=0):
    rng = np.random.default_rng(seed)
    byte_ids = np.arange(n_byte)
    common_ids = rng.choice(np.arange(256, 1000), size=n_common, replace=False)
    rare_ids = rng.choice(np.arange(1000, 40000), size=n_rare, replace=False)
    token_ids = np.concatenate([byte_ids, common_ids, rare_ids])
    classes = np.array(["byte"] * n_byte + ["common"] * n_common + ["rare"] * n_rare)
    texts = np.array([f"tok{i}" if i % 7 else " " for i in token_ids], dtype=object)  # sprinkle whitespace text
    return token_ids, texts, classes, rng


def test_plot_token_effects_smoke(tmp_path):
    token_ids, texts, classes, rng = _token_effects_inputs()
    n = len(token_ids)

    mean_a = rng.normal(size=n)
    mean_a[rng.random(n) < 0.2] = np.nan  # unmeasured for some tokens
    lo_a, hi_a = mean_a - 0.1, mean_a + 0.1

    mean_b = rng.normal(loc=0.05, scale=0.02, size=n)
    lo_b, hi_b = mean_b - 0.01, mean_b + 0.01

    panels = {
        "rare contexts (Δ sink_mass)": (mean_a, lo_a, hi_a),
        "common contexts": (mean_b, lo_b, hi_b),
    }
    path = tmp_path / "effects.png"
    plot_token_effects(token_ids, texts, classes, panels, path, "token effects smoke", "Δ sink_mass", top_k=10, zero_line=True)

    assert path.exists() and path.stat().st_size > 0


def test_plot_token_effects_rejects_length_mismatch(tmp_path):
    token_ids, texts, classes, rng = _token_effects_inputs()
    mean = rng.normal(size=len(token_ids))
    panels = {"p": (mean, mean, mean)}

    with pytest.raises(ValueError):
        plot_token_effects(token_ids, texts[:-1], classes, panels, tmp_path / "x.png", "t", "y", top_k=5)


def test_plot_token_effects_rejects_wrong_length_panel(tmp_path):
    token_ids, texts, classes, rng = _token_effects_inputs()
    short = rng.normal(size=len(token_ids) - 1)
    panels = {"p": (short, short, short)}

    with pytest.raises(ValueError):
        plot_token_effects(token_ids, texts, classes, panels, tmp_path / "x.png", "t", "y", top_k=5)


def test_plot_token_effects_rejects_empty_panels(tmp_path):
    token_ids, texts, classes, _ = _token_effects_inputs()

    with pytest.raises(ValueError):
        plot_token_effects(token_ids, texts, classes, {}, tmp_path / "x.png", "t", "y", top_k=5)


def test_plot_token_effects_rejects_unknown_class(tmp_path):
    token_ids, texts, classes, rng = _token_effects_inputs()
    classes = classes.copy()
    classes[0] = "weird"
    mean = rng.normal(size=len(token_ids))
    panels = {"p": (mean, mean, mean)}

    with pytest.raises(ValueError):
        plot_token_effects(token_ids, texts, classes, panels, tmp_path / "x.png", "t", "y", top_k=5)


# --- plot_token_head_heatmap ---------------------------------------------------------------


def test_plot_token_head_heatmap_smoke(tmp_path):
    rng = _rng()
    n_tokens, n_heads = 40, 15
    matrix = rng.normal(size=(n_tokens, n_heads))
    matrix[rng.random((n_tokens, n_heads)) < 0.1] = np.nan
    token_labels = [f"tok{i}" for i in range(n_tokens)]
    head_labels = [f"L{layer}H{head}" for layer in range(3) for head in range(5)]

    path = tmp_path / "heatmap.png"
    plot_token_head_heatmap(matrix, token_labels, head_labels, path, "heatmap smoke", "effect")

    assert path.exists() and path.stat().st_size > 0


def test_plot_token_head_heatmap_rejects_label_mismatch(tmp_path):
    matrix = np.zeros((3, 2))

    with pytest.raises(ValueError):
        plot_token_head_heatmap(matrix, ["a", "b", "c"], ["h1"], tmp_path / "x.png", "t", "cbar")


# --- plot_attention_received ---------------------------------------------------------------


def test_plot_attention_received_smoke(tmp_path):
    rng = _rng()
    ids = np.concatenate([np.arange(50), rng.choice(np.arange(1000, 40000), size=30, replace=False)])
    texts = np.array([f"id{i}" if i % 5 else "\n" for i in ids], dtype=object)
    values = rng.random(len(ids))
    values[rng.random(len(ids)) < 0.1] = np.nan
    counts = rng.integers(1, 100, size=len(ids))

    path = tmp_path / "received.png"
    plot_attention_received(ids, texts, values, counts, path, "received smoke", "mean attention", top_k=20)

    assert path.exists() and path.stat().st_size > 0


def test_plot_attention_received_rejects_length_mismatch(tmp_path):
    ids = np.arange(10)
    texts = np.array([f"id{i}" for i in ids], dtype=object)
    values = np.zeros(10)
    counts = np.zeros(9)

    with pytest.raises(ValueError):
        plot_attention_received(ids, texts, values, counts, tmp_path / "x.png", "t", "x", top_k=5)
