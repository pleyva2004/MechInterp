"""Tests for exp1ext.plots_cells: smoke tests (PNGs render) and input validation."""
from __future__ import annotations

import numpy as np
import pytest

from exp1ext.plots_cells import (
    _group_color, _order_by_group, plot_decomposition, plot_dose_response, plot_interaction_grid,
)

HEADS = ["L7H7", "L2H4", "L4H11", "L9H6", "L0H1"]
GROUPS = ["sink rises", "sink falls", "control", "sink rises", "unclassified"]  # last: unknown -> fallback color
N = len(HEADS)


def _triplet(n, base=0.5, spread=0.1, ci=0.03, seed=0):
    rng = np.random.default_rng(seed)
    mean = np.clip(rng.uniform(base - spread, base + spread, size=n), 0.0, 1.0)
    return mean, np.clip(mean - ci, 0.0, 1.0), np.clip(mean + ci, 0.0, 1.0)


# ---- _order_by_group / _group_color -------------------------------------------

def test_order_by_group_blocks_and_preserves_relative_order():
    groups = ["b", "a", "b", "a", "c"]
    order = _order_by_group(groups)
    ordered = [groups[i] for i in order]
    # blocks: first-seen group order is b, a, c
    assert ordered == ["b", "b", "a", "a", "c"]
    # relative order within a block preserved
    b_indices = [i for i in order if groups[i] == "b"]
    assert b_indices == [0, 2]
    a_indices = [i for i in order if groups[i] == "a"]
    assert a_indices == [1, 3]


def test_group_color_known_and_fallback():
    assert _group_color("sink rises") != _group_color("sink falls")
    assert _group_color("sink falls") != _group_color("control")
    fallback = _group_color("some unseen label")
    assert fallback not in {"sink rises", "sink falls", "control"}
    assert _group_color("also unseen") == fallback  # deterministic fallback


# ---- plot_interaction_grid -----------------------------------------------------

def test_plot_interaction_grid_smoke(tmp_path):
    cells = {k: _triplet(N, base=b, seed=i) for i, (k, b) in enumerate({"AA": 0.2, "AC": 0.5, "CA": 0.3, "CC": 0.6}.items())}
    path = tmp_path / "interaction_grid.png"
    plot_interaction_grid(cells, HEADS, GROUPS, path, title="t", ylabel="mean sink_mass")
    assert path.exists() and path.stat().st_size > 0


def test_plot_interaction_grid_missing_key_raises(tmp_path):
    cells = {k: _triplet(N) for k in ("AA", "AC", "CA")}  # missing CC
    with pytest.raises(ValueError):
        plot_interaction_grid(cells, HEADS, GROUPS, tmp_path / "x.png", title="t", ylabel="y")


def test_plot_interaction_grid_extra_key_raises(tmp_path):
    cells = {k: _triplet(N) for k in ("AA", "AC", "CA", "CC", "EXTRA")}
    with pytest.raises(ValueError):
        plot_interaction_grid(cells, HEADS, GROUPS, tmp_path / "x.png", title="t", ylabel="y")


def test_plot_interaction_grid_groups_length_mismatch_raises(tmp_path):
    cells = {k: _triplet(N) for k in ("AA", "AC", "CA", "CC")}
    with pytest.raises(ValueError):
        plot_interaction_grid(cells, HEADS, GROUPS[:-1], tmp_path / "x.png", title="t", ylabel="y")


def test_plot_interaction_grid_array_length_mismatch_raises(tmp_path):
    cells = {k: _triplet(N) for k in ("AA", "AC", "CA")}
    cells["CC"] = _triplet(N - 1)  # wrong length
    with pytest.raises(ValueError):
        plot_interaction_grid(cells, HEADS, GROUPS, tmp_path / "x.png", title="t", ylabel="y")


def test_plot_interaction_grid_empty_heads_raises(tmp_path):
    cells = {k: (np.array([]), np.array([]), np.array([])) for k in ("AA", "AC", "CA", "CC")}
    with pytest.raises(ValueError):
        plot_interaction_grid(cells, [], [], tmp_path / "x.png", title="t", ylabel="y")


def test_plot_interaction_grid_single_head(tmp_path):
    cells = {k: _triplet(1) for k in ("AA", "AC", "CA", "CC")}
    path = tmp_path / "single.png"
    plot_interaction_grid(cells, ["L0H0"], ["control"], path, title="t", ylabel="y")
    assert path.exists() and path.stat().st_size > 0


def test_plot_interaction_grid_uneven_grid_row(tmp_path):
    # 7 heads with the default ~6 columns leaves a partially filled last row
    heads = [f"L{i}H{i}" for i in range(7)]
    groups = ["sink rises"] * 3 + ["sink falls"] * 3 + ["control"]
    cells = {k: _triplet(7, seed=i) for i, k in enumerate(("AA", "AC", "CA", "CC"))}
    path = tmp_path / "uneven.png"
    plot_interaction_grid(cells, heads, groups, path, title="t", ylabel="y")
    assert path.exists() and path.stat().st_size > 0


# ---- plot_decomposition ---------------------------------------------------------

def test_plot_decomposition_smoke(tmp_path):
    effects = {
        k: _triplet(N, base=b, spread=0.3, ci=0.05, seed=i)
        for i, (k, b) in enumerate({"total": 0.1, "query": 0.05, "context": 0.02, "interaction": 0.0}.items())
    }
    path = tmp_path / "decomposition.png"
    plot_decomposition(effects, HEADS, GROUPS, path, title="t", ylabel="effect")
    assert path.exists() and path.stat().st_size > 0


def test_plot_decomposition_missing_key_raises(tmp_path):
    effects = {k: _triplet(N) for k in ("total", "query", "context")}  # missing interaction
    with pytest.raises(ValueError):
        plot_decomposition(effects, HEADS, GROUPS, tmp_path / "x.png", title="t", ylabel="y")


def test_plot_decomposition_groups_length_mismatch_raises(tmp_path):
    effects = {k: _triplet(N) for k in ("total", "query", "context", "interaction")}
    with pytest.raises(ValueError):
        plot_decomposition(effects, HEADS, GROUPS + ["control"], tmp_path / "x.png", title="t", ylabel="y")


def test_plot_decomposition_array_length_mismatch_raises(tmp_path):
    effects = {k: _triplet(N) for k in ("total", "query", "context")}
    effects["interaction"] = _triplet(N + 1)
    with pytest.raises(ValueError):
        plot_decomposition(effects, HEADS, GROUPS, tmp_path / "x.png", title="t", ylabel="y")


def test_plot_decomposition_empty_heads_raises(tmp_path):
    effects = {k: (np.array([]), np.array([]), np.array([])) for k in ("total", "query", "context", "interaction")}
    with pytest.raises(ValueError):
        plot_decomposition(effects, [], [], tmp_path / "x.png", title="t", ylabel="y")


# ---- plot_dose_response ---------------------------------------------------------

def test_plot_dose_response_smoke(tmp_path):
    ks = np.array([0, 1, 2, 4, 8, 16, 31])
    curves = {head: _triplet(len(ks), seed=i) for i, head in enumerate(HEADS)}
    path = tmp_path / "dose_response.png"
    plot_dose_response(ks, curves, GROUPS, path, title="t", ylabel="mean sink_mass")
    assert path.exists() and path.stat().st_size > 0


def test_plot_dose_response_empty_ks_raises(tmp_path):
    with pytest.raises(ValueError):
        plot_dose_response(np.array([]), {"L0H0": (np.array([]),) * 3}, ["control"], tmp_path / "x.png", title="t", ylabel="y")


def test_plot_dose_response_empty_curves_raises(tmp_path):
    ks = np.array([0, 1, 2])
    with pytest.raises(ValueError):
        plot_dose_response(ks, {}, [], tmp_path / "x.png", title="t", ylabel="y")


def test_plot_dose_response_groups_length_mismatch_raises(tmp_path):
    ks = np.array([0, 1, 2])
    curves = {head: _triplet(len(ks), seed=i) for i, head in enumerate(HEADS)}
    with pytest.raises(ValueError):
        plot_dose_response(ks, curves, GROUPS[:-1], tmp_path / "x.png", title="t", ylabel="y")


def test_plot_dose_response_array_length_mismatch_raises(tmp_path):
    ks = np.array([0, 1, 2, 4])
    curves = {head: _triplet(len(ks), seed=i) for i, head in enumerate(HEADS)}
    curves[HEADS[0]] = _triplet(len(ks) - 1)  # wrong length
    with pytest.raises(ValueError):
        plot_dose_response(ks, curves, GROUPS, tmp_path / "x.png", title="t", ylabel="y")


def test_plot_dose_response_single_head(tmp_path):
    ks = np.array([0, 1, 2, 4, 8, 16, 31])
    curves = {"L0H0": _triplet(len(ks))}
    path = tmp_path / "single.png"
    plot_dose_response(ks, curves, ["control"], path, title="t", ylabel="y")
    assert path.exists() and path.stat().st_size > 0
