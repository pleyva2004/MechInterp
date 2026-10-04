"""Smoke tests for exp2.plots: each figure renders a non-empty PNG, including degenerate inputs."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from exp2.plots import plot_layer_profile, plot_meaningful_counts, plot_tracked_forest

BASES = ["rare", "common"]
PHRASE = " I like sushi"
PLACEMENTS = ["near", "far"]
CONTRASTS = ["total", "order", "identity", "form", "placebo"]
METRICS = ["sink_mass", "slot_mass", "prev_mass", "entropy"]


def make_effects(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for b, pl, c, m, layer, head in itertools.product(BASES, PLACEMENTS, CONTRASTS, METRICS, range(12), range(12)):
        scale = 0.002 if c == "placebo" else 0.03
        mean = float(rng.normal(0, scale))
        half = abs(rng.normal(0.01, 0.003))
        rows.append(dict(
            base=b, phrase=PHRASE, placement=pl, contrast=c, metric=m, name=f"L{layer}H{head}",
            layer=layer, head=head, mean=mean, lo=mean - half, hi=mean + half, p=0.5, effect_size=0.1,
            q=0.5, sig=abs(mean) > 0.04, meaningful=abs(mean) > 0.04,
        ))
    return pd.DataFrame(rows)


HEADS = pd.DataFrame({
    "name": ["L1H5", "L7H7", "L2H4", "L5H3", "L0H1", "L4H11", "L11H11", "L99H9"],  # last not in effects
    "layer": [1, 7, 2, 5, 0, 4, 11, 99],
    "head": [5, 7, 4, 3, 1, 11, 11, 9],
    "group": ["sink rises", "sink rises", "sink falls", "label mix only", "control", "control", "control", "control"],
})


def _check(path):
    assert path.exists() and path.stat().st_size > 0


def test_forest(tmp_path):
    p = tmp_path / "forest.png"
    plot_tracked_forest(make_effects(), HEADS, "rare", PHRASE, "sink_mass", p)
    _check(p)


def test_forest_missing_placement_and_contrast_arg(tmp_path):
    eff = make_effects()
    eff = eff[eff["placement"] == "near"]
    p = tmp_path / "forest1.png"
    plot_tracked_forest(eff, HEADS, "common", PHRASE, "entropy", p, contrasts=("total", "placebo"))
    _check(p)


def test_counts(tmp_path):
    p = tmp_path / "counts.png"
    plot_meaningful_counts(make_effects(), "slot_mass", p)
    _check(p)


def test_layer_profile(tmp_path):
    p = tmp_path / "layers.png"
    plot_layer_profile(make_effects(), "rare", PHRASE, "prev_mass", p)
    _check(p)


@pytest.mark.parametrize("fn", ["forest", "layers"])
def test_no_matching_rows(tmp_path, fn):
    p = tmp_path / "empty.png"
    if fn == "forest":
        plot_tracked_forest(make_effects(), HEADS, "nope", PHRASE, "entropy", p)
    else:
        plot_layer_profile(make_effects(), "nope", PHRASE, "entropy", p)
    _check(p)
