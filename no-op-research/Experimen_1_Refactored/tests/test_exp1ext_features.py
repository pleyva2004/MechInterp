"""Tests for exp1ext.features."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from transformers import AutoTokenizer

from exp1ext.features import fit_feature_model, token_features


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("gpt2")


# ---- token_features -----------------------------------------------------

def test_token_features_known_ids(tokenizer):
    # 262 " the", 15 "0", 13 ".", 32 "A", 198 "\n", 100 a byte-fallback token
    ids = np.array([262, 15, 13, 32, 198, 100])
    df = token_features(ids, tokenizer)
    assert list(df["id"]) == list(ids)

    the = df.iloc[0]
    assert the["leading_space"] and the["is_alpha"] and not the["is_capitalized"]

    zero = df.iloc[1]
    assert zero["is_digit"]

    dot = df.iloc[2]
    assert dot["is_punct"]

    cap_a = df.iloc[3]
    assert cap_a["is_alpha"] and cap_a["is_capitalized"]

    newline = df.iloc[4]
    assert newline["n_chars"] == 0
    for flag in ("is_alpha", "is_digit", "is_punct", "is_capitalized"):
        assert not newline[flag], flag

    byte_tok = df.iloc[5]
    assert byte_tok["is_byte"]
    # documented behavior: an undecodable byte token's U+FFFD core has no alnum char -> counts as punct
    assert byte_tok["is_punct"]


def test_token_features_dtypes_and_log_id(tokenizer):
    ids = np.array([262, 15, 13, 32, 198, 100, 5000])
    df = token_features(ids, tokenizer)
    for col in ("is_byte", "leading_space", "is_alpha", "is_digit", "is_punct", "is_capitalized"):
        assert df[col].dtype == bool, col
    assert np.issubdtype(df["n_chars"].dtype, np.integer)
    assert np.issubdtype(df["log_id"].dtype, np.floating)
    np.testing.assert_allclose(df["log_id"].to_numpy(), np.log1p(ids.astype(np.float64)))


def test_token_features_is_byte_boundary(tokenizer):
    ids = np.array([0, 255, 256, 1000])
    df = token_features(ids, tokenizer)
    np.testing.assert_array_equal(df["is_byte"], [True, True, False, False])


def test_token_features_row_order_preserved(tokenizer):
    ids = np.array([1000, 32, 262])
    df = token_features(ids, tokenizer)
    assert list(df["id"]) == [1000, 32, 262]


# ---- fit_feature_model -----------------------------------------------------

def test_fit_feature_model_recovers_known_coefficients_and_drops_constant():
    rng = np.random.default_rng(0)
    T = 300
    x1 = rng.normal(size=T)
    x2 = rng.normal(size=T)
    const_col = np.full(T, 7.0)
    noise = rng.normal(scale=0.01, size=T)
    effect = 2.0 + 3.0 * x1 - 1.0 * x2 + noise
    features = pd.DataFrame({"x1": x1, "x2": x2, "const": const_col})

    with pytest.warns(UserWarning, match="const"):
        result = fit_feature_model(effect, features, columns=["x1", "x2", "const"], ci=0.95)

    assert list(result["term"]) == ["intercept", "x1", "x2"]
    assert (result["n"] == T).all()
    assert result["r2"].iloc[0] > 0.99

    truth = {"intercept": 2.0, "x1": 3.0, "x2": -1.0}
    by_term = result.set_index("term")
    for term, val in truth.items():
        assert by_term.loc[term, "coef"] == pytest.approx(val, abs=0.05)
        assert by_term.loc[term, "lo"] <= val <= by_term.loc[term, "hi"]


def test_fit_feature_model_drops_non_finite_effect_rows():
    rng = np.random.default_rng(1)
    T = 50
    x1 = rng.normal(size=T)
    effect = 1.0 + 2.0 * x1
    effect[0] = np.nan
    effect[1] = np.inf
    features = pd.DataFrame({"x1": x1})

    result = fit_feature_model(effect, features, columns=["x1"], ci=0.9)

    assert (result["n"] == T - 2).all()
    assert list(result["term"]) == ["intercept", "x1"]


def test_fit_feature_model_handles_boolean_columns():
    rng = np.random.default_rng(2)
    T = 200
    flag = rng.integers(0, 2, size=T).astype(bool)
    effect = 5.0 + 4.0 * flag.astype(np.float64) + rng.normal(scale=0.01, size=T)
    features = pd.DataFrame({"flag": flag})

    result = fit_feature_model(effect, features, columns=["flag"], ci=0.95)

    by_term = result.set_index("term")
    assert by_term.loc["flag", "coef"] == pytest.approx(4.0, abs=0.05)


def test_fit_feature_model_ci_widens_with_higher_confidence():
    rng = np.random.default_rng(3)
    T = 100
    x1 = rng.normal(size=T)
    effect = 1.0 + x1 + rng.normal(scale=0.5, size=T)
    features = pd.DataFrame({"x1": x1})

    narrow = fit_feature_model(effect, features, columns=["x1"], ci=0.5)
    wide = fit_feature_model(effect, features, columns=["x1"], ci=0.99)

    narrow_width = (narrow.set_index("term").loc["x1", "hi"] - narrow.set_index("term").loc["x1", "lo"])
    wide_width = (wide.set_index("term").loc["x1", "hi"] - wide.set_index("term").loc["x1", "lo"])
    assert wide_width > narrow_width
