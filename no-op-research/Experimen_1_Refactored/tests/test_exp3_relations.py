"""Tests for exp3.relations."""
from __future__ import annotations

import numpy as np
import pytest

from exp3.relations import (LABELS, arm_labels, arm_means, build_row, category_masses, check_row, double_centre,
                            grid_cells, head_effects, item_residuals, pairing_effect, permutation_p,
                            random_derangements, role_switches, verdict)

BOOT = {"iters": 500, "seed": 0}
N = 3
CP, PP = 30, 31          # model positions; array index = position - 1
KEY_LEN = 33
Q = KEY_LEN - 1


# ---- grid_cells ----------------------------------------------------------

def test_grid_cells_row_major():
    cells = grid_cells(3)
    assert cells.shape == (9, 2)
    for c in range(3):
        for k in range(3):
            assert tuple(cells[c * 3 + k]) == (c, k)


# ---- build_row / check_row -----------------------------------------------

@pytest.fixture
def stim():
    base = 1000 + np.arange(32)
    return {"base": base, "countries": np.array([1, 2, 3]), "capitals": np.array([10, 11, 12]),
            "cities": np.array([20, 21, 22]), "ids": [1, 2, 3, 10, 11, 12, 20, 21, 22]}


def _build(s):
    return build_row(s["base"], s["countries"], s["capitals"], s["cities"], CP, PP)


def _check(seqs, s, base=None):
    check_row(seqs, s["base"] if base is None else base, s["countries"], s["capitals"], s["cities"], CP, PP,
              s["ids"], bos_id=0)


def test_build_row_shape_and_last_row_is_base(stim):
    seqs = _build(stim)
    assert seqs.shape == (2 * N * N + 1, 32)
    np.testing.assert_array_equal(seqs[-1], stim["base"])


def test_build_row_cells_place_ids_and_keep_rest(stim):
    seqs = _build(stim)
    keep = np.ones(32, dtype=bool)
    keep[[29, 30]] = False
    cells = grid_cells(N)
    for g, partners in enumerate((stim["capitals"], stim["cities"])):
        for i, (c, k) in enumerate(cells):
            row = seqs[g * N * N + i]
            assert row[29] == stim["countries"][c]
            assert row[30] == partners[k]
            np.testing.assert_array_equal(row[keep], stim["base"][keep])


def test_build_row_does_not_mutate_base(stim):
    before = stim["base"].copy()
    _build(stim)
    np.testing.assert_array_equal(stim["base"], before)


def test_check_row_passes_on_valid(stim):
    _check(_build(stim), stim)


def test_check_row_rejects_corrupted_position_30(stim):
    seqs = _build(stim)
    seqs[4, 29] = 999
    with pytest.raises(AssertionError):
        _check(seqs, stim)


def test_check_row_rejects_corrupted_partner(stim):
    seqs = _build(stim)
    seqs[N * N + 2, 30] = 999
    with pytest.raises(AssertionError):
        _check(seqs, stim)


def test_check_row_rejects_repeated_token_elsewhere(stim):
    seqs = _build(stim)
    seqs[0, 5] = seqs[0, 6]
    with pytest.raises(AssertionError):
        _check(seqs, stim)


def test_check_row_rejects_stimulus_id_in_base(stim):
    base = stim["base"].copy()
    base[0] = stim["countries"][0]
    seqs = build_row(base, stim["countries"], stim["capitals"], stim["cities"], CP, PP)
    with pytest.raises(AssertionError):
        _check(seqs, stim, base=base)


# ---- category_masses -----------------------------------------------------

def test_category_masses_sum_to_one_and_batch_shape():
    rng = np.random.default_rng(0)
    attn = rng.dirichlet(np.ones(KEY_LEN), size=(2, 4))
    m = category_masses(attn, CP, PP)
    assert m.shape == (2, 4, 5)
    np.testing.assert_allclose(m.sum(axis=-1), 1.0)


def test_category_masses_assigns_right_categories():
    attn = np.zeros(KEY_LEN)
    attn[0], attn[30], attn[31], attn[32] = 0.1, 0.2, 0.3, 0.15
    attn[5] = 0.25  # filler
    m = category_masses(attn, CP, PP)
    np.testing.assert_allclose(m, [0.1, 0.2, 0.3, 0.15, 0.25])


# ---- arm_labels ----------------------------------------------------------

def _peaked(key, peak=0.8):
    a = np.full(KEY_LEN, (1 - peak) / (KEY_LEN - 1))
    a[key] = peak
    return a


def test_arm_labels_flat_is_diffuse():
    arg, mass = arm_labels(np.full(KEY_LEN, 1 / KEY_LEN), CP, PP, diffuse_threshold=0.2)
    assert LABELS[arg] == "Diffuse" and arg == 0
    assert LABELS[mass] == "Filler"  # mass label is never Diffuse


@pytest.mark.parametrize("key,label", [(0, "Sink"), (31, "Partner"), (30, "Country"), (32, "Self"), (10, "Filler")])
def test_arm_labels_peak_categories(key, label):
    arg, mass = arm_labels(_peaked(key), CP, PP, diffuse_threshold=0.2)
    assert LABELS[arg] == label
    assert LABELS[mass] == label


def test_arm_labels_mass_label_uses_summed_category_mass():
    a = np.zeros(KEY_LEN)
    a[1:30] = 0.9 / 29
    a[0] = 0.1
    arg, mass = arm_labels(a, CP, PP, diffuse_threshold=0.05)
    assert LABELS[arg] == "Sink"      # BOS is the single largest key
    assert LABELS[mass] == "Filler"   # but filler holds most of the mass


def test_arm_labels_batched_dtype_and_shape():
    attn = np.stack([_peaked(0), _peaked(30)])
    arg, mass = arm_labels(attn, CP, PP, diffuse_threshold=0.2)
    assert arg.dtype == np.int8 and mass.dtype == np.int8
    np.testing.assert_array_equal(arg, [1, 2])


# ---- arm_means -----------------------------------------------------------

def test_arm_means_correct_and_wrong():
    rng = np.random.default_rng(0)
    attn = rng.normal(size=(N * N, 4))
    g = attn.reshape(N, N, 4)
    out = arm_means(attn, N, np.array([[1, 2, 0]]))
    diag = np.mean([g[i, i] for i in range(N)], axis=0)
    off = np.mean([g[i, k] for i in range(N) for k in range(N) if i != k], axis=0)
    np.testing.assert_allclose(out["correct"], diag)
    np.testing.assert_allclose(out["wrong"], off)


def test_arm_means_permutation_arms():
    rng = np.random.default_rng(1)
    attn = rng.normal(size=(N * N, 2))
    g = attn.reshape(N, N, 2)
    perms = np.array([[1, 2, 0], [2, 0, 1]])
    out = arm_means(attn, N, perms)
    for i, p in enumerate(perms):
        in_p = np.mean([g[j, p[j]] for j in range(N)], axis=0)
        rest = np.mean([g[j, k] for j in range(N) for k in range(N) if k != p[j]], axis=0)
        np.testing.assert_allclose(out[f"p{i}"], in_p)
        np.testing.assert_allclose(out[f"p{i}_rest"], rest)


def test_arm_means_weights_sum_to_total():
    rng = np.random.default_rng(2)
    attn = rng.normal(size=(N * N, 3))
    out = arm_means(attn, N, np.array([[1, 0, 2]]))
    np.testing.assert_allclose(N * out["correct"] + (N * N - N) * out["wrong"], attn.sum(axis=0))


# ---- pairing_effect ------------------------------------------------------

def _additive(n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 1)) + rng.normal(size=(1, n))


def test_pairing_effect_additive_is_zero():
    n = 5
    grid = _additive(n)
    assert pairing_effect(grid) == pytest.approx(0.0, abs=1e-12)
    for p in ([1, 2, 3, 4, 0], [4, 3, 2, 1, 0], [0, 2, 1, 4, 3]):
        assert pairing_effect(grid, np.array(p)) == pytest.approx(0.0, abs=1e-12)


def test_pairing_effect_diagonal_bump_gives_delta():
    n, delta = 5, 0.7
    grid = _additive(n, seed=3) + delta * np.eye(n)
    assert pairing_effect(grid) == pytest.approx(delta)


def test_pairing_effect_equals_scaled_double_centred_diagonal():
    n = 6
    grid = np.random.default_rng(4).normal(size=(n, n))
    expected = n / (n - 1) * np.diagonal(double_centre(grid)).mean()
    assert pairing_effect(grid) == pytest.approx(expected)


def test_pairing_effect_batch_dims():
    n = 4
    grids = np.random.default_rng(5).normal(size=(7, n, n))
    out = pairing_effect(grids)
    assert out.shape == (7,)
    for r in range(7):
        assert out[r] == pytest.approx(pairing_effect(grids[r]))


# ---- random_derangements -------------------------------------------------

def test_random_derangements_properties():
    n = 5
    d = random_derangements(n, 30, seed=0)
    assert d.shape == (30, n)
    for p in d:
        assert not (p == np.arange(n)).any()
        np.testing.assert_array_equal(np.sort(p), np.arange(n))


def test_random_derangements_deterministic():
    np.testing.assert_array_equal(random_derangements(5, 10, seed=7), random_derangements(5, 10, seed=7))


# ---- permutation_p -------------------------------------------------------

def _perms(n, count, seed):
    from exp3.relations import random_permutations
    return random_permutations(n, count, seed)


def test_permutation_p_strong_bump():
    n = 6
    noise = np.random.default_rng(0).normal(size=(n, n)) * 0.01
    grid = noise + np.eye(n)
    out = permutation_p(grid, float(pairing_effect(grid)), _perms(n, 1000, 1))
    assert out["p_greater"] < 0.01
    assert out["p_less"] > 0.9


def test_permutation_p_no_bump_not_significant():
    n = 6
    grid = _additive(n, seed=2) + np.random.default_rng(0).normal(size=(n, n)) * 0.1
    out = permutation_p(grid, float(pairing_effect(grid)), _perms(n, 1000, 1))
    assert out["p_greater"] > 0.01


def test_permutation_p_ranges_and_two_sided():
    n = 5
    rng = np.random.default_rng(3)
    for _ in range(5):
        grid = rng.normal(size=(n, n))
        out = permutation_p(grid, float(pairing_effect(grid)), _perms(n, 200, 2))
        for key in ("p_greater", "p_less", "p_two_sided"):
            assert 0.0 < out[key] <= 1.0
        assert out["p_two_sided"] == pytest.approx(min(1.0, 2 * min(out["p_greater"], out["p_less"])))


# ---- verdict -------------------------------------------------------------

def _rows(mean, seed=0, n=40, sd=0.01):
    return mean + np.random.default_rng(seed).normal(size=n) * sd


def test_verdict_positive():
    v = verdict(_rows(0.5), {"p_greater": 0.001, "p_less": 1.0, "p_two_sided": 0.002}, 0.05, BOOT)
    assert v["verdict"] == "positive"
    assert v["lo"] > 0


def test_verdict_negative():
    v = verdict(_rows(-0.5), {"p_greater": 1.0, "p_less": 0.001, "p_two_sided": 0.002}, 0.05, BOOT)
    assert v["verdict"] == "negative"
    assert v["hi"] < 0


def test_verdict_equivalent():
    v = verdict(_rows(0.0, sd=0.001), {"p_greater": 0.5, "p_less": 0.5, "p_two_sided": 1.0}, 0.05, BOOT)
    assert v["verdict"] == "equivalent"


def test_verdict_inconclusive_when_perm_disagrees():
    v = verdict(_rows(0.5), {"p_greater": 0.5, "p_less": 0.5, "p_two_sided": 1.0}, 0.05, BOOT)
    assert v["verdict"] == "inconclusive"


# ---- item_residuals ------------------------------------------------------

def test_item_residuals_is_diagonal_of_double_centre():
    grid = np.random.default_rng(0).normal(size=(5, 5))
    out = item_residuals(grid)
    np.testing.assert_allclose(out, np.diagonal(double_centre(grid)))
    assert out.shape == (5,)


# ---- head_effects --------------------------------------------------------

def test_head_effects_shapes_and_meaningful():
    rng = np.random.default_rng(0)
    rows, L, H = 60, 2, 3
    t = rng.normal(size=(rows, L, H)) * 0.01
    t[:, 0, 0] += 0.05
    out = head_effects(t, BOOT, q_max=0.05, min_abs=0.02)
    for key in ("mean", "lo", "hi", "p", "q", "meaningful"):
        assert out[key].shape == (L, H)
    assert out["meaningful"][0, 0]
    assert out["meaningful"].sum() == 1


# ---- role_switches -------------------------------------------------------

def test_role_switches_single_switch():
    rows, L, H = 20, 2, 2
    lab_a = np.full((rows, L, H), 5, dtype=np.int8)
    lab_b = lab_a.copy()
    lab_a[:, 0, 0] = 1
    lab_b[:, 0, 0] = 3
    out = role_switches(lab_a, lab_b, q_max=0.05, min_shift=0.5)
    assert len(out) == 1
    s = out[0]
    assert s["to"] == "Sink" and s["from"] == "Partner"
    assert s["shift"] == pytest.approx(1.0)
    assert s["layer"] == 0 and s["head"] == 0


def test_role_switches_identical_arms_none():
    lab = np.random.default_rng(0).integers(0, len(LABELS), size=(20, 2, 2)).astype(np.int8)
    assert role_switches(lab, lab.copy(), q_max=0.05, min_shift=0.1) == []
