"""Tests for exp3.engine: synthetic inputs only (a tiny fake TL-shaped module stands in for the model)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.stats import binom, binomtest

from exp1.backends import get_final_attention
from exp1.sequences import prepend_bos
from exp3 import engine as eng
from exp3.engine import C, LABELS, M

BOOT = {"iters": 400, "seed": 0}
BOS_ID = 0


# ---- helpers -------------------------------------------------------------------------------------------------

def _base(rows: int, units: int, seq_len: int = 32, start: int = 1000) -> np.ndarray:
    """[rows, units, seq_len]: distinct ids per row (no repeats), identical across units."""
    b = start + np.arange(rows)[:, None] * 100 + np.arange(seq_len)[None, :]
    return np.repeat(b[:, None, :], units, axis=1).astype(np.int64)


def _write(base: np.ndarray, writes: dict[int, np.ndarray | int]) -> np.ndarray:
    s = base.copy()
    for pos, ids in writes.items():
        s[..., pos - 1] = ids
    return s


def _toy_design(rows: int = 4, units: int = 2) -> eng.Design:
    base = _base(rows, units)
    a, b, c = 90001, 90002, 90003
    seqs = {
        "none": base.copy(),
        "rep": _write(base, {10: a, 11: b, 12: c, 31: a, 32: b}),
        "plain": _write(base, {10: a, 11: b, 12: c}),
    }
    meta = {
        "none": eng.VariantMeta(()),
        "rep": eng.VariantMeta((10, 11, 12, 31, 32), {"Copy": (12,), "Dup": (11,)}, ((10, 31), (11, 32))),
        "plain": eng.VariantMeta((10, 11, 12)),
    }
    return eng.Design(seqs, base, meta, tuple(f"u{i}" for i in range(units)))


class FakeTL(torch.nn.Module):
    """TL-shaped stand-in: run_with_cache returns causal softmax patterns whose key scores depend only on each key's
    own token id, layer and head (so every sequence's pattern is independent of its batch)."""

    def __init__(self, n_layers: int = 4, n_heads: int = 3):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        self.cfg = SimpleNamespace(n_layers=n_layers, n_heads=n_heads)

    def run_with_cache(self, batch, names_filter=None, return_type=None):
        b, n = batch.shape
        tok = batch.to(torch.float64)
        causal = torch.tril(torch.ones(n, n, dtype=torch.bool))
        heads = torch.arange(self.cfg.n_heads, dtype=torch.float64)[None, :, None]
        cache = {}
        for layer in range(self.cfg.n_layers):
            score = 3.0 * torch.sin(tok[:, None, :] * 0.37 * (layer + 1) + heads)  # [b, H, n]
            score = score[:, :, None, :].expand(b, self.cfg.n_heads, n, n).masked_fill(~causal, float("-inf"))
            cache[("pattern", layer)] = torch.softmax(score, dim=-1).float()
        return None, cache


# ---- category codes, masses, measures ------------------------------------------------------------------------

def test_category_codes_full_row():
    meta = eng.VariantMeta((10, 11, 12, 31, 32), {"Copy": (12,), "Dup": (11,)})
    codes = eng.category_codes(meta, 33)
    assert codes[0] == C["BOS"] and codes[31] == C["Previous"] and codes[32] == C["Self"]
    assert codes[10] == C["Span"] and codes[11] == C["Dup"] and codes[12] == C["Copy"]
    others = np.setdiff1d(np.arange(33), [0, 10, 11, 12, 31, 32])
    assert (codes[others] == C["Filler"]).all()


def test_category_codes_short_row_padded():
    meta = eng.VariantMeta((21, 22, 23))
    codes = eng.category_codes(meta, 25, width=33)
    assert codes.shape == (33,)
    assert (codes[25:] == -1).all()
    assert codes[24] == C["Self"] and codes[23] == C["Previous"]
    assert codes[21] == C["Span"] and codes[22] == C["Span"]


def test_category_codes_delimiter_on_unwritten_position():
    codes = eng.category_codes(eng.VariantMeta((26, 27, 28)), 33)
    assert codes[29] == C["Filler"]
    codes = eng.category_codes(eng.VariantMeta((26, 27, 28), {"Delimiter": (29,)}), 33)
    assert codes[29] == C["Delimiter"]


def test_category_masses_sum_to_one_and_padding_ignored():
    rng = np.random.default_rng(0)
    att = np.zeros((5, 33))
    att[:, :25] = rng.dirichlet(np.ones(25), size=5)
    codes = eng.category_codes(eng.VariantMeta((21, 22, 23)), 25, width=33)
    m = eng.category_masses(att, codes)
    np.testing.assert_allclose(m.sum(-1), 1.0)
    np.testing.assert_allclose(m[:, C["BOS"]], att[:, 0])
    np.testing.assert_allclose(m[:, C["Span"]], att[:, 21] + att[:, 22])
    assert (m[:, C["Copy"]] == 0).all()


def test_measures_phrase_mass_overlaps_previous():
    meta = eng.VariantMeta((10, 11, 12, 31, 32), {"Copy": (12,), "Dup": (11,)})
    att = np.random.default_rng(1).dirichlet(np.ones(33), size=3)
    out = eng.measures(att, eng.category_codes(meta, 33), eng.phrase_positions(meta, 33))
    assert out.shape == (3, len(eng.MEASURES))
    assert eng.phrase_positions(meta, 33) == (10, 11, 12, 31)
    np.testing.assert_allclose(out[:, M["Phrase"]], att[:, [10, 11, 12, 31]].sum(-1))


# ---- design checks -------------------------------------------------------------------------------------------

def test_check_design_accepts_toy():
    out = eng.check_design(_toy_design(), BOS_ID)
    assert out["rep"]["repeat_groups"] == 2 * 2  # two groups x two units


def test_check_design_rejects_change_outside_written():
    d = _toy_design()
    d.seqs["plain"][1, 0, 19] = 77777
    with pytest.raises(AssertionError, match="outside the written"):
        eng.check_design(d, BOS_ID)


def test_check_design_rejects_filler_holding_written_id():
    d = _toy_design()
    d.base[0, :, 19] = 90001  # filler (position 20) now equals A, which is written at 10
    for v in d.seqs:
        d.seqs[v][0, :, 19] = 90001
    with pytest.raises(AssertionError, match="filler position 20"):
        eng.check_design(d, BOS_ID)


def test_check_design_rejects_undeclared_repeat_among_written():
    d = _toy_design()
    d.seqs["plain"][:, :, 11] = 90001  # C := A at 12, both written, but no repeat declared
    with pytest.raises(AssertionError, match="not a declared repeat"):
        eng.check_design(d, BOS_ID)


def test_check_design_rejects_missing_declared_repeat():
    d = _toy_design()
    d.seqs["rep"][2, 1, 31] = 90009  # query no longer equals position 11 in one row
    with pytest.raises(AssertionError, match="declared repeat"):
        eng.check_design(d, BOS_ID)


def test_check_design_rejects_extra_repeat_in_repeat_variant():
    d = _toy_design()
    d.seqs["rep"][:, :, 11] = 90001  # C := A: 10, 12, 31 now share one id; only (10, 31) is declared
    with pytest.raises(AssertionError, match="not a declared repeat"):
        eng.check_design(d, BOS_ID)


def test_check_design_rejects_bos_and_bad_category():
    d = _toy_design()
    d.seqs["none"][0, 0, 5] = BOS_ID
    d.base[0, 0, 5] = BOS_ID
    with pytest.raises(AssertionError, match="BOS"):
        eng.check_design(d, BOS_ID)
    d = _toy_design()
    d.meta["rep"] = eng.VariantMeta((10, 11, 12, 31, 32), {"Copy": (31,)}, ((10, 31), (11, 32)))
    with pytest.raises(AssertionError, match="named categories"):
        eng.check_design(d, BOS_ID)


def test_check_design_unit_repeats_and_short_variant():
    base = _base(3, 2)
    s = base.copy()
    s[:, 0, [9, 19]] = 90001  # unit 0 repeats at 10 and 20
    s[:, 1, [9, 19]] = [90002, 90003]  # unit 1 does not
    short = base[..., :24].copy()
    short[..., 22] = 90004
    d = eng.Design({"p": s, "short": short}, base,
                   {"p": eng.VariantMeta((10, 20), unit_repeats=(((10, 20),), ())),
                    "short": eng.VariantMeta((23,))}, ("a", "b"))
    out = eng.check_design(d, BOS_ID)
    assert out["short"]["seq_len"] == 24
    d.meta["p"] = eng.VariantMeta((10, 20))  # without the declaration unit 0's repeat is caught
    with pytest.raises(AssertionError, match="unit a"):
        eng.check_design(d, BOS_ID)


def test_check_differ_only_at():
    d = _toy_design()
    eng.check_differ_only_at(d, "rep", "plain", [31, 32])
    with pytest.raises(AssertionError):
        eng.check_differ_only_at(d, "rep", "plain", [31])


def test_draw_per_row_avoids_and_is_row_stable():
    avoid = np.arange(40).reshape(4, 10)
    pools = [np.arange(0, 30)] * 5
    out = eng.draw_per_row(avoid, pools, seed=7)
    assert out.shape == (4, 5)
    for i in range(4):
        assert not np.isin(out[i], avoid[i]).any()
        assert len(set(out[i].tolist())) == 5
    np.testing.assert_array_equal(eng.draw_per_row(avoid[:2], pools, seed=7), out[:2])
    assert not np.array_equal(eng.draw_per_row(avoid, pools, seed=8), out)


# ---- extraction (fake model) ---------------------------------------------------------------------------------

def _short_design() -> eng.Design:
    d = _toy_design(rows=5, units=2)
    short = d.base[..., :24].copy()
    short[..., 23] = 90005
    d.seqs["short"] = short
    d.meta["short"] = eng.VariantMeta((24,))
    return d


def test_extract_matches_direct_computation():
    model = FakeTL()
    d = _short_design()
    ext = eng.extract(model, d, BOS_ID, (2, 3), chunk_rows=2, row_sum_atol=1e-4)
    rows, units = d.rows, len(d.units)
    assert ext.att_row.shape == (rows, 4, 4, 3, 33)
    assert ext.att_late_unit.shape == (rows, units, 4, 33)
    assert ext.att_unit.shape == (units, 4, 4, 3, 33)
    assert list(ext.key_len) == [33, 33, 33, 25]
    assert ext.n_sequences == rows * units * 4
    for vi, v in enumerate(d.variants):
        s = d.seqs[v]
        direct = get_final_attention(model, prepend_bos(s.reshape(-1, s.shape[-1]), BOS_ID), 7, -1)
        direct = direct.reshape(rows, units, 4, 3, -1)
        kl = direct.shape[-1]
        np.testing.assert_allclose(ext.att_row[:, vi, ..., :kl], direct.mean(axis=1), rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(ext.att_late_unit[:, :, vi, :kl], direct[:, :, 2:4].mean(axis=(2, 3)), rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(ext.att_unit[:, vi, ..., :kl], direct.mean(axis=0), rtol=1e-6, atol=1e-7)
        assert (ext.att_row[:, vi, ..., kl:] == 0).all()
    np.testing.assert_allclose(ext.att_row.sum(-1), 1.0, atol=1e-5)


def test_determinism_and_hf_check_plumbing():
    model = FakeTL()
    d = _short_design()
    ext = eng.extract(model, d, BOS_ID, (2, 3), chunk_rows=2, row_sum_atol=1e-4)
    det = eng.determinism_check(model, d, ext, BOS_ID, chunk_rows=2, row_sum_atol=1e-4, k=3)
    assert det["rows"] == 4 and det["exact"]
    hf = eng.hf_check(model, model, d, BOS_ID, n_rows=2, batch_size=3, row_sum_atol=1e-4)
    assert hf["max_abs_delta"] == 0.0 and hf["argmax_agreement"] == 1.0 and hf["sequences"] == 2 * 2 * 4


# ---- contrasts -----------------------------------------------------------------------------------------------

def _fake_extraction(att_row: np.ndarray, variants: tuple[str, ...], units: int = 2) -> eng.Extraction:
    rows, V, L, H, K = att_row.shape
    late = att_row[:, :, 6:12].mean(axis=(2, 3))  # [rows, V, K]
    return eng.Extraction(variants, tuple(f"u{i}" for i in range(units)), np.full(V, K), (6, 11), att_row.astype(np.float32),
                          np.repeat(late[:, None], units, axis=1).astype(np.float32),
                          np.repeat(att_row.mean(axis=0)[None], units, axis=0).astype(np.float32), 0.0, 0, 1.0)


def test_make_contrast_masks_from_treated_or_override():
    d = _toy_design()
    c = eng.make_contrast(d, "x", "rep", "plain")
    assert c.codes[12] == C["Copy"] and c.phrase == (10, 11, 12, 31)
    c2 = eng.make_contrast(d, "plc", "plain", "none", mask_from="rep")
    assert c2.codes[12] == C["Copy"] and c2.mask_from == "rep"
    c3 = eng.make_contrast(d, "y", "plain", "rep")
    assert c3.codes[12] == C["Span"] and c3.phrase == (10, 11, 12)
    d.seqs["short"] = d.base[..., :24].copy()
    d.meta["short"] = eng.VariantMeta(())
    with pytest.raises(ValueError, match="key lengths"):
        eng.make_contrast(d, "bad", "rep", "short")


def test_arm_and_unit_measures_shapes():
    d = _toy_design()
    rng = np.random.default_rng(0)
    att = rng.dirichlet(np.ones(33), size=(4, 3, 12, 12))
    ext = _fake_extraction(att, d.variants)
    c = eng.make_contrast(d, "x", "rep", "none")
    ma, mb = eng.arm_measures(ext, c)
    assert ma.shape == (4, 12, 12, len(eng.MEASURES))
    np.testing.assert_allclose(ma[..., :len(eng.CATEGORIES)].sum(-1), 1.0, atol=1e-5)
    ua, ub = eng.late_unit_measures(ext, c)
    assert ua.shape == (4, 2, len(eng.MEASURES))
    np.testing.assert_allclose(ua[:, 0], ua[:, 1])  # the fake extraction repeats one unit
    np.testing.assert_allclose(ua[:, 0, M["BOS"]], eng.late_mean(ma[..., M["BOS"]], (6, 11)), rtol=1e-5)


def test_late_mean():
    x = np.zeros((2, 12, 12))
    x[:, 6:] = 1.0
    np.testing.assert_allclose(eng.late_mean(x, (6, 11)), 1.0)
    np.testing.assert_allclose(eng.late_mean(x, (0, 5)), 0.0)


# ---- intervals and decisions ---------------------------------------------------------------------------------

def _rows(mean: float, sd: float = 0.01, n: int = 300, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(mean, sd, n)


def test_verdict_positive_negative_equivalent_inconclusive():
    assert eng.verdict(_rows(0.05), BOOT, 0.005)["verdict"] == "positive"
    assert eng.verdict(_rows(-0.05), BOOT, 0.005)["verdict"] == "negative"
    eq = eng.verdict(_rows(0.0, sd=0.001), BOOT, 0.005)
    assert eq["verdict"] in ("equivalent",) and eq["tost"]
    assert eng.verdict(_rows(0.0, sd=0.5, n=20), BOOT, 0.005)["verdict"] == "inconclusive"


def test_interval_ci_nesting_and_noninferior():
    s = eng.interval(_rows(0.01, sd=0.05), BOOT)
    assert s["lo"] <= s["lo90"] <= s["mean"] <= s["hi90"] <= s["hi"]
    assert eng.noninferior({"lo90": -0.04}, 0.05) and not eng.noninferior({"lo90": -0.06}, 0.05)
    assert eng.equivalent({"lo90": -0.004, "hi90": 0.004}, 0.005)
    assert not eng.equivalent({"lo90": -0.004, "hi90": 0.006}, 0.005)


def test_unit_ci_sign_count():
    u = eng.unit_ci(np.array([0.1, 0.2, -0.05, 0.3]), BOOT)
    assert u["units"] == 4 and u["n_pos"] == 3 and u["n_neg"] == 1
    assert u["lo"] <= u["mean"] <= u["hi"]


# ---- labels and switches -------------------------------------------------------------------------------------

def test_labels_argmax_diffuse_and_mass():
    codes = eng.category_codes(eng.VariantMeta((10, 11, 12, 31, 32), {"Copy": (12,), "Dup": (11,)}), 33)
    att = np.zeros((3, 33))
    att[0, 0], att[0, 1:] = 0.68, 0.01   # Sink
    att[1, 12], att[1, 0] = 0.6, 0.4     # Copy
    att[2] = 1 / 33                      # flat: Diffuse; filler has the most summed mass
    am, ms = eng.labels(att, codes, 0.2)
    assert [LABELS[i] for i in am] == ["Sink", "Copy", "Diffuse"]
    assert [LABELS[i] for i in ms] == ["Sink", "Copy", "Filler"]


def test_role_switches_detects_sink_to_copy_and_matches_binomtest():
    rows = 30
    lab_b = np.full((rows, 12, 12), LABELS.index("Sink"), dtype=np.int8)
    lab_a = lab_b.copy()
    lab_a[:, 5, 5] = LABELS.index("Copy")
    lab_a[:4, 7, 1] = LABELS.index("Copy")  # 4 discordant rows: shift 0.13 < 0.15
    out = eng.role_switches(lab_a, lab_b, 0.05, 0.15)
    assert len(out) == 1
    s = out[0]
    assert (s["layer"], s["head"], s["from"], s["to"]) == (5, 5, "Sink", "Copy")
    assert s["shift"] == pytest.approx(1.0)
    # the vectorized McNemar p equals scipy's exact binomial test
    for k, m in ((3, 10), (0, 7), (5, 10), (12, 13)):
        exp = binomtest(k, m, 0.5).pvalue
        got = min(1.0, 2.0 * binom.cdf(min(k, m - k), m, 0.5))
        assert got == pytest.approx(exp)


def test_role_switches_identical_none_and_switching_heads_both_directions():
    lab = np.random.default_rng(0).integers(0, len(LABELS), size=(25, 12, 12)).astype(np.int8)
    assert eng.role_switches(lab, lab.copy(), 0.05, 0.1) == []
    a = np.full((25, 12, 12), 1, dtype=np.int8)
    b = a.copy()
    a[:, 0, 0] = 4
    b[:, 3, 3] = 4
    assert eng.switching_heads(a, b, 0.05, 0.15) == {(0, 0), (3, 3)}


def test_transition_matrix_and_head_roles():
    sw = [{"from": "Sink", "to": "Copy"}, {"from": "Sink", "to": "Copy"}, {"from": "Previous", "to": "Span"}]
    t = eng.transition_matrix(sw)
    assert t[LABELS.index("Sink"), LABELS.index("Copy")] == 2 and t.sum() == 3
    lab = np.array([[[1]], [[1]], [[4]]], dtype=np.int8)
    assert eng.head_roles(lab)[0, 0] == 1


def test_tv_distance():
    a = np.zeros((3, 1, 1, 8))
    b = np.zeros((3, 1, 1, 8))
    a[..., 0], b[..., 1] = 1.0, 1.0
    assert eng.tv_distance(a, b)[0, 0] == pytest.approx(1.0)
    assert eng.tv_distance(a, a)[0, 0] == 0.0


# ---- E3.4-style helpers --------------------------------------------------------------------------------------

def test_proportional_excess_zero_under_pure_renormalization():
    rng = np.random.default_rng(0)
    m_c = rng.dirichlet(np.ones(5), size=50)  # column 0 = BOS, column 1 = copy target
    copy_t = m_c[:, 1] + 0.3 * (1 - m_c[:, 1])
    m_t = m_c * ((1 - copy_t) / (1 - m_c[:, 1]))[:, None]
    m_t[:, 1] = copy_t
    np.testing.assert_allclose(m_t.sum(1), 1.0)
    ex = eng.proportional_excess(m_c[:, 0], m_t[:, 0], m_c[:, 1], m_t[:, 1])
    np.testing.assert_allclose(ex, 0.0, atol=1e-12)
    m_t2 = m_t.copy()
    m_t2[:, 0] -= 0.01  # extra BOS loss, moved to another key
    m_t2[:, 2] += 0.01
    np.testing.assert_allclose(eng.proportional_excess(m_c[:, 0], m_t2[:, 0], m_c[:, 1], m_t2[:, 1]), 0.01, atol=1e-12)


def test_random_set_counts():
    hit = np.zeros((12, 12), dtype=bool)
    pool = [(l_, h_) for l_ in range(6, 12) for h_ in range(12)]
    assert (eng.random_set_counts(hit, pool, 10, 50, 0) == 0).all()
    hit[6:12] = True
    assert (eng.random_set_counts(hit, pool, 10, 50, 0) == 10).all()
    hit[:] = False
    hit[6, :5] = True
    null = eng.random_set_counts(hit, pool, 10, 200, 1)
    assert null.min() >= 0 and null.max() <= 5
    np.testing.assert_array_equal(null, eng.random_set_counts(hit, pool, 10, 200, 1))


def test_parse_heads():
    assert eng.parse_heads(["L7H0", "L10H11"]) == [(7, 0), (10, 11)]
