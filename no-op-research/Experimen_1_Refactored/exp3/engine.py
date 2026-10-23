"""Experiment 3, class 1: the shared engine for E3.1-E3.5 (exp3/DESIGN.md section 2).

A sub-experiment's builder returns one Design per base class:

  seqs[variant]  int64 [rows, units, seq_len_v], WITHOUT BOS (array index = model position - 1). Units are phrases
                 or passages. seq_len_v may differ per variant (E3.3's 24-token rows); the query is always the final
                 position, q = seq_len_v.
  base           int64 [rows, units, base_len]: the row each unit's variants were written into. It is per unit so a
                 builder may redraw colliding filler per passage (E3.2). A variant equals base[..., :seq_len_v]
                 outside its written positions.
  meta[variant]  VariantMeta: written positions, named key categories (Copy / Dup / Delimiter), declared repeats.

Category masks are not fixed per variant. A Contrast takes them from its treated arm (rule 1) and applies them to
both arms. extract() runs TransformerLens in batches and keeps three arrays: unit-averaged attention per row, the
late-head mean per row and unit, and the per-unit mean over rows. The analysis helpers work on those arrays.
Positions are model positions (BOS = 0) throughout.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
from scipy.stats import binom

from exp1.backends import check_rows_sum_to_one, get_final_attention, model_shape
from exp1.compare import bh_fdr
from exp1.sequences import prepend_bos
from exp1ext.effects import bootstrap_mean_per_unit
from exp2.analysis import paired_test, parse_head
from exp3.relations import head_effects  # mean, 95% CI, Wilcoxon p, BH q over heads, meaningful

CATEGORIES = ("BOS", "Previous", "Self", "Copy", "Dup", "Delimiter", "Span", "Filler")  # disjoint, sum to 1
NAMED = ("Copy", "Dup", "Delimiter")  # declared by a builder; the other categories follow from q and the writes
MEASURES = CATEGORIES + ("Phrase",)  # Phrase = every written key before the query; overlaps the categories
LABELS = ("Diffuse", "Sink") + CATEGORIES[1:]  # label code = category index + 1; 0 = Diffuse
C = {name: i for i, name in enumerate(CATEGORIES)}
M = {name: i for i, name in enumerate(MEASURES)}

__all__ = ["CATEGORIES", "NAMED", "MEASURES", "LABELS", "VariantMeta", "Design", "check_design", "check_differ_only_at",
           "draw_per_row", "Extraction", "extract", "determinism_check", "hf_check", "category_codes",
           "phrase_positions", "category_masses", "measures", "Contrast", "make_contrast", "arm_measures",
           "late_unit_measures", "late_mean", "head_effects", "interval", "verdict", "equivalent", "noninferior",
           "unit_ci", "labels", "role_switches", "switching_heads", "head_roles", "transition_matrix", "tv_distance",
           "proportional_excess", "random_set_counts", "parse_heads"]


# ---- design --------------------------------------------------------------------------------------------------

@dataclass
class VariantMeta:
    """written: model positions whose ids differ from the unit's base row; may include the query.
    categories: named key categories {Copy / Dup / Delimiter: positions}, each in 1..q-2. A category may sit on an
    unwritten position, e.g. E3.5's Delimiter at 29 in the r arm.
    repeats: groups of written positions that must hold one id in every row. These are the only repeats allowed.
    unit_repeats: one such tuple of groups per unit, used instead of `repeats` when the pattern differs by unit
    (passages)."""
    written: tuple[int, ...]
    categories: dict[str, tuple[int, ...]] = field(default_factory=dict)
    repeats: tuple[tuple[int, ...], ...] = ()
    unit_repeats: tuple[tuple[tuple[int, ...], ...], ...] | None = None

    def repeat_groups(self, unit: int) -> tuple[tuple[int, ...], ...]:
        return self.repeats if self.unit_repeats is None else self.unit_repeats[unit]


@dataclass
class Design:
    seqs: dict[str, np.ndarray]
    base: np.ndarray
    meta: dict[str, VariantMeta]
    units: tuple[str, ...]

    @property
    def variants(self) -> tuple[str, ...]:
        return tuple(self.seqs)

    @property
    def rows(self) -> int:
        return self.base.shape[0]

    def key_len(self, variant: str) -> int:
        """Keys at the final query, BOS included: seq_len_v + 1. The query sits at key_len - 1."""
        return self.seqs[variant].shape[-1] + 1


def check_design(design: Design, bos_id: int) -> dict[str, dict[str, Any]]:
    """Raise unless every variant:
    - is int64 [rows, units, <= base_len] and contains no BOS;
    - equals its base row outside its written positions;
    - has filler that holds no written id;
    - repeats an id only inside a declared group, and every declared group holds a single id in every row;
    - declares named categories that are disjoint, known, and lie in 1..q-2.
    Returns a per-variant summary."""
    rows, units, base_len = design.base.shape
    if len(design.units) != units:
        raise AssertionError(f"{len(design.units)} unit names for {units} units in the base array")
    if set(design.meta) != set(design.seqs):
        raise AssertionError(f"meta and seqs disagree on the variants: {sorted(set(design.meta) ^ set(design.seqs))}")
    out: dict[str, dict[str, Any]] = {}
    for v, s in design.seqs.items():
        meta = design.meta[v]
        if s.dtype != np.int64 or s.ndim != 3 or s.shape[:2] != (rows, units) or s.shape[2] > base_len:
            raise AssertionError(f"{v}: shape {s.shape} / dtype {s.dtype}; expected int64 [{rows}, {units}, <= {base_len}]")
        n = s.shape[2]
        if (s == bos_id).any():
            raise AssertionError(f"{v}: BOS id among the tokens")
        w = np.asarray(meta.written, dtype=np.int64)
        if len(set(w.tolist())) != len(w) or (len(w) and (w.min() < 1 or w.max() > n)):
            raise AssertionError(f"{v}: written positions {meta.written} must be distinct and lie in [1, {n}]")
        keep = np.ones(n, dtype=bool)
        keep[w - 1] = False
        if not (s[..., keep] == design.base[..., :n][..., keep]).all():
            raise AssertionError(f"{v}: differs from its base row outside the written positions")
        named = [p for pos in meta.categories.values() for p in pos]
        if set(meta.categories) - set(NAMED) or len(set(named)) != len(named) or any(not 1 <= p <= n - 2 for p in named):
            raise AssertionError(f"{v}: named categories {meta.categories} must be among {NAMED}, disjoint, in [1, {n - 2}]")
        eq = s[..., :, None] == s[..., None, :]  # [rows, units, n, n]
        cross = eq[..., keep, :][..., w - 1]
        if cross.any():
            r, u, i, j = np.argwhere(cross)[0]
            raise AssertionError(f"{v}, unit {design.units[u]}, row {r}: filler position {np.flatnonzero(keep)[i] + 1} "
                                 f"holds the written id {s[r, u, w[j] - 1]} (position {w[j]})")
        allowed = np.zeros((units, n, n), dtype=bool)
        n_groups = 0
        for u in range(units):
            for g in meta.repeat_groups(u):
                idx = np.asarray(g, dtype=np.int64) - 1
                if len(idx) < 2 or not np.isin(np.asarray(g), w).all():
                    raise AssertionError(f"{v}: repeat group {g} must hold >= 2 written positions")
                if not eq[:, u, idx[0]][:, idx[1:]].all():
                    raise AssertionError(f"{v}, unit {design.units[u]}: declared repeat {g} does not hold one id in every row")
                allowed[u][np.ix_(idx, idx)] = True
                n_groups += 1
        bad = eq & ~allowed[None] & ~np.eye(n, dtype=bool)
        if bad.any():
            r, u, i, j = np.argwhere(bad)[0]
            raise AssertionError(f"{v}, unit {design.units[u]}, row {r}: positions {i + 1} and {j + 1} repeat the id "
                                 f"{s[r, u, i]}, which is not a declared repeat")
        out[v] = {"seq_len": n, "written": [int(p) for p in meta.written],
                  "categories": {k: list(p) for k, p in meta.categories.items()}, "repeat_groups": n_groups}
    return out


def check_differ_only_at(design: Design, a: str, b: str, positions: Sequence[int]) -> None:
    """Raise unless variants a and b have the same shape and hold the same ids everywhere outside `positions`."""
    sa, sb = design.seqs[a], design.seqs[b]
    if sa.shape != sb.shape:
        raise AssertionError(f"{a} {sa.shape} and {b} {sb.shape} differ in shape")
    keep = np.ones(sa.shape[-1], dtype=bool)
    keep[np.asarray(positions, dtype=np.int64) - 1] = False
    if not (sa[..., keep] == sb[..., keep]).all():
        raise AssertionError(f"{a} and {b} differ outside positions {list(positions)}")


def draw_per_row(avoid: np.ndarray, pools: Sequence[np.ndarray], seed: int) -> np.ndarray:
    """avoid: [n, m] ids that row i must not reuse (usually its whole base row). Returns int64 [n, len(pools)]: slot j
    is drawn uniformly from pools[j], never equal to an id in avoid[i] or to an id already drawn for row i. Row i
    uses default_rng([seed, i]), so a row's draw does not depend on how many rows there are."""
    out = np.empty((len(avoid), len(pools)), dtype=np.int64)
    for i, row in enumerate(avoid):
        rng = np.random.default_rng([seed, i])
        taken = set(row.tolist())
        for j, pool in enumerate(pools):
            for _ in range(10_000):
                tok = int(pool[rng.integers(len(pool))])
                if tok not in taken:
                    break
            else:
                raise RuntimeError(f"row {i}: no free id for slot {j} among {len(pool)} candidates")
            out[i, j] = tok
            taken.add(tok)
    return out


# ---- extraction ----------------------------------------------------------------------------------------------

@dataclass
class Extraction:
    variants: tuple[str, ...]
    units: tuple[str, ...]
    key_len: np.ndarray        # int [V]; keys >= key_len[v] are zero-padded
    late_layers: tuple[int, int]
    att_row: np.ndarray        # float32 [rows, V, L, H, K]: per row, the mean over units
    att_late_unit: np.ndarray  # float32 [rows, U, V, K]: the mean over the late layers' heads
    att_unit: np.ndarray       # float32 [U, V, L, H, K]: per unit, the mean over rows
    max_row_sum_dev: float
    n_sequences: int
    seconds: float

    def index(self, variant: str) -> int:
        return self.variants.index(variant)


def extract(model, design: Design, bos_id: int, late_layers: Sequence[int], chunk_rows: int, row_sum_atol: float,
            n_rows: int | None = None, log: Callable[[str], None] | None = None) -> Extraction:
    """Final-query attention for every variant x row x unit, through `model`. One forward pass covers chunk_rows
    rows x all units of one variant. Chunks start at multiples of chunk_rows, so a rerun on the first k * chunk_rows
    rows repeats the same batches exactly. Each batch is checked to sum to 1. n_rows limits the run to the first
    rows."""
    variants = design.variants
    rows = design.rows if n_rows is None else min(n_rows, design.rows)
    units = len(design.units)
    n_layers, n_heads = model_shape(model)
    width = max(design.key_len(v) for v in variants)
    lo, hi = int(late_layers[0]), int(late_layers[1])
    att_row = np.zeros((rows, len(variants), n_layers, n_heads, width), dtype=np.float32)
    att_late_unit = np.zeros((rows, units, len(variants), width), dtype=np.float32)
    unit_sum = np.zeros((units, len(variants), n_layers, n_heads, width), dtype=np.float64)
    dev, n_seq, t0 = 0.0, 0, time.time()
    for vi, v in enumerate(variants):
        kl = design.key_len(v)
        for r0 in range(0, rows, chunk_rows):
            block = design.seqs[v][r0:min(rows, r0 + chunk_rows)]
            flat = block.reshape(-1, block.shape[-1])
            attn = get_final_attention(model, prepend_bos(flat, bos_id), len(flat), -1)
            dev = max(dev, check_rows_sum_to_one(attn, row_sum_atol))
            attn = attn.reshape(block.shape[:2] + attn.shape[1:])  # [n, U, L, H, kl]
            att_row[r0:r0 + len(block), vi, ..., :kl] = attn.mean(axis=1)
            att_late_unit[r0:r0 + len(block), :, vi, :kl] = attn[:, :, lo:hi + 1].mean(axis=(2, 3))
            unit_sum[:, vi, ..., :kl] += attn.sum(axis=0, dtype=np.float64)
            n_seq += len(flat)
        if log is not None:
            log(f"{v}: {rows} rows x {units} units ({n_seq} sequences so far, {time.time() - t0:.0f}s)")
    return Extraction(variants=variants, units=design.units,
                      key_len=np.array([design.key_len(v) for v in variants]), late_layers=(lo, hi),
                      att_row=att_row, att_late_unit=att_late_unit, att_unit=(unit_sum / rows).astype(np.float32),
                      max_row_sum_dev=dev, n_sequences=n_seq, seconds=time.time() - t0)


def determinism_check(model, design: Design, ext: Extraction, bos_id: int, chunk_rows: int, row_sum_atol: float,
                      k: int) -> dict[str, Any]:
    """Re-extract the first chunks covering k rows (same batches as the main pass); the stored arrays must match
    exactly."""
    n = min(ext.att_row.shape[0], -(-k // chunk_rows) * chunk_rows)
    again = extract(model, design, bos_id, ext.late_layers, chunk_rows, row_sum_atol, n_rows=n)
    d = max(float(np.abs(again.att_row - ext.att_row[:n]).max()),
            float(np.abs(again.att_late_unit - ext.att_late_unit[:n]).max()))
    return {"rows": n, "max_abs_delta": d, "exact": d == 0.0}


def hf_check(tl, hf, design: Design, bos_id: int, n_rows: int, batch_size: int, row_sum_atol: float) -> dict[str, Any]:
    """TL vs HF on identical ids: every variant and unit of the first n_rows rows, per sequence. Returns the max
    |delta attention| and the argmax-key agreement."""
    max_delta, agree, n_seq = 0.0, [], 0
    for v, s in design.seqs.items():
        tokens = prepend_bos(s[:n_rows].reshape(-1, s.shape[-1]), bos_id)
        a_tl = get_final_attention(tl, tokens, batch_size, -1)
        a_hf = get_final_attention(hf, tokens, batch_size, -1)
        check_rows_sum_to_one(a_tl, row_sum_atol)
        check_rows_sum_to_one(a_hf, row_sum_atol)
        max_delta = max(max_delta, float(np.abs(a_tl - a_hf).max()))
        agree.append(float((a_tl.argmax(-1) == a_hf.argmax(-1)).mean()))
        n_seq += len(tokens)
    return {"rows": n_rows, "sequences": n_seq, "max_abs_delta": max_delta, "argmax_agreement": float(np.mean(agree))}


# ---- masks and measures --------------------------------------------------------------------------------------

def category_codes(meta: VariantMeta, key_len: int, width: int | None = None) -> np.ndarray:
    """int [width] codes into CATEGORIES for the keys at the final query q = key_len - 1; -1 marks padding
    (keys >= key_len). Precedence: BOS 0, Self q, Previous q - 1, named categories, Span (other written positions
    before q - 1), Filler (the rest)."""
    q = key_len - 1
    codes = np.full(width or key_len, -1, dtype=np.int64)
    codes[:key_len] = C["Filler"]
    codes[[p for p in meta.written if p < q - 1]] = C["Span"]
    for name, pos in meta.categories.items():
        codes[list(pos)] = C[name]
    codes[q - 1], codes[q], codes[0] = C["Previous"], C["Self"], C["BOS"]
    return codes


def phrase_positions(meta: VariantMeta, key_len: int) -> tuple[int, ...]:
    """Every written key before the query (q - 1 included when written): phrase-position mass."""
    return tuple(sorted(p for p in meta.written if p < key_len - 1))


def category_masses(att: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """att [..., K] -> float64 [..., len(CATEGORIES)]: summed weight per category (empty categories are 0)."""
    out = np.zeros(att.shape[:-1] + (len(CATEGORIES),))
    for c in range(len(CATEGORIES)):
        sel = np.flatnonzero(codes == c)
        if len(sel):
            out[..., c] = att[..., sel].sum(axis=-1, dtype=np.float64)
    return out


def measures(att: np.ndarray, codes: np.ndarray, phrase: Sequence[int]) -> np.ndarray:
    """att [..., K] -> float64 [..., len(MEASURES)]: the category masses, then phrase-position mass."""
    ph = att[..., list(phrase)].sum(axis=-1, dtype=np.float64) if len(phrase) else np.zeros(att.shape[:-1])
    return np.concatenate([category_masses(att, codes), ph[..., None]], axis=-1)


@dataclass
class Contrast:
    name: str
    a: str          # treated arm
    b: str          # control arm
    mask_from: str  # the variant whose positions define the masks (rule 1: the treated arm, unless stated)
    codes: np.ndarray
    phrase: tuple[int, ...]


def make_contrast(design: Design, name: str, a: str, b: str, mask_from: str | None = None,
                  width: int | None = None) -> Contrast:
    """Masks come from `mask_from` (default: a) and apply to both arms; the arms' key lengths must agree."""
    src = mask_from or a
    kl = {design.key_len(x) for x in (a, b, src)}
    if len(kl) != 1:
        raise ValueError(f"contrast {name}: key lengths differ across {a}, {b}, {src}; masks would not align")
    k = kl.pop()
    width = width or max(design.key_len(v) for v in design.variants)
    return Contrast(name, a, b, src, category_codes(design.meta[src], k, width), phrase_positions(design.meta[src], k))


def arm_measures(ext: Extraction, c: Contrast) -> tuple[np.ndarray, np.ndarray]:
    """Per-row measures [rows, L, H, len(MEASURES)] of arms a and b on unit-averaged attention, with c's masks."""
    return (measures(ext.att_row[:, ext.index(c.a)], c.codes, c.phrase),
            measures(ext.att_row[:, ext.index(c.b)], c.codes, c.phrase))


def late_unit_measures(ext: Extraction, c: Contrast) -> tuple[np.ndarray, np.ndarray]:
    """Per row and unit, the late-head-mean measures [rows, U, len(MEASURES)] of arms a and b."""
    return (measures(ext.att_late_unit[:, :, ext.index(c.a)], c.codes, c.phrase),
            measures(ext.att_late_unit[:, :, ext.index(c.b)], c.codes, c.phrase))


def late_mean(x: np.ndarray, layers: Sequence[int]) -> np.ndarray:
    """x [..., L, H] -> [...]: the mean over the heads of layers[0]..layers[1] (inclusive)."""
    return x[..., int(layers[0]):int(layers[1]) + 1, :].mean(axis=(-2, -1))


# ---- intervals and decisions ---------------------------------------------------------------------------------

def interval(x: np.ndarray, boot: dict[str, Any]) -> dict[str, float]:
    """x [n] per-row values: the mean, 95% and 90% percentile-bootstrap CIs (resampling rows) and Wilcoxon p."""
    x = np.asarray(x, dtype=np.float64)
    mean, lo, hi = (float(v) for v in bootstrap_mean_per_unit(x, boot["iters"], boot["seed"], 0.95))
    _, lo90, hi90 = (float(v) for v in bootstrap_mean_per_unit(x, boot["iters"], boot["seed"], 0.90))
    return {"n": int(len(x)), "mean": mean, "lo": lo, "hi": hi, "lo90": lo90, "hi90": hi90,
            "wilcoxon_p": float(paired_test(x[:, None])["p"][0])}


def equivalent(stats: dict[str, float], margin: float) -> bool:
    """TOST at alpha 0.05: the 90% CI lies inside (-margin, margin)."""
    return bool(-margin < stats["lo90"] and stats["hi90"] < margin)


def noninferior(stats: dict[str, float], margin: float) -> bool:
    """The one-sided 95% lower bound (= the lower end of the two-sided 90% CI) lies above -margin."""
    return bool(stats["lo90"] > -margin)


def verdict(x: np.ndarray, boot: dict[str, Any], margin: float) -> dict[str, Any]:
    """interval() plus a three-way decision. 'positive' / 'negative' when the 95% CI excludes 0 on that side;
    otherwise 'equivalent' when TOST passes (90% CI inside +-margin); otherwise 'inconclusive'."""
    s = interval(x, boot)
    eq = equivalent(s, margin)
    v = "positive" if s["lo"] > 0 else "negative" if s["hi"] < 0 else "equivalent" if eq else "inconclusive"
    return {**s, "tost": eq, "verdict": v}


def unit_ci(x: np.ndarray, boot: dict[str, Any], ci: float = 0.95) -> dict[str, float]:
    """x [U] per-unit means: the CI pooled over units (bootstrap over units), plus a sign count."""
    x = np.asarray(x, dtype=np.float64)
    mean, lo, hi = (float(v) for v in bootstrap_mean_per_unit(x, boot["iters"], boot["seed"], ci))
    return {"units": int(len(x)), "mean": mean, "lo": lo, "hi": hi, "n_pos": int((x > 0).sum()), "n_neg": int((x < 0).sum())}


# ---- labels and role switches --------------------------------------------------------------------------------

def labels(att: np.ndarray, codes: np.ndarray, diffuse_threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """att [..., K] (unit-averaged per row). Returns int8 codes into LABELS. The argmax label is Diffuse when the max
    key weight < threshold, else the argmax key's category. The mass label is the category with the largest
    summed mass, never Diffuse."""
    top = codes[att.argmax(axis=-1)] + 1
    argmax_label = np.where(att.max(axis=-1) < diffuse_threshold, 0, top).astype(np.int8)
    mass_label = (category_masses(att, codes).argmax(axis=-1) + 1).astype(np.int8)
    return argmax_label, mass_label


def role_switches(lab_a: np.ndarray, lab_b: np.ndarray, q_max: float, min_shift: float) -> list[dict[str, Any]]:
    """lab_a, lab_b: int codes [rows, L, H] for arm a (treated) and arm b, one label per row and arm. For each
    destination label, run McNemar's exact two-sided test (binomial 0.5 on the discordant rows) with BH over heads.
    A head switches to that label when q <= q_max and the label's row share rises by >= min_shift. The source is
    the label whose share falls most."""
    out = []
    shares = np.stack([(lab_a == k).mean(axis=0) - (lab_b == k).mean(axis=0) for k in range(len(LABELS))])
    for code in range(len(LABELS)):
        in_a, in_b = lab_a == code, lab_b == code
        b = (in_a & ~in_b).sum(axis=0)
        m = b + (~in_a & in_b).sum(axis=0)
        p = np.where(m > 0, np.minimum(1.0, 2.0 * binom.cdf(np.minimum(b, m - b), np.maximum(m, 1), 0.5)), 1.0)
        q = bh_fdr(p)
        # tolerance: a share difference of exactly min_shift (e.g. 30/200 rows) can round to just below it
        for layer, head in zip(*np.nonzero((q <= q_max) & (shares[code] >= min_shift - 1e-9))):
            falls = shares[:, layer, head].copy()
            falls[code] = np.inf
            out.append({"layer": int(layer), "head": int(head), "to": LABELS[code], "from": LABELS[int(falls.argmin())],
                        "shift": float(shares[code, layer, head]), "q": float(q[layer, head])})
    return out


def switching_heads(lab_a: np.ndarray, lab_b: np.ndarray, q_max: float, min_shift: float) -> set[tuple[int, int]]:
    """Distinct (layer, head) that switch in either direction (a vs b or b vs a): the placebo count."""
    sw = role_switches(lab_a, lab_b, q_max, min_shift) + role_switches(lab_b, lab_a, q_max, min_shift)
    return {(s["layer"], s["head"]) for s in sw}


def head_roles(lab: np.ndarray) -> np.ndarray:
    """[rows, L, H] codes -> [L, H]: each head's most common label (ties go to the lower code)."""
    return np.stack([(lab == k).sum(axis=0) for k in range(len(LABELS))]).argmax(axis=0)


def transition_matrix(switches: list[dict[str, Any]]) -> np.ndarray:
    """int [len(LABELS), len(LABELS)]: switching heads counted by (source row, destination column)."""
    t = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    for s in switches:
        t[LABELS.index(s["from"]), LABELS.index(s["to"])] += 1
    return t


def tv_distance(mass_a: np.ndarray, mass_b: np.ndarray) -> np.ndarray:
    """[rows, L, H, n_categories] per arm -> [L, H]: total-variation distance between the row-mean category
    profiles."""
    return 0.5 * np.abs(mass_a.mean(axis=0) - mass_b.mean(axis=0)).sum(axis=-1)


# ---- E3.4-style helpers that other sub-experiments can reuse -------------------------------------------------

def proportional_excess(target_c: np.ndarray, target_t: np.ndarray, gain_c: np.ndarray, gain_t: np.ndarray) -> np.ndarray:
    """Loss of the target category beyond its proportional share of another category's gain. A plain
    renormalization would take gain * target_c / (1 - gain_c) from the target. Returns
    (target_c - target_t) - (gain_t - gain_c) * target_c / (1 - gain_c), elementwise; c = control, t = treated."""
    return (target_c - target_t) - (gain_t - gain_c) * target_c / np.maximum(1.0 - gain_c, 1e-12)


def random_set_counts(hit: np.ndarray, pool: Sequence[tuple[int, int]], size: int, n_sets: int, seed: int) -> np.ndarray:
    """hit [L, H] bool. For n_sets random sets of `size` heads drawn without replacement from `pool`, the number of
    hits in each set (the null for "k of these heads hit")."""
    rng = np.random.default_rng(seed)
    pool = np.asarray(pool, dtype=np.int64)
    flags = hit[pool[:, 0], pool[:, 1]]
    return np.array([int(flags[rng.choice(len(pool), size, replace=False)].sum()) for _ in range(n_sets)])


def parse_heads(names: Sequence[str]) -> list[tuple[int, int]]:
    """['L7H0', ...] -> [(7, 0), ...]."""
    return [parse_head(name) for name in names]
