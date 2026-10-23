"""Experiment 3, class 2: country / capital (or city) grids written into random-token rows.

Every country is paired with every capital (and, in a second grid, with every city), so each token sits once in a
correct cell and n - 1 times in a wrong one. The fact effect T = mean(correct cells) - mean(wrong cells) therefore
cancels each token's own effect (it equals n / (n - 1) times the mean diagonal residual of the double-centred grid).
Inference is per row (Wilcoxon, bootstrap) and per item (a permutation test over pairings). See
exp3/DESIGN_relations.md.

Positions are model positions (BOS = 0); id arrays here are WITHOUT BOS, so model position p is array index p - 1.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy.stats import binomtest

from exp1.compare import bh_fdr
from exp1ext.effects import bootstrap_mean_per_unit
from exp2.analysis import paired_test

CATEGORIES = ("BOS", "country", "partner", "self", "filler")  # key categories at the final query
# argmax label: Diffuse, else category + 1. "Partner" is the capital / city: the previous token in R1-out, the query
# itself in R1-in (where "Self" is then empty).
LABELS = ("Diffuse", "Sink", "Country", "Partner", "Self", "Filler")
GRIDS = ("capital", "city")


def grid_cells(n: int) -> np.ndarray:
    """[n * n, 2] (country index, partner index) in row-major order: cell c * n + k pairs country c with partner k."""
    c, k = np.divmod(np.arange(n * n), n)
    return np.stack([c, k], axis=1)


def build_row(base: np.ndarray, countries: np.ndarray, capitals: np.ndarray, cities: np.ndarray,
              country_pos: int, partner_pos: int) -> np.ndarray:
    """One base row [seq_len] -> [2 n^2 + 1, seq_len]: the capital grid, the city grid, then the base row ('none')."""
    n = len(countries)
    cells = grid_cells(n)
    out = []
    for partners in (capitals, cities):
        seqs = np.repeat(base[None, :], n * n, axis=0)
        seqs[:, country_pos - 1] = countries[cells[:, 0]]
        seqs[:, partner_pos - 1] = partners[cells[:, 1]]
        out.append(seqs)
    out.append(base[None, :].copy())
    return np.concatenate(out)


def check_row(seqs: np.ndarray, base: np.ndarray, countries: np.ndarray, capitals: np.ndarray, cities: np.ndarray,
              country_pos: int, partner_pos: int, stimulus_ids: Sequence[int], bos_id: int) -> None:
    """Raise unless every cell holds its country / partner at the two positions, equals the base elsewhere, repeats
    no token, contains no BOS, and the base itself contains no stimulus id."""
    n = len(countries)
    cells = grid_cells(n)
    if np.isin(base, np.asarray(stimulus_ids)).any():
        raise AssertionError("a stimulus id occurs in the base row")
    keep = np.ones(seqs.shape[1], dtype=bool)
    keep[[country_pos - 1, partner_pos - 1]] = False
    for g, partners in enumerate((capitals, cities)):
        block = seqs[g * n * n:(g + 1) * n * n]
        if not (block[:, country_pos - 1] == countries[cells[:, 0]]).all():
            raise AssertionError(f"{GRIDS[g]} grid: wrong country at position {country_pos}")
        if not (block[:, partner_pos - 1] == partners[cells[:, 1]]).all():
            raise AssertionError(f"{GRIDS[g]} grid: wrong partner at position {partner_pos}")
        if not (block[:, keep] == base[keep]).all():
            raise AssertionError(f"{GRIDS[g]} grid: differs from the base outside the two positions")
    if not (seqs[-1] == base).all():
        raise AssertionError("the last sequence is not the base row ('none')")
    s = np.sort(seqs, axis=1)
    if (s[:, 1:] == s[:, :-1]).any() or (seqs == bos_id).any():
        raise AssertionError("a sequence repeats a token or contains BOS")


def build_cells(base: np.ndarray, writes: Sequence[Sequence[tuple[int, int]]]) -> np.ndarray:
    """One base row [seq_len] -> [len(writes), seq_len]: cell i is the base with each (model position, id) of writes[i]
    written in (e.g. R1-in's anchor and capital-alone cells)."""
    out = np.repeat(base[None, :], len(writes), axis=0)
    for i, cell in enumerate(writes):
        for pos, tok in cell:
            out[i, pos - 1] = tok
    return out


def check_cells(seqs: np.ndarray, base: np.ndarray, writes: Sequence[Sequence[tuple[int, int]]], bos_id: int) -> None:
    """Raise unless each cell holds its written ids, equals the base elsewhere, and repeats no token / has no BOS."""
    for i, cell in enumerate(writes):
        keep = np.ones(seqs.shape[1], dtype=bool)
        for pos, tok in cell:
            keep[pos - 1] = False
            if seqs[i, pos - 1] != tok:
                raise AssertionError(f"extra cell {i}: id {tok} missing at position {pos}")
        if not (seqs[i, keep] == base[keep]).all():
            raise AssertionError(f"extra cell {i}: differs from the base outside its positions")
    s = np.sort(seqs, axis=1)
    if (s[:, 1:] == s[:, :-1]).any() or (seqs == bos_id).any():
        raise AssertionError("an extra cell repeats a token or contains BOS")


def category_masses(attn: np.ndarray, country_pos: int, partner_pos: int) -> np.ndarray:
    """attn [..., key_len] at the final query -> [..., 5] mass on BOS, the country, the partner, the query itself,
    and everything else (positions are the same in every cell, so the masks are identical across arms). When the
    partner is the query itself (R1-in), its mass counts under 'partner' and 'self' is 0."""
    q = attn.shape[-1] - 1
    named = [0, country_pos, partner_pos, q]
    m = attn[..., named].copy()
    if partner_pos == q:
        m[..., 3] = 0.0
    return np.concatenate([m, 1.0 - m.sum(axis=-1, keepdims=True)], axis=-1)


def arm_labels(attn: np.ndarray, country_pos: int, partner_pos: int, diffuse_threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """attn [..., key_len], averaged over an arm's cells. Returns (argmax label, mass label), int8 codes into LABELS.
    Argmax label: Diffuse if the max key weight < threshold, else the argmax key's category. Mass label: the category
    with the largest summed mass (never Diffuse)."""
    q = attn.shape[-1] - 1
    cat_of_key = np.full(attn.shape[-1], CATEGORIES.index("filler"))
    cat_of_key[q] = 3
    cat_of_key[[0, country_pos, partner_pos]] = [0, 1, 2]  # after q, so a partner at the query labels as Partner
    top = cat_of_key[attn.argmax(axis=-1)] + 1
    argmax_label = np.where(attn.max(axis=-1) < diffuse_threshold, 0, top).astype(np.int8)
    mass_label = (category_masses(attn, country_pos, partner_pos).argmax(axis=-1) + 1).astype(np.int8)
    return argmax_label, mass_label


def arm_means(attn: np.ndarray, n: int, perms: np.ndarray) -> dict[str, np.ndarray]:
    """attn [n * n, ...] for one grid of one row. Mean over the cells of each arm:
    'correct' (diagonal), 'wrong' (off-diagonal), and for each permutation i in perms [P, n] 'p{i}' (its n cells)
    and 'p{i}_rest' (the other n^2 - n cells)."""
    g = attn.reshape((n, n) + attn.shape[1:])
    idx = np.arange(n)
    total = g.sum(axis=(0, 1))
    out = {}
    for name, cols in [("correct", idx)] + [(f"p{i}", p) for i, p in enumerate(perms)]:
        sel = g[idx, cols].sum(axis=0)
        key_rest = "wrong" if name == "correct" else f"{name}_rest"
        out[name] = sel / n
        out[key_rest] = (total - sel) / (n * n - n)
    return out


def pairing_effect(grids: np.ndarray, perm: np.ndarray | None = None) -> np.ndarray:
    """grids [..., n, n] (country x partner). Mean of the cells (j, perm[j]) minus the mean of all other cells;
    perm = identity gives the fact effect T (correct - wrong). Returns grids.shape[:-2]."""
    n = grids.shape[-1]
    perm = np.arange(n) if perm is None else np.asarray(perm)
    sel = grids[..., np.arange(n), perm].sum(axis=-1)
    total = grids.sum(axis=(-2, -1))
    return sel / n - (total - sel) / (n * n - n)


def double_centre(grid: np.ndarray) -> np.ndarray:
    """[..., n, n] -> residuals after removing row (country) and column (partner) means."""
    return (grid - grid.mean(axis=-1, keepdims=True) - grid.mean(axis=-2, keepdims=True)
            + grid.mean(axis=(-2, -1), keepdims=True))


def random_permutations(n: int, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack([rng.permutation(n) for _ in range(count)])


def random_derangements(n: int, count: int, seed: int) -> np.ndarray:
    """count permutations of range(n) with no fixed point (rejection sampling)."""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < count:
        p = rng.permutation(n)
        if not (p == np.arange(n)).any():
            out.append(p)
    return np.stack(out)


def permutation_p(grid_mean: np.ndarray, observed: float, perms: np.ndarray) -> dict[str, float]:
    """grid_mean [n, n] (mean over rows). The pairing effect of every permutation in perms is the null for
    `observed`. One-sided p for each direction, add-one corrected, and the two-sided p (twice the smaller, capped)."""
    null = np.array([pairing_effect(grid_mean, p) for p in perms])
    greater = (1 + (null >= observed - 1e-15).sum()) / (1 + len(null))
    less = (1 + (null <= observed + 1e-15).sum()) / (1 + len(null))
    return {"p_greater": float(greater), "p_less": float(less), "p_two_sided": float(min(1.0, 2 * min(greater, less)))}


def verdict(row_values: np.ndarray, perm: dict[str, float], margin: float, boot: dict[str, Any], alpha: float = 0.05) -> dict[str, Any]:
    """Three-way decision on a per-row score: 'positive' / 'negative' when the 95% row CI and the one-sided
    permutation test agree on the side, 'equivalent' when the 90% row CI lies inside +-margin, else 'inconclusive'."""
    x = np.asarray(row_values, dtype=np.float64)
    mean, lo, hi = (float(v) for v in bootstrap_mean_per_unit(x, boot["iters"], boot["seed"], 0.95))
    _, lo90, hi90 = (float(v) for v in bootstrap_mean_per_unit(x, boot["iters"], boot["seed"], 0.90))
    if lo > 0 and perm["p_greater"] < alpha:
        v = "positive"
    elif hi < 0 and perm["p_less"] < alpha:
        v = "negative"
    elif -margin < lo90 and hi90 < margin:
        v = "equivalent"
    else:
        v = "inconclusive"
    return {"mean": mean, "lo": lo, "hi": hi, "lo90": lo90, "hi90": hi90,
            "wilcoxon_p": float(paired_test(x[:, None])["p"][0]), **perm, "verdict": v}


def item_residuals(grid_mean: np.ndarray) -> np.ndarray:
    """[n, n] -> [n]: each country's diagonal residual after double-centring (its own correct-pairing effect)."""
    return np.diagonal(double_centre(grid_mean)).copy()


def head_effects(t_rows: np.ndarray, boot: dict[str, Any], q_max: float, min_abs: float) -> dict[str, np.ndarray]:
    """t_rows [rows, L, H]: per-row fact effect per head. Mean, 95% CI, Wilcoxon p, BH q over all heads, and
    meaningful = q <= q_max and |mean| >= min_abs."""
    mean, lo, hi = bootstrap_mean_per_unit(t_rows, boot["iters"], boot["seed"], 0.95)
    p = paired_test(t_rows)["p"]
    q = bh_fdr(p)
    return {"mean": mean, "lo": lo, "hi": hi, "p": p, "q": q, "meaningful": (q <= q_max) & (np.abs(mean) >= min_abs)}


def role_switches(lab_a: np.ndarray, lab_b: np.ndarray, q_max: float, min_shift: float) -> list[dict[str, Any]]:
    """lab_a, lab_b: int codes [rows, L, H] for arm a (treated) and arm b, one label per row and arm. For each
    destination label: McNemar (exact binomial on discordant rows), BH over heads; a head switches to that label
    when q <= q_max and its row share rises by >= min_shift. The source is the label whose share falls most."""
    rows, n_layers, n_heads = lab_a.shape
    out = []
    shares = {}
    pvals = np.ones((len(LABELS), n_layers, n_heads))
    for code in range(len(LABELS)):
        in_a, in_b = lab_a == code, lab_b == code
        shares[code] = in_a.mean(axis=0) - in_b.mean(axis=0)
        b = (in_a & ~in_b).sum(axis=0)
        c = (~in_a & in_b).sum(axis=0)
        for layer in range(n_layers):
            for head in range(n_heads):
                k, m = int(b[layer, head]), int(b[layer, head] + c[layer, head])
                if m:
                    pvals[code, layer, head] = binomtest(k, m, 0.5).pvalue
    for code in range(len(LABELS)):
        q = bh_fdr(pvals[code])
        hit = (q <= q_max) & (shares[code] >= min_shift - 1e-9)  # exact min_shift can round to just below it
        for layer, head in zip(*np.nonzero(hit)):
            falls = {c: shares[c][layer, head] for c in range(len(LABELS)) if c != code}
            src = min(falls, key=falls.get)
            out.append({"layer": int(layer), "head": int(head), "to": LABELS[code], "from": LABELS[src],
                        "shift": float(shares[code][layer, head]), "q": float(q[layer, head])})
    return out
