"""Experiment 3, E3.2: span length, from one word to natural text (exp3/DESIGN.md section 3).

Each passage contributes its span: the last K = 31 tokens before its true next token t. For k in the grid, the span's
last k tokens are written at positions 32 - k .. 31 of a base row: random filler before, the row's own random query at
32. A passage token therefore sits at the same position for every k. The units are the passages, and every base row
carries every variant once per passage.

Exclusion is per passage. The rows are shared across passages. Design.base[:, u] is the shared row with every id of
passage u (and its t) redrawn from the class pool minus the passage and the row.

Variants (36 on the default grid):
- none;
- nl_k;
- shuffled_k (k >= 3): one fixed order per passage and k, with the token at 31 kept;
- matched_k: same-form words, punctuation in place, one draw per distinct word id; drawn once for the 31-token span and
  nested over k;
- passage-rr: nl_31 with 30-31 back to the row's tokens;
- the 2 x 2: passage+t, passage+w, none+t, none+w;
- matched_b_k: the placebo.
Positions are model positions (BOS = 0); id arrays are WITHOUT BOS (array index = position - 1).
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

import numpy as np
from scipy.optimize import curve_fit

from exp2.analysis import head_name
from exp2.design import derive_seed
from exp3 import engine as eng

KINDS = ("argmax", "mass")
FIXED = ("passage-rr", "passage+t", "passage+w", "none+t", "none+w")
MEASURES = tuple(m for m in eng.MEASURES if m not in eng.NAMED)  # E3.2 declares no Copy / Dup / Delimiter: always 0
TESTS = ("positive", "negative", "equivalent", "switch_count")


# ---- layout --------------------------------------------------------------------------------------------------

def check_layout(layout: dict[str, Any]) -> tuple[list[int], list[int], list[int], int]:
    """(ks, shuffled_ks, placebo_ks, q). The largest k must fill 1..q-1; shuffled needs k >= 3 (k <= 2 would equal nl);
    passage-rr resets q-2 and q-1."""
    ks = [int(k) for k in layout["ks"]]
    sks = [int(k) for k in layout["shuffled_ks"]]
    pks = [int(k) for k in layout["placebo_ks"]]
    q = int(layout["query"])
    if ks != sorted(set(ks)) or ks[0] < 1 or ks[-1] != q - 1:
        raise ValueError(f"ks {ks} must be ascending, distinct, >= 1 and end at q - 1 = {q - 1}")
    if sks != sorted(set(sks)) or not set(sks) <= set(ks) or min(sks) < 3:
        raise ValueError(f"shuffled_ks {sks} must be a subset of ks with every k >= 3")
    if pks != sorted(set(pks)) or not set(pks) <= set(ks):
        raise ValueError(f"placebo_ks {pks} must be a subset of ks")
    if [int(p) for p in layout["rr"]] != [q - 2, q - 1]:
        raise ValueError(f"passage-rr must reset {[q - 2, q - 1]}, got {layout['rr']}")
    return ks, sks, pks, q


def variant_names(layout: dict[str, Any]) -> tuple[str, ...]:
    ks, sks, pks, _ = check_layout(layout)
    return (("none",) + tuple(f"nl_{k}" for k in ks) + tuple(f"shuffled_{k}" for k in sks)
            + tuple(f"matched_{k}" for k in ks) + FIXED + tuple(f"matched_b_{k}" for k in pks))


def span_positions(k: int, q: int) -> np.ndarray:
    """Model positions q - k .. q - 1 of the span's last k tokens."""
    return np.arange(q - k, q, dtype=np.int64)


def contrast_specs(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The sweep contrasts generated from the layout, then the fixed contrasts from the config:
    E_k = nl_k - none, order_k = nl_k - shuffled_k, content_k = nl_k - matched_k, placebo_k = matched_b_k - matched_k
    (masks from matched_k)."""
    ks, sks, pks, _ = check_layout(cfg["layout"])
    out: dict[str, dict[str, Any]] = {f"E_{k}": {"a": f"nl_{k}", "b": "none"} for k in ks}
    out.update({f"order_{k}": {"a": f"nl_{k}", "b": f"shuffled_{k}"} for k in sks})
    out.update({f"content_{k}": {"a": f"nl_{k}", "b": f"matched_{k}"} for k in ks})
    out.update({f"placebo_{k}": {"a": f"matched_b_{k}", "b": f"matched_{k}", "mask_from": f"matched_{k}"} for k in pks})
    clash = set(out) & set(cfg["contrasts"])
    if clash:
        raise ValueError(f"config contrasts {sorted(clash)} clash with the generated sweep contrasts")
    out.update(cfg["contrasts"])
    return out


def validate_specs(specs: list[dict], variants: Sequence[str], contrasts: dict[str, dict]) -> None:
    """Raise before any extraction if a prediction names an unknown variant, contrast, test or label."""
    for s in specs:
        for p in s["parts"]:
            if p["test"] not in TESTS:
                raise ValueError(f"{s['id']}/{p['name']}: unknown test {p['test']!r}")
            if p["test"] == "switch_count":
                missing = [c for c in [p["contrast"], *p["placebo"]] if c not in contrasts]
                bad = [x for x in (p["from"], p["to"]) if x not in eng.LABELS]
                if missing or bad or p["labels"] not in KINDS:
                    raise ValueError(f"{s['id']}/{p['name']}: unknown contrasts {missing} / labels {bad} / {p['labels']}")
            else:
                missing = [v for v in p["combo"] if v not in variants]
                if missing:
                    raise ValueError(f"{s['id']}/{p['name']}: unknown variants {missing}")
    for name, c in contrasts.items():
        missing = [v for v in (c["a"], c["b"], c.get("mask_from", c["a"])) if v not in variants]
        if missing:
            raise ValueError(f"contrast {name}: unknown variants {missing}")


# ---- spans ---------------------------------------------------------------------------------------------------

def shuffle_order(ids: Sequence[int], seed: int) -> np.ndarray:
    """A permutation of range(len(ids)) that keeps the last index and changes the id sequence."""
    ids = np.asarray(ids, dtype=np.int64)
    k = len(ids)
    rng = np.random.default_rng(seed)
    for _ in range(1000):
        perm = np.append(rng.permutation(k - 1), k - 1)
        if (ids[perm] != ids).any():
            return perm
    raise ValueError(f"no order of {ids.tolist()} keeps the last token and changes the sequence")


def shuffle_orders(spans: dict[str, Sequence[int]], ks: Sequence[int], seed: int) -> dict[str, dict[int, list[int]]]:
    """{passage: {k: permutation of the span's last k tokens}}: one fixed order per passage and k, from `seed`."""
    return {name: {int(k): shuffle_order(list(span)[len(span) - k:], derive_seed(seed, f"{name}/{k}")).tolist()
                   for k in ks}
            for name, span in spans.items()}


def repeat_groups(positions: Sequence[int], ids: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    """Groups of positions that hold the same id (only ids that occur more than once)."""
    by: dict[int, list[int]] = {}
    for p, x in zip(positions, ids):
        by.setdefault(int(x), []).append(int(p))
    return tuple(tuple(g) for g in by.values() if len(g) > 1)


def span_repeats(ids: Sequence[int]) -> dict[str, Any]:
    """Repeats inside a written span: distinct repeated ids, extra occurrences (positions whose id occurred earlier),
    and the shortest distance between two occurrences of one id (None without repeats)."""
    last: dict[int, int] = {}
    extra, min_d = 0, None
    for j, x in enumerate(int(v) for v in ids):
        if x in last:
            extra += 1
            min_d = j - last[x] if min_d is None else min(min_d, j - last[x])
        last[x] = j
    repeated = sorted(x for x, c in Counter(int(v) for v in ids).items() if c > 1)
    return {"repeated_ids": len(repeated), "extra_occurrences": extra, "min_distance": min_d, "ids": repeated}


def redraw_collisions(rows: np.ndarray, exclude: Sequence[int], pool: np.ndarray, seed: int) -> np.ndarray:
    """rows [n, L]. Every id in `exclude` is replaced by a uniform draw from `pool` minus `exclude` and minus the row's
    ids (including earlier replacements). Row i uses default_rng([seed, i]), so a row's draw does not depend on how
    many rows there are; rows without a collision are returned unchanged."""
    out = rows.copy()
    excl = {int(x) for x in exclude}
    for i, row in enumerate(out):
        bad = [j for j, x in enumerate(row.tolist()) if x in excl]
        if not bad:
            continue
        rng = np.random.default_rng([seed, i])
        taken = set(row.tolist()) | excl
        for j in bad:
            for _ in range(10_000):
                tok = int(pool[rng.integers(len(pool))])
                if tok not in taken:
                    break
            else:
                raise RuntimeError(f"row {i}: no free id to replace position {j + 1} among {len(pool)} candidates")
            out[i, j] = tok
            taken.add(tok)
    return out


def matched_span(span: Sequence[int], word_pools: dict[int, np.ndarray], avoid: np.ndarray, seed: int) -> np.ndarray:
    """[n, len(span)]: per row, each distinct word id (a key of word_pools) replaced by one draw from its pool; every
    other id (punctuation) stays. Draws avoid avoid[i] and each other, so distinct words get distinct ids and the
    span's repeat pattern is kept."""
    span = np.asarray(span, dtype=np.int64)
    out = np.tile(span, (len(avoid), 1))
    words = list(dict.fromkeys(int(x) for x in span if int(x) in word_pools))
    if words:
        draws = eng.draw_per_row(avoid, [word_pools[w] for w in words], seed)
        for j, w in enumerate(words):
            out[:, span == w] = draws[:, [j]]
    return out


# ---- design --------------------------------------------------------------------------------------------------

def build_e32(rows: np.ndarray, spans: dict[str, Sequence[int]], t_ids: dict[str, int],
              exclude: dict[str, Sequence[int]], word_pools: dict[str, dict[int, np.ndarray]],
              t_pools: dict[str, np.ndarray], orders: dict[str, dict[int, Sequence[int]]], base_pool: np.ndarray,
              seed: int, layout: dict[str, Any]) -> eng.Design:
    """Build the E3.2 Design; the units are the passages.
    rows: [n, seq_len] shared base rows, without BOS. spans: {passage: its last K ids before t}. t_ids: {passage: t}.
    exclude: {passage: every id of its text} (t is added). word_pools: {passage: {word id: matched pool}}; span ids
    without a pool are punctuation and stay in place. t_pools: {passage: t's matched pool} for w. orders: from
    shuffle_orders. base_pool: the ids the rows were drawn from."""
    ks, sks, pks, q = check_layout(layout)
    K = ks[-1]
    n, seq_len = rows.shape
    if q != seq_len:
        raise ValueError(f"the query ({q}) must be the final position ({seq_len})")
    units = tuple(spans)
    names = variant_names(layout)
    base = np.repeat(rows[:, None, :], len(units), axis=1)
    seqs = {v: np.empty_like(base) for v in names}
    groups: dict[str, list] = {v: [] for v in names}
    written: dict[str, tuple[int, ...]] = {}
    full = span_positions(K, q)
    for u, name in enumerate(units):
        span = np.asarray(spans[name], dtype=np.int64)
        if len(span) != K:
            raise ValueError(f"{name}: span of {len(span)} ids, expected {K}")
        t = int(t_ids[name])
        bu = redraw_collisions(rows, sorted({int(x) for x in exclude[name]} | {t}), base_pool,
                               derive_seed(seed, f"redraw/{name}"))
        base[:, u] = bu
        avoid = np.concatenate([bu, np.tile(span, (n, 1))], axis=1)
        mat = matched_span(span, word_pools[name], avoid, derive_seed(seed, f"matched/{name}"))
        mat_b = matched_span(span, word_pools[name], avoid, derive_seed(seed, f"matched_b/{name}"))
        w = eng.draw_per_row(avoid, [t_pools[name]], derive_seed(seed, f"wrong/{name}"))[:, 0]
        # variant: (positions, ids [len] or [n, len], the id pattern that fixes the repeat groups)
        cells: dict[str, tuple] = {"none": (np.zeros(0, dtype=np.int64), None, ())}
        for k in ks:
            pos, ids = span_positions(k, q), span[K - k:]
            cells[f"nl_{k}"] = (pos, ids, ids)
            cells[f"matched_{k}"] = (pos, mat[:, K - k:], ids)
            if k in sks:
                perm = np.asarray(orders[name][k], dtype=np.int64)
                cells[f"shuffled_{k}"] = (pos, ids[perm], ids[perm])
            if k in pks:
                cells[f"matched_b_{k}"] = (pos, mat_b[:, K - k:], ids)
        with_q = np.append(full, q)
        cells["passage-rr"] = (full[:-2], span[:-2], span[:-2])
        cells["passage+t"] = (with_q, np.append(span, t), np.append(span, t))
        cells["passage+w"] = (with_q, np.concatenate([np.tile(span, (n, 1)), w[:, None]], axis=1), np.append(span, -1))
        cells["none+t"] = (np.array([q]), np.array([t]), (t,))
        cells["none+w"] = (np.array([q]), w[:, None], (-1,))
        for v in names:
            pos, ids, pattern = cells[v]
            s = bu.copy()
            if len(pos):
                s[:, pos - 1] = ids
            seqs[v][:, u] = s
            groups[v].append(repeat_groups(pos, pattern))
            written[v] = tuple(int(p) for p in pos)
    meta = {v: eng.VariantMeta(written[v], {}, (), tuple(groups[v])) for v in names}
    return eng.Design(seqs, base, meta, units)


def check_e32(design: eng.Design, rows: np.ndarray, spans: dict[str, Sequence[int]], t_ids: dict[str, int],
              exclude: dict[str, Sequence[int]], word_pools: dict[str, dict[int, np.ndarray]],
              t_pools: dict[str, np.ndarray], orders: dict[str, dict[int, Sequence[int]]], base_pool: np.ndarray,
              layout: dict[str, Any]) -> dict[str, Any]:
    """Raise unless the E3.2 layout holds (run after eng.check_design):
    - each passage's base row holds no id of the passage or t, and differs from the shared row only where that row
      held one, with replacements from the base pool;
    - nl, shuffled, matched and matched_b differ only on the span; nl holds the span; shuffled holds the fixed order,
      keeps the token at 31 and changes the sequence;
    - matched / matched_b keep punctuation in place, draw each word from its pool (never a passage id), map equal ids
      to equal draws and distinct ids to distinct draws, and are nested over k;
    - passage-rr differs from nl_31 only at 30-31 (the row's tokens there);
    - the 2 x 2 cells differ only at 1..31 (context) or 32 (query); t and w sit at 32, w from t's pool and the same
      in passage+w and none+w.
    Returns the number of redrawn filler positions per passage."""
    ks, sks, pks, q = check_layout(layout)
    K = ks[-1]
    s = design.seqs
    full = span_positions(K, q)
    for k in ks:
        pos = span_positions(k, q)
        eng.check_differ_only_at(design, f"nl_{k}", "none", pos)
        eng.check_differ_only_at(design, f"nl_{k}", f"matched_{k}", pos)
        if k in sks:
            eng.check_differ_only_at(design, f"nl_{k}", f"shuffled_{k}", pos)
        if k in pks:
            eng.check_differ_only_at(design, f"matched_{k}", f"matched_b_{k}", pos)
    eng.check_differ_only_at(design, f"nl_{K}", "passage-rr", layout["rr"])
    eng.check_differ_only_at(design, f"nl_{K}", "passage+t", [q])
    eng.check_differ_only_at(design, "passage+t", "passage+w", [q])
    eng.check_differ_only_at(design, "none+t", "none+w", [q])
    eng.check_differ_only_at(design, "passage+t", "none+t", full)
    eng.check_differ_only_at(design, "passage+w", "none+w", full)
    eng.check_differ_only_at(design, "none", "none+t", [q])
    redrawn = {}
    rr = np.asarray(layout["rr"], dtype=np.int64) - 1
    for u, name in enumerate(design.units):
        span, t = np.asarray(spans[name], dtype=np.int64), int(t_ids[name])
        excl = np.asarray(sorted({int(x) for x in exclude[name]} | {t}), dtype=np.int64)
        if t in span or not np.isin(span, excl).all():
            raise AssertionError(f"{name}: t occurs in the span, or the exclusion list misses a span id")
        bu = design.base[:, u]
        hit = np.isin(rows, excl)
        if np.isin(bu, excl).any():
            raise AssertionError(f"{name}: the base row holds an id of the passage or t")
        if not (bu[~hit] == rows[~hit]).all() or not np.isin(bu[hit], base_pool).all():
            raise AssertionError(f"{name}: the base row differs from the shared row off its collisions, or a "
                                 "replacement is outside the base pool")
        redrawn[name] = int(hit.sum())
        for k in ks:
            ids, idx = span[K - k:], span_positions(k, q) - 1
            if not (s[f"nl_{k}"][:, u][:, idx] == ids).all():
                raise AssertionError(f"nl_{k}, {name}: the span does not hold the passage's last {k} ids")
            if k in sks:
                perm = np.asarray(orders[name][k])
                if sorted(perm.tolist()) != list(range(k)) or perm[-1] != k - 1 or (ids[perm] == ids).all():
                    raise AssertionError(f"shuffled_{k}, {name}: order {perm.tolist()} must keep 31 and change the span")
                if not (s[f"shuffled_{k}"][:, u][:, idx] == ids[perm]).all():
                    raise AssertionError(f"shuffled_{k}, {name}: the span does not hold the fixed order")
        for fam, fam_ks in (("matched", ks), ("matched_b", pks)):
            kmax = max(fam_ks)
            ref_ids = span[K - kmax:]
            ref = s[f"{fam}_{kmax}"][:, u][:, span_positions(kmax, q) - 1]
            for k in fam_ks:
                if not (s[f"{fam}_{k}"][:, u][:, span_positions(k, q) - 1] == ref[:, kmax - k:]).all():
                    raise AssertionError(f"{fam}_{k}, {name}: not the last {k} tokens of {fam}_{kmax}")
            word = np.array([int(x) in word_pools[name] for x in ref_ids])
            if not (ref[:, ~word] == ref_ids[~word]).all():
                raise AssertionError(f"{fam}, {name}: punctuation moved or was redrawn")
            for j in np.flatnonzero(word):
                if not np.isin(ref[:, j], word_pools[name][int(ref_ids[j])]).all():
                    raise AssertionError(f"{fam}, {name}: position {j + q - kmax} holds an id outside its matched pool")
            if np.isin(ref[:, word], excl).any():
                raise AssertionError(f"{fam}, {name}: a drawn word is an id of the passage or t")
            same = ref_ids[:, None] == ref_ids[None, :]
            if not ((ref[:, :, None] == ref[:, None, :]) == same).all():
                raise AssertionError(f"{fam}, {name}: the draws do not keep the span's repeat pattern")
        if not (s["passage-rr"][:, u][:, rr] == bu[:, rr]).all():
            raise AssertionError(f"passage-rr, {name}: positions {layout['rr']} are not the row's own tokens")
        if not (s["passage+t"][:, u, q - 1] == t).all() or not (s["none+t"][:, u, q - 1] == t).all():
            raise AssertionError(f"{name}: t is not at the query in passage+t / none+t")
        w = s["passage+w"][:, u, q - 1]
        if not np.isin(w, t_pools[name]).all() or np.isin(w, excl).any() or not (s["none+w"][:, u, q - 1] == w).all():
            raise AssertionError(f"{name}: w is outside t's pool, is a passage id, or differs between passage+w and none+w")
    return {"redrawn": redrawn}


# ---- analysis ------------------------------------------------------------------------------------------------

def late_bos_rows(ext: eng.Extraction, variant: str, layers: Sequence[int]) -> np.ndarray:
    """[rows]: late pooled BOS mass on passage-averaged attention (BOS is key 0 under every mask)."""
    lo, hi = int(layers[0]), int(layers[1])
    return ext.att_row[:, ext.index(variant), lo:hi + 1, :, 0].mean(axis=(1, 2), dtype=np.float64)


def combo_rows(ext: eng.Extraction, combo: dict[str, float], layers: Sequence[int]) -> np.ndarray:
    """[rows]: sum of coefficient x late pooled BOS mass over the combo's variants."""
    return sum(float(c) * late_bos_rows(ext, v, layers) for v, c in combo.items())


def combo_units(ext: eng.Extraction, combo: dict[str, float]) -> np.ndarray:
    """[U]: per passage, the mean over rows of the combo on late pooled BOS mass."""
    return sum(float(c) * ext.att_late_unit[:, :, ext.index(v), 0].astype(np.float64) for v, c in combo.items()).mean(axis=0)


def saturation(k: np.ndarray, e_inf: float, kappa: float) -> np.ndarray:
    return e_inf * (1.0 - np.exp(-np.asarray(k, dtype=np.float64) / kappa))


def _fit(ks: np.ndarray, y: np.ndarray, sat: dict[str, Any]) -> np.ndarray:
    lo = (float(sat["e_inf_bounds"][0]), float(sat["kappa_bounds"][0]))
    hi = (float(sat["e_inf_bounds"][1]), float(sat["kappa_bounds"][1]))
    eps = 1e-9
    p0 = (float(np.clip(y[-1], lo[0] + eps, hi[0] - eps)), float(np.clip(sat["kappa_start"], lo[1] + eps, hi[1] - eps)))
    try:
        p, _ = curve_fit(saturation, ks, y, p0=p0, bounds=(lo, hi), maxfev=20_000)
    except (RuntimeError, ValueError):
        return np.array([np.nan, np.nan])
    return np.asarray(p, dtype=np.float64)


def fit_saturation(ks: Sequence[int], e_rows: np.ndarray, boot: dict[str, Any], sat: dict[str, Any]) -> dict[str, Any]:
    """Least-squares E(k) = E_inf * (1 - exp(-k / kappa)) on the row means of e_rows [rows, len(ks)], with 95%
    percentile CIs from refitting row resamples. Reports failed fits and the share of resamples whose kappa ends on
    a bound."""
    x = np.asarray(ks, dtype=np.float64)
    y = e_rows.mean(axis=0)
    point = _fit(x, y, sat)
    rng = np.random.default_rng(boot["seed"])
    n = len(e_rows)
    bs = np.array([_fit(x, e_rows[rng.integers(0, n, n)].mean(axis=0), sat) for _ in range(boot["iters"])])
    ok = np.isfinite(bs).all(axis=1)
    kb = [float(b) for b in sat["kappa_bounds"]]
    at_bound = ok & (np.isclose(bs[:, 1], kb[0], rtol=1e-3) | np.isclose(bs[:, 1], kb[1], rtol=1e-3))
    lo, hi = (np.percentile(bs[ok], [2.5, 97.5], axis=0) if ok.any() else np.full((2, 2), np.nan))
    fitted = saturation(x, *point) if np.isfinite(point).all() else np.full_like(y, np.nan)
    return {"ks": [int(k) for k in ks], "mean_E": y.tolist(), "fitted": fitted.tolist(),
            "e_inf": float(point[0]), "kappa": float(point[1]),
            "e_inf_ci": [float(lo[0]), float(hi[0])], "kappa_ci": [float(lo[1]), float(hi[1])],
            "rmse": float(np.sqrt(np.mean((fitted - y) ** 2))), "n_boot": int(len(bs)), "n_failed": int((~ok).sum()),
            "share_kappa_at_bound": float(at_bound.sum() / max(int(ok.sum()), 1)),
            "point_kappa_at_bound": bool(np.isclose(point[1], kb[0], rtol=1e-3) or np.isclose(point[1], kb[1], rtol=1e-3))}


def type_sweep(unit_e: np.ndarray, units: Sequence[str], types: dict[str, str]) -> dict[str, dict[str, Any]]:
    """unit_e [U, len(ks)] per-passage E(k) -> {final-pair type: {passages, E: mean over its passages}}."""
    out = {}
    for ty in dict.fromkeys(types[u] for u in units):
        idx = [i for i, u in enumerate(units) if types[u] == ty]
        out[ty] = {"passages": [units[i] for i in idx], "E": unit_e[idx].mean(axis=0).tolist()}
    return out


def switched_mask(switches: list[dict[str, Any]], shape: tuple[int, int]) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    for s in switches:
        m[s["layer"], s["head"]] = True
    return m


def switch_lengths(ks: Sequence[int], switched: dict[int, np.ndarray]) -> np.ndarray:
    """switched {k: bool [L, H]} -> float [L, H]: the smallest k such that the head is switched at k and at every
    larger k in the grid; NaN when it is not switched at the largest k."""
    out = np.full(next(iter(switched.values())).shape, np.nan)
    run = np.ones(out.shape, dtype=bool)
    for k in sorted(ks, reverse=True):
        run &= switched[k]
        out[run] = k
    return out


def analyse(ext: eng.Extraction, design: eng.Design, cfg: dict[str, Any], contrasts: dict[str, dict],
            specs: list[dict], types: dict[str, str]) -> dict[str, Any]:
    """Every contrast: per-head tests of each measure in MEASURES, the late pooled per-row verdict and the CI pooled over
    passages, role switches (argmax and mass labels), transition matrices and TV distance. Then the sweep outputs
    (E(k), order and content series, saturation fit, per-passage and per-type E(k), per-head ΔBOS curves, switch
    lengths), the placebo switch counts, and each prediction's result in this base class."""
    a = cfg["analysis"]
    boot, layers, margin = a["bootstrap"], a["late_layers"], a["margin"]
    mq, mabs = a["meaningful"]["q"], a["meaningful"]["min_abs"]
    sq, shift = a["switch"]["q"], a["switch"]["min_shift"]
    ks, sks, _, _ = check_layout(cfg["layout"])
    cons = {n: eng.make_contrast(design, n, s["a"], s["b"], s.get("mask_from")) for n, s in contrasts.items()}
    placebos = [n for n in cons if n.startswith("placebo")]
    if not placebos:
        raise ValueError("the contrasts must include at least one named 'placebo...'")

    pooled, unit_rows, heads, switches, trans, tv, layer_profile, labs = [], [], {}, {}, {}, {}, {}, {}
    for name, c in cons.items():
        ma, mb = eng.arm_measures(ext, c)
        d = ma - mb
        ua, ub = eng.late_unit_measures(ext, c)
        du = (ua - ub).mean(axis=0)  # [U, M]: per-passage means over rows
        for m in MEASURES:
            k = eng.M[m]
            heads[(name, m)] = eng.head_effects(d[..., k], boot, mq, mabs)
            pooled.append({"contrast": name, "a": c.a, "b": c.b, "measure": m,
                           **eng.verdict(eng.late_mean(d[..., k], layers), boot, margin)})
            u = eng.unit_ci(du[:, k], boot)
            unit_rows.append({"contrast": name, "measure": m, **{f"unit_{key}": val for key, val in u.items()}})
            layer_profile[(name, m)] = d[..., k].mean(axis=(0, 2))
        la = eng.labels(ext.att_row[:, ext.index(c.a)], c.codes, cfg["diffuse_threshold"])
        lb = eng.labels(ext.att_row[:, ext.index(c.b)], c.codes, cfg["diffuse_threshold"])
        for i, kind in enumerate(KINDS):
            labs[(name, kind)] = (la[i], lb[i])
            switches[(name, kind)] = eng.role_switches(la[i], lb[i], sq, shift)
            trans[(name, kind)] = eng.transition_matrix(switches[(name, kind)])
        n_cat = len(eng.CATEGORIES)
        tv[name] = eng.tv_distance(ma[..., :n_cat], mb[..., :n_cat])
    placebo = {n: {kind: sorted(eng.switching_heads(*labs[(n, kind)], sq, shift)) for kind in KINDS} for n in placebos}
    roles = {(name, kind, arm): eng.head_roles(labs[(name, kind)][j])
             for (name, kind) in labs for j, arm in enumerate(("a", "b"))}

    # sweep over k (late pooled BOS; the same numbers as the pooled BOS rows of E_k / order_k / content_k)
    by = {(p["contrast"], p["measure"]): p for p in pooled}
    units_by = {(u["contrast"], u["measure"]): u for u in unit_rows}
    sweep = []
    for series, prefix in (("E", "E"), ("order", "order"), ("content", "content")):
        for k in ks:
            key = (f"{prefix}_{k}", "BOS")
            if key in by:
                sweep.append({"series": series, "k": k, **{x: by[key][x] for x in ("mean", "lo", "hi", "lo90", "hi90",
                                                                                  "wilcoxon_p", "verdict")},
                              **{x: units_by[key][x] for x in ("unit_mean", "unit_lo", "unit_hi", "unit_n_pos",
                                                               "unit_units")}})
    e_rows = np.stack([combo_rows(ext, {f"nl_{k}": 1, "none": -1}, layers) for k in ks], axis=1)
    unit_e = np.stack([combo_units(ext, {f"nl_{k}": 1, "none": -1}) for k in ks], axis=1)  # [U, n_k]
    shape = next(iter(heads.values()))["mean"].shape
    sw_len, sw_dest = {}, {}
    for kind in KINDS:
        sw_len[kind] = switch_lengths(ks, {k: switched_mask(switches[(f"E_{k}", kind)], shape) for k in ks})
        dest: dict[str, list[str]] = {}
        for s in switches[(f"E_{ks[-1]}", kind)]:  # a head may switch toward more than one label
            dest.setdefault(head_name(s["layer"], s["head"]), []).append(f"{s['from']}→{s['to']}")
        sw_dest[kind] = {h: ", ".join(v) for h, v in dest.items()}
    res = {"contrasts": cons, "pooled": pooled, "units": unit_rows, "heads": heads, "switches": switches,
           "transitions": trans, "tv": tv, "layer_profile": layer_profile, "placebo_switching": placebo,
           "roles": roles, "labels": {f"{n}/{k}/{arm}": labs[(n, k)][j] for (n, k) in labs for j, arm in enumerate("ab")},
           "sweep": sweep, "e_rows": e_rows, "unit_e": unit_e,
           "saturation": fit_saturation(ks, e_rows, boot, a["saturation"]),
           "types": type_sweep(unit_e, design.units, types),
           "head_curves": np.stack([heads[(f"E_{k}", "BOS")]["mean"] for k in ks]),  # [n_k, L, H]
           "switch_length": sw_len, "switch_dest": sw_dest}
    res["predictions"] = {s["id"]: evaluate(s, res, ext, cfg) for s in specs}
    return res


def _part(part: dict[str, Any], res: dict[str, Any], ext: eng.Extraction, cfg: dict[str, Any]) -> dict[str, Any]:
    a = cfg["analysis"]
    boot, test = a["bootstrap"], part["test"]
    if test == "switch_count":
        lo, hi = part["layers"]
        sw = res["switches"][(part["contrast"], part["labels"])]
        hits = [s for s in sw if s["from"] == part["from"] and s["to"] == part["to"] and lo <= s["layer"] <= hi]
        counts = {p: len(res["placebo_switching"][p][part["labels"]]) for p in part["placebo"]}
        n_p = max(counts.values())
        beyond = len(hits) - n_p
        names = ", ".join(f"{head_name(s['layer'], s['head'])} ({s['shift']:+.2f})" for s in hits) or "none"
        return {"ok": beyond >= part["min_beyond_placebo"],
                "detail": f"{len(hits)} heads in L{lo}-{hi} switch {part['from']}→{part['to']} in {part['contrast']} "
                          f"[{names}]; placebo {counts} (max {n_p}); beyond placebo {beyond} (need ≥ "
                          f"{part['min_beyond_placebo']})",
                "stats": {"count": len(hits), "placebo": counts, "beyond": beyond,
                          "heads": [head_name(s["layer"], s["head"]) for s in hits]}}
    combo = part["combo"]
    s = eng.interval(combo_rows(ext, combo, a["late_layers"]), boot)
    u = eng.unit_ci(combo_units(ext, combo), boot)
    unit_txt = f"per passage {u['mean']:+.4f} [{u['lo']:+.4f}, {u['hi']:+.4f}], {u['n_pos']}/{u['units']} > 0"
    ci95 = f"{s['mean']:+.4f} [{s['lo']:+.4f}, {s['hi']:+.4f}]"
    if test == "positive":
        ok, detail = s["lo"] > 0, f"{ci95} (need 95% CI > 0)"
    elif test == "negative":
        ok, detail = s["hi"] < 0, f"{ci95} (need 95% CI < 0)"
    elif test == "equivalent":
        ok = eng.equivalent(s, a["margin"])
        detail = f"{s['mean']:+.4f}, 90% CI [{s['lo90']:+.4f}, {s['hi90']:+.4f}] (need inside ±{a['margin']})"
    else:
        raise ValueError(f"unknown test {test!r}")
    return {"ok": bool(ok), "detail": f"{detail}; {unit_txt}", "stats": {**s, "unit": u}}


def evaluate(spec: dict[str, Any], res: dict[str, Any], ext: eng.Extraction, cfg: dict[str, Any]) -> dict[str, Any]:
    """One prediction in one base class -> {ok, detail, stats}. It holds when every part holds."""
    parts = [(p["name"], _part(p, res, ext, cfg)) for p in spec["parts"]]
    return {"ok": all(r["ok"] for _, r in parts),
            "detail": " | ".join(f"{n} [{'holds' if r['ok'] else 'fails'}]: {r['detail']}" for n, r in parts),
            "stats": {n: {"ok": r["ok"], **r["stats"]} for n, r in parts}}


def check_predictions(specs: list[dict], results: dict[str, dict]) -> list[dict]:
    """PASS only if the prediction holds in every base class."""
    out = []
    for s in specs:
        parts = [(res["predictions"][s["id"]]["ok"], f"{base}: {res['predictions'][s['id']]['detail']}")
                 for base, res in results.items()]
        ok = bool(parts) and all(p for p, _ in parts)
        out.append({"id": s["id"], "verdict": "PASS" if ok else "FAIL", "detail": "; ".join(d for _, d in parts),
                    "per_base": {base: bool(res["predictions"][s["id"]]["ok"]) for base, res in results.items()}})
    return out
