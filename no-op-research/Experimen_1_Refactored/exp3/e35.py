"""Experiment 3, E3.5: punctuation, does a delimiter replace BOS as the resting place? (exp3/DESIGN.md section 3).

Phrase P V O at 26-28 (or, in the random context, the row's own tokens there), d at 29, and the row's own tokens
at 30-32 (query at 32). d is a byte token (delimiters ".", ",", "\\n"; non-delimiter controls "x", "7") or r, the
row's own token. The full 2 x 6 grid (context x d) plus the placebo random_b+r: random+r with 26-28 redrawn once per
row. Delimiter = position 29 in every variant (rule 1), so its mask is the same in both arms of every contrast.

The units are the 12 phrases. The random-context variants and the placebo do not depend on the phrase, so every
unit holds the same sequence. extract_split() computes them once per row and broadcasts the result to every unit,
which halves the forward passes. The engine's extract() has no notion of unit-invariant variants.
Positions are model positions (BOS = 0); id arrays are WITHOUT BOS (array index = position - 1).
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from exp2.analysis import head_name
from exp2.design import derive_seed
from exp3 import engine as eng
from exp3.e34 import check_predictions  # PASS only if a prediction holds in every base class

CONTEXTS = ("phrase", "random")
PLACEBO = "random_b+r"
KINDS = ("argmax", "mass")
MEASURES = eng.MEASURES + ("BOS+Delimiter",)  # E5b's resting mass, appended to the engine's measures
MAX_BYTE_ID = 255  # GPT-2 ids 0-255 are the single-byte tokens

__all__ = ["CONTEXTS", "PLACEBO", "MEASURES", "variant", "d_names", "variant_names", "shared_variants", "build_e35",
           "check_e35", "split_design", "merge_extractions", "extract_split", "analyse", "previous_check", "evaluate",
           "check_predictions"]


def variant(context: str, d: str) -> str:
    return f"{context}+{d}"


def d_names(byte_ids: dict[str, int]) -> tuple[str, ...]:
    """The grid's d axis: the byte tokens in config order, then r."""
    if "r" in byte_ids:
        raise ValueError("'r' is reserved for the row's own token")
    return tuple(byte_ids) + ("r",)


def variant_names(byte_ids: dict[str, int]) -> tuple[str, ...]:
    """phrase+d for every d, random+d for every d, then the placebo."""
    return tuple(variant(c, d) for c in CONTEXTS for d in d_names(byte_ids)) + (PLACEBO,)


def shared_variants(byte_ids: dict[str, int]) -> tuple[str, ...]:
    """The variants that hold one sequence in every unit: the random context and the placebo."""
    return tuple(variant("random", d) for d in d_names(byte_ids)) + (PLACEBO,)


def _layout(layout: dict[str, Any]) -> tuple[list[int], int, int]:
    ph, delim, q = [int(p) for p in layout["phrase"]], int(layout["delimiter"]), int(layout["query"])
    if len(ph) != 3 or ph != list(range(ph[0], ph[0] + 3)) or delim != ph[-1] + 1 or delim > q - 2:
        raise ValueError(f"layout {layout}: expected the phrase on 3 consecutive positions, the delimiter right after "
                         "it, and the delimiter off q - 1")
    return ph, delim, q


# ---- design --------------------------------------------------------------------------------------------------

def build_e35(base: np.ndarray, phrases: dict[str, Sequence[int]], byte_ids: dict[str, int], base_pool: np.ndarray,
              seed: int, layout: dict[str, Any]) -> eng.Design:
    """Build the 13-variant Design; the units are the phrases.
    base: [rows, seq_len] base rows, without BOS. phrases: {name: [P, V, O]}. byte_ids: {name: id}, each 0-255.
    base_pool: the ids the base rows were drawn from; the placebo redraws 26-28 from it, avoiding the row."""
    ph, delim, q = _layout(layout)
    rows, seq_len = base.shape
    if q != seq_len:
        raise ValueError(f"the query ({q}) must be the final position ({seq_len})")
    bad = {n: t for n, t in byte_ids.items() if not 0 <= int(t) <= MAX_BYTE_ID}
    if bad:
        raise ValueError(f"byte ids must lie in 0-{MAX_BYTE_ID}: {bad}")
    if (base <= MAX_BYTE_ID).any():
        raise ValueError("the base rows hold a byte token; the byte at 29 must be the row's only one")
    units = tuple(phrases)
    ip = np.asarray(ph) - 1
    stacked = np.repeat(base[:, None, :], len(units), axis=1)
    seqs = {v: stacked.copy() for v in variant_names(byte_ids)}
    named = {"Delimiter": (delim,)}  # position 29 in every variant, also where it holds r
    meta = {}
    for d in d_names(byte_ids):
        tok = byte_ids.get(d)
        at_delim = () if tok is None else (delim,)
        for ctx in CONTEXTS:
            v = variant(ctx, d)
            if tok is not None:
                seqs[v][:, :, delim - 1] = int(tok)
            if ctx == "phrase":
                for u, name in enumerate(units):
                    seqs[v][:, u, ip] = np.asarray(phrases[name], dtype=np.int64)
            meta[v] = eng.VariantMeta((tuple(ph) if ctx == "phrase" else ()) + at_delim, dict(named))
    alt = eng.draw_per_row(base, [base_pool] * len(ph), derive_seed(seed, "placebo"))  # once per row
    seqs[PLACEBO][:, :, ip] = alt[:, None, :]
    meta[PLACEBO] = eng.VariantMeta(tuple(ph), dict(named))
    return eng.Design(seqs, stacked, meta, units)


def check_e35(design: eng.Design, phrases: dict[str, Sequence[int]], byte_ids: dict[str, int], base_pool: np.ndarray,
              layout: dict[str, Any]) -> None:
    """Raise unless the E3.5 layout holds:
    - phrase+d vs phrase+r and random+d vs random+r differ only at 29;
    - phrase+d vs random+d and the placebo vs random+r differ only at 26-28;
    - random+r is the base row; the phrase and byte ids sit where the design says;
    - the random variants and the placebo are identical across units;
    - the byte at 29 is the row's only id <= 255 (none in the r arms);
    - the placebo's ids come from the base pool and differ from the row's own."""
    ph, delim, q = _layout(layout)
    s = design.seqs
    if set(s) != set(variant_names(byte_ids)):
        raise AssertionError(f"variants {sorted(s)} do not match the grid {sorted(variant_names(byte_ids))}")
    for d in byte_ids:
        for ctx in CONTEXTS:
            eng.check_differ_only_at(design, variant(ctx, d), variant(ctx, "r"), [delim])
    for d in d_names(byte_ids):
        eng.check_differ_only_at(design, variant("phrase", d), variant("random", d), ph)
    eng.check_differ_only_at(design, PLACEBO, variant("random", "r"), ph)
    if not (s[variant("random", "r")] == design.base).all():
        raise AssertionError("random+r (none) is not the base row")
    ip = np.asarray(ph) - 1
    for u, name in enumerate(design.units):
        for d in d_names(byte_ids):
            if not (s[variant("phrase", d)][:, u][:, ip] == np.asarray(phrases[name])).all():
                raise AssertionError(f"phrase+{d}, {name}: positions {ph} do not hold {list(phrases[name])}")
    for d, tok in byte_ids.items():
        for ctx in CONTEXTS:
            if not (s[variant(ctx, d)][..., delim - 1] == tok).all():
                raise AssertionError(f"{variant(ctx, d)}: position {delim} does not hold {tok}")
    for v in shared_variants(byte_ids):
        if not (s[v] == s[v][:, :1]).all():
            raise AssertionError(f"{v} differs across units; it must not depend on the phrase")
    for v, x in s.items():
        expect = np.zeros(x.shape[-1], dtype=bool)
        expect[delim - 1] = v.split("+")[-1] in byte_ids
        if not ((x <= MAX_BYTE_ID) == expect).all():
            raise AssertionError(f"{v}: a byte token (id <= {MAX_BYTE_ID}) off position {delim}, or none at it")
    alt = s[PLACEBO][..., ip]
    if not np.isin(alt, base_pool).all() or (alt == design.base[..., ip]).any():
        raise AssertionError(f"{PLACEBO}: redrawn ids outside the base pool, or equal to the row's own")


# ---- extraction ----------------------------------------------------------------------------------------------

def split_design(design: eng.Design, shared: Sequence[str]) -> dict[str, eng.Design]:
    """{'per_unit': the unit-dependent variants, 'shared': the `shared` variants with one unit}. Raises unless each
    shared variant holds one sequence in every unit."""
    per = [v for v in design.variants if v not in shared]
    for v in shared:
        if not (design.seqs[v] == design.seqs[v][:, :1]).all():
            raise AssertionError(f"{v} differs across units; it cannot be computed once per row")
    out = {}
    if per:
        out["per_unit"] = eng.Design({v: design.seqs[v] for v in per}, design.base, {v: design.meta[v] for v in per},
                                     design.units)
    if shared:
        out["shared"] = eng.Design({v: np.ascontiguousarray(design.seqs[v][:, :1]) for v in shared},
                                   np.ascontiguousarray(design.base[:, :1]), {v: design.meta[v] for v in shared},
                                   ("shared",))
    return out


def merge_extractions(design: eng.Design, parts: dict[str, tuple[eng.Design, eng.Extraction]]) -> eng.Extraction:
    """One Extraction over every variant of `design`, in its order. A one-unit part is broadcast to every unit:
    its per-row unit mean is its single instance, and its per-unit arrays repeat across units."""
    exts = [e for _, e in parts.values()]
    rows = {e.att_row.shape[0] for e in exts}
    if len(rows) != 1 or len({e.late_layers for e in exts}) != 1:
        raise ValueError("parts disagree on the row count or the late layers")
    rows, units, variants = rows.pop(), len(design.units), design.variants
    _, _, n_layers, n_heads, _ = exts[0].att_row.shape
    width = max(e.att_row.shape[-1] for e in exts)
    att_row = np.zeros((rows, len(variants), n_layers, n_heads, width), dtype=np.float32)
    att_late_unit = np.zeros((rows, units, len(variants), width), dtype=np.float32)
    att_unit = np.zeros((units, len(variants), n_layers, n_heads, width), dtype=np.float32)
    for d, e in parts.values():
        k = e.att_row.shape[-1]
        for v in d.variants:
            i, j = variants.index(v), e.index(v)
            att_row[:, i, ..., :k] = e.att_row[:, j]
            att_late_unit[:, :, i, :k] = e.att_late_unit[:, :, j]
            att_unit[:, i, ..., :k] = e.att_unit[:, j]
    return eng.Extraction(variants=variants, units=design.units,
                          key_len=np.array([design.key_len(v) for v in variants]), late_layers=exts[0].late_layers,
                          att_row=att_row, att_late_unit=att_late_unit, att_unit=att_unit,
                          max_row_sum_dev=max(e.max_row_sum_dev for e in exts),
                          n_sequences=sum(e.n_sequences for e in exts), seconds=sum(e.seconds for e in exts))


def extract_split(model, design: eng.Design, shared: Sequence[str], bos_id: int, late_layers: Sequence[int],
                  chunk_rows: dict[str, int], row_sum_atol: float, n_rows: int | None = None,
                  log: Callable[[str], None] | None = None
                  ) -> tuple[eng.Extraction, dict[str, tuple[eng.Design, eng.Extraction]]]:
    """eng.extract() on each part of split_design() (chunk_rows per part name), merged. Also returns the parts, for
    the determinism and cross-library checks, which run per part on the batches that were actually used."""
    parts = {}
    for name, d in split_design(design, shared).items():
        plog = None if log is None else (lambda m, name=name: log(f"{name}: {m}"))
        parts[name] = (d, eng.extract(model, d, bos_id, late_layers, chunk_rows[name], row_sum_atol, n_rows, plog))
    return merge_extractions(design, parts), parts


# ---- analysis ------------------------------------------------------------------------------------------------

def _with_rest(m: np.ndarray) -> np.ndarray:
    """[..., len(eng.MEASURES)] -> [..., len(MEASURES)]: appends BOS + Delimiter mass."""
    return np.concatenate([m, m[..., eng.C["BOS"], None] + m[..., eng.C["Delimiter"], None]], axis=-1)


def analyse(ext: eng.Extraction, design: eng.Design, cfg: dict[str, Any], contrasts: dict[str, dict],
            interactions: dict[str, dict], specs: list[dict]) -> dict[str, Any]:
    """Every contrast:
    - per-head tests of each measure;
    - the late pooled per-row verdict and the CI pooled over units;
    - role switches (argmax and mass labels), transition matrices, TV distance.
    Every interaction (treated contrast - control contrast, identical masks): the same tests, without labels.
    Also the placebo switch counts, the previous-token-head check, and each prediction's per-class result."""
    a = cfg["analysis"]
    boot, layers, margin = a["bootstrap"], a["late_layers"], a["margin"]
    mq, mabs = a["meaningful"]["q"], a["meaningful"]["min_abs"]
    sq, shift = a["switch"]["q"], a["switch"]["min_shift"]
    cons = {n: eng.make_contrast(design, n, s["a"], s["b"], s.get("mask_from")) for n, s in contrasts.items()}
    if "placebo" not in cons:
        raise ValueError("the contrasts must include one named 'placebo'")
    if set(interactions) & set(cons):
        raise ValueError(f"names used for both a contrast and an interaction: {sorted(set(interactions) & set(cons))}")

    diffs, switches, trans, tv, labs, what = {}, {}, {}, {}, {}, {}
    for name, c in cons.items():
        ma, mb = (_with_rest(x) for x in eng.arm_measures(ext, c))
        ua, ub = (_with_rest(x) for x in eng.late_unit_measures(ext, c))
        diffs[name] = (ma - mb, (ua - ub).mean(axis=0))  # per row [rows, L, H, M]; per unit [U, M]
        what[name] = f"{c.a} − {c.b}"
        la = eng.labels(ext.att_row[:, ext.index(c.a)], c.codes, cfg["diffuse_threshold"])
        lb = eng.labels(ext.att_row[:, ext.index(c.b)], c.codes, cfg["diffuse_threshold"])
        for i, kind in enumerate(KINDS):
            labs[(name, kind)] = (la[i], lb[i])
            switches[(name, kind)] = eng.role_switches(la[i], lb[i], sq, shift)
            trans[(name, kind)] = eng.transition_matrix(switches[(name, kind)])
        n_cat = len(eng.CATEGORIES)
        tv[name] = eng.tv_distance(ma[..., :n_cat], mb[..., :n_cat])
    for name, s in interactions.items():
        t, k = cons[s["treated"]], cons[s["control"]]
        if not np.array_equal(t.codes, k.codes) or t.phrase != k.phrase:
            raise ValueError(f"interaction {name}: {t.name} and {k.name} use different masks")
        diffs[name] = (diffs[t.name][0] - diffs[k.name][0], diffs[t.name][1] - diffs[k.name][1])
        what[name] = f"({what[t.name]}) − ({what[k.name]})"

    pooled, unit_rows, heads, layer_profile = [], [], {}, {}
    for name, (d, du) in diffs.items():
        for k, m in enumerate(MEASURES):
            heads[(name, m)] = eng.head_effects(d[..., k], boot, mq, mabs)
            pooled.append({"contrast": name, "what": what[name], "measure": m,
                           **eng.verdict(eng.late_mean(d[..., k], layers), boot, margin)})
            u = eng.unit_ci(du[:, k], boot)
            unit_rows.append({"contrast": name, "measure": m, **{f"unit_{key}": val for key, val in u.items()}})
            layer_profile[(name, m)] = d[..., k].mean(axis=(0, 2))
    placebo = {kind: sorted(eng.switching_heads(*labs[("placebo", kind)], sq, shift)) for kind in KINDS}
    roles = {(name, kind, arm): eng.head_roles(labs[(name, kind)][j])
             for (name, kind) in labs for j, arm in enumerate(("a", "b"))}

    res = {"contrasts": cons, "interactions": dict(interactions), "what": what, "pooled": pooled, "units": unit_rows,
           "heads": heads, "switches": switches, "transitions": trans, "tv": tv, "layer_profile": layer_profile,
           "placebo_switching": placebo, "roles": roles,
           "labels": {f"{n}/{k}/{arm}": labs[(n, k)][j] for (n, k) in labs for j, arm in enumerate("ab")},
           "previous": previous_check(ext, design, cfg["previous_heads"], cfg["diffuse_threshold"], switches)}
    res["predictions"] = {s["id"]: evaluate(s, res) for s in specs}
    return res


def previous_check(ext: eng.Extraction, design: eng.Design, heads: Sequence[str], diffuse_threshold: float,
                   switches: dict[tuple[str, str], list[dict]]) -> dict[str, Any]:
    """Exp 1's previous-token heads keep Previous throughout: in every variant each head's role (most common argmax
    label over rows, that variant's own masks) is Previous, and none has an argmax switch whose source is Previous."""
    listed = eng.parse_heads(heads)
    prev = eng.LABELS.index("Previous")
    width = ext.att_row.shape[-1]
    roles, shares, mass = {}, {}, {}
    for v in design.variants:
        vi, kl = ext.index(v), design.key_len(v)
        lab = eng.labels(ext.att_row[:, vi], eng.category_codes(design.meta[v], kl, width), diffuse_threshold)[0]
        role = eng.head_roles(lab)
        roles[v] = {head_name(l_, h_): eng.LABELS[int(role[l_, h_])] for l_, h_ in listed}
        shares[v] = {head_name(l_, h_): float((lab[:, l_, h_] == prev).mean()) for l_, h_ in listed}
        mass[v] = {head_name(l_, h_): float(ext.att_row[:, vi, l_, h_, kl - 2].mean()) for l_, h_ in listed}  # key q-1
    lost = [{"contrast": name, **s} for (name, kind), sw in switches.items() if kind == "argmax"
            for s in sw if (s["layer"], s["head"]) in set(listed) and s["from"] == "Previous"]
    off = [{"variant": v, "head": n, "role": r} for v, rr in roles.items() for n, r in rr.items() if r != "Previous"]
    return {"ok": not off and not lost, "heads": list(heads), "roles": roles, "previous_share": shares,
            "previous_mass": mass, "not_previous": off, "switches_from_previous": lost}


def _pooled(res: dict[str, Any], name: str, measure: str) -> tuple[dict, dict]:
    p = next((x for x in res["pooled"] if x["contrast"] == name and x["measure"] == measure), None)
    if p is None:
        raise KeyError(f"no pooled result for {name} / {measure}")
    u = next(x for x in res["units"] if x["contrast"] == name and x["measure"] == measure)
    return p, u


def evaluate(spec: dict[str, Any], res: dict[str, Any]) -> dict[str, Any]:
    """One prediction in one base class -> {ok, detail, stats}. Kinds: all (every part holds), switch_count,
    positive (95% CI of the late pooled per-row value > 0), noninferior (one-sided 95% bound > -margin)."""
    kind = spec["kind"]
    if kind == "all":
        parts = [evaluate(p, res) for p in spec["parts"]]
        return {"ok": all(p["ok"] for p in parts),
                "detail": " AND ".join(f"({'ok' if p['ok'] else 'fails'}) {p['detail']}" for p in parts),
                "stats": {"parts": [p["stats"] for p in parts]}}
    if kind == "switch_count":
        lo, hi = spec["layers"]
        sw = res["switches"][(spec["contrast"], spec["labels"])]
        hits = [s for s in sw if s["from"] == spec["from"] and s["to"] == spec["to"] and lo <= s["layer"] <= hi]
        n_p = len(res["placebo_switching"][spec["labels"]])
        beyond = len(hits) - n_p
        names = ", ".join(f"{head_name(s['layer'], s['head'])} ({s['shift']:+.2f})" for s in hits) or "none"
        return {"ok": beyond >= spec["min_beyond_placebo"],
                "detail": f"{spec['contrast']}: {len(hits)} heads in L{lo}-{hi} switch {spec['from']}→{spec['to']} "
                          f"[{names}]; placebo {n_p}; beyond placebo {beyond} (need ≥ {spec['min_beyond_placebo']})",
                "stats": {"count": len(hits), "placebo": n_p, "beyond": beyond,
                          "heads": [head_name(s["layer"], s["head"]) for s in hits]}}
    p, u = _pooled(res, spec["of"], spec["measure"])
    unit_txt = (f"per phrase {u['unit_mean']:+.4f} [{u['unit_lo']:+.4f}, {u['unit_hi']:+.4f}], "
                f"{u['unit_n_pos']}/{u['unit_units']} > 0")
    stats = {**{k: v for k, v in p.items() if k not in ("contrast", "what", "measure")}, "unit": u}
    if kind == "positive":
        return {"ok": p["lo"] > 0,
                "detail": f"{spec['of']} on {spec['measure']}: {p['mean']:+.4f} [{p['lo']:+.4f}, {p['hi']:+.4f}]; "
                          f"{unit_txt}", "stats": stats}
    if kind == "noninferior":
        return {"ok": eng.noninferior(p, spec["margin"]),
                "detail": f"{spec['of']} on {spec['measure']}: {p['mean']:+.4f}, one-sided 95% bound {p['lo90']:+.4f} "
                          f"(need > {-spec['margin']:+.3f}); {unit_txt}", "stats": stats}
    raise ValueError(f"unknown prediction kind {kind!r}")
