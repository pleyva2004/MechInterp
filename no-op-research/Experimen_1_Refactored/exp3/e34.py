"""Experiment 3, E3.4: repeated phrase, induction as the positive control (exp3/DESIGN.md section 3).

Phrase A B C (pronoun-verb-object) sits early at 10-12 and the query is B at 32, so Copy = 12 (holds C) and Dup = 11
(the earlier B). Every base row carries every variant once per phrase (the unit). The phrase-independent variants
(repeat-rand, norepeat-rand, norepeat-rand_b) draw a fresh X Y Z (and r') for each row and unit, from the base pool
minus that row's ids.

Fallback: a 15-id random block at 2-16, its first 13 ids again at 19-31, query = its 14th id (Dup 15, Copy 16), one
instance per row. Positions are model positions (BOS = 0); id arrays are WITHOUT BOS (array index = position - 1).
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from exp2.analysis import head_name
from exp2.design import derive_seed
from exp3 import engine as eng

VARIANTS = ("repeat-nl", "norepeat-nl", "other-nl", "repeat-nl-1", "repeat-matched", "repeat-rand", "norepeat-rand",
            "norepeat-rand_b")
FALLBACK_VARIANTS = ("fb-repeat", "fb-norepeat", "fb-norepeat_b")
KINDS = ("argmax", "mass")


def _layout(layout: dict[str, Any]) -> tuple[list[int], int, int]:
    ph, prev, q = [int(p) for p in layout["phrase"]], int(layout["previous"]), int(layout["query"])
    if len(ph) != 3 or layout["dup"] != ph[1] or layout["copy"] != ph[2] or prev != q - 1 or ph[2] >= prev:
        raise ValueError(f"layout {layout}: expected phrase A B C with dup = B's position, copy = C's, previous = q - 1")
    return ph, prev, q


def build_e34(base: np.ndarray, phrases: dict[str, Sequence[int]], other: dict[str, str],
              matched: dict[str, Sequence[np.ndarray]], base_pool: np.ndarray, seed: int,
              layout: dict[str, Any]) -> eng.Design:
    """Build the 8-variant Design; the units are the phrases.
    base: [rows, seq_len] base rows, without BOS. phrases: {name: [A, B, C]}. other: {name: its other-nl phrase}.
    matched: {name: three matched pools}. base_pool: the ids the base rows were drawn from."""
    ph, prev, q = _layout(layout)
    rows, seq_len = base.shape
    if q != seq_len:
        raise ValueError(f"the query ({q}) must be the final position ({seq_len})")
    units = tuple(phrases)
    ip, iprev, iq = np.asarray(ph) - 1, prev - 1, q - 1
    stacked = np.repeat(base[:, None, :], len(units), axis=1)
    seqs = {v: stacked.copy() for v in VARIANTS}
    for u, name in enumerate(units):
        abc = np.asarray(phrases[name], dtype=np.int64)
        oth = np.asarray(phrases[other[name]], dtype=np.int64)
        mat = eng.draw_per_row(base, matched[name], derive_seed(seed, f"matched/{name}"))
        xyz = eng.draw_per_row(base, [base_pool] * 3, derive_seed(seed, f"rand/{name}"))
        alt = eng.draw_per_row(np.concatenate([base, xyz], axis=1), [base_pool] * 3, derive_seed(seed, f"rand_b/{name}"))
        writes = {  # variant: (10-12, 31, 32); None keeps the base row's own tokens
            "repeat-nl": (abc, abc[0], abc[1]),
            "norepeat-nl": (None, abc[0], abc[1]),
            "other-nl": (oth, abc[0], abc[1]),
            "repeat-nl-1": (abc, None, abc[1]),
            "repeat-matched": (mat, mat[:, 0], mat[:, 1]),
            "repeat-rand": (xyz, xyz[:, 0], xyz[:, 1]),
            "norepeat-rand": (None, xyz[:, 0], xyz[:, 1]),
            "norepeat-rand_b": (alt, xyz[:, 0], xyz[:, 1]),
        }
        for v, (span, at_prev, at_q) in writes.items():
            s = seqs[v][:, u]  # a view into seqs[v]
            if span is not None:
                s[:, ip] = span
            if at_prev is not None:
                s[:, iprev] = at_prev
            s[:, iq] = at_q
    full = tuple(ph) + (prev, q)
    named = {"Copy": (ph[2],), "Dup": (ph[1],)}
    rep = ((ph[0], prev), (ph[1], q))
    meta = {
        "repeat-nl": eng.VariantMeta(full, named, rep),
        "norepeat-nl": eng.VariantMeta((prev, q)),
        "other-nl": eng.VariantMeta(full),
        "repeat-nl-1": eng.VariantMeta(tuple(ph) + (q,), named, ((ph[1], q),)),
        "repeat-matched": eng.VariantMeta(full, named, rep),
        "repeat-rand": eng.VariantMeta(full, named, rep),
        "norepeat-rand": eng.VariantMeta((prev, q)),
        "norepeat-rand_b": eng.VariantMeta(full),
    }
    return eng.Design(seqs, stacked, meta, units)


def check_e34(design: eng.Design, phrases: dict[str, Sequence[int]], other: dict[str, str],
              matched: dict[str, Sequence[np.ndarray]], base_pool: np.ndarray, layout: dict[str, Any]) -> None:
    """Raise unless the E3.4 layout holds:
    - each control differs from its treated variant only at 10-12, and repeat-nl-1 from repeat-nl only at 31;
    - the phrase / other-phrase / matched / random ids sit where the table says;
    - other-nl shares no id with its phrase;
    - the query is identical within each contrast."""
    ph, prev, q = _layout(layout)
    s = design.seqs
    for ctrl in ("norepeat-nl", "other-nl"):
        eng.check_differ_only_at(design, "repeat-nl", ctrl, ph)
    eng.check_differ_only_at(design, "repeat-nl", "repeat-nl-1", [prev])
    for ctrl in ("norepeat-rand", "norepeat-rand_b"):
        eng.check_differ_only_at(design, "repeat-rand", ctrl, ph)
    ip = np.asarray(ph) - 1
    for u, name in enumerate(design.units):
        abc, oth = np.asarray(phrases[name]), np.asarray(phrases[other[name]])
        if other[name] == name or np.isin(oth, abc).any():
            raise AssertionError(f"other-nl for {name} is {other[name]}, which shares an id with it")
        for v, ids in (("repeat-nl", abc), ("repeat-nl-1", abc), ("other-nl", oth)):
            if not (s[v][:, u][:, ip] == ids).all():
                raise AssertionError(f"{v}, {name}: positions {ph} do not hold {ids.tolist()}")
        if not (s["repeat-nl"][:, u, q - 1] == abc[1]).all() or not (s["repeat-nl"][:, u, prev - 1] == abc[0]).all():
            raise AssertionError(f"repeat-nl, {name}: positions {prev}/{q} do not hold A/B")
        for j in range(3):
            if not np.isin(s["repeat-matched"][:, u, ph[j] - 1], matched[name][j]).all():
                raise AssertionError(f"repeat-matched, {name}: position {ph[j]} holds an id outside its matched pool")
        for v in ("repeat-rand", "norepeat-rand_b"):
            if not np.isin(s[v][:, u][:, ip], base_pool).all():
                raise AssertionError(f"{v}, {name}: a random id outside the base pool")


def build_fallback(base: np.ndarray, base_pool: np.ndarray, seed: int, layout: dict[str, Any]) -> eng.Design:
    """The pre-registered fallback, one unit ('random'):
    - fb-repeat: block B (15 ids) at 2-16, B[0..12] again at 19-31, query B[13];
    - fb-norepeat: 2-16 left as the row's own tokens;
    - fb-norepeat_b: 2-16 replaced by an independent draw (the placebo).
    B and the redraw avoid every id of the row."""
    b0, b1 = (int(x) for x in layout["block"])
    r0, r1 = (int(x) for x in layout["repeat"])
    q = int(layout["query"])
    block, rep = np.arange(b0, b1 + 1), np.arange(r0, r1 + 1)
    dup, copy = b0 + len(rep), b0 + len(rep) + 1
    if (dup, copy) != (layout["dup"], layout["copy"]) or q != base.shape[1] or copy > b1 or r0 <= b1 or r1 != q - 1:
        raise ValueError(f"fallback layout {layout} is inconsistent (derived dup {dup}, copy {copy})")
    blk = eng.draw_per_row(base, [base_pool] * len(block), derive_seed(seed, "fallback/block"))
    alt = eng.draw_per_row(np.concatenate([base, blk], axis=1), [base_pool] * len(block), derive_seed(seed, "fallback/alt"))
    seqs = {v: base[:, None, :].copy() for v in FALLBACK_VARIANTS}
    for v, first in (("fb-repeat", blk), ("fb-norepeat", None), ("fb-norepeat_b", alt)):
        s = seqs[v][:, 0]
        if first is not None:
            s[:, block - 1] = first
        s[:, rep - 1] = blk[:, :len(rep)]
        s[:, q - 1] = blk[:, len(rep)]
    tail = tuple(int(p) for p in rep) + (q,)
    groups = tuple((b0 + j, r0 + j) for j in range(len(rep))) + ((dup, q),)
    meta = {"fb-repeat": eng.VariantMeta(tuple(int(p) for p in block) + tail, {"Copy": (copy,), "Dup": (dup,)}, groups),
            "fb-norepeat": eng.VariantMeta(tail),
            "fb-norepeat_b": eng.VariantMeta(tuple(int(p) for p in block) + tail)}
    return eng.Design(seqs, base[:, None, :].copy(), meta, ("random",))


def check_fallback(design: eng.Design, base_pool: np.ndarray, layout: dict[str, Any]) -> None:
    block = list(range(int(layout["block"][0]), int(layout["block"][1]) + 1))
    for ctrl in ("fb-norepeat", "fb-norepeat_b"):
        eng.check_differ_only_at(design, "fb-repeat", ctrl, block)
    for v in ("fb-repeat", "fb-norepeat_b"):
        if not np.isin(design.seqs[v][:, 0, np.asarray(block) - 1], base_pool).all():
            raise AssertionError(f"{v}: a block id outside the base pool")


# ---- analysis ------------------------------------------------------------------------------------------------

def _names(mask: np.ndarray) -> list[str]:
    return [head_name(int(layer), int(head)) for layer, head in zip(*np.nonzero(mask))]


def _unit_head_diff(ext: eng.Extraction, c: eng.Contrast, k: int, heads: np.ndarray) -> np.ndarray:
    """[U]: per unit, the mean over rows of measure k in a - b, averaged over the heads in mask `heads`."""
    ua = eng.measures(ext.att_unit[:, ext.index(c.a)], c.codes, c.phrase)[..., k]
    ub = eng.measures(ext.att_unit[:, ext.index(c.b)], c.codes, c.phrase)[..., k]
    return (ua - ub)[:, heads].mean(axis=1)


def analyse(ext: eng.Extraction, design: eng.Design, cfg: dict[str, Any], contrasts: dict[str, dict],
            induction: dict[str, Any], specs: list[dict]) -> dict[str, Any]:
    """Every contrast:
    - per-head tests of each measure;
    - the late pooled per-row verdict and the CI pooled over units;
    - role switches (argmax and mass labels), transition matrices, TV distance.
    A contrast spec with `measures: [...]` is restricted to those measures, with no labels, switches or TV.
    Also the induction heads, the placebo switch counts, and each prediction's per-class result."""
    a = cfg["analysis"]
    boot, layers, margin = a["bootstrap"], a["late_layers"], a["margin"]
    mq, mabs = a["meaningful"]["q"], a["meaningful"]["min_abs"]
    sq, shift = a["switch"]["q"], a["switch"]["min_shift"]
    cons = {n: eng.make_contrast(design, n, s["a"], s["b"], s.get("mask_from")) for n, s in contrasts.items()}
    if "placebo" not in cons:
        raise ValueError("the contrasts must include one named 'placebo'")

    iv = induction["variant"]
    copy_pos = design.meta[iv].categories["Copy"][0]
    copy_mean = ext.att_row[:, ext.index(iv), :, :, copy_pos].mean(axis=0, dtype=np.float64)
    ind = copy_mean >= induction["threshold"]

    pooled, unit_rows, heads, switches, trans, tv, layer_profile, arms, labs = [], [], {}, {}, {}, {}, {}, {}, {}
    for name, c in cons.items():
        ma, mb = eng.arm_measures(ext, c)
        d = ma - mb
        arms[name] = (ma, mb)
        ua, ub = eng.late_unit_measures(ext, c)
        du = (ua - ub).mean(axis=0)  # [U, M]: per-unit means over rows
        keep = contrasts[name].get("measures", eng.MEASURES)
        for k, m in enumerate(eng.MEASURES):
            if m not in keep:
                continue
            heads[(name, m)] = eng.head_effects(d[..., k], boot, mq, mabs)
            pooled.append({"contrast": name, "a": c.a, "b": c.b, "measure": m,
                           **eng.verdict(eng.late_mean(d[..., k], layers), boot, margin)})
            u = eng.unit_ci(du[:, k], boot)
            unit_rows.append({"contrast": name, "measure": m, **{f"unit_{key}": val for key, val in u.items()}})
            layer_profile[(name, m)] = d[..., k].mean(axis=(0, 2))
        if "measures" in contrasts[name]:
            continue  # a restricted comparison (different tokens at 31-32): no labels, switches or TV
        la = eng.labels(ext.att_row[:, ext.index(c.a)], c.codes, cfg["diffuse_threshold"])
        lb = eng.labels(ext.att_row[:, ext.index(c.b)], c.codes, cfg["diffuse_threshold"])
        for i, kind in enumerate(KINDS):
            labs[(name, kind)] = (la[i], lb[i])
            switches[(name, kind)] = eng.role_switches(la[i], lb[i], sq, shift)
            trans[(name, kind)] = eng.transition_matrix(switches[(name, kind)])
        n_cat = len(eng.CATEGORIES)
        tv[name] = eng.tv_distance(ma[..., :n_cat], mb[..., :n_cat])
    placebo = {kind: sorted(eng.switching_heads(*labs[("placebo", kind)], sq, shift)) for kind in KINDS}
    roles = {(name, kind, arm): eng.head_roles(labs[(name, kind)][j])
             for (name, kind) in labs for j, arm in enumerate(("a", "b"))}

    res = {"contrasts": cons, "pooled": pooled, "units": unit_rows, "heads": heads, "switches": switches,
           "transitions": trans, "tv": tv, "layer_profile": layer_profile, "placebo_switching": placebo,
           "roles": roles, "labels": {f"{n}/{k}/{arm}": labs[(n, k)][j] for (n, k) in labs for j, arm in enumerate("ab")},
           "induction": {"variant": iv, "copy_position": copy_pos, "copy_mean": copy_mean, "mask": ind,
                         "heads": _names(ind), "threshold": induction["threshold"]}}
    res["predictions"] = {s["id"]: evaluate(s, res, arms, ext, cfg) for s in specs}
    return res


def evaluate(spec: dict[str, Any], res: dict[str, Any], arms: dict[str, tuple], ext: eng.Extraction,
             cfg: dict[str, Any]) -> dict[str, Any]:
    """One prediction in one base class -> {ok, detail, stats}."""
    a = cfg["analysis"]
    boot = a["bootstrap"]
    ind = res["induction"]["mask"]
    kind = spec["kind"]
    if kind == "switch_count":
        lo, hi = spec["layers"]
        sw = res["switches"][(spec["contrast"], spec["labels"])]
        hits = [s for s in sw if s["from"] == spec["from"] and s["to"] == spec["to"] and lo <= s["layer"] <= hi]
        n_p = len(res["placebo_switching"][spec["labels"]])
        beyond = len(hits) - n_p
        names = ", ".join(f"{head_name(s['layer'], s['head'])} ({s['shift']:+.2f})" for s in hits) or "none"
        return {"ok": beyond >= spec["min_beyond_placebo"],
                "detail": f"{len(hits)} heads in L{lo}-{hi} switch {spec['from']}→{spec['to']} [{names}]; placebo "
                          f"{n_p}; beyond placebo {beyond} (need ≥ {spec['min_beyond_placebo']})",
                "stats": {"count": len(hits), "placebo": n_p, "beyond": beyond,
                          "heads": [head_name(s["layer"], s["head"]) for s in hits]}}
    if kind == "head_set_loss":
        k = eng.M[spec["measure"]]
        h = res["heads"][(spec["contrast"], spec["measure"])]
        hit = h["meaningful"] & (h["mean"] < 0)
        listed = eng.parse_heads(cfg["e4d_heads"])
        count = int(sum(hit[l_, h_] for l_, h_ in listed))
        plc = res["heads"][("placebo", spec["measure"])]["meaningful"]
        n_p = int(sum(plc[l_, h_] for l_, h_ in listed))
        lo, hi = a["late_layers"]
        pool = [(l_, h_) for l_ in range(lo, hi + 1) for h_ in range(hit.shape[1]) if (l_, h_) not in listed]
        null = eng.random_set_counts(hit, pool, len(listed), a["e4d_null"]["sets"], a["e4d_null"]["seed"])
        p95 = float(np.percentile(null, spec["null_quantile"]))
        ok = count >= spec["min_count"] + n_p and count > p95
        per = ", ".join(f"{head_name(l_, h_)} {h['mean'][l_, h_]:+.3f}{'*' if hit[l_, h_] else ''}" for l_, h_ in listed)
        return {"ok": bool(ok),
                "detail": f"{count}/10 listed heads lose {spec['measure']} (meaningful, Δ<0) [{per}]; placebo {n_p}; "
                          f"null {spec['null_quantile']}th pct {p95:.1f} over {len(null)} sets of {len(listed)} "
                          f"from {len(pool)} other late heads (null mean {null.mean():.2f})",
                "stats": {"count": count, "placebo": n_p, "null_p95": p95, "null_mean": float(null.mean()),
                          "null": null.tolist(), "late_hits_total": int(hit[lo:hi + 1].sum()), "k": k}}
    if not ind.any():
        return {"ok": False, "detail": "no induction heads in this base class", "stats": {}}
    ma, mb = arms[spec["contrast"]]
    if kind == "renorm_excess":
        t, g = eng.M[spec["target"]], eng.M[spec["gain"]]
        ex = eng.proportional_excess(mb[..., t], ma[..., t], mb[..., g], ma[..., g])[:, ind].mean(axis=1)
        s = eng.interval(ex, boot)
        loss = (mb[..., t] - ma[..., t])[:, ind].mean()
        gain = (ma[..., g] - mb[..., g])[:, ind].mean()
        return {"ok": s["lo"] > 0,
                "detail": f"excess {s['mean']:+.4f} [{s['lo']:+.4f}, {s['hi']:+.4f}] over {int(ind.sum())} induction "
                          f"heads; mean {spec['target']} loss {loss:+.4f}, {spec['gain']} gain {gain:+.4f}",
                "stats": {**s, "loss": float(loss), "gain": float(gain)}}
    k = eng.M[spec["measure"]]
    x = (ma[..., k] - mb[..., k])[:, ind].mean(axis=1)
    s = eng.interval(x, boot)
    u = eng.unit_ci(_unit_head_diff(ext, res["contrasts"][spec["contrast"]], k, ind), boot)
    unit_txt = f"per phrase {u['mean']:+.4f} [{u['lo']:+.4f}, {u['hi']:+.4f}], {u['n_pos']}/{u['units']} > 0"
    if kind == "noninferior":
        ok = eng.noninferior(s, spec["margin"])
        detail = (f"{s['mean']:+.4f}, one-sided 95% bound {s['lo90']:+.4f} (need > {-spec['margin']:+.3f}); {unit_txt}")
        stats = {**s, "unit": u}
        if spec.get("alongside"):
            ca, cb = arms[spec["alongside"]]
            s2 = eng.interval((ca[..., k] - cb[..., k])[:, ind].mean(axis=1), boot)
            detail += f"; alongside {spec['alongside']}: {s2['mean']:+.4f} [{s2['lo']:+.4f}, {s2['hi']:+.4f}]"
            stats["alongside"] = s2
        return {"ok": ok, "detail": detail, "stats": stats}
    if kind == "positive":
        return {"ok": s["lo"] > 0, "detail": f"{s['mean']:+.4f} [{s['lo']:+.4f}, {s['hi']:+.4f}]; {unit_txt}",
                "stats": {**s, "unit": u}}
    raise ValueError(f"unknown prediction kind {kind!r}")


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
