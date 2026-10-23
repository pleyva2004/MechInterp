"""Experiment 3, E3.1: query inside vs outside the phrase (exp3/DESIGN.md section 3).

Context at 30-31 x query at 32. The units are the 12 pronoun-verb-object phrases P V O, and every base row carries
every variant once per phrase. r = the base row's own token at that position.

  variant    30  31  32
  PV+O       P   V   O      the query completes the phrase
  rr+O       r   r   O
  PV+Op      P   V   O'     O' = another phrase's object, implausible (fixed per phrase in the config)
  rr+Op      r   r   O'
  PV+M       P   V   M      M = per row and phrase, a same-form word from O's matched pool
  rr+M       r   r   M
  PV+r       P   V   r
  rr+r       r   r   r      the base row
  VP+O       V   P   O
  VP+r       V   P   r
  PVO+r      P V O at 29-31, query r (iteration 3's nl)
  matched    per row, one draw per token from P's, V's and O's matched pools, at 30-32 (placebo)
  matched_b  an independent second draw

Writes land on the query (32) in nine variants. No repeats are declared, so the engine's generic check rejects any
repeated id in a row, the query included. Two-arm contrasts use the engine; the differences of two contrasts
(interaction, completion, plausibility, order x query) are built here, with one variant's masks on all four arms.
Positions are model positions (BOS = 0); id arrays are WITHOUT BOS (array index = position - 1).
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from exp2.analysis import head_name
from exp2.design import derive_seed
from exp3 import engine as eng

VARIANTS = ("PV+O", "rr+O", "PV+Op", "rr+Op", "PV+M", "rr+M", "PV+r", "rr+r", "VP+O", "VP+r", "PVO+r", "matched",
            "matched_b")
QUERIES = ("O", "Op", "M", "r")
KINDS = ("argmax", "mass")


def _layout(layout: dict[str, Any]) -> tuple[list[int], list[int], int]:
    ctx, sanity, q = [int(p) for p in layout["context"]], [int(p) for p in layout["sanity"]], int(layout["query"])
    if ctx != [q - 2, q - 1] or sanity != [q - 3, q - 2, q - 1]:
        raise ValueError(f"layout {layout}: expected context = [q-2, q-1] and sanity = [q-3, q-2, q-1]")
    return ctx, sanity, q


def written_positions(layout: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    """Model positions each variant overwrites (the query included where it is written)."""
    ctx, sanity, q = _layout(layout)
    pvq, pv = tuple(ctx) + (q,), tuple(ctx)
    return {"PV+O": pvq, "rr+O": (q,), "PV+Op": pvq, "rr+Op": (q,), "PV+M": pvq, "rr+M": (q,), "PV+r": pv, "rr+r": (),
            "VP+O": pvq, "VP+r": pv, "PVO+r": tuple(sanity), "matched": pvq, "matched_b": pvq}


def check_oprime(phrases: dict[str, Sequence[int]], oprime: dict[str, str]) -> None:
    """Raise unless every phrase has an O' that is another phrase's object and shares no id with the phrase."""
    if set(oprime) != set(phrases):
        raise ValueError(f"oprime must name one O' per phrase; differs on {sorted(set(oprime) ^ set(phrases))}")
    for name, other in oprime.items():
        if other not in phrases or other == name:
            raise ValueError(f"O' for {name} is {other!r}: it must be another phrase's name")
        if int(phrases[other][2]) in [int(x) for x in phrases[name]]:
            raise ValueError(f"O' for {name} ({phrases[other][2]}) is one of the phrase's own ids")


def build_e31(base: np.ndarray, phrases: dict[str, Sequence[int]], oprime: dict[str, str],
              matched: dict[str, Sequence[np.ndarray]], seed: int, layout: dict[str, Any]) -> eng.Design:
    """Build the 13-variant Design; the units are the phrases.
    base: [rows, seq_len] base rows, without BOS. phrases: {name: [P, V, O]}. oprime: {name: the phrase whose object
    is O'}. matched: {name: matched pools of P, V and O}. M and the placebo ids avoid every id of the row."""
    ctx, sanity, q = _layout(layout)
    rows, seq_len = base.shape
    if q != seq_len:
        raise ValueError(f"the query ({q}) must be the final position ({seq_len})")
    check_oprime(phrases, oprime)
    units = tuple(phrases)
    written = written_positions(layout)
    stacked = np.repeat(base[:, None, :], len(units), axis=1)
    seqs = {v: stacked.copy() for v in VARIANTS}
    for u, name in enumerate(units):
        p, v, o = (int(x) for x in phrases[name])
        op = int(phrases[oprime[name]][2])
        m = eng.draw_per_row(base, [matched[name][2]], derive_seed(seed, f"M/{name}"))[:, 0]
        mat = eng.draw_per_row(base, matched[name], derive_seed(seed, f"matched/{name}"))
        mat_b = eng.draw_per_row(base, matched[name], derive_seed(seed, f"matched_b/{name}"))
        writes = {  # variant: {model position: id, or per-row ids}
            "PV+O": {ctx[0]: p, ctx[1]: v, q: o},
            "rr+O": {q: o},
            "PV+Op": {ctx[0]: p, ctx[1]: v, q: op},
            "rr+Op": {q: op},
            "PV+M": {ctx[0]: p, ctx[1]: v, q: m},
            "rr+M": {q: m},
            "PV+r": {ctx[0]: p, ctx[1]: v},
            "rr+r": {},
            "VP+O": {ctx[0]: v, ctx[1]: p, q: o},
            "VP+r": {ctx[0]: v, ctx[1]: p},
            "PVO+r": {sanity[0]: p, sanity[1]: v, sanity[2]: o},
            "matched": {ctx[0]: mat[:, 0], ctx[1]: mat[:, 1], q: mat[:, 2]},
            "matched_b": {ctx[0]: mat_b[:, 0], ctx[1]: mat_b[:, 1], q: mat_b[:, 2]},
        }
        for var, w in writes.items():
            if tuple(sorted(w)) != written[var]:
                raise AssertionError(f"{var}: writes {sorted(w)} disagree with its declared positions {written[var]}")
            s = seqs[var][:, u]  # a view into seqs[var]
            for pos, ids in w.items():
                s[:, pos - 1] = ids
    meta = {var: eng.VariantMeta(written[var]) for var in VARIANTS}
    return eng.Design(seqs, stacked, meta, units)


def check_e31(design: eng.Design, phrases: dict[str, Sequence[int]], oprime: dict[str, str],
              matched: dict[str, Sequence[np.ndarray]], layout: dict[str, Any]) -> None:
    """Raise unless the E3.1 layout holds:
    - rr+x differs from PV+x only at 30-31 (x in O, O', M, r), VP+O / VP+r from PV+O / PV+r only at 30-31, so the
      query is identical within C(x) and within each order contrast;
    - the PV+x variants differ from each other only at 32, and so do the rr+x variants;
    - PVO+r differs from rr+r only at 29-31; matched and matched_b differ from rr+r only at 30-32;
    - the phrase / O' ids sit where the table says, M and the placebo ids lie in their matched pools;
    - O' is another phrase's object, shares no id with the phrase and lies in O's matched pool (same form and class)."""
    ctx, sanity, q = _layout(layout)
    check_oprime(phrases, oprime)
    for x in QUERIES:
        eng.check_differ_only_at(design, f"PV+{x}", f"rr+{x}", ctx)
    for x in ("O", "r"):
        eng.check_differ_only_at(design, f"PV+{x}", f"VP+{x}", ctx)
    for x in QUERIES[1:]:
        eng.check_differ_only_at(design, "PV+O", f"PV+{x}", [q])
        eng.check_differ_only_at(design, "rr+O", f"rr+{x}", [q])
    eng.check_differ_only_at(design, "rr+r", "PVO+r", sanity)
    for v in ("matched", "matched_b"):
        eng.check_differ_only_at(design, "rr+r", v, ctx + [q])
    s = design.seqs
    for u, name in enumerate(design.units):
        p, v, o = (int(x) for x in phrases[name])
        op = int(phrases[oprime[name]][2])
        if op not in set(np.asarray(matched[name][2]).tolist()):
            raise AssertionError(f"O' for {name} ({op}) is not in O's matched pool: form or class differ from O ({o})")
        expect = {"PV+O": (ctx + [q], [p, v, o]), "PV+Op": (ctx + [q], [p, v, op]), "PV+M": (ctx, [p, v]),
                  "VP+O": (ctx + [q], [v, p, o]), "PV+r": (ctx, [p, v]), "VP+r": (ctx, [v, p]),
                  "PVO+r": (sanity, [p, v, o])}
        for var, (pos, ids) in expect.items():
            if not (s[var][:, u][:, np.asarray(pos) - 1] == np.asarray(ids)).all():
                raise AssertionError(f"{var}, {name}: positions {pos} do not hold {ids}")
        if not np.isin(s["PV+M"][:, u, q - 1], matched[name][2]).all():
            raise AssertionError(f"PV+M, {name}: M outside O's matched pool")
        for var in ("matched", "matched_b"):
            for j, pos in enumerate(ctx + [q]):
                if not np.isin(s[var][:, u, pos - 1], matched[name][j]).all():
                    raise AssertionError(f"{var}, {name}: position {pos} holds an id outside its matched pool")


# ---- analysis ------------------------------------------------------------------------------------------------

def _names(heads) -> list[str]:
    return [head_name(int(layer), int(head)) for layer, head in heads]


def present_measures(c: eng.Contrast) -> tuple[str, ...]:
    """The measures with a non-empty mask under c (Phrase when c has phrase positions). Empty categories are 0 in both
    arms by construction, so they are not tested."""
    out = tuple(m for i, m in enumerate(eng.CATEGORIES) if (c.codes == i).any())
    return out + (("Phrase",) if c.phrase else ())


def difference_pair(design: eng.Design, name: str, spec: dict[str, Any],
                    contrasts: dict[str, dict]) -> tuple[eng.Contrast, eng.Contrast]:
    """The two two-arm contrasts of a difference, both with the masks of spec['mask_from'] (default: plus's a)."""
    plus, minus = contrasts[spec["plus"]], contrasts[spec["minus"]]
    src = spec.get("mask_from", plus["a"])
    return (eng.make_contrast(design, f"{name}:plus", plus["a"], plus["b"], src),
            eng.make_contrast(design, f"{name}:minus", minus["a"], minus["b"], src))


def _logit(p: np.ndarray, eps: float) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def analyse(ext: eng.Extraction, design: eng.Design, cfg: dict[str, Any], contrasts: dict[str, dict],
            differences: dict[str, dict], specs: list[dict]) -> dict[str, Any]:
    """Every two-arm contrast and every difference:
    - per-head tests of each non-empty measure;
    - the late pooled per-row verdict and the CI pooled over units;
    - layer profiles.
    Two-arm contrasts also get role switches (argmax and mass labels), transition matrices and TV distance.
    Differences also get the logit(BOS) check. Then the placebo switch counts and each prediction's per-class result."""
    a = cfg["analysis"]
    boot, layers, margin = a["bootstrap"], a["late_layers"], a["margin"]
    mq, mabs = a["meaningful"]["q"], a["meaningful"]["min_abs"]
    sq, shift = a["switch"]["q"], a["switch"]["min_shift"]
    cons = {n: eng.make_contrast(design, n, s["a"], s["b"], s.get("mask_from")) for n, s in contrasts.items()}
    if "placebo" not in cons:
        raise ValueError("the contrasts must include one named 'placebo'")
    if set(cons) & set(differences):
        raise ValueError(f"contrast and difference names clash: {sorted(set(cons) & set(differences))}")
    diffs = {n: difference_pair(design, n, s, contrasts) for n, s in differences.items()}

    deltas, keep, switches, trans, tv, labs, logit = {}, {}, {}, {}, {}, {}, []
    for name, c in cons.items():
        ma, mb = eng.arm_measures(ext, c)
        ua, ub = eng.late_unit_measures(ext, c)
        deltas[name] = (ma - mb, (ua - ub).mean(axis=0))  # [rows, L, H, M], [U, M]
        keep[name] = present_measures(c)
        la = eng.labels(ext.att_row[:, ext.index(c.a)], c.codes, cfg["diffuse_threshold"])
        lb = eng.labels(ext.att_row[:, ext.index(c.b)], c.codes, cfg["diffuse_threshold"])
        for i, kind in enumerate(KINDS):
            labs[(name, kind)] = (la[i], lb[i])
            switches[(name, kind)] = eng.role_switches(la[i], lb[i], sq, shift)
            trans[(name, kind)] = eng.transition_matrix(switches[(name, kind)])
        n_cat = len(eng.CATEGORIES)
        tv[name] = eng.tv_distance(ma[..., :n_cat], mb[..., :n_cat])
    k_bos = eng.M["BOS"]
    for name, (cp, cm) in diffs.items():
        ma1, mb1 = eng.arm_measures(ext, cp)
        ma2, mb2 = eng.arm_measures(ext, cm)
        ua1, ub1 = eng.late_unit_measures(ext, cp)
        ua2, ub2 = eng.late_unit_measures(ext, cm)
        deltas[name] = ((ma1 - mb1) - (ma2 - mb2), ((ua1 - ub1) - (ua2 - ub2)).mean(axis=0))
        keep[name] = present_measures(cp)
        lg = [_logit(x[..., k_bos], a["logit_eps"]) for x in (ma1, mb1, ma2, mb2)]
        dl = eng.late_mean((lg[0] - lg[1]) - (lg[2] - lg[3]), layers)
        logit.append({"contrast": name, "measure": "logit(BOS)", **eng.verdict(dl, boot, margin)})

    pooled, unit_rows, heads, layer_profile = [], [], {}, {}
    for name, (d, du) in deltas.items():
        arms = (f"{cons[name].a} − {cons[name].b}" if name in cons else
                f"({diffs[name][0].a} − {diffs[name][0].b}) − ({diffs[name][1].a} − {diffs[name][1].b})")
        for k, m in enumerate(eng.MEASURES):
            if m not in keep[name]:
                continue
            heads[(name, m)] = eng.head_effects(d[..., k], boot, mq, mabs)
            pooled.append({"contrast": name, "type": "two-arm" if name in cons else "difference", "arms": arms,
                           "measure": m, **eng.verdict(eng.late_mean(d[..., k], layers), boot, margin)})
            u = eng.unit_ci(du[:, k], boot)
            unit_rows.append({"contrast": name, "measure": m, **{f"unit_{key}": val for key, val in u.items()}})
            layer_profile[(name, m)] = d[..., k].mean(axis=(0, 2))
    placebo = {kind: sorted(eng.switching_heads(*labs[("placebo", kind)], sq, shift)) for kind in KINDS}
    roles = {(name, kind, arm): eng.head_roles(labs[(name, kind)][j])
             for (name, kind) in labs for j, arm in enumerate(("a", "b"))}

    res = {"contrasts": cons, "differences": diffs, "pooled": pooled, "units": unit_rows, "heads": heads,
           "switches": switches, "transitions": trans, "tv": tv, "layer_profile": layer_profile,
           "placebo_switching": placebo, "roles": roles, "logit": logit,
           "labels": {f"{n}/{k}/{arm}": labs[(n, k)][j] for (n, k) in labs for j, arm in enumerate("ab")}}
    res["predictions"] = {s["id"]: evaluate(s, res, deltas, cfg) for s in specs}
    return res


def evaluate(spec: dict[str, Any], res: dict[str, Any], deltas: dict[str, tuple], cfg: dict[str, Any]) -> dict[str, Any]:
    """One prediction in one base class -> {ok, detail, stats}."""
    a = cfg["analysis"]
    boot, layers = a["bootstrap"], a["late_layers"]
    kind = spec["kind"]
    if kind == "switch_count":
        lo, hi = spec["layers"]
        dest = spec["to"] if isinstance(spec["to"], list) else [spec["to"]]
        sw = res["switches"][(spec["contrast"], spec["labels"])]
        hits = [s for s in sw if s["from"] == spec["from"] and s["to"] in dest and lo <= s["layer"] <= hi]
        distinct = sorted({(s["layer"], s["head"]) for s in hits})
        n_p = len(res["placebo_switching"][spec["labels"]])
        beyond = len(distinct) - n_p
        names = ", ".join(f"{head_name(s['layer'], s['head'])} →{s['to']} ({s['shift']:+.2f})" for s in hits) or "none"
        return {"ok": beyond >= spec["min_beyond_placebo"],
                "detail": f"{len(distinct)} heads in L{lo}-{hi} switch {spec['from']}→{' or '.join(dest)} [{names}]; "
                          f"placebo {n_p}; beyond placebo {beyond} (need ≥ {spec['min_beyond_placebo']})",
                "stats": {"count": len(distinct), "placebo": n_p, "beyond": beyond, "heads": _names(distinct)}}
    if kind not in ("sign", "unit_sign"):
        raise ValueError(f"unknown prediction kind {kind!r}")
    if spec["sign"] not in ("positive", "negative"):
        raise ValueError(f"sign must be 'positive' or 'negative', not {spec['sign']!r}")
    k = eng.M[spec["measure"]]
    d, du = deltas[spec["contrast"]]
    s = eng.interval(eng.late_mean(d[..., k], layers), boot)
    u = eng.unit_ci(du[:, k], boot)
    pos = spec["sign"] == "positive"
    row_ok = s["lo"] > 0 if pos else s["hi"] < 0
    unit_ok = u["lo"] > 0 if pos else u["hi"] < 0
    row_txt = f"per row {s['mean']:+.4f} [{s['lo']:+.4f}, {s['hi']:+.4f}]"
    unit_txt = (f"per phrase {u['mean']:+.4f} [{u['lo']:+.4f}, {u['hi']:+.4f}], "
                f"{u['n_pos'] if pos else u['n_neg']}/{u['units']} {'> 0' if pos else '< 0'}")
    need = "> 0" if pos else "< 0"
    if kind == "sign":
        return {"ok": bool(row_ok), "detail": f"{row_txt} (need 95% CI {need}); {unit_txt}", "stats": {**s, "unit": u}}
    return {"ok": bool(unit_ok), "detail": f"{unit_txt} (need 95% CI over phrases {need}); {row_txt}",
            "stats": {**s, "unit": u}}


def check_predictions(specs: list[dict], results: dict[str, dict]) -> list[dict]:
    """PASS only if the prediction holds in every base class."""
    out = []
    for s in specs:
        parts = [(res["predictions"][s["id"]]["ok"], f"{base}: {res['predictions'][s['id']]['detail']}")
                 for base, res in results.items()]
        ok = bool(parts) and all(p for p, _ in parts)
        out.append({"id": s["id"], "note": s.get("note", ""), "verdict": "PASS" if ok else "FAIL",
                    "detail": "; ".join(d for _, d in parts),
                    "per_base": {base: bool(res["predictions"][s["id"]]["ok"]) for base, res in results.items()}})
    return out


# ---- query log P (manipulation; reported, not gating) ---------------------------------------------------------

CONTEXTS = {"PV": "PV+O", "VP": "VP+O", "rr": "rr+O"}  # one variant per context; its prefix 0..31 is the context's


def query_candidates(design: eng.Design, n_rows: int) -> np.ndarray:
    """int64 [n_rows, U, len(QUERIES)]: the ids each row and phrase puts at the query as O, O', M and r."""
    return np.stack([design.seqs[f"PV+{x}"][:n_rows, :, -1] for x in QUERIES], axis=-1)


def summarize_query_logp(logp: dict[str, np.ndarray], units: Sequence[str]) -> dict[str, Any]:
    """logp[context] float [rows, U, len(QUERIES)]: log P of each query id at 32 given that context's prefix.
    Returns the mean per context x query, how often O beats O' / M / r under PV, the context lift
    (PV - rr) per query, and the per-phrase PV log P of O minus O'."""
    qi = {x: i for i, x in enumerate(QUERIES)}
    pv = logp["PV"]
    return {
        "mean": {ctx: {x: float(lp[..., i].mean()) for x, i in qi.items()} for ctx, lp in logp.items()},
        "pv_O_beats": {x: float((pv[..., qi["O"]] > pv[..., qi[x]]).mean()) for x in QUERIES[1:]},
        "lift_pv_minus_rr": {x: float((pv[..., i] - logp["rr"][..., i]).mean()) for x, i in qi.items()},
        "per_phrase_O_minus_Op": {u: float((pv[:, j, qi["O"]] - pv[:, j, qi["Op"]]).mean()) for j, u in enumerate(units)},
        "rows": int(pv.shape[0]),
    }
