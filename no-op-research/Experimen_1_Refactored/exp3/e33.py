"""Experiment 3, E3.3: distance sweep (exp3/DESIGN.md section 3).

Phrase P V O ends at 31 - g for every gap g, with the row's own random tokens between the phrase and the query (32).
Main contrast nl - shuffled (shuffled = V P O, O kept last); none gives the total effect; matched vs matched_b at
g = 0 is the placebo. Length control: the row's first 24 tokens (query 24), phrase at 21-23, nl - shuffled within
that length only. Every base row carries every variant once per phrase (the unit); `none` is the same sequence in
each unit slot. Positions are model positions (BOS = 0); id arrays are WITHOUT BOS (array index = position - 1).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from exp2.analysis import head_name
from exp2.design import derive_seed
from exp3 import engine as eng

KINDS = ("argmax", "mass")


@dataclass(frozen=True)
class Layout:
    gaps: tuple[int, ...]
    end: int            # the phrase's last position at g = 0 (q - 1)
    query: int
    placebo_gap: int
    lc_len: int         # length control: tokens kept (its query position)
    lc_phrase: tuple[int, ...]

    def slots(self, gap: int) -> tuple[int, int, int]:
        """Model positions of P V O when the phrase ends at end - gap."""
        return (self.end - 2 - gap, self.end - 1 - gap, self.end - gap)

    @property
    def nl_len(self) -> str:
        return f"nl_len{self.lc_len}"

    @property
    def shuffled_len(self) -> str:
        return f"shuffled_len{self.lc_len}"

    @property
    def variants(self) -> tuple[str, ...]:
        names = ["none"]
        for g in self.gaps:
            names += [f"nl_g{g}", f"shuffled_g{g}"]
        return tuple(names + ["matched", "matched_b", self.nl_len, self.shuffled_len])


def parse_layout(layout: dict[str, Any]) -> Layout:
    gaps = tuple(int(g) for g in layout["gaps"])
    lc = layout["length_control"]
    lay = Layout(gaps, int(layout["phrase_end"]), int(layout["query"]), int(layout["placebo_gap"]), int(lc["seq_len"]),
                 tuple(int(p) for p in lc["phrase"]))
    if lay.end != lay.query - 1:
        raise ValueError(f"layout {layout}: the phrase must end at q - 1 when g = 0")
    if list(gaps) != sorted(set(gaps)) or gaps[0] != 0 or lay.slots(gaps[-1])[0] < 1:
        raise ValueError(f"layout {layout}: gaps must be distinct, ascending, start at 0 and keep the phrase after BOS")
    if lay.placebo_gap not in gaps:
        raise ValueError(f"layout {layout}: the placebo gap must be one of the gaps")
    if lay.lc_phrase != (lay.lc_len - 3, lay.lc_len - 2, lay.lc_len - 1) or not 4 <= lay.lc_len < lay.query:
        raise ValueError(f"layout {layout}: the length-control phrase must end at its own q - 1")
    return lay


def build_e33(base: np.ndarray, phrases: dict[str, Sequence[int]], shuffled: dict[str, Sequence[int]],
              matched: dict[str, Sequence[np.ndarray]], seed: int, layout: dict[str, Any]) -> eng.Design:
    """Build the E3.3 Design; the units are the phrases.
    base: [rows, seq_len] base rows, without BOS. phrases: {name: [P, V, O]}. shuffled: {name: [V, P, O]}.
    matched: {name: three matched pools}. The length-control variants hold base[:, :lc_len] with the phrase written
    at its own q - 3 .. q - 1."""
    lay = parse_layout(layout)
    rows, seq_len = base.shape
    if lay.query != seq_len:
        raise ValueError(f"the query ({lay.query}) must be the final position ({seq_len})")
    units = tuple(phrases)
    stacked = np.repeat(base[:, None, :], len(units), axis=1)
    short = (lay.nl_len, lay.shuffled_len)
    seqs = {v: (stacked[..., :lay.lc_len] if v in short else stacked).copy() for v in lay.variants}
    pidx, lidx = np.asarray(lay.slots(lay.placebo_gap)) - 1, np.asarray(lay.lc_phrase) - 1
    for u, name in enumerate(units):
        pvo = np.asarray(phrases[name], dtype=np.int64)
        vpo = np.asarray(shuffled[name], dtype=np.int64)
        for g in lay.gaps:
            idx = np.asarray(lay.slots(g)) - 1
            seqs[f"nl_g{g}"][:, u, idx] = pvo
            seqs[f"shuffled_g{g}"][:, u, idx] = vpo
        seqs["matched"][:, u, pidx] = eng.draw_per_row(base, matched[name], derive_seed(seed, f"matched/{name}"))
        seqs["matched_b"][:, u, pidx] = eng.draw_per_row(base, matched[name], derive_seed(seed, f"matched_b/{name}"))
        seqs[lay.nl_len][:, u, lidx] = pvo
        seqs[lay.shuffled_len][:, u, lidx] = vpo
    meta = {"none": eng.VariantMeta(())}
    for g in lay.gaps:
        meta[f"nl_g{g}"] = eng.VariantMeta(lay.slots(g))
        meta[f"shuffled_g{g}"] = eng.VariantMeta(lay.slots(g))
    meta["matched"] = eng.VariantMeta(lay.slots(lay.placebo_gap))
    meta["matched_b"] = eng.VariantMeta(lay.slots(lay.placebo_gap))
    meta[lay.nl_len] = eng.VariantMeta(lay.lc_phrase)
    meta[lay.shuffled_len] = eng.VariantMeta(lay.lc_phrase)
    return eng.Design(seqs, stacked, meta, units)


def check_e33(design: eng.Design, phrases: dict[str, Sequence[int]], shuffled: dict[str, Sequence[int]],
              matched: dict[str, Sequence[np.ndarray]], layout: dict[str, Any]) -> None:
    """Raise unless the E3.3 layout holds:
    - shuffled is V P O for every phrase;
    - nl_g and shuffled_g differ only at 29 - g and 30 - g, and each differs from none only at its phrase positions;
    - matched and matched_b differ from each other and from none only at the placebo positions, with matched-pool ids;
    - the length-control variants are the row's first lc_len tokens and differ from each other only at q - 3, q - 2;
    - every variant's query is the row's own token (identical within every contrast)."""
    lay = parse_layout(layout)
    s, base = design.seqs, design.base
    for name in design.units:
        pvo, vpo = list(phrases[name]), list(shuffled[name])
        if vpo != [pvo[1], pvo[0], pvo[2]]:
            raise AssertionError(f"{name}: shuffled {vpo} is not V P O of {pvo}")
    if set(s) != set(lay.variants):
        raise AssertionError(f"variants {sorted(s)} differ from the layout's {sorted(lay.variants)}")
    if not (s["none"] == base).all():
        raise AssertionError("none differs from the base row")
    for g in lay.gaps:
        sl = lay.slots(g)
        eng.check_differ_only_at(design, f"nl_g{g}", f"shuffled_g{g}", sl[:2])
        for v in (f"nl_g{g}", f"shuffled_g{g}"):
            eng.check_differ_only_at(design, v, "none", sl)
    pl = lay.slots(lay.placebo_gap)
    eng.check_differ_only_at(design, "matched", "matched_b", pl)
    for v in ("matched", "matched_b"):
        eng.check_differ_only_at(design, v, "none", pl)
    eng.check_differ_only_at(design, lay.nl_len, lay.shuffled_len, lay.lc_phrase[:2])
    for v, x in s.items():
        n = lay.lc_len if v in (lay.nl_len, lay.shuffled_len) else lay.query
        if x.shape[-1] != n or not (x[..., n - 1] == base[..., n - 1]).all():
            raise AssertionError(f"{v}: expected {n} tokens ending in the row's own query token")
    for u, name in enumerate(design.units):
        pvo, vpo = list(phrases[name]), list(shuffled[name])
        placed = [(f"nl_g{g}", lay.slots(g), pvo) for g in lay.gaps] + \
                 [(f"shuffled_g{g}", lay.slots(g), vpo) for g in lay.gaps] + \
                 [(lay.nl_len, lay.lc_phrase, pvo), (lay.shuffled_len, lay.lc_phrase, vpo)]
        for v, pos, ids in placed:
            if not (s[v][:, u][:, np.asarray(pos) - 1] == np.asarray(ids)).all():
                raise AssertionError(f"{v}, {name}: positions {pos} do not hold {ids}")
        for v in ("matched", "matched_b"):
            for j, p in enumerate(pl):
                if not np.isin(s[v][:, u, p - 1], matched[name][j]).all():
                    raise AssertionError(f"{v}, {name}: position {p} holds an id outside its matched pool")


# ---- analysis ------------------------------------------------------------------------------------------------

def _names(mask: np.ndarray) -> list[str]:
    return [head_name(int(layer), int(head)) for layer, head in zip(*np.nonzero(mask))]


def half_distance(effect: np.ndarray, gaps: Sequence[int], eligible: np.ndarray, ratio: float) -> np.ndarray:
    """effect [G, L, H]: a per-head effect at each gap (gaps ascending, gaps[0] = 0). Returns float [L, H]: the
    smallest gap g > 0 with effect(g) / effect(0) <= ratio (sign-aware); inf when the ratio never falls that far;
    nan where `eligible` is False or effect(0) = 0."""
    g = np.asarray(gaps)
    if g[0] != 0 or (np.diff(g) <= 0).any() or len(g) != effect.shape[0]:
        raise ValueError(f"gaps {list(gaps)} must be ascending from 0, one per effect slice")
    with np.errstate(divide="ignore", invalid="ignore"):
        below = effect[1:] / effect[0] <= ratio
    first = np.where(below.any(axis=0), g[1:][below.argmax(axis=0)], np.inf).astype(np.float64)
    return np.where(eligible & (effect[0] != 0), first, np.nan)


def analyse(ext: eng.Extraction, design: eng.Design, cfg: dict[str, Any], contrasts: dict[str, dict],
            specs: list[dict]) -> dict[str, Any]:
    """Every contrast:
    - per-head tests of each analysed measure;
    - the late pooled per-row verdict and the CI pooled over units;
    - role switches (argmax and mass labels), transition matrices, TV distance.
    Then the sweep table, per-head half-distances, the length-control comparison and each prediction's result."""
    a = cfg["analysis"]
    boot, layers, margin = a["bootstrap"], a["late_layers"], a["margin"]
    mq, mabs = a["meaningful"]["q"], a["meaningful"]["min_abs"]
    sq, shift = a["switch"]["q"], a["switch"]["min_shift"]
    keep = a.get("measures", eng.MEASURES)
    cons = {n: eng.make_contrast(design, n, s["a"], s["b"], s.get("mask_from")) for n, s in contrasts.items()}
    if "placebo" not in cons:
        raise ValueError("the contrasts must include one named 'placebo'")

    pooled, unit_rows, heads, switches, trans, tv, layer_profile, labs, late, unit_late = [], [], {}, {}, {}, {}, {}, {}, {}, {}
    for name, c in cons.items():
        ma, mb = eng.arm_measures(ext, c)
        d = ma - mb
        ua, ub = eng.late_unit_measures(ext, c)
        du = (ua - ub).mean(axis=0)  # [U, M]: per-unit means over rows
        for k, m in enumerate(eng.MEASURES):
            if m not in keep:
                continue
            heads[(name, m)] = eng.head_effects(d[..., k], boot, mq, mabs)
            late[(name, m)] = eng.late_mean(d[..., k], layers)
            unit_late[(name, m)] = du[:, k]
            pooled.append({"contrast": name, "a": c.a, "b": c.b, "measure": m,
                           **eng.verdict(late[(name, m)], boot, margin)})
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
    placebo = {kind: sorted(eng.switching_heads(*labs[("placebo", kind)], sq, shift)) for kind in KINDS}

    res = {"contrasts": cons, "pooled": pooled, "units": unit_rows, "heads": heads, "switches": switches,
           "transitions": trans, "tv": tv, "layer_profile": layer_profile, "placebo_switching": placebo,
           "late": late, "unit_late": unit_late,
           "labels": {f"{n}/{k}/{arm}": labs[(n, k)][j] for (n, k) in labs for j, arm in enumerate("ab")}}
    res["sweep"] = sweep_table(res, cfg)
    res["half_distance"] = half_distances(res, cfg)
    res["length_control"] = length_control(res, cfg)
    res["predictions"] = {s["id"]: evaluate(s, res, cfg) for s in specs}
    return res


def _late_counts(h: dict[str, np.ndarray], layers: Sequence[int]) -> tuple[int, int, int]:
    """(meaningful positive, meaningful negative, total) late heads."""
    lo, hi = int(layers[0]), int(layers[1])
    mean, sig = h["mean"][lo:hi + 1], h["meaningful"][lo:hi + 1]
    return int((sig & (mean > 0)).sum()), int((sig & (mean < 0)).sum()), int(sig.size)


def sweep_table(res: dict[str, Any], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per (effect, gap): the late pooled sink (BOS) effect with its CIs and the late meaningful-head counts."""
    lay = parse_layout(cfg["layout"])
    layers = cfg["analysis"]["late_layers"]
    pooled = {(p["contrast"], p["measure"]): p for p in res["pooled"]}
    out = []
    for eff, prefix in cfg["sweep"].items():
        for g in lay.gaps:
            name = f"{prefix}{g}"
            p = pooled[(name, "BOS")]
            pos, neg, n = _late_counts(res["heads"][(name, "BOS")], layers)
            out.append({"effect": eff, "g": g, "contrast": name, "positions": "-".join(map(str, (lay.slots(g)[0], lay.slots(g)[-1]))),
                        **{k: p[k] for k in ("mean", "lo", "hi", "lo90", "hi90", "wilcoxon_p", "verdict")},
                        "late_meaningful_pos": pos, "late_meaningful_neg": neg, "late_heads": n})
    return out


def half_distances(res: dict[str, Any], cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Per effect (order = the pre-registered output, total = supplementary): per-head half-distance on the row-mean
    effect, for heads whose g = 0 effect is meaningful, and the long-range candidates (half-distance >= the minimum,
    "> max gap" included)."""
    lay = parse_layout(cfg["layout"])
    hd = cfg["half_distance"]
    m = hd["measure"]
    out = {}
    for eff, prefix in cfg["sweep"].items():
        effect = np.stack([res["heads"][(f"{prefix}{g}", m)]["mean"] for g in lay.gaps])
        eligible = res["heads"][(f"{prefix}{lay.gaps[0]}", m)]["meaningful"]
        dist = half_distance(effect, lay.gaps, eligible, hd["ratio"])
        long = np.nan_to_num(dist, nan=-1.0) >= hd["long_range_min"]
        order = sorted(zip(*np.nonzero(long)), key=lambda lh: (-dist[lh], -abs(effect[0][lh])))
        out[eff] = {"effect": effect, "eligible": eligible, "half_distance": dist, "long_range": long,
                    "long_range_heads": [head_name(int(l_), int(h_)) for l_, h_ in order],
                    "eligible_heads": _names(eligible)}
    return out


def length_control(res: dict[str, Any], cfg: dict[str, Any]) -> dict[str, dict[str, float]]:
    """E (late pooled sink, nl - shuffled) in the short rows next to E at the same distance (g = 0) and at the same
    absolute position (g = 8) in the full rows; each E is computed within its own length, and the differences are
    paired per row."""
    lc, boot = cfg["length_control"], cfg["analysis"]["bootstrap"]
    e_len, e_dist, e_pos = (res["late"][(lc[k], "BOS")] for k in ("contrast", "same_distance", "same_position"))
    return {"E_length_control": eng.interval(e_len, boot), "E_same_distance": eng.interval(e_dist, boot),
            "E_same_position": eng.interval(e_pos, boot),
            "length_control_minus_same_distance": eng.interval(e_len - e_dist, boot),
            "length_control_minus_same_position": eng.interval(e_len - e_pos, boot)}


def _ci(s: dict[str, float], lo: str = "lo", hi: str = "hi") -> str:
    return f"[{s[lo]:+.4f}, {s[hi]:+.4f}]"


def evaluate(spec: dict[str, Any], res: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """One prediction in one base class -> {ok, detail, stats}. Every kind works on the late pooled per-row values of
    `measure`; the CI pooled over phrases is reported alongside and does not gate."""
    boot = cfg["analysis"]["bootstrap"]
    m, kind = spec["measure"], spec["kind"]
    late, ul = res["late"], res["unit_late"]
    if kind == "combination_positive":
        terms = [(str(c), float(w)) for c, w in spec["terms"]]
        x = sum(w * late[(c, m)] for c, w in terms)
        xu = sum(w * ul[(c, m)] for c, w in terms)
        label = " ".join(f"{'+' if w >= 0 else '−'} {abs(w):g}·{c}" for c, w in terms).lstrip("+ ")
        parts = {c: eng.interval(late[(c, m)], boot) for c, _ in terms}
        s, u = eng.interval(x, boot), eng.unit_ci(xu, boot)
        ok = s["lo"] > 0
        part_txt = ", ".join(f"{c} {p['mean']:+.4f} {_ci(p)}" for c, p in parts.items())
        detail = (f"{label} = {s['mean']:+.4f} {_ci(s)} (need 95% CI > 0); {part_txt}; per phrase {u['mean']:+.4f} "
                  f"{_ci(u)}, {u['n_pos']}/{u['units']} > 0")
        return {"ok": bool(ok), "detail": detail, "stats": {**s, "unit": u, "parts": parts}}
    c = spec["contrast"]
    s, u = eng.interval(late[(c, m)], boot), eng.unit_ci(ul[(c, m)], boot)
    unit_txt = f"per phrase {u['mean']:+.4f} {_ci(u)}, {u['n_pos']}/{u['units']} > 0"
    if kind == "equivalent":
        ok = eng.equivalent(s, spec["margin"])
        detail = (f"{c} {s['mean']:+.4f}, 90% CI {_ci(s, 'lo90', 'hi90')} (need inside ±{spec['margin']}); 95% CI "
                  f"{_ci(s)}; {unit_txt}")
        return {"ok": bool(ok), "detail": detail, "stats": {**s, "unit": u}}
    if kind == "positive":
        pos, neg, n = _late_counts(res["heads"][(c, m)], cfg["analysis"]["late_layers"])
        detail = (f"{c} {s['mean']:+.4f} {_ci(s)} (need 95% CI > 0); late heads meaningful + / −: {pos} / {neg} of {n}; "
                  f"{unit_txt}")
        return {"ok": bool(s["lo"] > 0), "detail": detail,
                "stats": {**s, "unit": u, "late_meaningful_pos": pos, "late_meaningful_neg": neg}}
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
