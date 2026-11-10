"""TDA analysis 4: independent verification of the saved run results/20261025-113614_exp4tda4, plus an addendum report.

Run:  uv run --offline python -m exp4.tda4_finalize

The saved run is only read, never modified. Output goes to a new results/<timestamp>_exp4tda4/ folder: addendum.md,
verify.json, verify_arrays.npz and versions.json.

What is independent of exp4/tda4_graph.py:
- forward passes use the HuggingFace GPT-2 (eager attention), not TransformerLens, and the sequences are built here;
- H0 comes from a minimum spanning tree (scipy), H1 from ripser over Z/11, and W1 from an explicit assignment problem;
- the paired label-swap null, the split halves and the stability bootstrap are rebuilt with a different seed;
- a matching-pair cycle (MPC) is found by locating each persistent H1 class's birth edge from its birth value. Because
  hop-shortest paths are not unique, it is scored two ways: E1 on *every* shortest representative (strict) or on
  *some* (lenient). The run's single BFS representative must lie between the two.
Only the stimuli, the base rows and the wrong-capital derangement come from the shared loaders, because they define
the design.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from ripser import ripser
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import minimum_spanning_tree

from exp1.runlog import make_run_dir, save_versions
from exp2.design import cached_base_block
from exp3.run_positions import latin_derangements
from exp3.run_relations import stimuli

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN = PROJECT_ROOT / "results" / "20261025-113614_exp4tda4"
SMOKE_VERSIONS = Path("/private/tmp/claude-501/-Users-pabloleyva/cf17c3a1-b0ec-440e-916a-f5ea33bfe8a3/scratchpad/"
                      "tda4_smoke/20261025-113548_exp4tda4/versions.json")
N_LAYERS, T = 12, 32
EXTRACT_TAUS = (31, 24, 16, 8, 1)  # τ 30 and 28 have no own MPC in any layer mean, so cannot hold a G2 cell
SEED = 20261025
CONDS = ("control", "own", "wrong", "random")
PAIRS = (("own", "wrong"), ("own", "random"), ("own", "control"))
VARIANTS = (("noBOS", False), ("withBOS", True))
ROLE_ABBR = {"merge": "M", "cycle": "C", "instant": "I"}


def log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------- persistence, written without gudhi

def dmat(a: np.ndarray, bos: bool) -> np.ndarray:
    """Causal graph A[q, k] -> distances 1 − w, w(i, j) = A[i, j] + A[j, i] (one of the two is 0 under the mask)."""
    low = np.tril(np.asarray(a, dtype=np.float64), -1)
    d = 1.0 - (low + low.T)
    np.fill_diagonal(d, 0.0)
    return d if bos else d[1:, 1:]


def mst(d: np.ndarray) -> np.ndarray:
    """Minimum spanning tree as a dense matrix of 1 + edge length (0 = not a tree edge). scipy reads a zero entry as a
    missing edge and some edges have w = 1 (distance 0), so the tree is built on d + 1, which has the same tree."""
    shifted = d + 1.0
    np.fill_diagonal(shifted, 0.0)
    return minimum_spanning_tree(shifted).toarray()


def h0(d: np.ndarray) -> np.ndarray:
    """Finite H0 bars of a complete weighted graph: born at 0, dying at the minimum-spanning-tree edge lengths (zero-length
    bars dropped, as gudhi does)."""
    m = mst(d)
    deaths = np.sort(m[m > 0] - 1.0)
    deaths = deaths[deaths > 0]
    return np.column_stack([np.zeros_like(deaths), deaths])


def h1(d: np.ndarray) -> np.ndarray:
    return ripser(d.astype(np.float32), distance_matrix=True, maxdim=1, coeff=11)["dgms"][1].astype(np.float64)


def w1(a: np.ndarray, b: np.ndarray) -> float:
    """1-Wasserstein distance between diagrams, L-inf ground metric, points may go to the diagonal."""
    n, m = len(a), len(b)
    if n + m == 0:
        return 0.0
    c = np.full((n + m, n + m), 1e9)
    if n and m:
        c[:n, :m] = np.maximum(np.abs(a[:, None, 0] - b[None, :, 0]), np.abs(a[:, None, 1] - b[None, :, 1]))
    c[np.arange(n), m + np.arange(n)] = (a[:, 1] - a[:, 0]) / 2
    c[n + np.arange(m), np.arange(m)] = (b[:, 1] - b[:, 0]) / 2
    c[n:, m:] = 0.0
    r, k = linear_sum_assignment(c)
    return float(c[r, k].sum())


def diag(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return h0(d), h1(d)


def dist_d(x: tuple, y: tuple) -> float:
    return w1(x[0], y[0]) + w1(x[1], y[1])


def bfs_counts(adj: np.ndarray, s: int) -> tuple[np.ndarray, np.ndarray]:
    """Hop distance from s and the number of hop-shortest paths from s to each node."""
    dist = np.full(len(adj), -1)
    cnt = np.zeros(len(adj))
    dist[s], cnt[s] = 0, 1.0
    frontier = [s]
    while frontier:
        nxt = []
        for a in frontier:
            for b in np.flatnonzero(adj[a]):
                if dist[b] == -1:
                    dist[b] = dist[a] + 1
                    nxt.append(int(b))
                if dist[b] == dist[a] + 1:
                    cnt[b] += cnt[a]
        frontier = nxt
    return dist, cnt


def analyse(d: np.ndarray, tau: int, bos: bool, t: dict) -> dict:
    """Role of E1 = {32, τ}, number of persistent H1 classes and whether one of them is a matching-pair cycle with E1
    on every (strict) or some (lenient) hop-shortest representative."""
    off = 0 if bos else 1
    x, y = tau - off, T - off
    d32 = d.astype(np.float32)
    dg = ripser(d32, distance_matrix=True, maxdim=1, coeff=11)["dgms"][1]
    tree = mst(d)
    if tree[x, y] > 0 or tree[y, x] > 0:
        role = "merge"
    elif any(np.float32(b) == d32[x, y] for b, _ in dg):
        role = "cycle"
    else:
        role = "instant"
    upper = np.triu(np.ones(d32.shape, dtype=bool), 1)
    n_pers, amb = 0, 0
    best = {"some": None, "all": None}
    for b, dd in dg:
        b, dd = float(b), float(dd)
        if not (dd - b >= t["persist_min"] and (1.0 - dd) <= t["persist_ratio"] * (1.0 - b)):
            continue
        n_pers += 1
        adj = d32 < np.float32(b)
        np.fill_diagonal(adj, False)
        flags = []
        for u, v in np.argwhere(upper & (d32 == np.float32(b))):
            du, cu = bfs_counts(adj, int(u))
            if du[v] < 0:
                continue  # endpoints not yet connected, so this edge cannot birth an H1 class
            if {int(u), int(v)} == {x, y}:
                flags.append((True, True))
                continue
            dv, cv = bfs_counts(adj, int(v))
            thru = sum(cu[p] * cv[q] for p, q in ((x, y), (y, x))
                       if adj[p, q] and du[p] >= 0 and dv[q] >= 0 and du[p] + 1 + dv[q] == du[v])
            flags.append((thru > 0, thru == cu[v]))
        if len(flags) != 1:
            amb += 1
        if not flags:
            continue
        some, every = any(f[0] for f in flags), all(f[1] for f in flags)
        for key, ok in (("some", some), ("all", every)):
            if ok and (best[key] is None or dd - b > best[key]):
                best[key] = dd - b
    return {"role": role, "n_pers": n_pers, "some": best["some"], "all": best["all"], "ambiguous": amb}


# ---------------------------------------------------------------- extraction with the HuggingFace model

def build(rows: np.ndarray, countries: np.ndarray, finals: np.ndarray, tau: int, bos: int) -> np.ndarray:
    """[R, 32] rows -> [R, 11, 33]: BOS at 0, the row at 1..32, country c at position τ, finals[c] at 32."""
    R, n = len(rows), len(countries)
    s = np.repeat(rows[:, None, :], n, axis=1)
    s[:, :, tau - 1] = countries[None, :]
    s[:, :, T - 1] = finals[None, :]
    out = np.concatenate([np.full((R, n, 1), bos, dtype=np.int64), s.astype(np.int64)], axis=2)
    if not ((out[:, :, tau] == countries).all() and (out[:, :, T] == finals).all() and (out[:, :, 0] == bos).all()):
        raise AssertionError("ids at τ, 32 or 0 are wrong")
    keep = np.ones(T, dtype=bool)
    keep[[tau - 1, T - 1]] = False
    if not (out[:, :, 1:][:, :, keep] == rows[:, None, keep]).all():
        raise AssertionError("a sequence differs from its row outside τ and 32")
    srt = np.sort(out[:, :, 1:].reshape(-1, T), axis=1)
    if (srt[:, 1:] == srt[:, :-1]).any() or (out[:, :, 1:] == bos).any():
        raise AssertionError("a sequence repeats a token or contains BOS after position 0")
    return out


@torch.inference_mode()
def hf_row_graphs(model, seqs: np.ndarray, k: int = 25) -> tuple[np.ndarray, float, float]:
    """[R, n, 33] -> per-row layer-mean graphs [R, 12, 33, 33] (mean over the row's n sequences), max |row sum − 1|,
    max upper-triangle weight."""
    R, n, K = seqs.shape
    out = np.empty((R, N_LAYERS, K, K), dtype=np.float32)
    dev = up = 0.0
    for s in range(0, R, k):
        ids = torch.as_tensor(seqs[s:s + k].reshape(-1, K))
        att = torch.stack(model(input_ids=ids, output_attentions=True).attentions, dim=1)  # [b, L, H, q, k]
        dev = max(dev, float((att.sum(-1) - 1).abs().max()))
        up = max(up, float(torch.triu(att, 1).max()))
        out[s:s + k] = att.mean(dim=2).reshape(-1, n, N_LAYERS, K, K).mean(dim=1).numpy()
    return out, dev, up


# ---------------------------------------------------------------- report helpers

def provenance() -> dict:
    start = datetime.strptime(RUN.name[:15], "%Y%m%d-%H%M%S").timestamp()
    me = Path(__file__).resolve()
    later = [f"{cd}/{p.name}" for cd in ("exp1", "exp1ext", "exp2", "exp3", "exp4")
             for p in sorted((PROJECT_ROOT / cd).glob("*.py")) if p.stat().st_mtime > start and p.resolve() != me]
    tda4 = PROJECT_ROOT / "exp4" / "tda4_graph.py"
    run_hash = json.loads((RUN / "versions.json").read_text())["code_sha256"]
    smoke = json.loads(SMOKE_VERSIONS.read_text())["code_sha256"] if SMOKE_VERSIONS.exists() else None
    return {"run_start": datetime.fromtimestamp(start).strftime("%H:%M:%S"),
            "tda4_graph_mtime": datetime.fromtimestamp(tda4.stat().st_mtime).strftime("%H:%M:%S"),
            "tda4_graph_sha256": hashlib.sha256(tda4.read_bytes()).hexdigest(),
            "modified_after_start": later, "run_code_sha256": run_hash, "smoke2_code_sha256": smoke,
            "prereg_copy_identical": (RUN / "PREDICTIONS.md").read_bytes() == (PROJECT_ROOT / "exp4" / "PREDICTIONS_tda4.md").read_bytes(),
            "config_copy_identical": (RUN / "config.yaml").read_bytes() == (PROJECT_ROOT / "configs" / "exp4_tda4.yaml").read_bytes()}


def pval(null: np.ndarray, obs: float) -> float:
    return (1 + int((null >= obs).sum())) / (1 + len(null))


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    ap.add_argument("--rows", type=int, default=None, help="first rows only (smoke test)")
    ap.add_argument("--iters", type=int, default=None, help="override swaps / splits / resamples (smoke test)")
    args = ap.parse_args(argv)
    t0 = time.time()
    cfg = yaml.safe_load((RUN / "base_config.yaml").read_text()) | yaml.safe_load((RUN / "config.yaml").read_text())
    t = cfg["tda"]
    n_swap, n_split, n_boot = (args.iters,) * 3 if args.iters else (t["n_swap"], t["n_split"], t["n_boot"])
    out_dir = make_run_dir(args.results_dir, tag="exp4tda4")
    save_versions(out_dir, PROJECT_ROOT, code_dirs=("exp1", "exp2", "exp3", "exp4"))
    log(f"output folder {out_dir}", t0)

    G = np.load(RUN / "graphs.npz")
    A = np.load(RUN / "tda_arrays.npz")
    J = json.loads((RUN / "tda4.json").read_text())
    preds = {p["id"]: p for p in json.loads((RUN / "predictions.json").read_text())}
    checks = json.loads((RUN / "checks.json").read_text())
    units = [str(u) for u in G["units"]]
    taus = [int(x) for x in G["taus"]]
    fillers = list(cfg["bases"])
    U = len(units)

    # ---- 1. coverage of the saved run
    coverage = {"graphs_keys": sorted(k for k in G.files if "/" in k),
                "graph_shape": list(G[f"{fillers[0]}/own"].shape), "taus": taus, "units": units,
                "null_shape": list(A[f"{fillers[0]}/t31/null"].shape), "split_shape": list(A[f"{fillers[0]}/t31/split"].shape),
                "cells_with_arrays": sum(f"{b}/t{tau}/obs" in A.files for b in fillers for tau in taus),
                "timing_s": J["timing_s"]}
    prov = provenance()
    log("provenance and coverage read", t0)

    # ---- 2. re-derive every saved distance from graphs.npz with this file's persistence code
    obs_dev, obs_rel, tr_dev = 0.0, 0.0, 0.0
    full = {}  # (b, tau, cond, unit, variant) -> analyse()
    for b in fillers:
        for ti, tau in enumerate(taus):
            g = {c: G[f"{b}/{c}"][ti] for c in CONDS}
            dg = {}
            for c in CONDS:
                for u in range(U):
                    for v, bos in VARIANTS:
                        d = dmat(g[c][u], bos)
                        dg[(c, u, v)] = diag(d)
                        full[(b, tau, c, u, v)] = analyse(d, tau, bos, t)
            obs = A[f"{b}/t{tau}/obs"]
            for pi, (x, y) in enumerate(PAIRS):
                for u in range(U):
                    for vi, (v, _) in enumerate(VARIANTS):
                        dd = abs(dist_d(dg[(x, u, v)], dg[(y, u, v)]) - obs[pi, u, vi, 0])
                        obs_dev = max(obs_dev, dd)
                        obs_rel = max(obs_rel, dd / max(obs[pi, u, vi, 0], 1e-9))
            own_t = g["own"].astype(np.float64).copy()
            own_t[:, T, tau] = g["wrong"][:, T, tau]
            for u in range(U):
                for vi, (v, bos) in enumerate(VARIANTS):
                    tr_dev = max(tr_dev, abs(dist_d(diag(dmat(own_t[u], bos)), dg[("wrong", u, v)]) - A[f"{b}/t{tau}/transplant"][u, vi]))
    log(f"saved distances re-derived: max |Δ| obs {obs_dev:.1e} (relative {obs_rel:.1e}), transplant {tr_dev:.1e}", t0)

    # Full-sample graph summaries vs tda4.json: role, persistent-class count, MPC between strict and lenient.
    role_mis, npers_mis, mpc_bad, amb_total, mpc_run_n, mpc_run_strict = [], [], [], 0, 0, 0
    for (b, tau, c, u, v), s in full.items():
        r = J["cells"][f"{b}/t{tau}"]["graphs"][f"{c}/{units[u]}/{v}"]
        amb_total += s["ambiguous"]
        if r["role"] != s["role"]:
            role_mis.append(f"{b} τ{tau} {c} {units[u]} {v}: run {r['role']}, here {s['role']}")
        if r["n_persistent"] != s["n_pers"]:
            npers_mis.append(f"{b} τ{tau} {c} {units[u]} {v}: run {r['n_persistent']}, here {s['n_pers']}")
        has = r["mpc"] is not None
        mpc_run_n += has
        mpc_run_strict += has and s["all"] is not None
        if (has and s["some"] is None) or (not has and s["all"] is not None):
            mpc_bad.append(f"{b} τ{tau} {c} {units[u]} {v}")
    b1_found = [f"{b} τ{tau} {c} L{u}" for (b, tau, c, u, v), s in full.items()
                if v == "withBOS" and 1 <= u < N_LAYERS and s["some"] is not None]
    no_mpc_30_28 = all(full[(b, tau, "own", u, "noBOS")]["some"] is None for b in fillers for tau in (30, 28) for u in range(N_LAYERS))
    log(f"full-sample graphs: {len(role_mis)} role, {len(npers_mis)} count, {len(mpc_bad)} MPC disagreements", t0)

    # ---- 3. independent extraction (HF GPT-2) of own and wrong at EXTRACT_TAUS
    from transformers import AutoTokenizer, GPT2Model

    tok = AutoTokenizer.from_pretrained(cfg["figures"]["tokenizer"])
    countries, capitals, cities = stimuli(cfg, tok)
    stim = np.concatenate([countries, capitals, cities, cfg["demonstration_ids"]])
    wrong = latin_derangements(len(countries), 1, cfg["wrong_capital_seed"])[0]
    if (wrong == np.arange(len(countries))).any():
        raise AssertionError("derangement has a fixed point")
    mapping = ", ".join(f"{tok.decode([int(a)]).strip()}→{tok.decode([int(k)]).strip()}" for a, k in zip(countries, capitals[wrong]))
    model = GPT2Model.from_pretrained(cfg["backends"]["huggingface"]["model_name"], attn_implementation="eager").eval()
    rows_all, H, hf_dev, hf_up, prefix, hf_vs_saved = {}, {}, 0.0, 0.0, 0.0, 0.0
    for b, rng_ in cfg["bases"].items():
        block = cached_base_block(f"exp3rel_{b}", rng_, stim, cfg["base_cache_n"], T, cfg["seed"], cfg["bos_token_id"],
                                  PROJECT_ROOT / cfg["paths"]["sequences_dir"])
        rows = block[: (args.rows or cfg["n_rows"])]
        if np.isin(rows, stim).any():
            raise AssertionError("stimulus id in a base row")
        rows_all[b] = rows
        for tau in EXTRACT_TAUS:
            for c, fin in (("own", capitals), ("wrong", capitals[wrong])):
                g, dv, up = hf_row_graphs(model, build(rows, countries, fin, tau, cfg["bos_token_id"]))
                H[(b, tau, c)] = g
                hf_dev, hf_up = max(hf_dev, dv), max(hf_up, up)
            prefix = max(prefix, float(np.abs(H[(b, tau, "own")][:, :, :T] - H[(b, tau, "wrong")][:, :, :T]).max()))
            if not args.rows:
                ti = taus.index(tau)
                for c in ("own", "wrong"):
                    hf_vs_saved = max(hf_vs_saved, float(np.abs(H[(b, tau, c)].mean(axis=0, dtype=np.float64)
                                                                - G[f"{b}/{c}"][ti][:N_LAYERS]).max()))
        log(f"{b}: HF graphs for τ {EXTRACT_TAUS}", t0)
    R = len(next(iter(rows_all.values())))

    # ---- 4. own swap null, split halves, transplant and bootstrap on the HF graphs
    rng = np.random.default_rng(SEED)
    res = {}
    for b in fillers:
        for tau in EXTRACT_TAUS:
            own, wro = H[(b, tau, "own")], H[(b, tau, "wrong")]
            mo, mw = own.mean(axis=0, dtype=np.float64), wro.mean(axis=0, dtype=np.float64)
            dgo = [diag(dmat(mo[u], False)) for u in range(N_LAYERS)]
            dgw = [diag(dmat(mw[u], False)) for u in range(N_LAYERS)]
            obs = np.array([dist_d(dgo[u], dgw[u]) for u in range(N_LAYERS)])
            mt = mo.copy()
            mt[:, T, tau] = mw[:, T, tau]
            transplant = np.array([dist_d(diag(dmat(mt[u], False)), dgw[u]) for u in range(N_LAYERS)])
            diff = own.astype(np.float64) - wro
            null = np.empty((n_swap, N_LAYERS))
            for s in range(n_swap):
                delta = diff[rng.random(R) < 0.5].sum(axis=0) / R
                xo, xw = mo - delta, mw + delta
                for u in range(N_LAYERS):
                    null[s, u] = dist_d(diag(dmat(xo[u], False)), diag(dmat(xw[u], False)))
            split = np.empty((n_split, N_LAYERS)) if tau == t["g1_tau"] else None
            if split is not None:
                for s in range(n_split):
                    perm = rng.permutation(R)
                    ga, gb = own[perm[:R // 2]].mean(axis=0, dtype=np.float64), own[perm[R // 2:]].mean(axis=0, dtype=np.float64)
                    for u in range(N_LAYERS):
                        split[s, u] = dist_d(diag(dmat(ga[u], False)), diag(dmat(gb[u], False)))
            fs = {c: [analyse(dmat(m[u], False), tau, False, t) for u in range(N_LAYERS)] for c, m in (("own", mo), ("wrong", mw))}
            boot = {(c, k): np.zeros(N_LAYERS) for c in ("own", "wrong") for k in ("some", "all")}
            for _ in range(n_boot):
                idx = rng.integers(0, R, size=R)
                for c, x in (("own", own), ("wrong", wro)):
                    gm = x[idx].mean(axis=0, dtype=np.float64)
                    for u in range(N_LAYERS):
                        s_ = analyse(dmat(gm[u], False), tau, False, t)
                        for k in ("some", "all"):
                            boot[(c, k)][u] += s_[k] is not None
            boot = {k: v / n_boot for k, v in boot.items()}
            e1 = diff[:, :, T, tau]  # [R, 12] own − wrong capital→country weight per row
            sign_consistency = (np.sign(e1) == np.sign(e1.mean(axis=0))).mean(axis=0)
            res[(b, tau)] = {"obs": obs, "transplant": transplant, "null": null, "split": split, "full": fs,
                             "boot": boot, "sign_consistency": sign_consistency,
                             "p": np.array([pval(null[:, u], obs[u]) for u in range(N_LAYERS)]),
                             "null95": np.quantile(null, 1 - t["alpha"], axis=0)}
            log(f"{b} τ {tau}: own null / bootstrap done", t0)

    # ---- 5. verdicts re-derived from this file's numbers
    early = range(t["early_layers"][0], t["early_layers"][1] + 1)
    late = range(t["late_layers"][0], t["late_layers"][1] + 1)
    g1_tau = t["g1_tau"]
    g1_ok = all(res[(b, g1_tau)]["p"][l] <= t["alpha"] for b in fillers for l in early)

    def g2(strict: bool, a: str = "own", z: str = "wrong") -> list[tuple[int, int]]:
        pos, neg = ("all", "some") if strict else ("some", "all")
        cells = []
        for tau in EXTRACT_TAUS:
            for u in range(N_LAYERS):
                ok = True
                for b in fillers:
                    r = res[(b, tau)]
                    ok &= (r["full"][a][u][pos] is not None and r["boot"][(a, pos)][u] >= t["boot_stable"]
                           and r["full"][z][u][neg] is None and r["boot"][(z, neg)][u] <= t["boot_absent"])
                if ok:
                    cells.append((u, tau))
        return cells

    g2_strict, g2_len = g2(True), g2(False)
    g2_rev = sorted(set(g2(True, "wrong", "own")) | set(g2(False, "wrong", "own")))

    def transplant_cells(layers: range) -> list[tuple]:
        return [(b, tau, l, res[(b, tau)]["obs"][l], res[(b, tau)]["transplant"][l], res[(b, tau)]["null95"][l])
                for b in fillers for tau in t["far_taus"] for l in layers if res[(b, tau)]["p"][l] <= t["alpha"]]

    ce, cl = transplant_cells(early), transplant_cells(late)
    g3a_in = [c for c in ce if c[4] <= c[5]]
    g3b_out = [c for c in cl if c[4] > c[5]]
    mine = {"G1": "PASS" if g1_ok else "FAIL", "G2": "PASS" if g2_strict and g2_len else ("FAIL" if not g2_len else "PASS (lenient only)"),
            "G3a": "PASS" if ce and len(g3a_in) == len(ce) else "FAIL",
            "G3b": "PASS" if cl and len(g3b_out) >= len(cl) / 2 else "FAIL", "B1": "PASS" if not b1_found else "FAIL"}
    log("verdicts: " + ", ".join(f"{k} {v}" for k, v in mine.items()), t0)

    # ---- 6. tightness of the swap null in the saved run (noBOS, own vs wrong, every unit, τ and filler)
    ratio, sig = [], 0
    for b in fillers:
        for tau in taus:
            o, n95, p = A[f"{b}/t{tau}/obs"][0, :, 0, 0], A[f"{b}/t{tau}/null95"][0, :, 0], A[f"{b}/t{tau}/p"][0, :, 0]
            ratio += list(o / n95)
            sig += int((p <= t["alpha"]).sum())
    ratio = np.array(ratio)

    # ---- 7. write
    np.savez_compressed(out_dir / "verify_arrays.npz", **{f"{b}/t{tau}/{k}": v for (b, tau), r in res.items()
                                                         for k, v in r.items() if isinstance(v, np.ndarray)})
    summary = {"provenance": prov, "coverage": coverage, "rows_used": R, "iterations": [n_swap, n_split, n_boot],
               "hf": {"row_sum_dev": hf_dev, "upper_max": hf_up, "prefix_max": prefix, "vs_saved_graphs_max": hf_vs_saved,
                      "wrong_mapping": mapping},
               "rederive": {"obs_max_dev": obs_dev, "obs_max_rel_dev": obs_rel, "transplant_max_dev": tr_dev, "role_mismatch": role_mis,
                            "n_persistent_mismatch": npers_mis, "mpc_outside_bounds": mpc_bad,
                            "run_mpcs": mpc_run_n, "run_mpcs_strict": mpc_run_strict, "ambiguous_birth_edges": amb_total,
                            "graphs": len(full)},
               "verdicts_here": mine, "verdicts_run": {k: v["verdict"] for k, v in preds.items()},
               "g2_strict": g2_strict, "g2_lenient": g2_len, "g2_reverse": g2_rev, "no_own_mpc_tau30_28": no_mpc_30_28,
               "g3a": {"eligible": len(ce), "within_null": len(g3a_in)}, "g3b": {"eligible": len(cl), "beyond_null": len(g3b_out)},
               "b1_found": b1_found, "null_tightness": {"cells": len(ratio), "p_le_alpha": sig,
                                                        "ratio_quartiles": np.quantile(ratio, [0.25, 0.5, 0.75]).tolist()},
               "timing_s": time.time() - t0}
    (out_dir / "verify.json").write_text(json.dumps(summary, indent=2, default=float))
    write_addendum(out_dir, cfg, units, fillers, taus, A, J, preds, checks, coverage, prov, summary, res, ce, cl, R)
    log(f"done: {out_dir / 'addendum.md'}", t0)
    return out_dir


def write_addendum(out: Path, cfg: dict, units: list[str], fillers: list[str], taus: list[int], A, J: dict, preds: dict,
                   checks: dict, cov: dict, prov: dict, S: dict, res: dict, ce: list, cl: list, R: int) -> None:
    t = cfg["tda"]
    g1_tau = t["g1_tau"]
    hf, rd = S["hf"], S["rederive"]
    tm = cov["timing_s"]
    L = ["# TDA analysis 4: verification and addendum", "",
         f"Verifies the saved run `{RUN.name}` (pre-registration `exp4/PREDICTIONS_tda4.md`, code `exp4/tda4_graph.py`). "
         f"Written by `exp4/tda4_finalize.py` into this new folder (`{out.name}`); the run folder is unchanged. "
         "The original report is `tda4_report.md` in the run folder; this file adds the independent checks, a plain-language "
         "reading, and the pre-registered 'reported, not predicted' tables that the report left out.", "",
         "## 1. Completeness and provenance", "",
         f"- Coverage: fillers {', '.join(fillers)}; conditions control, own, wrong, random; τ ∈ {{{', '.join(map(str, taus))}}}; "
         f"{len(units)} units (12 layer means, 8 named heads, 2 pooled); {cfg['n_rows']} rows per filler (the whole cached block, "
         f"base_cache_n = {cfg['base_cache_n']}). Saved graph arrays {cov['graph_shape']} per filler × condition; "
         f"swap null {cov['null_shape']} [pair, swap, unit, variant]; split-half {cov['split_shape']}; "
         f"{cov['cells_with_arrays']}/14 (filler, τ) cells have TDA arrays. Bootstrap: {t['n_boot']} resamples.",
         f"- Runtime: extraction {tm['extract'] / 60:.1f} min, TDA {tm['tda'] / 60:.1f} min, total {tm['total'] / 60:.1f} min (CPU).",
         f"- Code: `tda4_graph.py` last modified {prov['tda4_graph_mtime']}, before the run started at {prov['run_start']}. "
         f"Files in exp1/exp1ext/exp2/exp3/exp4 modified after the start: {', '.join(prov['modified_after_start']) or 'none'} "
         "(not imported by `tda4_graph.py`). The run's recorded code hash "
         + ("equals" if prov["smoke2_code_sha256"] == prov["run_code_sha256"] else "differs from")
         + " that of the smoke run started at 11:35:48. The HF re-extraction and the re-derived distances below reproduce the "
         "saved arrays, so the run used the final code.",
         f"- The run's `PREDICTIONS.md` copy is {'identical to' if prov['prereg_copy_identical'] else 'DIFFERENT from'} "
         f"`exp4/PREDICTIONS_tda4.md`; its `config.yaml` is {'identical to' if prov['config_copy_identical'] else 'DIFFERENT from'} "
         "`configs/exp4_tda4.yaml`.", "",
         "## 2. Checks", "", "Run's own checks (`checks.json`):", "", "| check | result |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in checks.items()]
    L += ["", "Independent checks here (HuggingFace GPT-2, eager attention, sequences built in this file; own and wrong at "
          f"τ ∈ {{{', '.join(map(str, EXTRACT_TAUS))}}}, {R} rows per filler):", "",
          "| check | result |", "|---|---|",
          f"| design (country at τ, capital at 32, row elsewhere, no repeats, no BOS after 0, no stimulus ids in rows) | PASS (asserted) |",
          f"| wrong capitals | {hf['wrong_mapping']} |",
          f"| attention rows sum to 1 | max abs(sum − 1) {hf['row_sum_dev']:.1e} |",
          f"| causal (upper triangle) | max {hf['upper_max']:.1e} |",
          f"| prefix identity, own vs wrong query rows 0..31 | max abs Δ {hf['prefix_max']:.1e} |",
          f"| HF per-filler mean graphs vs saved `graphs.npz` (TransformerLens), layer means | max abs Δ {hf['vs_saved_graphs_max']:.1e} |",
          f"| every saved distance D (3 pairs × 22 units × 2 variants × 14 cells) re-derived from `graphs.npz` with MST + ripser + assignment W1 | max abs Δ {rd['obs_max_dev']:.1e}, max relative Δ {rd['obs_max_rel_dev']:.1e} (the run's gudhi.hera W1 uses its default delta = 0.01, i.e. it is exact only to within 1%; ripser works in float32) |",
          f"| every saved transplant distance re-derived | max abs Δ {rd['transplant_max_dev']:.1e} |",
          f"| E1 role, all {rd['graphs']} full-sample graphs | {len(rd['role_mismatch'])} disagreements |",
          f"| persistent H1 count, all full-sample graphs | {len(rd['n_persistent_mismatch'])} disagreements |",
          f"| run's MPC between strict (E1 on every shortest representative) and lenient (on some) | {len(rd['mpc_outside_bounds'])} violations; "
          f"{rd['run_mpcs_strict']}/{rd['run_mpcs']} run MPCs hold under the strict reading |",
          f"| H1 birth edges matched by value | {rd['ambiguous_birth_edges']} ambiguous |", ""]
    for name in ("role_mismatch", "n_persistent_mismatch", "mpc_outside_bounds"):
        if rd[name]:
            L += [f"{name}: " + "; ".join(rd[name][:20]), ""]

    L += [f"## 3. Headline statistic reproduced: own vs wrong D at τ = {g1_tau} (noBOS, layer means)", "",
          f"D(run) is the saved value; D(here) is recomputed on the HF graphs. The swap null here uses {S['iterations'][0]} new "
          f"paired swaps (seed {SEED}), and the split-half uses {S['iterations'][1]} new splits. The split-half is "
          "descriptive in the pre-registration. It compares two 100-row halves that differ in all 528 edges, so it is "
          "not the null for own vs wrong.", "",
          "| filler | layer | D run | D here | null95 run | null95 here | p run | p here | split-half median run | split-half median here | rows with own−wrong E1 of the mean's sign |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for b in fillers:
        r = res[(b, g1_tau)]
        for u in range(N_LAYERS):
            L.append(f"| {b} | L{u} | {A[f'{b}/t{g1_tau}/obs'][0, u, 0, 0]:.4f} | {r['obs'][u]:.4f} | "
                     f"{A[f'{b}/t{g1_tau}/null95'][0, u, 0]:.4f} | {r['null95'][u]:.4f} | {A[f'{b}/t{g1_tau}/p'][0, u, 0]:.3f} | "
                     f"{r['p'][u]:.3f} | {np.median(A[f'{b}/t{g1_tau}/split'][:, u, 0]):.4f} | {np.median(r['split'][:, u]):.4f} | "
                     f"{r['sign_consistency'][u]:.2f} |")

    v = S["verdicts_here"]
    nt = S["null_tightness"]
    vi_alpha = t["alpha"]
    run_in = [f"{b} τ{tau} L{l}" for b in fillers for tau in t["far_taus"] for l in range(5)
              if A[f"{b}/t{tau}/p"][0, l, 0] <= vi_alpha and A[f"{b}/t{tau}/transplant"][l, 0] <= A[f"{b}/t{tau}/null95"][0, l, 0]]
    run_el = sum(A[f"{b}/t{tau}/p"][0, l, 0] <= vi_alpha for b in fillers for tau in t["far_taus"] for l in range(5))
    L += ["", "## 4. Pre-registered verdicts (primary variant noBOS)", "",
          "| id | rule (short) | run | re-derived here | numbers |", "|---|---|---|---|---|",
          f"| G1 | τ {g1_tau}, own-vs-wrong D beyond the swap null (p ≤ {t['alpha']}) in each of L0–L4, both fillers | "
          f"{preds['G1']['verdict']} | {v['G1']} | all 10 cells p = 0.005 in the run (observed above all 200 swaps); here "
          f"max p {max(res[(b, g1_tau)]['p'][l] for b in fillers for l in range(5)):.3f}; D / null95 from "
          f"{min(A[f'{b}/t{g1_tau}/obs'][0, l, 0, 0] / A[f'{b}/t{g1_tau}/null95'][0, l, 0] for b in fillers for l in range(5)):.1f} (rare L4) to "
          f"{max(A[f'{b}/t{g1_tau}/obs'][0, l, 0, 0] / A[f'{b}/t{g1_tau}/null95'][0, l, 0] for b in fillers for l in range(5)):.1f} |",
          f"| G2 | some layer-mean cell where, in both fillers, own has a stable MPC (full sample, bootstrap ≥ {t['boot_stable']}) "
          f"and wrong has none (bootstrap ≤ {t['boot_absent']}) | {preds['G2']['verdict']} | {v['G2']} | run: "
          "4 cells (L4 τ 24, L0 τ 16, L0 τ 8, L0 τ 1; own bootstrap 1.00, wrong 0.00 in both fillers); reverse 0. Here strict: "
          + (", ".join(f"L{u} τ {tau}" for u, tau in S["g2_strict"]) or "none") + "; lenient: "
          + (", ".join(f"L{u} τ {tau}" for u, tau in S["g2_lenient"]) or "none") + f"; reverse {len(S['g2_reverse'])}. "
          f"τ 30 and 28 have no own MPC in any layer mean: {S['no_own_mpc_tau30_28']} |",
          f"| G3a | τ ∈ {{16, 8, 1}}, L0–L4: every cell beyond the null falls to ≤ null95 after transplanting own's A[32, τ] "
          f"to wrong's value | {preds['G3a']['verdict']} | {v['G3a']} | run {len(run_in)}/{run_el} fall within the null "
          f"({', '.join(run_in)}); here {S['g3a']['within_null']}/{S['g3a']['eligible']} |",
          f"| G3b | τ ∈ {{16, 8, 1}}, L6–L11: ≥ half of the cells beyond the null stay above null95 after the transplant | "
          f"{preds['G3b']['verdict']} | {v['G3b']} | run 34/36; here {S['g3b']['beyond_null']}/{S['g3b']['eligible']} |",
          f"| B1 | withBOS: no MPC in any layer 1–11, any τ, either filler | {preds['B1']['verdict']} | {v['B1']} | "
          f"run none (all four conditions); here, lenient reading, {len(S['b1_found'])} |", "",
          "Transplant cells here (filler, τ, layer: D → D after transplant, null95):", ""]
    for name, cells in (("early (G3a)", ce), ("late (G3b)", cl)):
        L.append(f"- {name}: " + "; ".join(f"{b} τ{tau} L{l}: {o:.4f} → {tr:.4f} ({n95:.4f})" for b, tau, l, o, tr, n95 in cells))
    L += ["", "## 5. What the topology shows", "",
          "- **What can differ.** Own, wrong and random share tokens 0..31, so their graphs are identical except for node 32's "
          "32 edges (prefix identity, max abs Δ 0). Every own-vs-wrong topological difference is about how the capital's "
          "node attaches to a fixed network of the other 31 tokens.",
          "- **BOS.** With BOS kept (withBOS), each token's heavy edge to BOS cones off any loop as soon as the loop forms. "
          "No matching-pair cycle exists in layers 1–11 in any condition (B1 PASS). So the question is asked on the noBOS "
          "graph: node 0 is deleted and the remaining weights are kept raw (no renormalisation). The capital's late drop in "
          "BOS attention then shows up only as extra weight on its content edges.",
          "- **Adjacent country (τ = 31): no loop.** E1 = E3 is the capital's previous-token edge. In almost every layer "
          "mean it is an H0 merge edge, so node 32 hangs off the network as a leaf. The exception is L2 own, where it is "
          "'instant': the matching capital's weaker previous-token edge lets node 32 attach through another edge first. "
          "The topological change is how tightly node 32 attaches: tighter with the matching capital in L0/L1/L4 (the "
          "same-country heads; L0 E1 +0.050 common, +0.048 rare) and looser in L2/L3 (the previous-token heads look away; "
          "L2 −0.048, L3 −0.040/−0.037). G1 passes on these weight changes at node 32, not on a cycle. The only τ = 31 "
          "loop through E1 is in common L10.",
          "- **Distant country, layer 0: a loop that only the matching capital closes.** At τ ≤ 16 (both fillers) and at "
          "τ = 24 (rare), the matching capital's heavier edge to its country (L0 E1 own − wrong about +0.03) joins node 32 "
          "to the network through the country first (E1 is a merge edge). When the capital's own previous-token edge "
          "{31, 32} enters, it closes a long loop: 32 → τ, then a chain of short back edges (mostly 1–5 back) up to 31, then "
          "32. At τ = 1 the hop from position 1 is a long edge. These are 7- to 13-node cycles with persistence "
          "0.010–0.018 and bootstrap frequency ≥ 0.99. With a wrong capital, E1 enters only after 32 and τ are already "
          "linked, so it is filled at once or makes a short-lived cycle; bootstrap 0.00. Random and control never have "
          "this loop.",
          "- **Layer 4, τ = 24: a short loop closed by E1 itself.** The cycle is 24–26–28–30–32 over layer 4's 2-back chain "
          "(persistence 0.017 common, 0.011 rare). Own has it in both fillers and wrong in neither.",
          "- **Layers 1, 3, 4 at τ ≤ 16: the loop often exists with both capitals.** Own and wrong both have a stable loop "
          "through E1 in common L1, L3 and L4 at τ 16/8/1, and in rare L4 at τ 16/8/1, rare L3 at τ 8/1 and rare L1 at "
          "τ 1. Rare L3 at τ 16 is own-only, but common L3 at τ 16 is shared, so it is not a G2 cell. A wrong capital "
          "still attends to the country token. In L1 it attends about as much as the matching one (E1 own − wrong −0.003 "
          "to −0.005 common, within ±0.001 rare, at τ ≤ 16), so the loop marks 'a country is present', not the match. "
          "In L3/L4 the match makes the loop longer-lived (e.g. common L4 τ 8: persistence 0.030 vs 0.009; L3 τ 8: 0.024 "
          "vs 0.013). The pooled L0–4 unit is own-only at τ ≤ 16 in rare and at τ 16 in common. The single head "
          "L3H0 has a loop through E1 at τ 8/1 in own, wrong and also random and/or control, so it is not specific to the "
          "capital.",
          "- **Late layers (5–11).** There are no stable matching-pair cycles, except common L10 at τ 31 (one filler only, so "
          "not a G2 cell). Own-vs-wrong distances are beyond the null, but transplanting the country edge leaves them "
          "unchanged in most far-τ cells (G3b PASS). The late difference sits on node 32's other content edges, where the "
          "BOS mass freed by the matching capital goes, not on the country edge.",
          "- **G3a fails because only layer 0 is the country edge alone.** In L0 the transplant removes the whole difference "
          "(D 0.017–0.030 → 0.0005, 6/6 cells). In L2 the transplant changes nothing, because E1 is not a critical edge "
          "there and the difference is in the capital's other edges. In L3/L4 the transplant removes 33–87% of D (run "
          "values) but not all of it.", "",
          "## 6. Caveats added by this verification", "",
          f"- **The swap null is very tight, so 'beyond the null' means 'consistent', not 'large'.** {nt['p_le_alpha']}/{nt['cells']} "
          "noBOS own-vs-wrong cells (every unit, τ and filler) have p ≤ 0.05. D / null95 has quartiles "
          f"{nt['ratio_quartiles'][0]:.1f} / {nt['ratio_quartiles'][1]:.1f} / {nt['ratio_quartiles'][2]:.1f}, which is close to "
          "√200 / 1.96 ≈ 7.2. That is the value expected when a unit's per-row own − wrong difference has the same sign in "
          "nearly every row. For example, for the country edge in L0–L4 at τ 31, 75–100% of rows share the sign of the "
          "mean (last column of section 3). Each row graph averages the same 11 country–capital pairs, so the effect is "
          "nearly deterministic across rows. The null tests exchangeability across rows given these 11 pairs; it "
          "says nothing about other countries. G1 passing therefore shows a consistent difference, and G3a/G3b compare "
          "against a very small yardstick.",
          "- In absolute terms the own-vs-wrong D is small next to the row-sampling variation of the whole graph: the "
          "split-half median is of similar size or larger for most layer means. That comparison is not like for like "
          "(split-half differs in all 528 edges, own vs wrong in 32), which is why the pre-registration kept it descriptive.",
          f"- Representative cycles: {rd['run_mpcs_strict']}/{rd['run_mpcs']} of the run's full-sample MPCs keep E1 on every hop-shortest "
          "representative. The G2 cells are re-derived here under both the strict and the lenient reading (section 4).",
          "- The G2 cells are bootstrap-stable over rows, but the 11 country–capital pairs are fixed. Cycle lengths and "
          "persistences come from mean graphs over 2,200 sequences, not from per-sequence topology.", "",
          "## 7. Pre-registered 'reported' items missing from the original report", "",
          "### 7a. Stability bootstrap: E1 role frequencies (noBOS; own / wrong; modal role and its frequency over "
          f"{t['n_boot']} resamples)", ""]
    roles = [str(r) for r in A["roles"]]
    conds = [str(c) for c in A["conds"]]
    show = list(range(N_LAYERS)) + [units.index("L0-4")]
    for b in fillers:
        L += [f"**{b}**", "", "| unit | " + " | ".join(f"τ {tau}" for tau in taus) + " |", "|---" * (len(taus) + 1) + "|"]
        for u in show:
            cells = []
            for tau in taus:
                br = A[f"{b}/t{tau}/boot_role"]
                parts = []
                for c in ("own", "wrong"):
                    f = br[conds.index(c), u, 0]
                    k = int(np.argmax(f))
                    parts.append(f"{ROLE_ABBR[roles[k]]}{f[k]:.2f}")
                cells.append("/".join(parts))
            L.append(f"| {units[u]} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["### 7b. Role of E1 in the random and control conditions (full sample, noBOS; random / control)", ""]
    for b in fillers:
        L += [f"**{b}**", "", "| unit | " + " | ".join(f"τ {tau}" for tau in taus) + " |", "|---" * (len(taus) + 1) + "|"]
        for u in show:
            g = lambda tau, c: J["cells"][f"{b}/t{tau}"]["graphs"][f"{c}/{units[u]}/noBOS"]["role"]
            L.append(f"| {units[u]} | " + " | ".join(f"{ROLE_ABBR[g(tau, 'random')]}/{ROLE_ABBR[g(tau, 'control')]}" for tau in taus) + " |")
        L.append("")
    for name in ("E1 own−wrong", "E1 own−control"):
        L += [f"### 7c. {name} with 95% bootstrap CI over rows (layer means L0–L4 and the same-country heads)", ""]
        for b in fillers:
            L += [f"**{b}**", "", "| unit | " + " | ".join(f"τ {tau}" for tau in taus) + " |", "|---" * (len(taus) + 1) + "|"]
            for u in list(range(5)) + [units.index(h) for h in ("L0H2", "L1H11", "L3H0", "L4H10")]:
                cells = []
                for tau in taus:
                    e = J["cells"][f"{b}/t{tau}"]["edges"]["diff"][name]
                    cells.append(f"{e['mean'][u]:+.4f} [{e['lo'][u]:+.4f}, {e['hi'][u]:+.4f}]")
                L.append(f"| {units[u]} | " + " | ".join(cells) + " |")
            L.append("")
    L += ["### 7d. Units beyond their swap null, per pair (noBOS, count out of 22 units)", "",
          "| filler | τ | own–wrong | own–random | own–control |", "|---|---|---|---|---|"]
    for b in fillers:
        for tau in taus:
            p = A[f"{b}/t{tau}/p"][:, :, 0]
            L.append(f"| {b} | {tau} | " + " | ".join(str(int((p[i] <= t["alpha"]).sum())) for i in range(3)) + " |")
    L += ["", "### 7e. Persistent H1 classes per graph, wrong condition (layer means; noBOS / withBOS); own is in the report", "",
          "| filler | τ | " + " | ".join(units[:N_LAYERS]) + " |", "|---|---" + "|---" * N_LAYERS + "|"]
    for b in fillers:
        for tau in taus:
            g = J["cells"][f"{b}/t{tau}"]["graphs"]
            L.append(f"| {b} | {tau} | " + " | ".join(f"{g[f'wrong/{units[u]}/noBOS']['n_persistent']}/{g[f'wrong/{units[u]}/withBOS']['n_persistent']}"
                                                   for u in range(N_LAYERS)) + " |")
    L += ["", "## Files", "",
          "- `verify.json`: provenance, coverage, independent checks, re-derived verdicts and null-tightness summary.",
          f"- `verify_arrays.npz`: per (filler, τ ∈ {{{', '.join(map(str, EXTRACT_TAUS))}}}) the HF-based observed D, transplant D, "
          "swap null [swap, layer], p, null95, split-half (τ 31 only) and per-row E1 sign consistency, for layer means L0–L11.",
          f"- Original run: `results/{RUN.name}/` (`tda4_report.md`, `figures/`, arrays).", ""]
    (out / "addendum.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
