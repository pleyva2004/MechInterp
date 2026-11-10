"""Experiment 5, analysis P: activation patching of the final token's stream (P-a) and of the same-country heads'
writes (P-b). Hypotheses: exp5/PREDICTIONS.md; operational choices: exp5/PREDICTIONS_P.md.

Run:  uv run --offline python -m exp5.patch --config configs/exp5_residual.yaml [--rows N] [--results-dir DIR]

Capital-paired cells (exp5.extract.cells): capital k at T in both runs; own = country k at tau, wrong = the country
whose deranged capital is k. For each (filler, tau in {16, 1}, row block) every run uses the same batch layout:
own and wrong (clean, caching x_l(T) and z(T)); own with x_l(T) from wrong (l = 0..11) and the reverse (l = 1..11);
own with the z(T) of L0H2, L1H11, L3H0, L4H10 from wrong (together and each alone) and the reverse. Measured: the
attention from T to BOS of every layer and head. Writes results/<timestamp>_exp5P/.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy.stats import rankdata

from exp1.backends import check_rows_sum_to_one, load_model, set_determinism
from exp1.runlog import make_run_dir, save_config_copy, save_versions
from exp1.sequences import prepend_bos
from exp2.design import cached_base_block
from exp3.run_relations import PROJECT_ROOT, log, stimuli
from exp4.run_controlled import check_variants, variants
from exp5.extract import cells

L, H, DH, DM = 12, 12, 64, 768
TAUS = (16, 1)
PATCH_LAYERS = tuple(range(1, L))  # P-a: l = 1..11
LATE = (6, 11)
HEADS = ((0, 2), (1, 11), (3, 0), (4, 10))
HEAD_SETS = {"heads": HEADS, **{f"L{l}H{h}": ((l, h),) for l, h in HEADS}}
BLOCK_ROWS = 30  # 30 rows x 11 cells = 330 sequences, the config's batch size
BOOT, BOOT_SEED, CI = 5000, 0, 0.95
DET_ROWS, DET_TAU = 2, 16
# Exp 4b late BOS fact (its capital − wrong capital, layers 6–11), exp4_capital_report.md, by tau = 32 − d
EXP4B = {16: {"common": -0.0291, "rare": -0.0257}, 1: {"common": -0.0403, "rare": -0.0332}}
RUNS = (["own", "wrong"] + [f"fwd_l{l}" for l in range(L)] + [f"rev_l{l}" for l in PATCH_LAYERS]
        + [f"fwd_{k}" for k in HEAD_SETS] + [f"rev_{k}" for k in HEAD_SETS])
RI = {r: i for i, r in enumerate(RUNS)}
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"
INK, MUTED = "#2b2b2a", "#898781"


def bos_run(model, x: torch.Tensor, patches: list, atol: float, cache: bool = False) -> dict:
    """One forward pass of x [N, T + 1] (BOS first) with the given (hook name, fn) patches. Returns the attention from
    T to BOS [N, L, H] (float32) and the max |row sum − 1| at T; with cache, x_l(T) [N, L, 768] and z(T) [N, L, H, 64]."""
    n = x.shape[0]
    bos = np.zeros((n, L, H), np.float32)
    dev = [0.0]
    resid: dict[int, torch.Tensor] = {}
    z: dict[int, torch.Tensor] = {}

    def pat_hook(pattern, hook):
        row = pattern[:, :, -1, :]
        dev[0] = max(dev[0], check_rows_sum_to_one(row.numpy(), atol))
        bos[:, hook.layer()] = row[:, :, 0].numpy()

    hooks = [(f"blocks.{l}.attn.hook_pattern", pat_hook) for l in range(L)]
    if cache:
        hooks += [(f"blocks.{l}.hook_resid_pre", lambda a, hook: resid.__setitem__(hook.layer(), a[:, -1].clone()))
                  for l in range(L)]
        hooks += [(f"blocks.{l}.attn.hook_z", lambda a, hook: z.__setitem__(hook.layer(), a[:, -1].clone()))
                  for l in range(L)]
    with torch.inference_mode():
        model.run_with_hooks(x, fwd_hooks=hooks + patches, return_type=None)
    out = {"bos": bos, "dev": dev[0]}
    if cache:
        out["resid"] = torch.stack([resid[l] for l in range(L)], dim=1)
        out["z"] = torch.stack([z[l] for l in range(L)], dim=1)
    return out


def resid_patch(l: int, src: torch.Tensor) -> tuple:
    """Replace x_l(T) (blocks.l.hook_resid_pre at the last position) with src [N, 768]."""
    def fn(act, hook):
        act[:, -1] = src
        return act
    return (f"blocks.{l}.hook_resid_pre", fn)


def z_patches(heads: tuple, src: torch.Tensor) -> list:
    """Replace z_{l,h}(T) with src[:, l, h] for each (l, h); the head's write at T is z W_O^{l,h} (= its hook_result)."""
    def make(l: int, h: int):
        def fn(act, hook):
            act[:, -1, h] = src[:, l, h]
            return act
        return (f"blocks.{l}.attn.hook_z", fn)
    return [make(l, h) for l, h in heads]


def run_block(model, own_x: torch.Tensor, wrong_x: torch.Tensor, atol: float, identity: bool,
              only: tuple[str, ...] | None = None) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """All runs of RUNS on one block (or only those in `only`). Returns {run: BOS attention [N, L, H]} and check maxima."""
    own = bos_run(model, own_x, [], atol, cache=True)
    wr = bos_run(model, wrong_x, [], atol, cache=True)
    out = {"own": own["bos"], "wrong": wr["bos"]}
    ck = {"rowsum": max(own["dev"], wr["dev"]), "causal": 0.0, "identity": 0.0,
          "null0": float((own["resid"][:, 0] - wr["resid"][:, 0]).abs().max())}
    jobs = []  # (run name, base run, base input, patches, first patched layer)
    for l in range(L):
        jobs.append((f"fwd_l{l}", "own", own_x, [resid_patch(l, wr["resid"][:, l])], l))
    for l in PATCH_LAYERS:
        jobs.append((f"rev_l{l}", "wrong", wrong_x, [resid_patch(l, own["resid"][:, l])], l))
    for k, hs in HEAD_SETS.items():
        first = min(l for l, _ in hs)
        jobs.append((f"fwd_{k}", "own", own_x, z_patches(hs, wr["z"]), first))
        jobs.append((f"rev_{k}", "wrong", wrong_x, z_patches(hs, own["z"]), first))
    for name, base, x, patches, first in jobs:
        if only is not None and name not in only:
            continue
        r = bos_run(model, x, patches, atol)
        out[name] = r["bos"]
        ck["rowsum"] = max(ck["rowsum"], r["dev"])
        if first > 0:  # layers before the patch cannot change
            ck["causal"] = max(ck["causal"], float(np.abs(r["bos"][:, :first] - out[base][:, :first]).max()))
    ck["l0"] = float(np.abs(out["fwd_l0"] - out["own"]).max()) if "fwd_l0" in out else 0.0
    if identity:  # own patched with its own activations: must change nothing
        for l in PATCH_LAYERS:
            r = bos_run(model, own_x, [resid_patch(l, own["resid"][:, l])], atol)
            ck["identity"] = max(ck["identity"], float(np.abs(r["bos"] - own["bos"]).max()))
        r = bos_run(model, own_x, z_patches(HEADS, own["z"]), atol)
        ck["identity"] = max(ck["identity"], float(np.abs(r["bos"] - own["bos"]).max()))
    return out, ck


def sequences(bases: np.ndarray, cl: dict, tau: int, bos_id: int) -> tuple[np.ndarray, np.ndarray]:
    """own and wrong sequences [R, n, T] for one tau, with the design asserts."""
    nr = len(bases)
    cap = np.tile(cl["capital"], (nr, 1))
    out = []
    for c in ("own", "wrong"):
        fill = np.tile(cl[c], (nr, 1))
        seqs = variants(bases, fill, tau, cap)
        check_variants(seqs, bases, fill, tau, bos_id, cap)
        if not (seqs[:, :, -1] == cap).all():
            raise AssertionError("the final token is not capital k")
        out.append(seqs)
    if not (out[0] != out[1]).sum(axis=-1).max() == 1:
        raise AssertionError("own and wrong differ at more than one position")
    return out[0], out[1]


def to_x(seqs: np.ndarray, bos_id: int) -> torch.Tensor:
    return torch.as_tensor(prepend_bos(seqs.reshape(-1, seqs.shape[-1]), bos_id), dtype=torch.long)


# ---------------------------------------------------------------- analysis

def late(x: np.ndarray, lo: int, hi: int = LATE[1]) -> np.ndarray:
    return x[..., lo:hi + 1].mean(axis=-1)


def ratio_boot(num: np.ndarray, den: np.ndarray, idx: np.ndarray) -> dict[str, np.ndarray]:
    """num, den [R, K] per-row quantities: ratio of row means, its row-bootstrap CI, and the denominator's mean and CI."""
    a = (1 - CI) / 2 * 100
    nb, db = num[idx].mean(axis=1), den[idx].mean(axis=1)  # [BOOT, K]
    fb = nb / db
    return {"F": num.mean(0) / den.mean(0), "lo": np.percentile(fb, a, axis=0), "hi": np.percentile(fb, 100 - a, axis=0),
            "den": den.mean(0), "den_lo": np.percentile(db, a, axis=0), "den_hi": np.percentile(db, 100 - a, axis=0),
            "num": num.mean(0), "boot": fb}


def spearman_l(F: np.ndarray, Fb: np.ndarray) -> dict[str, float]:
    """Spearman rho between l = 1..K and F(l), and its bootstrap CI from Fb [BOOT, K]."""
    a = (1 - CI) / 2 * 100
    rl = rankdata(np.arange(F.shape[-1]))

    def rho(f: np.ndarray) -> np.ndarray:
        rf = rankdata(f, axis=-1)
        rf = rf - rf.mean(axis=-1, keepdims=True)
        r0 = rl - rl.mean()
        return (rf * r0).sum(-1) / np.sqrt((rf ** 2).sum(-1) * (r0 ** 2).sum())

    rb = rho(Fb)
    return {"rho": float(rho(F[None])[0]), "lo": float(np.percentile(rb, a)), "hi": float(np.percentile(rb, 100 - a))}


def analyse(bl: np.ndarray, own_heads: np.ndarray, wrong_heads: np.ndarray) -> dict:
    """bl [R, n, runs, L] head-mean BOS attention at T per layer. Returns every quantity of the report."""
    R = bl.shape[0]
    idx = np.random.default_rng(BOOT_SEED).integers(0, R, size=(BOOT, R))
    c = bl.mean(axis=1)  # cell mean, [R, runs, L]
    g = lambda run: c[:, RI[run]]
    res: dict = {}
    d_late = late(g("own"), LATE[0]) - late(g("wrong"), LATE[0])
    db = d_late[idx].mean(axis=1)
    res["own_minus_wrong"] = {"mean": float(d_late.mean()), "lo": float(np.percentile(db, 2.5)),
                              "hi": float(np.percentile(db, 97.5)), "own": float(late(g("own"), LATE[0]).mean()),
                              "wrong": float(late(g("wrong"), LATE[0]).mean())}
    num_f = np.stack([late(g(f"fwd_l{l}"), max(l, LATE[0])) - late(g("own"), max(l, LATE[0])) for l in PATCH_LAYERS], 1)
    den_f = np.stack([late(g("wrong"), max(l, LATE[0])) - late(g("own"), max(l, LATE[0])) for l in PATCH_LAYERS], 1)
    num_r = np.stack([late(g(f"rev_l{l}"), max(l, LATE[0])) - late(g("wrong"), max(l, LATE[0])) for l in PATCH_LAYERS], 1)
    fwd, rev = ratio_boot(num_f, den_f, idx), ratio_boot(num_r, -den_f, idx)
    res["Pa"] = {"fwd": fwd, "rev": rev, "rho": spearman_l(fwd["F"], fwd["boot"]),
                 "rho_rev": spearman_l(rev["F"], rev["boot"])}
    per_layer = {}
    for m in range(LATE[0], L):
        num = np.stack([g(f"fwd_l{l}")[:, m] - g("own")[:, m] for l in PATCH_LAYERS], 1)
        den = np.repeat((g("wrong")[:, m] - g("own")[:, m])[:, None], len(PATCH_LAYERS), 1)
        per_layer[m] = ratio_boot(num, den, idx)
    res["Pa_layers"] = per_layer
    heads = {}
    den = late(g("wrong"), LATE[0]) - late(g("own"), LATE[0])
    for k in HEAD_SETS:
        nf = late(g(f"fwd_{k}"), LATE[0]) - late(g("own"), LATE[0])
        nr = late(g(f"rev_{k}"), LATE[0]) - late(g("wrong"), LATE[0])
        heads[k] = {"fwd": ratio_boot(nf[:, None], den[:, None], idx), "rev": ratio_boot(nr[:, None], -den[:, None], idx)}
    res["Pb"] = heads
    # per pair (cell): own − wrong and the four-head transfer, ratio of row means within the cell
    cl = bl[:, :, :, LATE[0]:].mean(-1)  # [R, n, runs]
    res["pairs"] = {"own_minus_wrong": (cl[:, :, RI["own"]] - cl[:, :, RI["wrong"]]).mean(0),
                    "G_heads": (cl[:, :, RI["fwd_heads"]] - cl[:, :, RI["own"]]).mean(0)
                    / (cl[:, :, RI["wrong"]] - cl[:, :, RI["own"]]).mean(0),
                    "F_l6": (cl[:, :, RI["fwd_l6"]] - cl[:, :, RI["own"]]).mean(0)
                    / (cl[:, :, RI["wrong"]] - cl[:, :, RI["own"]]).mean(0)}
    # per head, own − wrong BOS attention (row means), for the late heads that move most
    res["head_delta"] = (own_heads - wrong_heads).mean(0)  # [L, H]
    return res


# ---------------------------------------------------------------- figures

def style(ax) -> None:
    ax.grid(True, color="#e6e5e0", linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=8)


def plot_F(res: dict, fillers: list[str], path: Path) -> None:
    fig, axes = plt.subplots(len(fillers), len(TAUS), figsize=(9, 3.4 * len(fillers)), squeeze=False, sharey=True)
    ls = np.array(PATCH_LAYERS)
    for i, b in enumerate(fillers):
        for j, tau in enumerate(TAUS):
            ax, r = axes[i, j], res[b][tau]["Pa"]
            style(ax)
            ax.axhline(0, color=MUTED, linewidth=0.8)
            ax.axhline(1, color=MUTED, linewidth=0.8, linestyle=":")
            for key, col, lab in (("fwd", BLUE, "wrong → own (P-a)"), ("rev", ORANGE, "own → wrong (reverse)")):
                ax.fill_between(ls, r[key]["lo"], r[key]["hi"], color=col, alpha=0.18, linewidth=0)
                ax.plot(ls, r[key]["F"], color=col, linewidth=2, marker="o", markersize=4, label=lab)
            rho = r["rho"]
            ax.set_title(f"{b}, τ = {tau} (d = {32 - tau}):  ρ = {rho['rho']:+.2f} [{rho['lo']:+.2f}, {rho['hi']:+.2f}]",
                         fontsize=9, color=INK)
            ax.set_xticks(ls)
            ax.set_xlabel("patched layer l (x_l(T) replaced)", fontsize=8, color=INK)
            if j == 0:
                ax.set_ylabel("F(l): fraction of own − wrong\nlate BOS moved (layers max(l,6)–11)", fontsize=8, color=INK)
            if i == 0 and j == 0:
                ax.legend(fontsize=8, frameon=False)
    fig.suptitle("Patching the final token's stream: fraction of the late-BOS effect transferred (95% row-bootstrap CI)",
                 fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_layers(res: dict, fillers: list[str], path: Path) -> None:
    cols = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN]
    fig, axes = plt.subplots(len(fillers), len(TAUS), figsize=(9, 3.4 * len(fillers)), squeeze=False, sharey=True)
    for i, b in enumerate(fillers):
        for j, tau in enumerate(TAUS):
            ax = axes[i, j]
            style(ax)
            ax.axhline(0, color=MUTED, linewidth=0.8)
            ax.axhline(1, color=MUTED, linewidth=0.8, linestyle=":")
            for k, m in enumerate(range(LATE[0], L)):
                F = res[b][tau]["Pa_layers"][m]["F"][:m]  # l = 1..m (beyond m it is 0 by causality)
                ax.plot(np.arange(1, m + 1), F, color=cols[k], linewidth=2, marker="o", markersize=3.5, label=f"layer {m}")
            ax.set_title(f"{b}, τ = {tau}", fontsize=9, color=INK)
            ax.set_xticks(PATCH_LAYERS)
            ax.set_xlabel("patched layer l", fontsize=8, color=INK)
            if j == 0:
                ax.set_ylabel("F_m(l), BOS attention of layer m", fontsize=8, color=INK)
            if i == 0 and j == len(TAUS) - 1:
                ax.legend(fontsize=7, frameon=False, ncol=2)
    fig.suptitle("Per measured layer m: fraction of its own − wrong BOS attention moved by patching x_l(T)",
                 fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_heads(res: dict, fillers: list[str], path: Path) -> None:
    keys = list(HEAD_SETS)
    labels = ["all four"] + keys[1:]
    cols = [BLUE, ORANGE]
    fig, axes = plt.subplots(2, len(TAUS), figsize=(9, 6.4), squeeze=False, sharey=True)
    w = 0.38
    for i, direc in enumerate(("fwd", "rev")):
        for j, tau in enumerate(TAUS):
            ax = axes[i, j]
            style(ax)
            ax.axhline(0, color=MUTED, linewidth=0.8)
            if direc == "fwd" and tau == 16:
                ax.axhline(0.20, color=INK, linewidth=0.8, linestyle="--")
                ax.text(2, 0.21, "P-b threshold 0.20", fontsize=7, color=INK, ha="center", va="bottom")
            x = np.arange(len(keys))
            for k, b in enumerate(fillers):
                v = np.array([res[b][tau]["Pb"][h][direc]["F"][0] for h in keys])
                lo = np.array([res[b][tau]["Pb"][h][direc]["lo"][0] for h in keys])
                hi = np.array([res[b][tau]["Pb"][h][direc]["hi"][0] for h in keys])
                off = (k - (len(fillers) - 1) / 2) * w
                ax.bar(x + off, v, width=w - 0.04, color=cols[k], label=b)
                ax.errorbar(x + off, v, yerr=[v - lo, hi - v], fmt="none", ecolor=INK, elinewidth=0.9, capsize=2)
            ax.set_xticks(x, labels, fontsize=8)
            name = "wrong → own" if direc == "fwd" else "own → wrong (reverse)"
            ax.set_title(f"{name}, τ = {tau}", fontsize=9, color=INK)
            if j == 0:
                ax.set_ylabel("fraction of late BOS effect\n(layers 6–11) transferred", fontsize=8, color=INK)
            if i == 0 and j == 0:
                ax.legend(fontsize=8, frameon=False)
    fig.suptitle("Patching the same-country heads' writes at T (z at T): transfer with 95% row-bootstrap CI",
                 fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- report

def fmt(v: float, lo: float, hi: float, p: int = 3) -> str:
    return f"{v:+.{p}f} [{lo:+.{p}f}, {hi:+.{p}f}]"


def verdicts(res: dict, fillers: list[str]) -> list[dict]:
    pa = [res[b][t]["Pa"]["rho"] for b in fillers for t in TAUS]
    pa_ok = all(r["rho"] > 0 and r["lo"] > 0 for r in pa)
    pa_short = "; ".join(f"{b} τ={t}: ρ {fmt(res[b][t]['Pa']['rho']['rho'], res[b][t]['Pa']['rho']['lo'], res[b][t]['Pa']['rho']['hi'], 2)}"
                         for b in fillers for t in TAUS)
    pb = {b: res[b][16]["Pb"]["heads"]["fwd"] for b in fillers}
    pb_ok = all(pb[b]["F"][0] >= 0.20 for b in fillers)
    pb_short = "; ".join(f"{b}: {fmt(pb[b]['F'][0], pb[b]['lo'][0], pb[b]['hi'][0])}" for b in fillers)
    return [{"id": "P-a", "verdict": "PASS" if pa_ok else "FAIL", "short": pa_short},
            {"id": "P-b", "verdict": "PASS" if pb_ok else "FAIL", "short": f"four-head transfer at τ = 16: {pb_short} (threshold 0.20 on the point estimate)"}]


def write_report(run: Path, res: dict, checks: dict, preds: list[dict], fillers: list[str], n_rows: int,
                 secs: float, capitals: list[str]) -> None:
    Lr = ["# Experiment 5, analysis P: activation patching", "",
          f"Run `{run.name}`, {n_rows} rows per filler type × 11 capital-paired cells, τ ∈ {{16, 1}}, {secs / 60:.1f} min on "
          "CPU. Hypotheses: `PREDICTIONS.md` (P-a, P-b); operational choices: `PREDICTIONS_P.md` (both copied here). "
          f"All fractions are ratios of row means ({n_rows} rows; each row first averaged over its 11 cells), CIs are 95% "
          f"row-bootstrap percentile intervals ({BOOT} resamples, seed {BOOT_SEED}).", "",
          "## Checks", "", "| check | result |", "|---|---|"] + [f"| {k} | {v} |" for k, v in checks.items()]
    Lr += ["", "## Predictions", "", "| id | verdict | numbers |", "|---|---|---|"]
    Lr += [f"| {p['id']} | {p['verdict']} | {p['short']} |" for p in preds]
    for b in fillers:
        for tau in TAUS:
            r = res[b][tau]
            om = r["own_minus_wrong"]
            Lr += ["", f"## {b}, τ = {tau} (d = {32 - tau})", "",
                   f"Unpatched late BOS (layers 6–11): own {om['own']:.4f}, wrong {om['wrong']:.4f}, own − wrong "
                   f"{fmt(om['mean'], om['lo'], om['hi'], 4)} (Exp 4b at this distance: {EXP4B[tau][b]:+.4f}).", "",
                   "**P-a.** F(l) = (patched − own) / (wrong − own) on layers max(l, 6)–11; reverse F_rev(l) = (patched − "
                   "wrong) / (own − wrong). den = mean over rows of wrong − own on those layers.", "",
                   "| l | layers | F(l) | den (wrong − own) | F_rev(l) | F − F_rev |", "|---|---|---|---|---|---|"]
            f, rv = r["Pa"]["fwd"], r["Pa"]["rev"]
            for i, l in enumerate(PATCH_LAYERS):
                flag = "" if (f["den_lo"][i] > 0 or f["den_hi"][i] < 0) else " (den CI includes 0)"
                Lr.append(f"| {l} | {max(l, LATE[0])}–11 | {fmt(f['F'][i], f['lo'][i], f['hi'][i])} | "
                          f"{fmt(f['den'][i], f['den_lo'][i], f['den_hi'][i], 4)}{flag} | {fmt(rv['F'][i], rv['lo'][i], rv['hi'][i])} | "
                          f"{f['F'][i] - rv['F'][i]:+.3f} |")
            rho, rr = r["Pa"]["rho"], r["Pa"]["rho_rev"]
            Lr += ["", f"Spearman(l, F(l)) = {fmt(rho['rho'], rho['lo'], rho['hi'], 2)}; reverse "
                   f"{fmt(rr['rho'], rr['lo'], rr['hi'], 2)}.", "",
                   "Per measured layer m (descriptive): F_m(l) for l = 1..m (0 for l > m by causality).", "",
                   "| m | den (wrong − own) | " + " | ".join(f"l={l}" for l in PATCH_LAYERS) + " |",
                   "|---|---|" + "---|" * len(PATCH_LAYERS)]
            for m in range(LATE[0], L):
                pm = r["Pa_layers"][m]
                Lr.append(f"| {m} | {fmt(pm['den'][0], pm['den_lo'][0], pm['den_hi'][0], 4)} | "
                          + " | ".join(f"{pm['F'][i]:+.2f}" if l <= m else "0" for i, l in enumerate(PATCH_LAYERS)) + " |")
            Lr += ["", "**P-b and per-head (layers 6–11).**", "", "| heads patched (z at T) | wrong → own | own → wrong |",
                   "|---|---|---|"]
            for k in HEAD_SETS:
                hf, hr = r["Pb"][k]["fwd"], r["Pb"][k]["rev"]
                Lr.append(f"| {'all four' if k == 'heads' else k} | {fmt(hf['F'][0], hf['lo'][0], hf['hi'][0])} | "
                          f"{fmt(hr['F'][0], hr['lo'][0], hr['hi'][0])} |")
            hd = r["head_delta"]
            top = np.argsort(hd[LATE[0]:].ravel())[:5]
            Lr += ["", "Late heads with the largest drop in BOS attention (own − wrong): " + ", ".join(
                f"L{LATE[0] + t // H}H{t % H} {hd[LATE[0]:].ravel()[t]:+.3f}" for t in top) + "."]
    Lr += ["", "## Per pair (descriptive, τ = 16)", "",
           "Late BOS own − wrong and the four-head transfer G within each cell (ratio of row means over the cell's rows); "
           "F at l = 6 likewise.", "", "| capital at T (own vs wrong country) | " + " | ".join(f"{b}: own − wrong | {b}: G heads | {b}: F(6)" for b in fillers) + " |",
           "|---|" + "---|---|---|" * len(fillers)]
    for k, cap in enumerate(capitals):
        Lr.append(f"| {cap} | " + " | ".join(
            f"{res[b][16]['pairs']['own_minus_wrong'][k]:+.4f} | {res[b][16]['pairs']['G_heads'][k]:+.2f} | "
            f"{res[b][16]['pairs']['F_l6'][k]:+.2f}" for b in fillers) + " |")
    Lr += ["", "## Caveats", "",
           "- The same 11 country–capital pairs (and 11 fixed wrong pairings) throughout: row-bootstrap CIs mean consistent "
           "over contexts, not general over pairs. The per-pair table shows how much the pairs vary.",
           "- Patching x_l(T) leaves positions τ..T−1 different, so F(l) is what T's stream carries relative to what later "
           "layers can still read from those positions; F(l) for l > 6 is measured on fewer layers (max(l, 6)–11), as "
           "pre-registered, so the curve mixes a change of site with a change of measured layers.",
           "- P-b patches the four heads' z (equivalently their writes, `hook_result`) at T only; their writes at positions "
           "< T still differ between the runs. Heads alone need not add up to the four together (interactions).",
           "- Fractions are ratios of row means; a fraction outside [0, 1] means overshoot (patched beyond the other run) or "
           "a push in the opposite direction, not an error.",
           "- TransformerLens with default weight processing; BOS attention is unaffected by the processing (checked in "
           "Exp 4/4b against Hugging Face), and no HF cross-check of the patched runs was made here."]
    Lr += ["", "## Files", "",
           "- `p_results.json`: every number above.",
           "- `p_arrays.npz`: `bos_<filler>_<tau>` [rows, 11 cells, runs, 12 layers] head-mean BOS attention at T (float32), "
           "run names in `runs`; `heads_<filler>_<tau>` [rows, 2 (own, wrong), 12, 12] per-head BOS attention, mean over cells.",
           "- `figures/F_curves.png`, `figures/F_by_layer.png`, `figures/heads_transfer.png`."]
    (run / "p_report.md").write_text("\n".join(Lr) + "\n")


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items() if k != "boot"}
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


# ---------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Experiment 5, analysis P: activation patching")
    parser.add_argument("--config", default="configs/exp5_residual.yaml", type=Path)
    parser.add_argument("--rows", default=None, type=int, help="use only the first rows (smoke test)")
    parser.add_argument("--results-dir", default=None, type=Path, help="results root (smoke tests)")
    args = parser.parse_args(argv)
    cfg_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    cfg = yaml.safe_load(cfg_path.read_text())
    if args.rows:
        cfg["n_rows"] = args.rows
    t0 = time.time()
    run = make_run_dir(args.results_dir or PROJECT_ROOT / "results", tag="exp5P")
    save_config_copy(cfg_path, run)
    for f in ("PREDICTIONS.md", "PREDICTIONS_P.md"):
        shutil.copy2(PROJECT_ROOT / "exp5" / f, run / f)
    save_versions(run, PROJECT_ROOT, code_dirs=("exp1", "exp2", "exp3", "exp4", "exp5"))
    (run / "figures").mkdir()
    log(f"run folder {run}", t0)

    from transformers import AutoTokenizer

    set_determinism(cfg["seed"])
    tl = load_model("transformer_lens", cfg["backends"]["transformer_lens"]["model_name"], cfg["device"], cfg["dtype"])
    countries, capitals, cities = stimuli(cfg, AutoTokenizer.from_pretrained("gpt2"))
    stim_ids = np.concatenate([countries, capitals, cities, cfg["demonstration_ids"]])
    cl = cells(countries, capitals, cfg["wrong_capital_seed"])
    T, bos_id, atol = cfg["seq_len"], cfg["bos_token_id"], cfg["checks"]["row_sum_atol"]
    n = len(capitals)
    acc = {"rowsum": 0.0, "causal": 0.0, "null0": 0.0, "identity": 0.0, "l0": 0.0}
    det: dict[str, float] = {}
    res: dict = {}
    arrays: dict[str, np.ndarray] = {"runs": np.array(RUNS)}
    fillers = list(cfg["bases"])
    for b in fillers:
        block = cached_base_block(f"exp3rel_{b}", cfg["bases"][b], stim_ids, cfg["base_cache_n"], T, cfg["seed"],
                                  bos_id, PROJECT_ROOT / cfg["paths"]["sequences_dir"])
        bases = block[: cfg["n_rows"]]
        if np.isin(bases, stim_ids).any():
            raise AssertionError(f"{b}: a stimulus or demonstration id occurs in a base row")
        R = len(bases)
        res[b] = {}
        for tau in TAUS:
            own_s, wrong_s = sequences(bases, cl, tau, bos_id)
            bl = np.zeros((R, n, len(RUNS), L), np.float32)
            heads = np.zeros((R, 2, L, H), np.float64)
            first = {}
            for r0 in range(0, R, BLOCK_ROWS):
                nr = min(BLOCK_ROWS, R - r0)
                out, ck = run_block(tl, to_x(own_s[r0:r0 + nr], bos_id), to_x(wrong_s[r0:r0 + nr], bos_id), atol,
                                    identity=(r0 == 0))
                for k in acc:
                    acc[k] = max(acc[k], ck[k])
                for name, v in out.items():
                    bl[r0:r0 + nr, :, RI[name]] = v.reshape(nr, n, L, H).mean(axis=-1)
                heads[r0:r0 + nr, 0] = out["own"].reshape(nr, n, L, H).mean(axis=1)
                heads[r0:r0 + nr, 1] = out["wrong"].reshape(nr, n, L, H).mean(axis=1)
                if r0 == 0:
                    first = {k: out[k][: DET_ROWS * n] for k in ("own", "wrong", "fwd_l6", "fwd_heads")}
                log(f"{b} tau={tau}: rows {r0 + nr}/{R}", t0)
            if tau == DET_TAU:
                again, _ = run_block(tl, to_x(own_s[:DET_ROWS], bos_id), to_x(wrong_s[:DET_ROWS], bos_id), atol,
                                     identity=False, only=("fwd_l6", "fwd_heads"))
                det[b] = max(float(np.abs(again[k] - first[k]).max()) for k in first)
            res[b][tau] = analyse(bl.astype(np.float64), heads[:, 0], heads[:, 1])
            arrays[f"bos_{b}_{tau}"] = bl
            arrays[f"heads_{b}_{tau}"] = heads.astype(np.float32)

    checks: dict[str, str] = {
        "design (check_variants on own and wrong; final token = capital k; one differing position)": "PASS (asserted)",
        "attention rows at T sum to 1, every run": f"PASS (max |sum − 1| {acc['rowsum']:.1e} ≤ {atol:g}, asserted)",
        "built-in null: own vs wrong x_0(T)": f"{'PASS' if acc['null0'] == 0 else 'FAIL'} (max |Δ| {acc['null0']:.1e})",
        "l = 0 patch leaves every BOS attention unchanged": f"{'PASS' if acc['l0'] == 0 else 'FAIL'} (max |Δ| {acc['l0']:.1e})",
        "identity patch (own's own x_l(T), l = 1..11, and own z of the 4 heads; first block of each filler × τ)":
            f"{'PASS' if acc['identity'] == 0 else 'FAIL'} (max |Δ| {acc['identity']:.1e})",
        "causal locality: layers before the patched layer unchanged": f"{'PASS' if acc['causal'] == 0 else 'FAIL'} (max |Δ| {acc['causal']:.1e})",
    }
    for b in fillers:
        v = det[b]
        checks[f"determinism, rows 0–1 recomputed alone (τ = {DET_TAU}), {b}"] = (
            f"{'PASS' if v <= 1e-6 else 'FAIL'} (max |Δ| {v:.1e}{'' if v == 0 else '; batch-layout rounding'})")
    for b in fillers:
        for tau in TAUS:
            om = res[b][tau]["own_minus_wrong"]
            ref = EXP4B[tau][b]
            ok = om["hi"] < 0 and 0.5 <= om["mean"] / ref <= 2
            checks[f"unpatched own − wrong late BOS < 0 and within 2× of Exp 4b, {b} τ = {tau}"] = (
                f"{'PASS' if ok else 'FAIL'} ({fmt(om['mean'], om['lo'], om['hi'], 4)} vs Exp 4b {ref:+.4f}; ratio "
                f"{om['mean'] / ref:.2f})")
    preds = verdicts(res, fillers)
    np.savez_compressed(run / "p_arrays.npz", **arrays)
    (run / "p_results.json").write_text(json.dumps(jsonable({"checks": checks, "predictions": preds,
                                                            "results": res}), indent=1))
    plot_F(res, fillers, run / "figures" / "F_curves.png")
    plot_layers(res, fillers, run / "figures" / "F_by_layer.png")
    plot_heads(res, fillers, run / "figures" / "heads_transfer.png")
    names = [s["country"].strip() for s in cfg["stimuli"]]
    pair_labels = [f"{s['capital'].strip()} ({names[k]} vs {names[cl['inv'][k]]})" for k, s in enumerate(cfg["stimuli"])]
    write_report(run, res, checks, preds, fillers, cfg["n_rows"], time.time() - t0, pair_labels)
    (run / "DONE").write_text(f"{time.time() - t0:.0f} s\n")
    log(f"done: {run}", t0)
    for k, v in checks.items():
        print(f"{k}: {v}")
    for p in preds:
        print(f"{p['id']}: {p['verdict']} — {p['short']}")
    return run


if __name__ == "__main__":
    main()
