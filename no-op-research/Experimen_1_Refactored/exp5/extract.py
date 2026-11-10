"""Experiment 5: residual-stream extraction for the topology analyses R1–R5 and P (exp5/PREDICTIONS.md).

Run:  uv run --offline python -m exp5.extract --config configs/exp5_residual.yaml [--rows N] [--results-dir DIR]

Capital-paired cells: capital k at T (= 32) in both conditions; own = country k at tau, wrong = the country whose
deranged capital is k (Exp 4b derangement), control = the row's own token at tau with capital k at T (shared by own
and wrong). random = country k at tau with the row's own final token (Exp 4's condition), control = the base row.
Everything is read at the final token T. Writes results/<timestamp>_exp5res/ (per filler type b):

  seq_<b>.npz     tau in seq_taus, cond = (own, wrong, random), cell k = 0..10:
                  resid [R, 3, 3, n, 13, 768] fp16 (x_l(T) for l = 0..11 from hook_resid_pre, then
                  blocks.11.hook_resid_post), z [R, 3, 3, n, 12, 12, 64] fp16 (hook_z at T; write = z @ W_O[l, i]),
                  att [R, 3, 3, n, 12, 12, 3] fp16 (attention from T to BOS, tau, T - 1);
                  ctrl_resid [R, n, 13, 768], ctrl_z [R, n, 12, 12, 64], ctrl_att_row [R, n, 12, 12, 33];
                  rnd_ctrl_resid [R, 13, 768], rnd_ctrl_z [R, 12, 12, 64], rnd_ctrl_att_row [R, 12, 12, 33].
  means_<b>.npz   every tau, indexed by distance d - 1 (tau = 32 - d): resid [R, 3, 31, 13, 768] fp16, the mean over
                  the row's 11 cells; ctrl_resid [R, 13, 768] (mean of the 11 capital controls); rnd_ctrl_resid [R, 13, 768].
  graphs_<b>.npz  tau in graph_taus: {raw,vw}_{own,wrong,random} [R, 7, 12, 33, 33] fp16, per-row means over cells of
                  the layer graph (raw = mean over heads of attention; vw = sum over heads of attention x ||v_k W_O^h||);
                  {raw,vw}_control and {raw,vw}_rnd_control [R, 12, 33, 33]; head_{raw,vw}_{cond} [7, 12, 12, 33, 33]
                  float32 and head_{raw,vw}_{control,rnd_control} [12, 12, 33, 33], means over rows and cells.
  weights.npz     processed W_O [12, 12, 64, 768] and b_O [12, 768], countries, capitals, wrong countries per cell.
  checks.json, extract_report.md, DONE (written last).

TransformerLens's default weight processing matters for these vectors (not for attention): centred writing weights
make every stream state and write mean-free (= HF's minus its mean), and folded biases mean z and the writes exclude
each head's constant (β_ln1 W_V + b_V) W_O (moved into b_O), as do the value norms in the vw graphs. The HF check
compares on those terms.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from exp1.backends import check_rows_sum_to_one, load_model, set_determinism
from exp1.runlog import make_run_dir, save_config_copy, save_versions
from exp1.sequences import prepend_bos
from exp2.design import cached_base_block
from exp3.run_positions import latin_derangements
from exp3.run_relations import PROJECT_ROOT, log, stimuli
from exp4.run_controlled import check_variants, variants, with_final

CONDS = ("own", "wrong", "random")
L, H, DH, DM, NS = 12, 12, 64, 768, 13  # layers, heads, head dim, model dim, stream states (12 pre + final post)


def cells(countries: np.ndarray, capitals: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    """Per cell k: the country at tau for own (k) and wrong (the c whose deranged capital is k), and the capital at T."""
    wrong = latin_derangements(len(countries), 1, seed)[0]  # country c -> capital wrong[c], as in Exp 4b
    if (wrong == np.arange(len(countries))).any():
        raise AssertionError("the derangement has a fixed point")
    inv = np.argsort(wrong)  # inv[k] = c with wrong[c] = k
    return {"own": countries, "wrong": countries[inv], "capital": capitals, "inv": inv}


def hook_filter(want_z: bool, want_graphs: bool):
    def keep(name: str) -> bool:
        if name.endswith("hook_resid_pre") or name == "blocks.11.hook_resid_post":
            return True
        if name.endswith("attn.hook_pattern"):
            return want_z or want_graphs
        if name.endswith("attn.hook_z"):
            return want_z
        if name.endswith("attn.hook_v"):
            return want_graphs
        return False
    return keep


def forward(model, seqs: np.ndarray, cfg: dict, tau: int | None, want_z: bool, want_graphs: bool,
            gram: torch.Tensor) -> dict[str, Any]:
    """seqs [N, T] (no BOS). Returns stream states at T [N, 13, 768]; with want_z, z at T [N, 12, 12, 64] and the
    attention from T (full rows [N, 12, 12, 33] when tau is None, else [N, 12, 12, 3] on BOS, tau, T - 1); with
    want_graphs, layer graphs raw / vw [N, 12, 33, 33] and the per-head sums over the N sequences [12, 12, 33, 33]."""
    T, bs = cfg["seq_len"], cfg["batch_size"]
    keep = hook_filter(want_z, want_graphs)
    out: dict[str, list] = {"resid": [], "z": [], "att": [], "raw": [], "vw": []}
    head_raw = head_vw = None
    with torch.inference_mode():
        for s in range(0, len(seqs), bs):
            x = torch.as_tensor(prepend_bos(seqs[s:s + bs], cfg["bos_token_id"]), dtype=torch.long)
            _, cache = model.run_with_cache(x, names_filter=keep, return_type=None)
            resid = [cache[f"blocks.{l}.hook_resid_pre"][:, -1] for l in range(L)] + [cache["blocks.11.hook_resid_post"][:, -1]]
            out["resid"].append(torch.stack(resid, dim=1).float().numpy())
            if not (want_z or want_graphs):
                continue
            pat = torch.stack([cache[f"blocks.{l}.attn.hook_pattern"] for l in range(L)], dim=1)  # [b, L, H, q, k]
            check_rows_sum_to_one(pat[:, :, :, -1, :].float().numpy(), cfg["checks"]["row_sum_atol"])
            if want_z:
                out["z"].append(torch.stack([cache[f"blocks.{l}.attn.hook_z"][:, -1] for l in range(L)], dim=1).numpy())
                row = pat[:, :, :, -1, :]
                att = row if tau is None else torch.stack([row[..., 0], row[..., tau], row[..., T - 1]], dim=-1)
                out["att"].append(att.numpy())
            if want_graphs:
                v = torch.stack([cache[f"blocks.{l}.attn.hook_v"] for l in range(L)], dim=1)  # [b, L, pos, H, DH]
                # ||v_k W_O^h|| via the Gram matrix W_O^h W_O^h^T, without building the [b, L, pos, H, 768] writes
                vnorm = torch.einsum("blphd,lhde,blphe->blhp", v, gram, v).clamp_min(0).sqrt()
                vw = pat * vnorm[:, :, :, None, :]
                out["raw"].append(pat.mean(dim=2).numpy())
                out["vw"].append(vw.sum(dim=2).numpy())
                hr, hv = pat.sum(dim=0).numpy(), vw.sum(dim=0).numpy()
                head_raw = hr if head_raw is None else head_raw + hr
                head_vw = hv if head_vw is None else head_vw + hv
    res = {k: np.concatenate(v) for k, v in out.items() if v}
    if want_graphs:
        res["head_raw_sum"], res["head_vw_sum"] = head_raw, head_vw
    return res


def extract_filler(model, bases: np.ndarray, cl: dict, cfg: dict, gram: torch.Tensor, t0: float, b: str) -> dict:
    T, n, R = cfg["seq_len"], len(cl["capital"]), len(bases)
    st, gt = cfg["seq_taus"], cfg["graph_taus"]
    f16 = np.float16
    seq = {"resid": np.zeros((R, 3, len(st), n, NS, DM), f16), "z": np.zeros((R, 3, len(st), n, L, H, DH), f16),
           "att": np.zeros((R, 3, len(st), n, L, H, 3), f16),
           "ctrl_resid": np.zeros((R, n, NS, DM), f16), "ctrl_z": np.zeros((R, n, L, H, DH), f16),
           "ctrl_att_row": np.zeros((R, n, L, H, T + 1), f16), "rnd_ctrl_resid": np.zeros((R, NS, DM), f16),
           "rnd_ctrl_z": np.zeros((R, L, H, DH), f16), "rnd_ctrl_att_row": np.zeros((R, L, H, T + 1), f16)}
    means = {"resid": np.zeros((R, 3, T - 1, NS, DM), f16), "ctrl_resid": np.zeros((R, NS, DM), f16),
             "rnd_ctrl_resid": np.zeros((R, NS, DM), f16)}
    graphs: dict[str, np.ndarray] = {}
    for kind in ("raw", "vw"):
        for c in CONDS:
            graphs[f"{kind}_{c}"] = np.zeros((R, len(gt), L, T + 1, T + 1), f16)
            graphs[f"head_{kind}_{c}"] = np.zeros((len(gt), L, H, T + 1, T + 1), np.float64)
        for c in ("control", "rnd_control"):
            graphs[f"{kind}_{c}"] = np.zeros((R, L, T + 1, T + 1), f16)
            graphs[f"head_{kind}_{c}"] = np.zeros((L, H, T + 1, T + 1), np.float64)
    block = max(1, cfg["batch_size"] // n)
    for r0 in range(0, R, block):
        rb = bases[r0:r0 + block]
        nr = len(rb)
        sl = slice(r0, r0 + nr)
        cap = np.tile(cl["capital"], (nr, 1))
        fills = {"own": (np.tile(cl["own"], (nr, 1)), cap), "wrong": (np.tile(cl["wrong"], (nr, 1)), cap),
                 "random": (np.tile(cl["own"], (nr, 1)), None)}
        f = forward(model, with_final(rb, n, cap).reshape(-1, T), cfg, None, True, True, gram)  # capital controls
        seq["ctrl_resid"][sl] = f["resid"].reshape(nr, n, NS, DM)
        seq["ctrl_z"][sl] = f["z"].reshape(nr, n, L, H, DH)
        seq["ctrl_att_row"][sl] = f["att"].reshape(nr, n, L, H, T + 1)
        means["ctrl_resid"][sl] = f["resid"].reshape(nr, n, NS, DM).mean(axis=1)
        for kind in ("raw", "vw"):
            graphs[f"{kind}_control"][sl] = f[kind].reshape(nr, n, L, T + 1, T + 1).mean(axis=1)
            graphs[f"head_{kind}_control"] += f[f"head_{kind}_sum"]
        f = forward(model, rb, cfg, None, True, True, gram)  # base rows: the random-final-token control
        seq["rnd_ctrl_resid"][sl], seq["rnd_ctrl_z"][sl], seq["rnd_ctrl_att_row"][sl] = f["resid"], f["z"], f["att"]
        means["rnd_ctrl_resid"][sl] = f["resid"]
        for kind in ("raw", "vw"):
            graphs[f"{kind}_rnd_control"][sl] = f[kind]
            graphs[f"head_{kind}_rnd_control"] += f[f"head_{kind}_sum"]
        for ci, cond in enumerate(CONDS):
            fill, final = fills[cond]
            for tau in range(1, T):
                seqs = variants(rb, fill, tau, final)
                check_variants(seqs, rb, fill, tau, cfg["bos_token_id"], final)
                want_z, want_g = tau in st, tau in gt
                f = forward(model, seqs.reshape(-1, T), cfg, tau, want_z, want_g, gram)
                resid = f["resid"].reshape(nr, n, NS, DM)
                means["resid"][sl, ci, T - tau - 1] = resid.mean(axis=1)
                if want_z:
                    j = st.index(tau)
                    seq["resid"][sl, ci, j] = resid
                    seq["z"][sl, ci, j] = f["z"].reshape(nr, n, L, H, DH)
                    seq["att"][sl, ci, j] = f["att"].reshape(nr, n, L, H, 3)
                if want_g:
                    j = gt.index(tau)
                    for kind in ("raw", "vw"):
                        graphs[f"{kind}_{cond}"][sl, j] = f[kind].reshape(nr, n, L, T + 1, T + 1).mean(axis=1)
                        graphs[f"head_{kind}_{cond}"][j] += f[f"head_{kind}_sum"]
        log(f"{b}: rows {r0 + nr}/{R}", t0)
    for kind in ("raw", "vw"):
        for c in CONDS:
            graphs[f"head_{kind}_{c}"] = (graphs[f"head_{kind}_{c}"] / (R * n)).astype(np.float32)
        graphs[f"head_{kind}_control"] = (graphs[f"head_{kind}_control"] / (R * n)).astype(np.float32)
        graphs[f"head_{kind}_rnd_control"] = (graphs[f"head_{kind}_rnd_control"] / R).astype(np.float32)
    return {"seq": seq, "means": means, "graphs": graphs}


def hf_check(tl, bases: np.ndarray, cl: dict, cfg: dict) -> dict[str, float]:
    """Plain Hugging Face GPT-2 on a few sequences: stream states, the final LayerNorm output, z and per-head writes."""
    T, k, tau = cfg["seq_len"], cfg["checks"]["hf_rows"], 16
    hf = load_model("huggingface", cfg["backends"]["huggingface"]["model_name"], cfg["device"], cfg["dtype"],
                    cfg["backends"]["huggingface"]["attn_implementation"])
    rows = bases[:k]
    seqs = variants(rows, np.tile(cl["own"], (k, 1)), tau, np.tile(cl["capital"], (k, 1)))[:, :4].reshape(-1, T)
    x = torch.as_tensor(prepend_bos(seqs, cfg["bos_token_id"]), dtype=torch.long)
    cproj_in: dict[int, torch.Tensor] = {}
    hooks = [hf.h[l].attn.c_proj.register_forward_hook(lambda m, i, o, l=l: cproj_in.__setitem__(l, i[0].detach()))
             for l in range(L)]
    rel = lambda a, b: float((a - b).abs().max() / b.abs().max())
    centre = lambda a: a - a.mean(dim=-1, keepdim=True)
    with torch.inference_mode():  # the comparisons too: weights and inference tensors can't mix under autograd
        hs = hf(input_ids=x, output_hidden_states=True).hidden_states
        _, cache = tl.run_with_cache(x, return_type=None)
        for h in hooks:
            h.remove()
        out = {"stream (l <= 11, centred)": max(rel(cache[f"blocks.{l}.hook_resid_pre"][:, -1], centre(hs[l][:, -1]))
                                                 for l in range(L))}
        ln_f = hf.ln_f
        out["final LayerNorm output"] = rel(cache["ln_final.hook_normalized"][:, -1] * ln_f.weight + ln_f.bias,
                                            hs[L][:, -1])
        zmax = wmax = 0.0
        for l in range(L):
            # TL folds ln_1's bias into the value bias (fold_ln), then the whole value bias into b_O (attention rows
            # sum to 1, so it adds the same vector whatever is attended); so TL's z = HF's z − (β W_V + b_V)
            attn = hf.h[l].attn
            b_v = (hf.h[l].ln_1.bias @ attn.c_attn.weight[:, 2 * DM:] + attn.c_attn.bias[2 * DM:]).reshape(H, DH)
            hf_z = cproj_in[l][:, -1].reshape(-1, H, DH) - b_v
            zmax = max(zmax, rel(cache[f"blocks.{l}.attn.hook_z"][:, -1], hf_z))
            w = hf.h[l].attn.c_proj.weight.reshape(H, DH, DM)  # Conv1D: x @ weight, weight [in, out]
            hf_write = centre(torch.einsum("bhd,hdm->bhm", hf_z, w))
            tl_write = torch.einsum("bhd,hdm->bhm", cache[f"blocks.{l}.attn.hook_z"][:, -1], tl.W_O[l])
            wmax = max(wmax, rel(tl_write, hf_write))
    out["z (HF minus value bias)"], out["per-head writes (centred, without the value bias)"] = zmax, wmax
    return out


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Experiment 5, residual-stream extraction")
    parser.add_argument("--config", default="configs/exp5_residual.yaml", type=Path)
    parser.add_argument("--rows", default=None, type=int, help="use only the first rows (smoke test)")
    parser.add_argument("--results-dir", default=None, type=Path, help="override paths.results_dir (smoke tests)")
    args = parser.parse_args(argv)
    cfg_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    cfg = yaml.safe_load(cfg_path.read_text())
    if args.rows:
        cfg["n_rows"] = args.rows
    t0 = time.time()
    run = make_run_dir(args.results_dir or PROJECT_ROOT / cfg["paths"]["results_dir"], tag="exp5res")
    save_config_copy(cfg_path, run)
    shutil.copy2(PROJECT_ROOT / "exp5" / "PREDICTIONS.md", run / "PREDICTIONS.md")
    save_versions(run, PROJECT_ROOT, code_dirs=("exp1", "exp2", "exp3", "exp4", "exp5"))
    log(f"run folder {run}", t0)

    from transformers import AutoTokenizer

    set_determinism(cfg["seed"])
    tl = load_model("transformer_lens", cfg["backends"]["transformer_lens"]["model_name"], cfg["device"], cfg["dtype"])
    countries, capitals, cities = stimuli(cfg, AutoTokenizer.from_pretrained("gpt2"))
    stim_ids = np.concatenate([countries, capitals, cities, cfg["demonstration_ids"]])
    cl = cells(countries, capitals, cfg["wrong_capital_seed"])
    gram = torch.einsum("lhdm,lhem->lhde", tl.W_O, tl.W_O).detach()
    np.savez(run / "weights.npz", W_O=tl.W_O.detach().numpy(), b_O=tl.b_O.detach().numpy(), countries=countries,
             capitals=capitals, wrong_country_per_cell=cl["wrong"], inv=cl["inv"])

    checks: dict[str, str] = {"design (variants equal their control except at tau; no repeats/BOS)": "PASS (asserted)",
                              "attention rows sum to 1": f"PASS (atol {cfg['checks']['row_sum_atol']:g}, asserted)"}
    bases_all = {}
    for b, rng_ in cfg["bases"].items():
        block = cached_base_block(f"exp3rel_{b}", rng_, stim_ids, cfg["base_cache_n"], cfg["seq_len"], cfg["seed"],
                                  cfg["bos_token_id"], PROJECT_ROOT / cfg["paths"]["sequences_dir"])
        bases = block[: cfg["n_rows"]]
        if np.isin(bases, stim_ids).any():
            raise AssertionError(f"{b}: a stimulus or demonstration id occurs in a base row")
        bases_all[b] = bases
        ex = extract_filler(tl, bases, cl, cfg, gram, t0, b)
        s = ex["seq"]
        null0 = float(np.abs(s["resid"][:, 0, :, :, 0].astype(np.float32) - s["resid"][:, 1, :, :, 0].astype(np.float32)).max())
        checks[f"built-in null: own vs wrong x_0(T) identical, {b}"] = f"{'PASS' if null0 == 0 else 'FAIL'} (max |Δ| {null0:.1e})"
        r = cfg["checks"]["determinism_rows"]
        tau = cfg["seq_taus"][1]
        again = forward(tl, variants(bases[:r], np.tile(cl["own"], (r, 1)), tau, np.tile(cl["capital"], (r, 1))).reshape(-1, cfg["seq_len"]),
                        cfg, tau, True, False, gram)
        det = max(float(np.abs(again["resid"].astype(np.float16).reshape(r, -1, NS, DM) - s["resid"][:r, 0, 1]).max()),
                  float(np.abs(again["z"].astype(np.float16).reshape(r, -1, L, H, DH) - s["z"][:r, 0, 1]).max()))
        checks[f"determinism, {b}"] = f"{'PASS' if det == 0 else 'FAIL'} (max |Δ| {det:.1e})"
        np.savez(run / f"seq_{b}.npz", **s)
        np.savez(run / f"means_{b}.npz", **ex["means"])
        np.savez(run / f"graphs_{b}.npz", **ex["graphs"])
        log(f"{b}: saved", t0)
    hf = hf_check(tl, bases_all["common"], cl, cfg)
    for k, v in hf.items():
        checks[f"TL vs HF: {k}"] = f"{'PASS' if v < cfg['checks']['hf_rtol'] else 'FAIL'} (max rel |Δ| {v:.1e})"
    (run / "checks.json").write_text(json.dumps(checks, indent=2))
    lines = ["# Experiment 5: residual-stream extraction", "", f"Run `{run.name}`, {cfg['n_rows']} rows per filler type, "
             f"{time.time() - t0:.0f} s. Arrays and shapes: see the module docstring of `exp5/extract.py`.", "",
             "| check | result |", "|---|---|"] + [f"| {k} | {v} |" for k, v in checks.items()]
    (run / "extract_report.md").write_text("\n".join(lines) + "\n")
    (run / "DONE").write_text(f"{time.time() - t0:.0f} s\n")
    log(f"done: {run}", t0)
    for k, v in checks.items():
        print(f"{k}: {v}")
    return run


if __name__ == "__main__":
    main()
