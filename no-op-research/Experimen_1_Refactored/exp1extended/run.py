"""exp1extended entrypoint.   Run:  uv run python -m exp1extended.run --config configs/exp1extended.yaml --stage 0

Stages, run in order with the report read in between (the gates are decisions, not automatic):
  dryrun  time the forward pass and project the Stage 1-2 run times
  0       route each target head from the saved Exp 1 sequences (gate G0)
  1       paired position factorial + surface-matched rare pool + context dose-response (gates G1, G2)
  2       per-token sweep through the final position
Each run writes results/<YYYYMMDD-HHMMSS>_exp1extended_stage<stage>/.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from exp1.backends import bos_token_id, load_model, set_determinism
from exp1.runlog import make_run_dir, save_config_copy, save_versions
from exp1.sequences import prepend_bos
from exp1extended.config import load_config, target_heads
from exp1extended.extract import run_final

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGES = ("dryrun", "0", "1", "2")


def run_dryrun(cfg, run_dir: Path, model, root: Path) -> None:
    """Time run_final on the saved C_common_clean sequences and project the later stages' forward passes."""
    from exp1extended.stage0 import load_exp1_sequences

    tokens = prepend_bos(load_exp1_sequences(cfg, root, "C_common_clean"), cfg["bos_token_id"])
    start = time.perf_counter()
    run_final(model, tokens, cfg["batch_size"], target_heads(cfg), cfg["diffuse_threshold"])
    rate = len(tokens) / (time.perf_counter() - start)

    s1, s2 = cfg["stage1"], cfg["stage2"]
    n_tokens = (s2["tokens"]["all"][1] - s2["tokens"]["all"][0] + 1 + s2["tokens"]["rare_uniform"]
                + s2["tokens"]["rare_matched"] + s2["tokens"]["rare_transition"]["n"])
    stage1 = len(s1["cells"]) * s1["n_pairs"] + len(s1["dose"]["qp_arms"]) * len(s1["dose"]["levels"]) * s1["dose"]["n_pairs"]
    lines = ["# exp1extended dry run", "", f"- {len(tokens)} sequences in {len(tokens) / rate:.1f} s: **{rate:.0f} sequences/s** "
             f"(device {cfg['device']}, batch {cfg['batch_size']}).", "",
             "| stage | forward passes | projected minutes |", "|---|---|---|",
             f"| 0 (A, B, C re-run with q/k hooks) | {3 * len(tokens)} | {3 * len(tokens) / rate / 60:.1f} |",
             f"| 1 (cells + dose) | {stage1} | {stage1 / rate / 60:.1f} |"]
    for per_type in (8, s2["backgrounds"]["rare"]):  # half the configured backgrounds, and as configured
        n = n_tokens * 2 * per_type * len(s2["positions"])
        lines.append(f"| 2, {2 * per_type} backgrounds × {n_tokens} tokens × {len(s2['positions'])} position(s) | {n} | "
                     f"{n / rate / 60:.1f} |")
    (run_dir / "dryrun_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="exp1extended: which positions and tokens drive the head shifts")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=STAGES)
    args = parser.parse_args(argv)

    cfg = load_config(args.config, PROJECT_ROOT)
    run_dir = make_run_dir(PROJECT_ROOT / cfg["paths"]["results_dir"], tag=f"exp1extended_stage{args.stage}")
    save_config_copy(args.config, run_dir)
    save_versions(run_dir, PROJECT_ROOT, code_dirs=("exp1", "exp1extended"))
    (run_dir / "figures").mkdir()
    set_determinism(cfg["seed"])
    print(f"run folder: {run_dir}")

    model = load_model("transformer_lens", cfg["model"], cfg["device"], cfg["dtype"])
    if bos_token_id(model) != cfg["bos_token_id"]:
        raise ValueError(f"model BOS id {bos_token_id(model)} != config bos_token_id {cfg['bos_token_id']}")

    if args.stage == "dryrun":
        run_dryrun(cfg, run_dir, model, PROJECT_ROOT)
    elif args.stage == "0":
        from exp1extended.stage0 import run_stage0

        print(run_stage0(cfg, run_dir, model, PROJECT_ROOT).to_string(index=False))
    elif args.stage == "1":
        from exp1extended.stage1 import run_stage1

        run_stage1(cfg, run_dir, model, PROJECT_ROOT)
    else:
        from exp1extended.stage2 import run_stage2

        run_stage2(cfg, run_dir, model, PROJECT_ROOT)
    print(f"wrote {run_dir}")
    return run_dir


if __name__ == "__main__":
    main()
