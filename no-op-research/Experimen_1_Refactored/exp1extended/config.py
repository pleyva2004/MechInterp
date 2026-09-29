"""Load and sanity-check configs/exp1extended.yaml, the single source of this experiment's parameters."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from exp1.config import load_config as load_exp1_config
from exp1.phase2 import SHARED_KEYS

REQUIRED = (
    "model", "exp1_config", "exp1_runs", "seq_len", "bos_token_id", "seed", "diffuse_threshold", "batch_size",
    "device", "dtype", "target_heads", "primary_metrics", "pools", "matching", "stage0", "stage1", "stage2",
    "stats", "validation", "paths", "figures",
)
N_LAYERS = N_HEADS = 12  # GPT-2 small
METRIC_NAMES = ("sink_logit", "prev_logit", "other_gap", "entropy", "sink_mass", "prev_mass")


def load_config(path: str | Path, project_root: str | Path) -> dict[str, Any]:
    """The parsed config, after checks that would otherwise surface mid-run. Also verifies that every
    Exp 1 reference run was produced with the same shared settings as configs/exp1.yaml."""
    path, root = Path(path), Path(project_root)
    cfg = yaml.safe_load(path.read_text())

    missing = [key for key in REQUIRED if key not in cfg]
    if missing:
        raise ValueError(f"{path}: missing keys {missing}")

    seen: set[tuple[int, int]] = set()
    for group, heads in cfg["target_heads"].items():
        for layer, head in heads:
            if not (0 <= layer < N_LAYERS and 0 <= head < N_HEADS):
                raise ValueError(f"{path}: target head {[layer, head]} in {group} is outside {N_LAYERS}x{N_HEADS}")
            if (layer, head) in seen:
                raise ValueError(f"{path}: target head {[layer, head]} listed twice")
            seen.add((layer, head))
    if set(cfg["primary_metrics"]) != set(cfg["target_heads"]):
        raise ValueError(f"{path}: primary_metrics groups {sorted(cfg['primary_metrics'])} != target_heads groups "
                         f"{sorted(cfg['target_heads'])}")
    unknown = sorted({m for ms in cfg["primary_metrics"].values() for m in ms} - set(METRIC_NAMES))
    if unknown:
        raise ValueError(f"{path}: unknown primary metrics {unknown}; expected a subset of {METRIC_NAMES}")

    ranges = sorted((lo, hi, name) for name, (lo, hi) in cfg["pools"].items())
    for (lo_a, hi_a, a), (lo_b, _, b) in zip(ranges, ranges[1:]):
        if lo_b <= hi_a:
            raise ValueError(f"{path}: pools {a} and {b} overlap")

    for cell in cfg["stage1"]["cells"]:
        if set(cell) != {"Q", "P", "X"} or not set(cell.values()) <= {"r", "m", "c"}:
            raise ValueError(f"{path}: stage1 cell {cell} must map Q, P, X to r, m or c")
        if {"r", "m"} <= set(cell.values()):
            raise ValueError(f"{path}: stage1 cell {cell} mixes r and m, which share an id range and can collide")
    levels = cfg["stage1"]["dose"]["levels"]
    if levels != sorted(set(levels)) or levels[0] != 0 or levels[-1] > 30:
        raise ValueError(f"{path}: dose levels {levels} must be strictly increasing from 0 to at most 30")

    exp1_cfg = load_exp1_config(root / cfg["exp1_config"])
    for condition, run in cfg["exp1_runs"].items():
        ref_cfg = yaml.safe_load((root / run / "config.yaml").read_text())
        mismatched = [key for key in SHARED_KEYS if ref_cfg.get(key) != exp1_cfg[key]]
        if condition not in ref_cfg["run_conditions"]:
            mismatched.append(f"run_conditions (does not include {condition})")
        if mismatched:
            raise ValueError(f"exp1 run {run} does not match {cfg['exp1_config']} for {condition}: {mismatched}")
    cfg["_exp1"] = exp1_cfg
    return cfg


def target_heads(cfg: dict[str, Any]) -> list[tuple[int, int]]:
    """All target heads in config order (group by group)."""
    return [(layer, head) for heads in cfg["target_heads"].values() for layer, head in heads]


def head_group(cfg: dict[str, Any]) -> dict[tuple[int, int], str]:
    return {(layer, head): group for group, heads in cfg["target_heads"].items() for layer, head in heads}


def head_name(head: tuple[int, int]) -> str:
    return f"L{head[0]}H{head[1]}"
