"""Load and sanity-check configs/exp1.yaml, the single source of experimental parameters."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from exp1.backends import BACKENDS

REQUIRED = (
    "model", "conditions", "run_conditions", "seq_len", "prepend_bos", "bos_token_id",
    "sample_without_replacement", "n_sequences", "seed", "query_position", "diffuse_threshold",
    "batch_size", "device", "dtype", "backends", "primary_backend", "paths", "validation", "figures", "comparison",
)


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f)

    missing = [key for key in REQUIRED if key not in cfg]
    if missing:
        raise ValueError(f"{path}: missing keys {missing}")
    unknown = [c for c in cfg["run_conditions"] if c not in cfg["conditions"]]
    if unknown:
        raise ValueError(f"{path}: run_conditions {unknown} are not defined under conditions")
    for name, cond in cfg["conditions"].items():
        lo, hi = cond["id_range"]
        if not 0 <= lo <= hi:
            raise ValueError(f"{path}: condition {name} has invalid id_range {cond['id_range']}")
    bad_backends = [b for b in cfg["backends"] if b not in BACKENDS]
    if bad_backends:
        raise ValueError(f"{path}: unknown backends {bad_backends}; expected a subset of {BACKENDS}")
    if cfg["primary_backend"] not in cfg["backends"]:
        raise ValueError(f"{path}: primary_backend {cfg['primary_backend']!r} is not listed under backends")
    comp = cfg["comparison"]
    if comp["enabled"]:
        needed = {c for pair in comp["pairs"] for c in pair} | set(comp["monotonic_order"])
        unavailable = sorted(needed - set(cfg["run_conditions"]) - set(comp["reference_runs"]))
        if unavailable:
            raise ValueError(f"{path}: comparison needs {unavailable}, which are neither in run_conditions "
                             "nor in comparison.reference_runs")
    return cfg
