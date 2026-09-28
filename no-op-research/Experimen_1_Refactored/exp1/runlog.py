"""Per-run bookkeeping: a timestamped results folder, a verbatim config copy, and library versions."""
from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

PACKAGES = ("torch", "transformer-lens", "transformers", "numpy", "pandas")


def make_run_dir(results_dir: str | Path) -> Path:
    run_dir = Path(results_dir) / f"{datetime.now():%Y%m%d-%H%M%S}_exp1"
    run_dir.mkdir(parents=True, exist_ok=False)  # never write into an earlier run's folder
    return run_dir


def save_config_copy(config_path: str | Path, run_dir: Path) -> None:
    shutil.copy2(config_path, run_dir / "config.yaml")


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def save_versions(run_dir: Path, repo_dir: Path) -> dict[str, object]:
    versions: dict[str, object] = {"python": platform.python_version()}
    versions.update({pkg.replace("-", "_"): version(pkg) for pkg in PACKAGES})
    versions["git_commit"] = _git(["rev-parse", "HEAD"], repo_dir)
    # The commit alone does not pin the code if the working tree had uncommitted edits.
    status = _git(["status", "--porcelain", "--", "."], repo_dir)
    versions["git_dirty"] = None if status is None else bool(status)
    # Pins the code that actually ran even when it is uncommitted or untracked.
    code = hashlib.sha256()
    for path in sorted((repo_dir / "exp1").glob("*.py")):
        code.update(path.name.encode() + b"\0" + path.read_bytes())
    versions["code_sha256"] = code.hexdigest()
    (run_dir / "versions.json").write_text(json.dumps(versions, indent=2) + "\n")
    return versions
