"""Tests for exp1.runlog: run-folder naming and the code hash used in versions.json."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from exp1.runlog import make_run_dir, save_versions


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_make_run_dir_default_tag_is_exp1(tmp_path):
    assert make_run_dir(tmp_path).name.endswith("_exp1")


def test_make_run_dir_custom_tag(tmp_path):
    assert make_run_dir(tmp_path, tag="exp1extended_stage0").name.endswith("_exp1extended_stage0")


def test_save_versions_default_hash_unchanged(tmp_path):
    repo = tmp_path / "repo"
    _write(repo / "exp1" / "a.py", "x = 1\n")
    _write(repo / "exp1" / "b.py", "y = 2\n")
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    versions = save_versions(run_dir, repo)

    # The pre-change formula: bare file names over exp1/*.py, sorted.
    expected = hashlib.sha256()
    for path in sorted((repo / "exp1").glob("*.py")):
        expected.update(path.name.encode() + b"\0" + path.read_bytes())
    assert versions["code_sha256"] == expected.hexdigest()
    assert json.loads((run_dir / "versions.json").read_text())["code_sha256"] == expected.hexdigest()


def test_save_versions_extra_dir_changes_hash(tmp_path):
    repo = tmp_path / "repo"
    _write(repo / "exp1" / "a.py", "x = 1\n")
    _write(repo / "exp1extended" / "a.py", "z = 3\n")
    run_a, run_b = tmp_path / "a", tmp_path / "b"
    run_a.mkdir()
    run_b.mkdir()

    only_exp1 = save_versions(run_a, repo)["code_sha256"]
    both = save_versions(run_b, repo, code_dirs=("exp1", "exp1extended"))["code_sha256"]
    assert only_exp1 != both
