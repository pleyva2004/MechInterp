"""Smoke tests: each exp1extended stage runs end to end on a tiny configuration and writes its report,
tables and figures. Uses the real GPT-2 small (TransformerLens, local cache) on a few hundred sequences at
most; every cache and output goes to tmp_path, so nothing under data/ or results/ is touched."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from exp1extended.config import load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def model():
    try:
        from exp1.backends import load_model, set_determinism

        set_determinism(0)
        return load_model("transformer_lens", "gpt2-small", "cpu", "float32")
    except Exception as exc:  # no local weights / no network
        pytest.skip(f"GPT-2 small unavailable: {exc}")


@pytest.fixture(scope="module")
def base_cfg():
    return load_config(ROOT / "configs" / "exp1extended.yaml", ROOT)


def _tiny(cfg: dict, tmp_path: Path) -> dict:
    cfg = copy.deepcopy(cfg)
    cfg["paths"] = {"sequences_dir": str(tmp_path / "sequences"), "results_dir": str(tmp_path / "results")}
    cfg["stats"].update({"bootstrap_iters": 20, "n_permutations": 20})
    cfg["validation"]["determinism_n"] = 5
    cfg["stage1"]["n_pairs"] = 24
    cfg["stage1"]["dose"].update({"n_pairs": 12, "levels": [0, 15, 30]})
    cfg["stage2"].update({
        "tokens": {"all": [200, 299], "rare_uniform": 20, "rare_matched": 20,
                   "rare_transition": {"range": [1000, 2999], "n": 10}},
        "backgrounds": {"rare": 3, "common": 3},
        "confirm": {"top_n": 2, "backgrounds": 2, "baseline_tokens": 5},
        "feature_bootstrap_iters": 3,
        "stage1_run": None,
    })
    return cfg


def _run_dir(tmp_path: Path, name: str) -> Path:
    run_dir = tmp_path / name
    (run_dir / "figures").mkdir(parents=True)
    return run_dir


def test_stage0_smoke(model, base_cfg, tmp_path, monkeypatch):
    import exp1extended.stage0 as stage0

    real = stage0.load_exp1_sequences
    monkeypatch.setattr(stage0, "load_exp1_sequences", lambda cfg, root, c: real(cfg, root, c)[:150])
    cfg = _tiny(base_cfg, tmp_path)
    run_dir = _run_dir(tmp_path, "s0")

    routes = stage0.run_stage0(cfg, run_dir, model, ROOT)

    assert len(routes) == sum(len(v) for v in cfg["target_heads"].values())
    assert set(routes["route"]) <= {"query_token", "query_context", "key_side", "mixed", "no_gap"}
    for name in ("stage0_report.md", "validation_report.md", "figures/stage0_qk_swap.png", "stage0_qk_swap.parquet"):
        assert (run_dir / name).stat().st_size > 0, name
    assert "FAIL" not in (run_dir / "validation_report.md").read_text()


def test_stage0_qk_swap_parts_sum_to_total(model, base_cfg, tmp_path, monkeypatch):
    """query + key = total exactly (per pair, hence in the means) in the offline q/k swap."""
    import exp1extended.stage0 as stage0

    real = stage0.load_exp1_sequences
    monkeypatch.setattr(stage0, "load_exp1_sequences", lambda cfg, root, c: real(cfg, root, c)[:40])
    cfg = _tiny(base_cfg, tmp_path)
    heads = [(7, 7), (2, 4)]
    runs, _ = stage0.extract_conditions(cfg, ROOT, model, heads)
    swap = stage0.qk_swap_table(cfg, runs, heads, model.cfg.d_head)
    wide = swap.pivot_table(index=["comparison", "head", "metric"], columns="part", values="mean")
    np.testing.assert_allclose(wide["query"] + wide["key"], wide["total"], atol=1e-9)


def test_stage1_smoke(model, base_cfg, tmp_path, monkeypatch):
    import exp1extended.stage1 as stage1

    cfg = _tiny(base_cfg, tmp_path)
    run_dir = _run_dir(tmp_path, "s1")
    # V6 compares 24 fresh sequences with 1000 saved ones; exercised separately below, not gated here.
    monkeypatch.setattr(stage1, "replication_checks", lambda *a, **k: [])

    g1 = stage1.run_stage1(cfg, run_dir, model, ROOT)

    assert "G1: sweep position 31" in g1.columns
    for name in ("stage1_report.md", "validation_report.md", "figures/s1_main_effects.png", "figures/s1_positions.png",
                 "figures/s1_surface_vs_rarity.png", "figures/s1_dose.png", "stage1_yates.parquet"):
        assert (run_dir / name).stat().st_size > 0, name
    yates = __import__("pandas").read_parquet(run_dir / "stage1_yates.parquet")
    wide = yates.pivot_table(index=["metric", "head"], columns="effect", values="mean")
    np.testing.assert_allclose(wide["Q"] + wide["P"] + wide["X"] + wide["QPX"], wide["total (ccc − rrr)"], atol=1e-6)


def test_stage1_replication_checks_run(model, base_cfg, tmp_path):
    import exp1extended.stage1 as stage1
    from exp1.sequences import prepend_bos
    from exp1extended.config import target_heads
    from exp1extended.extract import run_final
    from exp1extended.stage0 import load_exp1_sequences

    cfg = _tiny(base_cfg, tmp_path)
    heads = target_heads(cfg)
    # The saved Exp 1 sequences themselves: V6 must find no difference and V8 must pass.
    runs = {code: run_final(model, prepend_bos(load_exp1_sequences(cfg, ROOT, cond), cfg["bos_token_id"]),
                            cfg["batch_size"], heads, cfg["diffuse_threshold"])
            for code, cond in (("rrr", "A_rare"), ("ccc", "C_common_clean"))}
    checks = stage1.replication_checks(cfg, ROOT, runs, heads)
    assert checks and all(c["result"] == "PASS" for c in checks), checks


def test_stage2_smoke(model, base_cfg, tmp_path):
    import exp1extended.stage2 as stage2

    cfg = _tiny(base_cfg, tmp_path)
    run_dir = _run_dir(tmp_path, "s2")

    summary = stage2.run_stage2(cfg, run_dir, model, ROOT)

    assert {"reliability", "omnibus p", "omnibus q"} <= set(summary.columns)
    for name in ("stage2_report.md", "validation_report.md", "stage2_effects.parquet", "stage2_confirm.parquet",
                 "figures/s2_token_profile_rare_active_pos32.png", "figures/s2_rarity_x_wordstart_pos32.png",
                 "figures/s2_feature_effects_pos32.png", "figures/s2_top_tokens_heatmap_pos32.png"):
        assert (run_dir / name).stat().st_size > 0, name
    assert "FAIL" not in (run_dir / "validation_report.md").read_text()
