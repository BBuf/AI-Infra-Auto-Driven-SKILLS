from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HISTORY_ROOT = ROOT / "model-pr-optimization-history"


def test_rebuild_history_dry_run_does_not_update_indexes(tmp_path) -> None:
    script = ROOT / "tools" / "rebuild_model_pr_history_from_git.py"
    spec = importlib.util.spec_from_file_location(
        "rebuild_model_pr_history_from_git", script
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["rebuild_model_pr_history_from_git"] = module
    spec.loader.exec_module(module)

    module.HISTORY_ROOT = tmp_path
    module.MODEL_ORDER = ["qwen3-core"]
    (tmp_path / "sglang").mkdir()
    (tmp_path / "vllm").mkdir()

    module.update_indexes(dry_run=True)

    assert not (tmp_path / "sglang" / "README.md").exists()
    assert not (tmp_path / "vllm" / "README.md").exists()


def test_model_pr_history_is_queryable_knowledge_base() -> None:
    list_result = subprocess.run(
        [sys.executable, "scripts/query.py", "--list"],
        cwd=HISTORY_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    assert "sglang/qwen3-core" in list_result.stdout
    assert "vllm/qwen3-core" in list_result.stdout

    paths_result = subprocess.run(
        [
            sys.executable,
            "scripts/query.py",
            "--framework",
            "sglang",
            "--model",
            "qwen3-core",
            "--paths-only",
        ],
        cwd=HISTORY_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    assert "sglang/qwen3-core/README.en.md" in paths_result.stdout

    search_result = subprocess.run(
        [
            sys.executable,
            "scripts/query.py",
            "--framework",
            "sglang",
            "--model",
            "qwen3-core",
            "fused qk norm",
        ],
        cwd=HISTORY_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    assert "sglang/qwen3-core" in search_result.stdout
    assert "Read:" in search_result.stdout

    model_id_result = subprocess.run(
        [sys.executable, "scripts/query.py", "Qwen/Qwen3-8B", "--limit", "1"],
        cwd=HISTORY_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    assert model_id_result.stdout.startswith("## sglang/qwen3-core")
