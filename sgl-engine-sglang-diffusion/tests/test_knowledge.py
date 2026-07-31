from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from sgl_engine_sglang_diffusion.knowledge import (
    KnowledgeSyncError,
    load_registry,
    sync_source,
)
from sgl_engine_sglang_diffusion.resources import KNOWLEDGE_REGISTRY


def _run(argv: list[str], cwd: Path) -> str:
    completed = subprocess.run(
        argv, cwd=cwd, text=True, capture_output=True, check=True
    )
    return completed.stdout.strip()


def _init_repo(path: Path) -> Path:
    path.mkdir()
    _run(["git", "init", "-b", "main"], path)
    _run(["git", "config", "user.name", "Test"], path)
    _run(["git", "config", "user.email", "test@example.com"], path)
    return path


def _commit_all(path: Path, message: str) -> str:
    _run(["git", "add", "."], path)
    _run(["git", "commit", "-m", message], path)
    return _run(["git", "rev-parse", "HEAD"], path)


def test_sync_reads_only_allowlisted_paths(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    (repo / "allowed.md").write_text("# QKNorm\nUse a fused op.\n")
    (repo / "secret.txt").write_text("HF_TOKEN=secret\n")
    commit = _commit_all(repo, "knowledge")

    snapshot = sync_source(
        name="fake",
        checkout=repo,
        commit=commit,
        patterns=["allowed.md"],
        output_dir=tmp_path / "out",
    )

    assert [entry.path for entry in snapshot.entries] == ["allowed.md"]
    output_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (tmp_path / "out").rglob("*")
        if path.is_file()
    )
    assert "HF_TOKEN=secret" not in output_text


def test_sync_redacts_secrets_and_absolute_paths(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    (repo / "allowed.md").write_text(
        "HF_TOKEN=secret\nRead /Users/example/private/model.bin\n"
    )
    commit = _commit_all(repo, "knowledge")

    sync_source(
        name="fake",
        checkout=repo,
        commit=commit,
        patterns=["allowed.md"],
        output_dir=tmp_path / "out",
    )

    reference = (tmp_path / "out" / "references" / "allowed.md").read_text()
    assert "secret" not in reference
    assert "/Users/example" not in reference
    assert "<redacted>" in reference
    assert "<redacted-absolute-path>" in reference


def test_load_registry_exposes_expected_sources() -> None:
    registry = load_registry(KNOWLEDGE_REGISTRY)
    assert set(registry) == {
        "sglang",
        "fastvideo",
        "kda_pilot",
        "kernel_wiki",
        "ncu_report_skill",
        "warp_specialization_report_skill",
    }
    assert ".claude/skills/add-jit-kernel/**" in registry["sglang"]
    assert "python/sglang/multimodal_gen/.claude/skills/**" in registry["sglang"]
    assert "diffusion/**" in registry["kda_pilot"]
    assert "sources/**" in registry["kernel_wiki"]
    assert "reference/**" in registry["ncu_report_skill"]
    assert any("kernels/ops/diffusion" in path for path in registry["sglang"])
    assert any("kernels/aot/CMakeLists.txt" in path for path in registry["sglang"])
    assert any("kernels/fused_op.py" in path for path in registry["sglang"])


def test_sync_rejects_empty_required_source(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    (repo / "unrelated.txt").write_text("not selected\n")
    commit = _commit_all(repo, "empty")

    with pytest.raises(KnowledgeSyncError, match="matched no allowlisted"):
        sync_source(
            name="required",
            checkout=repo,
            commit=commit,
            patterns=["missing/**"],
            output_dir=tmp_path / "out",
        )
