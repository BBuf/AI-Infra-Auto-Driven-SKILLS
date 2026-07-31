from __future__ import annotations

from pathlib import Path

import pytest

from sgl_engine_sglang_diffusion.process import run
from sgl_engine_sglang_diffusion.sources import (
    SourceManager,
    derive_submodule_sources,
)

pytest_plugins = ("helpers",)


def test_lock_source_resolves_full_commit(fake_git_repo: Path, tmp_path: Path) -> None:
    manager = SourceManager(tmp_path / "sources")
    lock = manager.lock("sglang", str(fake_git_repo), "main")
    assert len(lock.commit) == 40
    assert manager.checkout_path(lock).is_dir()


def test_create_worktree_is_clean_and_detached(
    fake_git_repo: Path, tmp_path: Path
) -> None:
    manager = SourceManager(tmp_path / "sources")
    lock = manager.lock("sglang", str(fake_git_repo), "main")
    worktree = manager.create_worktree(lock, tmp_path / "candidate")
    status = run(["git", "status", "--porcelain"], cwd=worktree)
    symbolic_head = run(
        ["git", "symbolic-ref", "-q", "HEAD"], cwd=worktree, check=False
    )
    assert status.stdout == ""
    assert symbolic_head.returncode != 0


def test_assert_clean_worktree_rejects_untracked_file(
    fake_git_repo: Path, tmp_path: Path
) -> None:
    manager = SourceManager(tmp_path / "sources")
    lock = manager.lock("sglang", str(fake_git_repo), "main")
    worktree = manager.create_worktree(lock, tmp_path / "candidate")
    (worktree / "untracked.txt").write_text("not part of the locked source\n")

    with pytest.raises(RuntimeError, match="worktree is dirty"):
        manager.assert_clean_worktree(worktree)


def test_create_worktree_refuses_existing_destination(
    fake_git_repo: Path, tmp_path: Path
) -> None:
    manager = SourceManager(tmp_path / "sources")
    lock = manager.lock("sglang", str(fake_git_repo), "main")
    destination = tmp_path / "candidate"
    destination.mkdir()
    marker = destination / "keep.txt"
    marker.write_text("do not overwrite\n")

    with pytest.raises(FileExistsError, match="already exists"):
        manager.create_worktree(lock, destination)
    assert marker.read_text() == "do not overwrite\n"


def test_create_worktree_refuses_destination_inside_shared_cache(
    fake_git_repo: Path, tmp_path: Path
) -> None:
    manager = SourceManager(tmp_path / "sources")
    lock = manager.lock("sglang", str(fake_git_repo), "main")
    destination = manager.checkout_path(lock) / "candidate"

    with pytest.raises(ValueError, match="outside the shared cache"):
        manager.create_worktree(lock, destination)


def test_source_name_cannot_escape_cache_root(
    fake_git_repo: Path, tmp_path: Path
) -> None:
    manager = SourceManager(tmp_path / "sources")

    with pytest.raises(ValueError, match="unsafe source name"):
        manager.lock("../sglang", str(fake_git_repo), "main")


def test_source_cache_name_cannot_be_rebound(
    fake_git_repo: Path, tmp_path: Path
) -> None:
    manager = SourceManager(tmp_path / "sources")
    manager.lock("sglang", str(fake_git_repo), "main")
    other = tmp_path / "other"
    other.mkdir()
    run(["git", "init"], cwd=other)

    with pytest.raises(RuntimeError, match="already bound"):
        manager.lock("sglang", str(other), "main")


def test_old_lock_remains_materializable_after_a_new_lock(
    fake_git_repo: Path, tmp_path: Path
) -> None:
    manager = SourceManager(tmp_path / "sources")
    old_lock = manager.lock("sglang", str(fake_git_repo), "main")
    (fake_git_repo / "README.md").write_text("# updated repository\n")
    run(["git", "add", "README.md"], cwd=fake_git_repo)
    run(["git", "commit", "-m", "update"], cwd=fake_git_repo)
    new_lock = manager.lock("sglang", str(fake_git_repo), "main")

    assert new_lock.commit != old_lock.commit
    old_worktree = manager.create_worktree(old_lock, tmp_path / "old-candidate")
    assert run(["git", "rev-parse", "HEAD"], cwd=old_worktree).stdout.strip() == (
        old_lock.commit
    )


def test_derive_submodule_sources_uses_exact_parent_gitlinks(
    fake_git_repo: Path, tmp_path: Path
) -> None:
    child = tmp_path / "kernel-wiki"
    child.mkdir()
    run(["git", "init"], cwd=child)
    run(["git", "config", "user.email", "tests@example.invalid"], cwd=child)
    run(["git", "config", "user.name", "Test Author"], cwd=child)
    (child / "SKILL.md").write_text("# Kernel knowledge\n")
    run(["git", "add", "SKILL.md"], cwd=child)
    run(["git", "commit", "-m", "skill"], cwd=child)
    child_commit = run(["git", "rev-parse", "HEAD"], cwd=child).stdout.strip()

    (fake_git_repo / ".gitmodules").write_text(
        '[submodule "external/KernelWiki"]\n'
        "\tpath = external/KernelWiki\n"
        f"\turl = {child}\n"
    )
    run(["git", "add", ".gitmodules"], cwd=fake_git_repo)
    run(
        [
            "git",
            "update-index",
            "--add",
            "--cacheinfo",
            f"160000,{child_commit},external/KernelWiki",
        ],
        cwd=fake_git_repo,
    )
    run(["git", "commit", "-m", "pin kernel knowledge"], cwd=fake_git_repo)

    manager = SourceManager(tmp_path / "sources")
    parent = manager.lock("kda_pilot", str(fake_git_repo), "main")
    specs = derive_submodule_sources(
        manager,
        parent,
        {"external/KernelWiki": "kernel_wiki"},
    )

    assert specs["kernel_wiki"].commit == child_commit
    assert specs["kernel_wiki"].repository == str(child)
