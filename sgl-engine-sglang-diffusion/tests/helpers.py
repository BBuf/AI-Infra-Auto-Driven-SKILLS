from __future__ import annotations

from pathlib import Path

import pytest

from sgl_engine_sglang_diffusion.process import run


@pytest.fixture
def fake_git_repo(tmp_path: Path) -> Path:
    repository = tmp_path / "upstream"
    repository.mkdir()
    run(["git", "init"], cwd=repository)
    run(["git", "config", "user.email", "tests@example.invalid"], cwd=repository)
    run(["git", "config", "user.name", "Test Author"], cwd=repository)
    (repository / "README.md").write_text("# fake repository\n")
    run(["git", "add", "README.md"], cwd=repository)
    run(["git", "commit", "-m", "initial"], cwd=repository)
    run(["git", "branch", "-M", "main"], cwd=repository)
    return repository
