from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sgl_engine_sglang_diffusion.patcher import (
    PatchError,
    PatchPackager,
    SGLangPathPolicy,
)
from sgl_engine_sglang_diffusion.process import run


def init_repo(tmp_path: Path) -> tuple[Path, str]:
    repository = tmp_path / "sglang"
    repository.mkdir()
    run(["git", "init"], cwd=repository)
    run(["git", "config", "user.email", "test@example.invalid"], cwd=repository)
    run(["git", "config", "user.name", "Test"], cwd=repository)
    (repository / "README.md").write_text("base\n")
    run(["git", "add", "README.md"], cwd=repository)
    run(["git", "commit", "-m", "base"], cwd=repository)
    base = run(["git", "rev-parse", "HEAD"], cwd=repository).stdout.strip()
    return repository, base


def write_candidate(repository: Path, base: str, *, derived: object = None) -> None:
    agent = repository / "python/sglang/kernels/agent"
    agent.mkdir(parents=True)
    for name in ("registry.py", "manifest.py", "runtime.py", "receipt.py"):
        (agent / name).write_text(
            'OPTION = "--agent-optimization"\nMODES = ("off", "auto")\n'
        )
    model = agent / "diffusion/test-model"
    model.mkdir(parents=True)
    source = model / "kernel.py"
    source.write_text("VALUE = 1\n")
    profile = {
        "schema_version": 1,
        "profile_id": "profile-1",
        "campaign_id": "campaign-1",
        "model_ids": ["test/model"],
        "sglang_base_sha": base,
        "hardware": {"gpu": "test"},
        "workload": {"width": 64},
        "techniques": {"kernel": {"enabled": True}},
        "server_args": {"agent_optimization": "profile-1"},
        "fallback_policy": "native",
        "source_hashes": {
            "python/sglang/kernels/agent/diffusion/test-model/kernel.py": (
                hashlib.sha256(source.read_bytes()).hexdigest()
            )
        },
        "integrated_delivery_sha256": "b" * 64,
        "speedup": 2.0,
    }
    if derived is not None:
        profile["derived_checkpoint"] = derived
    (model / "manifest.json").write_text(json.dumps(profile))
    run(["git", "add", "."], cwd=repository)
    run(["git", "commit", "-m", "candidate"], cwd=repository)


def test_path_policy_rejects_generated_kernel_outside_agent_folder() -> None:
    policy = SGLangPathPolicy("test-model")
    with pytest.raises(PatchError, match="outside"):
        policy.validate_changed_paths(
            [("A", "python/sglang/kernels/ops/diffusion/generated.py")]
        )
    policy.validate_changed_paths(
        [
            (
                "A",
                "python/sglang/kernels/agent/diffusion/test-model/kernel.py",
            ),
            (
                "A",
                "test/registered/kernels/ops/diffusion/agent/test-model/test.py",
            ),
        ]
    )


def test_packager_validates_profile_and_clean_room_patch(tmp_path: Path) -> None:
    repository, base = init_repo(tmp_path)
    write_candidate(repository, base)
    output = tmp_path / "bundle"
    bundle = PatchPackager(repository, base_sha=base).package(
        output,
        model_slug="test-model",
        profile_id="profile-1",
        cpu_validation_commands=(("git", "diff", "--check"),),
        gpu_validation_command=("python", "-m", "pytest", "gpu_test.py"),
    )
    assert bundle.patch.read_text().startswith("diff --git")
    assert bundle.apply_script.stat().st_mode & 0o111
    assert "sglang.patch" in bundle.checksums.read_text()


def test_packager_rejects_host_paths_and_credentials(tmp_path: Path) -> None:
    repository, base = init_repo(tmp_path)
    write_candidate(repository, base)
    runtime = repository / "python/sglang/kernels/agent/runtime.py"
    runtime.write_text('PATH = "/Users/person/model"\nHF_TOKEN = "secret"\n')
    run(["git", "add", "."], cwd=repository)
    run(["git", "commit", "-m", "leak"], cwd=repository)
    with pytest.raises(PatchError, match="forbidden"):
        PatchPackager(repository, base_sha=base).validate(model_slug="test-model")


def test_quantized_profile_requires_immutable_derived_weights(
    tmp_path: Path,
) -> None:
    repository, base = init_repo(tmp_path)
    write_candidate(repository, base, derived={"uri": "hf://weights"})
    with pytest.raises(PatchError, match="not immutable"):
        PatchPackager(repository, base_sha=base).validate(model_slug="test-model")


def test_quantized_profile_accepts_fully_locked_weights(tmp_path: Path) -> None:
    repository, base = init_repo(tmp_path)
    write_candidate(
        repository,
        base,
        derived={
            "uri": "hf://org/model",
            "revision": "c" * 40,
            "size_bytes": 1024,
            "sha256": "d" * 64,
        },
    )
    profile = PatchPackager(repository, base_sha=base).validate(model_slug="test-model")
    assert profile.profile_id == "profile-1"
