from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from sgl_engine_sglang_diffusion.driver import BenchmarkRun
from sgl_engine_sglang_diffusion.integrator import (
    CandidateActivation,
    IntegrationError,
    IntegrationManager,
    IntegrationVerificationOutcome,
    IntegrationVerificationRequest,
    VerifiedCandidate,
)
from sgl_engine_sglang_diffusion.models import (
    BaselineRecord,
    CampaignGoal,
    CorrectnessMode,
    QualityRecord,
)
from sgl_engine_sglang_diffusion.process import run
from sgl_engine_sglang_diffusion.sources import SourceManager


pytest_plugins = ("helpers",)

BASE_HASH = "a" * 64


class RecordingVerifier:
    def __init__(self) -> None:
        self.requests: list[IntegrationVerificationRequest] = []

    def verify_integrated(
        self, request: IntegrationVerificationRequest
    ) -> IntegrationVerificationOutcome:
        self.requests.append(request)
        if request.correctness is CorrectnessMode.LOSSLESS:
            assert not any(
                "lpips" in str(path).lower() for path in request.run_dir.rglob("*")
            )
            quality = QualityRecord(
                mode="not_gated",
                lpips_max=None,
                lpips_mean=None,
                visual_overall="authenticity_only",
                visual_verdict=request.run_dir / "equivalence.json",
                relation="equivalent",
            )
        else:
            quality = QualityRecord(
                mode="quality_gated",
                lpips_max=0.2,
                lpips_mean=0.1,
                visual_overall="pass",
                visual_verdict=request.run_dir / "visual-verdict.json",
                relation="equivalent",
            )
        return IntegrationVerificationOutcome(
            accepted=True,
            quality=quality,
            artifacts=[quality.visual_verdict],
            implementation_manifest={
                "verification_correctness": request.correctness.value
            },
        )


class RejectingVerifier(RecordingVerifier):
    def verify_integrated(
        self, request: IntegrationVerificationRequest
    ) -> IntegrationVerificationOutcome:
        self.requests.append(request)
        return IntegrationVerificationOutcome(
            accepted=False,
            issues=["combined candidate failed correctness"],
        )


class FakeDriver:
    calls: list[dict[str, Any]] = []

    def __init__(self, checkout: Path) -> None:
        self.checkout = checkout

    def run(
        self,
        goal: CampaignGoal,
        run_dir: Path,
        *,
        activation: Any,
        profile: bool,
    ) -> BenchmarkRun:
        run_dir.mkdir(parents=True)
        output_file = run_dir / "outputs" / "benchmark.jsonl"
        media_dir = run_dir / "outputs" / "media"
        normalized_file = run_dir / "PERFORMANCE.json"
        receipt = run_dir / "COMMAND.json"
        output_file.parent.mkdir(parents=True)
        media_dir.mkdir()
        output_file.write_text('{"results": {"total_s": 5.0}}\n')
        normalized_file.write_text(
            json.dumps(
                {
                    "total_s": 5.0,
                    "peak_memory_mib": 80.0,
                    "timing_scope": goal.workload.timing_scope,
                }
            )
        )
        receipt.write_text("{}\n")
        (media_dir / "out.mp4").write_bytes(b"fake")
        self.calls.append(
            {
                "checkout": self.checkout,
                "activation": activation,
                "profile": profile,
            }
        )
        return BenchmarkRun(
            run_dir=run_dir,
            output_file=output_file,
            normalized_file=normalized_file,
            media_dir=media_dir,
            command_receipt=receipt,
            normalized={
                "total_s": 5.0,
                "peak_memory_mib": 80.0,
                "timing_scope": goal.workload.timing_scope,
            },
            stdout="",
            stderr="",
        )


@pytest.fixture(autouse=True)
def clear_driver_calls() -> None:
    FakeDriver.calls.clear()


def _commit(repository: Path, branch: str, files: dict[str, str]) -> str:
    run(["git", "switch", "--detach", "main"], cwd=repository)
    run(["git", "switch", "-C", branch], cwd=repository)
    for relative, content in files.items():
        target = repository / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    run(["git", "add", *files], cwd=repository)
    run(["git", "commit", "-m", branch], cwd=repository)
    commit = run(["git", "rev-parse", "HEAD"], cwd=repository).stdout.strip()
    run(["git", "switch", "main"], cwd=repository)
    return commit


def _setup(
    tmp_path: Path,
    repository: Path,
    verifier: RecordingVerifier | None = None,
) -> tuple[IntegrationManager, str, RecordingVerifier]:
    source_manager = SourceManager(tmp_path / "sources")
    source_lock = source_manager.lock("sglang", str(repository), "main")
    selected_verifier = verifier or RecordingVerifier()
    manager = IntegrationManager(
        source_manager,
        source_lock,
        selected_verifier,
        driver_type=FakeDriver,
    )
    return manager, source_lock.commit, selected_verifier


def _candidate(
    candidate_id: str,
    technique: str,
    base: str,
    commit: str,
    *,
    correctness: CorrectnessMode = CorrectnessMode.LOSSLESS,
    verified: bool = True,
    env: dict[str, str] | None = None,
    server_args: list[str] | None = None,
    source_hashes: dict[str, str] | None = None,
) -> VerifiedCandidate:
    return VerifiedCandidate(
        candidate_id=candidate_id,
        technique=technique,
        base_commit=base,
        candidate_commit=commit,
        correctness=correctness,
        activation=CandidateActivation(
            env=env or {},
            server_args=server_args or [],
        ),
        source_hashes=source_hashes or {"sglang": BASE_HASH},
        compatibility_notes=[f"{technique} is compatible"],
        verified=verified,
    )


def _goal(prompt_file: Path) -> CampaignGoal:
    return CampaignGoal.model_validate(
        {
            "schema_version": 1,
            "model": {"id": "fake/model"},
            "hardware": {"environment": "fake-b200", "gpu_count": 1},
            "workload": {
                "prompts": str(prompt_file),
                "prompt_count": 5,
                "seed": 42,
                "height": 64,
                "width": 64,
                "frames": 1,
                "fps": 24,
                "steps": 4,
                "guidance": 1.0,
                "dtype": "bfloat16",
                "timing_scope": "load_excluded_end_to_end",
            },
            "goal": {"target_speedup": 2.0, "allow_quality_gated": True},
            "source": {"sglang_repo": "fake"},
            "agent": {"command": ["fake-agent"]},
        }
    )


def _baseline(tmp_path: Path, base: str) -> BaselineRecord:
    return BaselineRecord(
        model_id="fake/model",
        total_s=10.0,
        peak_memory_mib=100.0,
        timing_scope="load_excluded_end_to_end",
        run_dir=tmp_path / "baseline-run",
        baseline_frames=tmp_path / "baseline-frames",
        sglang_commit=base,
    )


def test_unverified_candidate_is_rejected_before_worktree_creation(
    tmp_path: Path, fake_git_repo: Path
) -> None:
    commit = _commit(fake_git_repo, "candidate", {"candidate.txt": "change\n"})
    manager, base, _ = _setup(tmp_path, fake_git_repo)
    candidate = _candidate("candidate", "kernel", base, commit, verified=False)

    with pytest.raises(IntegrationError, match="not passed"):
        manager.compose(
            ["candidate"],
            {"candidate": candidate},
            tmp_path / "integration",
        )

    assert not (tmp_path / "integration").exists()


def test_composition_is_canonical_and_does_not_modify_candidate_repository(
    tmp_path: Path, fake_git_repo: Path
) -> None:
    kernel_commit = _commit(
        fake_git_repo, "kernel-candidate", {"kernel.txt": "kernel\n"}
    )
    topology_commit = _commit(
        fake_git_repo, "topology-candidate", {"topology.txt": "topology\n"}
    )
    manager, base, _ = _setup(tmp_path, fake_git_repo)
    before = run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=fake_git_repo,
    ).stdout
    candidates = {
        "kernel": _candidate("kernel", "kernel", base, kernel_commit),
        "topology": _candidate("topology", "topology", base, topology_commit),
    }

    result = manager.compose(
        ["kernel", "topology"],
        candidates,
        tmp_path / "integration",
    )

    assert result.status == "ready_for_verification"
    assert result.recipe.ordered_candidate_ids == ["topology", "kernel"]
    assert (result.worktree / "topology.txt").read_text() == "topology\n"
    assert (result.worktree / "kernel.txt").read_text() == "kernel\n"
    assert (
        run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=fake_git_repo,
        ).stdout
        == before
    )


def test_conflict_aborts_and_returns_agent_revision_diagnostics(
    tmp_path: Path, fake_git_repo: Path
) -> None:
    topology_commit = _commit(
        fake_git_repo, "topology-conflict", {"README.md": "topology\n"}
    )
    kernel_commit = _commit(fake_git_repo, "kernel-conflict", {"README.md": "kernel\n"})
    manager, base, _ = _setup(tmp_path, fake_git_repo)
    candidates = {
        "kernel": _candidate("kernel", "kernel", base, kernel_commit),
        "topology": _candidate("topology", "topology", base, topology_commit),
    }

    result = manager.compose(
        ["kernel", "topology"],
        candidates,
        tmp_path / "integration",
    )

    assert result.status == "needs_agent_revision"
    assert result.failed_candidate_id == "kernel"
    assert result.diagnostics_path is not None
    diagnostics = json.loads(result.diagnostics_path.read_text())
    assert diagnostics["status"] == "needs_agent_revision"
    assert diagnostics["conflict_files"] == ["README.md"]
    assert not (result.worktree / ".git" / "CHERRY_PICK_HEAD").exists()
    assert run(["git", "status", "--porcelain"], cwd=result.worktree).stdout == ""


def test_integrated_activation_and_quality_routing_are_complete(
    tmp_path: Path, fake_git_repo: Path
) -> None:
    kernel_commit = _commit(fake_git_repo, "kernel-quality", {"kernel.txt": "kernel\n"})
    cache_commit = _commit(fake_git_repo, "cache-quality", {"cache.txt": "cache\n"})
    manager, base, verifier = _setup(tmp_path, fake_git_repo)
    candidates = {
        "kernel": _candidate(
            "kernel",
            "kernel",
            base,
            kernel_commit,
            env={"KERNEL_MODE": "1"},
            server_args=["--kernel-mode"],
        ),
        "cache": _candidate(
            "cache",
            "cache",
            base,
            cache_commit,
            correctness=CorrectnessMode.QUALITY_GATED,
            env={"CACHE_MODE": "1"},
            server_args=["--cache-mode", "fast"],
            source_hashes={"sglang": BASE_HASH, "cache": "b" * 64},
        ),
    }
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("\n".join(f"prompt {i}" for i in range(5)))

    result = manager.integrate(
        _goal(prompts),
        _baseline(tmp_path, base),
        ["cache", "kernel"],
        candidates,
        tmp_path / "final-integration",
    )

    assert result.status == "integrated"
    assert len(verifier.requests) == 1
    request = verifier.requests[0]
    assert request.correctness is CorrectnessMode.QUALITY_GATED
    assert request.performance.speedup == pytest.approx(2.0)
    assert request.recipe.activation.env == {
        "KERNEL_MODE": "1",
        "CACHE_MODE": "1",
    }
    assert request.recipe.activation.server_args == [
        "--kernel-mode",
        "--cache-mode",
        "fast",
    ]
    assert request.recipe.source_hashes == {
        "sglang": BASE_HASH,
        "cache": "b" * 64,
    }
    assert FakeDriver.calls[0]["activation"].env == request.recipe.activation.env
    delivery = json.loads(result.delivery_path.read_text())
    assert delivery["schema_version"] == 2
    assert delivery["frontier_points"][0]["quality"]["mode"] == "quality_gated"


def test_all_lossless_integration_never_routes_to_lpips(
    tmp_path: Path, fake_git_repo: Path
) -> None:
    commit = _commit(fake_git_repo, "lossless-only", {"kernel.txt": "kernel\n"})
    manager, base, verifier = _setup(tmp_path, fake_git_repo)
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("\n".join(f"prompt {i}" for i in range(5)))

    result = manager.integrate(
        _goal(prompts),
        _baseline(tmp_path, base),
        ["kernel"],
        {"kernel": _candidate("kernel", "kernel", base, commit)},
        tmp_path / "lossless-integration",
    )

    assert result.status == "integrated"
    assert verifier.requests[0].correctness is CorrectnessMode.LOSSLESS
    delivery = json.loads(result.delivery_path.read_text())
    quality = delivery["frontier_points"][0]["quality"]
    assert quality["mode"] == "not_gated"
    assert quality["lpips_max"] is None
    assert quality["lpips_mean"] is None


def test_rejected_integrated_verification_never_writes_delivery(
    tmp_path: Path, fake_git_repo: Path
) -> None:
    commit = _commit(fake_git_repo, "rejected-integration", {"kernel.txt": "kernel\n"})
    verifier = RejectingVerifier()
    manager, base, _ = _setup(tmp_path, fake_git_repo, verifier)
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("\n".join(f"prompt {i}" for i in range(5)))
    integration_root = tmp_path / "rejected-final"

    with pytest.raises(IntegrationError, match="failed correctness"):
        manager.integrate(
            _goal(prompts),
            _baseline(tmp_path, base),
            ["kernel"],
            {"kernel": _candidate("kernel", "kernel", base, commit)},
            integration_root,
        )

    assert not (integration_root / "INTEGRATED-DELIVERY.json").exists()
