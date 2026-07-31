from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from sgl_engine_sglang_diffusion.models import BaselineRecord
from sgl_engine_sglang_diffusion.process import run
from sgl_engine_sglang_diffusion.request import FrozenBenchmarkCommand
from sgl_engine_sglang_diffusion.techniques import TechniqueRegistry
from sgl_engine_sglang_diffusion.verifier import (
    DeliveryVerifier,
    VerificationError,
    resolve_inside,
)


class AuditSpy:
    def __init__(self, verdict: bool = True) -> None:
        self.calls = 0
        self.verdict = verdict

    def audit(self, **_: Any) -> bool:
        self.calls += 1
        return self.verdict


class QualitySpy:
    def __init__(self) -> None:
        self.calls = 0

    def assess(self, **kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        return {
            "aligned": True,
            "lpips_mean": 0.1,
            "lpips_max": 0.2,
            "prompt_scores": [{"prompt": index, "lpips": 0.1} for index in range(5)],
        }


class ForbiddenQualityEvaluator:
    def assess(self, **_: Any) -> dict[str, Any]:
        raise AssertionError("lossless verification invoked a quality evaluator")


@pytest.fixture
def registry() -> TechniqueRegistry:
    root = Path(__file__).parents[1]
    return TechniqueRegistry.load(root / "techniques/registry.toml")


@pytest.fixture
def evidence(tmp_path: Path) -> dict[str, Any]:
    worktree = tmp_path / "executor"
    campaign = tmp_path / "campaign"
    run_dir = worktree / "runs" / "candidate"
    worktree.mkdir()
    run(["git", "init"], cwd=worktree)
    run(["git", "config", "user.email", "test@example.invalid"], cwd=worktree)
    run(["git", "config", "user.name", "Test"], cwd=worktree)
    (worktree / "README.md").write_text("base\n")
    run(["git", "add", "README.md"], cwd=worktree)
    run(["git", "commit", "-m", "base"], cwd=worktree)
    base_commit = run(["git", "rev-parse", "HEAD"], cwd=worktree).stdout.strip()
    baseline_run = campaign / "baseline" / "run"
    baseline_run.mkdir(parents=True)
    baseline_frames = campaign / "baseline" / "frames"
    (baseline_frames / "prompt-00").mkdir(parents=True)
    (baseline_frames / "prompt-00" / "frame.png").write_bytes(b"baseline")
    source = worktree / "python/sglang/kernels/agent/kernel.py"
    source.parent.mkdir(parents=True)
    source.write_text("def optimized(): return True\n", encoding="utf-8")
    run(["git", "add", str(source.relative_to(worktree))], cwd=worktree)
    run(["git", "commit", "-m", "candidate"], cwd=worktree)
    candidate_commit = run(["git", "rev-parse", "HEAD"], cwd=worktree).stdout.strip()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    relative_source = str(source.relative_to(worktree))

    run_dir.mkdir(parents=True)
    performance = {
        "schema_version": 1,
        "candidate_id": "candidate-1",
        "run_id": "run-1",
        "total_s": 5.0,
        "peak_memory_mib": 900.0,
        "timing_scope": "frozen_e2e",
        "fallback_count": 0,
    }
    (run_dir / "PERFORMANCE.json").write_text(json.dumps(performance))
    outputs = run_dir / "outputs"
    outputs.mkdir()
    (outputs / "benchmark.jsonl").write_text(
        json.dumps(
            {
                "results": {
                    "successful_requests": 5,
                    "failed_requests": 0,
                    "total_s": 5.0,
                    "peak_memory_mib": 900.0,
                }
            }
        )
        + "\n"
    )
    media = outputs / "media"
    media.mkdir()
    for index in range(5):
        (media / f"out-{index}.mp4").write_bytes(b"real candidate media")
    manifest = {
        "schema_version": 1,
        "candidate_id": "candidate-1",
        "technique": "kernel",
        "kind": "patch",
        "base_commit": base_commit,
        "candidate_commit": candidate_commit,
        "activation": {"enable_agent_kernel": True},
        "eval_profile": {"timing_scope": "frozen_e2e"},
        "knowledge_origin": [],
    }
    (run_dir / "implementation-manifest.json").write_text(json.dumps(manifest))
    source_hashes = {relative_source: digest}
    (run_dir / "source-hashes.json").write_text(json.dumps(source_hashes))
    engagement = {
        "schema_version": 1,
        "profile_id": "candidate-1",
        "model_match": True,
        "hardware_match": True,
        "workload_match": True,
        "techniques": {
            "kernel": {"engaged": True, "call_count": 8, "fallback_count": 0}
        },
        "source_hashes": source_hashes,
    }
    (run_dir / "engagement-receipt.json").write_text(json.dumps(engagement))
    equivalence = {
        "candidate_id": "candidate-1",
        "run_id": "run-1",
        "baseline": {"global_steps": 20, "dit_calls": 20},
        "candidate": {"global_steps": 20, "dit_calls": 20},
        "method_argument": "The fused kernel computes the same indexed contraction.",
        "logical_work_unchanged": True,
        "approximation": False,
        "step_skipping": False,
        "sparsity": False,
        "sub_16bit": False,
        "rank_reduction": False,
    }
    (run_dir / "equivalence.json").write_text(json.dumps(equivalence))
    authenticity = {
        "candidate_id": "candidate-1",
        "authentic": True,
        "overall": "authenticity_only",
    }
    (run_dir / "authenticity.json").write_text(json.dumps(authenticity))

    artifact_names = [
        "PERFORMANCE.json",
        "outputs/benchmark.jsonl",
        "outputs/media",
        "implementation-manifest.json",
        "source-hashes.json",
        "engagement-receipt.json",
        "equivalence.json",
        "authenticity.json",
    ]
    point = {
        "candidate_id": "candidate-1",
        "run_dir": str(run_dir),
        "activation": {"enable_agent_kernel": True},
        "implementation_manifest": manifest,
        "performance": {
            "frontier_axis": "latency",
            "baseline_total_s": 10.0,
            "candidate_total_s": 5.0,
            "speedup": 2.0,
        },
        # Deliberately non-null: lossless verification must ignore LPIPS movement.
        "quality": {
            "mode": "not_gated",
            "lpips_max": 0.9,
            "lpips_mean": 0.8,
            "visual_overall": "authenticity_only",
            "visual_verdict": "authenticity.json",
            "relation": "not_applicable",
        },
        "artifacts": artifact_names,
    }
    delivery = {
        "schema_version": 2,
        "status": "complete",
        "component": "kernel",
        "model_id": "test/model",
        "baseline": {"total_s": 10.0},
        "frontier_points": [point],
        "pareto_assessment": "latency improvement",
    }
    delivery_path = worktree / "DELIVERY.json"
    delivery_path.write_text(json.dumps(delivery))
    baseline = BaselineRecord(
        model_id="test/model",
        total_s=10.0,
        peak_memory_mib=1024.0,
        timing_scope="frozen_e2e",
        run_dir=baseline_run,
        baseline_frames=baseline_frames,
        sglang_commit=base_commit,
    )
    return {
        "worktree": worktree,
        "campaign": campaign,
        "run_dir": run_dir,
        "baseline": baseline,
        "delivery": delivery,
        "delivery_path": delivery_path,
        "source_hashes": source_hashes,
        "base_commit": base_commit,
        "candidate_commit": candidate_commit,
    }


def make_verifier(
    evidence: dict[str, Any],
    registry: TechniqueRegistry,
    *,
    auditor: AuditSpy | None = None,
    quality: Any = None,
    command_template: FrozenBenchmarkCommand | None = None,
) -> DeliveryVerifier:
    return DeliveryVerifier(
        registry=registry,
        baseline=evidence["baseline"],
        campaign_artifact_root=evidence["campaign"],
        method_auditor=auditor or AuditSpy(),
        quality_evaluator=quality,
        command_template=command_template,
    )


def write_delivery(evidence: dict[str, Any], delivery: dict[str, Any]) -> Path:
    path = evidence["delivery_path"]
    path.write_text(json.dumps(delivery))
    return path


def test_valid_lossless_never_calls_quality_evaluator(
    evidence: dict[str, Any], registry: TechniqueRegistry
) -> None:
    auditor = AuditSpy()
    verifier = make_verifier(
        evidence,
        registry,
        auditor=auditor,
        quality=ForbiddenQualityEvaluator(),
    )
    result = verifier.verify(
        evidence["delivery_path"],
        technique="kernel",
        executor_worktree=evidence["worktree"],
    )
    assert result.accepted, result.findings
    assert result.lossless_required is True
    assert result.verified_points[0].authoritative_speedup == 2.0
    assert result.verified_points[0].activation == {"enable_agent_kernel": True}
    assert result.verified_points[0].source_hashes == evidence["source_hashes"]
    assert auditor.calls == 1


def test_launched_campaign_rejects_candidate_command_drift(
    evidence: dict[str, Any], registry: TechniqueRegistry
) -> None:
    campaign = evidence["campaign"]
    prompts = campaign / "validation-prompts.txt"
    prompts.write_text("\n".join(f"prompt {index}" for index in range(5)) + "\n")
    frozen = {
        "--model-path": "test/model",
        "--dataset": "vbench",
        "--dataset-path": "{{prompts}}",
        "--num-prompts": "5",
    }
    template = FrozenBenchmarkCommand(
        adapter="sglang_diffusion_offline",
        mode="script",
        argv_template=[
            "python",
            "{{benchmark}}",
            "--model-path",
            "test/model",
            "--dataset",
            "vbench",
            "--dataset-path",
            "{{prompts}}",
            "--num-prompts",
            "5",
            "--output-file",
            "{{output_file}}",
            "--output-path",
            "{{media_dir}}",
        ],
        original_cwd=evidence["worktree"],
        original_command_sha256="a" * 64,
        template_sha256="b" * 64,
        frozen_flags=frozen,
    )
    delivery = deepcopy(evidence["delivery"])
    activation = {"env": {"FAST_PATH": "1"}, "server_args": ["--enable-fast"]}
    point = delivery["frontier_points"][0]
    point["activation"] = activation
    point["implementation_manifest"]["activation"] = activation
    (evidence["run_dir"] / "implementation-manifest.json").write_text(
        json.dumps(point["implementation_manifest"])
    )
    argv, environment = template.render(
        checkout=evidence["worktree"],
        prompts=prompts,
        output_file=evidence["run_dir"] / "outputs" / "benchmark.jsonl",
        media_dir=evidence["run_dir"] / "outputs" / "media",
        activation_env=activation["env"],
        activation_args=activation["server_args"],
    )
    command = {
        "argv": list(argv),
        "declared_env": environment,
        "cwd": str(evidence["worktree"]),
        "profile": False,
        "baseline_command_template_sha256": template.template_sha256,
    }
    command_path = evidence["run_dir"] / "COMMAND.json"
    command_path.write_text(json.dumps(command))
    point["artifacts"].append("COMMAND.json")
    delivery_path = write_delivery(evidence, delivery)
    verifier = make_verifier(evidence, registry, command_template=template)

    accepted = verifier.verify(
        delivery_path,
        technique="kernel",
        executor_worktree=evidence["worktree"],
    )
    assert accepted.accepted, accepted.findings

    command["argv"].append("--num-inference-steps=1")
    command_path.write_text(json.dumps(command))
    rejected = verifier.verify(
        delivery_path,
        technique="kernel",
        executor_worktree=evidence["worktree"],
    )
    assert not rejected.accepted
    assert "baseline_command_mismatch" in {
        finding.code for finding in rejected.findings
    }


def test_resolve_inside_and_verifier_reject_path_escape(
    evidence: dict[str, Any], registry: TechniqueRegistry, tmp_path: Path
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(VerificationError, match="escapes"):
        resolve_inside(evidence["worktree"], outside)
    delivery = deepcopy(evidence["delivery"])
    delivery["frontier_points"][0]["run_dir"] = str(outside)
    result = make_verifier(evidence, registry).verify(
        write_delivery(evidence, delivery),
        technique="kernel",
        executor_worktree=evidence["worktree"],
    )
    assert not result.accepted
    assert {finding.code for finding in result.findings} == {"invalid_run_dir"}


def test_recomputes_speedup_and_rejects_tamper(
    evidence: dict[str, Any], registry: TechniqueRegistry
) -> None:
    delivery = deepcopy(evidence["delivery"])
    delivery["frontier_points"][0]["performance"]["speedup"] = 20.0
    result = make_verifier(evidence, registry).verify(
        write_delivery(evidence, delivery),
        technique="kernel",
        executor_worktree=evidence["worktree"],
    )
    assert not result.accepted
    assert "speedup_tamper" in {finding.code for finding in result.findings}


def test_raw_benchmark_must_match_normalized_performance(
    evidence: dict[str, Any], registry: TechniqueRegistry
) -> None:
    performance_path = evidence["run_dir"] / "PERFORMANCE.json"
    performance = json.loads(performance_path.read_text())
    performance["total_s"] = 4.0
    performance_path.write_text(json.dumps(performance))
    delivery = deepcopy(evidence["delivery"])
    point = delivery["frontier_points"][0]["performance"]
    point["candidate_total_s"] = 4.0
    point["speedup"] = 2.5
    result = make_verifier(evidence, registry).verify(
        write_delivery(evidence, delivery),
        technique="kernel",
        executor_worktree=evidence["worktree"],
    )
    assert not result.accepted
    assert "benchmark_performance_mismatch" in {
        finding.code for finding in result.findings
    }


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ("outputs/benchmark.jsonl", "missing_benchmark"),
        ("outputs/media", "missing_media"),
    ],
)
def test_rejects_missing_real_run_evidence(
    evidence: dict[str, Any],
    registry: TechniqueRegistry,
    target: str,
    expected: str,
) -> None:
    target_path = evidence["run_dir"] / target
    if target_path.is_dir():
        for path in target_path.iterdir():
            path.unlink()
    else:
        target_path.unlink()
    result = make_verifier(evidence, registry).verify(
        evidence["delivery_path"],
        technique="kernel",
        executor_worktree=evidence["worktree"],
    )
    assert not result.accepted
    assert expected in {finding.code for finding in result.findings}


def test_rejects_source_hash_mismatch(
    evidence: dict[str, Any], registry: TechniqueRegistry
) -> None:
    source = evidence["worktree"] / next(iter(evidence["source_hashes"]))
    source.write_text("tampered\n")
    result = make_verifier(evidence, registry).verify(
        evidence["delivery_path"],
        technique="kernel",
        executor_worktree=evidence["worktree"],
    )
    assert not result.accepted
    assert "invalid_source_hash" in {finding.code for finding in result.findings}


def test_rejects_zero_engagement_as_noop(
    evidence: dict[str, Any], registry: TechniqueRegistry
) -> None:
    path = evidence["run_dir"] / "engagement-receipt.json"
    receipt = json.loads(path.read_text())
    receipt["techniques"]["kernel"] = {"engaged": False, "call_count": 0}
    path.write_text(json.dumps(receipt))
    result = make_verifier(evidence, registry).verify(
        evidence["delivery_path"],
        technique="kernel",
        executor_worktree=evidence["worktree"],
    )
    assert not result.accepted
    assert "invalid_engagement" in {finding.code for finding in result.findings}
    assert "zero engagement" in " ".join(f.message for f in result.findings)


def make_quality_delivery(evidence: dict[str, Any]) -> Path:
    run_dir = evidence["run_dir"]
    delivery = deepcopy(evidence["delivery"])
    delivery["component"] = "cache"
    point = delivery["frontier_points"][0]
    point["implementation_manifest"]["technique"] = "cache"
    (run_dir / "implementation-manifest.json").write_text(
        json.dumps(point["implementation_manifest"])
    )
    receipt_path = run_dir / "engagement-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["techniques"] = {
        "cache": {"engaged": True, "cache_hit_count": 5, "fallback_count": 0}
    }
    receipt_path.write_text(json.dumps(receipt))
    verdict = {
        "candidate_id": "candidate-1",
        "overall": "pass",
        "producer": "interactive-root-agent",
        "external_api": False,
        "prompt_evidence": [{"prompt": index, "verdict": "pass"} for index in range(5)],
    }
    (run_dir / "visual_verdict.json").write_text(json.dumps(verdict))
    point["quality"] = {
        "mode": "quality_gated",
        "lpips_max": 0.2,
        "lpips_mean": 0.1,
        "visual_overall": "pass",
        "visual_verdict": "visual_verdict.json",
        "relation": "equivalent",
    }
    point["artifacts"].append("visual_verdict.json")
    return write_delivery(evidence, delivery)


def test_quality_path_calls_locked_evaluator_and_visual_gate(
    evidence: dict[str, Any], registry: TechniqueRegistry
) -> None:
    quality = QualitySpy()
    result = make_verifier(evidence, registry, quality=quality).verify(
        make_quality_delivery(evidence),
        technique="cache",
        executor_worktree=evidence["worktree"],
    )
    assert result.accepted, result.findings
    assert result.lossless_required is False
    assert quality.calls == 1


def test_quality_path_rejects_missing_lpips_after_calling_evaluator(
    evidence: dict[str, Any], registry: TechniqueRegistry
) -> None:
    delivery_path = make_quality_delivery(evidence)
    delivery = json.loads(delivery_path.read_text())
    delivery["frontier_points"][0]["quality"]["lpips_mean"] = None
    quality = QualitySpy()
    result = make_verifier(evidence, registry, quality=quality).verify(
        write_delivery(evidence, delivery),
        technique="cache",
        executor_worktree=evidence["worktree"],
    )
    assert not result.accepted
    assert quality.calls == 1
    assert "missing_lpips" in {finding.code for finding in result.findings}


def test_quality_path_reports_lpips_tamper_without_crashing(
    evidence: dict[str, Any], registry: TechniqueRegistry
) -> None:
    delivery_path = make_quality_delivery(evidence)
    delivery = json.loads(delivery_path.read_text())
    delivery["frontier_points"][0]["quality"]["lpips_mean"] = 0.9
    result = make_verifier(evidence, registry, quality=QualitySpy()).verify(
        write_delivery(evidence, delivery),
        technique="cache",
        executor_worktree=evidence["worktree"],
    )
    assert not result.accepted
    assert "lpips_tamper" in {finding.code for finding in result.findings}


def make_topology_delivery(evidence: dict[str, Any]) -> Path:
    run_dir = evidence["run_dir"]
    delivery = deepcopy(evidence["delivery"])
    delivery["component"] = "topology"
    point = delivery["frontier_points"][0]
    point["implementation_manifest"]["technique"] = "topology"
    (run_dir / "implementation-manifest.json").write_text(
        json.dumps(point["implementation_manifest"])
    )
    receipt_path = run_dir / "engagement-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["techniques"] = {
        "topology": {"engaged": True, "rank_count": 2, "fallback_count": 0}
    }
    receipt_path.write_text(json.dumps(receipt))
    common = {"candidate_id": "candidate-1", "run_id": "run-1"}
    documents = {
        "topology_preflight.json": {**common, "checks": {"nccl": True, "ranks": True}},
        "topology_manifest.json": {
            **common,
            "groups": [{"name": "world", "ranks": [0, 1]}],
            "rank_map": {"0": "cuda:0", "1": "cuda:1"},
            "collectives": ["all_reduce"],
            "source_hashes": evidence["source_hashes"],
            "fallback_count": 0,
        },
        "topology_trace.json": {
            **common,
            "world_size": 2,
            "ranks": [
                {"rank": 0, "participated": True, "timing_ms": 4.0, "memory_mib": 100},
                {"rank": 1, "participated": True, "timing_ms": 4.1, "memory_mib": 101},
            ],
            "fallback_count": 0,
        },
    }
    for name, document in documents.items():
        (run_dir / name).write_text(json.dumps(document))
        point["artifacts"].append(name)
    return write_delivery(evidence, delivery)


def test_topology_requires_consistent_durable_artifacts(
    evidence: dict[str, Any], registry: TechniqueRegistry
) -> None:
    delivery_path = make_topology_delivery(evidence)
    verifier = make_verifier(evidence, registry)
    accepted = verifier.verify(
        delivery_path,
        technique="topology",
        executor_worktree=evidence["worktree"],
    )
    assert accepted.accepted, accepted.findings

    trace_path = evidence["run_dir"] / "topology_trace.json"
    trace = json.loads(trace_path.read_text())
    trace["run_id"] = "fabricated-run"
    trace_path.write_text(json.dumps(trace))
    rejected = verifier.verify(
        delivery_path,
        technique="topology",
        executor_worktree=evidence["worktree"],
    )
    assert not rejected.accepted
    assert "topology_run_mismatch" in {finding.code for finding in rejected.findings}
