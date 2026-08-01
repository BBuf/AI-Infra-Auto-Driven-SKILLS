from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from pydantic import ValidationError

from .agents import redact_argv, redact_environment
from .models import (
    BaselineRecord,
    CandidateManifest,
    Delivery,
    EngagementReceipt,
    FrontierPoint,
    KernelEvidence,
)
from .process import run
from .request import FrozenBenchmarkCommand
from .techniques import TechniqueRegistry


_FALLBACK_MARKERS = (
    "falling back to diffusers backend",
    "using diffusers backend",
    "loaded diffusers pipeline",
)
_MEDIA_SUFFIXES = {
    ".avi",
    ".gif",
    ".jpeg",
    ".jpg",
    ".mkv",
    ".mov",
    ".mp4",
    ".png",
    ".webm",
    ".webp",
}
_IMPLEMENTATION_NAMES = (
    "implementation-manifest.json",
    "implementation_manifest.json",
    "IMPLEMENTATION.json",
)
_SOURCE_HASH_NAMES = (
    "source-hashes.json",
    "source_hashes.json",
    "SOURCE-HASHES.json",
)
_ENGAGEMENT_NAMES = (
    "engagement-receipt.json",
    "engagement_receipt.json",
    "ENGAGEMENT.json",
)
_KERNEL_EVIDENCE_NAMES = ("KERNEL-EVIDENCE.json", "kernel-evidence.json")


class VerificationError(RuntimeError):
    """Raised for unsafe paths or malformed durable evidence."""


class MethodEquivalenceAuditor(Protocol):
    """Independent code/method auditor used only by lossless techniques."""

    def audit(
        self,
        *,
        technique: str,
        executor_worktree: Path,
        manifest: CandidateManifest,
        equivalence: Mapping[str, Any],
    ) -> bool | str | Sequence[str]: ...


class QualityEvaluator(Protocol):
    """Adapter around the locked Sol plan-eval command."""

    def assess(
        self,
        *,
        baseline_frames: Path,
        candidate_frames: Path,
        run_dir: Path,
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class VerificationFinding:
    code: str
    message: str
    candidate_id: str | None = None


@dataclass(frozen=True)
class VerifiedFrontierPoint:
    candidate_id: str
    run_dir: Path
    authoritative_speedup: float
    candidate_total_s: float
    peak_memory_mib: float
    activation: dict[str, Any]
    implementation_manifest: CandidateManifest
    source_hashes: dict[str, str]


@dataclass(frozen=True)
class VerificationResult:
    accepted: bool
    technique: str
    findings: tuple[VerificationFinding, ...]
    verified_points: tuple[VerifiedFrontierPoint, ...]
    lossless_required: bool


def resolve_inside(root: Path, relative: Path) -> Path:
    """Resolve an existing path below *root*, rejecting `..` and symlink escape."""

    root = root.resolve()
    candidate = (
        relative.resolve() if relative.is_absolute() else (root / relative).resolve()
    )
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise VerificationError(f"artifact escapes allowed root: {relative}") from error
    if not candidate.exists():
        raise VerificationError(f"missing artifact: {relative}")
    return candidate


class DeliveryVerifier:
    """Independently verify an executor delivery without trusting its numbers."""

    def __init__(
        self,
        *,
        registry: TechniqueRegistry,
        baseline: BaselineRecord,
        campaign_artifact_root: Path,
        method_auditor: MethodEquivalenceAuditor,
        quality_evaluator: QualityEvaluator | None = None,
        command_template: FrozenBenchmarkCommand | None = None,
    ) -> None:
        self.registry = registry
        self.baseline = baseline
        self.campaign_artifact_root = campaign_artifact_root.resolve()
        self.method_auditor = method_auditor
        self.quality_evaluator = quality_evaluator
        self.command_template = command_template

    def verify(
        self,
        delivery_path: Path,
        *,
        technique: str,
        executor_worktree: Path,
    ) -> VerificationResult:
        contract = self.registry[technique]
        lossless = contract.correctness == "lossless"
        worktree = executor_worktree.resolve()
        findings: list[VerificationFinding] = []
        verified: list[VerifiedFrontierPoint] = []

        try:
            durable_delivery = self._resolve_allowed(
                delivery_path, (worktree, self.campaign_artifact_root)
            )
            if not durable_delivery.is_file():
                raise VerificationError(f"delivery is not a file: {durable_delivery}")
            delivery = Delivery.model_validate_json(
                durable_delivery.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, ValidationError, VerificationError) as error:
            findings.append(
                VerificationFinding("invalid_delivery", str(error), candidate_id=None)
            )
            return VerificationResult(
                accepted=False,
                technique=technique,
                findings=tuple(findings),
                verified_points=(),
                lossless_required=lossless,
            )

        if delivery.component != technique:
            findings.append(
                VerificationFinding(
                    "technique_mismatch",
                    f"delivery component {delivery.component!r} is not {technique!r}",
                )
            )
        if delivery.model_id != self.baseline.model_id:
            findings.append(
                VerificationFinding(
                    "model_mismatch",
                    "delivery model_id differs from the frozen baseline",
                )
            )
        if not delivery.frontier_points:
            findings.append(
                VerificationFinding(
                    "empty_frontier", "delivery contains no frontier points"
                )
            )

        for point in delivery.frontier_points:
            point_findings, verified_point = self._verify_point(
                point,
                technique=technique,
                lossless=lossless,
                executor_worktree=worktree,
            )
            findings.extend(point_findings)
            if not point_findings and verified_point is not None:
                verified.append(verified_point)

        return VerificationResult(
            accepted=not findings and len(verified) == len(delivery.frontier_points),
            technique=technique,
            findings=tuple(findings),
            verified_points=tuple(verified),
            lossless_required=lossless,
        )

    def _verify_point(
        self,
        point: FrontierPoint,
        *,
        technique: str,
        lossless: bool,
        executor_worktree: Path,
    ) -> tuple[list[VerificationFinding], VerifiedFrontierPoint | None]:
        candidate_id = point.candidate_id
        issues: list[VerificationFinding] = []

        def issue(code: str, message: str) -> None:
            issues.append(VerificationFinding(code, message, candidate_id))

        roots = (executor_worktree, self.campaign_artifact_root)
        try:
            run_dir = self._resolve_allowed(point.run_dir, roots)
            if not run_dir.is_dir():
                raise VerificationError(f"run_dir is not a directory: {run_dir}")
            baseline_run = self._resolve_allowed(self.baseline.run_dir, roots)
            if run_dir == baseline_run:
                raise VerificationError(
                    "candidate run_dir resubmits the frozen baseline"
                )
        except VerificationError as error:
            issue("invalid_run_dir", str(error))
            return issues, None

        artifacts: list[Path] = []
        for artifact in point.artifacts:
            try:
                artifacts.append(self._resolve_point_path(artifact, run_dir, roots))
            except VerificationError as error:
                issue("invalid_artifact", str(error))

        if self.command_template is not None:
            try:
                self._verify_frozen_command(
                    point,
                    run_dir=run_dir,
                    artifacts=artifacts,
                    executor_worktree=executor_worktree,
                )
            except VerificationError as error:
                issue("baseline_command_mismatch", str(error))

        try:
            performance_path = self._required_artifact(
                run_dir, artifacts, ("PERFORMANCE.json",)
            )
            performance = self._json_object(performance_path)
        except VerificationError as error:
            issue("missing_performance", str(error))
            performance = {}

        try:
            benchmark_path = self._required_benchmark(run_dir, artifacts)
            benchmark = self._validate_benchmark(benchmark_path)
        except VerificationError as error:
            issue("missing_benchmark", str(error))
            benchmark = {}

        try:
            candidate_frames = self._required_media(run_dir, artifacts)
        except VerificationError as error:
            issue("missing_media", str(error))
            candidate_frames = run_dir

        manifest: CandidateManifest | None = None
        try:
            manifest_path = self._required_artifact(
                run_dir, artifacts, _IMPLEMENTATION_NAMES
            )
            manifest_raw = self._json_object(manifest_path)
            if manifest_raw != point.implementation_manifest:
                raise VerificationError(
                    "inline implementation_manifest differs from durable manifest"
                )
            manifest = CandidateManifest.model_validate(manifest_raw)
            if manifest.candidate_id != candidate_id:
                raise VerificationError("manifest candidate_id does not match delivery")
            if manifest.technique != technique:
                raise VerificationError(
                    "manifest technique does not match verifier lane"
                )
            if manifest.kind != "patch":
                raise VerificationError("frontier point must contain a real patch")
            if manifest.base_commit == manifest.candidate_commit:
                raise VerificationError("candidate commit is unchanged from its base")
            if manifest.base_commit != self.baseline.sglang_commit:
                raise VerificationError(
                    "manifest base_commit differs from frozen SGLang commit"
                )
            if manifest.activation != point.activation:
                raise VerificationError(
                    "manifest activation differs from delivery activation"
                )
            if manifest.eval_profile.get("timing_scope") != self.baseline.timing_scope:
                raise VerificationError(
                    "manifest eval_profile timing scope differs from baseline"
                )
            self._verify_candidate_commit(manifest, executor_worktree)
        except (ValidationError, VerificationError) as error:
            issue("invalid_implementation", str(error))

        source_hashes: dict[str, str] = {}
        try:
            source_hash_path = self._required_artifact(
                run_dir, artifacts, _SOURCE_HASH_NAMES
            )
            source_hashes = self._load_source_hashes(source_hash_path)
            self._verify_source_hashes(
                source_hashes, executor_worktree, manifest=manifest
            )
        except VerificationError as error:
            issue("invalid_source_hash", str(error))

        engagement: EngagementReceipt | None = None
        try:
            engagement_path = self._required_artifact(
                run_dir, artifacts, _ENGAGEMENT_NAMES
            )
            engagement = EngagementReceipt.model_validate(
                self._json_object(engagement_path)
            )
            self._verify_engagement(
                engagement,
                candidate_id=candidate_id,
                technique=technique,
                source_hashes=source_hashes,
                activation=point.activation,
            )
        except (ValidationError, VerificationError) as error:
            issue("invalid_engagement", str(error))

        authoritative = self._verify_performance(point, performance, benchmark, issue)
        self._reject_fallback(run_dir, performance, engagement, issue)

        if lossless:
            self._verify_lossless(
                point,
                run_dir=run_dir,
                artifacts=artifacts,
                manifest=manifest,
                technique=technique,
                executor_worktree=executor_worktree,
                issue=issue,
            )
        else:
            self._verify_quality(
                point,
                run_dir=run_dir,
                candidate_frames=candidate_frames,
                issue=issue,
            )

        if technique == "topology":
            self._verify_topology(
                run_dir,
                artifacts,
                candidate_id=candidate_id,
                source_hashes=source_hashes,
                issue=issue,
            )
        if technique == "kernel":
            self._verify_kernel_evidence(
                run_dir,
                artifacts,
                candidate_id=candidate_id,
                executor_worktree=executor_worktree,
                issue=issue,
            )

        if issues or authoritative is None or manifest is None:
            return issues, None
        speedup, candidate_total, peak_memory = authoritative
        return issues, VerifiedFrontierPoint(
            candidate_id=candidate_id,
            run_dir=run_dir,
            authoritative_speedup=speedup,
            candidate_total_s=candidate_total,
            peak_memory_mib=peak_memory,
            activation=dict(point.activation),
            implementation_manifest=manifest,
            source_hashes=dict(source_hashes),
        )

    def _verify_kernel_evidence(
        self,
        run_dir: Path,
        artifacts: Sequence[Path],
        *,
        candidate_id: str,
        executor_worktree: Path,
        issue: Any,
    ) -> None:
        try:
            evidence_path = self._required_artifact(
                run_dir, artifacts, _KERNEL_EVIDENCE_NAMES
            )
            evidence = KernelEvidence.model_validate(self._json_object(evidence_path))
            if evidence.candidate_id != candidate_id:
                raise VerificationError(
                    "kernel evidence candidate_id differs from the delivery"
                )
            if evidence.candidate_family not in self.registry["kernel"].coverage:
                raise VerificationError(
                    "kernel evidence candidate_family is outside the reviewed coverage"
                )
            profile = (
                self.campaign_artifact_root
                / "profiles"
                / "0"
                / "PROFILE-DIGEST.json"
            )
            if not profile.is_file():
                raise VerificationError("campaign raw profile digest is missing")
            if self._sha256_file(profile) != evidence.profile_digest_sha256:
                raise VerificationError(
                    "kernel evidence is not bound to the active profile digest"
                )

            knowledge_root = self.campaign_artifact_root / "knowledge" / "kda_pilot"
            pinned_kernelwiki = self._pinned_kernelwiki_files(knowledge_root)
            for source in evidence.kernelwiki.sources:
                resolved = self._verify_evidence_file(
                    source.path,
                    source.sha256,
                    roots=(knowledge_root,),
                )
                pinned_digest = pinned_kernelwiki.get(resolved)
                if pinned_digest is None:
                    raise VerificationError(
                        "KernelWiki citation is absent from the pinned knowledge index"
                    )
                if pinned_digest != source.sha256:
                    raise VerificationError(
                        "KernelWiki citation hash differs from the pinned index"
                    )

            roots = (run_dir, executor_worktree, self.campaign_artifact_root)
            files = [evidence.microbenchmark]
            files.extend(
                item
                for item in (
                    evidence.ncu.before_report,
                    evidence.ncu.after_report,
                    evidence.ncu.metrics_digest,
                    evidence.warp_specialization.timeline_report,
                    evidence.warp_specialization.reconciliation,
                )
                if item is not None
            )
            resolved_files = [
                self._verify_evidence_file(item.path, item.sha256, roots=roots)
                for item in files
            ]
            if evidence.ncu.applicable:
                assert evidence.ncu.before_report is not None
                assert evidence.ncu.after_report is not None
                before = self._resolve_allowed(evidence.ncu.before_report.path, roots)
                after = self._resolve_allowed(evidence.ncu.after_report.path, roots)
                if before == after or before.suffix != ".ncu-rep" or after.suffix != ".ncu-rep":
                    raise VerificationError(
                        "NCU before/after evidence must be distinct .ncu-rep files"
                    )
            if any(path.stat().st_size <= 0 for path in resolved_files):
                raise VerificationError("kernel evidence references an empty artifact")
        except (ValidationError, VerificationError) as error:
            issue("invalid_kernel_evidence", str(error))

    def _verify_evidence_file(
        self,
        path: Path,
        expected_sha256: str,
        *,
        roots: Sequence[Path],
    ) -> Path:
        resolved = self._resolve_allowed(path, roots)
        if not resolved.is_file() or resolved.is_symlink():
            raise VerificationError(f"evidence is not a regular file: {path}")
        if self._sha256_file(resolved) != expected_sha256:
            raise VerificationError(f"evidence SHA-256 mismatch: {path}")
        return resolved

    def _pinned_kernelwiki_files(self, knowledge_root: Path) -> dict[Path, str]:
        manifest_path = self.campaign_artifact_root / "KNOWLEDGE.json"
        try:
            manifest = self._json_object(manifest_path)
            snapshots = manifest.get("snapshots")
            if not isinstance(snapshots, Mapping):
                raise VerificationError("campaign knowledge manifest is malformed")
            raw_index = snapshots.get("kda_pilot")
            if not isinstance(raw_index, str):
                raise VerificationError("campaign has no pinned KDA-Pilot snapshot")
            index_path = self._resolve_allowed(Path(raw_index), (knowledge_root,))
            index = self._json_object(index_path)
            entries = index.get("entries")
            if not isinstance(entries, list):
                raise VerificationError("KDA-Pilot knowledge index has no entries")
        except (OSError, VerificationError) as error:
            raise VerificationError(f"cannot load pinned KernelWiki index: {error}") from error
        result: dict[Path, str] = {}
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            relative = entry.get("path")
            digest = entry.get("reference_sha256")
            if (
                not isinstance(relative, str)
                or not relative.startswith("external/KernelWiki/")
                or not isinstance(digest, str)
            ):
                continue
            reference = self._resolve_allowed(
                index_path.parent / "references" / relative,
                (knowledge_root,),
            )
            result[reference] = digest
        if not result:
            raise VerificationError("pinned KDA-Pilot snapshot contains no KernelWiki files")
        return result

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _verify_frozen_command(
        self,
        point: FrontierPoint,
        *,
        run_dir: Path,
        artifacts: Sequence[Path],
        executor_worktree: Path,
    ) -> None:
        assert self.command_template is not None
        receipt_path = self._required_artifact(run_dir, artifacts, ("COMMAND.json",))
        receipt = self._json_object(receipt_path)
        activation_env = point.activation.get("env", {})
        activation_args = point.activation.get("server_args", [])
        if not isinstance(activation_env, Mapping) or not isinstance(
            activation_args, list
        ):
            raise VerificationError(
                "candidate activation must contain env and server_args"
            )
        prompts = self.campaign_artifact_root / "validation-prompts.txt"
        if not prompts.is_file():
            raise VerificationError("frozen validation prompt file is missing")
        expected_argv, expected_env = self.command_template.render(
            checkout=executor_worktree,
            prompts=prompts,
            output_file=run_dir / "outputs" / "benchmark.jsonl",
            media_dir=run_dir / "outputs" / "media",
            activation_env={
                str(name): str(value) for name, value in activation_env.items()
            },
            activation_args=[str(value) for value in activation_args],
        )
        expected = {
            "argv": redact_argv(list(expected_argv)),
            "declared_env": redact_environment(expected_env),
            "cwd": str(executor_worktree.resolve()),
            "profile": False,
            "baseline_command_template_sha256": (self.command_template.template_sha256),
        }
        mismatches = [
            name for name, value in expected.items() if receipt.get(name) != value
        ]
        if mismatches:
            raise VerificationError(
                "candidate command differs from the frozen user baseline "
                "template: " + ", ".join(mismatches)
            )

    def _verify_performance(
        self,
        point: FrontierPoint,
        performance: Mapping[str, Any],
        benchmark: Mapping[str, Any],
        issue: Any,
    ) -> tuple[float, float, float] | None:
        candidate_total = self._positive_number(performance.get("total_s"))
        peak_memory = self._positive_number(performance.get("peak_memory_mib"))
        if candidate_total is None or peak_memory is None:
            issue(
                "invalid_performance",
                "PERFORMANCE.json requires positive total_s and peak_memory_mib",
            )
            return None
        benchmark_total = self._positive_number(benchmark.get("total_s"))
        benchmark_peak = self._positive_number(benchmark.get("peak_memory_mib"))
        if benchmark_total is None or benchmark_peak is None:
            issue(
                "invalid_benchmark_metrics",
                "benchmark requires positive latency and peak memory",
            )
            return None
        if not math.isclose(
            candidate_total,
            benchmark_total,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ) or not math.isclose(
            peak_memory,
            benchmark_peak,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            issue(
                "benchmark_performance_mismatch",
                "PERFORMANCE.json differs from the raw benchmark result",
            )
        candidate_total = benchmark_total
        peak_memory = benchmark_peak
        timing_scope = performance.get("timing_scope")
        if timing_scope != self.baseline.timing_scope:
            issue(
                "timing_scope_mismatch",
                "candidate timing_scope differs from the frozen baseline",
            )
        if not math.isclose(
            point.performance.baseline_total_s,
            self.baseline.total_s,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            issue(
                "baseline_tamper",
                "reported baseline_total_s differs from the frozen baseline",
            )
        if not math.isclose(
            point.performance.candidate_total_s,
            candidate_total,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            issue(
                "candidate_tamper",
                "reported candidate_total_s differs from PERFORMANCE.json",
            )
        speedup = self.baseline.total_s / candidate_total
        if not math.isclose(
            speedup,
            point.performance.speedup,
            rel_tol=1e-6,
            abs_tol=1e-9,
        ):
            issue(
                "speedup_tamper",
                "reported speedup does not match durable benchmark",
            )
        if point.performance.frontier_axis == "latency":
            if candidate_total >= self.baseline.total_s:
                issue(
                    "no_improvement",
                    "latency frontier point does not improve frozen baseline latency",
                )
        elif peak_memory >= self.baseline.peak_memory_mib:
            issue(
                "dominated_memory",
                "memory frontier point does not improve frozen baseline memory",
            )
        return speedup, candidate_total, peak_memory

    def _verify_lossless(
        self,
        point: FrontierPoint,
        *,
        run_dir: Path,
        artifacts: Sequence[Path],
        manifest: CandidateManifest | None,
        technique: str,
        executor_worktree: Path,
        issue: Any,
    ) -> None:
        try:
            equivalence_path = self._required_artifact(
                run_dir, artifacts, ("equivalence.json",)
            )
            equivalence = self._json_object(equivalence_path)
            self._validate_equivalence(equivalence, point.candidate_id)
        except VerificationError as error:
            issue("invalid_equivalence", str(error))
            return

        try:
            authenticity = self._json_object(
                self._resolve_point_path(
                    point.quality.visual_verdict,
                    run_dir,
                    (executor_worktree, self.campaign_artifact_root),
                )
            )
            if not (
                authenticity.get("authentic") is True
                or authenticity.get("overall") in {"pass", "authenticity_only"}
            ):
                raise VerificationError("lossless authenticity receipt did not pass")
        except VerificationError as error:
            issue("invalid_authenticity", str(error))

        if manifest is None:
            return
        try:
            result = self.method_auditor.audit(
                technique=technique,
                executor_worktree=executor_worktree,
                manifest=manifest,
                equivalence=equivalence,
            )
            audit_issues = self._auditor_issues(result)
            for message in audit_issues:
                issue("method_equivalence_rejected", message)
        except Exception as error:  # auditor failure must fail closed
            issue("method_auditor_failed", str(error))

    def _verify_quality(
        self,
        point: FrontierPoint,
        *,
        run_dir: Path,
        candidate_frames: Path,
        issue: Any,
    ) -> None:
        if point.quality.mode != "quality_gated":
            issue(
                "quality_mode_mismatch", "quality-gated technique is marked not_gated"
            )
        if point.quality.lpips_mean is None or point.quality.lpips_max is None:
            issue("missing_lpips", "quality-gated point requires aligned LPIPS metrics")
        if self.quality_evaluator is None:
            issue(
                "missing_quality_evaluator",
                "locked Sol quality evaluator is unavailable",
            )
            return
        try:
            baseline_frames = resolve_inside(
                self.campaign_artifact_root, self.baseline.baseline_frames
            )
            assessed = self.quality_evaluator.assess(
                baseline_frames=baseline_frames,
                candidate_frames=candidate_frames,
                run_dir=run_dir,
            )
        except Exception as error:
            issue("quality_evaluator_failed", str(error))
            return

        if assessed.get("aligned") is not True:
            issue(
                "unaligned_lpips", "quality evaluator did not confirm frame alignment"
            )
        prompt_scores = assessed.get("prompt_scores")
        if not isinstance(prompt_scores, list) or len(prompt_scores) < 5:
            issue(
                "missing_prompt_scores",
                "quality evidence must include all five prompt-level scores",
            )
        self._compare_quality_metric(
            "lpips_mean", point.quality.lpips_mean, assessed.get("lpips_mean"), issue
        )
        self._compare_quality_metric(
            "lpips_max", point.quality.lpips_max, assessed.get("lpips_max"), issue
        )

        try:
            verdict_path = self._resolve_point_path(
                point.quality.visual_verdict,
                run_dir,
                (self.campaign_artifact_root,),
            )
            verdict = self._json_object(verdict_path)
            if (
                verdict.get("overall") != "pass"
                or point.quality.visual_overall != "pass"
            ):
                raise VerificationError(
                    "built-in multimodal visual verdict did not pass"
                )
            if verdict.get("producer") != "coding-agent-built-in-vision":
                raise VerificationError(
                    "visual verdict was not produced by coding-agent built-in vision"
                )
            if verdict.get("external_api") is not False:
                raise VerificationError("external visual API verdict is disallowed")
            evidence = verdict.get("prompt_evidence")
            if not isinstance(evidence, list) or len(evidence) < 5:
                raise VerificationError("visual verdict lacks prompt-level evidence")
            if assessed.get("visual_overall") != "pass":
                raise VerificationError(
                    "independent built-in multimodal assessment did not pass"
                )
            assessed_digest = assessed.get("visual_verdict_sha256")
            actual_digest = hashlib.sha256(verdict_path.read_bytes()).hexdigest()
            if assessed_digest != actual_digest:
                raise VerificationError(
                    "visual verdict is not the one approved by the "
                    "independent quality evaluator"
                )
        except VerificationError as error:
            issue("invalid_visual_verdict", str(error))

    def _verify_topology(
        self,
        run_dir: Path,
        artifacts: Sequence[Path],
        *,
        candidate_id: str,
        source_hashes: Mapping[str, str],
        issue: Any,
    ) -> None:
        required = (
            "topology_preflight.json",
            "topology_manifest.json",
            "topology_trace.json",
            "equivalence.json",
        )
        documents: dict[str, Mapping[str, Any]] = {}
        for name in required:
            try:
                document = self._json_object(
                    self._required_artifact(run_dir, artifacts, (name,))
                )
                if document.get("candidate_id") != candidate_id:
                    raise VerificationError(f"{name} candidate_id is inconsistent")
                run_id = document.get("run_id")
                if not isinstance(run_id, str) or not run_id:
                    raise VerificationError(f"{name} lacks a durable run_id")
                documents[name] = document
            except VerificationError as error:
                issue("invalid_topology_artifact", str(error))
        if len(documents) != len(required):
            return
        run_ids = {document["run_id"] for document in documents.values()}
        if len(run_ids) != 1:
            issue(
                "topology_run_mismatch",
                "topology artifacts do not share the same run_id",
            )

        preflight = documents["topology_preflight.json"]
        checks = preflight.get("checks")
        if (
            not isinstance(checks, Mapping)
            or not checks
            or any(value is not True for value in checks.values())
        ):
            issue("topology_preflight_failed", "not all topology preflight checks pass")

        manifest = documents["topology_manifest.json"]
        for field in ("groups", "rank_map", "collectives"):
            value = manifest.get(field)
            if not isinstance(value, (list, dict)) or not value:
                issue("invalid_topology_manifest", f"topology manifest lacks {field}")
        if manifest.get("source_hashes") != dict(source_hashes):
            issue(
                "topology_source_mismatch",
                "topology manifest source hashes differ from verified hashes",
            )

        trace = documents["topology_trace.json"]
        world_size = trace.get("world_size")
        ranks = trace.get("ranks")
        if (
            not isinstance(world_size, int)
            or isinstance(world_size, bool)
            or world_size <= 0
            or not isinstance(ranks, list)
        ):
            issue("invalid_topology_trace", "trace lacks a valid world_size/rank list")
            return
        observed: set[int] = set()
        for rank in ranks:
            if not isinstance(rank, Mapping):
                issue("invalid_topology_trace", "rank evidence must be an object")
                continue
            rank_id = rank.get("rank")
            if not isinstance(rank_id, int) or isinstance(rank_id, bool):
                issue("invalid_topology_trace", "rank evidence lacks integer rank")
                continue
            if rank_id in observed:
                issue("invalid_topology_trace", f"duplicate rank evidence: {rank_id}")
            observed.add(rank_id)
            if rank.get("participated") is not True:
                issue("invalid_topology_trace", f"rank {rank_id} did not participate")
            if self._positive_number(rank.get("timing_ms")) is None:
                issue(
                    "invalid_topology_trace", f"rank {rank_id} has no positive timing"
                )
            if self._positive_number(rank.get("memory_mib")) is None:
                issue(
                    "invalid_topology_trace", f"rank {rank_id} has no positive memory"
                )
        if observed != set(range(world_size)):
            issue(
                "topology_rank_coverage",
                "topology trace does not contain exactly one record for every rank",
            )
        if (
            self._has_fallback(preflight)
            or self._has_fallback(manifest)
            or self._has_fallback(trace)
        ):
            issue("topology_fallback", "topology evidence reports a silent fallback")

    def _verify_engagement(
        self,
        receipt: EngagementReceipt,
        *,
        candidate_id: str,
        technique: str,
        source_hashes: Mapping[str, str],
        activation: Mapping[str, Any],
    ) -> None:
        if receipt.profile_id != candidate_id:
            raise VerificationError("engagement profile_id does not match candidate_id")
        if not (
            receipt.model_match and receipt.hardware_match and receipt.workload_match
        ):
            raise VerificationError(
                "engagement receipt does not match frozen evaluation"
            )
        if not activation:
            raise VerificationError("candidate has no requested activation")
        evidence = receipt.techniques.get(technique)
        if not isinstance(evidence, Mapping):
            raise VerificationError(f"engagement receipt lacks {technique!r}")
        if not self._positive_engagement(evidence):
            raise VerificationError("requested technique has zero engagement")
        if self._has_fallback(evidence):
            raise VerificationError("engagement receipt reports a disallowed fallback")
        if dict(receipt.source_hashes) != dict(source_hashes):
            raise VerificationError(
                "engagement source hashes differ from verified hashes"
            )

    @staticmethod
    def _validate_equivalence(
        equivalence: Mapping[str, Any], candidate_id: str
    ) -> None:
        declared = equivalence.get("candidate_id")
        if declared is not None and declared != candidate_id:
            raise VerificationError("equivalence candidate_id does not match delivery")
        method_argument = equivalence.get("method_argument")
        if not isinstance(method_argument, str) or not method_argument.strip():
            raise VerificationError("equivalence lacks a nonempty method_argument")
        baseline = equivalence.get("baseline")
        candidate = equivalence.get("candidate")
        if not isinstance(baseline, Mapping) or not isinstance(candidate, Mapping):
            raise VerificationError(
                "equivalence lacks baseline/candidate logical counts"
            )
        for field in ("global_steps", "dit_calls"):
            before = baseline.get(field)
            after = candidate.get(field)
            if (
                not isinstance(before, int)
                or isinstance(before, bool)
                or not isinstance(after, int)
                or isinstance(after, bool)
                or before <= 0
                or after != before
            ):
                raise VerificationError(
                    f"lossless equivalence does not preserve {field}"
                )
        if equivalence.get("logical_work_unchanged") is not True:
            raise VerificationError("logical model work is not declared unchanged")
        for forbidden in (
            "approximation",
            "step_skipping",
            "sparsity",
            "sub_16bit",
            "rank_reduction",
        ):
            if equivalence.get(forbidden) not in (False, None):
                raise VerificationError(f"lossless candidate enables {forbidden}")

    @staticmethod
    def _auditor_issues(result: bool | str | Sequence[str]) -> list[str]:
        if result is True:
            return []
        if result is False:
            return ["independent method-equivalence audit rejected the candidate"]
        if isinstance(result, str):
            return [result] if result else ["method auditor returned an empty verdict"]
        messages = [str(value) for value in result if str(value)]
        return messages or ["method auditor did not return an affirmative verdict"]

    @staticmethod
    def _compare_quality_metric(
        name: str, reported: float | None, assessed: Any, issue: Any
    ) -> None:
        assessed_number = DeliveryVerifier._positive_or_zero_number(assessed)
        if (
            reported is None
            or assessed_number is None
            or not math.isclose(
                float(reported), assessed_number, rel_tol=1e-6, abs_tol=1e-9
            )
        ):
            issue(
                "lpips_tamper",
                f"reported {name} differs from locked evaluator assessment",
            )

    def _resolve_allowed(self, path: Path, roots: Sequence[Path]) -> Path:
        errors: list[str] = []
        for root in roots:
            try:
                return resolve_inside(root, path)
            except VerificationError as error:
                errors.append(str(error))
        raise VerificationError(
            f"path is missing or outside allowed roots: {path} ({'; '.join(errors)})"
        )

    def _resolve_point_path(
        self, path: Path, run_dir: Path, roots: Sequence[Path]
    ) -> Path:
        if not path.is_absolute():
            try:
                return resolve_inside(run_dir, path)
            except VerificationError:
                pass
        return self._resolve_allowed(path, roots)

    @staticmethod
    def _required_artifact(
        run_dir: Path, artifacts: Sequence[Path], names: Sequence[str]
    ) -> Path:
        for artifact in artifacts:
            if artifact.name in names:
                return artifact
        for name in names:
            candidate = run_dir / name
            if candidate.exists():
                return resolve_inside(run_dir, Path(name))
        names_text = ", ".join(names)
        raise VerificationError(f"required artifact is missing: {names_text}")

    @staticmethod
    def _required_benchmark(run_dir: Path, artifacts: Sequence[Path]) -> Path:
        for artifact in artifacts:
            if artifact.name in {"benchmark.json", "benchmark.jsonl"}:
                return artifact
        for relative in (
            Path("outputs/benchmark.json"),
            Path("outputs/benchmark.jsonl"),
        ):
            candidate = run_dir / relative
            if candidate.exists():
                return resolve_inside(run_dir, relative)
        raise VerificationError("required benchmark artifact is missing")

    @staticmethod
    def _validate_benchmark(path: Path) -> dict[str, float]:
        if not path.is_file() or path.stat().st_size == 0:
            raise VerificationError("benchmark artifact is empty or not a file")
        text = path.read_text(encoding="utf-8")
        objects = []
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise VerificationError(
                    f"benchmark JSON is invalid: {error}"
                ) from error
            if isinstance(value, Mapping):
                objects.append(value)
        if not objects:
            raise VerificationError("benchmark contains no JSON object evidence")
        raw = objects[-1]
        results = raw.get("results", raw)
        if not isinstance(results, Mapping):
            raise VerificationError("benchmark results must be an object")
        successful = results.get("successful_requests")
        failed = results.get("failed_requests")
        if successful is not None and successful != 5:
            raise VerificationError(
                f"benchmark has {successful!r} successful requests, expected 5"
            )
        if failed is not None and failed != 0:
            raise VerificationError(f"benchmark reports {failed!r} failed requests")
        total = DeliveryVerifier._first_positive(
            results, ("total_duration_seconds", "total_s", "latency")
        )
        peak = DeliveryVerifier._first_positive(
            results, ("peak_memory_mb", "peak_memory_mib")
        )
        if total is None or peak is None:
            raise VerificationError(
                "benchmark lacks positive latency or peak-memory metrics"
            )
        return {"total_s": total, "peak_memory_mib": peak}

    @staticmethod
    def _required_media(run_dir: Path, artifacts: Sequence[Path]) -> Path:
        candidates = list(artifacts)
        for relative in (Path("outputs/media"), Path("outputs/frames")):
            candidate = run_dir / relative
            if candidate.exists():
                candidates.append(resolve_inside(run_dir, relative))
        for candidate in candidates:
            if candidate.is_file():
                media_files = (
                    [candidate] if candidate.suffix.lower() in _MEDIA_SUFFIXES else []
                )
            elif candidate.is_dir():
                media_files = [
                    path
                    for path in candidate.rglob("*")
                    if path.is_file() and path.suffix.lower() in _MEDIA_SUFFIXES
                ]
            else:
                media_files = []
            if len(media_files) >= 5:
                return candidate
        raise VerificationError(
            "run has fewer than five durable media outputs/aligned prompt frames"
        )

    @staticmethod
    def _json_object(path: Path) -> dict[str, Any]:
        if not path.is_file():
            raise VerificationError(f"JSON artifact is not a file: {path}")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise VerificationError(f"invalid JSON artifact {path}: {error}") from error
        if not isinstance(value, dict):
            raise VerificationError(f"JSON artifact must contain an object: {path}")
        return value

    @classmethod
    def _load_source_hashes(cls, path: Path) -> dict[str, str]:
        value = cls._json_object(path)
        raw = value.get("source_hashes", value)
        if not isinstance(raw, Mapping) or not raw:
            raise VerificationError("source hash artifact is empty")
        hashes = {str(name): str(digest) for name, digest in raw.items()}
        if any(
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            for digest in hashes.values()
        ):
            raise VerificationError("source hash artifact contains an invalid sha256")
        return hashes

    @staticmethod
    def _verify_source_hashes(
        source_hashes: Mapping[str, str],
        executor_worktree: Path,
        *,
        manifest: CandidateManifest | None,
    ) -> None:
        for relative, expected in source_hashes.items():
            source = resolve_inside(executor_worktree, Path(relative))
            if not source.is_file():
                raise VerificationError(f"hashed source is not a file: {relative}")
            actual = hashlib.sha256(source.read_bytes()).hexdigest()
            if actual != expected:
                raise VerificationError(f"source hash differs for {relative}")
        if manifest is None:
            return
        changed = run(
            [
                "git",
                "diff",
                "--name-status",
                manifest.base_commit,
                manifest.candidate_commit,
            ],
            cwd=executor_worktree,
        ).stdout.splitlines()
        required: set[str] = set()
        for line in changed:
            fields = line.split("\t")
            if len(fields) < 2 or fields[0].startswith("D"):
                continue
            required.add(fields[-1])
        missing = sorted(required - set(source_hashes))
        if missing:
            raise VerificationError(
                "source hash artifact omits changed files: " + ", ".join(missing)
            )

    @staticmethod
    def _verify_candidate_commit(
        manifest: CandidateManifest, executor_worktree: Path
    ) -> None:
        head = run(
            ["git", "rev-parse", "--verify", "HEAD^{commit}"],
            cwd=executor_worktree,
            check=False,
        )
        if head.returncode != 0:
            raise VerificationError("executor worktree is not a valid Git checkout")
        if head.stdout.strip() != manifest.candidate_commit:
            raise VerificationError(
                "manifest candidate_commit differs from executor HEAD"
            )
        ancestor = run(
            [
                "git",
                "merge-base",
                "--is-ancestor",
                manifest.base_commit,
                manifest.candidate_commit,
            ],
            cwd=executor_worktree,
            check=False,
        )
        if ancestor.returncode != 0:
            raise VerificationError(
                "manifest base_commit is not an ancestor of candidate_commit"
            )
        changed = run(
            [
                "git",
                "diff",
                "--name-only",
                manifest.base_commit,
                manifest.candidate_commit,
            ],
            cwd=executor_worktree,
        ).stdout.splitlines()
        if not changed:
            raise VerificationError("candidate commit contains no source changes")

    @staticmethod
    def _positive_engagement(evidence: Mapping[str, Any]) -> bool:
        direct = evidence.get("engaged")
        if direct is True:
            return True
        for name, value in evidence.items():
            lowered = str(name).lower()
            if "fallback" in lowered:
                continue
            if (
                isinstance(value, int)
                and not isinstance(value, bool)
                and value > 0
                and any(
                    token in lowered
                    for token in ("count", "call", "hit", "engage", "saved", "applied")
                )
            ):
                return True
        return False

    @classmethod
    def _has_fallback(cls, evidence: Any) -> bool:
        if isinstance(evidence, Mapping):
            for name, value in evidence.items():
                lowered = str(name).lower()
                if "fallback" in lowered:
                    if value is True:
                        return True
                    if (
                        isinstance(value, int)
                        and not isinstance(value, bool)
                        and value > 0
                    ):
                        return True
                    if isinstance(value, str) and value.lower() not in {
                        "",
                        "0",
                        "false",
                        "none",
                        "native",
                    }:
                        return True
                if cls._has_fallback(value):
                    return True
        elif isinstance(evidence, list):
            return any(cls._has_fallback(value) for value in evidence)
        return False

    @staticmethod
    def _reject_fallback(
        run_dir: Path,
        performance: Mapping[str, Any],
        engagement: EngagementReceipt | None,
        issue: Any,
    ) -> None:
        if DeliveryVerifier._has_fallback(performance):
            issue("fallback_detected", "PERFORMANCE.json reports a fallback")
        if engagement is not None and DeliveryVerifier._has_fallback(
            engagement.model_dump()
        ):
            issue("fallback_detected", "engagement receipt reports a fallback")
        for name in ("stdout.log", "stderr.log"):
            log_path = run_dir / name
            if not log_path.is_file():
                continue
            lowered = log_path.read_text(encoding="utf-8", errors="replace").lower()
            marker = next(
                (value for value in _FALLBACK_MARKERS if value in lowered), None
            )
            if marker is not None:
                issue(
                    "fallback_detected", f"run log contains fallback marker: {marker}"
                )

    @staticmethod
    def _positive_number(value: Any) -> float | None:
        number = DeliveryVerifier._positive_or_zero_number(value)
        return number if number is not None and number > 0 else None

    @staticmethod
    def _first_positive(
        values: Mapping[str, Any], names: Sequence[str]
    ) -> float | None:
        for name in names:
            number = DeliveryVerifier._positive_number(values.get(name))
            if number is not None:
                return number
        return None

    @staticmethod
    def _positive_or_zero_number(value: Any) -> float | None:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            number = float(value)
            if math.isfinite(number) and number >= 0:
                return number
        return None
