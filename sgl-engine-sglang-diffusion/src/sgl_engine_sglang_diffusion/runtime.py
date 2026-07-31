from __future__ import annotations

import hashlib
import json
import math
import os
import re
import statistics
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .baseline import BaselineRunner
from .config import load_goal
from .controller import CampaignController, StepResult
from .driver import SGLangDiffusionDriver
from .integrator import (
    CandidateActivation,
    IntegrationError,
    IntegrationManager,
    IntegrationVerificationOutcome,
    IntegrationVerificationRequest,
    VerifiedCandidate,
)
from .knowledge import check_contract_hashes
from .knowledge import load_registry as load_knowledge_registry
from .knowledge import read_source_lock, sync_source
from .models import (
    CampaignGoal,
    CampaignStatus,
    CorrectnessMode,
    EngagementReceipt,
    IntegratedDelivery,
    ProfileDigest,
    QualityRecord,
    SourceLock,
)
from .patcher import PatchPackager, sha256_file
from .process import run
from .profiler import Profiler, TechniqueRouter
from .request import FrozenBenchmarkCommand
from .review import SameAgentReviewValidator
from .sources import SourceManager
from .state import LeaseUnavailable, RECOVERABLE_STATUSES, StateStore
from .techniques import TechniqueRegistry
from .work_orders import WorkOrderManager


class CampaignRuntimeError(RuntimeError):
    """The durable campaign cannot safely advance in its current state."""


_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_NAMES = ("sglang", "sol_engine", "fastvideo", "kda_pilot")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CampaignRuntimeError(f"invalid JSON artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise CampaignRuntimeError(f"JSON artifact must contain an object: {path}")
    return value


def _model_slug(model_id: str) -> str:
    slug = re.sub(r"[^a-z0-9_-]+", "-", model_id.lower()).strip("-_")
    if not slug:
        raise CampaignRuntimeError(f"model ID has no safe patch slug: {model_id!r}")
    return slug


def _source_specs(goal: CampaignGoal) -> dict[str, tuple[str, str]]:
    return {
        "sglang": (goal.source.sglang_repo, goal.source.sglang_ref),
        "sol_engine": (
            goal.source.sol_engine_repo,
            goal.source.sol_engine_ref,
        ),
        "fastvideo": (goal.source.fastvideo_repo, goal.source.fastvideo_ref),
        "kda_pilot": (
            goal.source.kda_pilot_repo,
            goal.source.kda_pilot_ref,
        ),
    }


def _validate_sol_contract(lock: SourceLock, checkout: Path | None = None) -> None:
    source_lock = _PACKAGE_ROOT / "contracts" / "sol_engine" / "source-lock.json"
    expected_hashes = _PACKAGE_ROOT / "contracts" / "sol_engine" / "source-hashes.json"
    reviewed = read_source_lock(source_lock)
    if lock.commit != reviewed["commit"]:
        raise CampaignRuntimeError(
            "campaign Sol-Engine commit differs from the reviewed correctness "
            f"contract: {lock.commit} != {reviewed['commit']}"
        )
    if checkout is not None:
        issues = check_contract_hashes(source_lock, checkout, expected_hashes)
        if issues:
            raise CampaignRuntimeError(
                "locked Sol-Engine contract drift: " + "; ".join(issues)
            )


class LockedSolQualityEvaluator:
    """Recompute locked Sol LPIPS without launching an AI reviewer."""

    def __init__(
        self,
        *,
        sol_checkout: Path,
        campaign_dir: Path,
    ) -> None:
        self.sol_checkout = sol_checkout.resolve()
        self.campaign_dir = campaign_dir.resolve()

    def assess(
        self,
        *,
        baseline_frames: Path,
        candidate_frames: Path,
        run_dir: Path,
    ) -> Mapping[str, Any]:
        lpips_judge = self.sol_checkout / "tools/vision/lpips_judge.py"
        if not lpips_judge.is_file():
            raise CampaignRuntimeError(
                f"locked Sol LPIPS evaluator is missing: {lpips_judge}"
            )
        run_dir = run_dir.resolve()
        prompt_pairs = self._aligned_prompt_pairs(
            baseline_frames.resolve(),
            candidate_frames.resolve(),
            run_dir,
        )
        input_manifest = [
            {
                "baseline": str(baseline),
                "baseline_sha256": sha256_file(baseline),
                "candidate": str(candidate),
                "candidate_sha256": sha256_file(candidate),
            }
            for pairs in prompt_pairs
            for baseline, candidate in pairs
        ]
        input_digest = hashlib.sha256(
            json.dumps(input_manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        key = hashlib.sha256(f"{run_dir}\0{input_digest}".encode()).hexdigest()[:20]
        review_dir = self.campaign_dir / "quality-metrics" / key
        review_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(
            review_dir / "INPUTS.json",
            {
                "schema_version": 1,
                "input_sha256": input_digest,
                "pairs": input_manifest,
            },
        )

        prompt_scores: list[dict[str, Any]] = []
        all_scores: list[float] = []
        for index, pairs in enumerate(prompt_pairs):
            prompt_output = review_dir / f"lpips-prompt-{index:02d}.json"
            argv = [sys.executable, str(lpips_judge)]
            for baseline, candidate in pairs:
                argv.extend(
                    [
                        "--baseline-frame",
                        str(baseline),
                        "--candidate-frame",
                        str(candidate),
                    ]
                )
            argv.extend(["--out", str(prompt_output)])
            if not prompt_output.is_file():
                result = run(argv, cwd=self.sol_checkout, check=False)
                (review_dir / f"lpips-prompt-{index:02d}.stdout.log").write_text(
                    result.stdout, encoding="utf-8"
                )
                (review_dir / f"lpips-prompt-{index:02d}.stderr.log").write_text(
                    result.stderr, encoding="utf-8"
                )
                if result.returncode != 0 or not prompt_output.is_file():
                    raise CampaignRuntimeError(
                        f"locked Sol LPIPS failed for prompt {index}"
                    )
            payload = _read_object(prompt_output)
            raw_scores = payload.get("per_frame")
            if (
                payload.get("status") != "ok"
                or not isinstance(raw_scores, list)
                or len(raw_scores) != len(pairs)
                or any(
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(float(value))
                    or float(value) < 0
                    for value in raw_scores
                )
            ):
                raise CampaignRuntimeError(
                    f"locked Sol LPIPS is unavailable for prompt {index}"
                )
            scores = [float(value) for value in raw_scores]
            all_scores.extend(scores)
            prompt_scores.append(
                {
                    "prompt_index": index,
                    "frame_count": len(scores),
                    "mean": statistics.fmean(scores),
                    "max": max(scores),
                }
            )
        lpips_summary = {
            "schema_version": 1,
            "input_sha256": input_digest,
            "aligned": True,
            "prompt_scores": prompt_scores,
            "lpips_mean": statistics.fmean(all_scores),
            "lpips_max": max(all_scores),
            "frame_count": len(all_scores),
        }
        _atomic_json(review_dir / "lpips-assessment.json", lpips_summary)
        return {
            "aligned": True,
            "prompt_scores": prompt_scores,
            "lpips_mean": lpips_summary["lpips_mean"],
            "lpips_max": lpips_summary["lpips_max"],
        }

    @staticmethod
    def _aligned_prompt_pairs(
        baseline_frames: Path,
        candidate_frames: Path,
        run_dir: Path,
    ) -> list[list[tuple[Path, Path]]]:
        baseline_prompts = LockedSolQualityEvaluator._safe_prompt_dirs(
            baseline_frames, baseline_frames
        )
        candidate_root = run_dir / "outputs" / "frames"
        if not candidate_root.is_dir():
            candidate_root = candidate_frames
        candidate_prompts = LockedSolQualityEvaluator._safe_prompt_dirs(
            candidate_root, run_dir
        )
        if len(baseline_prompts) != 5 or len(candidate_prompts) != 5:
            raise CampaignRuntimeError(
                "locked LPIPS requires exactly five aligned prompt directories"
            )
        suffixes = {".png", ".jpg", ".jpeg", ".webp"}
        result: list[list[tuple[Path, Path]]] = []
        for index, (baseline_prompt, candidate_prompt) in enumerate(
            zip(baseline_prompts, candidate_prompts, strict=True)
        ):
            baseline = LockedSolQualityEvaluator._safe_frames(
                baseline_prompt, baseline_frames, suffixes
            )
            candidate = LockedSolQualityEvaluator._safe_frames(
                candidate_prompt, run_dir, suffixes
            )
            if not baseline or len(baseline) != len(candidate):
                raise CampaignRuntimeError(
                    "locked LPIPS frame alignment failed for prompt "
                    f"{index}: {len(baseline)} != {len(candidate)}"
                )
            result.append(list(zip(baseline, candidate, strict=True)))
        return result

    @staticmethod
    def _safe_prompt_dirs(root: Path, allowed_root: Path) -> list[Path]:
        if (
            not root.is_dir()
            or root.is_symlink()
            or not allowed_root.is_dir()
            or allowed_root.is_symlink()
        ):
            raise CampaignRuntimeError(
                f"aligned prompt root is unsafe or missing: {root}"
            )
        resolved_allowed = allowed_root.resolve()
        resolved_root = root.resolve()
        try:
            resolved_root.relative_to(resolved_allowed)
        except ValueError as error:
            raise CampaignRuntimeError(
                f"aligned prompt root escapes allowed evidence: {root}"
            ) from error
        prompts: list[Path] = []
        for path in root.iterdir():
            if not path.name.startswith("prompt-"):
                continue
            if not path.is_dir() or path.is_symlink():
                raise CampaignRuntimeError(f"unsafe aligned prompt path: {path}")
            resolved = path.resolve()
            try:
                resolved.relative_to(resolved_root)
            except ValueError as error:
                raise CampaignRuntimeError(
                    f"aligned prompt path escapes evidence root: {path}"
                ) from error
            prompts.append(resolved)
        return sorted(prompts)

    @staticmethod
    def _safe_frames(
        prompt: Path, allowed_root: Path, suffixes: set[str]
    ) -> list[Path]:
        resolved_allowed = allowed_root.resolve()
        frames: list[Path] = []
        for path in prompt.iterdir():
            if path.suffix.lower() not in suffixes:
                continue
            if not path.is_file() or path.is_symlink():
                raise CampaignRuntimeError(f"unsafe aligned frame path: {path}")
            resolved = path.resolve()
            try:
                resolved.relative_to(resolved_allowed)
            except ValueError as error:
                raise CampaignRuntimeError(
                    f"aligned frame escapes evidence root: {path}"
                ) from error
            frames.append(resolved)
        return sorted(frames)


class RuntimeIntegrationVerifier:
    """Reapply engagement and correctness gates to the composed full run."""

    def __init__(self, quality_evaluator: LockedSolQualityEvaluator | None) -> None:
        self.quality_evaluator = quality_evaluator

    def verify_integrated(
        self, request: IntegrationVerificationRequest
    ) -> IntegrationVerificationOutcome:
        issues: list[str] = []
        artifacts: list[Path] = []

        media = [
            path
            for path in request.media_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower()
            in {".png", ".jpg", ".jpeg", ".webp", ".mp4", ".webm", ".mov"}
        ]
        if len(media) < 5:
            issues.append("integrated run has fewer than five durable media outputs")
        artifacts.extend(media)

        engagement_path = _find_named(
            request.run_dir,
            ("engagement-receipt.json", "ENGAGEMENT.json"),
        )
        if engagement_path is None:
            issues.append("integrated run lacks an engagement receipt")
            engagement = None
        else:
            artifacts.append(engagement_path)
            try:
                engagement = EngagementReceipt.model_validate(
                    _read_object(engagement_path)
                )
            except (ValueError, CampaignRuntimeError) as error:
                issues.append(f"invalid integrated engagement receipt: {error}")
                engagement = None

        if engagement is not None:
            if not (
                engagement.model_match
                and engagement.hardware_match
                and engagement.workload_match
            ):
                issues.append("integrated engagement does not match frozen inputs")
            if dict(engagement.source_hashes) != request.recipe.source_hashes:
                issues.append("integrated engagement source hashes differ from recipe")
            for technique in request.recipe.techniques:
                evidence = engagement.techniques.get(technique)
                if not isinstance(evidence, dict) or not _positive_engagement(evidence):
                    issues.append(f"integrated {technique} has zero engagement")
                elif _has_fallback(evidence):
                    issues.append(f"integrated {technique} reports a fallback")

        quality: QualityRecord | None = None
        if request.correctness is CorrectnessMode.LOSSLESS:
            equivalence_path = _find_named(request.run_dir, ("equivalence.json",))
            authenticity_path = _find_named(
                request.run_dir, ("authenticity.json", "visual_verdict.json")
            )
            if equivalence_path is None:
                issues.append("integrated lossless run lacks equivalence.json")
            else:
                artifacts.append(equivalence_path)
                try:
                    equivalence = _read_object(equivalence_path)
                    _validate_lossless_equivalence(equivalence)
                except CampaignRuntimeError as error:
                    issues.append(str(error))
            if authenticity_path is None:
                issues.append("integrated lossless run lacks authenticity receipt")
            else:
                artifacts.append(authenticity_path)
                authenticity = _read_object(authenticity_path)
                if not (
                    authenticity.get("authentic") is True
                    or authenticity.get("overall") in {"pass", "authenticity_only"}
                ):
                    issues.append("integrated authenticity receipt did not pass")
                quality = QualityRecord(
                    mode="not_gated",
                    lpips_max=None,
                    lpips_mean=None,
                    visual_overall="authenticity_only",
                    visual_verdict=authenticity_path,
                    relation="equivalent",
                )
        else:
            visual_path = request.run_dir / "visual_verdict.json"
            if self.quality_evaluator is None:
                issues.append("integrated quality evaluator is unavailable")
            elif not visual_path.is_file():
                issues.append("integrated run lacks visual_verdict.json")
            else:
                try:
                    assessed = self.quality_evaluator.assess(
                        baseline_frames=request.baseline.baseline_frames,
                        candidate_frames=request.media_dir,
                        run_dir=request.run_dir,
                    )
                    if (
                        assessed.get("aligned") is not True
                        or assessed.get("visual_overall") != "pass"
                    ):
                        issues.append("integrated quality assessment did not pass")
                    else:
                        quality = QualityRecord(
                            mode="quality_gated",
                            lpips_max=float(assessed["lpips_max"]),
                            lpips_mean=float(assessed["lpips_mean"]),
                            visual_overall="pass",
                            visual_verdict=visual_path,
                            relation="equivalent",
                        )
                        artifacts.append(visual_path)
                except (KeyError, TypeError, ValueError, CampaignRuntimeError) as error:
                    issues.append(f"integrated quality assessment failed: {error}")

        return IntegrationVerificationOutcome(
            accepted=not issues,
            issues=issues,
            quality=quality if not issues else None,
            artifacts=artifacts,
            implementation_manifest={
                "engagement_receipt": (
                    str(engagement_path) if engagement_path is not None else None
                ),
                "reapplied_correctness": request.correctness.value,
            },
        )


def _find_named(root: Path, names: Sequence[str]) -> Path | None:
    for name in names:
        direct = root / name
        if direct.is_file():
            return direct.resolve()
    wanted = set(names)
    return next(
        (
            path.resolve()
            for path in root.rglob("*")
            if path.is_file() and path.name in wanted
        ),
        None,
    )


def _positive_engagement(evidence: Mapping[str, Any]) -> bool:
    if evidence.get("engaged") is True:
        return True
    return any(
        isinstance(value, int)
        and not isinstance(value, bool)
        and value > 0
        and any(
            token in str(name).lower()
            for token in ("count", "call", "hit", "engage", "applied")
        )
        for name, value in evidence.items()
        if "fallback" not in str(name).lower()
    )


def _has_fallback(value: Any) -> bool:
    if isinstance(value, Mapping):
        for name, nested in value.items():
            if "fallback" in str(name).lower() and nested not in (
                False,
                0,
                "",
                None,
                "native",
            ):
                return True
            if _has_fallback(nested):
                return True
    elif isinstance(value, list):
        return any(_has_fallback(item) for item in value)
    return False


def _validate_lossless_equivalence(value: Mapping[str, Any]) -> None:
    baseline = value.get("baseline")
    candidate = value.get("candidate")
    if not isinstance(baseline, Mapping) or not isinstance(candidate, Mapping):
        raise CampaignRuntimeError(
            "integrated equivalence lacks baseline/candidate counts"
        )
    for name in ("global_steps", "dit_calls"):
        before = baseline.get(name)
        after = candidate.get(name)
        if (
            not isinstance(before, int)
            or isinstance(before, bool)
            or before <= 0
            or after != before
        ):
            raise CampaignRuntimeError(f"integrated lossless run changed {name}")
    if (
        not isinstance(value.get("method_argument"), str)
        or not value["method_argument"].strip()
        or value.get("logical_work_unchanged") is not True
    ):
        raise CampaignRuntimeError("integrated lossless method argument is incomplete")


class FileCampaignHooks:
    """File-backed, restart-safe implementation of the controller hook protocol."""

    def __init__(
        self,
        *,
        campaign_dir: Path,
        campaign_id: str,
        goal: CampaignGoal,
        store: StateStore,
    ) -> None:
        self.campaign_dir = campaign_dir.resolve()
        self.campaign_id = campaign_id
        self.goal = goal
        self.store = store
        self.source_manager = SourceManager(
            self.campaign_dir.parent / ".sgl-diffusion-source-cache"
        )
        self.registry = TechniqueRegistry.load(
            _PACKAGE_ROOT / "techniques" / "registry.toml"
        )

    def _driver(self, checkout: Path) -> SGLangDiffusionDriver:
        template = self.campaign_dir / "BASELINE-COMMAND.json"
        if template.is_file():
            return SGLangDiffusionDriver.from_template(checkout, template)
        return SGLangDiffusionDriver(checkout)

    def _command_template(self) -> FrozenBenchmarkCommand | None:
        path = self.campaign_dir / "BASELINE-COMMAND.json"
        if not path.is_file():
            return None
        return FrozenBenchmarkCommand.model_validate_json(
            path.read_text(encoding="utf-8")
        )

    def freeze_sources_and_baseline(self) -> StepResult:
        locks = self._ensure_source_locks()
        worktrees = self._ensure_source_worktrees(locks)
        self._sync_knowledge(locks, worktrees)

        baseline_path = self.campaign_dir / "BASELINE.json"
        if baseline_path.is_file():
            baseline = BaselineRunner.load(baseline_path)
            if baseline.sglang_commit != locks["sglang"].commit:
                raise CampaignRuntimeError(
                    "frozen baseline does not match SOURCE-LOCKS.json"
                )
        else:
            baseline = BaselineRunner(self._driver(worktrees["sglang"])).freeze(
                self.goal,
                self.campaign_dir,
                sglang_commit=locks["sglang"].commit,
            )
        return StepResult(
            CampaignStatus.BASELINE_LOCKED,
            payload={
                "baseline": str(baseline_path),
                "sglang_commit": baseline.sglang_commit,
            },
        )

    def profile_and_route(self) -> StepResult:
        route_path = self.campaign_dir / "ROUTES.json"
        if route_path.is_file():
            value = _read_object(route_path)
            routes = [str(item) for item in value["routes"]]
            return StepResult(
                CampaignStatus.PROFILED,
                payload={"routes": routes, "route_artifact": str(route_path)},
            )

        locks = self._load_locks()
        worktree = self.campaign_dir / "source-worktrees" / "sglang"
        profile_path = self.campaign_dir / "profiles" / "0" / "PROFILE-DIGEST.json"
        if profile_path.is_file():
            digest = ProfileDigest.model_validate_json(
                profile_path.read_text(encoding="utf-8")
            )
        else:
            digest = Profiler(self._driver(worktree)).collect(
                self.goal,
                self.campaign_dir,
                epoch=0,
            )
        router = TechniqueRouter()
        routes = router.route(
            digest,
            allow_quality_gated=self.goal.goal.allow_quality_gated,
            gpu_count=self.goal.hardware.gpu_count,
        )
        unknown = set(routes) - set(self.registry.names())
        if unknown:
            raise CampaignRuntimeError(
                "router selected unregistered techniques: " + ", ".join(sorted(unknown))
            )
        _atomic_json(
            route_path,
            {
                "schema_version": 1,
                "routes": routes,
                "evidence": router.last_evidence,
                "profile_digest": str(profile_path),
                "sglang_commit": locks["sglang"].commit,
            },
        )
        return StepResult(
            CampaignStatus.PROFILED,
            payload={"routes": routes, "route_artifact": str(route_path)},
        )

    def enter_agent_wait(self, epoch: int) -> StepResult:
        return StepResult(
            CampaignStatus.AWAITING_AGENT,
            payload={
                "epoch": epoch,
                "reason": "profile_ready_for_interactive_claim",
                "routes": self._routes(),
            },
        )

    def verify_submitted_delivery(self, epoch: int) -> StepResult:
        from .verifier import DeliveryVerifier

        work_orders = WorkOrderManager(
            self.campaign_dir,
            campaign_id=self.campaign_id,
            store=self.store,
            source_manager=self.source_manager,
            registry=self.registry,
        )
        order = work_orders.active_work_order()
        if order.epoch != epoch:
            raise CampaignRuntimeError(
                f"active work order epoch {order.epoch} differs from state epoch {epoch}"
            )
        submitted = [
            event
            for event in self.store.events(
                self.campaign_id, event_type="candidate_submitted"
            )
            if event["payload"].get("epoch") == epoch
        ]
        if not submitted:
            return StepResult(
                None,
                payload={
                    "reason": "awaiting_explicit_submit",
                    "technique": order.technique,
                    "delivery": str(order.delivery_path),
                },
            )
        if not order.delivery_path.is_file() or order.delivery_path.is_symlink():
            return StepResult(
                CampaignStatus.AWAITING_AGENT,
                payload={
                    "reason": "submitted_delivery_missing",
                    "technique": order.technique,
                },
            )
        submitted_digest = str(submitted[-1]["payload"].get("delivery_sha256", ""))
        if sha256_file(order.delivery_path) != submitted_digest:
            return StepResult(
                CampaignStatus.AWAITING_AGENT,
                payload={
                    "reason": "delivery_changed_after_submit",
                    "technique": order.technique,
                },
            )

        verified = self._load_verified(epoch)
        baseline = BaselineRunner.load(self.campaign_dir / "BASELINE.json")
        quality = LockedSolQualityEvaluator(
            sol_checkout=self.campaign_dir / "source-worktrees" / "sol_engine",
            campaign_dir=self.campaign_dir,
        )
        verifier = DeliveryVerifier(
            registry=self.registry,
            baseline=baseline,
            campaign_artifact_root=self.campaign_dir,
            review_validator=SameAgentReviewValidator(
                campaign_id=self.campaign_id,
                epoch=epoch,
                review_path=order.review_path,
            ),
            quality_evaluator=quality,
            command_template=self._command_template(),
        )
        result = verifier.verify(
            order.delivery_path,
            technique=order.technique,
            executor_worktree=order.worktree,
        )
        if not result.accepted or not result.verified_points:
            findings = [
                {
                    "code": finding.code,
                    "message": finding.message,
                    "candidate_id": finding.candidate_id,
                }
                for finding in result.findings
            ]
            if not findings:
                findings.append(
                    {
                        "code": "empty_verified_frontier",
                        "message": "verifier accepted no durable frontier point",
                        "candidate_id": None,
                    }
                )
            self.store.record_event(
                self.campaign_id,
                "work_rejected",
                f"{self.campaign_id}:verify:{epoch}:rejected",
                {
                    "epoch": epoch,
                    "technique": order.technique,
                    "findings": findings,
                },
            )
            return StepResult(
                CampaignStatus.AWAITING_AGENT,
                payload={
                    "reason": "candidate_rejected",
                    "technique": order.technique,
                    "findings": findings,
                },
            )
        point = max(
            result.verified_points,
            key=lambda item: item.authoritative_speedup,
        )
        manifest = point.implementation_manifest
        candidate = VerifiedCandidate(
            candidate_id=point.candidate_id,
            technique=order.technique,
            base_commit=manifest.base_commit,
            candidate_commit=manifest.candidate_commit,
            correctness=CorrectnessMode(self.registry[order.technique].correctness),
            activation=CandidateActivation(
                env=dict(point.activation.get("env", {})),
                server_args=list(point.activation.get("server_args", [])),
            ),
            source_hashes=point.source_hashes,
            compatibility_notes=[
                f"verified from interactive {order.technique} work order",
                f"authoritative speedup {point.authoritative_speedup:.8g}x",
            ],
            verified_speedup=point.authoritative_speedup,
            verified=True,
        )
        verified[order.technique] = candidate
        self._write_verified(epoch, verified)
        self.store.record_event(
            self.campaign_id,
            "work_accepted",
            f"{self.campaign_id}:verify:{epoch}:accepted",
            {
                "epoch": epoch,
                "technique": order.technique,
                "candidate_id": candidate.candidate_id,
                "verified_speedup": candidate.verified_speedup,
            },
        )
        if candidate.verified_speedup <= 1.0:
            return StepResult(
                CampaignStatus.AWAITING_AGENT,
                payload={
                    "reason": "verified_non_latency_frontier",
                    "technique": order.technique,
                    "candidate_id": candidate.candidate_id,
                },
            )

        return StepResult(
            CampaignStatus.INTEGRATING,
            payload={
                "verified_candidates": str(self._verified_path(epoch)),
                "candidate_ids": [
                    item.candidate_id
                    for item in verified.values()
                    if item.verified_speedup > 1.0
                ],
            },
        )

    def integrate_and_gate(self, epoch: int) -> StepResult:
        integration_receipt = (
            self.campaign_dir / "integration" / str(epoch) / ("INTEGRATION.json")
        )
        if integration_receipt.is_file():
            value = _read_object(integration_receipt)
            return StepResult(
                CampaignStatus.FINAL_VERIFYING,
                payload={
                    "integrated_delivery": value["delivery_path"],
                    "integration_receipt": str(integration_receipt),
                },
            )

        locks = self._load_locks()
        baseline = BaselineRunner.load(self.campaign_dir / "BASELINE.json")
        verified = self._load_verified(epoch)
        selected = {
            name: candidate
            for name, candidate in verified.items()
            if candidate.verified and candidate.verified_speedup > 1.0
        }
        if not selected:
            return StepResult(
                CampaignStatus.AWAITING_AGENT,
                payload={"reason": "no_verified_latency_positive_candidate"},
            )
        quality = LockedSolQualityEvaluator(
            sol_checkout=self.campaign_dir / "source-worktrees" / "sol_engine",
            campaign_dir=self.campaign_dir,
        )
        manager = IntegrationManager(
            self.source_manager,
            locks["sglang"],
            RuntimeIntegrationVerifier(quality),
            command_template=self._command_template(),
        )
        epoch_root = integration_receipt.parent
        attempt = 1
        while (epoch_root / f"attempt-{attempt:03d}").exists():
            attempt += 1
        try:
            result = manager.integrate(
                self.goal,
                baseline,
                [selected[name].candidate_id for name in sorted(selected)],
                {item.candidate_id: item for item in selected.values()},
                epoch_root / f"attempt-{attempt:03d}",
            )
        except IntegrationError as error:
            signature = hashlib.sha256(str(error).encode()).hexdigest()
            self.store.record_failure(
                self.campaign_id,
                "integration",
                signature,
                {
                    "epoch": epoch,
                    "candidate_ids": [
                        selected[name].candidate_id for name in sorted(selected)
                    ],
                    "detail": str(error),
                },
            )
            self.store.record_event(
                self.campaign_id,
                "integration_rejected",
                f"{self.campaign_id}:integration:{epoch}:rejected",
                {
                    "epoch": epoch,
                    "candidate_ids": [
                        selected[name].candidate_id for name in sorted(selected)
                    ],
                    "detail": str(error),
                    "failure_signature": signature,
                },
            )
            return StepResult(
                CampaignStatus.AWAITING_AGENT,
                payload={
                    "reason": "integrated_gate_rejected",
                    "detail": str(error),
                    "failure_signature": signature,
                },
            )
        if result.status == "needs_agent_revision":
            assert result.failed_candidate_id is not None
            failed_technique = next(
                name
                for name, candidate in selected.items()
                if candidate.candidate_id == result.failed_candidate_id
            )
            self._remove_verified(epoch, failed_technique)
            self.store.record_event(
                self.campaign_id,
                "integration_conflict",
                (
                    f"{self.campaign_id}:integration:{epoch}:conflict:"
                    f"{failed_technique}"
                ),
                {
                    "epoch": epoch,
                    "technique": failed_technique,
                    "candidate_id": result.failed_candidate_id,
                    "diagnostics": str(result.diagnostics_path),
                },
            )
            return StepResult(
                CampaignStatus.AWAITING_AGENT,
                payload={
                    "reason": "integration_conflict",
                    "technique": failed_technique,
                    "diagnostics": str(result.diagnostics_path),
                },
            )
        if result.delivery_path is None or result.integration_commit is None:
            raise CampaignRuntimeError("integrator returned no durable delivery")
        _atomic_json(
            integration_receipt,
            {
                "schema_version": 1,
                "epoch": epoch,
                "attempt": attempt,
                "worktree": str(result.worktree),
                "integration_commit": result.integration_commit,
                "delivery_path": str(result.delivery_path),
            },
        )
        return StepResult(
            CampaignStatus.FINAL_VERIFYING,
            payload={
                "integrated_delivery": str(result.delivery_path),
                "integration_receipt": str(integration_receipt),
            },
        )

    def package_or_continue(self, epoch: int) -> StepResult:
        receipt_path = (
            self.campaign_dir / "integration" / str(epoch) / "INTEGRATION.json"
        )
        receipt = _read_object(receipt_path)
        delivery_path = Path(receipt["delivery_path"])
        delivery = IntegratedDelivery.model_validate_json(
            delivery_path.read_text(encoding="utf-8")
        )
        speedup = max(point.performance.speedup for point in delivery.frontier_points)
        if speedup < self.goal.goal.target_speedup:
            return StepResult(
                CampaignStatus.AWAITING_AGENT,
                payload={
                    "reason": "target_not_reached",
                    "verified_speedup": speedup,
                },
                verified_speedup=speedup,
                new_hypothesis=True,
            )

        locks = self._load_locks()
        worktree = Path(receipt["worktree"]).resolve()
        packager = PatchPackager(worktree, base_sha=locks["sglang"].commit)
        profile = packager.validate(model_slug=_model_slug(self.goal.model.id))
        if not math.isclose(profile.speedup, speedup, rel_tol=1e-6, abs_tol=1e-9):
            raise CampaignRuntimeError(
                "packaged profile speedup differs from integrated measurement"
            )
        patch_dir = self.campaign_dir / "patch"
        package_receipt = self.campaign_dir / "PACKAGE.json"
        if not package_receipt.is_file():
            if patch_dir.exists():
                self._validate_existing_patch(patch_dir, locks["sglang"].commit)
            else:
                temporary = self.campaign_dir / (f".patch.{epoch}.{os.getpid()}.tmp")
                bundle = packager.package(
                    temporary,
                    model_slug=_model_slug(self.goal.model.id),
                    evidence=[
                        delivery_path,
                        self.campaign_dir / "BASELINE.json",
                        self.campaign_dir / "SOURCE-LOCKS.json",
                    ],
                    cpu_validation_commands=(),
                    gpu_validation_command=self._gpu_validation_command(delivery),
                    clean_room=True,
                )
                if not bundle.patch.is_file():
                    raise CampaignRuntimeError("patch packager produced no patch")
                os.replace(temporary, patch_dir)
            _atomic_json(
                package_receipt,
                {
                    "schema_version": 1,
                    "epoch": epoch,
                    "verified_speedup": speedup,
                    "base_sha": locks["sglang"].commit,
                    "patch": str(patch_dir / "sglang.patch"),
                    "manifest": str(patch_dir / "manifest.json"),
                    "clean_room_verified": True,
                },
            )
        else:
            self._validate_existing_patch(patch_dir, locks["sglang"].commit)
        return StepResult(
            CampaignStatus.TARGET_REACHED,
            payload={
                "package_receipt": str(package_receipt),
                "patch": str(patch_dir / "sglang.patch"),
            },
            verified_speedup=speedup,
            clean_room_verified=True,
        )

    def _ensure_source_locks(self) -> dict[str, SourceLock]:
        path = self.campaign_dir / "SOURCE-LOCKS.json"
        if path.is_file():
            return self._load_locks()
        locks: dict[str, SourceLock] = {}
        for name, (repository, requested_ref) in _source_specs(self.goal).items():
            locks[name] = self.source_manager.lock(name, repository, requested_ref)
        _validate_sol_contract(locks["sol_engine"])
        _atomic_json(
            path,
            {
                "schema_version": 1,
                **{name: lock.model_dump(mode="json") for name, lock in locks.items()},
            },
        )
        return locks

    def _load_locks(self) -> dict[str, SourceLock]:
        value = _read_object(self.campaign_dir / "SOURCE-LOCKS.json")
        locks = {name: SourceLock.model_validate(value[name]) for name in _SOURCE_NAMES}
        _validate_sol_contract(locks["sol_engine"])
        expected = _source_specs(self.goal)
        for name, lock in locks.items():
            repository, requested_ref = expected[name]
            if lock.repository != repository or lock.requested_ref != requested_ref:
                raise CampaignRuntimeError(
                    f"source lock {name} differs from frozen GOAL.yaml"
                )
        return locks

    def _ensure_source_worktrees(
        self, locks: Mapping[str, SourceLock]
    ) -> dict[str, Path]:
        roots: dict[str, Path] = {}
        for name, lock in locks.items():
            destination = self.campaign_dir / "source-worktrees" / name
            if destination.exists():
                head = run(["git", "rev-parse", "HEAD"], cwd=destination).stdout.strip()
                if head != lock.commit:
                    raise CampaignRuntimeError(
                        f"source worktree {name} is not at its locked commit"
                    )
            else:
                self.source_manager.create_worktree(lock, destination)
            roots[name] = destination.resolve()
        _validate_sol_contract(locks["sol_engine"], roots["sol_engine"])
        return roots

    def _sync_knowledge(
        self,
        locks: Mapping[str, SourceLock],
        worktrees: Mapping[str, Path],
    ) -> None:
        registry = load_knowledge_registry(
            _PACKAGE_ROOT / "knowledge" / "registry.toml"
        )
        snapshots: dict[str, str] = {}
        for name, patterns in registry.items():
            lock = locks[name]
            output = self.campaign_dir / "knowledge" / name / lock.commit
            snapshot = sync_source(
                name=name,
                checkout=worktrees[name],
                commit=lock.commit,
                patterns=patterns,
                output_dir=output,
            )
            if snapshot.commit != lock.commit:
                raise CampaignRuntimeError(
                    f"knowledge snapshot {name} has the wrong commit"
                )
            snapshots[name] = str(output / "index.json")
        _atomic_json(
            self.campaign_dir / "KNOWLEDGE.json",
            {"schema_version": 1, "snapshots": snapshots},
        )

    def _routes(self) -> list[str]:
        value = _read_object(self.campaign_dir / "ROUTES.json")
        routes = [str(item) for item in value["routes"]]
        if not routes or len(routes) != len(set(routes)):
            raise CampaignRuntimeError("ROUTES.json is empty or has duplicates")
        return routes

    def _verified_path(self, epoch: int) -> Path:
        del epoch
        return self.campaign_dir / "VERIFIED-CANDIDATES.json"

    def _load_verified(self, epoch: int) -> dict[str, VerifiedCandidate]:
        path = self._verified_path(epoch)
        if not path.is_file():
            return {}
        value = _read_object(path)
        return {
            name: VerifiedCandidate.model_validate(candidate)
            for name, candidate in value["candidates"].items()
        }

    def _write_verified(
        self, epoch: int, candidates: Mapping[str, VerifiedCandidate]
    ) -> None:
        _atomic_json(
            self._verified_path(epoch),
            {
                "schema_version": 1,
                "epoch": epoch,
                "candidates": {
                    name: candidate.model_dump(mode="json")
                    for name, candidate in candidates.items()
                },
            },
        )

    def _remove_verified(self, epoch: int, technique: str) -> None:
        candidates = self._load_verified(epoch)
        candidates.pop(technique, None)
        self._write_verified(epoch, candidates)

    @staticmethod
    def _gpu_validation_command(delivery: IntegratedDelivery) -> list[str]:
        if not delivery.frontier_points:
            return []
        receipt = delivery.frontier_points[0].run_dir / "COMMAND.json"
        if not receipt.is_file():
            return []
        value = _read_object(receipt)
        argv = value.get("argv")
        return [str(item) for item in argv] if isinstance(argv, list) else []

    @staticmethod
    def _validate_existing_patch(patch_dir: Path, base_sha: str) -> None:
        required = (
            patch_dir / "sglang.patch",
            patch_dir / "manifest.json",
            patch_dir / "SHA256SUMS",
            patch_dir / "apply_and_verify.sh",
        )
        if any(not path.is_file() for path in required):
            raise CampaignRuntimeError("existing patch bundle is incomplete")
        manifest = _read_object(patch_dir / "manifest.json")
        if manifest.get("base_sha") != base_sha:
            raise CampaignRuntimeError("existing patch bundle has the wrong base")
        if manifest.get("patch_sha256") != sha256_file(patch_dir / "sglang.patch"):
            raise CampaignRuntimeError("existing patch checksum is invalid")


def _resume_recoverable(
    store: StateStore, campaign_id: str, current: CampaignStatus
) -> CampaignStatus:
    events = store.events(campaign_id, event_type="transition")
    for ordinal, event in reversed(list(enumerate(events, start=1))):
        payload = event["payload"]
        if (
            payload.get("status") == current.value
            and payload.get("prior_status") is not None
        ):
            prior = CampaignStatus(payload["prior_status"])
            return store.transition(
                campaign_id,
                prior,
                idempotency_key=(
                    f"{campaign_id}:recover:{ordinal}:{current.value}:{prior.value}"
                ),
                payload={"reason": "explicit_resume"},
            )
    raise CampaignRuntimeError(
        f"{current.value} has no durable prior active state to resume"
    )


def run_campaign_command(command: str, campaign: Path) -> dict[str, Any]:
    """Execute one durable controller burst and return a machine-readable receipt."""

    if command not in {"run", "resume", "package"}:
        raise ValueError(f"unsupported campaign command: {command}")
    campaign = campaign.resolve()
    manifest = _read_object(campaign / "CAMPAIGN.json")
    campaign_id = str(manifest["campaign_id"])
    goal = load_goal(campaign / "GOAL.yaml")
    with StateStore.open(campaign / "state.sqlite", campaign / "events.jsonl") as store:
        prior = store.status(campaign_id)
        if prior in RECOVERABLE_STATUSES:
            if command != "resume":
                return _runtime_payload(campaign, campaign_id, prior, prior)
            prior = _resume_recoverable(store, campaign_id, prior)
        if command == "package" and prior not in {
            CampaignStatus.FINAL_VERIFYING,
            CampaignStatus.TARGET_REACHED,
        }:
            raise CampaignRuntimeError(
                "package requires FINAL_VERIFYING or TARGET_REACHED state"
            )
        if prior is CampaignStatus.TARGET_REACHED:
            return _runtime_payload(campaign, campaign_id, prior, prior)

        hooks = FileCampaignHooks(
            campaign_dir=campaign,
            campaign_id=campaign_id,
            goal=goal,
            store=store,
        )
        controller = CampaignController(
            store=store,
            campaign_id=campaign_id,
            goal=goal,
            hooks=hooks,
            campaign_dir=campaign,
            allowed_methods=(
                hooks._routes() if (campaign / "ROUTES.json").is_file() else ()
            ),
        )
        initial = store.status(campaign_id)
        current = initial
        max_steps = 1 if command == "package" else 32
        for _ in range(max_steps):
            before = store.status(campaign_id)
            try:
                current = controller.run_once()
            except LeaseUnavailable as error:
                ordinal = len(store.events(campaign_id)) + 1
                current = store.transition(
                    campaign_id,
                    CampaignStatus.WAITING_RESOURCE,
                    idempotency_key=(f"{campaign_id}:waiting-resource:{ordinal}"),
                    payload={
                        "reason": "executor_lease_unavailable",
                        "detail": str(error),
                    },
                )
            if current == before or current in RECOVERABLE_STATUSES:
                break
            if current in {
                CampaignStatus.TARGET_REACHED,
                CampaignStatus.UNREACHABLE_CERTIFIED,
                CampaignStatus.SEARCH_SPACE_EXHAUSTED,
                CampaignStatus.CANCELLED,
            }:
                break
        return _runtime_payload(campaign, campaign_id, initial, current)


def _runtime_payload(
    campaign: Path,
    campaign_id: str,
    prior: CampaignStatus,
    current: CampaignStatus,
) -> dict[str, Any]:
    artifacts = sorted(
        str(path.relative_to(campaign))
        for path in campaign.rglob("*")
        if path.is_file()
        and (
            path.suffix in {".json", ".patch"}
            or path.name in {"SHA256SUMS", "apply_and_verify.sh"}
        )
    )
    return {
        "campaign_id": campaign_id,
        "campaign": str(campaign),
        "prior_state": prior.value,
        "new_state": current.value,
        "artifacts": artifacts,
    }
