from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import statistics
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .agents import AgentRunner, build_agent_argv, redact_argv
from .baseline import BaselineRunner
from .config import load_goal
from .controller import CampaignController, StepResult
from .delivery_contract import build_delivery_contract
from .driver import SGLangDiffusionDriver
from .integrator import (
    CandidateActivation,
    IntegrationError,
    IntegrationManager,
    IntegrationVerificationOutcome,
    IntegrationVerificationRequest,
    VerifiedCandidate,
)
from .history_rules import HistoryRuleCatalog
from .knowledge import KnowledgeSyncError, check_contract_hashes
from .knowledge import load_registry as load_knowledge_registry
from .knowledge import read_source_lock, sync_source
from .models import (
    CampaignGoal,
    CampaignStatus,
    CorrectnessMode,
    EngagementReceipt,
    FinalQualityEvidence,
    GpuInventory,
    GpuInventoryDevice,
    IntegratedDelivery,
    ProfileDigest,
    QualityRecord,
    SourceLock,
    TechniqueDisposition,
)
from .orchestration import (
    ExecutorHandle,
    ExecutorManager,
    ExecutorPrompt,
    PromptSection,
    require_regular_delivery,
)
from .patcher import PatchPackager, sha256_file
from .placement import detect_placement_contract
from .process import run
from .profiler import ProfileError, Profiler, TechniqueRouter
from .request import FrozenBenchmarkCommand
from .sources import SourceManager
from .state import LeaseUnavailable, RECOVERABLE_STATUSES, StateStore
from .techniques import TechniqueRegistry


class CampaignRuntimeError(RuntimeError):
    """The durable campaign cannot safely advance in its current state."""


_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_NAMES = ("sglang", "sol_engine", "fastvideo", "kda_pilot")
_KDA_REQUIRED_SUBMODULES = (
    "external/KernelWiki",
    "external/ncu-report-skill",
    "external/warp-specialization-report-skill",
)
_KNOWLEDGE_REQUIRED_PREFIXES = {
    "kda_pilot": tuple(f"{path}/" for path in _KDA_REQUIRED_SUBMODULES),
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LOSSLESS_FORBIDDEN_ADDITIONS = re.compile(
    r"(?i)\b(?:step[-_ ]?skip|token[-_ ]?prun|spars|rank[-_ ]?reduc|"
    r"fp8|int8|nvfp4|int4|approximate)\b"
)


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


def _serialize_handle(handle: ExecutorHandle) -> dict[str, Any]:
    value = asdict(handle)
    for name in ("root", "worktree", "prompt", "delivery", "receipt"):
        value[name] = str(value[name])
    return value


def _load_handle(path: Path) -> ExecutorHandle:
    value = _read_object(path)
    handle = ExecutorHandle(
        executor_id=str(value["executor_id"]),
        campaign_id=str(value["campaign_id"]),
        technique=str(value["technique"]),
        root=Path(value["root"]).resolve(),
        worktree=Path(value["worktree"]).resolve(),
        prompt=Path(value["prompt"]).resolve(),
        delivery=Path(value["delivery"]).resolve(),
        receipt=Path(value["receipt"]).resolve(),
        pid=int(value["pid"]),
        attempt=int(value["attempt"]),
        lease_resource=str(value["lease_resource"]),
        lease_owner=str(value["lease_owner"]),
    )
    if path.resolve() != handle.root / "executor.json":
        raise CampaignRuntimeError(f"executor manifest has an unsafe root: {path}")
    if handle.worktree != handle.root / "worktree":
        raise CampaignRuntimeError(f"executor worktree escapes its root: {path}")
    if handle.delivery != handle.worktree / "DELIVERY.json":
        raise CampaignRuntimeError(f"executor delivery path changed: {path}")
    return handle


class IndependentMasterMethodAuditor:
    """Combine deterministic checks with a separate coding-agent code audit."""

    def __init__(
        self,
        *,
        campaign_dir: Path,
        registry: TechniqueRegistry,
        agent_command: Sequence[str],
        agent_model: str | None,
    ) -> None:
        self.campaign_dir = campaign_dir.resolve()
        self.registry = registry
        self.agent_command = tuple(agent_command)
        self.agent_model = agent_model

    def audit(
        self,
        *,
        technique: str,
        executor_worktree: Path,
        manifest: Any,
        equivalence: Mapping[str, Any],
    ) -> bool | str | Sequence[str]:
        worktree = executor_worktree.resolve()
        head = run(["git", "rev-parse", "HEAD"], cwd=worktree).stdout.strip()
        if head != manifest.candidate_commit:
            return "candidate_commit is not the executor worktree HEAD"
        ancestor = run(
            [
                "git",
                "merge-base",
                "--is-ancestor",
                manifest.base_commit,
                manifest.candidate_commit,
            ],
            cwd=worktree,
            check=False,
        )
        if ancestor.returncode != 0:
            return "candidate commit is not descended from its declared base"
        patch = run(
            [
                "git",
                "diff",
                "--no-ext-diff",
                "--unified=0",
                f"{manifest.base_commit}..{manifest.candidate_commit}",
            ],
            cwd=worktree,
        ).stdout
        additions = "\n".join(
            line[1:]
            for line in patch.splitlines()
            if line.startswith("+") and not line.startswith("+++")
        )
        if not additions.strip():
            return "candidate has no added implementation content"
        forbidden = _LOSSLESS_FORBIDDEN_ADDITIONS.search(additions)
        if forbidden is not None:
            return (
                "lossless candidate adds a forbidden approximation marker: "
                f"{forbidden.group(0)}"
            )
        if equivalence.get("logical_work_unchanged") is not True:
            return "lossless method argument does not preserve logical work"
        if technique not in {"residency", "kernel", "topology"}:
            return f"{technique} is not a registered lossless audit lane"

        method_argument = str(equivalence.get("method_argument", ""))
        method_digest = hashlib.sha256(method_argument.encode()).hexdigest()
        review_dir = (
            self.campaign_dir / "master-method-audits" / manifest.candidate_commit
        )
        review_dir.mkdir(parents=True, exist_ok=True)
        diff_path = review_dir / "candidate.patch"
        equivalence_path = review_dir / "equivalence.json"
        assessment_path = review_dir / "assessment.json"
        if not diff_path.is_file():
            diff_path.write_text(patch, encoding="utf-8")
        if not equivalence_path.is_file():
            _atomic_json(equivalence_path, dict(equivalence))
        if not assessment_path.is_file():
            prompt = review_dir / "master-method-prompt.md"
            prompt.write_text(
                "# Independent lossless method-equivalence audit\n\n"
                "You are the independent master, not the optimization executor. "
                "Read the actual candidate diff and the binding Sol-Engine "
                "contract/scope. Reject approximation, skipped/reduced logical "
                "work, sparsity, sub-16-bit behavior, rank reduction, changed "
                "global denoising steps or DiT calls, and unsupported method "
                "claims. Numeric output similarity is not a correctness gate.\n\n"
                f"Candidate worktree: {worktree}\n"
                f"Base commit: {manifest.base_commit}\n"
                f"Candidate commit: {manifest.candidate_commit}\n"
                f"Technique: {technique}\n"
                f"Diff: {diff_path}\n"
                f"Equivalence evidence: {equivalence_path}\n"
                "Binding master contract: "
                f"{_PACKAGE_ROOT / 'contracts/sol_engine/master.md'}\n"
                f"Binding technique scope: {self.registry[technique].scope}\n\n"
                f"Write only {assessment_path} as JSON with fields: "
                "accepted (boolean), findings (array of strings), "
                "producer='coding-agent-built-in-reasoning', external_api=false, "
                "technique, base_commit, candidate_commit, and "
                f"method_argument_sha256='{method_digest}'.\n",
                encoding="utf-8",
            )
            argv = build_agent_argv(self.agent_command, self.agent_model, prompt)
            stdout_path = review_dir / "stdout.log"
            _atomic_json(
                review_dir / "MASTER-METHOD-COMMAND.json",
                {
                    "schema_version": 1,
                    "argv": redact_argv(argv),
                    "cwd": str(review_dir),
                    "prompt_sha256": sha256_file(prompt),
                    "campaign_id": self.campaign_dir.name,
                    "agent_role": "master_method",
                    "technique": technique,
                    "invocation_id": (f"master-method:{manifest.candidate_commit}"),
                    "stdout": str(stdout_path),
                },
            )
            result = run(argv, cwd=review_dir, check=False)
            stdout_path.write_text(result.stdout, encoding="utf-8")
            (review_dir / "stderr.log").write_text(result.stderr, encoding="utf-8")
            if result.returncode != 0 or not assessment_path.is_file():
                return (
                    "independent coding-agent method audit failed to produce "
                    "a durable assessment"
                )

        assessment = _read_object(assessment_path)
        expected = {
            "producer": "coding-agent-built-in-reasoning",
            "external_api": False,
            "technique": technique,
            "base_commit": manifest.base_commit,
            "candidate_commit": manifest.candidate_commit,
            "method_argument_sha256": method_digest,
        }
        mismatches = [
            name for name, value in expected.items() if assessment.get(name) != value
        ]
        if mismatches:
            return "independent method audit provenance mismatch: " + ", ".join(
                mismatches
            )
        findings = assessment.get("findings")
        if assessment.get("accepted") is not True:
            if isinstance(findings, list) and findings:
                return [str(item) for item in findings]
            return "independent master rejected method equivalence"
        if not isinstance(findings, list):
            return "independent method audit findings must be an array"
        return True


class LockedSolQualityEvaluator:
    """Independently recompute locked Sol LPIPS and run a master visual review."""

    def __init__(
        self,
        *,
        sol_checkout: Path,
        campaign_dir: Path,
        agent_command: Sequence[str],
        agent_model: str | None,
    ) -> None:
        self.sol_checkout = sol_checkout.resolve()
        self.campaign_dir = campaign_dir.resolve()
        self.agent_command = tuple(agent_command)
        self.agent_model = agent_model

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
        review_dir = self.campaign_dir / "master-quality" / key
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

        visual_verdict = run_dir / "visual_verdict.json"
        if not visual_verdict.is_file():
            raise CampaignRuntimeError(
                "quality-gated delivery lacks visual_verdict.json"
            )
        verdict_digest = sha256_file(visual_verdict)
        master_assessment = review_dir / "master-visual-assessment.json"
        if not master_assessment.is_file():
            prompt = review_dir / "master-visual-prompt.md"
            prompt.write_text(
                "# Independent SGLang Diffusion visual review\n\n"
                f"Baseline aligned frames: {baseline_frames.resolve()}\n"
                f"Candidate frames/media: {candidate_frames.resolve()}\n"
                f"Executor visual verdict: {visual_verdict}\n"
                f"Executor verdict SHA-256: {verdict_digest}\n"
                f"Aligned input manifest: {review_dir / 'INPUTS.json'}\n"
                f"Independent LPIPS summary: "
                f"{review_dir / 'lpips-assessment.json'}\n"
                f"Write only {master_assessment} as JSON. Inspect all five "
                "prompts with your built-in multimodal vision. Do not use an "
                "external vision API. Required fields: overall ('pass' or "
                "'fail'), producer='coding-agent-built-in-vision', "
                "external_api=false, reviewed_verdict_sha256, and "
                "prompt_evidence with at least five entries.\n",
                encoding="utf-8",
            )
            argv = build_agent_argv(self.agent_command, self.agent_model, prompt)
            stdout_path = review_dir / "master-visual.stdout.log"
            _atomic_json(
                review_dir / "MASTER-VISUAL-COMMAND.json",
                {
                    "schema_version": 1,
                    "argv": redact_argv(argv),
                    "cwd": str(review_dir),
                    "prompt_sha256": sha256_file(prompt),
                    "campaign_id": self.campaign_dir.name,
                    "agent_role": "master_visual",
                    "technique": None,
                    "invocation_id": f"master-visual:{key}",
                    "stdout": str(stdout_path),
                },
            )
            result = run(argv, cwd=review_dir, check=False)
            stdout_path.write_text(result.stdout, encoding="utf-8")
            (review_dir / "master-visual.stderr.log").write_text(
                result.stderr, encoding="utf-8"
            )
            if result.returncode != 0 or not master_assessment.is_file():
                raise CampaignRuntimeError(
                    "independent coding-agent visual review failed"
                )

        master = _read_object(master_assessment)
        if (
            master.get("producer") != "coding-agent-built-in-vision"
            or master.get("external_api") is not False
            or master.get("reviewed_verdict_sha256") != verdict_digest
            or not isinstance(master.get("prompt_evidence"), list)
            or len(master["prompt_evidence"]) < 5
        ):
            raise CampaignRuntimeError(
                "independent visual assessment has invalid provenance"
            )
        return {
            "aligned": True,
            "prompt_scores": prompt_scores,
            "lpips_mean": lpips_summary["lpips_mean"],
            "lpips_max": lpips_summary["lpips_max"],
            "visual_overall": master.get("overall"),
            "visual_verdict_sha256": verdict_digest,
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
                "independent LPIPS requires exactly five aligned prompt directories"
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
                    "independent LPIPS frame alignment failed for prompt "
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


def _process_actually_alive(pid: int) -> bool:
    """Reap our exited child or reject a zombie recovered from a prior runner."""

    try:
        waited, _ = os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        status = run(
            ["ps", "-o", "stat=", "-p", str(pid)],
            cwd=Path.cwd(),
            check=False,
        )
        state = status.stdout.strip()
        return status.returncode == 0 and bool(state) and not state.startswith("Z")
    return waited == 0


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
        self.runner = AgentRunner(goal.agent.command, goal.agent.model)
        self.executors = ExecutorManager(
            self.campaign_dir,
            state=store,
            sources=self.source_manager,
            runner=self.runner,
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

    def _ensure_gpu_inventory(self) -> GpuInventory | None:
        """Freeze the baseline-selected GPU identity before executors can run."""

        inventory_path = self.campaign_dir / "GPU-INVENTORY.json"
        if inventory_path.is_file():
            inventory = GpuInventory.model_validate_json(
                inventory_path.read_text(encoding="utf-8")
            )
            if inventory.gpu_count != self.goal.hardware.gpu_count:
                raise CampaignRuntimeError(
                    "frozen GPU inventory differs from campaign gpu_count"
                )
            return inventory

        template = self._command_template()
        declared_visibility = (
            template.env.get("CUDA_VISIBLE_DEVICES") if template is not None else None
        )
        if declared_visibility is not None:
            visibility_source = "frozen_command_env"
            visibility = declared_visibility
        elif "CUDA_VISIBLE_DEVICES" in os.environ:
            visibility_source = "controller_env"
            visibility = os.environ["CUDA_VISIBLE_DEVICES"]
        else:
            visibility_source = "default_order"
            visibility = None

        query = [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.total",
            "--format=csv,noheader,nounits",
        ]
        try:
            result = run(query, cwd=self.campaign_dir, check=False)
            if result.returncode != 0:
                raise CampaignRuntimeError(
                    "nvidia-smi GPU inventory query failed: "
                    + result.stderr.strip()[:500]
                )
            available: list[GpuInventoryDevice] = []
            for line in result.stdout.splitlines():
                if not line.strip():
                    continue
                fields = [field.strip() for field in line.split(",")]
                if len(fields) != 3:
                    raise CampaignRuntimeError(
                        f"unexpected nvidia-smi inventory row: {line!r}"
                    )
                available.append(
                    GpuInventoryDevice(
                        physical_index=int(fields[0]),
                        uuid=fields[1],
                        total_mib=float(fields[2]),
                    )
                )
            if not available:
                raise CampaignRuntimeError("nvidia-smi returned an empty GPU inventory")

            tokens = None
            if visibility is not None:
                tokens = [token.strip() for token in visibility.split(",")]
                if not tokens or any(not token or token == "-1" for token in tokens):
                    raise CampaignRuntimeError(
                        "CUDA_VISIBLE_DEVICES selects no usable baseline GPU"
                    )
                selected = [
                    self._resolve_inventory_token(token, available) for token in tokens
                ]
            else:
                selected = available
            gpu_count = self.goal.hardware.gpu_count
            if len(selected) < gpu_count:
                raise CampaignRuntimeError(
                    "visible GPU inventory is smaller than frozen gpu_count"
                )
            selected = selected[:gpu_count]
            inventory = GpuInventory(
                gpu_count=gpu_count,
                visibility_source=visibility_source,
                visible_device_tokens=tokens,
                baseline_command_template_sha256=(
                    template.template_sha256 if template is not None else None
                ),
                devices=selected,
            )
            _atomic_json(inventory_path, inventory.model_dump(mode="json"))
            return inventory
        except (OSError, RuntimeError, ValueError) as error:
            _atomic_json(
                self.campaign_dir / "GPU-INVENTORY-UNAVAILABLE.json",
                {
                    "schema_version": 1,
                    "reason": str(error),
                    "effect": (
                        "residency candidates are fail-closed; other lanes may continue"
                    ),
                },
            )
            return None

    @staticmethod
    def _resolve_inventory_token(
        token: str, available: Sequence[GpuInventoryDevice]
    ) -> GpuInventoryDevice:
        if token.startswith("MIG-"):
            raise CampaignRuntimeError(
                "MIG CUDA visibility cannot yet be bound to the physical GPU inventory"
            )
        if token.isdigit():
            matches = [item for item in available if item.physical_index == int(token)]
        else:
            matches = [item for item in available if item.uuid.startswith(token)]
        if len(matches) != 1:
            raise CampaignRuntimeError(
                f"CUDA_VISIBLE_DEVICES token {token!r} does not resolve uniquely"
            )
        return matches[0]

    def freeze_sources_and_baseline(self) -> StepResult:
        locks = self._ensure_source_locks()
        worktrees = self._ensure_source_worktrees(locks)
        self._sync_knowledge(locks, worktrees)
        inventory = self._ensure_gpu_inventory()

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
                "gpu_inventory": (
                    str(self.campaign_dir / "GPU-INVENTORY.json")
                    if inventory is not None
                    else None
                ),
            },
        )

    def profile_and_route(self) -> StepResult:
        route_path = self.campaign_dir / "ROUTES.json"
        if route_path.is_file():
            value = _read_object(route_path)
            profile_artifact = Path(value["profile_digest"])
            try:
                if value.get("schema_version") != 2:
                    raise ProfileError("route manifest predates target-aware routing")
                if value.get("target_speedup") != self.goal.goal.target_speedup:
                    raise ProfileError("route manifest target differs from frozen goal")
                cached = ProfileDigest.model_validate_json(
                    profile_artifact.read_text(encoding="utf-8")
                )
                Profiler.validate_digest(cached)
            except (OSError, UnicodeError, ValueError, ProfileError):
                rejected = route_path.with_name(
                    f"ROUTES.rejected-{sha256_file(route_path)[:12]}.json"
                )
                if not rejected.exists():
                    route_path.replace(rejected)
            else:
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
            target_speedup=self.goal.goal.target_speedup,
        )
        unknown = set(routes) - set(self.registry.names())
        if unknown:
            raise CampaignRuntimeError(
                "router selected unregistered techniques: " + ", ".join(sorted(unknown))
            )
        _atomic_json(
            route_path,
            {
                "schema_version": 2,
                "routes": routes,
                "route_policy": router.route_policy,
                "target_speedup": self.goal.goal.target_speedup,
                "evidence": router.last_evidence,
                "profile_digest": str(profile_path),
                "sglang_commit": locks["sglang"].commit,
            },
        )
        return StepResult(
            CampaignStatus.PROFILED,
            payload={"routes": routes, "route_artifact": str(route_path)},
        )

    def start_search_epoch(self, epoch: int) -> StepResult:
        routes = self._routes()
        search_root = self.campaign_dir / "search" / str(epoch)
        manifest_path = search_root / "EXECUTORS.json"
        if not manifest_path.is_file():
            _atomic_json(
                manifest_path,
                {
                    "schema_version": 2,
                    "epoch": epoch,
                    "routes": routes,
                    "executors": {},
                    "active_technique": None,
                },
            )
        active = self._ensure_next_executor(epoch)
        return StepResult(
            CampaignStatus.SEARCHING,
            payload={
                "executors": str(manifest_path),
                "routes": routes,
                "active_technique": active,
                "gpu_measurements_serial": True,
            },
        )

    def poll_and_verify_executors(self, epoch: int) -> StepResult:
        from .verifier import DeliveryVerifier

        routes = self._routes()
        handles = self._epoch_handles(epoch)
        verified = self._load_verified(epoch)
        dispositions = self._load_dispositions()
        deferred = self._load_deferred_lanes(epoch)
        baseline = BaselineRunner.load(self.campaign_dir / "BASELINE.json")
        quality = LockedSolQualityEvaluator(
            sol_checkout=self.campaign_dir / "source-worktrees" / "sol_engine",
            campaign_dir=self.campaign_dir,
            agent_command=self.goal.agent.command,
            agent_model=self.goal.agent.model,
        )
        verifier = DeliveryVerifier(
            registry=self.registry,
            baseline=baseline,
            campaign_artifact_root=self.campaign_dir,
            method_auditor=IndependentMasterMethodAuditor(
                campaign_dir=self.campaign_dir,
                registry=self.registry,
                agent_command=self.goal.agent.command,
                agent_model=self.goal.agent.model,
            ),
            quality_evaluator=quality,
            command_template=self._command_template(),
        )

        for technique in routes:
            if technique in verified or technique in dispositions or technique in deferred:
                continue
            if technique not in handles:
                active = self._ensure_next_executor(epoch)
                if active is None:
                    break
                return StepResult(
                    None,
                    payload={"reason": "executor_started", "technique": active},
                )
            handle = handles[technique]
            polled = self.executors.poll(handle)
            if polled.alive and _process_actually_alive(handle.pid):
                return StepResult(
                    None,
                    payload={"reason": "agent_running", "technique": technique},
                )
            if not polled.delivered or polled.delivery is None:
                disposition_path = handle.worktree / "DISPOSITION.json"
                if disposition_path.exists() or disposition_path.is_symlink():
                    try:
                        disposition_path = require_regular_delivery(
                            handle.worktree, disposition_path
                        )
                        disposition = self._validate_disposition(
                            disposition_path, technique
                        )
                    except (OSError, ValueError, CampaignRuntimeError) as error:
                        return self._resume_or_exhaust(
                            handle, f"Invalid DISPOSITION.json: {error}", epoch=epoch
                        )
                    if disposition.classification == "blocked":
                        return self._resume_or_exhaust(
                            handle,
                            "A blocked disposition is recoverable and cannot close "
                            "the lane; repair the blocker or keep searching.",
                            epoch=epoch,
                        )
                    self._store_disposition(disposition, disposition_path)
                    active = self._ensure_next_executor(epoch)
                    if active is not None:
                        return StepResult(None, payload={
                            "reason": "lane_dispositioned",
                            "technique": technique,
                            "next_technique": active,
                        })
                    break
                return self._resume_or_exhaust(
                    handle,
                    "The process exited without a regular DELIVERY.json inside "
                    "its assigned worktree.",
                    epoch=epoch,
                )

            result = verifier.verify(
                polled.delivery,
                technique=technique,
                executor_worktree=handle.worktree,
            )
            if not result.accepted:
                for measurement in result.authenticated_measurements:
                    self._record_scientific_round(
                        handle,
                        polled.delivery,
                        outcome=(
                            "improved"
                            if measurement.authoritative_speedup > 1.0
                            else "regressed"
                        ),
                        measurement=measurement,
                    )
                feedback = "\n".join(
                    f"{index}. [{finding.code}] {finding.message}"
                    for index, finding in enumerate(result.findings, start=1)
                )
                return self._resume_or_exhaust(handle, feedback, epoch=epoch)
            if not result.verified_points:
                return self._resume_or_exhaust(
                    handle,
                    "Verifier accepted no durable frontier point.",
                    epoch=epoch,
                )
            point = max(
                result.verified_points,
                key=lambda item: item.authoritative_speedup,
            )
            self._record_scientific_round(
                handle,
                polled.delivery,
                outcome="improved",
                measurement=next(
                    (
                        item
                        for item in result.authenticated_measurements
                        if item.candidate_id == point.candidate_id
                    ),
                    None,
                ),
            )
            manifest = point.implementation_manifest
            candidate = VerifiedCandidate(
                candidate_id=point.candidate_id,
                technique=technique,
                base_commit=manifest.base_commit,
                candidate_commit=manifest.candidate_commit,
                correctness=CorrectnessMode(self.registry[technique].correctness),
                activation=CandidateActivation(
                    env=dict(point.activation.get("env", {})),
                    server_args=list(point.activation.get("server_args", [])),
                ),
                source_hashes=point.source_hashes,
                compatibility_notes=[
                    f"verified in isolated {technique} executor",
                    f"authoritative speedup {point.authoritative_speedup:.8g}x",
                ],
                verified_speedup=point.authoritative_speedup,
                verified=True,
            )
            verified[technique] = candidate
            self._write_verified(epoch, verified)
            self._register_candidate(candidate, epoch=epoch)
            active = self._ensure_next_executor(epoch)
            if active is not None:
                return StepResult(
                    None,
                    payload={
                        "reason": "next_serial_executor_started",
                        "technique": active,
                    },
                )

        selected = self._selected_candidates(epoch)
        if not selected:
            if self._load_deferred_lanes(epoch):
                return StepResult(
                    CampaignStatus.INFRA_BLOCKED,
                    payload={
                        "reason": "all_productive_lanes_finished_with_deferred_executors",
                        "deferred_lanes": sorted(self._load_deferred_lanes(epoch)),
                    },
                )
            return StepResult(
                CampaignStatus.SEARCH_SPACE_EXHAUSTED,
                payload={"reason": "all_lanes_dispositioned_without_positive_candidate"},
            )
        return StepResult(
            CampaignStatus.INTEGRATING,
            payload={
                "verified_candidates": str(self._verified_path(epoch)),
                "candidate_ids": [
                    selected[name].candidate_id
                    for name in routes
                    if name in selected
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
        verified = self._selected_candidates(epoch)
        routes = self._routes()
        if not verified:
            return StepResult(
                CampaignStatus.SEARCHING,
                payload={"reason": "no_positive_candidates_to_integrate"},
            )
        quality = LockedSolQualityEvaluator(
            sol_checkout=self.campaign_dir / "source-worktrees" / "sol_engine",
            campaign_dir=self.campaign_dir,
            agent_command=self.goal.agent.command,
            agent_model=self.goal.agent.model,
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
                [
                    verified[name].candidate_id
                    for name in routes
                    if name in verified
                ],
                {item.candidate_id: item for item in verified.values()},
                epoch_root / f"attempt-{attempt:03d}",
            )
        except IntegrationError as error:
            feedback = (
                "Independent combined full-workload integration gate failed: "
                f"{error}"
            )
            failed_technique = next(
                (name for name in reversed(routes) if name in verified), None
            )
            if failed_technique is None:
                raise CampaignRuntimeError("integration failed without a candidate")
            failed = verified[failed_technique]
            self._exclude_candidate(failed.candidate_id, feedback)
            self._remove_verified(epoch, failed_technique)
            handle = self._epoch_handles(epoch).get(failed_technique)
            if handle is None:
                self._ensure_next_executor(epoch, preferred=failed_technique)
            else:
                outcome = self._resume_or_exhaust(handle, feedback, epoch=epoch)
                if outcome.next_status is CampaignStatus.INFRA_BLOCKED:
                    return outcome
            return StepResult(
                CampaignStatus.SEARCHING,
                payload={
                    "reason": "integrated_gate_rejected",
                    "feedback": feedback,
                },
            )
        if result.status == "needs_executor_revision":
            assert result.failed_candidate_id is not None
            failed_technique = next(
                name
                for name, candidate in verified.items()
                if candidate.candidate_id == result.failed_candidate_id
            )
            handle = self._epoch_handles(epoch).get(failed_technique)
            feedback = (
                "Canonical integration conflict. Rebase the candidate on the "
                "locked SGLang base and make it composition-compatible. "
                f"Diagnostics: {result.diagnostics_path}"
            )
            rounds = self._technique_rounds(failed_technique)
            if rounds >= self.registry[failed_technique].round_budget:
                self._write_budget_disposition(failed_technique, rounds)
                self._exclude_candidate(result.failed_candidate_id, feedback)
                return StepResult(CampaignStatus.SEARCHING, payload={
                    "reason": "technique_round_budget_exhausted",
                    "technique": failed_technique,
                    "rounds": rounds,
                })
            self._remove_verified(epoch, failed_technique)
            self._exclude_candidate(result.failed_candidate_id, feedback)
            if handle is None:
                self._ensure_next_executor(epoch, preferred=failed_technique)
            else:
                self.executors.resume(
                    handle,
                    feedback=feedback,
                    idempotency_key=(
                        f"{self.campaign_id}:{epoch}:integration-conflict:"
                        f"{failed_technique}:{handle.attempt}"
                    ),
                )
            return StepResult(
                CampaignStatus.SEARCHING,
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
            if self._has_search_budget(epoch):
                return StepResult(
                    CampaignStatus.PROFILED,
                    payload={
                        "reason": "target_not_reached",
                        "verified_speedup": speedup,
                    },
                    verified_speedup=speedup,
                    new_hypothesis=True,
                )
            return StepResult(
                CampaignStatus.SEARCH_SPACE_EXHAUSTED,
                payload={
                    "reason": "technique_round_budgets_exhausted",
                    "verified_speedup": speedup,
                },
                verified_speedup=speedup,
            )

        locks = self._load_locks()
        worktree = Path(receipt["worktree"]).resolve()
        try:
            quality_evidence, quality_issues, quality_path = (
                self._ensure_final_quality_evidence(
                    epoch=epoch,
                    integrated_commit=str(receipt["integration_commit"]),
                    candidate_run=delivery.frontier_points[0].run_dir,
                )
            )
        except CampaignRuntimeError as error:
            return StepResult(
                CampaignStatus.INFRA_BLOCKED,
                payload={
                    "reason": "final_quality_evaluator_failed",
                    "detail": str(error),
                },
                verified_speedup=speedup,
            )
        if quality_issues:
            payload = {
                "reason": "final_quality_gate_rejected",
                "issues": quality_issues,
                "quality_evidence": str(quality_path),
            }
            if self._has_search_budget(epoch):
                return StepResult(
                    CampaignStatus.PROFILED,
                    payload=payload,
                    verified_speedup=speedup,
                    new_hypothesis=True,
                )
            return StepResult(
                CampaignStatus.SEARCH_SPACE_EXHAUSTED,
                payload=payload,
                verified_speedup=speedup,
            )
        packager = PatchPackager(worktree, base_sha=locks["sglang"].commit)
        profile = packager.validate(model_slug=_model_slug(self.goal.model.id))
        if not math.isclose(profile.speedup, speedup, rel_tol=1e-6, abs_tol=1e-9):
            raise CampaignRuntimeError(
                "agent profile speedup differs from integrated measurement"
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
                        quality_path,
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
                    "quality_evidence": str(quality_path),
                    "quality_producer": quality_evidence.producer,
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

    def _ensure_final_quality_evidence(
        self,
        *,
        epoch: int,
        integrated_commit: str,
        candidate_run: Path,
    ) -> tuple[FinalQualityEvidence, list[str], Path]:
        quality_root = self.campaign_dir / "final-quality" / str(epoch)
        quality_path = quality_root / "QUALITY-EVIDENCE.json"
        baseline = BaselineRunner.load(self.campaign_dir / "BASELINE.json")
        if not quality_path.is_file():
            quality_root.mkdir(parents=True, exist_ok=True)
            prompt = quality_root / "master-quality-prompt.md"
            prompt.write_text(
                "# Independent final media quality gate\n\n"
                f"Integrated commit: {integrated_commit}\n"
                f"Baseline run: {baseline.run_dir.resolve()}\n"
                f"Baseline aligned frames: {baseline.baseline_frames.resolve()}\n"
                f"Candidate run: {candidate_run.resolve()}\n"
                f"Validation prompts: {(self.campaign_dir / 'validation-prompts.txt').resolve()}\n"
                f"Expected width: {self.goal.workload.width}\n"
                f"Expected height: {self.goal.workload.height}\n"
                f"Expected fps: {self.goal.workload.fps}\n"
                f"Expected frames: {self.goal.workload.frames}\n"
                f"Required output: {quality_path}\n\n"
                "Act as the independent Master, not an Executor. Use the pinned "
                "Sol/KDA tools available in this campaign. Run and retain command "
                "receipts for LPIPS, VBench, audio analysis, AV-sync/media probing, "
                "and visual review. Inspect exactly five prompts. Write only the "
                "required QUALITY-EVIDENCE.json matching the checked-in schema. "
                "Set producer='independent-master' and external_api=false. Never "
                "convert missing tooling or evidence into a pass.\n",
                encoding="utf-8",
            )
            argv = build_agent_argv(self.goal.agent.command, self.goal.agent.model, prompt)
            _atomic_json(
                quality_root / "MASTER-QUALITY-COMMAND.json",
                {
                    "schema_version": 1,
                    "argv": redact_argv(argv),
                    "cwd": str(quality_root),
                    "prompt_sha256": sha256_file(prompt),
                    "campaign_id": self.campaign_id,
                    "agent_role": "master_final_quality",
                    "epoch": epoch,
                },
            )
            result = run(argv, cwd=quality_root, check=False)
            (quality_root / "master-quality.stdout.log").write_text(
                result.stdout, encoding="utf-8"
            )
            (quality_root / "master-quality.stderr.log").write_text(
                result.stderr, encoding="utf-8"
            )
            if result.returncode != 0 or not quality_path.is_file():
                raise CampaignRuntimeError(
                    "independent Master did not produce QUALITY-EVIDENCE.json"
                )
        try:
            evidence = FinalQualityEvidence.model_validate_json(
                quality_path.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, ValueError) as error:
            raise CampaignRuntimeError(
                f"invalid final quality evidence: {error}"
            ) from error
        issues = self._validate_final_quality_evidence(
            evidence,
            integrated_commit=integrated_commit,
            baseline=baseline,
            evidence_root=quality_root,
        )
        return evidence, issues, quality_path

    def _validate_final_quality_evidence(
        self,
        evidence: FinalQualityEvidence,
        *,
        integrated_commit: str,
        baseline: Any,
        evidence_root: Path,
    ) -> list[str]:
        issues: list[str] = []
        if evidence.integrated_commit != integrated_commit:
            issues.append("quality evidence is bound to another integrated commit")
        audio_required = self._baseline_audio_required(baseline.run_dir)
        if evidence.audio_required != audio_required:
            issues.append("audio requirement differs from the frozen baseline streams")
        receipt_paths: set[Path] = set()
        for receipt in evidence.command_receipts:
            path = receipt.path
            resolved = path.resolve() if path.is_absolute() else (evidence_root / path).resolve()
            if not self._inside(resolved, self.campaign_dir) or not resolved.is_file():
                issues.append(f"quality command receipt is unsafe or missing: {path}")
                continue
            if sha256_file(resolved) != receipt.sha256:
                issues.append(f"quality command receipt hash mismatch: {path}")
            receipt_paths.add(resolved)
        if len(receipt_paths) < 4:
            issues.append("quality gate requires four distinct independent tool receipts")

        vbench_baseline: list[float] = []
        vbench_candidate: list[float] = []
        for prompt in evidence.prompts:
            values = (
                prompt.lpips,
                prompt.media.fps,
                prompt.media.video_duration_s,
                *(prompt.vbench_baseline.model_dump().values()),
                *(prompt.vbench_candidate.model_dump().values()),
            )
            if any(not math.isfinite(float(value)) for value in values):
                issues.append(f"prompt {prompt.prompt_index} contains non-finite metrics")
            if prompt.lpips > evidence.thresholds.lpips_max:
                issues.append(f"prompt {prompt.prompt_index} exceeds LPIPS threshold")
            if prompt.visual != "pass":
                issues.append(f"prompt {prompt.prompt_index} failed visual review")
            media = prompt.media
            if (
                media.width != self.goal.workload.width
                or media.height != self.goal.workload.height
                or not math.isclose(media.fps, self.goal.workload.fps, rel_tol=1e-3)
                or media.frame_count != self.goal.workload.frames
            ):
                issues.append(f"prompt {prompt.prompt_index} violates media contract")
            vbench_baseline.extend(prompt.vbench_baseline.model_dump().values())
            vbench_candidate.extend(prompt.vbench_candidate.model_dump().values())
            if audio_required:
                audio = prompt.audio
                if (
                    not media.audio_codec
                    or not audio.present
                    or audio.duration_s <= 0
                    or audio.sample_rate <= 0
                    or audio.channels <= 0
                    or audio.silence_ratio > evidence.thresholds.silence_ratio_max
                    or audio.clipping_ratio > evidence.thresholds.clipping_ratio_max
                ):
                    issues.append(f"prompt {prompt.prompt_index} failed audio quality")
                if (
                    prompt.av_sync_drift_ms is None
                    or not math.isfinite(prompt.av_sync_drift_ms)
                    or abs(prompt.av_sync_drift_ms)
                    > evidence.thresholds.av_sync_drift_ms_max
                ):
                    issues.append(f"prompt {prompt.prompt_index} failed AV sync")
        if statistics.fmean(vbench_candidate) < (
            statistics.fmean(vbench_baseline)
            - evidence.thresholds.vbench_max_mean_regression
        ):
            issues.append("aggregate VBench regressed beyond the configured threshold")
        return issues

    @staticmethod
    def _baseline_audio_required(run_dir: Path) -> bool:
        videos = sorted(
            path
            for path in run_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm"}
        )
        if not videos:
            return False
        ffprobe = shutil.which("ffprobe")
        if ffprobe is None:
            raise CampaignRuntimeError(
                "ffprobe is required to freeze the baseline audio-stream contract"
            )
        for video in videos:
            result = run(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-select_streams",
                    "a",
                    "-show_entries",
                    "stream=index",
                    "-of",
                    "json",
                    str(video),
                ],
                cwd=run_dir,
                check=False,
            )
            if result.returncode != 0:
                raise CampaignRuntimeError(
                    f"ffprobe could not inspect frozen baseline media: {video}"
                )
            try:
                payload = json.loads(result.stdout)
            except json.JSONDecodeError as error:
                raise CampaignRuntimeError("ffprobe returned invalid JSON") from error
            if payload.get("streams"):
                return True
        return False

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
                if name == "kda_pilot":
                    self.source_manager.ensure_submodules(
                        destination,
                        required=_KDA_REQUIRED_SUBMODULES,
                    )
            else:
                self.source_manager.create_worktree(
                    lock,
                    destination,
                    initialize_submodules=name == "kda_pilot",
                    required_submodules=(
                        _KDA_REQUIRED_SUBMODULES if name == "kda_pilot" else ()
                    ),
                )
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
            kwargs = {
                "name": name,
                "checkout": worktrees[name],
                "commit": lock.commit,
                "patterns": patterns,
                "output_dir": output,
                "required_prefixes": _KNOWLEDGE_REQUIRED_PREFIXES.get(name, ()),
            }
            try:
                snapshot = sync_source(**kwargs)
            except KnowledgeSyncError:
                if not output.exists():
                    raise
                rejected_root = self.campaign_dir / "knowledge" / "rejected" / name
                rejected_root.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256(
                    (output / "index.json").read_bytes()
                    if (output / "index.json").is_file()
                    else str(output).encode()
                ).hexdigest()[:12]
                rejected = rejected_root / f"{lock.commit}-{digest}"
                if rejected.exists():
                    raise CampaignRuntimeError(
                        f"knowledge recovery target already exists: {rejected}"
                    )
                output.replace(rejected)
                snapshot = sync_source(**kwargs)
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

    def _executor_prompt(self, technique: str, epoch: int) -> ExecutorPrompt:
        placement = detect_placement_contract(
            self.campaign_dir / "source-worktrees" / "sglang"
        ).render(model_slug=_model_slug(self.goal.model.id))
        knowledge_manifest = _read_object(self.campaign_dir / "KNOWLEDGE.json")
        locked_knowledge = tuple(
            PromptSection.from_path(
                f"Locked {name} optimization knowledge",
                Path(index),
            )
            for name, index in sorted(knowledge_manifest["snapshots"].items())
        )
        history_catalog = HistoryRuleCatalog.load(
            _PACKAGE_ROOT / "knowledge" / "history-rules.toml", self.registry
        )
        history_rules = history_catalog.for_technique(technique)
        history_knowledge = (
            (
                PromptSection(
                    f"Diff-reviewed {technique} historical rules",
                    history_catalog.render(technique),
                    "checked-in:knowledge/history-rules.toml"
                    f"#sha256={history_catalog.sha256}",
                ),
            )
            if history_rules
            else ()
        )
        knowledge = history_knowledge + locked_knowledge
        baseline = (self.campaign_dir / "BASELINE.json").read_text(encoding="utf-8")
        command_template = self.campaign_dir / "BASELINE-COMMAND.json"
        if command_template.is_file():
            baseline += (
                "\n\nFrozen user baseline command template (binding):\n"
                "Materialize this template for your assigned worktree and "
                "candidate run directory. Preserve every frozen workload flag, "
                "append only your declared activation, and retain the generated "
                "COMMAND.json. The independent Master compares its full argv, "
                "cwd, environment, and template SHA-256; another benchmark "
                "command is rejected.\n" + command_template.read_text(encoding="utf-8")
            )
        inventory_path = self.campaign_dir / "GPU-INVENTORY.json"
        unavailable_inventory = self.campaign_dir / "GPU-INVENTORY-UNAVAILABLE.json"
        if inventory_path.is_file():
            baseline += (
                "\n\nController-owned baseline GPU inventory (binding):\n"
                + inventory_path.read_text(encoding="utf-8")
            )
        elif unavailable_inventory.is_file():
            baseline += (
                "\n\nGPU inventory unavailable; residency candidates must fail "
                "closed, while independent non-residency lanes may continue:\n"
                + unavailable_inventory.read_text(encoding="utf-8")
            )
        profile = (
            self.campaign_dir / "profiles" / "0" / "PROFILE-DIGEST.json"
        ).read_text(encoding="utf-8")
        contract = (
            (_PACKAGE_ROOT / "contracts" / "sol_engine" / "loop-and-gate.md").read_text(
                encoding="utf-8"
            )
            + "\n"
            + (_PACKAGE_ROOT / "contracts" / "sol_engine" / "master.md").read_text(
                encoding="utf-8"
            )
        )
        return ExecutorPrompt(
            correctness_contract=PromptSection(
                "Sol-Engine correctness and master contract",
                contract,
                "checked-in:contracts/sol_engine",
            ),
            technique_scope=PromptSection.from_path(
                f"{technique} technique scope",
                self.registry[technique].scope,
            ),
            placement_rules=PromptSection(
                "SGLang generated-kernel placement and registration",
                placement,
                "detected:locked-SGLang-layout",
            ),
            knowledge=knowledge,
            baseline=PromptSection(
                "Frozen baseline and profile evidence",
                baseline + "\n" + profile,
                str(self.campaign_dir / "BASELINE.json"),
            ),
            search_state={
                "campaign_id": self.campaign_id,
                "epoch": epoch,
                "technique": technique,
                "round_budget": self.registry[technique].round_budget,
                "target_speedup": self.goal.goal.target_speedup,
                "best_verified_speedup": self._best_verified_speedup(),
                "remaining_multiplier": (
                    self.goal.goal.target_speedup / self._best_verified_speedup()
                ),
                "prior_failures": self.store.failures(self.campaign_id),
            },
            rejected_signatures=tuple(
                item["signature"] for item in self.store.failures(self.campaign_id)
            ),
            delivery_contract=build_delivery_contract(
                campaign=self.campaign_dir,
                technique=technique,
                registry=self.registry,
                baseline=BaselineRunner.load(self.campaign_dir / "BASELINE.json"),
                command_template=self._command_template(),
            ),
        )

    def _best_verified_speedup(self) -> float:
        best = 1.0
        path = self._candidate_registry_path()
        if not path.is_file():
            return best
        for item in _read_object(path).get("history", []):
            candidate = item.get("candidate") if isinstance(item, dict) else None
            speedup = (
                candidate.get("verified_speedup")
                if isinstance(candidate, dict)
                else None
            )
            if isinstance(speedup, (int, float)) and not isinstance(speedup, bool):
                best = max(best, float(speedup))
        return best

    def _epoch_handles(self, epoch: int) -> dict[str, ExecutorHandle]:
        root = self.campaign_dir / "search" / str(epoch)
        manifest = _read_object(root / "EXECUTORS.json")
        handles: dict[str, ExecutorHandle] = {}
        for technique, value in manifest["executors"].items():
            executor_root = Path(value["root"])
            handles[str(technique)] = _load_handle(executor_root / "executor.json")
        return handles

    def _ensure_next_executor(
        self, epoch: int, *, preferred: str | None = None
    ) -> str | None:
        root = self.campaign_dir / "search" / str(epoch)
        manifest_path = root / "EXECUTORS.json"
        manifest = _read_object(manifest_path)
        handles = self._epoch_handles(epoch)
        verified = self._load_verified(epoch)
        dispositions = self._load_dispositions()
        deferred = self._load_deferred_lanes(epoch)
        active = manifest.get("active_technique")
        if (
            isinstance(active, str)
            and active in handles
            and active not in verified
            and active not in dispositions
            and active not in deferred
        ):
            return active

        routes = self._routes()
        ordered = list(routes)
        if preferred in ordered:
            ordered.remove(preferred)
            ordered.insert(0, preferred)
        selected: str | None = None
        for technique in ordered:
            if technique in verified or technique in dispositions or technique in deferred:
                continue
            rounds = self._technique_rounds(technique)
            if rounds >= self.registry[technique].round_budget:
                self._write_budget_disposition(technique, rounds)
                dispositions = self._load_dispositions()
                continue
            selected = technique
            break

        if selected is None:
            manifest["active_technique"] = None
            _atomic_json(manifest_path, manifest)
            return None
        if selected not in handles:
            lock = self._load_locks()["sglang"]
            handle = self.executors.spawn(
                campaign_id=self.campaign_id,
                technique=selected,
                source_lock=lock,
                prompt=self._executor_prompt(selected, epoch),
                idempotency_key=f"{self.campaign_id}:{epoch}:executor:{selected}",
            )
            manifest["executors"][selected] = _serialize_handle(handle)
        manifest["active_technique"] = selected
        _atomic_json(manifest_path, manifest)
        return selected

    def _dispositions_path(self) -> Path:
        return self.campaign_dir / "TECHNIQUE-DISPOSITIONS.json"

    def _load_dispositions(self) -> dict[str, TechniqueDisposition]:
        path = self._dispositions_path()
        if not path.is_file():
            return {}
        payload = _read_object(path)
        return {
            str(name): TechniqueDisposition.model_validate(value)
            for name, value in payload.get("techniques", {}).items()
        }

    def _validate_disposition(
        self, path: Path, technique: str
    ) -> TechniqueDisposition:
        disposition = TechniqueDisposition.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        if disposition.technique != technique:
            raise CampaignRuntimeError("disposition technique does not match lane")
        if disposition.classification == "budget_exhausted":
            raise CampaignRuntimeError(
                "only the controller may issue a budget-exhausted disposition"
            )
        if disposition.profile_digest_sha256 != self._profile_digest_sha256():
            raise CampaignRuntimeError("disposition is bound to a stale profile")
        required = set(self.registry[technique].coverage)
        observed = {item.id for item in disposition.coverage}
        if observed != required or len(observed) != len(disposition.coverage):
            raise CampaignRuntimeError(
                "disposition does not cover the exact required candidate families"
            )
        round_ids = {
            str(event["payload"].get("round_id"))
            for event in self.store.events(
                self.campaign_id, event_type="scientific_round_completed"
            )
            if event["payload"].get("technique") == technique
        }
        for item in disposition.coverage:
            if item.status == "measured" and (
                not item.scientific_round_ids
                or not set(item.scientific_round_ids).issubset(round_ids)
            ):
                raise CampaignRuntimeError(
                    f"measured coverage {item.id} lacks authenticated round IDs"
                )
            for evidence in item.evidence:
                resolved = evidence if evidence.is_absolute() else path.parent / evidence
                resolved = resolved.resolve()
                if not (
                    self._inside(resolved, self.campaign_dir)
                    or self._inside(resolved, path.parent)
                ) or not resolved.is_file():
                    raise CampaignRuntimeError(
                        f"disposition evidence is unsafe or missing: {evidence}"
                    )
        if disposition.classification in {"no_gain", "unsupported"} and any(
            item.status == "blocked" for item in disposition.coverage
        ):
            raise CampaignRuntimeError(
                "a terminal no-gain/unsupported disposition cannot hide blocked coverage"
            )
        return disposition

    def _store_disposition(
        self, disposition: TechniqueDisposition, source: Path
    ) -> None:
        current = self._load_dispositions()
        current[disposition.technique] = disposition
        existing_payload = (
            _read_object(self._dispositions_path())
            if self._dispositions_path().is_file()
            else {}
        )
        sources = dict(existing_payload.get("sources", {}))
        sources[disposition.technique] = str(source)
        _atomic_json(
            self._dispositions_path(),
            {
                "schema_version": 1,
                "techniques": {
                    name: value.model_dump(mode="json")
                    for name, value in sorted(current.items())
                },
                "sources": sources,
            },
        )

    def _write_budget_disposition(self, technique: str, rounds: int) -> None:
        if technique in self._load_dispositions():
            return
        disposition = TechniqueDisposition(
            technique=technique,
            classification="budget_exhausted",
            reason=(
                f"The controller authenticated {rounds} complete frozen-workload "
                "measurements, consuming this lane's scientific budget."
            ),
            coverage=[
                {
                    "id": coverage_id,
                    "status": "blocked",
                    "evidence": [self.store.event_log_path],
                    "scientific_round_ids": [],
                }
                for coverage_id in self.registry[technique].coverage
            ],
            profile_digest_sha256=self._profile_digest_sha256(),
        )
        self._store_disposition(disposition, self.store.event_log_path)

    def _record_scientific_round(
        self,
        handle: ExecutorHandle,
        delivery_path: Path,
        *,
        outcome: str,
        measurement: Any | None = None,
    ) -> str:
        if measurement is None:
            payload = _read_object(delivery_path)
            points = payload.get("frontier_points")
            if (
                not isinstance(points, list)
                or not points
                or not isinstance(points[0], dict)
            ):
                raise CampaignRuntimeError(
                    "cannot authenticate a scientific round without a frontier point"
                )
            point = points[0]
            candidate_id = str(point.get("candidate_id", ""))
            performance = point.get("performance")
            if not candidate_id or not isinstance(performance, dict):
                raise CampaignRuntimeError("scientific round delivery is incomplete")
            candidate_mean = performance.get("candidate_mean_e2e_s")
            workload_total = performance.get("candidate_workload_total_s")
            request_count = performance.get("request_count")
            measurement_digest = sha256_file(delivery_path)
            measurement_run = str(point.get("run_dir", ""))
        else:
            candidate_id = measurement.candidate_id
            candidate_mean = measurement.candidate_mean_e2e_s
            workload_total = measurement.candidate_workload_total_s
            request_count = measurement.request_count
            performance_path = measurement.run_dir / "PERFORMANCE.json"
            measurement_digest = sha256_file(performance_path)
            measurement_run = str(measurement.run_dir.resolve())
        round_id = hashlib.sha256(
            f"{handle.technique}\0{candidate_id}\0{measurement_run}\0"
            f"{measurement_digest}".encode()
        ).hexdigest()[:20]
        idempotency_key = f"{self.campaign_id}:scientific-round:{round_id}"
        if any(
            event["idempotency_key"] == idempotency_key
            for event in self.store.events(self.campaign_id)
        ):
            return round_id
        self.store.record_event(
            self.campaign_id,
            "scientific_round_completed",
            idempotency_key,
            {
                "round_id": round_id,
                "technique": handle.technique,
                "candidate_id": candidate_id,
                "delivery_sha256": sha256_file(delivery_path),
                "candidate_mean_e2e_s": candidate_mean,
                "candidate_workload_total_s": workload_total,
                "request_count": request_count,
                "measurement_sha256": measurement_digest,
                "outcome": outcome,
            },
        )
        return round_id

    def _candidate_registry_path(self) -> Path:
        return self.campaign_dir / "CANDIDATE-REGISTRY.json"

    def _register_candidate(self, candidate: VerifiedCandidate, *, epoch: int) -> None:
        path = self._candidate_registry_path()
        payload = _read_object(path) if path.is_file() else {
            "schema_version": 1,
            "history": [],
        }
        history = payload.setdefault("history", [])
        if not any(
            item.get("candidate", {}).get("candidate_id") == candidate.candidate_id
            for item in history
            if isinstance(item, dict)
        ):
            history.append({
                "epoch": epoch,
                "candidate": candidate.model_dump(mode="json"),
            })
        _atomic_json(path, payload)

    def _selected_candidates(self, epoch: int) -> dict[str, VerifiedCandidate]:
        candidates: dict[str, VerifiedCandidate] = {}
        path = self._candidate_registry_path()
        excluded = self._excluded_candidate_ids()
        if path.is_file():
            for item in _read_object(path).get("history", []):
                if not isinstance(item, dict):
                    continue
                candidate = VerifiedCandidate.model_validate(item["candidate"])
                if candidate.candidate_id in excluded:
                    continue
                prior = candidates.get(candidate.technique)
                if prior is None or (candidate.verified_speedup or 0) > (
                    prior.verified_speedup or 0
                ):
                    candidates[candidate.technique] = candidate
        for technique, candidate in self._load_verified(epoch).items():
            if candidate.candidate_id not in excluded:
                prior = candidates.get(technique)
                if prior is None or (candidate.verified_speedup or 0) > (
                    prior.verified_speedup or 0
                ):
                    candidates[technique] = candidate
        return candidates

    def _exclude_candidate(self, candidate_id: str, reason: str) -> None:
        path = self.campaign_dir / "COMPOSITION-EXCLUSIONS.json"
        payload = _read_object(path) if path.is_file() else {
            "schema_version": 1,
            "candidates": {},
        }
        payload["candidates"].setdefault(candidate_id, reason)
        _atomic_json(path, payload)

    def _excluded_candidate_ids(self) -> set[str]:
        path = self.campaign_dir / "COMPOSITION-EXCLUSIONS.json"
        if not path.is_file():
            return set()
        return set(_read_object(path).get("candidates", {}))

    def _profile_digest_sha256(self) -> str:
        return sha256_file(
            self.campaign_dir / "profiles" / "0" / "PROFILE-DIGEST.json"
        )

    @staticmethod
    def _inside(path: Path, root: Path) -> bool:
        try:
            path.resolve().relative_to(root.resolve())
        except ValueError:
            return False
        return True

    def _verified_path(self, epoch: int) -> Path:
        return self.campaign_dir / "search" / str(epoch) / "VERIFIED-CANDIDATES.json"

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

    def _deferred_lanes_path(self, epoch: int) -> Path:
        return self.campaign_dir / "search" / str(epoch) / "DEFERRED-LANES.json"

    def _load_deferred_lanes(self, epoch: int) -> dict[str, dict[str, Any]]:
        path = self._deferred_lanes_path(epoch)
        if not path.is_file():
            return {}
        value = _read_object(path).get("lanes", {})
        if not isinstance(value, dict):
            raise CampaignRuntimeError("DEFERRED-LANES.json has invalid lanes")
        return {
            str(name): dict(payload)
            for name, payload in value.items()
            if isinstance(payload, dict)
        }

    def _defer_lane(
        self,
        epoch: int,
        handle: ExecutorHandle,
        *,
        signature: str,
        feedback: str,
    ) -> None:
        path = self._deferred_lanes_path(epoch)
        payload = _read_object(path) if path.is_file() else {
            "schema_version": 1,
            "epoch": epoch,
            "lanes": {},
        }
        payload["lanes"].setdefault(
            handle.technique,
            {
                "executor_id": handle.executor_id,
                "attempt": handle.attempt,
                "failure_signature": signature,
                "feedback": feedback,
                "classification": "executor_protocol_deferred",
            },
        )
        _atomic_json(path, payload)
        manifest_path = self.campaign_dir / "search" / str(epoch) / "EXECUTORS.json"
        manifest = _read_object(manifest_path)
        if manifest.get("active_technique") == handle.technique:
            manifest["active_technique"] = None
            _atomic_json(manifest_path, manifest)

    def _resume_or_exhaust(
        self,
        handle: ExecutorHandle,
        feedback: str,
        *,
        epoch: int,
    ) -> StepResult:
        budget = self.registry[handle.technique].round_budget
        rounds = self._technique_rounds(handle.technique)
        if rounds >= budget:
            self._write_budget_disposition(handle.technique, rounds)
            return StepResult(
                None,
                payload={
                    "reason": "technique_round_budget_exhausted_lane_continues",
                    "technique": handle.technique,
                    "attempt": handle.attempt,
                    "rounds": rounds,
                },
            )
        signature = hashlib.sha256(
            f"{handle.technique}\0{feedback}".encode()
        ).hexdigest()
        resume_key = (
            f"{self.campaign_id}:executor-resume:{handle.executor_id}:"
            f"{handle.attempt + 1}:{signature}"
        )
        resume_already_recorded = any(
            event["idempotency_key"] == resume_key
            for event in self.store.events(self.campaign_id)
        )
        feedback_sha256 = hashlib.sha256(feedback.encode()).hexdigest()
        same_signature_resumes = sum(
            1
            for event in self.store.events(
                self.campaign_id,
                event_type="executor_resumed",
            )
            if event["payload"].get("executor_id") == handle.executor_id
            and event["payload"].get("feedback_sha256") == feedback_sha256
        )
        if (
            (same_signature_resumes >= 2 or handle.attempt >= 6)
            and not resume_already_recorded
        ):
            self._defer_lane(
                epoch,
                handle,
                signature=signature,
                feedback=feedback,
            )
            next_technique = self._ensure_next_executor(epoch)
            return StepResult(
                None,
                payload={
                    "reason": "executor_lane_deferred",
                    "technique": handle.technique,
                    "attempt": handle.attempt,
                    "failure_signature": signature,
                    "next_technique": next_technique,
                },
            )
        resumed = self.executors.resume(
            handle,
            feedback=feedback,
            idempotency_key=resume_key,
        )
        if not self.store.has_failure(signature):
            self.store.record_failure(
                self.campaign_id,
                handle.technique,
                signature,
                {"feedback": feedback, "attempt": handle.attempt},
            )
        return StepResult(
            None,
            payload={
                "reason": "executor_resumed",
                "technique": handle.technique,
                "attempt": resumed.attempt,
                "failure_signature": signature,
            },
        )

    def _technique_rounds(self, technique: str) -> int:
        """Count authenticated full-workload measurements, never processes."""
        return sum(
            1
            for event in self.store.events(
                self.campaign_id, event_type="scientific_round_completed"
            )
            if event["payload"].get("technique") == technique
        )

    def _has_search_budget(self, epoch: int) -> bool:
        del epoch
        dispositions = self._load_dispositions()
        return any(
            name not in dispositions
            and
            self._technique_rounds(name) < self.registry[name].round_budget
            for name in self._routes()
        )

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
