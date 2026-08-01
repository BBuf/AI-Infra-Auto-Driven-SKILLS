from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import ConfigDict, Field, model_validator

from .driver import Activation, BenchmarkRun, SGLangDiffusionDriver
from .models import (
    BaselineRecord,
    CampaignGoal,
    CorrectnessMode,
    FrontierPoint,
    IntegratedDelivery,
    PerformanceRecord,
    QualityRecord,
    SourceLock,
    StrictModel,
)
from .process import run
from .request import FrozenBenchmarkCommand
from .sources import SourceManager


_TECHNIQUE_ORDER = {
    "topology": 0,
    "residency": 1,
    "kernel": 2,
    "cache": 3,
    "pisa": 4,
    "quantization": 5,
    "token_pruning": 6,
}


class IntegrationError(RuntimeError):
    """Raised when a recipe cannot be composed or independently verified."""


class CandidateActivation(StrictModel):
    env: dict[str, str] = Field(default_factory=dict)
    server_args: list[str] = Field(default_factory=list)


class VerifiedCandidate(StrictModel):
    """The narrow, durable record handed from verification to integration."""

    schema_version: Literal[1] = 1
    candidate_id: str = Field(min_length=1)
    technique: str = Field(min_length=1)
    base_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    candidate_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    correctness: CorrectnessMode
    activation: CandidateActivation = Field(default_factory=CandidateActivation)
    source_hashes: dict[str, str] = Field(min_length=1)
    compatibility_notes: list[str] = Field(default_factory=list)
    verified_speedup: float | None = Field(default=None, gt=0)
    verified: bool

    @model_validator(mode="after")
    def require_source_hash_values(self) -> VerifiedCandidate:
        if any(not name or not digest for name, digest in self.source_hashes.items()):
            raise ValueError("source_hashes must contain nonempty names and values")
        return self


class IntegrationRecipe(StrictModel):
    """A deterministic, fully expanded recipe for one integration attempt."""

    schema_version: Literal[1] = 1
    base_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    ordered_candidate_ids: list[str] = Field(min_length=1)
    commit_shas: list[str] = Field(min_length=1)
    techniques: list[str] = Field(min_length=1)
    correctness_modes: list[CorrectnessMode] = Field(min_length=1)
    activation: CandidateActivation
    source_hashes: dict[str, str] = Field(min_length=1)
    compatibility_notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_aligned_canonical_entries(self) -> IntegrationRecipe:
        lengths = {
            len(self.ordered_candidate_ids),
            len(self.commit_shas),
            len(self.techniques),
            len(self.correctness_modes),
        }
        if len(lengths) != 1:
            raise ValueError("candidate recipe lists must have identical lengths")
        if len(self.ordered_candidate_ids) != len(set(self.ordered_candidate_ids)):
            raise ValueError("ordered_candidate_ids contains duplicates")
        if any(
            not re.fullmatch(r"[0-9a-f]{40}", commit) for commit in self.commit_shas
        ):
            raise ValueError("commit_shas must contain full lowercase Git SHAs")
        ranks = [_technique_rank(name) for name in self.techniques]
        if ranks != sorted(ranks):
            raise ValueError("techniques are not in canonical integration order")
        if any(not name or not digest for name, digest in self.source_hashes.items()):
            raise ValueError("source_hashes must contain nonempty names and values")
        return self

    @property
    def correctness(self) -> CorrectnessMode:
        if CorrectnessMode.QUALITY_GATED in self.correctness_modes:
            return CorrectnessMode.QUALITY_GATED
        return CorrectnessMode.LOSSLESS


class IntegrationVerificationRequest(StrictModel):
    """Stable adapter input for the independent delivery verifier."""

    model_config = ConfigDict(
        extra="forbid",
        protected_namespaces=(),
        arbitrary_types_allowed=True,
    )

    schema_version: Literal[1] = 1
    worktree: Path
    run_dir: Path
    benchmark_file: Path
    normalized_file: Path
    media_dir: Path
    command_receipt: Path
    baseline: BaselineRecord
    recipe: IntegrationRecipe
    correctness: CorrectnessMode
    performance: PerformanceRecord


class IntegrationVerificationOutcome(StrictModel):
    accepted: bool
    issues: list[str] = Field(default_factory=list)
    quality: QualityRecord | None = None
    artifacts: list[Path] = Field(default_factory=list)
    implementation_manifest: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def accepted_outcome_has_evidence(self) -> IntegrationVerificationOutcome:
        if self.accepted and self.quality is None:
            raise ValueError(
                "accepted integration verification requires quality evidence"
            )
        return self


class DeliveryVerifierAdapter(Protocol):
    """The only DeliveryVerifier surface the integrator depends on."""

    def verify_integrated(
        self, request: IntegrationVerificationRequest
    ) -> IntegrationVerificationOutcome: ...


class IntegrationResult(StrictModel):
    status: Literal[
        "ready_for_verification",
        "integrated",
        "needs_executor_revision",
    ]
    recipe: IntegrationRecipe
    worktree: Path
    integration_commit: str | None = None
    failed_candidate_id: str | None = None
    diagnostics_path: Path | None = None
    delivery_path: Path | None = None


def _technique_rank(technique: str) -> tuple[int, str]:
    """Put extension lanes last while keeping their order deterministic."""

    if technique in _TECHNIQUE_ORDER:
        return (_TECHNIQUE_ORDER[technique], "")
    return (len(_TECHNIQUE_ORDER), technique)


class IntegrationManager:
    """Compose verified commits and rerun the complete frozen workload."""

    def __init__(
        self,
        source_manager: SourceManager,
        source_lock: SourceLock,
        verifier: DeliveryVerifierAdapter,
        *,
        driver_type: type[SGLangDiffusionDriver] = SGLangDiffusionDriver,
        command_template: FrozenBenchmarkCommand | None = None,
    ) -> None:
        self.source_manager = source_manager
        self.source_lock = source_lock
        self.verifier = verifier
        self.driver_type = driver_type
        self.command_template = command_template

    def build_recipe(
        self,
        selected_candidate_ids: Sequence[str],
        candidates: Mapping[str, VerifiedCandidate],
    ) -> IntegrationRecipe:
        if not selected_candidate_ids:
            raise IntegrationError("at least one candidate must be selected")
        if len(selected_candidate_ids) != len(set(selected_candidate_ids)):
            raise IntegrationError("selected candidate IDs contain duplicates")

        selected: list[tuple[int, VerifiedCandidate]] = []
        for selection_index, candidate_id in enumerate(selected_candidate_ids):
            candidate = candidates.get(candidate_id)
            if candidate is None or not candidate.verified:
                raise IntegrationError(
                    f"candidate has not passed independent verification: {candidate_id}"
                )
            if candidate.candidate_id != candidate_id:
                raise IntegrationError(
                    f"candidate registry key does not match record: {candidate_id}"
                )
            if candidate.base_commit != self.source_lock.commit:
                raise IntegrationError(
                    f"candidate {candidate_id} was built from "
                    f"{candidate.base_commit}, not locked base "
                    f"{self.source_lock.commit}"
                )
            selected.append((selection_index, candidate))

        selected.sort(
            key=lambda item: (
                _technique_rank(item[1].technique),
                item[0],
            )
        )

        environment: dict[str, str] = {}
        server_args: list[str] = []
        source_hashes: dict[str, str] = {}
        compatibility_notes: list[str] = []
        for _, candidate in selected:
            for name, value in candidate.activation.env.items():
                old_value = environment.get(name)
                if old_value is not None and old_value != value:
                    raise IntegrationError(
                        f"activation environment conflict for {name!r}: "
                        f"{old_value!r} != {value!r}"
                    )
                environment[name] = value
            server_args.extend(candidate.activation.server_args)
            for name, digest in candidate.source_hashes.items():
                old_digest = source_hashes.get(name)
                if old_digest is not None and old_digest != digest:
                    raise IntegrationError(
                        f"source hash conflict for {name!r}: "
                        f"{old_digest!r} != {digest!r}"
                    )
                source_hashes[name] = digest
            compatibility_notes.extend(
                f"{candidate.candidate_id}: {note}"
                for note in candidate.compatibility_notes
            )

        ordered = [candidate for _, candidate in selected]
        return IntegrationRecipe(
            base_commit=self.source_lock.commit,
            ordered_candidate_ids=[candidate.candidate_id for candidate in ordered],
            commit_shas=[candidate.candidate_commit for candidate in ordered],
            techniques=[candidate.technique for candidate in ordered],
            correctness_modes=[candidate.correctness for candidate in ordered],
            activation=CandidateActivation(
                env=environment,
                server_args=server_args,
            ),
            source_hashes=source_hashes,
            compatibility_notes=compatibility_notes,
        )

    def compose(
        self,
        selected_candidate_ids: Sequence[str],
        candidates: Mapping[str, VerifiedCandidate],
        destination: Path,
        *,
        diagnostics_path: Path | None = None,
    ) -> IntegrationResult:
        recipe = self.build_recipe(selected_candidate_ids, candidates)
        worktree = self.source_manager.create_worktree(self.source_lock, destination)
        detached = run(
            ["git", "symbolic-ref", "-q", "HEAD"],
            cwd=worktree,
            check=False,
        )
        if detached.returncode == 0:
            raise IntegrationError("integration worktree must use detached HEAD")

        for candidate_id, candidate_commit in zip(
            recipe.ordered_candidate_ids,
            recipe.commit_shas,
            strict=True,
        ):
            exists = run(
                ["git", "cat-file", "-e", f"{candidate_commit}^{{commit}}"],
                cwd=worktree,
                check=False,
            )
            if exists.returncode != 0:
                raise IntegrationError(
                    f"candidate commit is unavailable: {candidate_commit}"
                )
            result = run(
                ["git", "cherry-pick", candidate_commit],
                cwd=worktree,
                env={
                    "GIT_COMMITTER_NAME": "SGL Diffusion Engine",
                    "GIT_COMMITTER_EMAIL": "sgl-diffusion-engine@localhost",
                },
                check=False,
            )
            if result.returncode == 0:
                continue

            conflict_files = run(
                ["git", "diff", "--name-only", "--diff-filter=U"],
                cwd=worktree,
                check=False,
            ).stdout.splitlines()
            status = run(
                ["git", "status", "--porcelain=v1", "--untracked-files=all"],
                cwd=worktree,
                check=False,
            ).stdout
            abort = run(
                ["git", "cherry-pick", "--abort"],
                cwd=worktree,
                check=False,
            )
            diagnostics_path = diagnostics_path or destination.with_name(
                f"{destination.name}.conflict.json"
            )
            _atomic_json(
                diagnostics_path,
                {
                    "schema_version": 1,
                    "status": "needs_executor_revision",
                    "failed_candidate_id": candidate_id,
                    "failed_commit": candidate_commit,
                    "conflict_files": conflict_files,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "status_before_abort": status,
                    "abort_returncode": abort.returncode,
                    "abort_stderr": abort.stderr,
                },
            )
            return IntegrationResult(
                status="needs_executor_revision",
                recipe=recipe,
                worktree=worktree,
                failed_candidate_id=candidate_id,
                diagnostics_path=diagnostics_path,
            )

        self.source_manager.assert_clean_worktree(worktree)
        integration_commit = run(
            ["git", "rev-parse", "--verify", "HEAD^{commit}"],
            cwd=worktree,
        ).stdout.strip()
        return IntegrationResult(
            status="ready_for_verification",
            recipe=recipe,
            worktree=worktree,
            integration_commit=integration_commit,
        )

    def integrate(
        self,
        goal: CampaignGoal,
        baseline: BaselineRecord,
        selected_candidate_ids: Sequence[str],
        candidates: Mapping[str, VerifiedCandidate],
        integration_root: Path,
    ) -> IntegrationResult:
        integration_root = integration_root.resolve()
        integration_root.mkdir(parents=True, exist_ok=True)
        delivery_path = integration_root / "INTEGRATED-DELIVERY.json"
        if delivery_path.exists() or delivery_path.is_symlink():
            raise IntegrationError(
                f"integrated delivery already exists: {delivery_path}"
            )
        if baseline.sglang_commit != self.source_lock.commit:
            raise IntegrationError(
                "frozen baseline commit differs from locked integration base: "
                f"{baseline.sglang_commit} != {self.source_lock.commit}"
            )
        if baseline.model_id != goal.model.id:
            raise IntegrationError(
                "frozen baseline model differs from campaign model: "
                f"{baseline.model_id!r} != {goal.model.id!r}"
            )

        composed = self.compose(
            selected_candidate_ids,
            candidates,
            integration_root / "worktree",
            diagnostics_path=integration_root / "CONFLICT.json",
        )
        if composed.status == "needs_executor_revision":
            return composed

        if (
            self.command_template is not None
            and self.driver_type is SGLangDiffusionDriver
        ):
            driver = SGLangDiffusionDriver.from_template(
                composed.worktree, self.command_template
            )
        else:
            driver = self.driver_type(composed.worktree)
        benchmark = driver.run(
            goal,
            integration_root / "run",
            activation=Activation(
                env=composed.recipe.activation.env,
                server_args=tuple(composed.recipe.activation.server_args),
            ),
            profile=False,
        )
        performance = self._recompute_performance(baseline, benchmark)
        request = IntegrationVerificationRequest(
            worktree=composed.worktree,
            run_dir=benchmark.run_dir,
            benchmark_file=benchmark.output_file,
            normalized_file=benchmark.normalized_file,
            media_dir=benchmark.media_dir,
            command_receipt=benchmark.command_receipt,
            baseline=baseline,
            recipe=composed.recipe,
            correctness=composed.recipe.correctness,
            performance=performance,
        )
        verification = self.verifier.verify_integrated(request)
        if not verification.accepted:
            issues = "; ".join(verification.issues) or "no issue detail supplied"
            raise IntegrationError(
                f"integrated candidate failed independent verification: {issues}"
            )
        if verification.quality is None:  # defensive for non-Pydantic adapters
            raise IntegrationError(
                "independent verification did not return quality evidence"
            )
        self._validate_quality_routing(
            composed.recipe.correctness, verification.quality
        )

        implementation_manifest = {
            "integration_commit": composed.integration_commit,
            "recipe": composed.recipe.model_dump(mode="json"),
            **verification.implementation_manifest,
        }
        artifacts = list(
            dict.fromkeys(
                [
                    benchmark.output_file,
                    benchmark.normalized_file,
                    benchmark.command_receipt,
                    *verification.artifacts,
                ]
            )
        )
        delivery = IntegratedDelivery(
            schema_version=2,
            status="complete",
            component="integrator",
            model_id=goal.model.id,
            baseline=baseline.model_dump(mode="json"),
            frontier_points=[
                FrontierPoint(
                    candidate_id="integrated",
                    run_dir=benchmark.run_dir,
                    activation=composed.recipe.activation.model_dump(mode="json"),
                    implementation_manifest=implementation_manifest,
                    performance=performance,
                    quality=verification.quality,
                    artifacts=artifacts,
                )
            ],
            pareto_assessment=(
                "Independently verified full frozen-workload integration."
            ),
        )
        _write_exclusive_model(delivery_path, delivery)
        return IntegrationResult(
            status="integrated",
            recipe=composed.recipe,
            worktree=composed.worktree,
            integration_commit=composed.integration_commit,
            delivery_path=delivery_path,
        )

    @staticmethod
    def _recompute_performance(
        baseline: BaselineRecord, benchmark: BenchmarkRun
    ) -> PerformanceRecord:
        timing_scope = benchmark.normalized.get("timing_scope")
        if timing_scope != baseline.timing_scope:
            raise IntegrationError(
                "integrated benchmark timing scope differs from frozen baseline: "
                f"{timing_scope!r} != {baseline.timing_scope!r}"
            )
        candidate_mean_e2e_s = float(benchmark.normalized["mean_e2e_s"])
        candidate_workload_total_s = float(benchmark.normalized["workload_total_s"])
        request_count = int(benchmark.normalized["request_count"])
        if not math.isfinite(candidate_mean_e2e_s) or candidate_mean_e2e_s <= 0:
            raise IntegrationError(
                "integrated benchmark latency must be finite and positive"
            )
        return PerformanceRecord(
            frontier_axis="latency",
            baseline_mean_e2e_s=baseline.mean_e2e_s,
            candidate_mean_e2e_s=candidate_mean_e2e_s,
            baseline_workload_total_s=baseline.workload_total_s,
            candidate_workload_total_s=candidate_workload_total_s,
            request_count=request_count,
            speedup=baseline.mean_e2e_s / candidate_mean_e2e_s,
        )

    @staticmethod
    def _validate_quality_routing(
        correctness: CorrectnessMode, quality: QualityRecord
    ) -> None:
        if correctness is CorrectnessMode.LOSSLESS:
            if (
                quality.mode != "not_gated"
                or quality.lpips_max is not None
                or quality.lpips_mean is not None
            ):
                raise IntegrationError(
                    "lossless integration must not contain LPIPS quality-gate "
                    "evidence"
                )
            return
        if (
            quality.mode != "quality_gated"
            or quality.lpips_max is None
            or quality.lpips_mean is None
            or quality.visual_overall != "pass"
        ):
            raise IntegrationError(
                "quality-gated integration requires LPIPS evidence and a "
                "passing built-in visual verdict"
            )


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_exclusive_model(path: Path, model: StrictModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    model.model_dump(mode="json"),
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
    except FileExistsError as error:
        raise IntegrationError(f"refusing to replace delivery: {path}") from error
