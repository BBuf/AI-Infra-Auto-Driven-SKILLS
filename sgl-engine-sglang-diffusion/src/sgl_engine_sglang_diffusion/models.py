from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class CorrectnessMode(StrEnum):
    LOSSLESS = "lossless"
    QUALITY_GATED = "quality_gated"


class CampaignStatus(StrEnum):
    NEW = "NEW"
    BASELINE_LOCKED = "BASELINE_LOCKED"
    PROFILED = "PROFILED"
    AWAITING_AGENT = "AWAITING_AGENT"
    SEARCHING = "SEARCHING"
    INTEGRATING = "INTEGRATING"
    FINAL_VERIFYING = "FINAL_VERIFYING"
    TARGET_REACHED = "TARGET_REACHED"
    UNREACHABLE_CERTIFIED = "UNREACHABLE_CERTIFIED"
    SEARCH_SPACE_EXHAUSTED = "SEARCH_SPACE_EXHAUSTED"
    WAITING_RESOURCE = "WAITING_RESOURCE"
    INFRA_BLOCKED = "INFRA_BLOCKED"
    PAUSED_BUDGET = "PAUSED_BUDGET"
    CANCELLED = "CANCELLED"


class ModelSpec(StrictModel):
    id: str = Field(min_length=1)


class HardwareSpec(StrictModel):
    environment: str = Field(min_length=1)
    gpu_count: int = Field(ge=1)


class WorkloadSpec(StrictModel):
    prompts: Path
    prompt_count: int = Field(ge=5)
    seed: int
    height: int = Field(gt=0)
    width: int = Field(gt=0)
    frames: int = Field(gt=0)
    fps: int = Field(gt=0)
    steps: int = Field(gt=0)
    guidance: float = Field(ge=0)
    dtype: str
    timing_scope: str


class GoalTarget(StrictModel):
    target_speedup: float = Field(gt=1.0)
    allow_quality_gated: bool


class SourceSpec(StrictModel):
    sglang_repo: str
    sglang_ref: str = "main"
    sol_engine_repo: str = "https://github.com/NVlabs/Sana.git"
    sol_engine_ref: str = "cee25847afdd34bc656abcca126262200b088dc8"
    fastvideo_repo: str = "https://github.com/hao-ai-lab/FastVideo.git"
    fastvideo_ref: str = "main"
    kda_pilot_repo: str = "https://github.com/BBuf/KDA-Pilot.git"
    kda_pilot_ref: str = "main"


class AgentSpec(StrictModel):
    command: list[str] = Field(min_length=1)
    model: str | None = None


class CampaignGoal(StrictModel):
    schema_version: Literal[2] = 2
    execution_mode: Literal["interactive_single_agent"] = "interactive_single_agent"
    model: ModelSpec
    hardware: HardwareSpec
    workload: WorkloadSpec
    goal: GoalTarget
    source: SourceSpec

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_agent_goal(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        migrated = dict(value)
        if migrated.get("schema_version", 1) == 1:
            migrated["schema_version"] = 2
        migrated.pop("agent", None)
        migrated.setdefault("execution_mode", "interactive_single_agent")
        return migrated

    @field_validator("workload")
    @classmethod
    def require_five_prompt_contract(cls, value: WorkloadSpec) -> WorkloadSpec:
        if value.prompt_count != 5:
            raise ValueError("prompt_count must be exactly 5 for Sol-Engine parity")
        return value


class SourceLock(StrictModel):
    name: str
    repository: str
    requested_ref: str
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")


class BaselineRecord(StrictModel):
    schema_version: Literal[1] = 1
    model_id: str
    total_s: float = Field(gt=0)
    peak_memory_mib: float = Field(gt=0)
    timing_scope: str
    run_dir: Path
    baseline_frames: Path
    sglang_commit: str = Field(pattern=r"^[0-9a-f]{40}$")


class PerformanceRecord(StrictModel):
    frontier_axis: Literal["latency", "peak_memory"]
    baseline_total_s: float = Field(gt=0)
    candidate_total_s: float = Field(gt=0)
    speedup: float = Field(gt=0)


class QualityRecord(StrictModel):
    mode: Literal["quality_gated", "not_gated"]
    lpips_max: float | None
    lpips_mean: float | None
    visual_overall: Literal["pass", "fail", "authenticity_only"]
    visual_verdict: Path
    relation: Literal["equivalent", "better", "worse", "not_applicable"]


class FrontierPoint(StrictModel):
    candidate_id: str
    run_dir: Path
    activation: dict[str, Any]
    implementation_manifest: dict[str, Any]
    performance: PerformanceRecord
    quality: QualityRecord
    artifacts: list[Path]


class Delivery(StrictModel):
    schema_version: Literal[2]
    status: Literal["complete"]
    component: str
    model_id: str
    baseline: dict[str, Any]
    frontier_points: list[FrontierPoint]
    pareto_assessment: str


class TechniqueContract(StrictModel):
    schema_version: Literal[1] = 1
    name: str
    workflow_uid: str
    correctness: CorrectnessMode
    round_budget: int = Field(gt=0)
    scope_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class AgentWorkOrder(StrictModel):
    schema_version: Literal[1] = 1
    campaign_id: str = Field(min_length=1)
    epoch: int = Field(ge=1)
    technique: str = Field(min_length=1)
    correctness: CorrectnessMode
    worktree: Path
    delivery_path: Path
    review_path: Path
    baseline_path: Path
    profile_path: Path
    technique_scope: Path
    knowledge_manifest_path: Path
    search_space_path: Path
    source_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    baseline_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    profile_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    technique_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    knowledge_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    search_space_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    scientific_rounds_used: int = Field(ge=0)
    scientific_rounds_remaining: int = Field(ge=0)


class TechniqueDisposition(StrictModel):
    schema_version: Literal[1] = 1
    technique: str = Field(min_length=1)
    classification: Literal["unsupported", "no_gain", "blocked"]
    reason: str = Field(min_length=1)
    closed: bool


class KnowledgeOrigin(StrictModel):
    source: str = Field(min_length=1)
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class CandidateManifest(StrictModel):
    schema_version: Literal[1] = 1
    candidate_id: str
    technique: str
    kind: Literal["patch", "control"]
    base_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    candidate_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    activation: dict[str, Any]
    eval_profile: dict[str, Any]
    knowledge_origin: list[KnowledgeOrigin] = Field(min_length=1)


class IntegratedDelivery(Delivery):
    component: Literal["integrator"]


class DerivedCheckpoint(StrictModel):
    uri: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class AgentProfile(StrictModel):
    schema_version: Literal[1] = 1
    profile_id: str
    campaign_id: str
    model_ids: list[str] = Field(min_length=1)
    sglang_base_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    hardware: dict[str, Any]
    workload: dict[str, Any]
    techniques: dict[str, Any]
    server_args: dict[str, Any]
    fallback_policy: Literal["native", "error"]
    source_hashes: dict[str, str]
    integrated_delivery_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    speedup: float = Field(gt=1)
    derived_checkpoint: DerivedCheckpoint | None = None


class EngagementReceipt(StrictModel):
    schema_version: Literal[1] = 1
    profile_id: str
    model_match: bool
    hardware_match: bool
    workload_match: bool
    techniques: dict[str, dict[str, int | bool | str]]
    source_hashes: dict[str, str]


class ProfileDigest(StrictModel):
    schema_version: Literal[1] = 1
    run_dir: Path
    timing_scope: str
    stage_ms: dict[str, float]
    hotspots: list[dict[str, Any]]
    trace_paths: list[Path]


class UnreachableCertificate(StrictModel):
    schema_version: Literal[1] = 1
    frozen_workload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    hardware: dict[str, Any]
    allowed_methods: list[str]
    target_latency_s: float = Field(gt=0)
    lower_bound_s: float = Field(gt=0)
    derivation: list[dict[str, Any]] = Field(min_length=1)
    source_evidence: list[str] = Field(min_length=1)
