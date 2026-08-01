from __future__ import annotations

import math
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
    schema_version: Literal[1]
    model: ModelSpec
    hardware: HardwareSpec
    workload: WorkloadSpec
    goal: GoalTarget
    source: SourceSpec
    agent: AgentSpec

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
    schema_version: Literal[2] = 2
    model_id: str
    mean_e2e_s: float = Field(gt=0)
    workload_total_s: float = Field(gt=0)
    request_count: Literal[5] = 5
    peak_memory_mib: float = Field(gt=0)
    timing_scope: str
    run_dir: Path
    baseline_frames: Path
    sglang_commit: str = Field(pattern=r"^[0-9a-f]{40}$")

    @model_validator(mode="after")
    def require_consistent_timing(self) -> BaselineRecord:
        expected_total = self.mean_e2e_s * self.request_count
        if not math.isclose(
            self.workload_total_s, expected_total, rel_tol=1e-9, abs_tol=1e-9
        ):
            raise ValueError("workload_total_s must equal mean_e2e_s * request_count")
        return self


class PerformanceRecord(StrictModel):
    schema_version: Literal[2] = 2
    frontier_axis: Literal["latency", "peak_memory"]
    baseline_mean_e2e_s: float = Field(gt=0)
    candidate_mean_e2e_s: float = Field(gt=0)
    baseline_workload_total_s: float = Field(gt=0)
    candidate_workload_total_s: float = Field(gt=0)
    request_count: Literal[5] = 5
    speedup: float = Field(gt=0)

    @model_validator(mode="after")
    def require_consistent_timing(self) -> PerformanceRecord:
        pairs = (
            (self.baseline_workload_total_s, self.baseline_mean_e2e_s),
            (self.candidate_workload_total_s, self.candidate_mean_e2e_s),
        )
        for workload_total, mean_e2e in pairs:
            if not math.isclose(
                workload_total,
                mean_e2e * self.request_count,
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                raise ValueError("workload total must equal mean_e2e_s * request_count")
        return self


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


class CoverageItem(StrictModel):
    id: str = Field(min_length=1)
    status: Literal["measured", "inapplicable", "blocked"]
    evidence: list[Path] = Field(min_length=1)
    scientific_round_ids: list[str] = Field(default_factory=list)


class TechniqueDisposition(StrictModel):
    schema_version: Literal[1] = 1
    technique: str = Field(min_length=1)
    classification: Literal["unsupported", "no_gain", "blocked", "budget_exhausted"]
    reason: str = Field(min_length=1)
    coverage: list[CoverageItem] = Field(min_length=1)
    profile_digest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class EvidenceFile(StrictModel):
    path: Path
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class KernelWikiEvidence(StrictModel):
    queries: list[str] = Field(min_length=1)
    sources: list[EvidenceFile] = Field(min_length=1)


class NcuEvidence(StrictModel):
    applicable: bool
    reason: str = Field(min_length=1)
    before_report: EvidenceFile | None = None
    after_report: EvidenceFile | None = None
    metrics_digest: EvidenceFile | None = None

    @model_validator(mode="after")
    def require_reports_when_applicable(self) -> NcuEvidence:
        if self.applicable and not all(
            (self.before_report, self.after_report, self.metrics_digest)
        ):
            raise ValueError(
                "applicable NCU evidence requires before, after, and metrics reports"
            )
        return self


class WarpSpecializationEvidence(StrictModel):
    applicable: bool
    reason: str = Field(min_length=1)
    timeline_report: EvidenceFile | None = None
    reconciliation: EvidenceFile | None = None

    @model_validator(mode="after")
    def require_timeline_when_applicable(self) -> WarpSpecializationEvidence:
        if self.applicable and not all((self.timeline_report, self.reconciliation)):
            raise ValueError(
                "warp-specialized candidates require timeline and reconciliation"
            )
        return self


class KernelEvidence(StrictModel):
    schema_version: Literal[1] = 1
    candidate_id: str = Field(min_length=1)
    candidate_family: str = Field(min_length=1)
    implementation_kind: Literal[
        "layout_only",
        "torch_compile",
        "triton",
        "cuda_cute",
        "upstream_reuse",
    ]
    hotspot: str = Field(min_length=1)
    profile_digest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    kernelwiki: KernelWikiEvidence
    ncu: NcuEvidence
    warp_specialization: WarpSpecializationEvidence
    correctness_shapes: list[str] = Field(min_length=1)
    microbenchmark: EvidenceFile

    @model_validator(mode="after")
    def require_ncu_for_implemented_kernel(self) -> KernelEvidence:
        if self.implementation_kind in {
            "triton",
            "cuda_cute",
            "upstream_reuse",
        } and not self.ncu.applicable:
            raise ValueError(
                "implemented kernel candidates require before/after NCU evidence"
            )
        if self.warp_specialization.applicable and self.implementation_kind != "cuda_cute":
            raise ValueError(
                "warp-specialization evidence applies only to CUDA/CuTe candidates"
            )
        return self


class ResidencyMemorySnapshot(StrictModel):
    gpu_uuid: str = Field(min_length=1)
    total_mib: float = Field(gt=0)
    min_free_before_mib: float = Field(ge=0)
    peak_reserved_mib: float = Field(ge=0)
    safety_margin_mib: float = Field(gt=0)

    @model_validator(mode="after")
    def require_measured_headroom(self) -> ResidencyMemorySnapshot:
        if self.min_free_before_mib > self.total_mib:
            raise ValueError("residency minimum-free memory exceeds total GPU memory")
        if (
            self.peak_reserved_mib + self.safety_margin_mib
            > self.min_free_before_mib
        ):
            raise ValueError(
                "residency peak plus safety margin exceeds measured free GPU memory"
            )
        return self


class ResidencyTransferMeasurement(StrictModel):
    h2d_copy_count: int = Field(ge=0)
    h2d_total_ms: float = Field(ge=0)


class ResidencyComponentPlacement(StrictModel):
    component: str = Field(min_length=1)
    baseline_strategy: Literal[
        "resident",
        "component_cpu_offload",
        "layerwise_offload",
        "partial_dit_resident",
        "fsdp",
        "compile_transient_offload",
    ]
    candidate_strategy: Literal[
        "resident",
        "component_cpu_offload",
        "layerwise_offload",
        "partial_dit_resident",
        "fsdp",
        "compile_transient_offload",
    ]
    footprint_mib: float = Field(gt=0)
    total_layers: int | None = Field(default=None, gt=0)
    resident_layers: int | None = Field(default=None, gt=0)
    prefetch_depth: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def require_strategy_specific_counts(self) -> ResidencyComponentPlacement:
        if self.candidate_strategy == "partial_dit_resident":
            if self.total_layers is None or self.resident_layers is None:
                raise ValueError("partial DiT residency requires layer counts")
            if self.resident_layers >= self.total_layers:
                raise ValueError("partial DiT resident layers must be below total layers")
        elif self.resident_layers is not None:
            raise ValueError("resident_layers is valid only for partial DiT residency")
        if self.prefetch_depth is not None and self.candidate_strategy not in {
            "layerwise_offload",
            "partial_dit_resident",
        }:
            raise ValueError("prefetch_depth requires a layerwise residency strategy")
        return self


class ResidencyConflictChecks(StrictModel):
    explicit_policy_declared: Literal[True]
    fsdp_compatible: Literal[True]
    cache_dit_compatible: Literal[True]
    steady_state_restored: Literal[True]
    unsupported_components_absent: Literal[True]


class GpuInventoryDevice(StrictModel):
    physical_index: int = Field(ge=0)
    uuid: str = Field(min_length=1)
    total_mib: float = Field(gt=0)


class GpuInventory(StrictModel):
    """Controller-owned identity of the GPUs selected by the frozen baseline."""

    schema_version: Literal[1] = 1
    gpu_count: int = Field(gt=0)
    visibility_source: Literal[
        "frozen_command_env",
        "controller_env",
        "default_order",
    ]
    visible_device_tokens: list[str] | None = None
    baseline_command_template_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    devices: list[GpuInventoryDevice] = Field(min_length=1)

    @model_validator(mode="after")
    def require_exact_unique_device_set(self) -> GpuInventory:
        if len(self.devices) != self.gpu_count:
            raise ValueError("GPU inventory device count differs from frozen gpu_count")
        if len({item.uuid for item in self.devices}) != len(self.devices):
            raise ValueError("GPU inventory contains duplicate UUIDs")
        if len({item.physical_index for item in self.devices}) != len(self.devices):
            raise ValueError("GPU inventory contains duplicate physical indices")
        if self.visible_device_tokens is not None and not self.visible_device_tokens:
            raise ValueError("visible_device_tokens must be nonempty when declared")
        return self


class ResidencyEvidence(StrictModel):
    schema_version: Literal[1] = 1
    candidate_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    coverage_id: str = Field(min_length=1)
    profile_digest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    gpu_uuids: list[str] = Field(min_length=1)
    baseline_memory: list[ResidencyMemorySnapshot] = Field(min_length=1)
    candidate_memory: list[ResidencyMemorySnapshot] = Field(min_length=1)
    baseline_transfers: ResidencyTransferMeasurement
    candidate_transfers: ResidencyTransferMeasurement
    components: list[ResidencyComponentPlacement] = Field(min_length=1)
    compile_strategy: str = Field(min_length=1)
    steady_state_strategy: str = Field(min_length=1)
    conflicts: ResidencyConflictChecks
    engagement_counts: dict[str, int] = Field(min_length=1)
    performance_artifact: EvidenceFile
    equivalence_artifact: EvidenceFile

    @model_validator(mode="after")
    def require_frozen_gpus_and_engagement(self) -> ResidencyEvidence:
        if len(self.gpu_uuids) != len(set(self.gpu_uuids)):
            raise ValueError("residency gpu_uuids contains duplicates")
        expected = set(self.gpu_uuids)
        baseline = {item.gpu_uuid for item in self.baseline_memory}
        candidate = {item.gpu_uuid for item in self.candidate_memory}
        if (
            baseline != expected
            or candidate != expected
            or len(self.baseline_memory) != len(expected)
            or len(self.candidate_memory) != len(expected)
        ):
            raise ValueError("residency memory snapshots drift from frozen GPU UUIDs")
        baseline_by_gpu = {item.gpu_uuid: item for item in self.baseline_memory}
        candidate_by_gpu = {item.gpu_uuid: item for item in self.candidate_memory}
        if any(
            baseline_by_gpu[gpu_uuid].total_mib
            != candidate_by_gpu[gpu_uuid].total_mib
            for gpu_uuid in expected
        ):
            raise ValueError("residency GPU total memory changed between measurements")
        if any(value < 0 for value in self.engagement_counts.values()) or not any(
            value > 0 for value in self.engagement_counts.values()
        ):
            raise ValueError("residency evidence requires positive engagement")
        return self


class VBenchDimensions(StrictModel):
    subject_consistency: float
    background_consistency: float
    motion_smoothness: float
    temporal_flickering: float
    aesthetic_quality: float
    imaging_quality: float


class AudioQualityEvidence(StrictModel):
    present: bool
    duration_s: float = Field(ge=0)
    sample_rate: int = Field(ge=0)
    channels: int = Field(ge=0)
    silence_ratio: float = Field(ge=0, le=1)
    clipping_ratio: float = Field(ge=0, le=1)


class MediaContractEvidence(StrictModel):
    container: str = Field(min_length=1)
    video_codec: str = Field(min_length=1)
    audio_codec: str | None = None
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: float = Field(gt=0)
    frame_count: int = Field(gt=0)
    video_duration_s: float = Field(gt=0)


class FinalPromptQuality(StrictModel):
    prompt_index: int = Field(ge=0, lt=5)
    lpips: float = Field(ge=0)
    vbench_baseline: VBenchDimensions
    vbench_candidate: VBenchDimensions
    audio: AudioQualityEvidence
    av_sync_drift_ms: float | None
    media: MediaContractEvidence
    visual: Literal["pass", "fail"]


class FinalQualityThresholds(StrictModel):
    lpips_max: float = Field(gt=0)
    vbench_max_mean_regression: float = Field(default=0, ge=0)
    silence_ratio_max: float = Field(default=0.98, ge=0, le=1)
    clipping_ratio_max: float = Field(default=0.01, ge=0, le=1)
    av_sync_drift_ms_max: float = Field(default=80, gt=0)


class FinalQualityEvidence(StrictModel):
    schema_version: Literal[1] = 1
    producer: Literal["independent-master"]
    external_api: Literal[False]
    integrated_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    audio_required: bool
    thresholds: FinalQualityThresholds
    prompts: list[FinalPromptQuality] = Field(min_length=5, max_length=5)
    command_receipts: list[EvidenceFile] = Field(min_length=4)

    @model_validator(mode="after")
    def require_exact_prompt_indices(self) -> FinalQualityEvidence:
        if {item.prompt_index for item in self.prompts} != set(range(5)):
            raise ValueError("final quality evidence requires prompt indices 0..4")
        return self


class CandidateManifest(StrictModel):
    schema_version: Literal[1] = 1
    candidate_id: str
    technique: str
    kind: Literal["patch", "control"]
    base_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    candidate_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    activation: dict[str, Any]
    eval_profile: dict[str, Any]
    knowledge_origin: list[dict[str, str]]


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
    trace_sha256: dict[str, str] = Field(min_length=1)
    parser_version: Literal["chrome-trace-v1"]
    event_count: int = Field(gt=0)


class ProfileTraceArtifact(StrictModel):
    path: Path
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ProfileInventory(StrictModel):
    schema_version: Literal[1] = 1
    parser_version: Literal["chrome-trace-v1"]
    event_count: int = Field(gt=0)
    traces: list[ProfileTraceArtifact] = Field(min_length=1)


class UnreachableCertificate(StrictModel):
    schema_version: Literal[1] = 1
    frozen_workload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    hardware: dict[str, Any]
    allowed_methods: list[str]
    target_latency_s: float = Field(gt=0)
    lower_bound_s: float = Field(gt=0)
    derivation: list[dict[str, Any]] = Field(min_length=1)
    source_evidence: list[str] = Field(min_length=1)
