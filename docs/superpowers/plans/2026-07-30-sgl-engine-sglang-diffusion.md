# SGL-Engine for SGLang Diffusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an executable, persistent Sol-Engine-compatible optimization
controller that improves a locked SGLang Diffusion checkout and emits a
clean-room-verified `sglang.patch`.

**Architecture:** A deterministic Python control plane owns source locks,
SQLite state, leases, artifact schemas, objective verification, integration,
and patch packaging. Coding agents work in isolated SGLang worktrees and use
Sol-Engine contracts plus locked SGLang, KDA-Pilot, and FastVideo knowledge.
The master independently verifies every delivery before an integration
worktree can advance.

**Tech Stack:** Python 3.11+, Pydantic 2, PyYAML, stdlib `sqlite3`,
`subprocess`, Git worktrees, pytest, TOML registries, JSON/JSONL artifacts,
Sol-Engine `plan_eval.py`, SGLang Diffusion benchmark/profile tools.

---

## Scope and sequencing

This is one pull request but four independently testable milestones:

1. durable control-plane foundation;
2. Sol-Engine contracts and knowledge indexing;
3. executor, verifier, integrator, and patch pipeline;
4. CLI, mocked end-to-end validation, and documentation.

Do not start GPU optimization during implementation. The pull request must
include the real-GPU validation command and artifact contract, while CI uses
fake repositories and fake runners.

## File map

Create:

```text
sgl-engine-sglang-diffusion/
├── .gitignore
├── README.md
├── pyproject.toml
├── examples/goal.yaml
├── examples/validation-prompts.txt
├── contracts/sol_engine/
│   ├── source-lock.json
│   ├── loop-and-gate.md
│   └── master.md
├── knowledge/registry.toml
├── techniques/registry.toml
├── techniques/{kernel,cache,pisa,topology,quantization,token_pruning}.md
├── prompts/executor.md
├── prompts/master.md
├── schemas/
├── src/sgl_engine_sglang_diffusion/
│   ├── __init__.py
│   ├── agents.py
│   ├── artifacts.py
│   ├── baseline.py
│   ├── cli.py
│   ├── config.py
│   ├── controller.py
│   ├── driver.py
│   ├── integrator.py
│   ├── knowledge.py
│   ├── models.py
│   ├── orchestration.py
│   ├── patcher.py
│   ├── process.py
│   ├── sources.py
│   ├── state.py
│   ├── techniques.py
│   ├── verifier.py
│   └── watchdog.py
├── tools/
│   ├── sync_optimization_knowledge.py
│   └── sync_sol_engine_contracts.py
└── tests/
    ├── conftest.py
    ├── helpers.py
    ├── test_artifacts.py
    ├── test_cli.py
    ├── test_config.py
    ├── test_controller.py
    ├── test_driver.py
    ├── test_integration_flow.py
    ├── test_knowledge.py
    ├── test_orchestration.py
    ├── test_patcher.py
    ├── test_profiler.py
    ├── test_sources.py
    ├── test_state.py
    ├── test_techniques.py
    └── test_verifier.py
```

Modify:

```text
README.md
tests/test_repository_metadata.py
```

## Task 1: Package scaffold and test entry point

**Files:**

- Create: `sgl-engine-sglang-diffusion/pyproject.toml`
- Create: `sgl-engine-sglang-diffusion/.gitignore`
- Create: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/__init__.py`
- Create: `sgl-engine-sglang-diffusion/tests/conftest.py`
- Create: `sgl-engine-sglang-diffusion/tests/test_config.py`

- [ ] **Step 1: Write the package smoke test**

```python
from sgl_engine_sglang_diffusion import __version__


def test_package_version() -> None:
    assert __version__ == "0.1.0"
```

- [ ] **Step 2: Add the source-layout test bootstrap**

```python
from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PACKAGE_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))
```

- [ ] **Step 3: Run the test and verify the package is missing**

Run:

```bash
python3 -m pytest sgl-engine-sglang-diffusion/tests/test_config.py -q
```

Expected: collection fails because `sgl_engine_sglang_diffusion` does not
exist.

- [ ] **Step 4: Add package metadata**

```toml
[build-system]
requires = ["setuptools>=75", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "sgl-engine-sglang-diffusion"
version = "0.1.0"
description = "Sol-Engine-compatible optimization controller for SGLang Diffusion"
requires-python = ">=3.11"
dependencies = [
  "pydantic>=2.7,<3",
  "PyYAML>=6,<7",
]

[project.optional-dependencies]
dev = ["pytest>=8,<10"]

[project.scripts]
sgl-diffusion-engine = "sgl_engine_sglang_diffusion.cli:main"

[tool.setuptools.packages.find]
where = ["src"]
```

```python
"""SGL-Engine for SGLang Diffusion."""

__version__ = "0.1.0"
```

```gitignore
runs/
*.sqlite
*.sqlite-shm
*.sqlite-wal
__pycache__/
.pytest_cache/
*.egg-info/
dist/
build/
```

- [ ] **Step 5: Run the smoke test**

Run:

```bash
python3 -m pytest sgl-engine-sglang-diffusion/tests/test_config.py -q
```

Expected: `1 passed`.

- [ ] **Step 6: Commit**

```bash
git add sgl-engine-sglang-diffusion
git commit -m "feat: scaffold SGLang Diffusion optimization engine"
```

## Task 2: Typed campaign and artifact models

**Files:**

- Create: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/models.py`
- Create: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/config.py`
- Create: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/artifacts.py`
- Create: `sgl-engine-sglang-diffusion/examples/goal.yaml`
- Create: `sgl-engine-sglang-diffusion/examples/validation-prompts.txt`
- Create: `sgl-engine-sglang-diffusion/tests/test_artifacts.py`
- Modify: `sgl-engine-sglang-diffusion/tests/test_config.py`

- [ ] **Step 1: Write validation tests**

```python
from pathlib import Path

import pytest

from sgl_engine_sglang_diffusion.config import load_goal


def test_load_goal_freezes_required_workload(tmp_path: Path) -> None:
    prompt_file = tmp_path / "prompts.txt"
    prompt_file.write_text("\n".join(f"prompt {i}" for i in range(5)) + "\n")
    goal_file = tmp_path / "goal.yaml"
    goal_file.write_text(
        f"""
schema_version: 1
model:
  id: test/model
hardware:
  environment: fake-b200
  gpu_count: 1
workload:
  prompts: {prompt_file}
  prompt_count: 5
  seed: 42
  height: 64
  width: 64
  frames: 1
  fps: 24
  steps: 4
  guidance: 1.0
  dtype: bfloat16
  timing_scope: load_excluded_end_to_end
goal:
  target_speedup: 2.0
  allow_quality_gated: true
source:
  sglang_repo: https://github.com/sgl-project/sglang.git
  sglang_ref: main
agent:
  command: [codex, exec]
"""
    )
    goal = load_goal(goal_file)
    assert goal.goal.target_speedup == 2.0
    assert goal.workload.prompt_count == 5


def test_goal_rejects_fewer_than_five_prompts(tmp_path: Path) -> None:
    goal_file = tmp_path / "goal.yaml"
    goal_file.write_text(
        """
schema_version: 1
model: {id: test/model}
hardware: {environment: fake, gpu_count: 1}
workload:
  prompts: missing.txt
  prompt_count: 4
  seed: 42
  height: 64
  width: 64
  frames: 1
  fps: 24
  steps: 4
  guidance: 1.0
  dtype: bfloat16
  timing_scope: load_excluded_end_to_end
goal: {target_speedup: 2.0, allow_quality_gated: true}
source:
  sglang_repo: https://github.com/sgl-project/sglang.git
  sglang_ref: main
agent: {command: [codex, exec]}
"""
    )
    with pytest.raises(ValueError, match="prompt_count"):
        load_goal(goal_file)
```

- [ ] **Step 2: Add models**

Implement Pydantic models with exact field names:

```python
from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


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
    schema_version: Literal[1] = 1
    model_id: str
    total_s: float = Field(gt=0)
    peak_memory_mib: float = Field(gt=0)
    timing_scope: str
    run_dir: Path
    baseline_frames: Path
    sglang_commit: str


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
```

- [ ] **Step 3: Implement YAML loading and schema emission**

```python
from pathlib import Path

import yaml

from .models import CampaignGoal


def load_goal(path: Path) -> CampaignGoal:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    goal = CampaignGoal.model_validate(data)
    if not goal.workload.prompts.is_absolute():
        goal.workload.prompts = (path.parent / goal.workload.prompts).resolve()
    if not goal.workload.prompts.is_file():
        raise ValueError(f"prompt file does not exist: {goal.workload.prompts}")
    prompts = [
        line for line in goal.workload.prompts.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(prompts) < goal.workload.prompt_count:
        raise ValueError("prompt file contains fewer than five non-empty prompts")
    return goal
```

```python
import json
from pathlib import Path

from .models import (
    AgentProfile,
    BaselineRecord,
    CampaignGoal,
    CandidateManifest,
    Delivery,
    EngagementReceipt,
    IntegratedDelivery,
    ProfileDigest,
    SourceLock,
    TechniqueContract,
    UnreachableCertificate,
)


SCHEMA_MODELS = {
    "goal": CampaignGoal,
    "baseline": BaselineRecord,
    "source-lock": SourceLock,
    "technique": TechniqueContract,
    "candidate": CandidateManifest,
    "delivery": Delivery,
    "integrated-delivery": IntegratedDelivery,
    "agent-profile": AgentProfile,
    "engagement-receipt": EngagementReceipt,
    "profile-digest": ProfileDigest,
    "unreachable-certificate": UnreachableCertificate,
}


def write_schemas(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, model in SCHEMA_MODELS.items():
        target = output_dir / f"{name}.schema.json"
        target.write_text(
            json.dumps(model.model_json_schema(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
```

- [ ] **Step 4: Add a valid example goal and artifact round-trip test**

Use five concrete prompts in `examples/validation-prompts.txt` and make
`examples/goal.yaml` match the schema. Test:

```python
def test_delivery_rejects_unknown_fields() -> None:
    from pydantic import ValidationError
    from sgl_engine_sglang_diffusion.models import Delivery

    with pytest.raises(ValidationError):
        Delivery.model_validate(
            {
                "schema_version": 2,
                "status": "complete",
                "component": "kernel",
                "model_id": "test/model",
                "baseline": {},
                "frontier_points": [],
                "pareto_assessment": "empty",
                "fabricated": True,
            }
        )
```

- [ ] **Step 5: Generate checked-in schemas and run tests**

Run:

```bash
PYTHONPATH=sgl-engine-sglang-diffusion/src python3 -c \
  "from pathlib import Path; from sgl_engine_sglang_diffusion.artifacts import write_schemas; write_schemas(Path('sgl-engine-sglang-diffusion/schemas'))"
python3 -m pytest sgl-engine-sglang-diffusion/tests/test_config.py \
  sgl-engine-sglang-diffusion/tests/test_artifacts.py -q
```

Expected: all tests pass and all eleven schema files are created. The test must
assert the exact schema filename set so a later model cannot be added without a
checked-in schema.

- [ ] **Step 6: Commit**

```bash
git add sgl-engine-sglang-diffusion
git commit -m "feat: define optimization campaign artifacts"
```

## Task 3: Sol-Engine technique contracts and registry

**Files:**

- Create: `sgl-engine-sglang-diffusion/contracts/sol_engine/source-lock.json`
- Create: `sgl-engine-sglang-diffusion/contracts/sol_engine/loop-and-gate.md`
- Create: `sgl-engine-sglang-diffusion/contracts/sol_engine/master.md`
- Create: `sgl-engine-sglang-diffusion/techniques/registry.toml`
- Create: `sgl-engine-sglang-diffusion/techniques/*.md`
- Create: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/techniques.py`
- Create: `sgl-engine-sglang-diffusion/tests/test_techniques.py`

- [ ] **Step 1: Write parity tests**

```python
from pathlib import Path

from sgl_engine_sglang_diffusion.techniques import TechniqueRegistry


ROOT = Path(__file__).resolve().parents[1]


def test_registry_preserves_sol_techniques_and_modes() -> None:
    registry = TechniqueRegistry.load(ROOT / "techniques" / "registry.toml")
    assert set(registry.names()) == {
        "kernel",
        "cache",
        "pisa",
        "topology",
        "quantization",
        "token_pruning",
    }
    assert registry["kernel"].correctness == "lossless"
    assert registry["topology"].correctness == "lossless"
    for name in ("cache", "pisa", "quantization", "token_pruning"):
        assert registry[name].correctness == "quality_gated"


def test_round_budgets_match_reviewed_contract() -> None:
    registry = TechniqueRegistry.load(ROOT / "techniques" / "registry.toml")
    assert registry["kernel"].round_budget == 40
    assert registry["cache"].round_budget == 20
    assert registry["pisa"].round_budget == 20
    assert registry["topology"].round_budget == 20
```

- [ ] **Step 2: Add the registry**

```toml
schema_version = 1
default_order = [
  "kernel",
  "cache",
  "pisa",
  "quantization",
  "token_pruning",
]

[techniques.kernel]
workflow_uid = "kernel_aw"
scope = "techniques/kernel.md"
correctness = "lossless"
round_budget = 40
origin = "sol-engine-lightweight"

[techniques.cache]
workflow_uid = "cache_ca"
scope = "techniques/cache.md"
correctness = "quality_gated"
round_budget = 20
origin = "sol-engine-lightweight"

[techniques.pisa]
workflow_uid = "attention_pa"
scope = "techniques/pisa.md"
correctness = "quality_gated"
round_budget = 20
origin = "sol-engine-lightweight"

[techniques.topology]
workflow_uid = "topology_ta"
scope = "techniques/topology.md"
correctness = "lossless"
round_budget = 20
origin = "sol-engine-lightweight"
optional = true

[techniques.quantization]
workflow_uid = "quantization_sglang"
scope = "techniques/quantization.md"
correctness = "quality_gated"
round_budget = 20
origin = "sol-engine-full-adaptation"

[techniques.token_pruning]
workflow_uid = "token_pruning_sglang"
scope = "techniques/token_pruning.md"
correctness = "quality_gated"
round_budget = 20
origin = "sol-engine-full-adaptation"
```

- [ ] **Step 3: Implement registry parsing**

```python
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Technique:
    name: str
    workflow_uid: str
    scope: Path
    correctness: str
    round_budget: int
    origin: str
    optional: bool = False


class TechniqueRegistry:
    def __init__(self, entries: dict[str, Technique], default_order: list[str]):
        self._entries = entries
        self.default_order = default_order

    @classmethod
    def load(cls, path: Path) -> "TechniqueRegistry":
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        entries: dict[str, Technique] = {}
        for name, raw in data["techniques"].items():
            scope = path.parent.parent / raw["scope"]
            if not scope.is_file():
                raise ValueError(f"missing technique scope: {scope}")
            entries[name] = Technique(
                name=name,
                workflow_uid=raw["workflow_uid"],
                scope=scope,
                correctness=raw["correctness"],
                round_budget=int(raw["round_budget"]),
                origin=raw["origin"],
                optional=bool(raw.get("optional", False)),
            )
        return cls(entries, list(data["default_order"]))

    def names(self) -> list[str]:
        return list(self._entries)

    def __getitem__(self, name: str) -> Technique:
        return self._entries[name]
```

- [ ] **Step 4: Add adapted contracts**

`source-lock.json`:

```json
{
  "repository": "https://github.com/NVlabs/Sana.git",
  "branch": "sol-engine",
  "commit": "cee25847afdd34bc656abcca126262200b088dc8",
  "authoritative_paths": [
    "orchestration/prompts/loop_and_gate_contract.md",
    "orchestration/prompts/master.md",
    "orchestration/techniques.toml",
    "workflow/kernel_aw/nodes/codex_executor/kernel_scope.md",
    "workflow/cache_ca/nodes/codex_executor/cache_scope.md",
    "workflow/attention_pa/nodes/codex_executor/attention_scope.md",
    "workflow/topology_ta/nodes/codex_executor/topology_scope.md"
  ]
}
```

`loop-and-gate.md` must state, in original wording:

```markdown
# Loop and gate contract

- The baseline is frozen and never remeasured by an executor.
- One round is one hypothesis, one candidate, one real run, and one gate.
- Lossless kernel/topology candidates are judged by mathematical and
  algorithmic equivalence, unchanged global logical denoising-step and
  DiT/model-call counts, and no approximation, sparsity, step skipping,
  sub-16-bit quantization, rank reduction, or changed logical work.
- Lossless candidates are never rejected using output differences, LPIPS,
  PSNR, or visual quality.
- Quality-gated cache, PISA, quantization, and token-pruning candidates require
  aligned LPIPS, built-in multimodal visual review, real engagement, and the
  complete frozen workload.
- Every retained point names a real run directory and durable artifacts.
- The master independently recomputes performance and verifies provenance,
  authenticity, quality, and the actual method before integration.
```

Create six focused scope files from the approved design. Preserve the cache
three-family matched-time rule, PISA approximate-remainder rule, topology
preflight/equivalence artifacts, and kernel ownership boundaries. Mark
quantization/token-pruning round budgets as SGLang adaptation defaults.

- [ ] **Step 5: Add contract-content assertions**

Assert the lossless contract contains `never rejected using output
differences` and the quality contract contains all of `LPIPS`, `multimodal`,
and `engagement`. Assert the Sol source lock commit is 40 hex characters.

- [ ] **Step 6: Run tests**

Run:

```bash
python3 -m pytest sgl-engine-sglang-diffusion/tests/test_techniques.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add sgl-engine-sglang-diffusion
git commit -m "feat: add Sol-Engine technique contracts"
```

## Task 4: Process runner and immutable source worktrees

**Files:**

- Create: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/process.py`
- Create: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/sources.py`
- Create: `sgl-engine-sglang-diffusion/tests/helpers.py`
- Create: `sgl-engine-sglang-diffusion/tests/test_sources.py`

- [ ] **Step 1: Write fake-repository tests**

```python
def test_lock_source_resolves_full_commit(fake_git_repo: Path, tmp_path: Path) -> None:
    manager = SourceManager(tmp_path / "sources")
    lock = manager.lock("sglang", str(fake_git_repo), "main")
    assert len(lock.commit) == 40
    assert manager.checkout_path(lock).is_dir()


def test_create_worktree_is_clean_and_detached(
    fake_git_repo: Path, tmp_path: Path
) -> None:
    manager = SourceManager(tmp_path / "sources")
    lock = manager.lock("sglang", str(fake_git_repo), "main")
    worktree = manager.create_worktree(lock, tmp_path / "candidate")
    status = run(["git", "status", "--porcelain"], cwd=worktree)
    assert status.stdout == ""
```

`fake_git_repo` initializes a temporary repository, configures a local test
identity, commits `README.md`, and renames the branch to `main`.

- [ ] **Step 2: Implement a non-shell process runner**

```python
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class CommandResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


def run(
    argv: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str] | None = None,
    check: bool = True,
) -> CommandResult:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    completed = subprocess.run(
        list(argv),
        cwd=cwd,
        env=merged_env,
        text=True,
        capture_output=True,
        check=False,
    )
    result = CommandResult(
        tuple(argv), completed.returncode, completed.stdout, completed.stderr
    )
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {list(argv)!r}\n"
            f"{completed.stderr}"
        )
    return result
```

- [ ] **Step 3: Implement source locking**

`SourceManager.lock()` clones with `--filter=blob:none` when the cache is
missing, fetches the requested ref, resolves `FETCH_HEAD^{commit}`, and returns
`SourceLock`. `create_worktree()` validates that the destination is outside
the shared bare cache, does `git worktree add --detach <path> <commit>`, and
asserts a clean status.

Do not use a shell string. Do not delete an existing destination.

- [ ] **Step 4: Add dirty-worktree and duplicate-destination tests**

Create an untracked file in a candidate worktree and assert
`assert_clean_worktree()` raises. Create the destination before
`create_worktree()` and assert it refuses to overwrite it.

- [ ] **Step 5: Run tests and commit**

```bash
python3 -m pytest sgl-engine-sglang-diffusion/tests/test_sources.py -q
git add sgl-engine-sglang-diffusion
git commit -m "feat: lock source revisions and isolate worktrees"
```

## Task 5: Transactional campaign state, events, and leases

**Files:**

- Create: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/state.py`
- Create: `sgl-engine-sglang-diffusion/tests/test_state.py`

- [ ] **Step 1: Write state-transition tests**

Cover:

```python
def test_transition_is_idempotent(tmp_path: Path) -> None:
    store = StateStore.open(tmp_path / "state.sqlite", tmp_path / "events.jsonl")
    store.create_campaign("c1")
    store.transition("c1", CampaignStatus.BASELINE_LOCKED, idempotency_key="base")
    store.transition("c1", CampaignStatus.BASELINE_LOCKED, idempotency_key="base")
    assert store.events("c1", event_type="transition") == [
        {
            "campaign_id": "c1",
            "event_type": "transition",
            "idempotency_key": "base",
            "payload": {"status": "BASELINE_LOCKED"},
        }
    ]


def test_expired_lease_can_be_reclaimed(tmp_path: Path) -> None:
    store = StateStore.open(tmp_path / "state.sqlite", tmp_path / "events.jsonl")
    store.create_campaign("c1")
    store.acquire_lease("executor:kernel", "worker-a", ttl_seconds=0)
    store.acquire_lease("executor:kernel", "worker-b", ttl_seconds=60)
    assert store.lease_owner("executor:kernel") == "worker-b"
```

- [ ] **Step 2: Implement the SQLite schema**

Create tables:

```sql
CREATE TABLE campaigns (
  campaign_id TEXT PRIMARY KEY,
  status TEXT NOT NULL,
  epoch INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE TABLE events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  campaign_id TEXT NOT NULL,
  event_type TEXT NOT NULL,
  idempotency_key TEXT NOT NULL UNIQUE,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE leases (
  resource TEXT PRIMARY KEY,
  owner TEXT NOT NULL,
  expires_at TEXT NOT NULL
);
CREATE TABLE failures (
  signature TEXT PRIMARY KEY,
  campaign_id TEXT NOT NULL,
  technique TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
```

Use `BEGIN IMMEDIATE` for transitions and lease acquisition. Enable WAL and
foreign keys. Mirror committed events to JSONL only after the database commit.

- [ ] **Step 3: Implement allowed transitions**

Encode an explicit transition map. Terminal statuses accept no outgoing
transition. `WAITING_RESOURCE`, `INFRA_BLOCKED`, and `PAUSED_BUDGET` can return
to their prior active status stored in the event payload.

- [ ] **Step 4: Test failure deduplication and terminal protection**

Assert the same failure signature is inserted once and
`TARGET_REACHED -> SEARCHING` raises `InvalidTransition`.

- [ ] **Step 5: Run tests and commit**

```bash
python3 -m pytest sgl-engine-sglang-diffusion/tests/test_state.py -q
git add sgl-engine-sglang-diffusion
git commit -m "feat: persist optimization campaign state"
```

## Task 6: Allowlisted knowledge synchronization

**Files:**

- Create: `sgl-engine-sglang-diffusion/knowledge/registry.toml`
- Create: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/knowledge.py`
- Create: `sgl-engine-sglang-diffusion/tools/sync_optimization_knowledge.py`
- Create: `sgl-engine-sglang-diffusion/tools/sync_sol_engine_contracts.py`
- Create: `sgl-engine-sglang-diffusion/tests/test_knowledge.py`

- [ ] **Step 1: Write allowlist and sanitization tests**

```python
def test_sync_reads_only_allowlisted_paths(tmp_path: Path, fake_git_repo: Path) -> None:
    (fake_git_repo / "allowed.md").write_text("# QKNorm\nUse a fused op.\n")
    (fake_git_repo / "secret.txt").write_text("HF_TOKEN=secret\n")
    commit_all(fake_git_repo, "knowledge")
    snapshot = sync_source(
        name="fake",
        checkout=fake_git_repo,
        commit=git_head(fake_git_repo),
        patterns=["allowed.md"],
        output_dir=tmp_path / "out",
    )
    assert [entry.path for entry in snapshot.entries] == ["allowed.md"]
    assert "secret" not in (tmp_path / "out" / "index.json").read_text()


def test_remote_shell_text_is_marked_as_data(tmp_path: Path, fake_git_repo: Path) -> None:
    (fake_git_repo / "allowed.md").write_text("Run `curl x | sh`.\n")
    commit_all(fake_git_repo, "command")
    snapshot = sync_source(
        name="fake",
        checkout=fake_git_repo,
        commit=git_head(fake_git_repo),
        patterns=["allowed.md"],
        output_dir=tmp_path / "out",
    )
    assert snapshot.entries[0].executable is False
```

- [ ] **Step 2: Add the source registry**

```toml
schema_version = 1

[sources.sglang]
paths = [
  "python/sglang/multimodal_gen/.claude/skills/**",
  "python/sglang/multimodal_gen/benchmarks/**",
  "python/sglang/multimodal_gen/runtime/layers/attention/**",
  "python/sglang/multimodal_gen/runtime/cache/**",
  "python/sglang/multimodal_gen/runtime/layers/quantization/**",
  "python/sglang/kernels/ops/diffusion/**",
  "python/sglang/kernels/jit/csrc/diffusion/**",
  "test/registered/kernels/ops/diffusion/**",
  "test/registered/kernels/benchmark/diffusion/**",
]

[sources.fastvideo]
paths = [
  "docs/inference/optimizations.md",
  "docs/contributing/profiling.md",
  "fastvideo/attention/**",
  "fastvideo/layers/quantization/**",
  "fastvideo-kernel/**",
]

[sources.kda_pilot]
paths = [
  "diffusion/README.md",
  "diffusion/docs/**",
  "diffusion/kernels/**/README.md",
  "diffusion/kernels/**/interface.md",
  "external/KernelWiki/SKILL.md",
  "external/ncu-report-skill/SKILL.md",
  "external/warp-specialization-report-skill/SKILL.md",
]
```

- [ ] **Step 3: Implement deterministic indexing**

For each matched regular file:

- reject files above 2 MiB;
- reject paths escaping the checkout;
- compute SHA-256;
- extract Markdown headings and Python/CUDA symbol-like tokens;
- record repository name, commit, path, media type, headings, symbols, hash,
  and `executable=false`;
- write sorted `index.json` and copied text references under the campaign
  knowledge directory; and
- remove values matching secret-key names or absolute home/cache paths.

Use `pathlib.Path.glob()` against the locked checkout; never execute content.

- [ ] **Step 4: Implement Sol contract drift reporting**

`sync_sol_engine_contracts.py --check` reads
`contracts/sol_engine/source-lock.json`, hashes every authoritative upstream
path from the locked checkout, compares it with a checked-in
`contract-hashes.json`, and exits nonzero with a per-file report on drift.
`--update` writes hashes only; it does not replace adapted local contracts.

- [ ] **Step 5: Run tests and commit**

```bash
python3 -m pytest sgl-engine-sglang-diffusion/tests/test_knowledge.py -q
git add sgl-engine-sglang-diffusion
git commit -m "feat: index optimization knowledge with provenance"
```

## Task 7: SGLang benchmark driver, frozen baseline, profiler, and router

**Files:**

- Create: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/driver.py`
- Create: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/baseline.py`
- Create: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/profiler.py`
- Create: `sgl-engine-sglang-diffusion/tests/test_driver.py`
- Create: `sgl-engine-sglang-diffusion/tests/test_profiler.py`

- [ ] **Step 1: Write command-discovery tests**

Create a fake SGLang tree containing:

```text
python/sglang/multimodal_gen/benchmarks/bench_offline_throughput.py
python/sglang/multimodal_gen/.claude/skills/
  sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py
```

Test that the driver builds an argv list containing:

```text
--model-path test/model
--dataset vbench
--dataset-path <absolute prompt file>
--num-prompts 5
--seed 42
--width 64
--height 64
--num-frames 1
--fps 24
--num-inference-steps 4
--guidance-scale 1.0
--output-file <run>/outputs/benchmark.jsonl
```

Also assert profile mode adds `--profile` and sets
`SGLANG_DIFFUSION_TORCH_PROFILER_DIR`.

- [ ] **Step 2: Implement `SGLangDiffusionDriver`**

The driver:

- resolves the current benchmark path from the locked checkout;
- fails if the SGLang native benchmark path is missing;
- constructs argv without shell interpolation;
- accepts candidate activation as `env: dict[str, str]` and
  `server_args: list[str]`;
- always uses exactly five prompts;
- keeps baseline and candidate argv identical except declared activation and
  output paths;
- treats any log containing `falling back to diffusers backend`,
  `using diffusers backend`, or `loaded diffusers pipeline` as invalid
  evidence; and
- writes the exact argv/environment receipt before launch.

- [ ] **Step 3: Normalize benchmark output**

Read the final valid JSONL row and write:

```json
{
  "schema_version": 1,
  "total_s": 1.0,
  "peak_memory_mib": 1024.0,
  "timing_scope": "load_excluded_end_to_end",
  "raw_result": {}
}
```

Reject unsuccessful rows, nonpositive latency, or missing peak memory.

- [ ] **Step 4: Implement `BaselineRunner.freeze()`**

The method refuses an existing `BASELINE.json`, runs the driver exactly once,
requires output video/image and aligned frame directories, records source and
environment provenance, validates `BaselineRecord`, and atomically writes
`BASELINE.json`.

- [ ] **Step 5: Test no-refresh and native-backend rejection**

Call `freeze()` twice and assert the second call raises before invoking the
fake runner. Return a fake fallback log and assert no baseline file is written.

- [ ] **Step 6: Write profiler and routing tests**

```python
def make_profile_digest(tmp_path: Path) -> ProfileDigest:
    return ProfileDigest(
        run_dir=tmp_path,
        timing_scope="load_excluded_end_to_end",
        stage_ms={"denoise": 900.0, "decode": 100.0},
        hotspots=[{"name": "aten::mul", "total_ms": 120.0}],
        trace_paths=[tmp_path / "trace.json.gz"],
    )


def test_profile_routes_attention_and_glue_hotspots(tmp_path: Path) -> None:
    digest = ProfileDigest(
        run_dir=tmp_path,
        timing_scope="load_excluded_end_to_end",
        stage_ms={"denoise": 900.0, "decode": 100.0},
        hotspots=[
            {"name": "scaled_dot_product_attention", "total_ms": 400.0},
            {"name": "aten::mul", "total_ms": 120.0},
        ],
        trace_paths=[tmp_path / "trace.json.gz"],
    )
    routed = TechniqueRouter().route(digest, allow_quality_gated=True, gpu_count=1)
    assert routed == ["kernel", "cache", "pisa", "quantization", "token_pruning"]


def test_profile_adds_topology_only_for_multi_gpu(tmp_path: Path) -> None:
    digest = make_profile_digest(tmp_path)
    routed = TechniqueRouter().route(digest, allow_quality_gated=False, gpu_count=4)
    assert routed == ["kernel", "topology"]
```

- [ ] **Step 7: Implement profiler collection and digest**

`Profiler.collect()` uses the same frozen workload through
`SGLangDiffusionDriver` with:

```text
--profile
--num-profiled-timesteps 5
SGLANG_DIFFUSION_TORCH_PROFILER_DIR=<campaign>/profiles/<epoch>
```

It requires a native-backend run, preserves the raw trace and performance
dump, and writes a schema-valid `ProfileDigest`. `stage_ms` comes from the
SGLang performance dump. `hotspots` is a sorted normalized table with name,
category, total time, call count, shapes when available, and source hint. A
missing detailed op table is allowed only when the raw trace is retained and
the stage timing still identifies a route.

- [ ] **Step 8: Implement deterministic technique routing**

`TechniqueRouter.route()` always considers `kernel`, adds `topology` only when
`gpu_count > 1`, and adds `cache`, `pisa`, `quantization`, and
`token_pruning` only when quality-gated methods are allowed. It records the
hotspot evidence and existing-fast-path knowledge entries used for each route.
The agent decides the actual hypothesis; the router only decides applicable
executor lanes.

- [ ] **Step 9: Run tests and commit**

```bash
python3 -m pytest \
  sgl-engine-sglang-diffusion/tests/test_driver.py \
  sgl-engine-sglang-diffusion/tests/test_profiler.py -q
git add sgl-engine-sglang-diffusion
git commit -m "feat: benchmark and profile SGLang Diffusion"
```

## Task 8: Agent runner and executor lifecycle

**Files:**

- Create: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/agents.py`
- Create: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/orchestration.py`
- Create: `sgl-engine-sglang-diffusion/prompts/executor.md`
- Create: `sgl-engine-sglang-diffusion/prompts/master.md`
- Create: `sgl-engine-sglang-diffusion/tests/test_orchestration.py`

- [ ] **Step 1: Write executor lifecycle tests**

Use a fake agent command implemented as a temporary Python script. It accepts a
prompt path and writes `DELIVERY.json` when instructed.

Test:

- one executor gets one worktree;
- repeated spawn with the same idempotency key returns the same executor;
- poll reports alive/delivered;
- resume appends exact feedback to the durable prompt and starts a new process;
  and
- an executor cannot reference a path outside its worktree.

- [ ] **Step 2: Implement `AgentRunner`**

```python
class AgentRunner:
    def __init__(self, command: list[str], model: str | None):
        self.command = command
        self.model = model

    def argv(self, prompt: Path) -> list[str]:
        argv = [*self.command]
        if self.model:
            argv.extend(["--model", self.model])
        argv.append(str(prompt))
        return argv
```

Launch with `subprocess.Popen(..., start_new_session=True)` and write a process
receipt containing PID, argv, cwd, start time, and prompt hash. Never persist
environment values whose key contains `TOKEN`, `SECRET`, `PASSWORD`, or `KEY`.

- [ ] **Step 3: Implement prompt assembly**

The executor prompt concatenates, in order:

1. `contracts/sol_engine/loop-and-gate.md`;
2. the selected technique scope;
3. generated SGLang placement rules;
4. relevant locked knowledge index entries;
5. frozen baseline JSON;
6. current search state and rejected signatures; and
7. the candidate worktree and required `DELIVERY.json` path.

Add a visible precedence statement and SHA-256 for every inserted section.

- [ ] **Step 4: Implement spawn, poll, and resume**

`ExecutorManager` uses `StateStore` idempotency and leases, creates a detached
worktree through `SourceManager`, writes `goal.md`, starts `AgentRunner`, and
records process metadata. Poll never trusts process liveness as delivery
success; delivery means a regular in-worktree `DELIVERY.json` exists.

Resume verifies the feedback is nonempty, writes a numbered feedback artifact,
and reuses the existing worktree and search ledger.

- [ ] **Step 5: Run tests and commit**

```bash
python3 -m pytest sgl-engine-sglang-diffusion/tests/test_orchestration.py -q
git add sgl-engine-sglang-diffusion
git commit -m "feat: orchestrate isolated optimization executors"
```

## Task 9: Independent Sol-Engine delivery verification

**Files:**

- Create: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/verifier.py`
- Create: `sgl-engine-sglang-diffusion/tests/test_verifier.py`

- [ ] **Step 1: Write anti-fabrication tests**

Cover:

- reported speedup differs from recomputed speedup;
- run directory escapes the worktree;
- benchmark or output video is missing;
- timing scope differs;
- source hash differs;
- quality-gated delivery lacks LPIPS;
- quality-gated delivery lacks a passing built-in visual verdict;
- lossless delivery contains LPIPS but is not rejected for numeric movement;
- lossless delivery lacks equivalence counts or method argument;
- requested activation has zero engagement; and
- valid lossless and quality-gated deliveries pass their respective objective
  checks.

- [ ] **Step 2: Implement safe artifact resolution**

```python
def resolve_inside(root: Path, relative: Path) -> Path:
    candidate = (root / relative).resolve()
    candidate.relative_to(root.resolve())
    if not candidate.exists():
        raise VerificationError(f"missing artifact: {relative}")
    return candidate
```

- [ ] **Step 3: Implement authoritative performance verification**

Load frozen `BASELINE.json` and the candidate's durable
`outputs/benchmark.json`. Require identical timing scope and compute:

```python
speedup = baseline.total_s / candidate_total_s
if not math.isclose(
    speedup, point.performance.speedup, rel_tol=1e-6, abs_tol=1e-9
):
    issues.append("reported speedup does not match durable benchmark")
```

Require a measurable latency improvement or a non-dominated memory point.

- [ ] **Step 4: Implement correctness routing**

For `lossless`:

- never invoke the plan-eval/LPIPS command;
- require `outputs/equivalence.json`;
- require equal baseline/candidate global steps and DiT calls;
- require a nonempty `method_argument`;
- require authenticity receipt and positive engagement; and
- return `lossless_required=true` so the master must audit actual code.

For `quality_gated`:

- invoke the configured locked Sol `search/plan_eval.py --no-gemini --assess`
  command through an adapter;
- require aligned `lpips_mean` and `lpips_max`;
- require `visual_verdict.json` produced by the coding agent's built-in vision;
- require `overall=pass`; and
- require positive technique engagement and zero disallowed silent fallback.

- [ ] **Step 5: Implement topology-specific verification**

Require `topology_preflight.json`, `topology_manifest.json`,
`topology_trace.json`, and `equivalence.json` with the same candidate/run IDs.
Check full rank coverage, all ranks participated, positive per-rank timing and
memory, nonempty groups/rank map/collectives, no silent fallback, and matching
source hashes.

- [ ] **Step 6: Run tests and commit**

```bash
python3 -m pytest sgl-engine-sglang-diffusion/tests/test_verifier.py -q
git add sgl-engine-sglang-diffusion
git commit -m "feat: independently verify optimization deliveries"
```

## Task 10: Integration and repeated full-workload gating

**Files:**

- Create: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/integrator.py`
- Create: `sgl-engine-sglang-diffusion/tests/test_integration_flow.py`

- [ ] **Step 1: Write integration tests**

Create two accepted fake candidate commits that modify different files and
assert the integration worktree contains both. Add a conflicting pair and
assert the result is `needs_executor_revision`, not an automatically resolved
merge.

Test that:

- unverified candidate IDs are rejected;
- topology is applied before local techniques;
- the combined activation includes every selected technique;
- any quality-gated component routes the integrated run through the lossy
  gate; and
- an all-lossless composition never invokes LPIPS.

- [ ] **Step 2: Implement `IntegrationRecipe`**

It records ordered candidate IDs, commit SHAs, activation environment,
server-argument changes, correctness modes, source hashes, and compatibility
notes. Use Pydantic `extra="forbid"`.

- [ ] **Step 3: Implement integration worktree composition**

Create a detached worktree from the locked base. Cherry-pick only verified
candidate commits in this order:

1. topology;
2. kernel;
3. cache;
4. PISA;
5. quantization;
6. token pruning;
7. extension lanes.

Abort the cherry-pick on conflict and preserve conflict diagnostics. Do not
modify executor worktrees.

- [ ] **Step 4: Run and verify the integrated recipe**

Use `SGLangDiffusionDriver` with the frozen workload. Recompute speedup from
the baseline, verify all receipts, and use `DeliveryVerifier` with
`quality_gated` if any component is lossy. Write schema-version-2
`INTEGRATED-DELIVERY.json` only after every selected component passes.

- [ ] **Step 5: Run tests and commit**

```bash
python3 -m pytest sgl-engine-sglang-diffusion/tests/test_integration_flow.py -q
git add sgl-engine-sglang-diffusion
git commit -m "feat: integrate verified optimization frontiers"
```

## Task 11: SGLang agent-profile rules and patch packaging

**Files:**

- Create: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/patcher.py`
- Create: `sgl-engine-sglang-diffusion/tests/test_patcher.py`

- [ ] **Step 1: Write SGLang path-policy tests**

Reject a generated model-specific kernel outside:

```text
python/sglang/kernels/agent/diffusion/<model_slug>/
python/sglang/kernels/jit/csrc/diffusion/agent/<model_slug>/
test/registered/kernels/ops/diffusion/agent/<model_slug>/
test/registered/kernels/benchmark/diffusion/agent/<model_slug>/
```

Allow modifications to the affected DiT/layer, ServerArgs, and tests. Reject
patches containing a home-directory path, `HF_TOKEN`, or edits below `.git`.

- [ ] **Step 2: Validate runtime profile artifacts**

Require the integration worktree to contain:

```text
python/sglang/kernels/agent/registry.py
python/sglang/kernels/agent/manifest.py
python/sglang/kernels/agent/runtime.py
python/sglang/kernels/agent/receipt.py
python/sglang/kernels/agent/diffusion/<model_slug>/manifest.json
```

Validate that the manifest contains profile ID, accepted model IDs, locked
base SHA, hardware/shape constraints, selected techniques, fallback policy,
source hashes, integrated-delivery hash, and measured speedup.

Inspect the SGLang diff for the literal CLI option `--agent-optimization` and
the modes `off` and `auto`.

- [ ] **Step 3: Generate a binary-safe patch**

Run:

```text
git diff --binary --full-index <base_sha>..HEAD
```

Write it atomically as `sglang.patch`. Generate `manifest.json`,
`SHA256SUMS`, evidence copies, and an executable `apply_and_verify.sh`.

The script:

1. checks `git rev-parse HEAD` equals the locked base;
2. runs `git apply --check`;
3. applies the patch;
4. runs configured import and unit-test commands; and
5. prints the exact GPU revalidation command without executing it unless
   `--run-gpu-validation` is supplied.

- [ ] **Step 4: Clean-room verify**

Create a new detached worktree from the base, apply the generated patch, assert
a clean `git diff --check`, execute CPU validation, and compare applied-tree
source hashes with the integration manifest.

- [ ] **Step 5: Test quantized-weight policy**

Reject an integrated manifest that names a derived checkpoint without immutable
URI, revision, size, and SHA-256. Allow patch-only or fully locked public
checkpoint profiles.

- [ ] **Step 6: Run tests and commit**

```bash
python3 -m pytest sgl-engine-sglang-diffusion/tests/test_patcher.py -q
git add sgl-engine-sglang-diffusion
git commit -m "feat: package clean-room verified SGLang patches"
```

## Task 12: Campaign controller, epochs, watchdog, and CLI

**Files:**

- Create: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/controller.py`
- Create: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/watchdog.py`
- Create: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/cli.py`
- Create: `sgl-engine-sglang-diffusion/tests/test_controller.py`
- Create: `sgl-engine-sglang-diffusion/tests/test_cli.py`

- [ ] **Step 1: Write target and epoch tests**

Cover:

- target reached only when verified integrated speedup meets the goal;
- target miss plus new hypothesis starts the next epoch;
- repeated failure signatures do not start another candidate;
- no hypothesis produces `SEARCH_SPACE_EXHAUSTED`;
- no plateau produces `UNREACHABLE_CERTIFIED`;
- a valid lower-bound certificate may produce
  `UNREACHABLE_CERTIFIED`;
- infrastructure failure enters `WAITING_RESOURCE`; and
- budget exhaustion enters `PAUSED_BUDGET`.

- [ ] **Step 2: Implement the controller state loop**

```python
def run_once(self) -> CampaignStatus:
    status = self.store.status(self.campaign_id)
    if status is CampaignStatus.NEW:
        return self.freeze_sources_and_baseline()
    if status is CampaignStatus.BASELINE_LOCKED:
        return self.profile_and_route()
    if status is CampaignStatus.PROFILED:
        return self.start_search_epoch()
    if status is CampaignStatus.SEARCHING:
        return self.poll_and_verify_executors()
    if status is CampaignStatus.INTEGRATING:
        return self.integrate_and_gate()
    if status is CampaignStatus.FINAL_VERIFYING:
        return self.package_or_continue()
    return status
```

Every method uses an idempotency key derived from campaign ID, epoch, state,
technique, and candidate ID.

- [ ] **Step 3: Implement scoped termination**

`TARGET_REACHED` requires a clean-room verified integrated delivery and
`speedup >= target_speedup`.

`UNREACHABLE_CERTIFIED` requires a schema-valid certificate containing frozen
workload, hardware, allowed-method set, lower-bound derivation, target latency,
and source evidence. Otherwise choose `SEARCH_SPACE_EXHAUSTED`.

- [ ] **Step 4: Implement watchdog**

The watchdog polls controller heartbeat and expired leases. It restarts only
the controller command recorded in the campaign manifest. It does not edit
SQLite scientific state except to reclaim an expired lease through
`StateStore`.

- [ ] **Step 5: Implement CLI**

Commands:

```text
sgl-diffusion-engine init --goal goal.yaml --run-root runs/
sgl-diffusion-engine run --campaign runs/<id>
sgl-diffusion-engine resume --campaign runs/<id>
sgl-diffusion-engine status --campaign runs/<id> [--json]
sgl-diffusion-engine sync-knowledge --campaign runs/<id>
sgl-diffusion-engine check-contracts --sol-checkout <path>
sgl-diffusion-engine package --campaign runs/<id>
sgl-diffusion-engine watchdog --campaign runs/<id>
```

Use `argparse`; every mutation command prints campaign ID, prior state, new
state, and artifact paths.

- [ ] **Step 6: Run tests and commit**

```bash
python3 -m pytest \
  sgl-engine-sglang-diffusion/tests/test_controller.py \
  sgl-engine-sglang-diffusion/tests/test_cli.py -q
git add sgl-engine-sglang-diffusion
git commit -m "feat: run persistent optimization campaigns"
```

## Task 13: CPU-only mocked end-to-end campaign

**Files:**

- Expand: `sgl-engine-sglang-diffusion/tests/helpers.py`
- Modify: `sgl-engine-sglang-diffusion/tests/test_integration_flow.py`

- [ ] **Step 1: Build the fake SGLang repository**

The fixture contains:

- a fake benchmark script that writes deterministic JSONL, video, frames,
  benchmark, equivalence, visual verdict, and receipt artifacts;
- a minimal `python/sglang/kernels` tree;
- a minimal ServerArgs file;
- a fake model file;
- test and benchmark directories; and
- commits that candidate agents can modify.

- [ ] **Step 2: Build a two-response fake agent**

First invocation writes a delivery with a fabricated 2x claim over a measured
1.1x candidate. Resume invocation writes a correct 1.1x delivery and a commit
containing a valid agent-profile skeleton.

- [ ] **Step 3: Exercise the full campaign**

The test must:

1. initialize the campaign;
2. lock sources;
3. freeze one baseline;
4. spawn an executor;
5. reject the fabricated delivery;
6. resume with exact feedback;
7. accept the second delivery;
8. stop and recreate the controller process object;
9. resume without duplicating the baseline or executor;
10. integrate the candidate;
11. clean-room apply the patch; and
12. assert the patch bundle and state are valid.

- [ ] **Step 4: Run the entire package test suite**

```bash
python3 -m pytest sgl-engine-sglang-diffusion/tests -q
```

Expected: all tests pass without CUDA, network, or an installed agent CLI.

- [ ] **Step 5: Commit**

```bash
git add sgl-engine-sglang-diffusion
git commit -m "test: cover the optimization campaign end to end"
```

## Task 14: Documentation and repository discoverability

**Files:**

- Create: `sgl-engine-sglang-diffusion/README.md`
- Modify: `README.md`
- Modify: `tests/test_repository_metadata.py`

- [ ] **Step 1: Write repository metadata tests**

```python
def test_sglang_diffusion_engine_is_discoverable() -> None:
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    engine_readme = (
        root / "sgl-engine-sglang-diffusion" / "README.md"
    ).read_text(encoding="utf-8")
    assert "[`sgl-engine-sglang-diffusion`]" in readme
    for required in [
        "sgl-diffusion-engine init",
        "sgl-diffusion-engine run",
        "sgl-diffusion-engine resume",
        "sglang.patch",
        "--agent-optimization",
        "Sol-Engine",
        "KDA-Pilot",
        "FastVideo",
    ]:
        assert required in engine_readme
```

- [ ] **Step 2: Write the engine README**

Include:

- what the engine does and does not guarantee;
- exact installation command;
- minimal goal file;
- init/run/resume/status/package commands;
- source-lock behavior;
- Sol-Engine lossless versus quality-gated semantics;
- KDA/SGLang/FastVideo knowledge precedence;
- run-directory layout;
- executor recovery;
- patch application;
- quantized-weight limitation;
- real-GPU release-validation procedure; and
- security guidance for tokens, remote documentation, and isolated worktrees.

- [ ] **Step 3: Add the root README section**

Add a concise section after the SGLang SOTA loop:

```markdown
## SGL-Engine for SGLang Diffusion

[`sgl-engine-sglang-diffusion`](sgl-engine-sglang-diffusion/) is an executable,
persistent Sol-Engine-compatible optimization controller. It locks an SGLang
revision, runs isolated technique agents, verifies their real GPU evidence,
integrates accepted candidates, and emits a clean-room-checked `sglang.patch`.
```

- [ ] **Step 4: Run documentation and package tests**

```bash
python3 -m pytest tests/test_repository_metadata.py \
  sgl-engine-sglang-diffusion/tests -q
pre-commit run --all-files
```

Expected: all tests and hooks pass.

- [ ] **Step 5: Commit**

```bash
git add README.md tests/test_repository_metadata.py \
  sgl-engine-sglang-diffusion
git commit -m "docs: document SGLang Diffusion optimization campaigns"
```

## Task 15: Final validation and draft-PR preparation

**Files:**

- Review all files changed by Tasks 1–14.

- [ ] **Step 1: Install the package in an isolated environment**

```bash
python3 -m venv /tmp/sgl-engine-sglang-diffusion-venv
/tmp/sgl-engine-sglang-diffusion-venv/bin/pip install \
  -e 'sgl-engine-sglang-diffusion[dev]'
```

Expected: installation succeeds and the console script is available.

- [ ] **Step 2: Run CLI smoke checks**

```bash
/tmp/sgl-engine-sglang-diffusion-venv/bin/sgl-diffusion-engine --help
/tmp/sgl-engine-sglang-diffusion-venv/bin/sgl-diffusion-engine init --help
/tmp/sgl-engine-sglang-diffusion-venv/bin/sgl-diffusion-engine resume --help
```

Expected: all commands exit zero.

- [ ] **Step 3: Run complete validation**

```bash
/tmp/sgl-engine-sglang-diffusion-venv/bin/python -m pytest \
  tests/test_repository_metadata.py \
  sgl-engine-sglang-diffusion/tests -q
pre-commit run --all-files
git diff --check origin/main...HEAD
```

Expected: tests and hooks pass; no whitespace errors.

- [ ] **Step 4: Audit scope and secrets**

```bash
git status -sb
git diff --stat origin/main...HEAD
git diff --name-only origin/main...HEAD
rg -n '(HF_TOKEN=|gho_|sk-[A-Za-z0-9]|/Users/|/home/[^<])' \
  sgl-engine-sglang-diffusion README.md tests || true
```

Expected: only intended engine, documentation, design, plan, and repository
metadata files are changed; no credentials or machine-specific paths appear in
shipping artifacts.

- [ ] **Step 5: Prepare the draft PR body**

The body must cover:

- Sol-Engine workflow parity;
- SGLang/KDA/FastVideo knowledge extensions;
- persistent recovery and anti-fabrication;
- SGLang patch/profile protocol;
- CPU-only validation results;
- the deferred real-GPU validation procedure; and
- known limitation that a universal 2x result is not part of this framework PR.

- [ ] **Step 6: Push and open a draft PR**

```bash
git push -u origin agent/sgl-engine-sglang-diffusion
```

Then create a draft PR targeting `BBuf/AI-Infra-Auto-Driven-SKILLS:main` with
the reviewed title:

```text
Add SGL-Engine workflow for SGLang Diffusion
```
