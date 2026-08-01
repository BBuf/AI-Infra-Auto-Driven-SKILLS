from __future__ import annotations

import json
import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import sgl_engine_sglang_diffusion.runtime as runtime_module

from sgl_engine_sglang_diffusion.integrator import VerifiedCandidate
from sgl_engine_sglang_diffusion.models import CorrectnessMode, SourceLock
from sgl_engine_sglang_diffusion.orchestration import ExecutorHandle
from sgl_engine_sglang_diffusion.runtime import FileCampaignHooks
from sgl_engine_sglang_diffusion.runtime import CampaignRuntimeError
from sgl_engine_sglang_diffusion.state import StateStore
from sgl_engine_sglang_diffusion.techniques import TechniqueRegistry


def test_gpu_inventory_resolves_frozen_visibility_before_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    hooks = object.__new__(FileCampaignHooks)
    hooks.campaign_dir = campaign
    hooks.goal = SimpleNamespace(hardware=SimpleNamespace(gpu_count=2))
    hooks._command_template = lambda: SimpleNamespace(  # type: ignore[method-assign]
        env={"CUDA_VISIBLE_DEVICES": "2,GPU-b"},
        template_sha256="a" * 64,
    )
    monkeypatch.setattr(
        runtime_module,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=(
                "0, GPU-a, 81920\n"
                "1, GPU-b, 81920\n"
                "2, GPU-c, 141312\n"
            ),
            stderr="",
        ),
    )

    inventory = hooks._ensure_gpu_inventory()

    assert inventory is not None
    assert inventory.visibility_source == "frozen_command_env"
    assert [item.uuid for item in inventory.devices] == ["GPU-c", "GPU-b"]
    assert inventory.baseline_command_template_sha256 == "a" * 64
    assert json.loads((campaign / "GPU-INVENTORY.json").read_text())["gpu_count"] == 2


def test_executor_prompt_injects_only_active_lane_history_rules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = tmp_path / "campaign"
    (campaign / "profiles/0").mkdir(parents=True)
    (campaign / "source-worktrees/sglang").mkdir(parents=True)
    (campaign / "KNOWLEDGE.json").write_text(
        json.dumps({"schema_version": 1, "snapshots": {}})
    )
    baseline_run = campaign / "baseline/run"
    baseline_frames = campaign / "baseline/frames"
    baseline_run.mkdir(parents=True)
    baseline_frames.mkdir(parents=True)
    (campaign / "BASELINE.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "model_id": "test/model",
                "mean_e2e_s": 10.0,
                "workload_total_s": 50.0,
                "request_count": 5,
                "peak_memory_mib": 1024.0,
                "timing_scope": "frozen_e2e",
                "run_dir": str(baseline_run),
                "baseline_frames": str(baseline_frames),
                "sglang_commit": "a" * 40,
            }
        )
    )
    (campaign / "profiles/0/PROFILE-DIGEST.json").write_text(
        '{"profile": "bound"}\n'
    )
    (campaign / "GPU-INVENTORY.json").write_text(
        '{"schema_version": 1, "gpu_count": 1, "devices": '
        '[{"uuid": "GPU-test"}]}\n'
    )
    store = StateStore.open(campaign / "state.sqlite", campaign / "events.jsonl")
    store.create_campaign("campaign-1")
    hooks = object.__new__(FileCampaignHooks)
    hooks.campaign_dir = campaign
    hooks.campaign_id = "campaign-1"
    hooks.goal = SimpleNamespace(
        model=SimpleNamespace(id="test/model"),
        goal=SimpleNamespace(target_speedup=5.0),
    )
    hooks.store = store
    hooks.registry = TechniqueRegistry.load(
        Path(__file__).resolve().parents[1] / "techniques/registry.toml"
    )
    monkeypatch.setattr(
        runtime_module,
        "detect_placement_contract",
        lambda _: SimpleNamespace(render=lambda **_: "placement\n"),
    )
    try:
        prompt = hooks._executor_prompt("residency", 1)
        history = prompt.knowledge[0]
        assert "Diff-reviewed residency historical rules" == history.name
        assert "#sha256=" in history.source
        assert "residency.partial-dit-layers" in history.content
        assert "kernel.regional-compile-graph" not in history.content
        assert "GPU-test" in prompt.baseline.content
    finally:
        store.close()


class RecordingExecutors:
    def __init__(self, campaign: Path) -> None:
        self.campaign = campaign
        self.spawned: list[str] = []

    def spawn(self, **kwargs: Any) -> ExecutorHandle:
        technique = str(kwargs["technique"])
        self.spawned.append(technique)
        root = self.campaign / "executors" / technique
        worktree = root / "worktree"
        worktree.mkdir(parents=True, exist_ok=True)
        handle = ExecutorHandle(
            executor_id=f"executor-{technique}",
            campaign_id="campaign-1",
            technique=technique,
            root=root,
            worktree=worktree,
            prompt=root / "goal.md",
            delivery=worktree / "DELIVERY.json",
            receipt=root / "process-001.json",
            pid=100 + len(self.spawned),
            attempt=1,
            lease_resource=f"executor:campaign-1:{technique}",
            lease_owner=f"agent:{technique}",
        )
        payload = {
            **handle.__dict__,
            "root": str(handle.root),
            "worktree": str(handle.worktree),
            "prompt": str(handle.prompt),
            "delivery": str(handle.delivery),
            "receipt": str(handle.receipt),
        }
        (root / "executor.json").write_text(json.dumps(payload))
        return handle


def test_search_epoch_activates_exactly_one_executor_at_a_time(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    store = StateStore.open(campaign / "state.sqlite", campaign / "events.jsonl")
    store.create_campaign("campaign-1")
    hooks = object.__new__(FileCampaignHooks)
    hooks.campaign_dir = campaign
    hooks.campaign_id = "campaign-1"
    hooks.store = store
    hooks.registry = TechniqueRegistry.load(
        Path(__file__).resolve().parents[1] / "techniques/registry.toml"
    )
    hooks.executors = RecordingExecutors(campaign)
    hooks._routes = lambda: ["kernel", "cache"]  # type: ignore[method-assign]
    hooks._load_locks = lambda: {  # type: ignore[method-assign]
        "sglang": SourceLock(
            name="sglang",
            repository="fake",
            requested_ref="main",
            commit="a" * 40,
        )
    }
    hooks._executor_prompt = lambda technique, epoch: None  # type: ignore[method-assign]
    try:
        result = hooks.start_search_epoch(1)
        assert result.payload["active_technique"] == "kernel"
        assert hooks.executors.spawned == ["kernel"]

        # Re-entering the same controller state must not start a second lane.
        repeated = hooks.start_search_epoch(1)
        assert repeated.payload["active_technique"] == "kernel"
        assert hooks.executors.spawned == ["kernel"]

        hooks._write_verified(
            1,
            {
                "kernel": VerifiedCandidate(
                    candidate_id="kernel-win",
                    technique="kernel",
                    base_commit="a" * 40,
                    candidate_commit="b" * 40,
                    correctness=CorrectnessMode.LOSSLESS,
                    source_hashes={"kernel.py": "c" * 64},
                    verified_speedup=1.01,
                    verified=True,
                )
            },
        )
        assert hooks._ensure_next_executor(1) == "cache"
        assert hooks.executors.spawned == ["kernel", "cache"]
        manifest = json.loads(
            (campaign / "search/1/EXECUTORS.json").read_text()
        )
        assert manifest["active_technique"] == "cache"
    finally:
        store.close()


def test_positive_candidates_compose_and_exclusion_preserves_other_wins(
    tmp_path: Path,
) -> None:
    hooks = object.__new__(FileCampaignHooks)
    hooks.campaign_dir = tmp_path
    candidates = [
        VerifiedCandidate(
            candidate_id=f"{technique}-win",
            technique=technique,
            base_commit="a" * 40,
            candidate_commit=commit * 40,
            correctness=CorrectnessMode.LOSSLESS,
            source_hashes={f"{technique}.py": "d" * 64},
            verified_speedup=speedup,
            verified=True,
        )
        for technique, commit, speedup in (
            ("residency", "e", 1.08),
            ("kernel", "b", 1.2),
            ("topology", "c", 1.15),
        )
    ]
    for candidate in candidates:
        hooks._register_candidate(candidate, epoch=1)

    selected = hooks._selected_candidates(2)
    assert set(selected) == {"residency", "kernel", "topology"}
    assert all(candidate.verified_speedup < 2.0 for candidate in selected.values())

    hooks._exclude_candidate("topology-win", "combined patch regressed")
    remaining = hooks._selected_candidates(2)
    assert set(remaining) == {"residency", "kernel"}
    registry = json.loads((tmp_path / "CANDIDATE-REGISTRY.json").read_text())
    assert {item["candidate"]["candidate_id"] for item in registry["history"]} == {
        "kernel-win",
        "residency-win",
        "topology-win",
    }


def test_lane_disposition_requires_complete_coverage_not_one_failed_hypothesis(
    tmp_path: Path,
) -> None:
    campaign = tmp_path / "campaign"
    worktree = campaign / "executor/worktree"
    worktree.mkdir(parents=True)
    evidence_path = worktree / "coverage.md"
    evidence_path.write_text("reviewed family evidence\n")
    profile = campaign / "profiles/0/PROFILE-DIGEST.json"
    profile.parent.mkdir(parents=True)
    profile.write_text('{"profile": "bound"}\n')
    store = StateStore.open(campaign / "state.sqlite", campaign / "events.jsonl")
    store.create_campaign("campaign-1")
    store.record_event(
        "campaign-1",
        "scientific_round_completed",
        "round-1",
        {"round_id": "round-1", "technique": "kernel"},
    )
    hooks = object.__new__(FileCampaignHooks)
    hooks.campaign_dir = campaign
    hooks.campaign_id = "campaign-1"
    hooks.store = store
    hooks.registry = TechniqueRegistry.load(
        Path(__file__).resolve().parents[1] / "techniques/registry.toml"
    )
    payload = {
        "schema_version": 1,
        "technique": "kernel",
        "classification": "no_gain",
        "reason": "Every required family was measured or shown inapplicable.",
        "profile_digest_sha256": hashlib.sha256(profile.read_bytes()).hexdigest(),
        "coverage": [
            {
                "id": coverage_id,
                "status": "measured" if index == 0 else "inapplicable",
                "evidence": [str(evidence_path)],
                "scientific_round_ids": ["round-1"] if index == 0 else [],
            }
            for index, coverage_id in enumerate(hooks.registry["kernel"].coverage)
        ],
    }
    disposition = worktree / "DISPOSITION.json"
    disposition.write_text(json.dumps(payload))
    try:
        assert hooks._validate_disposition(disposition, "kernel").classification == "no_gain"
        payload["coverage"] = payload["coverage"][:1]
        disposition.write_text(json.dumps(payload))
        with pytest.raises(CampaignRuntimeError, match="exact required"):
            hooks._validate_disposition(disposition, "kernel")
    finally:
        store.close()


def test_repeated_executor_protocol_failure_defers_lane_and_advances(
    tmp_path: Path,
) -> None:
    campaign = tmp_path / "campaign"
    search = campaign / "search/1"
    search.mkdir(parents=True)
    (search / "EXECUTORS.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "epoch": 1,
                "routes": ["kernel", "cache"],
                "executors": {},
                "active_technique": "kernel",
            }
        )
    )
    store = StateStore.open(campaign / "state.sqlite", campaign / "events.jsonl")
    store.create_campaign("campaign-1")
    feedback = "[invalid_kernel_evidence] citation is outside pinned knowledge"
    feedback_sha256 = hashlib.sha256(feedback.encode()).hexdigest()
    for attempt in (2, 3):
        store.record_event(
            "campaign-1",
            "executor_resumed",
            f"resume-{attempt}",
            {
                "executor_id": "executor-kernel",
                "attempt": attempt,
                "feedback_sha256": feedback_sha256,
            },
        )
    worktree = campaign / "executor/worktree"
    worktree.mkdir(parents=True)
    handle = ExecutorHandle(
        executor_id="executor-kernel",
        campaign_id="campaign-1",
        technique="kernel",
        root=worktree.parent,
        worktree=worktree,
        prompt=worktree.parent / "goal.md",
        delivery=worktree / "DELIVERY.json",
        receipt=worktree.parent / "process-003.json",
        pid=123,
        attempt=3,
        lease_resource="executor:campaign-1:kernel",
        lease_owner="agent:executor-kernel",
    )
    hooks = object.__new__(FileCampaignHooks)
    hooks.campaign_dir = campaign
    hooks.campaign_id = "campaign-1"
    hooks.store = store
    hooks.registry = TechniqueRegistry.load(
        Path(__file__).resolve().parents[1] / "techniques/registry.toml"
    )
    hooks._ensure_next_executor = lambda epoch: "cache"  # type: ignore[method-assign]
    try:
        result = hooks._resume_or_exhaust(handle, feedback, epoch=1)
        assert result.payload["reason"] == "executor_lane_deferred"
        assert result.payload["next_technique"] == "cache"
        deferred = json.loads((search / "DEFERRED-LANES.json").read_text())
        assert deferred["lanes"]["kernel"]["attempt"] == 3
        manifest = json.loads((search / "EXECUTORS.json").read_text())
        assert manifest["active_technique"] is None
    finally:
        store.close()
