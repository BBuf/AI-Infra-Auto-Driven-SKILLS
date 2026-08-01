from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

import pytest

from sgl_engine_sglang_diffusion.integrator import VerifiedCandidate
from sgl_engine_sglang_diffusion.models import CorrectnessMode, SourceLock
from sgl_engine_sglang_diffusion.orchestration import ExecutorHandle
from sgl_engine_sglang_diffusion.runtime import FileCampaignHooks
from sgl_engine_sglang_diffusion.runtime import CampaignRuntimeError
from sgl_engine_sglang_diffusion.state import StateStore
from sgl_engine_sglang_diffusion.techniques import TechniqueRegistry


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
            ("kernel", "b", 1.2),
            ("topology", "c", 1.15),
        )
    ]
    for candidate in candidates:
        hooks._register_candidate(candidate, epoch=1)

    selected = hooks._selected_candidates(2)
    assert set(selected) == {"kernel", "topology"}
    assert all(candidate.verified_speedup < 2.0 for candidate in selected.values())

    hooks._exclude_candidate("topology-win", "combined patch regressed")
    remaining = hooks._selected_candidates(2)
    assert set(remaining) == {"kernel"}
    registry = json.loads((tmp_path / "CANDIDATE-REGISTRY.json").read_text())
    assert {item["candidate"]["candidate_id"] for item in registry["history"]} == {
        "kernel-win",
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
