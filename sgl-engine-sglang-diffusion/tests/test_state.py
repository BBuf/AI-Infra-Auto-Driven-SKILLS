from __future__ import annotations

import json
from pathlib import Path

import pytest

from sgl_engine_sglang_diffusion.models import CampaignStatus
from sgl_engine_sglang_diffusion.state import (
    IdempotencyConflict,
    InvalidTransition,
    LeaseUnavailable,
    StateStore,
)


def make_store(tmp_path: Path) -> StateStore:
    return StateStore.open(tmp_path / "state.sqlite", tmp_path / "events.jsonl")


def test_transition_is_idempotent(tmp_path: Path) -> None:
    store = make_store(tmp_path)
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
    store = make_store(tmp_path)
    store.create_campaign("c1")
    store.acquire_lease("executor:kernel", "worker-a", ttl_seconds=0)
    store.acquire_lease("executor:kernel", "worker-b", ttl_seconds=60)
    assert store.lease_owner("executor:kernel") == "worker-b"


def test_live_lease_cannot_be_stolen(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.acquire_lease("executor:kernel", "worker-a", ttl_seconds=60)

    with pytest.raises(LeaseUnavailable, match="worker-a"):
        store.acquire_lease("executor:kernel", "worker-b", ttl_seconds=60)
    assert store.lease_owner("executor:kernel") == "worker-a"


def test_failure_signatures_are_deduplicated(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.create_campaign("c1")
    assert store.record_failure("c1", "kernel", "same", {"reason": "build"})
    assert not store.record_failure(
        "c1", "kernel", "same", {"reason": "changed payload"}
    )
    assert store.failures("c1") == [
        {
            "signature": "same",
            "campaign_id": "c1",
            "technique": "kernel",
            "payload": {"reason": "build"},
        }
    ]


def test_terminal_status_rejects_outgoing_transition(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.create_campaign("c1")
    store.transition("c1", CampaignStatus.BASELINE_LOCKED, idempotency_key="base")
    store.transition("c1", CampaignStatus.PROFILED, idempotency_key="profile")
    store.transition("c1", CampaignStatus.AWAITING_AGENT, idempotency_key="await")
    store.transition("c1", CampaignStatus.SEARCHING, idempotency_key="search")
    store.transition("c1", CampaignStatus.INTEGRATING, idempotency_key="integrate")
    store.transition(
        "c1", CampaignStatus.FINAL_VERIFYING, idempotency_key="final-verify"
    )
    store.transition("c1", CampaignStatus.TARGET_REACHED, idempotency_key="target")

    with pytest.raises(InvalidTransition, match="terminal"):
        store.transition(
            "c1", CampaignStatus.SEARCHING, idempotency_key="cannot-reopen"
        )


def test_interactive_agent_wait_is_a_first_class_transition(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.create_campaign("c1")
    store.transition("c1", CampaignStatus.BASELINE_LOCKED, idempotency_key="base")
    store.transition("c1", CampaignStatus.PROFILED, idempotency_key="profile")
    store.transition("c1", CampaignStatus.AWAITING_AGENT, idempotency_key="await")
    assert store.status("c1") is CampaignStatus.AWAITING_AGENT
    store.transition("c1", CampaignStatus.SEARCHING, idempotency_key="claim")
    store.transition("c1", CampaignStatus.AWAITING_AGENT, idempotency_key="reject")
    assert store.status("c1") is CampaignStatus.AWAITING_AGENT


def test_recoverable_status_only_returns_to_prior_active_status(
    tmp_path: Path,
) -> None:
    store = make_store(tmp_path)
    store.create_campaign("c1")
    store.transition("c1", CampaignStatus.BASELINE_LOCKED, idempotency_key="base")
    store.transition("c1", CampaignStatus.WAITING_RESOURCE, idempotency_key="waiting")

    with pytest.raises(InvalidTransition, match="can only resume BASELINE_LOCKED"):
        store.transition("c1", CampaignStatus.SEARCHING, idempotency_key="bad-resume")

    store.transition(
        "c1", CampaignStatus.BASELINE_LOCKED, idempotency_key="resume-baseline"
    )
    assert store.status("c1") is CampaignStatus.BASELINE_LOCKED


def test_idempotency_key_cannot_describe_two_operations(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.create_campaign("c1")
    store.transition("c1", CampaignStatus.BASELINE_LOCKED, idempotency_key="step")

    with pytest.raises(IdempotencyConflict):
        store.increment_epoch("c1", idempotency_key="step")


def test_generic_event_is_idempotent_and_detects_conflicting_reuse(
    tmp_path: Path,
) -> None:
    store = make_store(tmp_path)
    store.create_campaign("c1")
    expected = {
        "campaign_id": "c1",
        "event_type": "executor_spawned",
        "idempotency_key": "spawn:kernel:0",
        "payload": {"technique": "kernel", "epoch": 0},
    }

    assert (
        store.record_event(
            "c1",
            "executor_spawned",
            "spawn:kernel:0",
            {"technique": "kernel", "epoch": 0},
        )
        == expected
    )
    assert (
        store.record_event(
            "c1",
            "executor_spawned",
            "spawn:kernel:0",
            {"technique": "kernel", "epoch": 0},
        )
        == expected
    )
    assert store.events("c1", event_type="executor_spawned") == [expected]

    with pytest.raises(IdempotencyConflict):
        store.record_event(
            "c1",
            "executor_spawned",
            "spawn:kernel:0",
            {"technique": "cache", "epoch": 0},
        )


def test_events_are_mirrored_after_commit(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.create_campaign("c1")
    store.transition("c1", CampaignStatus.BASELINE_LOCKED, idempotency_key="base")

    lines = (tmp_path / "events.jsonl").read_text().splitlines()
    assert [json.loads(line) for line in lines] == store.events("c1")
