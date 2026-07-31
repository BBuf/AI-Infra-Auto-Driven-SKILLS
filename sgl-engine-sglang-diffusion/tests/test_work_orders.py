from __future__ import annotations

import json
from pathlib import Path

import pytest

from sgl_engine_sglang_diffusion.models import CampaignStatus
from sgl_engine_sglang_diffusion.sources import SourceManager
from sgl_engine_sglang_diffusion.state import StateStore
from sgl_engine_sglang_diffusion.work_orders import WorkOrderError, WorkOrderManager

pytest_plugins = ("helpers",)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _manager(
    tmp_path: Path,
    fake_git_repo: Path,
    *,
    routes: tuple[str, ...] = ("kernel", "cache"),
) -> tuple[WorkOrderManager, StateStore, Path]:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    sources = SourceManager(tmp_path / "source-cache")
    lock = sources.lock("sglang", str(fake_git_repo), "main")
    _write_json(
        campaign / "SOURCE-LOCKS.json",
        {"schema_version": 1, "sglang": lock.model_dump(mode="json")},
    )
    _write_json(campaign / "BASELINE.json", {"total_s": 10.0})
    _write_json(
        campaign / "profiles/0/PROFILE-DIGEST.json",
        {"stage_ms": {"denoise": 9000.0}},
    )
    _write_json(campaign / "KNOWLEDGE.json", {"schema_version": 1})
    _write_json(
        campaign / "SEARCH-SPACE.json",
        {"schema_version": 1, "families": {}},
    )
    _write_json(campaign / "ROUTES.json", {"schema_version": 1, "routes": routes})
    store = StateStore.open(campaign / "state.sqlite", campaign / "events.jsonl")
    store.create_campaign("campaign")
    store.transition(
        "campaign", CampaignStatus.BASELINE_LOCKED, idempotency_key="baseline"
    )
    store.transition("campaign", CampaignStatus.PROFILED, idempotency_key="profile")
    store.transition("campaign", CampaignStatus.AWAITING_AGENT, idempotency_key="await")
    return (
        WorkOrderManager(
            campaign,
            campaign_id="campaign",
            store=store,
            source_manager=sources,
        ),
        store,
        campaign,
    )


def test_claim_creates_one_exclusive_detached_worktree(
    tmp_path: Path, fake_git_repo: Path
) -> None:
    manager, store, campaign = _manager(tmp_path, fake_git_repo)

    order = manager.claim("kernel")

    assert order.epoch == 1
    assert order.technique == "kernel"
    assert order.worktree.is_dir()
    assert order.delivery_path == order.worktree / "DELIVERY.json"
    assert order.knowledge_manifest_path == campaign / "KNOWLEDGE.json"
    assert order.search_space_path == campaign / "SEARCH-SPACE.json"
    assert len(order.knowledge_manifest_sha256) == 64
    assert len(order.search_space_sha256) == 64
    assert (campaign / "search/1/AGENT-WORK.json").is_file()
    assert store.status("campaign") is CampaignStatus.SEARCHING
    assert store.epoch("campaign") == 1
    with pytest.raises(WorkOrderError, match="requires AWAITING_AGENT"):
        manager.claim("cache")
    store.close()


def test_active_work_order_rejects_bound_search_space_drift(
    tmp_path: Path, fake_git_repo: Path
) -> None:
    manager, store, campaign = _manager(tmp_path, fake_git_repo)
    manager.claim("kernel")
    _write_json(
        campaign / "SEARCH-SPACE.json",
        {"schema_version": 1, "families": {"tampered": {}}},
    )

    with pytest.raises(WorkOrderError, match="search space hash differs"):
        manager.active_work_order()
    store.close()


def test_unrouted_and_closed_techniques_fail_closed(
    tmp_path: Path, fake_git_repo: Path
) -> None:
    manager, store, _ = _manager(tmp_path, fake_git_repo, routes=("kernel",))
    with pytest.raises(WorkOrderError, match="not a routed suggestion"):
        manager.claim("cache")

    manager.skip(
        "kernel",
        classification="unsupported",
        reason="the locked hardware cannot execute this method",
    )
    assert store.status("campaign") is CampaignStatus.SEARCH_SPACE_EXHAUSTED
    with pytest.raises(WorkOrderError, match="requires AWAITING_AGENT"):
        manager.claim("kernel")
    store.close()


def test_submit_is_the_only_event_that_consumes_a_scientific_round(
    tmp_path: Path, fake_git_repo: Path
) -> None:
    manager, store, _ = _manager(tmp_path, fake_git_repo)
    order = manager.claim("kernel")
    order.delivery_path.write_text('{"schema_version": 2}', encoding="utf-8")

    payload = manager.submit(order.delivery_path)

    assert payload["technique"] == "kernel"
    assert len(store.events("campaign", event_type="candidate_submitted")) == 1
    with pytest.raises(WorkOrderError, match="already has a candidate submission"):
        manager.submit(order.delivery_path)
    work = manager.work()
    kernel = next(item for item in work["suggestions"] if item["technique"] == "kernel")
    assert kernel["scientific_rounds_used"] == 1
    store.close()


def test_blocked_skip_is_recoverable_and_consumes_no_round(
    tmp_path: Path, fake_git_repo: Path
) -> None:
    manager, store, _ = _manager(tmp_path, fake_git_repo)

    disposition = manager.skip(
        "kernel",
        classification="blocked",
        reason="the assigned GPU is temporarily occupied",
    )

    assert disposition.closed is False
    assert store.status("campaign") is CampaignStatus.AWAITING_AGENT
    assert store.events("campaign", event_type="candidate_submitted") == []
    assert any(
        item["action"] == "claim" and item["technique"] == "kernel"
        for item in manager.work()["legal_actions"]
    )
    store.close()


def test_skip_active_work_returns_to_interactive_boundary(
    tmp_path: Path, fake_git_repo: Path
) -> None:
    manager, store, _ = _manager(tmp_path, fake_git_repo)
    manager.claim("kernel")

    disposition = manager.skip(
        "kernel",
        classification="no_gain",
        reason="reviewed candidates have no latency-positive frontier",
    )

    assert disposition.closed is True
    assert store.status("campaign") is CampaignStatus.AWAITING_AGENT
    assert store.events("campaign", event_type="candidate_submitted") == []
    store.close()


def test_closing_verified_candidate_reintegrates_remaining_subset_without_round(
    tmp_path: Path,
    fake_git_repo: Path,
) -> None:
    manager, store, campaign = _manager(tmp_path, fake_git_repo)
    _write_json(
        campaign / "VERIFIED-CANDIDATES.json",
        {
            "schema_version": 1,
            "epoch": 0,
            "candidates": {
                "kernel": {"verified": True, "verified_speedup": 1.2},
                "cache": {"verified": True, "verified_speedup": 1.1},
            },
        },
    )

    manager.skip(
        "cache",
        classification="no_gain",
        reason="the composed stack regressed when cache was included",
    )

    assert store.status("campaign") is CampaignStatus.INTEGRATING
    assert store.epoch("campaign") == 1
    assert store.events("campaign", event_type="candidate_submitted") == []
    transition = store.events("campaign", event_type="transition")[-1]["payload"]
    assert transition["excluded_technique"] == "cache"
    store.close()


def test_delivery_must_be_the_regular_file_assigned_by_the_work_order(
    tmp_path: Path, fake_git_repo: Path
) -> None:
    manager, store, _ = _manager(tmp_path, fake_git_repo)
    order = manager.claim("kernel")
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    with pytest.raises(WorkOrderError, match="active work-order path"):
        manager.submit(outside)

    order.delivery_path.symlink_to(outside)
    with pytest.raises(WorkOrderError, match="regular, non-symlink"):
        manager.submit(order.delivery_path)
    store.close()
