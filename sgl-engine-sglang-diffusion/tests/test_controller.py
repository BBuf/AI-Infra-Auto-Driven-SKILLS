from __future__ import annotations

import hashlib
import json
from pathlib import Path

from sgl_engine_sglang_diffusion.controller import CampaignController, StepResult
from sgl_engine_sglang_diffusion.models import CampaignGoal, CampaignStatus
from sgl_engine_sglang_diffusion.state import StateStore

from test_driver import COMMIT, make_goal


class ScriptedHooks:
    def __init__(self, results: dict[str, StepResult]) -> None:
        self.results = results

    def freeze_sources_and_baseline(self) -> StepResult:
        return self.results["freeze"]

    def profile_and_route(self) -> StepResult:
        return self.results["profile"]

    def enter_agent_wait(self, epoch: int) -> StepResult:
        return self.results["await"]

    def verify_submitted_delivery(self, epoch: int) -> StepResult:
        return self.results["verify"]

    def integrate_and_gate(self, epoch: int) -> StepResult:
        return self.results["integrate"]

    def package_or_continue(self, epoch: int) -> StepResult:
        return self.results["package"]


def make_controller(
    tmp_path: Path,
    goal: CampaignGoal,
    hooks: ScriptedHooks,
    *,
    allowed_methods: tuple[str, ...] = ("kernel",),
) -> tuple[CampaignController, StateStore]:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    store = StateStore.open(campaign / "state.sqlite", campaign / "events.jsonl")
    store.create_campaign("c1")
    controller = CampaignController(
        store=store,
        campaign_id="c1",
        goal=goal,
        hooks=hooks,
        campaign_dir=campaign,
        allowed_methods=allowed_methods,
    )
    return controller, store


def default_results() -> dict[str, StepResult]:
    return {
        "freeze": StepResult(CampaignStatus.BASELINE_LOCKED),
        "profile": StepResult(CampaignStatus.PROFILED),
        "await": StepResult(CampaignStatus.AWAITING_AGENT),
        "verify": StepResult(CampaignStatus.INTEGRATING),
        "integrate": StepResult(CampaignStatus.FINAL_VERIFYING),
        "package": StepResult(
            CampaignStatus.TARGET_REACHED,
            verified_speedup=2.1,
            clean_room_verified=True,
        ),
    }


def claim_work(store: StateStore) -> int:
    epoch = store.increment_epoch("c1", idempotency_key="claim:1:epoch")
    store.transition(
        "c1",
        CampaignStatus.SEARCHING,
        idempotency_key="claim:1:searching",
        payload={"technique": "kernel", "epoch": epoch},
    )
    return epoch


def test_target_requires_verified_speedup_and_clean_room(tmp_path: Path) -> None:
    goal = make_goal(tmp_path)
    results = default_results()
    controller, store = make_controller(tmp_path, goal, ScriptedHooks(results))
    assert controller.run_until_wait() is CampaignStatus.AWAITING_AGENT
    claim_work(store)
    assert controller.run_until_wait() is CampaignStatus.TARGET_REACHED
    store.close()

    other = tmp_path / "other"
    other.mkdir()
    other_goal = make_goal(other)
    below = default_results()
    below["package"] = StepResult(
        CampaignStatus.TARGET_REACHED,
        verified_speedup=1.9,
        clean_room_verified=True,
    )
    controller, store = make_controller(other, other_goal, ScriptedHooks(below))
    assert controller.run_until_wait() is CampaignStatus.AWAITING_AGENT
    claim_work(store)
    assert controller.run_until_wait() is CampaignStatus.AWAITING_AGENT
    store.close()


def test_plateau_is_not_an_unreachable_proof(tmp_path: Path) -> None:
    goal = make_goal(tmp_path)
    results = default_results()
    results["verify"] = StepResult(CampaignStatus.UNREACHABLE_CERTIFIED)
    controller, store = make_controller(tmp_path, goal, ScriptedHooks(results))
    assert controller.run_until_wait() is CampaignStatus.AWAITING_AGENT
    claim_work(store)
    assert controller.run_until_wait() is CampaignStatus.AWAITING_AGENT
    store.close()


def test_valid_lower_bound_certificate_can_end_campaign(tmp_path: Path) -> None:
    goal = make_goal(tmp_path)
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    baseline = {
        "schema_version": 1,
        "model_id": goal.model.id,
        "total_s": 10.0,
        "peak_memory_mib": 1.0,
        "timing_scope": goal.workload.timing_scope,
        "run_dir": str(campaign / "baseline/run"),
        "baseline_frames": str(campaign / "baseline/frames"),
        "sglang_commit": COMMIT,
    }
    (campaign / "BASELINE.json").write_text(json.dumps(baseline))
    certificate_path = campaign / "UNREACHABLE.json"
    certificate_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "frozen_workload_sha256": hashlib.sha256(
                    goal.workload.prompts.read_bytes()
                ).hexdigest(),
                "hardware": goal.hardware.model_dump(mode="json"),
                "allowed_methods": ["kernel"],
                "target_latency_s": 5.0,
                "lower_bound_s": 5.5,
                "derivation": [{"bound": "serial decode"}],
                "source_evidence": ["profile.json"],
            }
        )
    )
    results = default_results()
    results["verify"] = StepResult(
        CampaignStatus.UNREACHABLE_CERTIFIED,
        unreachable_certificate=certificate_path,
    )
    store = StateStore.open(campaign / "state.sqlite", campaign / "events.jsonl")
    store.create_campaign("c1")
    controller = CampaignController(
        store=store,
        campaign_id="c1",
        goal=goal,
        hooks=ScriptedHooks(results),
        campaign_dir=campaign,
        allowed_methods=("kernel",),
    )
    assert controller.run_until_wait() is CampaignStatus.AWAITING_AGENT
    claim_work(store)
    assert controller.run_until_wait() is CampaignStatus.UNREACHABLE_CERTIFIED
    store.close()


def test_failure_signatures_are_not_retried(tmp_path: Path) -> None:
    goal = make_goal(tmp_path)
    controller, store = make_controller(
        tmp_path, goal, ScriptedHooks(default_results())
    )
    assert controller.admit_hypothesis(
        technique="kernel", failure_signature="same", payload={"reason": "build"}
    )
    assert not controller.admit_hypothesis(
        technique="kernel", failure_signature="same", payload={"reason": "build"}
    )
    store.close()


def test_delivery_verifier_can_wait_without_mutating_state(tmp_path: Path) -> None:
    goal = make_goal(tmp_path)
    results = default_results()
    results["verify"] = StepResult(None, payload={"reason": "delivery_missing"})
    controller, store = make_controller(tmp_path, goal, ScriptedHooks(results))
    assert controller.run_once() is CampaignStatus.BASELINE_LOCKED
    assert controller.run_once() is CampaignStatus.PROFILED
    assert controller.run_once() is CampaignStatus.AWAITING_AGENT
    claim_work(store)
    assert controller.run_once() is CampaignStatus.SEARCHING
    assert len(store.events("c1", event_type="transition")) == 4
    store.close()


def test_resource_and_budget_outcomes_are_recoverable(tmp_path: Path) -> None:
    for index, status in enumerate(
        (CampaignStatus.WAITING_RESOURCE, CampaignStatus.PAUSED_BUDGET)
    ):
        root = tmp_path / str(index)
        root.mkdir()
        goal = make_goal(root)
        results = default_results()
        results["freeze"] = StepResult(status)
        controller, store = make_controller(root, goal, ScriptedHooks(results))
        assert controller.run_once() is status
        store.close()
