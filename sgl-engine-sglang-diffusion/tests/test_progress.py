from __future__ import annotations

import json
from pathlib import Path

from sgl_engine_sglang_diffusion.cli import initialize
from sgl_engine_sglang_diffusion.progress import render_progress, write_progress
from sgl_engine_sglang_diffusion.models import CampaignStatus
from sgl_engine_sglang_diffusion.state import StateStore


def make_campaign(tmp_path: Path) -> Path:
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("\n".join(f"prompt {index}" for index in range(5)) + "\n")
    goal = tmp_path / "goal.yaml"
    goal.write_text(
        f"""schema_version: 2
execution_mode: interactive_single_agent
model: {{id: test/model}}
hardware: {{environment: test-b200, gpu_count: 1}}
workload:
  prompts: {prompts}
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
goal: {{target_speedup: 2.0, allow_quality_gated: true}}
source: {{sglang_repo: local}}
"""
    )
    campaign = initialize(goal, tmp_path / "runs")
    (campaign / "ROUTES.json").write_text(json.dumps({"routes": ["kernel", "cache"]}))
    return campaign


def test_progress_reports_single_agent_rounds_and_nonadditive_stack(
    tmp_path: Path,
) -> None:
    campaign = make_campaign(tmp_path)
    manifest = json.loads((campaign / "CAMPAIGN.json").read_text())
    campaign_id = manifest["campaign_id"]
    with StateStore.open(campaign / "state.sqlite", campaign / "events.jsonl") as store:
        store.record_event(
            campaign_id,
            "candidate_submitted",
            "submit-kernel",
            {
                "technique": "kernel",
                "epoch": 1,
                "delivery": "DELIVERY.json",
            },
        )

    (campaign / "VERIFIED-CANDIDATES.json").write_text(
        json.dumps(
            {
                "candidates": {
                    "kernel": {
                        "verified_speedup": 1.27,
                    }
                }
            }
        )
    )
    integrated = campaign / "integration" / "1" / "attempt-001"
    integrated.mkdir(parents=True)
    (campaign / "BASELINE.json").write_text(json.dumps({"total_s": 10.0}))
    (integrated / "INTEGRATED-DELIVERY.json").write_text(
        json.dumps(
            {
                "frontier_points": [
                    {
                        "performance": {
                            "speedup": 1.68,
                            "candidate_total_s": 5.95238095,
                        },
                        "implementation_manifest": {
                            "recipe": {"techniques": ["kernel"]}
                        },
                    }
                ]
            }
        )
    )
    progress = write_progress(campaign)
    assert progress["execution_mode"] == "interactive_single_agent"
    assert progress["best_verified_speedup"] == 1.68
    assert progress["performance_progress"] == 0.68
    assert progress["integrated_stack_speedup"] == 1.68
    assert progress["baseline_total_s"] == 10.0
    assert progress["integrated_total_s"] == 5.95238095
    assert progress["interactive_agent_usage"]["available"] is False
    assert "tokens" not in progress
    kernel = next(
        item for item in progress["techniques"] if item["technique"] == "kernel"
    )
    assert kernel["scientific_rounds_used"] == 1
    assert kernel["scientific_rounds_remaining"] == 39
    assert kernel["best_isolated_e2e_speedup"] == 1.27
    assert kernel["integrated"] is True
    assert kernel["marginal_attribution"] == "not_measured"
    rendered = render_progress(progress)
    assert "1.68x / 2.00x" in rendered
    assert "tokens" not in rendered
    assert "integrated stack" in rendered
    assert "10.0000s baseline" in rendered
    assert (campaign / "PROGRESS.json").is_file()


def test_progress_yields_at_interactive_agent_boundary(tmp_path: Path) -> None:
    campaign = make_campaign(tmp_path)
    manifest = json.loads((campaign / "CAMPAIGN.json").read_text())
    campaign_id = manifest["campaign_id"]
    with StateStore.open(campaign / "state.sqlite", campaign / "events.jsonl") as store:
        store.transition(
            campaign_id,
            CampaignStatus.BASELINE_LOCKED,
            idempotency_key="baseline",
        )
        store.transition(
            campaign_id,
            CampaignStatus.PROFILED,
            idempotency_key="profile",
        )
        store.transition(
            campaign_id,
            CampaignStatus.AWAITING_AGENT,
            idempotency_key="await",
        )

    progress = write_progress(campaign)

    assert progress["yielded"] is True
    assert progress["terminal"] is False
    assert "current root agent" in progress["current_work"]
    assert any(action["action"] == "claim" for action in progress["legal_actions"])
