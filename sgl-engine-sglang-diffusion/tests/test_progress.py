from __future__ import annotations

import json
from pathlib import Path

from sgl_engine_sglang_diffusion.cli import initialize
from sgl_engine_sglang_diffusion.progress import render_progress, write_progress
from sgl_engine_sglang_diffusion.state import StateStore


def make_campaign(tmp_path: Path) -> Path:
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("\n".join(f"prompt {index}" for index in range(5)) + "\n")
    goal = tmp_path / "goal.yaml"
    goal.write_text(
        f"""schema_version: 1
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
agent: {{command: [codex, exec], model: gpt-test}}
"""
    )
    campaign = initialize(goal, tmp_path / "runs")
    (campaign / "ROUTES.json").write_text(json.dumps({"routes": ["kernel", "cache"]}))
    return campaign


def test_progress_reports_tokens_techniques_and_nonadditive_stack(
    tmp_path: Path,
) -> None:
    campaign = make_campaign(tmp_path)
    manifest = json.loads((campaign / "CAMPAIGN.json").read_text())
    campaign_id = manifest["campaign_id"]
    with StateStore.open(campaign / "state.sqlite", campaign / "events.jsonl") as store:
        store.record_event(
            campaign_id,
            "executor_spawned",
            "spawn-kernel",
            {
                "executor_id": "kernel-1",
                "technique": "kernel",
                "attempt": 1,
            },
        )
        store.record_event(
            campaign_id,
            "executor_resumed",
            "resume-kernel",
            {"executor_id": "kernel-1", "attempt": 2},
        )
        store.record_event(
            campaign_id,
            "scientific_round_completed",
            "round-kernel-1",
            {"round_id": "round-kernel-1", "technique": "kernel"},
        )

    verified = campaign / "search" / "1"
    verified.mkdir(parents=True)
    (verified / "VERIFIED-CANDIDATES.json").write_text(
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
    agent = campaign / "executors" / "kernel-1"
    agent.mkdir(parents=True)
    stream = agent / "stdout-002.log"
    stream.write_text(
        json.dumps(
            {
                "type": "turn.completed",
                "usage": {"input_tokens": 100, "output_tokens": 20},
            }
        )
        + "\n"
    )
    (agent / "process-002.json").write_text(
        json.dumps(
            {
                "pid": 1,
                "argv": ["codex", "exec", "--json", "goal.md"],
                "stdout": str(stream),
                "context": {
                    "campaign_id": campaign_id,
                    "agent_role": "executor",
                    "technique": "kernel",
                    "attempt": 2,
                    "invocation_id": "kernel-1:2",
                },
            }
        )
    )

    progress = write_progress(campaign)
    assert progress["best_verified_speedup"] == 1.68
    assert progress["performance_progress"] == 0.68
    assert progress["integrated_stack_speedup"] == 1.68
    assert progress["baseline_total_s"] == 10.0
    assert progress["integrated_total_s"] == 5.95238095
    assert progress["tokens"]["total_tokens"] == 120
    assert progress["tokens"]["by_role"] == {"executor": 120}
    assert progress["tokens"]["by_technique"] == {"kernel": 120}
    kernel = next(
        item for item in progress["techniques"] if item["technique"] == "kernel"
    )
    assert kernel["attempts"] == 2
    assert kernel["scientific_rounds"] == 1
    assert progress["search"]["rounds_used"] == 1
    assert kernel["best_isolated_e2e_speedup"] == 1.27
    assert kernel["integrated"] is True
    assert kernel["marginal_attribution"] == "not_measured"
    rendered = render_progress(progress)
    assert "1.68x / 2.00x" in rendered
    assert "120 total" in rendered
    assert "executor=120" in rendered
    assert "integrated stack" in rendered
    assert "10.0000s baseline" in rendered
    assert (campaign / "PROGRESS.json").is_file()
