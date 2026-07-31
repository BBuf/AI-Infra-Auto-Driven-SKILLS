from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

from sgl_engine_sglang_diffusion.launcher import LaunchError, launch_campaign
from sgl_engine_sglang_diffusion.models import CampaignStatus
from sgl_engine_sglang_diffusion.state import StateStore


def write_request(tmp_path: Path) -> Path:
    checkout = tmp_path / "sglang"
    benchmark = (
        checkout / "python/sglang/multimodal_gen/benchmarks/bench_offline_throughput.py"
    )
    benchmark.parent.mkdir(parents=True)
    benchmark.write_text("# benchmark\n")
    prompts = checkout / "prompts.txt"
    prompts.write_text("\n".join(f"prompt {index}" for index in range(5)) + "\n")
    request = {
        "schema_version": 1,
        "machine": "test-b200",
        "model": "test/model",
        "sglang_checkout": str(checkout),
        "sglang_ref": "main",
        "gpu_count": 1,
        "target_speedup": 2.0,
        "agent": {"command": ["legacy-agent"]},
        "token_budget": 1000,
        "baseline": {
            "cwd": str(checkout),
            "command": (
                "CUDA_VISIBLE_DEVICES=0 python "
                "python/sglang/multimodal_gen/benchmarks/"
                "bench_offline_throughput.py --model-path test/model "
                "--dataset vbench --dataset-path prompts.txt --num-prompts 5"
            ),
        },
        "run_root": str(tmp_path / "runs"),
        "idempotency_key": "same-request",
    }
    path = tmp_path / "request.yaml"
    path.write_text(yaml.safe_dump(request, sort_keys=False))
    return path


def test_launch_is_one_shot_detached_and_idempotent(tmp_path: Path) -> None:
    calls: list[Path] = []

    def spawn(campaign: Path) -> int:
        calls.append(campaign)
        (campaign / "WATCHDOG.json").write_text(json.dumps({"pid": os.getpid()}) + "\n")
        return os.getpid()

    request = write_request(tmp_path)
    first = launch_campaign(request, detach=True, watchdog_spawner=spawn)
    second = launch_campaign(request, detach=True, watchdog_spawner=spawn)

    assert first["campaign"] == second["campaign"]
    assert first["watchdog_pid"] == os.getpid()
    assert second["reused"] is True
    assert calls == [Path(first["campaign"])]
    campaign = Path(first["campaign"])
    assert (campaign / "BASELINE-COMMAND.json").is_file()
    assert (campaign / "LAUNCH.json").is_file()
    assert (campaign / "validation-prompts.txt").is_file()
    frozen_request = (campaign / "REQUEST.yaml").read_text()
    assert "CUDA_VISIBLE_DEVICES" not in frozen_request
    assert "agent:" not in frozen_request
    assert "token_budget:" not in frozen_request
    launch_request = json.loads((campaign / "LAUNCH-REQUEST.json").read_text())
    assert "token_budget" not in launch_request
    assert second["progress_command"][-1] == "--watch"
    assert second["execution_mode"] == "interactive_single_agent"
    assert second["work_command"][-1] == "--json"


def test_reused_launch_recovers_missing_watchdog_and_rejects_key_conflict(
    tmp_path: Path,
) -> None:
    request = write_request(tmp_path)
    first = launch_campaign(request, detach=False)
    calls: list[Path] = []

    def spawn(campaign: Path) -> int:
        calls.append(campaign)
        return os.getpid()

    resumed = launch_campaign(request, detach=True, watchdog_spawner=spawn)
    assert resumed["campaign"] == first["campaign"]
    assert calls == [Path(first["campaign"])]

    value = yaml.safe_load(request.read_text())
    value["target_speedup"] = 3.0
    request.write_text(yaml.safe_dump(value, sort_keys=False))
    with pytest.raises(LaunchError, match="another request"):
        launch_campaign(request, detach=False)


def test_reused_launch_does_not_restart_watchdog_at_agent_boundary(
    tmp_path: Path,
) -> None:
    request = write_request(tmp_path)
    first = launch_campaign(request, detach=False)
    campaign = Path(first["campaign"])
    manifest = json.loads((campaign / "CAMPAIGN.json").read_text())
    with StateStore.open(campaign / "state.sqlite", campaign / "events.jsonl") as store:
        campaign_id = manifest["campaign_id"]
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
    calls: list[Path] = []

    reused = launch_campaign(
        request,
        detach=True,
        watchdog_spawner=lambda value: calls.append(value) or os.getpid(),
    )

    assert reused["reused"] is True
    assert reused["watchdog_pid"] is None
    assert calls == []
