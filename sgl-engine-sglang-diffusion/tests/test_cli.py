from __future__ import annotations

import json
from pathlib import Path

from sgl_engine_sglang_diffusion.cli import main


def test_init_and_status_are_cpu_only(tmp_path: Path, capsys: object) -> None:
    goal = tmp_path / "goal.yaml"
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("\n".join(f"prompt {index}" for index in range(5)) + "\n")
    goal.write_text(
        f"""schema_version: 1
model:
  id: test/model
hardware:
  environment: test
  gpu_count: 1
workload:
  prompts: {prompts.name}
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
  allow_quality_gated: false
source:
  sglang_repo: local
agent:
  command: [fake-agent]
"""
    )
    assert (
        main(["init", "--goal", str(goal), "--run-root", str(tmp_path / "runs")]) == 0
    )
    output = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    campaign = Path(output["campaign"])
    assert (campaign / "CAMPAIGN.json").is_file()
    assert (campaign / "GOAL.yaml").is_file()

    assert main(["status", "--campaign", str(campaign), "--json"]) == 0
    status = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert status["status"] == "NEW"
    assert status["epoch"] == 0

    assert main(["progress", "--campaign", str(campaign), "--json"]) == 0
    progress = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert progress["status"] == "NEW"
    assert progress["target_speedup"] == 2.0
    assert (campaign / "PROGRESS.json").is_file()


def test_help_exposes_all_campaign_commands(capsys: object) -> None:
    try:
        main(["--help"])
    except SystemExit as error:
        assert error.code == 0
    output = capsys.readouterr().out  # type: ignore[attr-defined]
    for command in (
        "init",
        "run",
        "resume",
        "status",
        "progress",
        "launch",
        "work",
        "claim",
        "submit",
        "skip",
        "sync-knowledge",
        "package",
        "watchdog",
    ):
        assert command in output
