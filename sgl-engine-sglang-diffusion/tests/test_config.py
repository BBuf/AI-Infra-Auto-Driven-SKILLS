from __future__ import annotations

from pathlib import Path

import pytest

from sgl_engine_sglang_diffusion import __version__
from sgl_engine_sglang_diffusion.config import load_goal


def test_package_version() -> None:
    assert __version__ == "0.1.0"


def test_load_goal_freezes_required_workload(tmp_path: Path) -> None:
    prompt_file = tmp_path / "prompts.txt"
    prompt_file.write_text(
        "\n".join(f"prompt {index}" for index in range(5)) + "\n",
        encoding="utf-8",
    )
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
""",
        encoding="utf-8",
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
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="prompt_count"):
        load_goal(goal_file)


def test_load_checked_in_example() -> None:
    example = Path(__file__).resolve().parents[1] / "examples" / "goal.yaml"

    goal = load_goal(example)

    assert goal.workload.prompts.is_absolute()
    assert goal.workload.prompt_count == 5
