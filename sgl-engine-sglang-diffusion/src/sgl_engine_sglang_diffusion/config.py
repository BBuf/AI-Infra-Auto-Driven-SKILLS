from __future__ import annotations

from pathlib import Path

import yaml

from .models import CampaignGoal


def load_goal(path: Path) -> CampaignGoal:
    """Load and validate a campaign goal and its frozen prompt set."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    goal = CampaignGoal.model_validate(data)
    if not goal.workload.prompts.is_absolute():
        goal.workload.prompts = (path.parent / goal.workload.prompts).resolve()
    if not goal.workload.prompts.is_file():
        raise ValueError(f"prompt file does not exist: {goal.workload.prompts}")
    prompts = [
        line
        for line in goal.workload.prompts.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(prompts) < goal.workload.prompt_count:
        raise ValueError("prompt file contains fewer than five non-empty prompts")
    return goal
