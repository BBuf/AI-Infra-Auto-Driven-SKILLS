from __future__ import annotations

from .config import KernelTask
from .lineage import LineageEntry


def is_stagnated(
    lineage: list[LineageEntry],
    *,
    window: int = 3,
    min_relative_gain: float = 0.005,
) -> bool:
    if len(lineage) <= window:
        return False
    scores = [
        float(entry.score.get("geomean_vs_baseline", 0.0)) for entry in lineage
    ]
    recent_best = max(scores[-window:])
    previous_best = max(scores[: -window])
    if previous_best <= 0:
        return False
    return (recent_best - previous_best) / previous_best < min_relative_gain


def supervisor_hints(task: KernelTask, lineage: list[LineageEntry]) -> list[str]:
    hints = [
        "Re-read the semantic spec and benchmark cases before editing.",
        "Change one hypothesis at a time and keep failed attempts in notes.",
    ]
    if "cuda" in task.lane.lower():
        hints.append(
            "Profile for memory bandwidth, instruction count, occupancy, and launch overhead before changing tiling."
        )
    if "triton" in task.lane.lower():
        hints.append(
            "Check program count, vector width, masks, and cache behavior for the hottest shape."
        )
    if is_stagnated(lineage):
        hints.append(
            "Plateau detected: switch direction instead of refining the last micro-change."
        )
    return hints
