from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from .config import KernelTask


@dataclass(frozen=True)
class ScoreSummary:
    correct: bool
    matched_cases: int
    missing_cases: tuple[str, ...]
    geomean_vs_baseline: float
    clearly_better_than_baseline: bool
    per_case: tuple[dict[str, float | str], ...]


def _geomean(values: list[float]) -> float:
    if not values:
        return 0.0
    if any(value <= 0 for value in values):
        return 0.0
    return math.exp(sum(math.log(value) for value in values) / len(values))


def score_against_baseline(
    task: KernelTask,
    result: dict[str, Any],
    *,
    improvement_threshold: float = 1.05,
    per_case_floor: float = 1.0,
) -> ScoreSummary:
    """Compare candidate latency rows to the configured baseline for one task.

    A ratio above 1.0 means the candidate is faster than the baseline for
    that case. A candidate is "clearly better" only when it is correct, covers
    every configured case, and beats the baseline geomean by the threshold.
    """

    correct = bool(result.get("correct", False))
    measurements = {
        str(row["case_id"]): float(row["latency_us"])
        for row in result.get("measurements", [])
        if "case_id" in row and "latency_us" in row
    }

    ratios: list[float] = []
    per_case: list[dict[str, float | str]] = []
    missing: list[str] = []
    for baseline in task.baseline:
        case_id = str(baseline["case_id"])
        baseline_latency = float(baseline["baseline_latency_us"])
        candidate_latency = measurements.get(case_id)
        if candidate_latency is None:
            missing.append(case_id)
            continue
        ratio = baseline_latency / candidate_latency if candidate_latency > 0 else 0.0
        ratios.append(ratio)
        per_case.append(
            {
                "case_id": case_id,
                "baseline_latency_us": baseline_latency,
                "candidate_latency_us": candidate_latency,
                "speedup_vs_baseline": ratio,
            }
        )

    geomean = _geomean(ratios)
    no_case_regressed = bool(ratios) and min(ratios) >= per_case_floor
    clearly_better = (
        correct
        and not missing
        and len(ratios) == len(task.baseline)
        and no_case_regressed
        and geomean >= improvement_threshold
    )
    if not correct:
        geomean = 0.0
        clearly_better = False

    return ScoreSummary(
        correct=correct,
        matched_cases=len(ratios),
        missing_cases=tuple(missing),
        geomean_vs_baseline=geomean,
        clearly_better_than_baseline=clearly_better,
        per_case=tuple(per_case),
    )
