from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from .models import CampaignStatus
from .state import StateStore, TERMINAL_STATUSES
from .techniques import TechniqueRegistry
from .telemetry import refresh_token_usage, token_totals


_PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def build_progress(campaign: Path) -> dict[str, Any]:
    campaign = campaign.resolve()
    manifest = _read_object(campaign / "CAMPAIGN.json")
    goal = yaml.safe_load((campaign / "GOAL.yaml").read_text(encoding="utf-8"))
    campaign_id = str(manifest["campaign_id"])
    with StateStore.open(campaign / "state.sqlite", campaign / "events.jsonl") as store:
        status = store.status(campaign_id)
        epoch = store.epoch(campaign_id)
        events = store.events(campaign_id)
        failures = store.failures(campaign_id)

    registry = TechniqueRegistry.load(_PACKAGE_ROOT / "techniques" / "registry.toml")
    routes = _routes(campaign, registry.default_order)
    attempts = _technique_attempts(events)
    scientific_rounds = _scientific_rounds(events)
    isolated = _isolated_speedups(campaign)
    (
        integrated_speedup,
        integrated_mean_e2e_s,
        integrated_workload_total_s,
        integrated_request_count,
        integrated_techniques,
    ) = _integrated(campaign)
    baseline_record = _read_object_optional(campaign / "BASELINE.json") or {}
    baseline_mean_e2e_s = _number(baseline_record.get("mean_e2e_s"))
    baseline_workload_total_s = _number(baseline_record.get("workload_total_s"))
    raw_baseline_count = baseline_record.get("request_count")
    baseline_request_count = (
        raw_baseline_count if type(raw_baseline_count) is int else None
    )
    rows: list[dict[str, Any]] = []
    for technique in routes:
        failure_count = sum(1 for item in failures if item["technique"] == technique)
        if technique in integrated_techniques:
            state = "integrated"
        elif technique in isolated:
            state = "verified"
        elif attempts.get(technique, 0) > 0:
            state = "running" if status is CampaignStatus.SEARCHING else "attempted"
        else:
            state = "pending"
        rows.append(
            {
                "technique": technique,
                "status": state,
                "attempts": attempts.get(technique, 0),
                "scientific_rounds": scientific_rounds.get(technique, 0),
                "round_budget": registry[technique].round_budget,
                "best_isolated_e2e_speedup": isolated.get(technique),
                "gate": (
                    "passed"
                    if technique in isolated
                    else "rejected_last"
                    if failure_count
                    else "pending"
                ),
                "failure_count": failure_count,
                "integrated": technique in integrated_techniques,
                "marginal_attribution": "not_measured",
            }
        )
    target = float(goal["goal"]["target_speedup"])
    best = max([1.0, integrated_speedup or 0.0, *isolated.values()])
    performance_fraction = round(_clamp((best - 1.0) / (target - 1.0)), 8)
    search_used = sum(scientific_rounds.get(name, 0) for name in routes)
    search_budget = sum(registry[name].round_budget for name in routes)

    token_records = refresh_token_usage(campaign)
    tokens = token_totals(token_records)
    tokens_by_role = _token_breakdown(token_records, "agent_role")
    tokens_by_technique = _token_breakdown(token_records, "technique")
    launch_request = _read_object_optional(campaign / "LAUNCH-REQUEST.json") or {}
    token_budget = launch_request.get("token_budget")
    if not isinstance(token_budget, int) or token_budget <= 0:
        token_budget = None
    token_fraction = (
        _clamp(tokens["total_tokens"] / token_budget)
        if token_budget is not None
        else None
    )
    unavailable = sum(1 for record in token_records if record.get("available") is False)
    created = datetime.fromisoformat(str(manifest["created_at"]))
    now = datetime.now(UTC)
    if created.tzinfo is None:
        created = created.replace(tzinfo=UTC)
    elapsed = max(0, int((now - created).total_seconds()))
    patch = campaign / "patch" / "sglang.patch"
    certificate = campaign / "UNREACHABLE.json"
    return {
        "schema_version": 2,
        "campaign_id": campaign_id,
        "campaign": str(campaign),
        "model": str(goal["model"]["id"]),
        "machine": str(goal["hardware"]["environment"]),
        "status": status.value,
        "terminal": status in TERMINAL_STATUSES,
        "epoch": epoch,
        "created_at": created.isoformat(),
        "updated_at": now.isoformat(),
        "elapsed_seconds": elapsed,
        "target_speedup": target,
        "baseline_mean_e2e_s": baseline_mean_e2e_s,
        "baseline_workload_total_s": baseline_workload_total_s,
        "baseline_request_count": baseline_request_count,
        "integrated_mean_e2e_s": integrated_mean_e2e_s,
        "integrated_workload_total_s": integrated_workload_total_s,
        "integrated_request_count": integrated_request_count,
        "best_verified_speedup": best,
        "integrated_stack_speedup": integrated_speedup,
        "performance_progress": performance_fraction,
        "search": {
            "rounds_used": search_used,
            "round_budget": search_budget,
            "fraction": _clamp(search_used / search_budget) if search_budget else 0.0,
        },
        "tokens": {
            **tokens,
            "budget": token_budget,
            "fraction": token_fraction,
            "exact_invocations": sum(
                1 for item in token_records if item.get("available") is True
            ),
            "unavailable_invocations": unavailable,
            "by_role": tokens_by_role,
            "by_technique": tokens_by_technique,
        },
        "techniques": rows,
        "current_work": _current_work(status, events),
        "artifacts": {
            "patch": str(patch) if patch.is_file() else None,
            "unreachable_certificate": (
                str(certificate) if certificate.is_file() else None
            ),
            "token_ledger": str(campaign / "TOKEN-USAGE.jsonl"),
            "events": str(campaign / "events.jsonl"),
        },
    }


def write_progress(campaign: Path) -> dict[str, Any]:
    projection = build_progress(campaign)
    path = campaign.resolve() / "PROGRESS.json"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(projection, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return projection


def render_progress(progress: dict[str, Any]) -> str:
    lines = [
        (
            f"{progress['model']} · {progress['machine']} · "
            f"TARGET {progress['target_speedup']:.2f}x"
        ),
        "",
        (
            f"performance {_bar(progress['performance_progress'])} "
            f"{progress['best_verified_speedup']:.2f}x / "
            f"{progress['target_speedup']:.2f}x"
        ),
        (
            f"search      {_bar(progress['search']['fraction'])} "
            f"{progress['search']['rounds_used']} / "
            f"{progress['search']['round_budget']} rounds"
        ),
        (
            f"phase       {progress['status']} · epoch {progress['epoch']} · "
            f"elapsed {_duration(progress['elapsed_seconds'])}"
        ),
    ]
    tokens = progress["tokens"]
    token_line = (
        f"tokens      {tokens['total_tokens']:,} total · "
        f"{tokens['input_tokens']:,} input · "
        f"{tokens['output_tokens']:,} output"
    )
    if tokens["budget"] is not None:
        token_line += (
            f"\n            {_bar(tokens['fraction'])} "
            f"{tokens['total_tokens']:,} / {tokens['budget']:,}"
        )
    if tokens["unavailable_invocations"]:
        token_line += f" · {tokens['unavailable_invocations']} runtime(s) unavailable"
    if tokens["by_role"]:
        token_line += "\n            by role: " + ", ".join(
            f"{name}={value:,}" for name, value in tokens["by_role"].items()
        )
    lines.extend(
        [
            token_line,
            "",
            "technique          state       gate           tries  isolated e2e",
        ]
    )
    if progress["baseline_mean_e2e_s"] is not None:
        latency = f"latency     {progress['baseline_mean_e2e_s']:.4f}s/request baseline"
        if progress["integrated_mean_e2e_s"] is not None:
            latency += (
                f" -> {progress['integrated_mean_e2e_s']:.4f}s/request integrated"
            )
        lines.insert(5, latency)
    if (
        progress["baseline_workload_total_s"] is not None
        and progress["baseline_request_count"] is not None
    ):
        workload = (
            f"workload    {progress['baseline_workload_total_s']:.4f}s/"
            f"{progress['baseline_request_count']} requests baseline"
        )
        if (
            progress["integrated_workload_total_s"] is not None
            and progress["integrated_request_count"] is not None
        ):
            workload += (
                f" -> {progress['integrated_workload_total_s']:.4f}s/"
                f"{progress['integrated_request_count']} requests integrated"
            )
        lines.insert(6, workload)
    for row in progress["techniques"]:
        speedup = row["best_isolated_e2e_speedup"]
        speedup_text = f"{speedup:.2f}x" if speedup is not None else "-"
        lines.append(
            f"{row['technique']:<18} {row['status']:<11} "
            f"{row['gate']:<14} {row['attempts']:>5}  {speedup_text:>12}"
        )
    integrated = progress["integrated_stack_speedup"]
    lines.extend(
        [
            "-" * 67,
            (
                "integrated stack"
                + (f"{integrated:>36.2f}x" if integrated is not None else f"{'-':>37}")
            ),
            "",
            f"current: {progress['current_work']}",
            f"updated: {progress['updated_at']}",
        ]
    )
    if progress["artifacts"]["patch"]:
        lines.append(f"patch: {progress['artifacts']['patch']}")
    return "\n".join(lines)


def watch_progress(
    campaign: Path,
    *,
    interval_seconds: float = 5.0,
    json_output: bool = False,
) -> None:
    if interval_seconds <= 0:
        raise ValueError("progress interval must be positive")
    while True:
        projection = write_progress(campaign)
        if json_output:
            print(json.dumps(projection, sort_keys=True), flush=True)
        else:
            print(render_progress(projection), flush=True)
        if projection["terminal"]:
            return
        time.sleep(interval_seconds)


def _routes(campaign: Path, defaults: list[str]) -> list[str]:
    value = _read_object_optional(campaign / "ROUTES.json")
    if value is None or not isinstance(value.get("routes"), list):
        return list(defaults)
    return [str(item) for item in value["routes"]]


def _token_breakdown(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    totals: dict[str, int] = {}
    for record in records:
        label = record.get(field)
        if (
            record.get("available") is not True
            or not isinstance(label, str)
            or not label
        ):
            continue
        totals[label] = totals.get(label, 0) + int(record.get("total_tokens", 0))
    return dict(sorted(totals.items()))


def _technique_attempts(events: list[dict[str, Any]]) -> dict[str, int]:
    executor_techniques: dict[str, str] = {}
    attempts: dict[str, int] = {}
    for event in events:
        payload = event["payload"]
        if event["event_type"] == "executor_spawned":
            technique = payload.get("technique")
            executor_id = payload.get("executor_id")
            if isinstance(technique, str) and isinstance(executor_id, str):
                executor_techniques[executor_id] = technique
                attempts[technique] = attempts.get(technique, 0) + 1
        elif event["event_type"] == "executor_resumed":
            technique = executor_techniques.get(str(payload.get("executor_id")))
            if technique is not None:
                attempts[technique] = attempts.get(technique, 0) + 1
    return attempts


def _scientific_rounds(events: list[dict[str, Any]]) -> dict[str, int]:
    rounds: dict[str, int] = {}
    for event in events:
        if event["event_type"] != "scientific_round_completed":
            continue
        technique = event["payload"].get("technique")
        if isinstance(technique, str) and technique:
            rounds[technique] = rounds.get(technique, 0) + 1
    return rounds


def _isolated_speedups(campaign: Path) -> dict[str, float]:
    result: dict[str, float] = {}
    for path in sorted(campaign.glob("search/*/VERIFIED-CANDIDATES.json")):
        value = _read_object_optional(path)
        if value is None or not isinstance(value.get("candidates"), dict):
            continue
        for technique, candidate in value["candidates"].items():
            if not isinstance(candidate, dict):
                continue
            speedup = candidate.get("verified_speedup")
            if isinstance(speedup, (int, float)) and not isinstance(speedup, bool):
                result[str(technique)] = max(
                    float(speedup), result.get(str(technique), 0.0)
                )
    return result


def _integrated(
    campaign: Path,
) -> tuple[float | None, float | None, float | None, int | None, set[str]]:
    speedup: float | None = None
    candidate_mean_e2e_s: float | None = None
    candidate_workload_total_s: float | None = None
    request_count: int | None = None
    techniques: set[str] = set()
    for path in sorted(
        campaign.glob("integration/*/attempt-*/INTEGRATED-DELIVERY.json")
    ):
        value = _read_object_optional(path)
        if value is None:
            continue
        points = value.get("frontier_points")
        if not isinstance(points, list):
            continue
        for point in points:
            if not isinstance(point, dict):
                continue
            implementation = point.get("implementation_manifest")
            recipe = (
                implementation.get("recipe")
                if isinstance(implementation, dict)
                else None
            )
            raw_techniques = (
                recipe.get("techniques") if isinstance(recipe, dict) else None
            )
            point_techniques = (
                {str(item) for item in raw_techniques}
                if isinstance(raw_techniques, list)
                else set()
            )
            performance = point.get("performance")
            if isinstance(performance, dict):
                raw = performance.get("speedup")
                if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                    measured = float(raw)
                    if speedup is None or measured >= speedup:
                        speedup = measured
                        techniques = point_techniques
                        candidate_mean_e2e_s = _number(
                            performance.get("candidate_mean_e2e_s")
                        )
                        candidate_workload_total_s = _number(
                            performance.get("candidate_workload_total_s")
                        )
                        raw_count = performance.get("request_count")
                        request_count = raw_count if type(raw_count) is int else None
    return (
        speedup,
        candidate_mean_e2e_s,
        candidate_workload_total_s,
        request_count,
        techniques,
    )


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _current_work(status: CampaignStatus, events: list[dict[str, Any]]) -> str:
    if events:
        payload = events[-1]["payload"]
        reason = payload.get("reason")
        technique = payload.get("technique")
        detail = " · ".join(str(item) for item in (technique, reason) if item)
        if detail:
            return detail
    return {
        CampaignStatus.NEW: "locking sources and preparing the baseline",
        CampaignStatus.BASELINE_LOCKED: "profiling the frozen baseline",
        CampaignStatus.PROFILED: "routing optimization techniques",
        CampaignStatus.SEARCHING: "optimization executors are running",
        CampaignStatus.INTEGRATING: "integrating verified candidates",
        CampaignStatus.FINAL_VERIFYING: "running final full-workload verification",
        CampaignStatus.TARGET_REACHED: "target reached and patch packaged",
        CampaignStatus.UNREACHABLE_CERTIFIED: "target independently certified unreachable",
        CampaignStatus.SEARCH_SPACE_EXHAUSTED: "reviewed search budgets exhausted",
        CampaignStatus.WAITING_RESOURCE: "waiting for an owned GPU/resource",
        CampaignStatus.INFRA_BLOCKED: "infrastructure blocked",
        CampaignStatus.PAUSED_BUDGET: "token/search budget paused",
        CampaignStatus.CANCELLED: "campaign cancelled",
    }[status]


def _bar(fraction: float | None, width: int = 20) -> str:
    value = _clamp(float(fraction or 0.0))
    filled = round(value * width)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def _duration(seconds: int) -> str:
    hours, remainder = divmod(seconds, 3600)
    minutes, second = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{second:02d}"


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, value))


def _read_object(path: Path) -> dict[str, Any]:
    value = _read_object_optional(path)
    if value is None:
        raise ValueError(f"invalid JSON object: {path}")
    return value


def _read_object_optional(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None
