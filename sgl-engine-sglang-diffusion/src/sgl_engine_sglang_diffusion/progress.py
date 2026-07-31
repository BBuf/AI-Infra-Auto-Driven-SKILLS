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
    rounds = _technique_rounds(events)
    dispositions = _dispositions(campaign)
    active_work_order = _active_work_order(campaign, status, epoch)
    isolated = _isolated_speedups(campaign)
    integrated_speedup, integrated_total_s, integrated_techniques = _integrated(
        campaign
    )
    baseline_record = _read_object_optional(campaign / "BASELINE.json") or {}
    raw_baseline_total = baseline_record.get("total_s")
    baseline_total_s = (
        float(raw_baseline_total)
        if isinstance(raw_baseline_total, (int, float))
        and not isinstance(raw_baseline_total, bool)
        else None
    )
    rows: list[dict[str, Any]] = []
    for technique in routes:
        failure_count = sum(1 for item in failures if item["technique"] == technique)
        if technique in integrated_techniques:
            state = "integrated"
        elif technique in isolated:
            state = "verified"
        elif (
            active_work_order is not None
            and active_work_order.get("technique") == technique
        ):
            state = "active"
        elif technique in dispositions:
            state = str(dispositions[technique].get("classification", "reviewed"))
        elif rounds.get(technique, 0) > 0:
            state = "measured"
        else:
            state = "suggested"
        rows.append(
            {
                "technique": technique,
                "status": state,
                "scientific_rounds_used": rounds.get(technique, 0),
                "round_budget": registry[technique].round_budget,
                "scientific_rounds_remaining": max(
                    0,
                    registry[technique].round_budget - rounds.get(technique, 0),
                ),
                "disposition": dispositions.get(technique),
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
    search_used = sum(rounds.get(name, 0) for name in routes)
    search_budget = sum(registry[name].round_budget for name in routes)
    created = datetime.fromisoformat(str(manifest["created_at"]))
    now = datetime.now(UTC)
    if created.tzinfo is None:
        created = created.replace(tzinfo=UTC)
    elapsed = max(0, int((now - created).total_seconds()))
    patch = campaign / "patch" / "sglang.patch"
    certificate = campaign / "UNREACHABLE.json"
    return {
        "schema_version": 1,
        "execution_mode": "interactive_single_agent",
        "campaign_id": campaign_id,
        "campaign": str(campaign),
        "model": str(goal["model"]["id"]),
        "machine": str(goal["hardware"]["environment"]),
        "status": status.value,
        "terminal": status in TERMINAL_STATUSES,
        "yielded": status is CampaignStatus.AWAITING_AGENT,
        "epoch": epoch,
        "created_at": created.isoformat(),
        "updated_at": now.isoformat(),
        "elapsed_seconds": elapsed,
        "target_speedup": target,
        "baseline_total_s": baseline_total_s,
        "integrated_total_s": integrated_total_s,
        "best_verified_speedup": best,
        "integrated_stack_speedup": integrated_speedup,
        "performance_progress": performance_fraction,
        "search": {
            "rounds_used": search_used,
            "round_budget": search_budget,
            "fraction": _clamp(search_used / search_budget) if search_budget else 0.0,
        },
        "interactive_agent_usage": {
            "available": False,
            "reason": "current-conversation token usage is not exposed to the CLI",
        },
        "techniques": rows,
        "active_work_order": active_work_order,
        "legal_actions": _legal_actions(status, active_work_order, rows),
        "current_work": _current_work(status, events),
        "artifacts": {
            "patch": str(patch) if patch.is_file() else None,
            "unreachable_certificate": (
                str(certificate) if certificate.is_file() else None
            ),
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
    lines.extend(
        [
            "",
            "technique          state       gate          rounds  isolated e2e",
        ]
    )
    if progress["baseline_total_s"] is not None:
        latency = f"latency     {progress['baseline_total_s']:.4f}s baseline"
        if progress["integrated_total_s"] is not None:
            latency += f" -> {progress['integrated_total_s']:.4f}s integrated"
        lines.insert(5, latency)
    for row in progress["techniques"]:
        speedup = row["best_isolated_e2e_speedup"]
        speedup_text = f"{speedup:.2f}x" if speedup is not None else "-"
        lines.append(
            f"{row['technique']:<18} {row['status']:<11} "
            f"{row['gate']:<13} {row['scientific_rounds_used']:>6}  "
            f"{speedup_text:>12}"
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
        if projection["terminal"] or projection["yielded"]:
            return
        time.sleep(interval_seconds)


def _routes(campaign: Path, defaults: list[str]) -> list[str]:
    value = _read_object_optional(campaign / "ROUTES.json")
    if value is None or not isinstance(value.get("routes"), list):
        return list(defaults)
    return [str(item) for item in value["routes"]]


def _technique_rounds(events: list[dict[str, Any]]) -> dict[str, int]:
    rounds: dict[str, int] = {}
    for event in events:
        payload = event["payload"]
        if event["event_type"] == "candidate_submitted":
            technique = payload.get("technique")
            if isinstance(technique, str):
                rounds[technique] = rounds.get(technique, 0) + 1
    return rounds


def _dispositions(campaign: Path) -> dict[str, dict[str, Any]]:
    value = _read_object_optional(campaign / "TECHNIQUE-DISPOSITIONS.json")
    raw = value.get("techniques") if value is not None else None
    if not isinstance(raw, dict):
        return {}
    return {
        str(name): dict(disposition)
        for name, disposition in raw.items()
        if isinstance(disposition, dict)
    }


def _active_work_order(
    campaign: Path,
    status: CampaignStatus,
    epoch: int,
) -> dict[str, Any] | None:
    if status is not CampaignStatus.SEARCHING:
        return None
    return _read_object_optional(campaign / "search" / str(epoch) / "AGENT-WORK.json")


def _legal_actions(
    status: CampaignStatus,
    active: dict[str, Any] | None,
    techniques: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if status is CampaignStatus.AWAITING_AGENT:
        actions = [
            {"action": "claim", "technique": item["technique"]}
            for item in techniques
            if item["scientific_rounds_remaining"] > 0
            and not (
                isinstance(item["disposition"], dict)
                and item["disposition"].get("closed") is True
            )
        ]
        actions.extend(
            {"action": "skip", "technique": item["technique"]} for item in techniques
        )
        return actions
    if status is CampaignStatus.SEARCHING and active is not None:
        return [
            {"action": "submit", "delivery": active.get("delivery_path")},
            {"action": "skip", "technique": active.get("technique")},
        ]
    return []


def _isolated_speedups(campaign: Path) -> dict[str, float]:
    result: dict[str, float] = {}
    paths = [campaign / "VERIFIED-CANDIDATES.json"]
    paths.extend(sorted(campaign.glob("search/*/VERIFIED-CANDIDATES.json")))
    for path in paths:
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
) -> tuple[float | None, float | None, set[str]]:
    speedup: float | None = None
    candidate_total_s: float | None = None
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
                        candidate_total_s = None
                        raw_total = performance.get("candidate_total_s")
                        if isinstance(raw_total, (int, float)) and not isinstance(
                            raw_total, bool
                        ):
                            candidate_total_s = float(raw_total)
    return speedup, candidate_total_s, techniques


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
        CampaignStatus.AWAITING_AGENT: (
            "waiting for the current root agent to claim or complete one work order"
        ),
        CampaignStatus.SEARCHING: "the current root agent owns one work order",
        CampaignStatus.INTEGRATING: "integrating verified candidates",
        CampaignStatus.FINAL_VERIFYING: "running final full-workload verification",
        CampaignStatus.TARGET_REACHED: "target reached and patch packaged",
        CampaignStatus.UNREACHABLE_CERTIFIED: "target certified unreachable",
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
