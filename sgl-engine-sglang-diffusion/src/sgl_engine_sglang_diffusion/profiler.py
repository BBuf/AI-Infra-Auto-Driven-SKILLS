from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .driver import Activation, SGLangDiffusionDriver
from .models import CampaignGoal, ProfileDigest


class ProfileError(RuntimeError):
    pass


class Profiler:
    """Collect a native SGLang trace and normalize routing evidence."""

    def __init__(self, driver: SGLangDiffusionDriver) -> None:
        self.driver = driver

    def collect(
        self,
        goal: CampaignGoal,
        campaign_dir: Path,
        *,
        epoch: int,
        activation: Activation | None = None,
    ) -> ProfileDigest:
        epoch_dir = campaign_dir.resolve() / "profiles" / str(epoch)
        existing = epoch_dir / "PROFILE-DIGEST.json"
        if existing.is_file():
            return ProfileDigest.model_validate_json(
                existing.read_text(encoding="utf-8")
            )
        attempt_numbers = [
            int(path.name.removeprefix("attempt-"))
            for path in epoch_dir.glob("attempt-[0-9][0-9][0-9]")
            if path.is_dir() and path.name.removeprefix("attempt-").isdigit()
        ]
        profile_dir = epoch_dir / (f"attempt-{max(attempt_numbers, default=0) + 1:03d}")
        benchmark = self.driver.run(
            goal, profile_dir, activation=activation, profile=True
        )
        traces = self._trace_paths(profile_dir)
        if not traces:
            raise ProfileError(f"profiler produced no durable trace in {profile_dir}")

        stage_ms, hotspots = self._performance_tables(profile_dir)
        if not stage_ms:
            stage_ms = {"end_to_end": float(benchmark.normalized["total_s"]) * 1000.0}
        digest = ProfileDigest(
            run_dir=profile_dir,
            timing_scope=goal.workload.timing_scope,
            stage_ms=stage_ms,
            hotspots=hotspots,
            trace_paths=traces,
        )
        target = epoch_dir / "PROFILE-DIGEST.json"
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(digest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
        return digest

    @staticmethod
    def _trace_paths(profile_dir: Path) -> list[Path]:
        ignored = {
            "COMMAND.json",
            "PERFORMANCE.json",
            "PROFILE-DIGEST.json",
            "benchmark.jsonl",
        }
        return sorted(
            path.resolve()
            for path in profile_dir.rglob("*")
            if path.is_file()
            and path.name not in ignored
            and (
                path.name.endswith(".trace.json")
                or path.name.endswith(".trace.json.gz")
                or path.name.endswith("trace.json")
                or path.name.endswith("trace.json.gz")
            )
        )

    @staticmethod
    def _performance_tables(
        profile_dir: Path,
    ) -> tuple[dict[str, float], list[dict[str, Any]]]:
        stage_ms: dict[str, float] = {}
        hotspots: list[dict[str, Any]] = []
        for path in sorted(profile_dir.rglob("*.json")):
            if path.name in {"COMMAND.json", "PERFORMANCE.json", "PROFILE-DIGEST.json"}:
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            raw_stages = payload.get("stage_ms")
            if isinstance(raw_stages, dict):
                for name, value in raw_stages.items():
                    if (
                        isinstance(value, (int, float))
                        and not isinstance(value, bool)
                        and value >= 0
                    ):
                        stage_ms[str(name)] = float(value)
            raw_hotspots = payload.get("hotspots", payload.get("operators"))
            if isinstance(raw_hotspots, list):
                hotspots.extend(
                    row
                    for row in (
                        Profiler._normalize_hotspot(value) for value in raw_hotspots
                    )
                    if row is not None
                )
        hotspots.sort(key=lambda item: (-item["total_ms"], item["name"]))
        return dict(sorted(stage_ms.items())), hotspots

    @staticmethod
    def _normalize_hotspot(value: Any) -> dict[str, Any] | None:
        if not isinstance(value, dict):
            return None
        name = value.get("name", value.get("operator"))
        total = value.get("total_ms", value.get("time_ms"))
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(total, (int, float))
            or isinstance(total, bool)
            or total < 0
        ):
            return None
        calls = value.get("calls", value.get("call_count", 0))
        return {
            "name": name,
            "category": str(value.get("category", "")),
            "total_ms": float(total),
            "call_count": int(calls) if isinstance(calls, int) else 0,
            "shapes": value.get("shapes", []),
            "source_hint": str(value.get("source_hint", "")),
        }


class TechniqueRouter:
    """Suggest applicable techniques without choosing their hypotheses."""

    def __init__(self) -> None:
        self.last_evidence: dict[str, dict[str, Any]] = {}

    def route(
        self,
        digest: ProfileDigest,
        *,
        allow_quality_gated: bool,
        gpu_count: int,
    ) -> list[str]:
        if gpu_count < 1:
            raise ValueError("gpu_count must be positive")
        hotspots = [
            {
                "name": str(row.get("name", "")),
                "total_ms": float(row.get("total_ms", 0.0)),
            }
            for row in digest.hotspots
        ]
        self.last_evidence = {
            "kernel": {
                "hotspots": hotspots,
                "knowledge": ["sglang-kernel-placement", "profile-evidence"],
            }
        }
        routes = ["kernel"]
        if gpu_count > 1:
            routes.append("topology")
            self.last_evidence["topology"] = {
                "gpu_count": gpu_count,
                "knowledge": ["sglang-distributed-runtime"],
            }
        if allow_quality_gated:
            for technique in ("cache", "pisa", "quantization", "token_pruning"):
                routes.append(technique)
                self.last_evidence[technique] = {
                    "hotspots": hotspots,
                    "knowledge": [
                        "sol-engine-contract",
                        "sglang-diffusion-performance",
                    ],
                }
        return routes
