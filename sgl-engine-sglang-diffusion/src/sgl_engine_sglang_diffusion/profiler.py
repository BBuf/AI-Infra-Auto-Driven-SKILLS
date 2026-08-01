from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from .driver import Activation, SGLangDiffusionDriver
from .models import CampaignGoal, ProfileDigest


class ProfileError(RuntimeError):
    pass


class Profiler:
    """Collect a native SGLang trace and normalize routing evidence."""

    PARSER_VERSION = "chrome-trace-v1"

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
            try:
                digest = ProfileDigest.model_validate_json(
                    existing.read_text(encoding="utf-8")
                )
                self.validate_digest(digest)
            except (OSError, UnicodeError, ValueError, ProfileError):
                rejected = existing.with_name(
                    f"PROFILE-DIGEST.rejected-{self._sha256_file(existing)[:12]}.json"
                )
                if not rejected.exists():
                    existing.replace(rejected)
            else:
                return digest
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

        del benchmark  # profile routing comes from the trace, never E2E fallback
        stage_ms, hotspots, event_count = self._performance_tables(
            profile_dir, traces
        )
        trace_sha256 = {
            str(path): self._sha256_file(path)
            for path in traces
        }
        digest = ProfileDigest(
            run_dir=profile_dir,
            timing_scope=goal.workload.timing_scope,
            stage_ms=stage_ms,
            hotspots=hotspots,
            trace_paths=traces,
            trace_sha256=trace_sha256,
            parser_version=self.PARSER_VERSION,
            event_count=event_count,
        )
        self.validate_digest(digest)
        self._write_json_atomic(
            epoch_dir / "PROFILE-INVENTORY.json",
            {
                "schema_version": 1,
                "parser_version": self.PARSER_VERSION,
                "event_count": event_count,
                "traces": [
                    {
                        "path": str(path),
                        "size_bytes": path.stat().st_size,
                        "sha256": trace_sha256[str(path)],
                    }
                    for path in traces
                ],
            },
        )
        target = epoch_dir / "PROFILE-DIGEST.json"
        self._write_json_atomic(target, digest.model_dump(mode="json"))
        return digest

    @classmethod
    def validate_digest(cls, digest: ProfileDigest) -> None:
        if digest.parser_version != cls.PARSER_VERSION:
            raise ProfileError(
                "profile digest was not extracted by the required raw-trace parser"
            )
        if digest.event_count <= 0:
            raise ProfileError("profile digest contains no timed trace events")
        if not digest.stage_ms or not digest.hotspots:
            raise ProfileError("profile digest requires non-empty stages and hotspots")
        for name, duration in digest.stage_ms.items():
            if not name or not math.isfinite(duration) or duration <= 0:
                raise ProfileError("profile digest contains an invalid stage duration")
        for hotspot in digest.hotspots:
            duration = hotspot.get("total_ms")
            if (
                not hotspot.get("name")
                or not isinstance(duration, (int, float))
                or isinstance(duration, bool)
                or not math.isfinite(float(duration))
                or float(duration) <= 0
            ):
                raise ProfileError("profile digest contains an invalid hotspot")
        if not digest.trace_paths or set(map(str, digest.trace_paths)) != set(
            digest.trace_sha256
        ):
            raise ProfileError("profile digest trace inventory is incomplete")
        for path in digest.trace_paths:
            if not path.is_file():
                raise ProfileError(f"profile trace is missing: {path}")
            if cls._sha256_file(path) != digest.trace_sha256[str(path)]:
                raise ProfileError(f"profile trace hash changed: {path}")

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
        traces: list[Path],
    ) -> tuple[dict[str, float], list[dict[str, Any]], int]:
        stage_totals: defaultdict[str, float] = defaultdict(float)
        hotspot_totals: defaultdict[tuple[str, str], list[float]] = defaultdict(
            lambda: [0.0, 0.0]
        )
        event_count = 0
        for path in traces:
            try:
                payload = Profiler._read_trace(path)
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise ProfileError(f"cannot parse profiler trace {path}: {error}") from error
            raw_events = payload.get("traceEvents") if isinstance(payload, dict) else payload
            if not isinstance(raw_events, list):
                raise ProfileError(f"profiler trace has no traceEvents array: {path}")
            for event in raw_events:
                if not isinstance(event, dict) or event.get("ph") != "X":
                    continue
                duration = event.get("dur")
                name = event.get("name")
                if (
                    not isinstance(name, str)
                    or not name.strip()
                    or not isinstance(duration, (int, float))
                    or isinstance(duration, bool)
                    or not math.isfinite(float(duration))
                    or float(duration) <= 0
                ):
                    continue
                total_ms = float(duration) / 1000.0
                category = Profiler._event_category(event)
                stage_totals[category] += total_ms
                totals = hotspot_totals[(name.strip(), category)]
                totals[0] += total_ms
                totals[1] += 1
                event_count += 1

        if event_count == 0:
            raise ProfileError("profiler traces contain no complete positive-duration events")
        total_traced_ms = sum(stage_totals.values())

        # Sidecars may contribute labels and shape hints, but a sidecar cannot
        # make an empty or corrupt raw trace valid.
        sidecar_hotspots: dict[str, dict[str, Any]] = {}
        for path in sorted(profile_dir.rglob("*.json")):
            if path in traces or path.name in {
                "COMMAND.json",
                "PERFORMANCE.json",
                "PROFILE-DIGEST.json",
                "PROFILE-INVENTORY.json",
            }:
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
                        # Keep explicitly instrumented stages without replacing
                        # raw-trace categories.
                        stage_totals[f"instrumented:{name}"] = float(value)
            raw_hotspots = payload.get("hotspots", payload.get("operators"))
            if isinstance(raw_hotspots, list):
                for row in (
                    Profiler._normalize_hotspot(value) for value in raw_hotspots
                ):
                    if row is not None:
                        sidecar_hotspots[row["name"]] = row
        hotspots = []
        for (name, category), (total_ms, calls) in hotspot_totals.items():
            supplemental = sidecar_hotspots.get(name, {})
            hotspots.append(
                {
                    "name": name,
                    "category": category,
                    "total_ms": total_ms,
                    "share": total_ms / total_traced_ms if total_traced_ms else 0.0,
                    "call_count": int(calls),
                    "shapes": supplemental.get("shapes", []),
                    "source_hint": supplemental.get("source_hint", ""),
                }
            )
        hotspots.sort(key=lambda item: (-item["total_ms"], item["name"]))
        return dict(sorted(stage_totals.items())), hotspots, event_count

    @staticmethod
    def _read_trace(path: Path) -> Any:
        opener = gzip.open if path.name.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _event_category(event: dict[str, Any]) -> str:
        category = str(event.get("cat", "")).lower()
        name = str(event.get("name", "")).lower()
        combined = f"{category} {name}"
        if any(token in combined for token in ("nccl", "collective", "sendrecv")):
            return "collective"
        if any(token in combined for token in ("memcpy", "copy", "permute", "contiguous")):
            return "copy_layout"
        if any(token in combined for token in ("kernel", "cuda", "gpu")):
            return "cuda_kernel"
        if any(token in combined for token in ("cpu", "python", "operator")):
            return "cpu_operator"
        return "other"

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _write_json_atomic(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)

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
    """Choose applicable executor lanes without choosing their hypotheses."""

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
        Profiler.validate_digest(digest)
        hotspots = [
            {
                "name": str(row.get("name", "")),
                "total_ms": float(row.get("total_ms", 0.0)),
            }
            for row in digest.hotspots
        ]
        self.last_evidence = {
            "residency": {
                "hotspots": hotspots,
                "stages": dict(digest.stage_ms),
                "knowledge": [
                    "sglang-residency-history",
                    "profile-evidence",
                ],
            },
            "kernel": {
                "hotspots": hotspots,
                "knowledge": ["sglang-kernel-placement", "profile-evidence"],
            }
        }
        # Parallel degrees and rank maps are frozen by the baseline contract.
        # Multi-GPU profiles still feed collective/layout candidates to kernel.
        routes = ["residency", "kernel"]
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
