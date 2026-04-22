#!/usr/bin/env python3
"""Summarize normalized cross-framework benchmark JSONL results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _get(row: dict[str, Any], path: str, default: Any = None) -> Any:
    current: Any = row
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _float(row: dict[str, Any], path: str, default: float = 0.0) -> float:
    value = _get(row, path, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bool(row: dict[str, Any], path: str, default: bool = False) -> bool:
    value = _get(row, path, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _rank_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _get(row, "status") == "ok",
        _bool(row, "sla.passed"),
        _float(row, "metrics.request_throughput"),
        _float(row, "metrics.output_token_throughput"),
        -_float(row, "metrics.p99_ttft_ms", 1e30),
        -_float(row, "metrics.p99_tpot_ms", 1e30),
        -_float(row, "hardware.gpu_count", 1e30),
    )


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise SystemExit(f"{path}:{line_no}: expected a JSON object")
            rows.append(row)
    return rows


def best_by_framework(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        framework = str(_get(row, "framework", "unknown"))
        if framework not in best or _rank_key(row) > _rank_key(best[framework]):
            best[framework] = row
    return sorted(best.values(), key=_rank_key, reverse=True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "framework",
        "candidate_id",
        "status",
        "sla_passed",
        "request_throughput",
        "output_token_throughput",
        "p99_ttft_ms",
        "p99_tpot_ms",
        "gpu_count",
        "failure_reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "framework": _get(row, "framework", ""),
                    "candidate_id": _get(row, "candidate_id", ""),
                    "status": _get(row, "status", ""),
                    "sla_passed": _bool(row, "sla.passed"),
                    "request_throughput": _get(row, "metrics.request_throughput", ""),
                    "output_token_throughput": _get(
                        row, "metrics.output_token_throughput", ""
                    ),
                    "p99_ttft_ms": _get(row, "metrics.p99_ttft_ms", ""),
                    "p99_tpot_ms": _get(row, "metrics.p99_tpot_ms", ""),
                    "gpu_count": _get(row, "hardware.gpu_count", ""),
                    "failure_reason": _get(row, "failure_reason", ""),
                }
            )


def render_markdown(rows: list[dict[str, Any]]) -> str:
    ranked = sorted(rows, key=_rank_key, reverse=True)
    winners = best_by_framework(rows)
    overall = ranked[0] if ranked else None

    lines = ["# Benchmark Summary", ""]
    if overall is None:
        lines.append("No rows found.")
        return "\n".join(lines) + "\n"

    lines.extend(
        [
            "## Overall Winner",
            "",
            f"- Framework: `{_get(overall, 'framework', 'unknown')}`",
            f"- Candidate: `{_get(overall, 'candidate_id', 'unknown')}`",
            f"- SLA passed: `{_bool(overall, 'sla.passed')}`",
            f"- Request throughput: `{_fmt(_get(overall, 'metrics.request_throughput'))}`",
            f"- Output token throughput: `{_fmt(_get(overall, 'metrics.output_token_throughput'))}`",
            "",
            "## Best Per Framework",
            "",
            "| Framework | Candidate | Status | SLA | Req/s | Output tok/s | P99 TTFT ms | P99 TPOT ms | GPUs |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in winners:
        lines.append(
            "| {framework} | {candidate} | {status} | {sla} | {rps} | {otps} | {ttft} | {tpot} | {gpus} |".format(
                framework=_get(row, "framework", ""),
                candidate=_get(row, "candidate_id", ""),
                status=_get(row, "status", ""),
                sla=_bool(row, "sla.passed"),
                rps=_fmt(_get(row, "metrics.request_throughput")),
                otps=_fmt(_get(row, "metrics.output_token_throughput")),
                ttft=_fmt(_get(row, "metrics.p99_ttft_ms")),
                tpot=_fmt(_get(row, "metrics.p99_tpot_ms")),
                gpus=_fmt(_get(row, "hardware.gpu_count")),
            )
        )

    failed = [
        row
        for row in rows
        if _get(row, "status") != "ok" or not _bool(row, "sla.passed")
    ]
    if failed:
        lines.extend(
            [
                "",
                "## Failed Or SLA-Failing Candidates",
                "",
                "| Framework | Candidate | Status | SLA | Reason |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for row in failed:
            lines.append(
                "| {framework} | {candidate} | {status} | {sla} | {reason} |".format(
                    framework=_get(row, "framework", ""),
                    candidate=_get(row, "candidate_id", ""),
                    status=_get(row, "status", ""),
                    sla=_bool(row, "sla.passed"),
                    reason=str(_get(row, "failure_reason", "")).replace("|", "\\|"),
                )
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, help="Normalized JSONL")
    parser.add_argument("--output", required=True, type=Path, help="Markdown summary")
    parser.add_argument("--csv", type=Path, help="Optional CSV table")
    args = parser.parse_args()

    rows = load_rows(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(rows), encoding="utf-8")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        write_csv(args.csv, sorted(rows, key=_rank_key, reverse=True))


if __name__ == "__main__":
    main()
