from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def parse_codex_usage(path: Path) -> dict[str, int] | None:
    """Return the last exact usage object from a Codex JSONL stream."""

    if not path.is_file():
        return None
    last: dict[str, int] | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        usage = _find_usage(event)
        if usage is not None:
            last = usage
    return last


def _find_usage(value: Any) -> dict[str, int] | None:
    if isinstance(value, Mapping):
        direct = _normalize_usage(value)
        if direct is not None:
            return direct
        for name in ("usage", "token_usage", "result", "turn", "data"):
            if name in value:
                nested = _find_usage(value[name])
                if nested is not None:
                    return nested
    return None


def _normalize_usage(value: Mapping[str, Any]) -> dict[str, int] | None:
    aliases = {
        "input_tokens": ("input_tokens", "input_token_count"),
        "cached_input_tokens": (
            "cached_input_tokens",
            "cached_tokens",
            "cache_read_input_tokens",
        ),
        "output_tokens": ("output_tokens", "output_token_count"),
        "reasoning_tokens": (
            "reasoning_tokens",
            "reasoning_output_tokens",
        ),
        "total_tokens": ("total_tokens", "total_token_count"),
    }
    result: dict[str, int] = {}
    for canonical, names in aliases.items():
        for name in names:
            raw = value.get(name)
            if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0:
                result[canonical] = raw
                break
    if "input_tokens" not in result and "output_tokens" not in result:
        return None
    result.setdefault("input_tokens", 0)
    result.setdefault("cached_input_tokens", 0)
    result.setdefault("output_tokens", 0)
    result.setdefault("reasoning_tokens", 0)
    result.setdefault(
        "total_tokens",
        result["input_tokens"] + result["output_tokens"],
    )
    return result


def refresh_token_usage(campaign: Path) -> list[dict[str, Any]]:
    """Append new invocation records and return only each invocation's latest."""

    campaign = campaign.resolve()
    ledger = campaign / "TOKEN-USAGE.jsonl"
    existing = _read_ledger(ledger)
    seen = {
        (str(item.get("invocation_id")), str(item.get("source_event_sha256")))
        for item in existing
    }
    additions: list[dict[str, Any]] = []
    for receipt_path in sorted(campaign.rglob("*.json")):
        if receipt_path.is_symlink():
            continue
        receipt = _read_object(receipt_path)
        if receipt is None:
            continue
        context, stdout_path, argv = _telemetry_source(receipt, receipt_path)
        if context is None or stdout_path is None:
            continue
        try:
            stdout_path.relative_to(campaign)
        except ValueError:
            continue
        if stdout_path.is_symlink():
            continue
        invocation_id = str(
            context.get("invocation_id") or receipt_path.relative_to(campaign)
        )
        role = context.get("agent_role")
        if not isinstance(role, str) or not role:
            continue
        source_digest = (
            hashlib.sha256(stdout_path.read_bytes()).hexdigest()
            if stdout_path.is_file()
            else hashlib.sha256(b"").hexdigest()
        )
        key = (invocation_id, source_digest)
        if key in seen:
            continue
        codex = _is_codex_argv(argv)
        usage = parse_codex_usage(stdout_path) if codex else None
        pid = receipt.get("pid")
        if (
            codex
            and usage is None
            and isinstance(pid, int)
            and not isinstance(pid, bool)
            and _pid_alive(pid)
        ):
            # A running Codex stream has no final usage event yet.
            continue
        record: dict[str, Any] = {
            "schema_version": 1,
            "recorded_at": datetime.now(UTC).isoformat(),
            "campaign_id": context.get("campaign_id"),
            "epoch": context.get("epoch"),
            "invocation_id": invocation_id,
            "pid": receipt.get("pid"),
            "agent_role": role,
            "technique": context.get("technique"),
            "attempt": context.get("attempt"),
            "runtime": (
                "codex" if codex else Path(argv[0]).name if argv else "unknown"
            ),
            "model": _model_from_argv(argv),
            "available": usage is not None,
            "exact": usage is not None,
            "source": str(stdout_path.relative_to(campaign)),
            "source_event_sha256": source_digest,
        }
        if usage is not None:
            record.update(usage)
        else:
            record["reason"] = "runtime_did_not_emit_supported_token_usage"
        additions.append(record)
        seen.add(key)
    if additions:
        descriptor = os.open(
            ledger,
            os.O_APPEND | os.O_CREAT | os.O_WRONLY,
            0o600,
        )
        try:
            for record in additions:
                os.write(
                    descriptor,
                    (
                        json.dumps(
                            record,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    ).encode(),
                )
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return latest_token_records([*existing, *additions])


def latest_token_records(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for record in records:
        invocation_id = str(record.get("invocation_id"))
        latest[invocation_id] = record
    return [latest[name] for name in sorted(latest)]


def token_totals(records: list[dict[str, Any]]) -> dict[str, int]:
    fields = (
        "input_tokens",
        "cached_input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "total_tokens",
    )
    return {
        name: sum(
            int(record.get(name, 0))
            for record in records
            if record.get("available") is True
        )
        for name in fields
    }


def _telemetry_source(
    receipt: Mapping[str, Any], receipt_path: Path
) -> tuple[dict[str, Any] | None, Path | None, list[str]]:
    raw_context = receipt.get("context")
    context = (
        dict(raw_context)
        if isinstance(raw_context, Mapping) and receipt_path.name.startswith("process-")
        else None
    )
    if (
        context is None
        and receipt_path.name.startswith("MASTER-")
        and receipt_path.name.endswith("-COMMAND.json")
        and isinstance(receipt.get("agent_role"), str)
    ):
        context = {
            name: receipt.get(name)
            for name in (
                "campaign_id",
                "epoch",
                "invocation_id",
                "agent_role",
                "technique",
                "attempt",
            )
        }
    stdout = receipt.get("stdout")
    stdout_path = Path(stdout).resolve() if isinstance(stdout, str) else None
    raw_argv = receipt.get("argv")
    argv = [str(item) for item in raw_argv] if isinstance(raw_argv, list) else []
    return context, stdout_path, argv


def _read_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def _read_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _is_codex_argv(argv: list[str]) -> bool:
    return (
        len(argv) >= 2
        and Path(argv[0]).name == "codex"
        and argv[1] == "exec"
        and "--json" in argv
    )


def _model_from_argv(argv: list[str]) -> str | None:
    try:
        return argv[argv.index("--model") + 1]
    except (ValueError, IndexError):
        return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
