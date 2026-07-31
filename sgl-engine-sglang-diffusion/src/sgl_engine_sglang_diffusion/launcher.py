from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from .agents import redact_argv
from .request import load_launch_request, normalize_launch_request
from .state import StateStore, TERMINAL_STATUSES


class LaunchError(RuntimeError):
    """A one-shot launch cannot safely create or recover a campaign."""


WatchdogSpawner = Callable[[Path], int]


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _spawn_watchdog(campaign: Path) -> int:
    argv = [
        sys.executable,
        "-m",
        "sgl_engine_sglang_diffusion.cli",
        "watchdog",
        "--campaign",
        str(campaign),
    ]
    stdout_path = campaign / "watchdog.stdout.log"
    stderr_path = campaign / "watchdog.stderr.log"
    with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
        process = subprocess.Popen(
            argv,
            cwd=campaign,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
            shell=False,
        )
    _atomic_json(
        campaign / "WATCHDOG.json",
        {
            "schema_version": 1,
            "pid": process.pid,
            "argv": redact_argv(argv),
            "cwd": str(campaign),
            "started_at": datetime.now(UTC).isoformat(),
            "start_new_session": True,
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        },
    )
    return process.pid


def launch_campaign(
    request_path: Path,
    *,
    detach: bool,
    watchdog_spawner: WatchdogSpawner = _spawn_watchdog,
) -> dict[str, Any]:
    """Create one frozen campaign and optionally hand it to a detached watchdog."""

    request_path = request_path.resolve()
    request = load_launch_request(request_path)
    goal, command_template = normalize_launch_request(request)
    request_payload = request.model_dump(mode="json")
    request_payload.pop("agent", None)
    request_digest = hashlib.sha256(
        json.dumps(request_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    key = request.idempotency_key or request_digest
    key_digest = hashlib.sha256(key.encode()).hexdigest()
    index_path = request.run_root / ".launch-index" / f"{key_digest}.json"
    if index_path.is_file():
        existing = json.loads(index_path.read_text(encoding="utf-8"))
        if existing.get("request_sha256") != request_digest:
            raise LaunchError(
                "launch idempotency key is already bound to another request"
            )
        campaign = Path(existing["campaign"]).resolve()
        if not (campaign / "CAMPAIGN.json").is_file():
            raise LaunchError(f"launch index points to a missing campaign: {campaign}")
        watchdog_pid = _recorded_watchdog_pid(campaign)
        if (
            detach
            and not _campaign_quiescent(campaign)
            and (watchdog_pid is None or not _pid_alive(watchdog_pid))
        ):
            watchdog_pid = watchdog_spawner(campaign)
        from .progress import write_progress

        write_progress(campaign)
        return _launch_payload(
            campaign,
            watchdog_pid=watchdog_pid,
            reused=True,
        )

    # Imported lazily to avoid a CLI/launcher import cycle.
    from .cli import initialize_goal

    campaign = initialize_goal(goal, request.run_root)
    frozen_request = dict(request_payload)
    baseline = dict(frozen_request["baseline"])
    if baseline.get("command") is not None:
        baseline["command_sha256"] = hashlib.sha256(
            baseline["command"].encode()
        ).hexdigest()
        baseline["command"] = "<frozen in BASELINE-COMMAND.json>"
    if baseline.get("argv") is not None:
        baseline["argv_sha256"] = hashlib.sha256(
            json.dumps(baseline["argv"], separators=(",", ":")).encode()
        ).hexdigest()
        baseline["argv"] = ["<frozen in BASELINE-COMMAND.json>"]
    frozen_request["baseline"] = baseline
    (campaign / "REQUEST.yaml").write_text(
        yaml.safe_dump(frozen_request, sort_keys=False),
        encoding="utf-8",
    )
    _atomic_json(
        campaign / "BASELINE-COMMAND.json",
        command_template.model_dump(mode="json"),
    )
    _atomic_json(
        campaign / "LAUNCH-REQUEST.json",
        {
            "schema_version": 1,
            "request_sha256": request_digest,
            "idempotency_key_sha256": key_digest,
            "request_source": str(request_path),
            "machine": request.machine,
            "model": request.model,
            "sglang_checkout": str(request.sglang_checkout),
            "token_budget": request.token_budget,
        },
    )
    campaign_id = json.loads((campaign / "CAMPAIGN.json").read_text(encoding="utf-8"))[
        "campaign_id"
    ]
    _atomic_json(
        index_path,
        {
            "schema_version": 1,
            "campaign": str(campaign),
            "campaign_id": campaign_id,
            "request_sha256": request_digest,
        },
    )
    watchdog_pid = watchdog_spawner(campaign) if detach else None
    receipt = {
        "schema_version": 1,
        "campaign": str(campaign),
        "campaign_id": campaign_id,
        "request_sha256": request_digest,
        "detached": detach,
        "watchdog_pid": watchdog_pid,
        "created_at": datetime.now(UTC).isoformat(),
    }
    _atomic_json(campaign / "LAUNCH.json", receipt)
    from .progress import write_progress

    write_progress(campaign)
    return _launch_payload(campaign, watchdog_pid=watchdog_pid, reused=False)


def _recorded_watchdog_pid(campaign: Path) -> int | None:
    path = campaign / "WATCHDOG.json"
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    pid = value.get("pid")
    return pid if isinstance(pid, int) and pid > 0 else None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _campaign_quiescent(campaign: Path) -> bool:
    manifest = json.loads((campaign / "CAMPAIGN.json").read_text(encoding="utf-8"))
    with StateStore.open(campaign / "state.sqlite", campaign / "events.jsonl") as store:
        status = store.status(str(manifest["campaign_id"]))
        return status in TERMINAL_STATUSES or status.value == "AWAITING_AGENT"


def _launch_payload(
    campaign: Path, *, watchdog_pid: int | None, reused: bool
) -> dict[str, Any]:
    manifest = json.loads((campaign / "CAMPAIGN.json").read_text(encoding="utf-8"))
    base = [
        "sgl-diffusion-engine",
        "progress",
        "--campaign",
        str(campaign),
    ]
    return {
        "campaign_id": manifest["campaign_id"],
        "campaign": str(campaign),
        "execution_mode": "interactive_single_agent",
        "watchdog_pid": watchdog_pid,
        "reused": reused,
        "progress_command": [*base, "--watch"],
        "status_command": [
            "sgl-diffusion-engine",
            "status",
            "--campaign",
            str(campaign),
        ],
        "work_command": [
            "sgl-diffusion-engine",
            "work",
            "--campaign",
            str(campaign),
            "--json",
        ],
        "resume_command": [
            "sgl-diffusion-engine",
            "resume",
            "--campaign",
            str(campaign),
        ],
    }
