from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from .state import LeaseUnavailable, StateStore, TERMINAL_STATUSES


class WatchdogError(RuntimeError):
    pass


class CampaignWatchdog:
    """Restart only the controller command declared by the campaign itself."""

    def __init__(
        self,
        campaign_dir: Path,
        store: StateStore,
        *,
        stale_after_seconds: float = 300.0,
    ) -> None:
        if stale_after_seconds <= 0:
            raise ValueError("stale_after_seconds must be positive")
        self.campaign_dir = campaign_dir.resolve()
        self.store = store
        self.stale_after_seconds = stale_after_seconds
        self._controller: subprocess.Popen[bytes] | None = None

    def tick(self) -> int | None:
        manifest = self._manifest()
        command = manifest.get("controller_command")
        campaign_id = manifest.get("campaign_id")
        if (
            not isinstance(command, list)
            or not command
            or any(not isinstance(value, str) for value in command)
            or not isinstance(campaign_id, str)
        ):
            raise WatchdogError("campaign manifest has no safe controller command")
        if self.store.status(campaign_id) in TERMINAL_STATUSES:
            return None

        if self._controller is not None:
            if self._controller.poll() is None:
                return None
            self._controller = None

        heartbeat = self.campaign_dir / "controller-heartbeat.json"
        if heartbeat.is_file():
            age = time.time() - heartbeat.stat().st_mtime
            pid = self._heartbeat_pid(heartbeat)
            if (
                age <= self.stale_after_seconds
                and pid is not None
                and self._pid_alive(pid)
            ):
                return None

        owner = f"watchdog:{os.getpid()}"
        resource = f"controller:{campaign_id}"
        try:
            self.store.acquire_lease(resource, owner, ttl_seconds=60)
        except LeaseUnavailable:
            return None
        with (
            (self.campaign_dir / "watchdog-controller.stdout.log").open("ab") as stdout,
            (self.campaign_dir / "watchdog-controller.stderr.log").open("ab") as stderr,
        ):
            process = subprocess.Popen(
                command,
                cwd=self.campaign_dir,
                start_new_session=True,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
            )
        self._controller = process
        receipt = self.campaign_dir / "watchdog-restart.json"
        receipt.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "campaign_id": campaign_id,
                    "pid": process.pid,
                    "command": command,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return process.pid

    @staticmethod
    def _heartbeat_pid(path: Path) -> int | None:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        pid = value.get("pid") if isinstance(value, dict) else None
        return (
            pid
            if isinstance(pid, int) and not isinstance(pid, bool) and pid > 0
            else None
        )

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def run_forever(self, *, interval_seconds: float = 30.0) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        while True:
            self.tick()
            manifest = self._manifest()
            campaign_id = manifest.get("campaign_id")
            if (
                isinstance(campaign_id, str)
                and self.store.status(campaign_id) in TERMINAL_STATUSES
            ):
                return
            time.sleep(interval_seconds)

    def _manifest(self) -> dict[str, Any]:
        path = self.campaign_dir / "CAMPAIGN.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise WatchdogError(f"invalid campaign manifest: {path}") from error
        if not isinstance(value, dict):
            raise WatchdogError("campaign manifest must be an object")
        return value
