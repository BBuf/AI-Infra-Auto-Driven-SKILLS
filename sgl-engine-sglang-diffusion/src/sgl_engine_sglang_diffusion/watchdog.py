from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from .state import LeaseUnavailable, StateStore


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

    def tick(self) -> int | None:
        heartbeat = self.campaign_dir / "controller-heartbeat.json"
        if heartbeat.is_file():
            age = time.time() - heartbeat.stat().st_mtime
            if age <= self.stale_after_seconds:
                return None

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

        owner = f"watchdog:{os.getpid()}"
        resource = f"controller:{campaign_id}"
        try:
            self.store.acquire_lease(resource, owner, ttl_seconds=60)
        except LeaseUnavailable:
            return None
        process = subprocess.Popen(
            command,
            cwd=self.campaign_dir,
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=(self.campaign_dir / "watchdog-controller.stdout.log").open("ab"),
            stderr=(self.campaign_dir / "watchdog-controller.stderr.log").open("ab"),
        )
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

    def run_forever(self, *, interval_seconds: float = 30.0) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        while True:
            self.tick()
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
