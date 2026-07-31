from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

from sgl_engine_sglang_diffusion.models import CampaignStatus
from sgl_engine_sglang_diffusion.state import StateStore
from sgl_engine_sglang_diffusion.watchdog import CampaignWatchdog, WatchdogError


class _CompletedProcess:
    next_pid = 10_000

    def __init__(self) -> None:
        type(self).next_pid += 1
        self.pid = type(self).next_pid

    @staticmethod
    def poll() -> int:
        return 0


def _watchdog(tmp_path: Path) -> tuple[CampaignWatchdog, StateStore, Path]:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    marker = campaign / "invocations.txt"
    command = [
        sys.executable,
        "-m",
        "sgl_engine_sglang_diffusion.cli",
        "resume",
        "--campaign",
        str(campaign),
    ]
    (campaign / "CAMPAIGN.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "campaign_id": "campaign-1",
                "execution_mode": "interactive_single_agent",
                "controller_command": command,
            }
        ),
        encoding="utf-8",
    )

    def spawn(argv: list[str], **_: object) -> _CompletedProcess:
        assert argv == command
        previous = int(marker.read_text()) if marker.is_file() else 0
        marker.write_text(str(previous + 1), encoding="utf-8")
        return _CompletedProcess()

    store = StateStore.open(campaign / "state.sqlite", campaign / "events.jsonl")
    store.create_campaign("campaign-1")
    return (
        CampaignWatchdog(
            campaign,
            store,
            stale_after_seconds=300,
            popen_factory=spawn,
        ),
        store,
        marker,
    )


def _wait_for(path: Path, expected: str) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if path.is_file() and path.read_text(encoding="utf-8") == expected:
            return
        time.sleep(0.01)
    raise AssertionError(f"{path} did not become {expected!r}")


def test_restarts_immediately_after_its_one_shot_controller_exits(
    tmp_path: Path,
) -> None:
    watchdog, store, marker = _watchdog(tmp_path)
    try:
        first_pid = watchdog.tick()
        assert first_pid is not None
        _wait_for(marker, "1")

        deadline = time.monotonic() + 5
        second_pid = None
        while second_pid is None and time.monotonic() < deadline:
            second_pid = watchdog.tick()
            time.sleep(0.01)
        assert second_pid is not None
        assert second_pid != first_pid
        _wait_for(marker, "2")
    finally:
        store.close()


def test_does_not_restart_a_live_controller(tmp_path: Path) -> None:
    watchdog, store, _ = _watchdog(tmp_path)
    heartbeat = watchdog.campaign_dir / "controller-heartbeat.json"
    heartbeat.write_text(
        json.dumps({"pid": 1, "campaign_id": "campaign-1"}),
        encoding="utf-8",
    )
    try:
        assert watchdog.tick() is None
    finally:
        store.close()


def test_terminal_campaign_is_never_restarted(tmp_path: Path) -> None:
    watchdog, store, marker = _watchdog(tmp_path)
    try:
        store.transition(
            "campaign-1",
            CampaignStatus.CANCELLED,
            idempotency_key="cancel",
        )
        assert watchdog.tick() is None
        assert not marker.exists()
        watchdog.run_forever(interval_seconds=0.01)
    finally:
        store.close()


def test_awaiting_agent_campaign_is_yielded_not_restarted(tmp_path: Path) -> None:
    watchdog, store, marker = _watchdog(tmp_path)
    try:
        store.transition(
            "campaign-1",
            CampaignStatus.BASELINE_LOCKED,
            idempotency_key="baseline",
        )
        store.transition(
            "campaign-1",
            CampaignStatus.PROFILED,
            idempotency_key="profile",
        )
        store.transition(
            "campaign-1",
            CampaignStatus.AWAITING_AGENT,
            idempotency_key="await",
        )
        assert watchdog.tick() is None
        assert not marker.exists()
        watchdog.run_forever(interval_seconds=0.01)
    finally:
        store.close()


def test_manifest_cannot_replace_resume_with_an_ai_or_shell_command(
    tmp_path: Path,
) -> None:
    watchdog, store, _ = _watchdog(tmp_path)
    manifest_path = watchdog.campaign_dir / "CAMPAIGN.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["controller_command"] = [sys.executable, "-c", "print('tampered')"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    try:
        with pytest.raises(WatchdogError, match="exact deterministic"):
            watchdog.tick()
    finally:
        store.close()
