from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from sgl_engine_sglang_diffusion.models import CampaignStatus
from sgl_engine_sglang_diffusion.state import StateStore
from sgl_engine_sglang_diffusion.watchdog import CampaignWatchdog


def _watchdog(tmp_path: Path) -> tuple[CampaignWatchdog, StateStore, Path]:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    marker = campaign / "invocations.txt"
    script = tmp_path / "controller.py"
    script.write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "path = Path(sys.argv[1])\n"
        "old = int(path.read_text()) if path.is_file() else 0\n"
        "path.write_text(str(old + 1))\n",
        encoding="utf-8",
    )
    (campaign / "CAMPAIGN.json").write_text(
        json.dumps(
            {
                "campaign_id": "campaign-1",
                "controller_command": [sys.executable, str(script), str(marker)],
            }
        ),
        encoding="utf-8",
    )
    store = StateStore.open(campaign / "state.sqlite", campaign / "events.jsonl")
    store.create_campaign("campaign-1")
    return (
        CampaignWatchdog(campaign, store, stale_after_seconds=300),
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
