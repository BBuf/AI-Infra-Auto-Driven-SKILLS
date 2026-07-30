from __future__ import annotations

import json
from pathlib import Path

from sgl_engine_sglang_diffusion.telemetry import (
    parse_codex_usage,
    refresh_token_usage,
    token_totals,
)


def test_parse_codex_usage_accepts_terminal_usage_event(tmp_path: Path) -> None:
    stream = tmp_path / "stdout.jsonl"
    stream.write_text(
        json.dumps({"type": "thread.started", "thread_id": "thread-1"})
        + "\n"
        + json.dumps(
            {
                "type": "turn.completed",
                "usage": {
                    "input_tokens": 120,
                    "cached_input_tokens": 20,
                    "output_tokens": 30,
                    "reasoning_output_tokens": 4,
                },
            }
        )
        + "\n"
    )
    assert parse_codex_usage(stream) == {
        "input_tokens": 120,
        "cached_input_tokens": 20,
        "output_tokens": 30,
        "reasoning_tokens": 4,
        "total_tokens": 150,
    }


def test_refresh_ledger_is_deduplicated_and_attributed(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    root = campaign / "executors" / "abc"
    root.mkdir(parents=True)
    stream = root / "stdout-001.log"
    stream.write_text(
        json.dumps(
            {
                "type": "turn.completed",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }
        )
        + "\n"
    )
    receipt = {
        "schema_version": 1,
        "pid": 12,
        "argv": ["codex", "exec", "--json", "--model", "gpt-test", "goal.md"],
        "stdout": str(stream),
        "context": {
            "campaign_id": "campaign-1",
            "epoch": 2,
            "agent_role": "executor",
            "technique": "kernel",
            "attempt": 1,
            "invocation_id": "executor:abc:attempt:1",
        },
    }
    (root / "process-001.json").write_text(json.dumps(receipt))

    first = refresh_token_usage(campaign)
    second = refresh_token_usage(campaign)
    assert first == second
    assert len((campaign / "TOKEN-USAGE.jsonl").read_text().splitlines()) == 1
    assert first[0]["technique"] == "kernel"
    assert first[0]["model"] == "gpt-test"
    assert token_totals(first)["total_tokens"] == 15


def test_unsupported_runtime_is_explicitly_unavailable(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    stream = campaign / "agent.log"
    stream.write_text("done\n")
    (campaign / "MASTER-OTHER-COMMAND.json").write_text(
        json.dumps(
            {
                "argv": ["other-agent", "goal.md"],
                "stdout": str(stream),
                "agent_role": "executor",
                "campaign_id": "campaign-1",
                "invocation_id": "other:1",
            }
        )
    )
    records = refresh_token_usage(campaign)
    assert records[0]["available"] is False
    assert records[0]["exact"] is False


def test_finished_codex_without_usage_is_explicitly_unavailable(
    tmp_path: Path,
) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    stream = campaign / "master.log"
    stream.write_text('{"type":"error","message":"failed"}\n')
    (campaign / "MASTER-FAILED-COMMAND.json").write_text(
        json.dumps(
            {
                "argv": ["codex", "exec", "--json", "goal.md"],
                "stdout": str(stream),
                "agent_role": "master_method",
                "campaign_id": "campaign-1",
                "invocation_id": "master:failed",
            }
        )
    )
    records = refresh_token_usage(campaign)
    assert records[0]["runtime"] == "codex"
    assert records[0]["available"] is False
